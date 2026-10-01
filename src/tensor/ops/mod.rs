//! Tensor operations, grouped by what they do.
//!
//! | module | ops |
//! |---|---|
//! | [`arith`] | `add` `sub` `mul` `div` and their scalar forms |
//! | [`unary`] | `neg` `exp` `log` `sqrt` `abs` `powf` `clamp` |
//! | [`activation`] | `relu` `sigmoid` `tanh` `gelu` `silu` `leaky_relu` `softmax` `log_softmax` |
//! | [`matmul`] | `matmul` and the transpose-fused `matmul_nt` / `matmul_tn` |
//! | [`reduce`] | `sum` `mean` `max` `min` `argmax` and their axis forms |
//! | [`view`] | `reshape` `permute` `transpose` `expand` `cat` `stack` `narrow` |
//! | [`index`] | `index_select` |
//! | [`conv`] | `im2col`, the unfold that turns convolution into a GEMM |
//! | [`pool`] | `max_pool2d` `avg_pool2d` `adaptive_avg_pool2d` |
//! | [`norm`] | `layer_norm` `batch_norm2d`, fused so the backward can reuse the statistics |
//!
//! Each public op computes its result on whichever device the inputs live on,
//! then attaches a backward rule from [`autograd::ops`](crate::autograd::ops).
//! The `raw_*` helpers here do the same compute with no graph, for use inside
//! backward rules where recording would be wrong.

pub mod activation;
pub mod arith;
pub mod conv;
pub mod index;
pub mod matmul;
pub mod norm;
pub mod pool;
pub mod reduce;
pub mod unary;
pub mod view;

use rayon::prelude::*;

use crate::cuda::{ffi, kernels};

use super::shape;
use super::storage::Storage;
use super::Tensor;

/// Elements below which rayon's overhead outweighs the parallelism.
const PARALLEL_THRESHOLD: usize = 1 << 15;

/// Apply an element-wise function of one tensor.
pub(crate) fn unary_op(
    x: &Tensor,
    cpu: impl Fn(f32) -> f32 + Send + Sync,
    gpu: ffi::UnaryKernel,
) -> Tensor {
    match x.storage() {
        Storage::Cpu(data) => Tensor::from_vec(map(data, cpu), x.shape()),
        Storage::Cuda(buf) => {
            let out = kernels::unary(buf, x.numel(), gpu).expect("cuda unary op");
            Tensor::raw(Storage::Cuda(out), x.shape().to_vec(), x.device())
        }
    }
}

/// Apply an element-wise function of a tensor and a scalar.
pub(crate) fn scalar_op(
    x: &Tensor,
    s: f32,
    cpu: impl Fn(f32, f32) -> f32 + Send + Sync,
    gpu: ffi::ScalarKernel,
) -> Tensor {
    match x.storage() {
        Storage::Cpu(data) => Tensor::from_vec(map(data, |v| cpu(v, s)), x.shape()),
        Storage::Cuda(buf) => {
            let out = kernels::scalar(buf, s, x.numel(), gpu).expect("cuda scalar op");
            Tensor::raw(Storage::Cuda(out), x.shape().to_vec(), x.device())
        }
    }
}

/// Apply an element-wise function of two tensors, broadcasting if their shapes differ.
pub(crate) fn binary_op(
    a: &Tensor,
    b: &Tensor,
    cpu: impl Fn(f32, f32) -> f32 + Send + Sync,
    gpu: ffi::BinaryKernel,
) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "device mismatch: {} and {} — move one with .to()",
        a.device(),
        b.device()
    );

    if a.shape() != b.shape() {
        return broadcast_binary_op(a, b, cpu, gpu);
    }

    match (a.storage(), b.storage()) {
        (Storage::Cpu(x), Storage::Cpu(y)) => Tensor::from_vec(zip(x, y, cpu), a.shape()),
        (Storage::Cuda(x), Storage::Cuda(y)) => {
            let out = kernels::binary(x, y, a.numel(), gpu).expect("cuda binary op");
            Tensor::raw(Storage::Cuda(out), a.shape().to_vec(), a.device())
        }
        _ => unreachable!("devices already checked equal"),
    }
}

/// Broadcast both operands to a common shape, then apply `cpu`/`gpu`.
///
/// On the GPU we materialise the expansion so the kernel stays on-device; on the
/// CPU we walk broadcast strides instead, which avoids the copy entirely.
fn broadcast_binary_op(
    a: &Tensor,
    b: &Tensor,
    cpu: impl Fn(f32, f32) -> f32 + Send + Sync,
    gpu: ffi::BinaryKernel,
) -> Tensor {
    let out_shape = shape::broadcast(a.shape(), b.shape());

    if a.is_cuda() {
        let a_full = view::broadcast_to(a, &out_shape);
        let b_full = view::broadcast_to(b, &out_shape);
        return binary_op(&a_full, &b_full, cpu, gpu);
    }

    let ndim = out_shape.len();
    let out_strides = shape::strides_for(&out_shape);
    let a_strides = shape::broadcast_strides(&shape::pad_left(a.shape(), ndim), &out_shape);
    let b_strides = shape::broadcast_strides(&shape::pad_left(b.shape(), ndim), &out_shape);
    let (a_data, b_data) = (a.to_vec(), b.to_vec());

    let data: Vec<f32> = (0..shape::numel(&out_shape))
        .map(|flat| {
            let (mut rest, mut ai, mut bi) = (flat, 0usize, 0usize);
            for d in 0..ndim {
                let coord = rest / out_strides[d];
                rest %= out_strides[d];
                ai += coord * a_strides[d];
                bi += coord * b_strides[d];
            }
            cpu(a_data[ai], b_data[bi])
        })
        .collect();

    Tensor::from_vec(data, &out_shape)
}

/// Map over host data, in parallel once it is large enough to pay off.
pub(crate) fn map(data: &[f32], f: impl Fn(f32) -> f32 + Send + Sync) -> Vec<f32> {
    if data.len() >= PARALLEL_THRESHOLD {
        data.par_iter().map(|&v| f(v)).collect()
    } else {
        data.iter().map(|&v| f(v)).collect()
    }
}

/// Zip two equally sized host buffers, in parallel once large enough.
pub(crate) fn zip(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32 + Send + Sync) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= PARALLEL_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| f(x, y))
            .collect()
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
    }
}
