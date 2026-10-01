//! Matrix multiplication, plain and transpose-fused.
//!
//! `matmul_nt` and `matmul_tn` read an operand transposed instead of building a
//! transposed copy first. That matters because the backward pass of every linear
//! layer needs exactly those two forms, and on the GPU they map onto a cuBLAS
//! transpose flag — no staging buffer, no extra kernel.

use rayon::prelude::*;

use crate::autograd::ops::matmul::{MatmulBackward, MatmulNtBackward, MatmulTnBackward};
use crate::cuda::kernels;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// Which operands are read transposed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    /// `C = A · B`, with `A` as `[m, k]` and `B` as `[k, n]`.
    Plain,
    /// `C = A · Bᵀ`, with `B` stored as `[n, k]`.
    RhsT,
    /// `C = Aᵀ · B`, with `A` stored as `[k, m]`.
    LhsT,
}

impl Tensor {
    /// `self · other`. Both must be at least 2-D; leading dimensions batch.
    ///
    /// A batch dimension of 1 on either side broadcasts against the other.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let out = multiply(self, other, Layout::Plain);
        out.with_grad(&[self, other], || MatmulBackward {
            a: self.detach(),
            b: other.detach(),
        })
    }

    /// `self · otherᵀ`, with `other` stored as `[.., n, k]`.
    pub fn matmul_nt(&self, other: &Tensor) -> Tensor {
        let out = multiply(self, other, Layout::RhsT);
        out.with_grad(&[self, other], || MatmulNtBackward {
            a: self.detach(),
            b: other.detach(),
        })
    }

    /// `selfᵀ · other`, with `self` stored as `[.., k, m]`.
    pub fn matmul_tn(&self, other: &Tensor) -> Tensor {
        let out = multiply(self, other, Layout::LhsT);
        out.with_grad(&[self, other], || MatmulTnBackward {
            a: self.detach(),
            b: other.detach(),
        })
    }
}

/// The `(m, n, k, batch)` of one multiplication, plus each side's batch count.
struct Dims {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    a_batch: usize,
    b_batch: usize,
}

/// Shared forward for all three layouts, with no graph node attached.
pub(crate) fn multiply(a: &Tensor, b: &Tensor, layout: Layout) -> Tensor {
    assert_eq!(
        a.device(),
        b.device(),
        "matmul device mismatch: {} and {}",
        a.device(),
        b.device()
    );
    assert!(
        a.ndim() >= 2 && b.ndim() >= 2,
        "matmul needs 2+ dimensions, got {:?} and {:?}",
        a.shape(),
        b.shape()
    );

    let dims = resolve_dims(a, b, layout);
    // The output's batch prefix comes from whichever side actually has one:
    // more batch items wins, and at equal counts the higher rank wins, so
    // [m,k] × [1,k,n] stays 3-D instead of collapsing to a plain matrix.
    let mut out_shape: Vec<usize> = a.shape()[..a.ndim() - 2].to_vec();
    if dims.b_batch > dims.a_batch || (dims.b_batch == dims.a_batch && b.ndim() > a.ndim()) {
        out_shape = b.shape()[..b.ndim() - 2].to_vec();
    }
    out_shape.push(dims.m);
    out_shape.push(dims.n);

    match (a.storage(), b.storage()) {
        (Storage::Cuda(x), Storage::Cuda(y)) if dims.a_batch == dims.b_batch => {
            let out = cuda_gemm(x, y, &dims, layout);
            Tensor::raw(Storage::Cuda(out), out_shape, a.device())
        }
        // cuBLAS strided-batched needs matching batch counts; broadcast one side
        // by materialising it rather than adding a second GPU path.
        (Storage::Cuda(_), Storage::Cuda(_)) => {
            let (x, y) = match_batches(a, b, &dims);
            multiply(&x, &y, layout)
        }
        _ => Tensor::from_vec(
            cpu_gemm(&a.to_vec(), &b.to_vec(), &dims, layout),
            &out_shape,
        )
        .to(a.device()),
    }
}

fn resolve_dims(a: &Tensor, b: &Tensor, layout: Layout) -> Dims {
    let (ar, ac) = (a.dim(a.ndim() - 2), a.dim(a.ndim() - 1));
    let (br, bc) = (b.dim(b.ndim() - 2), b.dim(b.ndim() - 1));

    // Read each side's (rows, inner) according to which operand is transposed.
    let (m, k) = match layout {
        Layout::LhsT => (ac, ar),
        _ => (ar, ac),
    };
    let (k_b, n) = match layout {
        Layout::RhsT => (bc, br),
        _ => (br, bc),
    };
    assert_eq!(
        k,
        k_b,
        "matmul inner dimension mismatch: {:?} and {:?} under {layout:?}",
        a.shape(),
        b.shape()
    );

    let a_batch: usize = a.shape()[..a.ndim() - 2].iter().product();
    let b_batch: usize = b.shape()[..b.ndim() - 2].iter().product();
    assert!(
        a_batch == b_batch || a_batch == 1 || b_batch == 1,
        "matmul batch mismatch: {a_batch} and {b_batch}"
    );

    Dims {
        m,
        n,
        k,
        batch: a_batch.max(b_batch),
        a_batch,
        b_batch,
    }
}

/// Tile whichever side has a single batch up to the other's batch count.
fn match_batches(a: &Tensor, b: &Tensor, dims: &Dims) -> (Tensor, Tensor) {
    let tile = |t: &Tensor| {
        let mut shape = vec![dims.batch];
        shape.extend_from_slice(&t.shape()[t.ndim() - 2..]);
        let flat = t.reshape(&[1, t.dim(t.ndim() - 2) as i64, t.dim(t.ndim() - 1) as i64]);
        flat.expand(&shape)
    };
    match (dims.a_batch, dims.b_batch) {
        (1, _) => (tile(a), b.clone()),
        (_, 1) => (a.clone(), tile(b)),
        _ => (a.clone(), b.clone()),
    }
}

fn cuda_gemm(
    a: &crate::cuda::CudaBuffer,
    b: &crate::cuda::CudaBuffer,
    dims: &Dims,
    layout: Layout,
) -> crate::cuda::CudaBuffer {
    let (m, n, k, batch) = (dims.m, dims.n, dims.k, dims.batch);
    let result = if batch == 1 {
        match layout {
            Layout::Plain => kernels::matmul(a, b, m, n, k),
            Layout::RhsT => kernels::matmul_nt(a, b, m, n, k),
            Layout::LhsT => kernels::matmul_tn(a, b, m, n, k),
        }
    } else {
        match layout {
            Layout::Plain => kernels::matmul_batched(a, b, m, n, k, batch),
            Layout::RhsT => kernels::matmul_batched_nt(a, b, m, n, k, batch),
            Layout::LhsT => kernels::matmul_batched_tn(a, b, m, n, k, batch),
        }
    };
    result.expect("cuda gemm")
}

/// Row-parallel GEMM.
///
/// Parallelism is per *output row*, not per batch: a transformer step's largest
/// multiplies — the feed-forward and head projections — have batch 1, and
/// batch-level chunks would leave every core but one idle on exactly the work
/// that dominates.
///
/// Each layout gets the loop order that streams its operands contiguously:
/// `Plain` and `LhsT` accumulate whole rows of `b` at a time, and `RhsT` — the
/// `q·kᵀ` of every attention score — is a dot product of two contiguous rows.
fn cpu_gemm(a: &[f32], b: &[f32], dims: &Dims, layout: Layout) -> Vec<f32> {
    let Dims {
        m,
        n,
        k,
        batch,
        a_batch,
        b_batch,
    } = *dims;
    let (a_stride, b_stride) = (
        if a_batch == 1 { 0 } else { m * k },
        if b_batch == 1 { 0 } else { k * n },
    );

    let fill_row = |row_index: usize, row: &mut [f32]| {
        let (batch_index, i) = (row_index / m, row_index % m);
        let a_base = batch_index * a_stride;
        let b_base = batch_index * b_stride;

        if layout == Layout::RhsT {
            let a_row = &a[a_base + i * k..a_base + (i + 1) * k];
            for (j, cell) in row.iter_mut().enumerate() {
                let b_row = &b[b_base + j * k..b_base + (j + 1) * k];
                *cell = a_row.iter().zip(b_row).map(|(&x, &y)| x * y).sum();
            }
            return;
        }
        for p in 0..k {
            let lhs = match layout {
                Layout::LhsT => a[a_base + p * m + i],
                _ => a[a_base + i * k + p],
            };
            if lhs == 0.0 {
                continue;
            }
            let b_row = &b[b_base + p * n..b_base + (p + 1) * n];
            for (cell, &rhs) in row.iter_mut().zip(b_row) {
                *cell += lhs * rhs;
            }
        }
    };

    let mut out = vec![0.0f32; batch * m * n];
    // Tiny multiplies are not worth a trip through the thread pool.
    if batch * m * n * k < 16_384 {
        out.chunks_mut(n)
            .enumerate()
            .for_each(|(r, row)| fill_row(r, row));
    } else {
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(r, row)| fill_row(r, row));
    }
    out
}
