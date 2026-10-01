//! Safe wrappers around the CUDA kernels.
//!
//! Each function allocates its output buffer, launches the kernel, and turns a
//! non-zero status into an [`Error::Cuda`]. Callers never touch raw pointers.

use std::ffi::c_int;

use super::buffer::CudaBuffer;
use super::ffi::{self, BinaryKernel, ScalarKernel, UnaryKernel};
use crate::error::{Error, Result};

/// Run `kernel`, returning its freshly allocated output of `out_len` floats.
fn launch(
    out_len: usize,
    name: &'static str,
    kernel: impl FnOnce(&CudaBuffer) -> c_int,
) -> Result<CudaBuffer> {
    let out = CudaBuffer::new(out_len)?;
    let code = kernel(&out);
    if code != 0 {
        return Err(Error::Cuda(format!("{name} failed (status {code})")));
    }
    Ok(out)
}

// ── Element-wise ─────────────────────────────────────────────────────────────

pub fn binary(
    a: &CudaBuffer,
    b: &CudaBuffer,
    n: usize,
    kernel: BinaryKernel,
) -> Result<CudaBuffer> {
    launch(n, "binary op", |out| unsafe {
        kernel(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n)
    })
}

pub fn unary(a: &CudaBuffer, n: usize, kernel: UnaryKernel) -> Result<CudaBuffer> {
    launch(n, "unary op", |out| unsafe {
        kernel(a.as_ptr(), out.as_mut_ptr(), n)
    })
}

pub fn scalar(a: &CudaBuffer, s: f32, n: usize, kernel: ScalarKernel) -> Result<CudaBuffer> {
    launch(n, "scalar op", |out| unsafe {
        kernel(a.as_ptr(), s, out.as_mut_ptr(), n)
    })
}

pub fn clamp(a: &CudaBuffer, lo: f32, hi: f32, n: usize) -> Result<CudaBuffer> {
    launch(n, "clamp", |out| unsafe {
        ffi::fastnn_cuda_clamp(a.as_ptr(), lo, hi, out.as_mut_ptr(), n)
    })
}

pub fn leaky_relu(a: &CudaBuffer, slope: f32, n: usize) -> Result<CudaBuffer> {
    launch(n, "leaky_relu", |out| unsafe {
        ffi::fastnn_cuda_leaky_relu(a.as_ptr(), slope, out.as_mut_ptr(), n)
    })
}

// ── Softmax ──────────────────────────────────────────────────────────────────

pub fn softmax(x: &CudaBuffer, rows: usize, cols: usize) -> Result<CudaBuffer> {
    launch(rows * cols, "softmax", |out| unsafe {
        ffi::fastnn_cuda_softmax(x.as_ptr(), out.as_mut_ptr(), rows as c_int, cols as c_int)
    })
}

pub fn log_softmax(x: &CudaBuffer, rows: usize, cols: usize) -> Result<CudaBuffer> {
    launch(rows * cols, "log_softmax", |out| unsafe {
        ffi::fastnn_cuda_log_softmax(x.as_ptr(), out.as_mut_ptr(), rows as c_int, cols as c_int)
    })
}

pub fn softmax_backward(
    grad: &CudaBuffer,
    y: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> Result<CudaBuffer> {
    launch(rows * cols, "softmax_backward", |out| unsafe {
        ffi::fastnn_cuda_softmax_backward(
            grad.as_ptr(),
            y.as_ptr(),
            out.as_mut_ptr(),
            rows as c_int,
            cols as c_int,
        )
    })
}

// ── GEMM ─────────────────────────────────────────────────────────────────────

/// `C[m,n] = A[m,k] · B[k,n]`
pub fn matmul(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize) -> Result<CudaBuffer> {
    launch(m * n, "matmul", |out| unsafe {
        ffi::fastnn_cuda_matmul(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            n as c_int,
            k as c_int,
            k as c_int,
            n as c_int,
            n as c_int,
            1.0,
            0.0,
        )
    })
}

/// `C[m,n] = A[m,k] · B[n,k]ᵀ` — B is read transposed, no staging buffer.
pub fn matmul_nt(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer> {
    launch(m * n, "matmul_nt", |out| unsafe {
        ffi::fastnn_cuda_matmul_nt(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            n as c_int,
            k as c_int,
        )
    })
}

/// `C[m,n] = A[k,m]ᵀ · B[k,n]` — A is read transposed, no staging buffer.
pub fn matmul_tn(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaBuffer> {
    launch(m * n, "matmul_tn", |out| unsafe {
        ffi::fastnn_cuda_matmul_tn(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            n as c_int,
            k as c_int,
        )
    })
}

pub fn matmul_batched(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) -> Result<CudaBuffer> {
    launch(batch * m * n, "matmul_batched", |out| unsafe {
        ffi::fastnn_cuda_matmul_batched(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            n as c_int,
            k as c_int,
            batch as c_int,
            1.0,
            0.0,
        )
    })
}

pub fn matmul_batched_nt(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) -> Result<CudaBuffer> {
    launch(batch * m * n, "matmul_batched_nt", |out| unsafe {
        ffi::fastnn_cuda_matmul_batched_nt(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            n as c_int,
            k as c_int,
            batch as c_int,
        )
    })
}

pub fn matmul_batched_tn(
    a: &CudaBuffer,
    b: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) -> Result<CudaBuffer> {
    launch(batch * m * n, "matmul_batched_tn", |out| unsafe {
        ffi::fastnn_cuda_matmul_batched_tn(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            m as c_int,
            n as c_int,
            k as c_int,
            batch as c_int,
        )
    })
}

// ── Layout ───────────────────────────────────────────────────────────────────

pub fn transpose_2d(x: &CudaBuffer, rows: usize, cols: usize) -> Result<CudaBuffer> {
    launch(rows * cols, "transpose", |out| unsafe {
        ffi::fastnn_cuda_transpose(x.as_ptr(), out.as_mut_ptr(), rows as c_int, cols as c_int)
    })
}

pub fn transpose_batched(
    x: &CudaBuffer,
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<CudaBuffer> {
    launch(batch * rows * cols, "transpose_batched", |out| unsafe {
        ffi::fastnn_cuda_transpose_batched(
            x.as_ptr(),
            out.as_mut_ptr(),
            batch as c_int,
            rows as c_int,
            cols as c_int,
        )
    })
}

/// Gather `numel` elements: output index `i` reads the input element whose
/// coordinates come from decoding `i` against `out_strides` and re-encoding with
/// `in_strides` permuted by `perm`. Stride 0 broadcasts a dimension.
pub fn permute_nd(
    x: &CudaBuffer,
    out_strides: &[usize],
    in_strides: &[usize],
    perm: &[usize],
    numel: usize,
) -> Result<CudaBuffer> {
    let out_s = to_c_ints(out_strides);
    let in_s = to_c_ints(in_strides);
    let p = to_c_ints(perm);
    launch(numel, "permute_nd", |out| unsafe {
        ffi::fastnn_cuda_permute_nd(
            x.as_ptr(),
            out.as_mut_ptr(),
            out_s.as_ptr(),
            in_s.as_ptr(),
            p.as_ptr(),
            perm.len() as c_int,
            numel as c_int,
        )
    })
}

// ── Reductions ───────────────────────────────────────────────────────────────

pub fn sum(x: &CudaBuffer, n: usize) -> Result<CudaBuffer> {
    launch(1, "sum", |out| unsafe {
        ffi::fastnn_cuda_sum(x.as_ptr(), out.as_mut_ptr(), n)
    })
}

pub fn mean(x: &CudaBuffer, n: usize) -> Result<CudaBuffer> {
    launch(1, "mean", |out| unsafe {
        ffi::fastnn_cuda_mean(x.as_ptr(), out.as_mut_ptr(), n)
    })
}

pub fn max(x: &CudaBuffer, n: usize) -> Result<CudaBuffer> {
    launch(1, "max", |out| unsafe {
        ffi::fastnn_cuda_max(x.as_ptr(), out.as_mut_ptr(), n)
    })
}

pub fn min(x: &CudaBuffer, n: usize) -> Result<CudaBuffer> {
    launch(1, "min", |out| unsafe {
        ffi::fastnn_cuda_min(x.as_ptr(), out.as_mut_ptr(), n)
    })
}

pub fn sum_axis(x: &CudaBuffer, shape: &[usize], axis: usize) -> Result<CudaBuffer> {
    let total = x.len();
    let shape_c = to_c_ints(shape);
    launch(total / shape[axis], "sum_axis", |out| unsafe {
        ffi::fastnn_cuda_sum_axis(
            x.as_ptr(),
            out.as_mut_ptr(),
            shape_c.as_ptr(),
            shape.len() as c_int,
            axis as c_int,
            total as c_int,
        )
    })
}

// ── Fused layers ─────────────────────────────────────────────────────────────

/// Returns `(output, mean, inv_std)`; the last two feed [`layer_norm_backward`].
pub fn layer_norm_forward(
    x: &CudaBuffer,
    gamma: &CudaBuffer,
    beta: &CudaBuffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer)> {
    let out = CudaBuffer::new(rows * cols)?;
    let mean = CudaBuffer::new(rows)?;
    let inv_std = CudaBuffer::new(rows)?;
    let code = unsafe {
        ffi::fastnn_cuda_layer_norm_forward(
            x.as_ptr(),
            gamma.as_ptr(),
            beta.as_ptr(),
            out.as_mut_ptr(),
            mean.as_mut_ptr(),
            inv_std.as_mut_ptr(),
            rows as c_int,
            cols as c_int,
            eps,
        )
    };
    if code != 0 {
        return Err(Error::Cuda(format!(
            "layer_norm_forward failed (status {code})"
        )));
    }
    Ok((out, mean, inv_std))
}

/// Returns `(grad_x, grad_gamma, grad_beta)`.
pub fn layer_norm_backward(
    grad: &CudaBuffer,
    x: &CudaBuffer,
    gamma: &CudaBuffer,
    mean: &CudaBuffer,
    inv_std: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer)> {
    let grad_x = CudaBuffer::new(rows * cols)?;
    let grad_gamma = CudaBuffer::new(cols)?;
    let grad_beta = CudaBuffer::new(cols)?;
    let code = unsafe {
        ffi::fastnn_cuda_layer_norm_backward(
            grad.as_ptr(),
            x.as_ptr(),
            gamma.as_ptr(),
            mean.as_ptr(),
            inv_std.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_gamma.as_mut_ptr(),
            grad_beta.as_mut_ptr(),
            rows as c_int,
            cols as c_int,
        )
    };
    if code != 0 {
        return Err(Error::Cuda(format!(
            "layer_norm_backward failed (status {code})"
        )));
    }
    Ok((grad_x, grad_gamma, grad_beta))
}

/// Upload token ids for the embedding kernels.
///
/// `CudaBuffer` is typed as f32, but the kernel takes `int*`; both are 4 bytes,
/// so we reinterpret the slice and the kernel casts the pointer back.
pub fn upload_ids(ids: &[i32]) -> Result<CudaBuffer> {
    let as_f32 = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const f32, ids.len()) };
    CudaBuffer::from_slice(as_f32)
}

pub fn embedding_forward(
    ids: &CudaBuffer,
    weight: &CudaBuffer,
    n_ids: usize,
    dim: usize,
) -> Result<CudaBuffer> {
    launch(n_ids * dim, "embedding_forward", |out| unsafe {
        ffi::fastnn_cuda_embedding_forward(
            ids.as_ptr() as *const c_int,
            weight.as_ptr(),
            out.as_mut_ptr(),
            n_ids as c_int,
            dim as c_int,
        )
    })
}

pub fn embedding_backward(
    ids: &CudaBuffer,
    grad: &CudaBuffer,
    n_ids: usize,
    dim: usize,
    vocab: usize,
) -> Result<CudaBuffer> {
    // The kernel scatter-adds, so the accumulator must start at zero.
    let out = CudaBuffer::zeros(vocab * dim)?;
    let code = unsafe {
        ffi::fastnn_cuda_embedding_backward(
            ids.as_ptr() as *const c_int,
            grad.as_ptr(),
            out.as_mut_ptr(),
            n_ids as c_int,
            dim as c_int,
            vocab as c_int,
        )
    };
    if code != 0 {
        return Err(Error::Cuda(format!(
            "embedding_backward failed (status {code})"
        )));
    }
    Ok(out)
}

// ── Convolution lowering ─────────────────────────────────────────────────────

/// Unfold `[n, c, h, w]` into `[n, c·kh·kw, out_h·out_w]` columns on the device.
pub fn im2col(
    input: &CudaBuffer,
    (n, c, h, w): (usize, usize, usize, usize),
    (kh, kw): (usize, usize),
    (sh, sw): (usize, usize),
    (ph, pw): (usize, usize),
    (dh, dw): (usize, usize),
    (out_h, out_w): (usize, usize),
) -> Result<CudaBuffer> {
    launch(n * c * kh * kw * out_h * out_w, "im2col", |out| unsafe {
        ffi::fastnn_cuda_im2col(
            input.as_ptr(),
            out.as_mut_ptr(),
            n as c_int,
            c as c_int,
            h as c_int,
            w as c_int,
            kh as c_int,
            kw as c_int,
            sh as c_int,
            sw as c_int,
            ph as c_int,
            pw as c_int,
            dh as c_int,
            dw as c_int,
            out_h as c_int,
            out_w as c_int,
        )
    })
}

/// Fold columns back into `[n, c, h, w]`, summing where windows overlapped.
pub fn col2im(
    cols: &CudaBuffer,
    (n, c, h, w): (usize, usize, usize, usize),
    (kh, kw): (usize, usize),
    (sh, sw): (usize, usize),
    (ph, pw): (usize, usize),
    (dh, dw): (usize, usize),
    (out_h, out_w): (usize, usize),
) -> Result<CudaBuffer> {
    launch(n * c * h * w, "col2im", |out| unsafe {
        ffi::fastnn_cuda_col2im(
            cols.as_ptr(),
            out.as_mut_ptr(),
            n as c_int,
            c as c_int,
            h as c_int,
            w as c_int,
            kh as c_int,
            kw as c_int,
            sh as c_int,
            sw as c_int,
            ph as c_int,
            pw as c_int,
            dh as c_int,
            dw as c_int,
            out_h as c_int,
            out_w as c_int,
        )
    })
}

fn to_c_ints(xs: &[usize]) -> Vec<c_int> {
    xs.iter().map(|&x| x as c_int).collect()
}
