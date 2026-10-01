//! Every raw declaration of a symbol in `cuda/kernels.cu`, in one place.
//!
//! Nothing outside this module calls these directly — [`super::kernels`] wraps
//! them in safe, `Result`-returning functions. When you add a kernel to
//! `kernels.cu`, add its signature here and a wrapper there.
//!
//! Every function returns `0` on success and non-zero on failure.

#![allow(dead_code)]

use std::ffi::c_int;

extern "C" {
    // ── Context ──────────────────────────────────────────────────────────────
    pub fn fastnn_cuda_init(device_id: c_int) -> c_int;
    pub fn fastnn_cuda_device_count() -> c_int;
    pub fn fastnn_cuda_synchronize() -> c_int;
    pub fn fastnn_cuda_get_memory_info(free: *mut usize, total: *mut usize);

    // ── Memory ───────────────────────────────────────────────────────────────
    pub fn fastnn_cuda_malloc(ptr: *mut *mut f32, bytes: usize) -> c_int;
    pub fn fastnn_cuda_free(ptr: *mut f32) -> c_int;
    pub fn fastnn_cuda_memcpy_h2d(dst: *mut f32, src: *const f32, bytes: usize) -> c_int;
    pub fn fastnn_cuda_memcpy_d2h(dst: *mut f32, src: *const f32, bytes: usize) -> c_int;
    pub fn fastnn_cuda_memcpy_d2d(dst: *mut f32, src: *const f32, bytes: usize) -> c_int;
    pub fn fastnn_cuda_memset(ptr: *mut f32, value: c_int, bytes: usize) -> c_int;
    pub fn fastnn_cuda_fill(data: *mut f32, value: f32, n: usize) -> c_int;

    // ── Element-wise: binary, scalar, unary ──────────────────────────────────
    pub fn fastnn_cuda_add(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_sub(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_mul(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_div(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> c_int;

    pub fn fastnn_cuda_add_scalar(a: *const f32, s: f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_mul_scalar(a: *const f32, s: f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_pow_scalar(a: *const f32, s: f32, out: *mut f32, n: usize) -> c_int;

    pub fn fastnn_cuda_sqrt(a: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_abs(a: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_neg(a: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_exp(a: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_log(a: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_clamp(a: *const f32, lo: f32, hi: f32, out: *mut f32, n: usize) -> c_int;

    // ── Activations ──────────────────────────────────────────────────────────
    pub fn fastnn_cuda_relu(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_relu_backward(
        grad: *const f32,
        x: *const f32,
        out: *mut f32,
        n: usize,
    ) -> c_int;
    pub fn fastnn_cuda_sigmoid(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_sigmoid_backward(
        grad: *const f32,
        y: *const f32,
        out: *mut f32,
        n: usize,
    ) -> c_int;
    pub fn fastnn_cuda_tanh_forward(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_tanh_backward(
        grad: *const f32,
        y: *const f32,
        out: *mut f32,
        n: usize,
    ) -> c_int;
    pub fn fastnn_cuda_gelu(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_gelu_backward(
        grad: *const f32,
        x: *const f32,
        out: *mut f32,
        n: usize,
    ) -> c_int;
    pub fn fastnn_cuda_silu(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_silu_backward(
        grad: *const f32,
        x: *const f32,
        out: *mut f32,
        n: usize,
    ) -> c_int;
    pub fn fastnn_cuda_leaky_relu(x: *const f32, slope: f32, out: *mut f32, n: usize) -> c_int;

    // ── Softmax (over the last dimension) ────────────────────────────────────
    pub fn fastnn_cuda_softmax(x: *const f32, out: *mut f32, rows: c_int, cols: c_int) -> c_int;
    pub fn fastnn_cuda_softmax_backward(
        grad: *const f32,
        y: *const f32,
        out: *mut f32,
        rows: c_int,
        cols: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_log_softmax(x: *const f32, out: *mut f32, rows: c_int, cols: c_int)
        -> c_int;

    // ── LayerNorm (fused; saves mean and 1/std for the backward pass) ────────
    pub fn fastnn_cuda_layer_norm_forward(
        x: *const f32,
        gamma: *const f32,
        beta: *const f32,
        out: *mut f32,
        mean: *mut f32,
        inv_std: *mut f32,
        rows: c_int,
        cols: c_int,
        eps: f32,
    ) -> c_int;
    pub fn fastnn_cuda_layer_norm_backward(
        grad: *const f32,
        x: *const f32,
        gamma: *const f32,
        mean: *const f32,
        inv_std: *const f32,
        grad_x: *mut f32,
        grad_gamma: *mut f32,
        grad_beta: *mut f32,
        rows: c_int,
        cols: c_int,
    ) -> c_int;

    // ── Embedding ────────────────────────────────────────────────────────────
    pub fn fastnn_cuda_embedding_forward(
        ids: *const c_int,
        weight: *const f32,
        out: *mut f32,
        n_ids: c_int,
        dim: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_embedding_backward(
        ids: *const c_int,
        grad: *const f32,
        grad_weight: *mut f32,
        n_ids: c_int,
        dim: c_int,
        vocab: c_int,
    ) -> c_int;

    // ── Convolution lowering ─────────────────────────────────────────────────
    pub fn fastnn_cuda_im2col(
        input: *const f32,
        cols: *mut f32,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
        kh: c_int,
        kw: c_int,
        sh: c_int,
        sw: c_int,
        ph: c_int,
        pw: c_int,
        dh: c_int,
        dw: c_int,
        out_h: c_int,
        out_w: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_col2im(
        cols: *const f32,
        image: *mut f32,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
        kh: c_int,
        kw: c_int,
        sh: c_int,
        sw: c_int,
        ph: c_int,
        pw: c_int,
        dh: c_int,
        dw: c_int,
        out_h: c_int,
        out_w: c_int,
    ) -> c_int;

    // ── GEMM (cuBLAS). `nt`/`tn` transpose in place, with no staging buffer. ─
    pub fn fastnn_cuda_matmul(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
        lda: c_int,
        ldb: c_int,
        ldc: c_int,
        alpha: f32,
        beta: f32,
    ) -> c_int;
    pub fn fastnn_cuda_matmul_batched(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
        batch: c_int,
        alpha: f32,
        beta: f32,
    ) -> c_int;
    pub fn fastnn_cuda_matmul_nt(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_matmul_tn(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_matmul_batched_nt(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
        batch: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_matmul_batched_tn(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
        batch: c_int,
    ) -> c_int;

    // ── Layout ───────────────────────────────────────────────────────────────
    pub fn fastnn_cuda_transpose(x: *const f32, out: *mut f32, rows: c_int, cols: c_int) -> c_int;
    pub fn fastnn_cuda_transpose_batched(
        x: *const f32,
        out: *mut f32,
        batch: c_int,
        rows: c_int,
        cols: c_int,
    ) -> c_int;
    /// General gather: `out[i] = x[dot(coords(i, out_strides), in_strides[perm])]`.
    /// A stride of 0 in `in_strides` broadcasts that dimension, which is how
    /// `expand` reuses this kernel.
    pub fn fastnn_cuda_permute_nd(
        x: *const f32,
        out: *mut f32,
        out_strides: *const c_int,
        in_strides: *const c_int,
        perm: *const c_int,
        ndim: c_int,
        numel: c_int,
    ) -> c_int;

    // ── Reductions ───────────────────────────────────────────────────────────
    pub fn fastnn_cuda_sum(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_mean(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_max(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_min(x: *const f32, out: *mut f32, n: usize) -> c_int;
    pub fn fastnn_cuda_argmax(x: *const f32, out: *mut c_int, n: usize) -> c_int;
    pub fn fastnn_cuda_argmax_axis(
        x: *const f32,
        out: *mut c_int,
        outer: c_int,
        axis: c_int,
        inner: c_int,
    ) -> c_int;
    pub fn fastnn_cuda_sum_axis(
        x: *const f32,
        out: *mut f32,
        shape: *const c_int,
        ndim: c_int,
        axis: c_int,
        total: c_int,
    ) -> c_int;
}

/// Signature shared by every element-wise binary kernel.
pub type BinaryKernel = unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize) -> c_int;
/// Signature shared by every element-wise unary kernel.
pub type UnaryKernel = unsafe extern "C" fn(*const f32, *mut f32, usize) -> c_int;
/// Signature shared by every tensor-scalar kernel.
pub type ScalarKernel = unsafe extern "C" fn(*const f32, f32, *mut f32, usize) -> c_int;
