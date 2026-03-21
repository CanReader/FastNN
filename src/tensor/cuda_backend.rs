//! CUDA backend for tensor operations — dispatches to CUDA kernels via FFI.

use crate::cuda::CudaBuffer;

// ============================================================================
// FFI declarations for CUDA kernels
// ============================================================================
extern "C" {
    // Element-wise
    pub fn fastnn_cuda_add(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_sub(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_mul(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_div(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_add_scalar(a: *const f32, scalar: f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_mul_scalar(a: *const f32, scalar: f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_pow_scalar(a: *const f32, scalar: f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_sqrt(a: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_abs(a: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_neg(a: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_exp(a: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_log(a: *const f32, out: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_clamp(a: *const f32, min_val: f32, max_val: f32, out: *mut f32, n: usize) -> i32;

    // Activations
    pub fn fastnn_cuda_relu(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_relu_backward(grad_output: *const f32, input: *const f32, grad_input: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_sigmoid(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_sigmoid_backward(grad_output: *const f32, output: *const f32, grad_input: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_tanh_forward(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_tanh_backward(grad_output: *const f32, output: *const f32, grad_input: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_gelu(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_gelu_backward(grad_output: *const f32, input: *const f32, grad_input: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_silu(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_silu_backward(grad_output: *const f32, input: *const f32, grad_input: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_leaky_relu(input: *const f32, neg_slope: f32, output: *mut f32, n: usize) -> i32;

    // Softmax
    pub fn fastnn_cuda_softmax(input: *const f32, output: *mut f32, batch_size: i32, num_classes: i32) -> i32;
    pub fn fastnn_cuda_log_softmax(input: *const f32, output: *mut f32, batch_size: i32, num_classes: i32) -> i32;

    // Matrix ops
    pub fn fastnn_cuda_matmul(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, lda: i32, ldb: i32, ldc: i32, alpha: f32, beta: f32) -> i32;
    pub fn fastnn_cuda_matmul_batched(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, batch_size: i32, alpha: f32, beta: f32) -> i32;
    pub fn fastnn_cuda_transpose(input: *const f32, output: *mut f32, rows: i32, cols: i32) -> i32;

    // Reductions
    pub fn fastnn_cuda_sum(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_mean(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_max(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_min(input: *const f32, output: *mut f32, n: usize) -> i32;
    pub fn fastnn_cuda_argmax(input: *const f32, output: *mut i32, n: usize) -> i32;
    pub fn fastnn_cuda_argmax_axis(input: *const f32, output: *mut i32, outer: i32, axis_size: i32, inner: i32) -> i32;

    // Fill/copy
    pub fn fastnn_cuda_fill(data: *mut f32, value: f32, n: usize) -> i32;
}

/// Helper to run an element-wise binary operation on GPU.
pub fn cuda_binary_op(
    a: &CudaBuffer, b: &CudaBuffer, n: usize,
    op: unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize) -> i32,
) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(n)?;
    let ret = unsafe { op(a.as_ptr(), b.as_ptr(), out.ptr(), n) };
    if ret != 0 { return Err("CUDA binary op failed".to_string()); }
    Ok(out)
}

/// Helper to run an element-wise unary operation on GPU.
pub fn cuda_unary_op(
    a: &CudaBuffer, n: usize,
    op: unsafe extern "C" fn(*const f32, *mut f32, usize) -> i32,
) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(n)?;
    let ret = unsafe { op(a.as_ptr(), out.ptr(), n) };
    if ret != 0 { return Err("CUDA unary op failed".to_string()); }
    Ok(out)
}

/// Scalar operation on GPU.
pub fn cuda_scalar_op(
    a: &CudaBuffer, scalar: f32, n: usize,
    op: unsafe extern "C" fn(*const f32, f32, *mut f32, usize) -> i32,
) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(n)?;
    let ret = unsafe { op(a.as_ptr(), scalar, out.ptr(), n) };
    if ret != 0 { return Err("CUDA scalar op failed".to_string()); }
    Ok(out)
}

/// Matrix multiplication on GPU.
pub fn cuda_matmul(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(m * n)?;
    let ret = unsafe {
        fastnn_cuda_matmul(
            a.as_ptr(), b.as_ptr(), out.ptr(),
            m as i32, n as i32, k as i32,
            k as i32, n as i32, n as i32,
            1.0, 0.0
        )
    };
    if ret != 0 { return Err("CUDA matmul failed".to_string()); }
    Ok(out)
}

/// Batched matrix multiplication on GPU.
pub fn cuda_matmul_batched(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize, batch: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(batch * m * n)?;
    let ret = unsafe {
        fastnn_cuda_matmul_batched(
            a.as_ptr(), b.as_ptr(), out.ptr(),
            m as i32, n as i32, k as i32, batch as i32,
            1.0, 0.0
        )
    };
    if ret != 0 { return Err("CUDA batched matmul failed".to_string()); }
    Ok(out)
}

/// Transpose on GPU.
pub fn cuda_transpose_2d(a: &CudaBuffer, rows: usize, cols: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(rows * cols)?;
    let ret = unsafe { fastnn_cuda_transpose(a.as_ptr(), out.ptr(), rows as i32, cols as i32) };
    if ret != 0 { return Err("CUDA transpose failed".to_string()); }
    Ok(out)
}

/// Sum reduction on GPU.
pub fn cuda_sum(a: &CudaBuffer, n: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(1)?;
    let ret = unsafe { fastnn_cuda_sum(a.as_ptr(), out.ptr(), n) };
    if ret != 0 { return Err("CUDA sum failed".to_string()); }
    Ok(out)
}

/// Mean reduction on GPU.
pub fn cuda_mean(a: &CudaBuffer, n: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(1)?;
    let ret = unsafe { fastnn_cuda_mean(a.as_ptr(), out.ptr(), n) };
    if ret != 0 { return Err("CUDA mean failed".to_string()); }
    Ok(out)
}

/// Softmax on GPU.
pub fn cuda_softmax(a: &CudaBuffer, batch_size: usize, num_classes: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(batch_size * num_classes)?;
    let ret = unsafe { fastnn_cuda_softmax(a.as_ptr(), out.ptr(), batch_size as i32, num_classes as i32) };
    if ret != 0 { return Err("CUDA softmax failed".to_string()); }
    Ok(out)
}

/// Log-softmax on GPU.
pub fn cuda_log_softmax(a: &CudaBuffer, batch_size: usize, num_classes: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(batch_size * num_classes)?;
    let ret = unsafe { fastnn_cuda_log_softmax(a.as_ptr(), out.ptr(), batch_size as i32, num_classes as i32) };
    if ret != 0 { return Err("CUDA log_softmax failed".to_string()); }
    Ok(out)
}
