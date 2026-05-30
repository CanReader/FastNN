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
    pub fn fastnn_cuda_softmax_backward(grad_output: *const f32, output: *const f32, grad_input: *mut f32, batch_size: i32, num_classes: i32) -> i32;
    pub fn fastnn_cuda_log_softmax(input: *const f32, output: *mut f32, batch_size: i32, num_classes: i32) -> i32;

    // Layer Normalisation
    pub fn fastnn_cuda_layer_norm_forward(
        input: *const f32, gamma: *const f32, beta: *const f32,
        output: *mut f32, mean: *mut f32, inv_var: *mut f32,
        batch_size: i32, normalized_size: i32, epsilon: f32) -> i32;
    pub fn fastnn_cuda_layer_norm_backward(
        grad_output: *const f32, input: *const f32,
        gamma: *const f32, mean: *const f32, inv_var: *const f32,
        grad_input: *mut f32, grad_gamma: *mut f32, grad_beta: *mut f32,
        batch_size: i32, normalized_size: i32) -> i32;

    // Embedding
    pub fn fastnn_cuda_embedding_forward(
        indices: *const i32, weight: *const f32, output: *mut f32,
        num_indices: i32, embedding_dim: i32) -> i32;
    pub fn fastnn_cuda_embedding_backward(
        indices: *const i32, grad_output: *const f32, grad_weight: *mut f32,
        num_indices: i32, embedding_dim: i32, num_embeddings: i32) -> i32;

    // Matrix ops
    pub fn fastnn_cuda_matmul(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, lda: i32, ldb: i32, ldc: i32, alpha: f32, beta: f32) -> i32;
    pub fn fastnn_cuda_matmul_batched(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, batch_size: i32, alpha: f32, beta: f32) -> i32;
    pub fn fastnn_cuda_matmul_nt(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32) -> i32;
    pub fn fastnn_cuda_matmul_tn(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32) -> i32;
    pub fn fastnn_cuda_matmul_batched_nt(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, batch: i32) -> i32;
    pub fn fastnn_cuda_matmul_batched_tn(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, batch: i32) -> i32;
    pub fn fastnn_cuda_transpose(input: *const f32, output: *mut f32, rows: i32, cols: i32) -> i32;
    pub fn fastnn_cuda_transpose_batched(input: *const f32, output: *mut f32, batch: i32, rows: i32, cols: i32) -> i32;
    pub fn fastnn_cuda_permute_nd(input: *const f32, output: *mut f32, out_strides: *const i32, in_strides: *const i32, perm: *const i32, ndim: i32, numel: i32) -> i32;

    // Axis-wise sum reduction
    pub fn fastnn_cuda_sum_axis(input: *const f32, output: *mut f32, shape: *const i32, ndim: i32, axis: i32, total: i32) -> i32;

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

/// Transpose on GPU (2D only).
pub fn cuda_transpose_2d(a: &CudaBuffer, rows: usize, cols: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(rows * cols)?;
    let ret = unsafe { fastnn_cuda_transpose(a.as_ptr(), out.ptr(), rows as i32, cols as i32) };
    if ret != 0 { return Err("CUDA transpose failed".to_string()); }
    Ok(out)
}

/// Sum along one axis on GPU.
pub fn cuda_sum_axis(input: &CudaBuffer, shape: &[usize], axis: usize) -> Result<CudaBuffer, String> {
    let total = input.len();
    let axis_size = shape[axis];
    let out_numel = total / axis_size;
    let out = CudaBuffer::new(out_numel)?;
    let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    let ret = unsafe {
        fastnn_cuda_sum_axis(
            input.as_ptr(), out.ptr(),
            shape_i32.as_ptr(), shape.len() as i32, axis as i32, total as i32,
        )
    };
    if ret != 0 { return Err("CUDA sum_axis failed".to_string()); }
    Ok(out)
}

/// C (m×n) = A (m×k) × B^T (k×n), B stored (n×k) row-major.
pub fn cuda_matmul_nt(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(m * n)?;
    let ret = unsafe { fastnn_cuda_matmul_nt(a.as_ptr(), b.as_ptr(), out.ptr(), m as i32, n as i32, k as i32) };
    if ret != 0 { return Err("CUDA matmul_nt failed".to_string()); }
    Ok(out)
}

/// C (m×n) = A^T (m×k) × B (k×n), A stored (k×m) row-major.
pub fn cuda_matmul_tn(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(m * n)?;
    let ret = unsafe { fastnn_cuda_matmul_tn(a.as_ptr(), b.as_ptr(), out.ptr(), m as i32, n as i32, k as i32) };
    if ret != 0 { return Err("CUDA matmul_tn failed".to_string()); }
    Ok(out)
}

/// Batched C (batch×m×n) = A (batch×m×k) × B^T (batch×k×n), B stored (batch×n×k).
pub fn cuda_matmul_batched_nt(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize, batch: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(batch * m * n)?;
    let ret = unsafe { fastnn_cuda_matmul_batched_nt(a.as_ptr(), b.as_ptr(), out.ptr(), m as i32, n as i32, k as i32, batch as i32) };
    if ret != 0 { return Err("CUDA matmul_batched_nt failed".to_string()); }
    Ok(out)
}

/// Batched C (batch×m×n) = A^T (batch×m×k) × B (batch×k×n), A stored (batch×k×m).
pub fn cuda_matmul_batched_tn(a: &CudaBuffer, b: &CudaBuffer, m: usize, n: usize, k: usize, batch: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(batch * m * n)?;
    let ret = unsafe { fastnn_cuda_matmul_batched_tn(a.as_ptr(), b.as_ptr(), out.ptr(), m as i32, n as i32, k as i32, batch as i32) };
    if ret != 0 { return Err("CUDA matmul_batched_tn failed".to_string()); }
    Ok(out)
}

/// Batched transpose of the last two dims: input [batch, rows, cols] → [batch, cols, rows].
pub fn cuda_transpose_batched(a: &CudaBuffer, batch: usize, rows: usize, cols: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(batch * rows * cols)?;
    let ret = unsafe { fastnn_cuda_transpose_batched(a.as_ptr(), out.ptr(), batch as i32, rows as i32, cols as i32) };
    if ret != 0 { return Err("CUDA batched transpose failed".to_string()); }
    Ok(out)
}

/// N-D permute on GPU.
pub fn cuda_permute_nd(
    input: &CudaBuffer,
    out_strides: &[usize],
    in_strides: &[usize],
    perm: &[usize],
    numel: usize,
) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(numel)?;
    let out_s: Vec<i32> = out_strides.iter().map(|&x| x as i32).collect();
    let in_s: Vec<i32> = in_strides.iter().map(|&x| x as i32).collect();
    let p: Vec<i32> = perm.iter().map(|&x| x as i32).collect();
    let ret = unsafe {
        fastnn_cuda_permute_nd(
            input.as_ptr(), out.ptr(),
            out_s.as_ptr(), in_s.as_ptr(), p.as_ptr(),
            perm.len() as i32, numel as i32,
        )
    };
    if ret != 0 { return Err("CUDA permute_nd failed".to_string()); }
    Ok(out)
}

/// Sum reduction on GPU.
pub fn cuda_sum(a: &CudaBuffer, n: usize) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(1)?;
    let ret = unsafe { fastnn_cuda_sum(a.as_ptr(), out.ptr(), n) };
    if ret != 0 { return Err("CUDA sum failed".to_string()); }
    Ok(out)
}

/// Upload integer indices to GPU as a CudaBuffer.
/// Safe because i32 and f32 are both 4 bytes; the CUDA kernel casts back to int*.
pub fn cuda_upload_indices(indices: &[i32]) -> Result<CudaBuffer, String> {
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(indices.as_ptr() as *const f32, indices.len())
    };
    CudaBuffer::from_slice(floats)
}

/// LayerNorm forward. Returns (output, mean, inv_var) all on GPU.
pub fn cuda_layer_norm_forward(
    input: &CudaBuffer, gamma: &CudaBuffer, beta: &CudaBuffer,
    batch: usize, norm_size: usize, eps: f32,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    let out = CudaBuffer::new(batch * norm_size)?;
    let mean = CudaBuffer::new(batch)?;
    let inv_var = CudaBuffer::new(batch)?;
    let ret = unsafe {
        fastnn_cuda_layer_norm_forward(
            input.as_ptr(), gamma.as_ptr(), beta.as_ptr(),
            out.ptr(), mean.ptr(), inv_var.ptr(),
            batch as i32, norm_size as i32, eps,
        )
    };
    if ret != 0 { return Err("CUDA layer_norm_forward failed".to_string()); }
    Ok((out, mean, inv_var))
}

/// LayerNorm backward. Returns (grad_input, grad_gamma, grad_beta) all on GPU.
pub fn cuda_layer_norm_backward(
    grad_output: &CudaBuffer, input: &CudaBuffer, gamma: &CudaBuffer,
    mean: &CudaBuffer, inv_var: &CudaBuffer,
    batch: usize, norm_size: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer), String> {
    let gi = CudaBuffer::new(batch * norm_size)?;
    let gg = CudaBuffer::new(norm_size)?;
    let gb = CudaBuffer::new(norm_size)?;
    let ret = unsafe {
        fastnn_cuda_layer_norm_backward(
            grad_output.as_ptr(), input.as_ptr(), gamma.as_ptr(),
            mean.as_ptr(), inv_var.as_ptr(),
            gi.ptr(), gg.ptr(), gb.ptr(),
            batch as i32, norm_size as i32,
        )
    };
    if ret != 0 { return Err("CUDA layer_norm_backward failed".to_string()); }
    Ok((gi, gg, gb))
}

/// Embedding gather on GPU. `indices_buf` holds i32 indices as reinterpreted f32 bytes.
pub fn cuda_embedding_forward(
    indices_buf: &CudaBuffer, weight: &CudaBuffer,
    num_indices: usize, emb_dim: usize,
) -> Result<CudaBuffer, String> {
    let out = CudaBuffer::new(num_indices * emb_dim)?;
    let ret = unsafe {
        fastnn_cuda_embedding_forward(
            indices_buf.as_ptr() as *const i32,
            weight.as_ptr(), out.ptr(),
            num_indices as i32, emb_dim as i32,
        )
    };
    if ret != 0 { return Err("CUDA embedding_forward failed".to_string()); }
    Ok(out)
}

/// Embedding backward scatter-add on GPU. Returns grad_weight buffer.
pub fn cuda_embedding_backward(
    indices_buf: &CudaBuffer, grad_output: &CudaBuffer,
    num_indices: usize, emb_dim: usize, vocab_size: usize,
) -> Result<CudaBuffer, String> {
    let gw = CudaBuffer::new(vocab_size * emb_dim)?;
    let ret = unsafe {
        fastnn_cuda_embedding_backward(
            indices_buf.as_ptr() as *const i32,
            grad_output.as_ptr(), gw.ptr(),
            num_indices as i32, emb_dim as i32, vocab_size as i32,
        )
    };
    if ret != 0 { return Err("CUDA embedding_backward failed".to_string()); }
    Ok(gw)
}

/// Softmax backward on GPU.
pub fn cuda_softmax_backward(
    grad_output: &CudaBuffer, softmax_output: &CudaBuffer,
    batch: usize, classes: usize,
) -> Result<CudaBuffer, String> {
    let gi = CudaBuffer::new(batch * classes)?;
    let ret = unsafe {
        fastnn_cuda_softmax_backward(
            grad_output.as_ptr(), softmax_output.as_ptr(), gi.ptr(),
            batch as i32, classes as i32,
        )
    };
    if ret != 0 { return Err("CUDA softmax_backward failed".to_string()); }
    Ok(gi)
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
