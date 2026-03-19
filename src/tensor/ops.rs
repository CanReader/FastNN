//! Tensor operations — dispatches to CPU or CUDA backend based on device.

use crate::tensor::tensor::{Tensor, TensorStorage, compute_strides};
use crate::tensor::cuda_backend;
use crate::cuda::CudaBuffer;

// ============================================================================
// Element-wise arithmetic
// ============================================================================

impl Tensor {
    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |a, b| a + b, cuda_backend::fastdl_cuda_add)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |a, b| a - b, cuda_backend::fastdl_cuda_sub)
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |a, b| a * b, cuda_backend::fastdl_cuda_mul)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |a, b| a / b, cuda_backend::fastdl_cuda_div)
    }

    /// Add scalar to all elements.
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        self.scalar_op(scalar, |a, s| a + s, cuda_backend::fastdl_cuda_add_scalar)
    }

    /// Multiply all elements by scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        self.scalar_op(scalar, |a, s| a * s, cuda_backend::fastdl_cuda_mul_scalar)
    }

    /// Raise all elements to a power.
    pub fn pow_scalar(&self, scalar: f32) -> Tensor {
        self.scalar_op(scalar, |a, s| a.powf(s), cuda_backend::fastdl_cuda_pow_scalar)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Tensor {
        self.unary_op(|a| a.sqrt(), cuda_backend::fastdl_cuda_sqrt)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Tensor {
        self.unary_op(|a| a.abs(), cuda_backend::fastdl_cuda_abs)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        self.unary_op(|a| -a, cuda_backend::fastdl_cuda_neg)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Tensor {
        self.unary_op(|a| a.exp(), cuda_backend::fastdl_cuda_exp)
    }

    /// Element-wise natural log.
    pub fn log(&self) -> Tensor {
        self.unary_op(|a| a.ln(), cuda_backend::fastdl_cuda_log)
    }

    /// Clamp all elements to [min_val, max_val].
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Tensor {
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let result: Vec<f32> = data.iter().map(|&a| a.clamp(min_val, max_val)).collect();
                Tensor::from_vec(result, &self.shape)
            }
            TensorStorage::Cuda(buf) => {
                let out = CudaBuffer::new(self.numel()).expect("alloc failed");
                unsafe {
                    cuda_backend::fastdl_cuda_clamp(buf.as_ptr(), min_val, max_val, out.ptr(), self.numel());
                }
                Tensor::from_cuda_buffer(out, self.shape.clone(), self.requires_grad)
            }
        }
    }

    // ========================================================================
    // Activations
    // ========================================================================

    pub fn relu(&self) -> Tensor {
        self.unary_op(|a| a.max(0.0), cuda_backend::fastdl_cuda_relu)
    }

    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(|a| 1.0 / (1.0 + (-a).exp()), cuda_backend::fastdl_cuda_sigmoid)
    }

    pub fn tanh_act(&self) -> Tensor {
        self.unary_op(|a| a.tanh(), cuda_backend::fastdl_cuda_tanh_forward)
    }

    pub fn gelu(&self) -> Tensor {
        self.unary_op(
            |a| 0.5 * a * (1.0 + libm::erff(a * std::f32::consts::FRAC_1_SQRT_2)),
            cuda_backend::fastdl_cuda_gelu,
        )
    }

    pub fn silu(&self) -> Tensor {
        self.unary_op(
            |a| a / (1.0 + (-a).exp()),
            cuda_backend::fastdl_cuda_silu,
        )
    }

    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let result: Vec<f32> = data.iter().map(|&a| if a > 0.0 { a } else { negative_slope * a }).collect();
                Tensor::from_vec(result, &self.shape)
            }
            TensorStorage::Cuda(buf) => {
                let out = CudaBuffer::new(self.numel()).expect("alloc failed");
                unsafe {
                    cuda_backend::fastdl_cuda_leaky_relu(buf.as_ptr(), negative_slope, out.ptr(), self.numel());
                }
                Tensor::from_cuda_buffer(out, self.shape.clone(), self.requires_grad)
            }
        }
    }

    /// Softmax along the last dimension.
    pub fn softmax(&self) -> Tensor {
        let shape = self.shape();
        let num_classes = *shape.last().unwrap();
        let batch_size = self.numel() / num_classes;

        match &self.storage {
            TensorStorage::Cpu(data) => {
                let mut result = vec![0.0f32; self.numel()];
                for b in 0..batch_size {
                    let offset = b * num_classes;
                    let row = &data[offset..offset + num_classes];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
                    for c in 0..num_classes {
                        result[offset + c] = (row[c] - max).exp() / exp_sum;
                    }
                }
                Tensor::from_vec(result, &self.shape)
            }
            TensorStorage::Cuda(buf) => {
                let out_buf = cuda_backend::cuda_softmax(buf, batch_size, num_classes)
                    .expect("CUDA softmax failed");
                Tensor::from_cuda_buffer(out_buf, self.shape.clone(), self.requires_grad)
            }
        }
    }

    /// Log-softmax along the last dimension.
    pub fn log_softmax(&self) -> Tensor {
        let shape = self.shape();
        let num_classes = *shape.last().unwrap();
        let batch_size = self.numel() / num_classes;

        match &self.storage {
            TensorStorage::Cpu(data) => {
                let mut result = vec![0.0f32; self.numel()];
                for b in 0..batch_size {
                    let offset = b * num_classes;
                    let row = &data[offset..offset + num_classes];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let log_sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
                    for c in 0..num_classes {
                        result[offset + c] = row[c] - max - log_sum_exp;
                    }
                }
                Tensor::from_vec(result, &self.shape)
            }
            TensorStorage::Cuda(buf) => {
                let out_buf = cuda_backend::cuda_log_softmax(buf, batch_size, num_classes)
                    .expect("CUDA log_softmax failed");
                Tensor::from_cuda_buffer(out_buf, self.shape.clone(), self.requires_grad)
            }
        }
    }

    // ========================================================================
    // Matrix operations
    // ========================================================================

    /// Matrix multiplication. Supports 2D and batched 3D tensors.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.ndim() >= 2 && other.ndim() >= 2, "matmul requires at least 2D tensors");
        assert_eq!(self.device, other.device, "Device mismatch");

        if self.ndim() == 2 && other.ndim() == 2 {
            self.matmul_2d(other)
        } else {
            self.matmul_batched(other)
        }
    }

    fn matmul_2d(&self, other: &Tensor) -> Tensor {
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        assert_eq!(k, other.shape[0], "Matmul shape mismatch: {:?} x {:?}", self.shape, other.shape);

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(a), TensorStorage::Cpu(b)) => {
                let mut result = vec![0.0f32; m * n];
                // Optimized cache-friendly matmul
                for i in 0..m {
                    for p in 0..k {
                        let a_val = a[i * k + p];
                        for j in 0..n {
                            result[i * n + j] += a_val * b[p * n + j];
                        }
                    }
                }
                Tensor::from_vec(result, &[m, n])
            }
            (TensorStorage::Cuda(a), TensorStorage::Cuda(b)) => {
                let out = cuda_backend::cuda_matmul(a, b, m, n, k)
                    .expect("CUDA matmul failed");
                Tensor::from_cuda_buffer(out, vec![m, n], self.requires_grad || other.requires_grad)
            }
            _ => panic!("Device mismatch"),
        }
    }

    fn matmul_batched(&self, other: &Tensor) -> Tensor {
        // Handle broadcasting for batched matmul
        let a_shape = self.shape();
        let b_shape = other.shape();

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];
        assert_eq!(k, b_shape[b_shape.len() - 2], "Matmul shape mismatch");

        // Calculate batch dimensions
        let batch_size: usize = a_shape[..a_shape.len() - 2].iter().product();
        let b_batch: usize = b_shape[..b_shape.len() - 2].iter().product();
        assert!(batch_size == b_batch || batch_size == 1 || b_batch == 1,
                "Batch dimensions must match or be broadcastable");
        let out_batch = batch_size.max(b_batch);

        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(a), TensorStorage::Cpu(b)) => {
                let mut result = vec![0.0f32; out_batch * m * n];
                for batch in 0..out_batch {
                    let a_batch = if batch_size == 1 { 0 } else { batch };
                    let b_batch_idx = if b_batch == 1 { 0 } else { batch };
                    let a_offset = a_batch * m * k;
                    let b_offset = b_batch_idx * k * n;
                    let c_offset = batch * m * n;
                    for i in 0..m {
                        for p in 0..k {
                            let a_val = a[a_offset + i * k + p];
                            for j in 0..n {
                                result[c_offset + i * n + j] += a_val * b[b_offset + p * n + j];
                            }
                        }
                    }
                }
                let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
                out_shape.push(m);
                out_shape.push(n);
                Tensor::from_vec(result, &out_shape)
            }
            (TensorStorage::Cuda(a), TensorStorage::Cuda(b)) => {
                let out = cuda_backend::cuda_matmul_batched(a, b, m, n, k, out_batch)
                    .expect("CUDA batched matmul failed");
                let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
                out_shape.push(m);
                out_shape.push(n);
                Tensor::from_cuda_buffer(out, out_shape, self.requires_grad || other.requires_grad)
            }
            _ => panic!("Device mismatch"),
        }
    }

    /// Transpose the last two dimensions (wrapper for GPU too).
    pub fn transpose(&self) -> Tensor {
        assert!(self.ndim() >= 2);
        let rows = self.shape[self.ndim() - 2];
        let cols = self.shape[self.ndim() - 1];

        match &self.storage {
            TensorStorage::Cpu(data) => {
                if self.ndim() == 2 {
                    let mut result = vec![0.0f32; data.len()];
                    for i in 0..rows {
                        for j in 0..cols {
                            result[j * rows + i] = data[i * cols + j];
                        }
                    }
                    Tensor::from_vec(result, &[cols, rows])
                } else {
                    self.t()
                }
            }
            TensorStorage::Cuda(buf) => {
                let out = cuda_backend::cuda_transpose_2d(buf, rows, cols)
                    .expect("CUDA transpose failed");
                let mut new_shape = self.shape.clone();
                let n = new_shape.len();
                new_shape.swap(n - 1, n - 2);
                Tensor::from_cuda_buffer(out, new_shape, self.requires_grad)
            }
        }
    }

    // ========================================================================
    // Reduction operations
    // ========================================================================

    /// Sum of all elements.
    pub fn sum(&self) -> Tensor {
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let s: f32 = data.iter().sum();
                Tensor::from_vec(vec![s], &[1])
            }
            TensorStorage::Cuda(buf) => {
                let out = cuda_backend::cuda_sum(buf, self.numel())
                    .expect("CUDA sum failed");
                Tensor::from_cuda_buffer(out, vec![1], false)
            }
        }
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Tensor {
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let s: f32 = data.iter().sum::<f32>() / data.len() as f32;
                Tensor::from_vec(vec![s], &[1])
            }
            TensorStorage::Cuda(buf) => {
                let out = cuda_backend::cuda_mean(buf, self.numel())
                    .expect("CUDA mean failed");
                Tensor::from_cuda_buffer(out, vec![1], false)
            }
        }
    }

    /// Max of all elements.
    pub fn max_val(&self) -> Tensor {
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let m = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                Tensor::from_vec(vec![m], &[1])
            }
            TensorStorage::Cuda(buf) => {
                let out = CudaBuffer::new(1).unwrap();
                unsafe { cuda_backend::fastdl_cuda_max(buf.as_ptr(), out.ptr(), self.numel()) };
                Tensor::from_cuda_buffer(out, vec![1], false)
            }
        }
    }

    /// Min of all elements.
    pub fn min_val(&self) -> Tensor {
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let m = data.iter().cloned().fold(f32::INFINITY, f32::min);
                Tensor::from_vec(vec![m], &[1])
            }
            TensorStorage::Cuda(buf) => {
                let out = CudaBuffer::new(1).unwrap();
                unsafe { cuda_backend::fastdl_cuda_min(buf.as_ptr(), out.ptr(), self.numel()) };
                Tensor::from_cuda_buffer(out, vec![1], false)
            }
        }
    }

    /// Sum along a specific axis.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.ndim(), "Axis out of range");
        let data = self.to_vec();
        let mut new_shape = self.shape.clone();
        let axis_size = new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        let out_numel: usize = new_shape.iter().product();
        let mut result = vec![0.0f32; out_numel];

        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();

        for o in 0..outer {
            for a in 0..axis_size {
                for i in 0..inner {
                    let src_idx = (o * axis_size + a) * inner + i;
                    let dst_idx = o * inner + i;
                    result[dst_idx] += data[src_idx];
                }
            }
        }

        Tensor::from_vec(result, &new_shape)
    }

    /// Mean along a specific axis.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let axis_size = self.shape[axis] as f32;
        self.sum_axis(axis).mul_scalar(1.0 / axis_size)
    }

    /// Argmax along a specific axis.
    pub fn argmax(&self, axis: usize) -> Vec<usize> {
        let data = self.to_vec();
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();

        let mut result = vec![0usize; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;
                for a in 0..axis_size {
                    let val = data[(o * axis_size + a) * inner + i];
                    if val > max_val {
                        max_val = val;
                        max_idx = a;
                    }
                }
                result[o * inner + i] = max_idx;
            }
        }
        result
    }

    /// Variance of all elements.
    pub fn var(&self) -> Tensor {
        let mean = self.mean().item();
        let data = self.to_vec();
        let var: f32 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / data.len() as f32;
        Tensor::from_vec(vec![var], &[1])
    }

    // ========================================================================
    // Comparison operations
    // ========================================================================

    /// Element-wise equality check, returns tensor of 0.0/1.0.
    pub fn eq_tensor(&self, other: &Tensor) -> Tensor {
        let a = self.to_vec();
        let b = other.to_vec();
        assert_eq!(a.len(), b.len());
        let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| if (x - y).abs() < 1e-7 { 1.0 } else { 0.0 }).collect();
        Tensor::from_vec(result, &self.shape)
    }

    /// Count number of matching elements.
    pub fn eq_count(&self, other: &Tensor) -> usize {
        let eq = self.eq_tensor(other);
        eq.to_vec().iter().filter(|&&v| v > 0.5).count()
    }

    // ========================================================================
    // Internal dispatch helpers
    // ========================================================================

    fn binary_op(
        &self, other: &Tensor,
        cpu_op: impl Fn(f32, f32) -> f32,
        cuda_op: unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize) -> i32,
    ) -> Tensor {
        assert_eq!(self.device, other.device, "Device mismatch: {:?} vs {:?}", self.device, other.device);

        // Handle broadcasting
        if self.shape != other.shape {
            return self.broadcast_binary_op(other, cpu_op, cuda_op);
        }

        let n = self.numel();
        match (&self.storage, &other.storage) {
            (TensorStorage::Cpu(a), TensorStorage::Cpu(b)) => {
                let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| cpu_op(x, y)).collect();
                Tensor::from_vec(result, &self.shape)
            }
            (TensorStorage::Cuda(a), TensorStorage::Cuda(b)) => {
                let out = cuda_backend::cuda_binary_op(a, b, n, cuda_op)
                    .expect("CUDA binary op failed");
                Tensor::from_cuda_buffer(out, self.shape.clone(), self.requires_grad || other.requires_grad)
            }
            _ => panic!("Device mismatch"),
        }
    }

    fn broadcast_binary_op(
        &self, other: &Tensor,
        cpu_op: impl Fn(f32, f32) -> f32,
        _cuda_op: unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize) -> i32,
    ) -> Tensor {
        // Compute broadcast shape
        let out_shape = broadcast_shape(&self.shape, &other.shape);
        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let out_numel: usize = out_shape.iter().product();
        let out_strides = compute_strides(&out_shape);

        let a_shape_padded = pad_shape(&self.shape, out_shape.len());
        let b_shape_padded = pad_shape(&other.shape, out_shape.len());
        let a_strides = compute_broadcast_strides(&a_shape_padded, &out_shape);
        let b_strides = compute_broadcast_strides(&b_shape_padded, &out_shape);

        let mut result = vec![0.0f32; out_numel];
        for flat in 0..out_numel {
            let mut remaining = flat;
            let mut a_flat = 0usize;
            let mut b_flat = 0usize;
            for d in 0..out_shape.len() {
                let idx = remaining / out_strides[d];
                remaining %= out_strides[d];
                a_flat += idx * a_strides[d];
                b_flat += idx * b_strides[d];
            }
            result[flat] = cpu_op(a_data[a_flat], b_data[b_flat]);
        }

        Tensor::from_vec(result, &out_shape)
    }

    fn unary_op(
        &self,
        cpu_op: impl Fn(f32) -> f32,
        cuda_op: unsafe extern "C" fn(*const f32, *mut f32, usize) -> i32,
    ) -> Tensor {
        let n = self.numel();
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let result: Vec<f32> = data.iter().map(|&x| cpu_op(x)).collect();
                Tensor::from_vec(result, &self.shape)
            }
            TensorStorage::Cuda(buf) => {
                let out = cuda_backend::cuda_unary_op(buf, n, cuda_op)
                    .expect("CUDA unary op failed");
                Tensor::from_cuda_buffer(out, self.shape.clone(), self.requires_grad)
            }
        }
    }

    fn scalar_op(
        &self, scalar: f32,
        cpu_op: impl Fn(f32, f32) -> f32,
        cuda_op: unsafe extern "C" fn(*const f32, f32, *mut f32, usize) -> i32,
    ) -> Tensor {
        let n = self.numel();
        match &self.storage {
            TensorStorage::Cpu(data) => {
                let result: Vec<f32> = data.iter().map(|&x| cpu_op(x, scalar)).collect();
                Tensor::from_vec(result, &self.shape)
            }
            TensorStorage::Cuda(buf) => {
                let out = cuda_backend::cuda_scalar_op(buf, scalar, n, cuda_op)
                    .expect("CUDA scalar op failed");
                Tensor::from_cuda_buffer(out, self.shape.clone(), self.requires_grad)
            }
        }
    }
}

// ============================================================================
// Operator overloading
// ============================================================================

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Tensor { self.add(rhs) }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Tensor { Tensor::sub(self, rhs) }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Tensor { self.mul(rhs) }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Tensor { Tensor::div(self, rhs) }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor { self.neg() }
}

// ============================================================================
// Broadcasting helpers
// ============================================================================

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_dims = a.len().max(b.len());
    let mut result = vec![0usize; max_dims];
    for i in 0..max_dims {
        let a_dim = if i < max_dims - a.len() { 1 } else { a[i - (max_dims - a.len())] };
        let b_dim = if i < max_dims - b.len() { 1 } else { b[i - (max_dims - b.len())] };
        assert!(a_dim == b_dim || a_dim == 1 || b_dim == 1,
                "Cannot broadcast shapes {:?} and {:?}", a, b);
        result[i] = a_dim.max(b_dim);
    }
    result
}

fn pad_shape(shape: &[usize], target_ndim: usize) -> Vec<usize> {
    let mut padded = vec![1usize; target_ndim - shape.len()];
    padded.extend_from_slice(shape);
    padded
}

fn compute_broadcast_strides(shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let orig_strides = compute_strides(shape);
    orig_strides.iter().zip(shape.iter()).zip(target_shape.iter())
        .map(|((&s, &dim), &_target)| if dim == 1 { 0 } else { s })
        .collect()
}
