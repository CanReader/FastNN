use std::fmt;
use std::sync::{Arc, Mutex};

use crate::cuda::CudaBuffer;

/// Specifies where tensor data lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

/// The underlying storage for tensor data.
#[derive(Debug, Clone)]
pub enum TensorStorage {
    Cpu(Vec<f32>),
    Cuda(Arc<CudaBuffer>),
}

/// A multi-dimensional array supporting both CPU and CUDA backends.
///
/// Tensors track their shape, strides, device, and optionally maintain
/// gradient information for automatic differentiation.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: TensorStorage,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) device: Device,
    pub(crate) requires_grad: bool,
    pub(crate) grad: Arc<Mutex<Option<Box<Tensor>>>>,
    /// Unique ID for autograd graph tracking.
    pub(crate) id: u64,
}

static TENSOR_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_tensor_id() -> u64 {
    TENSOR_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Compute default row-major (C-contiguous) strides for a given shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl Tensor {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor from a flat Vec and shape on CPU.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "Data length {} doesn't match shape {:?} ({})", data.len(), shape, total);
        let strides = compute_strides(shape);
        Tensor {
            storage: TensorStorage::Cpu(data),
            shape: shape.to_vec(),
            strides,
            device: Device::Cpu,
            requires_grad: false,
            grad: Arc::new(Mutex::new(None)),
            id: next_tensor_id(),
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        Self::from_vec(vec![0.0; total], shape)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        Self::from_vec(vec![1.0; total], shape)
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: &[usize], value: f32) -> Self {
        let total: usize = shape.iter().product();
        Self::from_vec(vec![value; total], shape)
    }

    /// Create a 1D tensor with evenly spaced values.
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut val = start;
        while val < end {
            data.push(val);
            val += step;
        }
        let len = data.len();
        Self::from_vec(data, &[len])
    }

    /// Create a 1D tensor of `n` evenly spaced values from start to end (inclusive).
    pub fn linspace(start: f32, end: f32, n: usize) -> Self {
        let step = if n > 1 { (end - start) / (n - 1) as f32 } else { 0.0 };
        let data: Vec<f32> = (0..n).map(|i| start + step * i as f32).collect();
        Self::from_vec(data, &[n])
    }

    /// Create a tensor with random values from uniform distribution [0, 1).
    pub fn rand(shape: &[usize]) -> Self {
        use rand::Rng;
        let total: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..total).map(|_| rng.gen::<f32>()).collect();
        Self::from_vec(data, shape)
    }

    /// Create a tensor with random values from standard normal distribution.
    pub fn randn(shape: &[usize]) -> Self {
        use rand_distr::{Distribution, StandardNormal};
        let total: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..total).map(|_| StandardNormal.sample(&mut rng)).collect();
        Self::from_vec(data, shape)
    }

    /// Kaiming (He) uniform initialization for conv/linear weights.
    pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Self {
        use rand::Rng;
        let bound = (6.0f32 / fan_in as f32).sqrt();
        let total: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..total).map(|_| rng.gen_range(-bound..bound)).collect();
        Self::from_vec(data, shape)
    }

    /// Xavier (Glorot) uniform initialization.
    pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Self {
        use rand::Rng;
        let bound = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
        let total: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..total).map(|_| rng.gen_range(-bound..bound)).collect();
        Self::from_vec(data, shape)
    }

    /// Create an identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_vec(data, &[n, n])
    }

    // ========================================================================
    // Properties
    // ========================================================================

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get gradient tensor, if one has been computed.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.lock().unwrap().as_ref().map(|g| *g.clone())
    }

    /// Set the gradient tensor.
    pub fn set_grad(&self, grad: Tensor) {
        *self.grad.lock().unwrap() = Some(Box::new(grad));
    }

    /// Accumulate gradient (add to existing or set if none).
    pub fn accumulate_grad(&self, grad: &Tensor) {
        let mut guard = self.grad.lock().unwrap();
        match guard.as_mut() {
            Some(existing) => {
                let new_grad = existing.add(grad);
                **existing = new_grad;
            }
            None => {
                *guard = Some(Box::new(grad.clone()));
            }
        }
    }

    /// Zero the gradient.
    pub fn zero_grad(&self) {
        *self.grad.lock().unwrap() = None;
    }

    /// Clone of the grad `Arc<Mutex<...>>` cell. Used by the autograd graph to
    /// write accumulated gradients back into leaf tensors after `backward()`.
    pub(crate) fn grad_cell(&self) -> Arc<Mutex<Option<Box<Tensor>>>> {
        self.grad.clone()
    }

    /// Run reverse-mode autodiff from this scalar tensor. Populates `.grad()` on
    /// every leaf tensor in the graph that had `requires_grad = true`.
    ///
    /// The tensor must be a scalar (numel == 1), typically the output of a loss.
    /// After backward, the graph tape is cleared — call `enable_grad()` again
    /// before the next forward pass you want tracked.
    pub fn backward(&self) {
        assert_eq!(self.numel(), 1,
                   "backward() requires a scalar tensor (numel=1), got shape {:?}", self.shape);
        crate::autograd::graph::backward(self.id);
    }

    /// In-place SGD-style update: `self -= lr * grad`. Used by optimizers to
    /// mutate weights while preserving the `grad` Arc and `id` so future
    /// backward passes still reference the same leaf.
    pub fn apply_sgd_update(&mut self, lr: f32, update: &Tensor) {
        assert_eq!(self.shape, update.shape,
                   "apply_sgd_update shape mismatch: {:?} vs {:?}", self.shape, update.shape);
        match (&mut self.storage, &update.storage) {
            (TensorStorage::Cpu(data), TensorStorage::Cpu(u)) => {
                for (p, g) in data.iter_mut().zip(u.iter()) {
                    *p -= lr * g;
                }
            }
            (TensorStorage::Cuda(_), _) | (_, TensorStorage::Cuda(_)) => {
                let p = self.to_vec();
                let u = update.to_vec();
                let new_data: Vec<f32> = p.iter().zip(u.iter()).map(|(&pv, &gv)| pv - lr * gv).collect();
                let buf = CudaBuffer::from_slice(&new_data).expect("upload failed");
                self.storage = TensorStorage::Cuda(Arc::new(buf));
            }
        }
    }

    /// In-place data replacement, preserving `id` and `grad` Arc.
    /// Used by optimizers (Adam/AdamW) that compute a fresh parameter vector.
    pub fn set_data_from_vec(&mut self, data: Vec<f32>) {
        assert_eq!(data.len(), self.numel(),
                   "set_data_from_vec length mismatch: {} vs {}", data.len(), self.numel());
        match &mut self.storage {
            TensorStorage::Cpu(cpu) => { *cpu = data; }
            TensorStorage::Cuda(_) => {
                let buf = CudaBuffer::from_slice(&data).expect("upload failed");
                self.storage = TensorStorage::Cuda(Arc::new(buf));
            }
        }
    }

    // ========================================================================
    // Data Access
    // ========================================================================

    /// Get a reference to CPU data. Panics if tensor is on GPU.
    pub fn data(&self) -> &[f32] {
        match &self.storage {
            TensorStorage::Cpu(data) => data,
            TensorStorage::Cuda(_) => panic!("Cannot access GPU tensor data directly. Use .to_device(Device::Cpu) first."),
        }
    }

    /// Get a mutable reference to CPU data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            TensorStorage::Cpu(data) => data,
            TensorStorage::Cuda(_) => panic!("Cannot access GPU tensor data directly. Use .to_device(Device::Cpu) first."),
        }
    }

    /// Download tensor data to a Vec (works for both CPU and GPU).
    pub fn to_vec(&self) -> Vec<f32> {
        match &self.storage {
            TensorStorage::Cpu(data) => data.clone(),
            TensorStorage::Cuda(buf) => buf.to_vec().expect("Failed to download GPU tensor"),
        }
    }

    /// Get a single scalar value (tensor must have exactly 1 element).
    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() requires exactly 1 element, got {}", self.numel());
        self.to_vec()[0]
    }

    /// Access element by flat index.
    pub fn get_flat(&self, idx: usize) -> f32 {
        self.data()[idx]
    }

    /// Access element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f32 {
        assert_eq!(indices.len(), self.ndim());
        let flat_idx: usize = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum();
        self.data()[flat_idx]
    }

    // ========================================================================
    // Device Transfer
    // ========================================================================

    /// Move tensor to the specified device.
    pub fn to_device(&self, device: Device) -> Tensor {
        if self.device == device {
            return self.clone();
        }

        match (&self.device, &device) {
            (Device::Cpu, Device::Cuda(_)) => {
                let data = self.data();
                let buf = CudaBuffer::from_slice(data).expect("Failed to upload to GPU");
                Tensor {
                    storage: TensorStorage::Cuda(Arc::new(buf)),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    device,
                    requires_grad: self.requires_grad,
                    grad: Arc::new(Mutex::new(None)),
                    id: next_tensor_id(),
                }
            }
            (Device::Cuda(_), Device::Cpu) => {
                let data = self.to_vec();
                Tensor {
                    storage: TensorStorage::Cpu(data),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    device: Device::Cpu,
                    requires_grad: self.requires_grad,
                    grad: Arc::new(Mutex::new(None)),
                    id: next_tensor_id(),
                }
            }
            (Device::Cuda(_), Device::Cuda(_)) => {
                // GPU to GPU copy
                if let TensorStorage::Cuda(buf) = &self.storage {
                    let new_buf = (**buf).clone();
                    Tensor {
                        storage: TensorStorage::Cuda(Arc::new(new_buf)),
                        shape: self.shape.clone(),
                        strides: self.strides.clone(),
                        device,
                        requires_grad: self.requires_grad,
                        grad: Arc::new(Mutex::new(None)),
                        id: next_tensor_id(),
                    }
                } else {
                    unreachable!()
                }
            }
            _ => unreachable!(),
        }
    }

    /// Shorthand for `.to_device(Device::Cuda(0))`.
    pub fn cuda(&self) -> Tensor {
        self.to_device(Device::Cuda(0))
    }

    /// Shorthand for `.to_device(Device::Cpu)`.
    pub fn cpu(&self) -> Tensor {
        self.to_device(Device::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    // ========================================================================
    // Shape Manipulation
    // ========================================================================

    /// Reshape tensor to new shape. -1 can be used for one dimension.
    pub fn reshape(&self, new_shape: &[i64]) -> Tensor {
        let numel = self.numel();
        let mut inferred_idx: Option<usize> = None;
        let mut product: usize = 1;

        for (i, &s) in new_shape.iter().enumerate() {
            if s == -1 {
                assert!(inferred_idx.is_none(), "Can only infer one dimension");
                inferred_idx = Some(i);
            } else {
                assert!(s > 0, "Shape dimensions must be positive (got {})", s);
                product *= s as usize;
            }
        }

        let final_shape: Vec<usize> = if let Some(idx) = inferred_idx {
            assert!(numel % product == 0, "Cannot reshape {} elements into shape {:?}", numel, new_shape);
            new_shape.iter().enumerate().map(|(i, &s)| {
                if i == idx { numel / product } else { s as usize }
            }).collect()
        } else {
            let shape: Vec<usize> = new_shape.iter().map(|&s| s as usize).collect();
            assert_eq!(shape.iter().product::<usize>(), numel, "Shape mismatch");
            shape
        };

        let strides = compute_strides(&final_shape);
        let mut out = Tensor {
            storage: self.storage.clone(),
            shape: final_shape,
            strides,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(Mutex::new(None)),
            id: next_tensor_id(),
        };

        if crate::autograd::graph::is_grad_enabled() && self.requires_grad {
            out.requires_grad = true;
            let grad_fn = std::sync::Arc::new(
                crate::autograd::backward_ops::ReshapeBackward {
                    input_ids: vec![self.id],
                    input_shape: self.shape.clone(),
                }
            );
            crate::autograd::graph::record_op_with_cells(
                grad_fn, out.id,
                vec![(self.id, self.grad.clone())],
            );
        }

        out
    }

    /// Flatten to 1D.
    pub fn flatten(&self) -> Tensor {
        self.reshape(&[self.numel() as i64])
    }

    /// Flatten dimensions from `start_dim` to `end_dim` (inclusive).
    pub fn flatten_range(&self, start_dim: usize, end_dim: usize) -> Tensor {
        let end_dim = if end_dim >= self.ndim() { self.ndim() - 1 } else { end_dim };
        let mut new_shape = Vec::new();
        for i in 0..start_dim {
            new_shape.push(self.shape[i] as i64);
        }
        let flat: usize = self.shape[start_dim..=end_dim].iter().product();
        new_shape.push(flat as i64);
        for i in (end_dim + 1)..self.ndim() {
            new_shape.push(self.shape[i] as i64);
        }
        self.reshape(&new_shape)
    }

    /// Add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape.iter().map(|&s| s as i64).collect::<Vec<_>>())
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self) -> Tensor {
        let new_shape: Vec<i64> = self.shape.iter().filter(|&&s| s != 1).map(|&s| s as i64).collect();
        if new_shape.is_empty() {
            self.reshape(&[1])
        } else {
            self.reshape(&new_shape)
        }
    }

    /// Transpose the last two dimensions.
    pub fn t(&self) -> Tensor {
        assert!(self.ndim() >= 2, "t() requires at least 2 dimensions");
        let n = self.ndim();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.swap(n - 1, n - 2);
        self.permute(&perm)
    }

    /// Permute dimensions.
    pub fn permute(&self, dims: &[usize]) -> Tensor {
        assert_eq!(dims.len(), self.ndim());
        let new_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();
        let numel = self.numel();

        // ── CUDA fast path ──────────────────────────────────────────────────
        #[cfg(feature = "cuda")]
        if let TensorStorage::Cuda(buf) = &self.storage {
            use crate::tensor::cuda_backend;
            let out_strides = compute_strides(&new_shape);
            let out_buf = cuda_backend::cuda_permute_nd(buf, &out_strides, &self.strides, dims, numel)
                .expect("CUDA permute_nd failed");
            let mut result = Tensor::from_cuda_buffer(out_buf, new_shape, false);
            result.requires_grad = self.requires_grad;
            if crate::autograd::graph::is_grad_enabled() && self.requires_grad {
                let grad_fn = std::sync::Arc::new(crate::autograd::backward_ops::PermuteBackward {
                    input_ids: vec![self.id],
                    dims: dims.to_vec(),
                });
                crate::autograd::graph::record_op_with_cells(
                    grad_fn,
                    result.id,
                    vec![(self.id, self.grad.clone())],
                );
            }
            return result;
        }

        // ── CPU path ────────────────────────────────────────────────────────
        let data = self.to_vec();
        let mut new_data = vec![0.0f32; numel];
        let ndim = self.ndim();
        let old_strides = &self.strides;
        let new_strides_contiguous = compute_strides(&new_shape);

        for flat in 0..numel {
            let mut remaining = flat;
            let mut old_flat = 0usize;
            for d in 0..ndim {
                let coord = remaining / new_strides_contiguous[d];
                remaining %= new_strides_contiguous[d];
                old_flat += coord * old_strides[dims[d]];
            }
            new_data[flat] = data[old_flat];
        }

        let mut result = Tensor::from_vec(new_data, &new_shape);
        if self.is_cuda() {
            result = result.to_device(self.device);
        }
        result.requires_grad = self.requires_grad;

        if crate::autograd::graph::is_grad_enabled() && self.requires_grad {
            let grad_fn = std::sync::Arc::new(crate::autograd::backward_ops::PermuteBackward {
                input_ids: vec![self.id],
                dims: dims.to_vec(),
            });
            crate::autograd::graph::record_op_with_cells(
                grad_fn,
                result.id,
                vec![(self.id, self.grad.clone())],
            );
        }

        result
    }

    /// Expand (broadcast) tensor to a new shape.
    pub fn expand(&self, shape: &[usize]) -> Tensor {
        assert_eq!(shape.len(), self.ndim());
        let numel: usize = shape.iter().product();

        // ── CUDA fast path: reuse permute_nd with stride=0 for broadcast dims ──
        // Output element at coord (d0,d1,...) reads input at the same coord,
        // but broadcast dims always read index 0 (stride=0 maps everything to 0).
        #[cfg(feature = "cuda")]
        if let TensorStorage::Cuda(buf) = &self.storage {
            use crate::tensor::cuda_backend;
            let out_strides = compute_strides(shape);
            let in_strides: Vec<usize> = self.strides.iter().enumerate()
                .map(|(d, &s)| if self.shape[d] == 1 { 0 } else { s })
                .collect();
            let perm: Vec<usize> = (0..self.ndim()).collect();
            let out_buf = cuda_backend::cuda_permute_nd(buf, &out_strides, &in_strides, &perm, numel)
                .expect("CUDA expand failed");
            let mut result = Tensor::from_cuda_buffer(out_buf, shape.to_vec(), false);
            if crate::autograd::graph::is_grad_enabled() && self.requires_grad {
                result.requires_grad = true;
                let grad_fn = std::sync::Arc::new(
                    crate::autograd::backward_ops::ExpandBackward {
                        input_ids: vec![self.id],
                        input_shape: self.shape.clone(),
                    }
                );
                crate::autograd::graph::record_op_with_cells(
                    grad_fn, result.id,
                    vec![(self.id, self.grad.clone())],
                );
            }
            return result;
        }

        // ── CPU path ────────────────────────────────────────────────────────────
        let data = self.to_vec();
        let total = numel;
        let mut new_data = vec![0.0f32; total];
        let new_strides = compute_strides(shape);

        for flat in 0..total {
            let mut remaining = flat;
            let mut old_flat = 0usize;
            for d in 0..self.ndim() {
                let idx = remaining / new_strides[d];
                remaining %= new_strides[d];
                let old_idx = if self.shape[d] == 1 { 0 } else { idx };
                old_flat += old_idx * self.strides[d];
            }
            new_data[flat] = data[old_flat];
        }

        let mut result = Tensor::from_vec(new_data, shape);
        if self.is_cuda() {
            result = result.to_device(self.device);
        }

        if crate::autograd::graph::is_grad_enabled() && self.requires_grad {
            result.requires_grad = true;
            let grad_fn = std::sync::Arc::new(
                crate::autograd::backward_ops::ExpandBackward {
                    input_ids: vec![self.id],
                    input_shape: self.shape.clone(),
                }
            );
            crate::autograd::graph::record_op_with_cells(
                grad_fn, result.id,
                vec![(self.id, self.grad.clone())],
            );
        }

        result
    }

    /// Repeat the tensor along each dimension.
    pub fn repeat(&self, repeats: &[usize]) -> Tensor {
        assert_eq!(repeats.len(), self.ndim());
        let new_shape: Vec<usize> = self.shape.iter().zip(repeats).map(|(s, r)| s * r).collect();
        self.expand(&new_shape)
    }

    /// Concatenate tensors along a dimension.
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        let ndim = tensors[0].ndim();
        let mut new_shape = tensors[0].shape().to_vec();

        // Validate shapes match except along cat dimension
        for t in &tensors[1..] {
            assert_eq!(t.ndim(), ndim);
            for d in 0..ndim {
                if d != dim {
                    assert_eq!(t.shape()[d], new_shape[d], "Shape mismatch at dim {}", d);
                }
            }
            new_shape[dim] += t.shape()[dim];
        }

        let total: usize = new_shape.iter().product();
        let mut new_data = vec![0.0f32; total];
        let new_strides = compute_strides(&new_shape);

        let mut dim_offset = 0;
        for t in tensors {
            let t_data = t.to_vec();
            let t_strides = compute_strides(t.shape());
            let t_numel = t.numel();

            for flat in 0..t_numel {
                let mut remaining = flat;
                let mut new_flat = 0usize;
                for d in 0..ndim {
                    let idx = remaining / t_strides[d];
                    remaining %= t_strides[d];
                    let new_idx = if d == dim { idx + dim_offset } else { idx };
                    new_flat += new_idx * new_strides[d];
                }
                new_data[new_flat] = t_data[flat];
            }
            dim_offset += t.shape()[dim];
        }

        Tensor::from_vec(new_data, &new_shape)
    }

    /// Stack tensors along a new dimension.
    pub fn stack(tensors: &[&Tensor], dim: usize) -> Tensor {
        let unsqueezed: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        Tensor::cat(&refs, dim)
    }

    /// Detach tensor from autograd graph (returns tensor that doesn't require grad).
    pub fn detach(&self) -> Tensor {
        let mut t = self.clone();
        t.requires_grad = false;
        t.grad = Arc::new(Mutex::new(None));
        t.id = next_tensor_id();
        t
    }

    // ========================================================================
    // Get CudaBuffer reference (for kernel dispatch)
    // ========================================================================

    pub(crate) fn cuda_buffer(&self) -> &CudaBuffer {
        match &self.storage {
            TensorStorage::Cuda(buf) => buf,
            TensorStorage::Cpu(_) => panic!("Tensor is on CPU, not CUDA"),
        }
    }

    pub(crate) fn from_cuda_buffer(buf: CudaBuffer, shape: Vec<usize>, requires_grad: bool) -> Self {
        let strides = compute_strides(&shape);
        Tensor {
            storage: TensorStorage::Cuda(Arc::new(buf)),
            shape,
            strides,
            device: Device::Cuda(0),
            requires_grad,
            grad: Arc::new(Mutex::new(None)),
            id: next_tensor_id(),
        }
    }
}

// ============================================================================
// Display
// ============================================================================

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, device={}, requires_grad={})", self.shape, self.device, self.requires_grad)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.to_vec();
        if self.ndim() == 0 || self.numel() == 0 {
            return write!(f, "Tensor([])");
        }
        if self.ndim() == 1 {
            write!(f, "Tensor([")?;
            let max_show = 8;
            let n = data.len();
            if n <= max_show {
                for (i, v) in data.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:.4}", v)?;
                }
            } else {
                for i in 0..4 {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:.4}", data[i])?;
                }
                write!(f, ", ..., ")?;
                for i in (n-3)..n {
                    if i > n - 3 { write!(f, ", ")?; }
                    write!(f, "{:.4}", data[i])?;
                }
            }
            write!(f, "], shape={:?}, device={})", self.shape, self.device)
        } else if self.ndim() == 2 {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            writeln!(f, "Tensor([")?;
            let max_rows = 6;
            let max_cols = 6;
            let show_rows = rows.min(max_rows);
            for r in 0..show_rows {
                write!(f, "  [")?;
                let show_cols = cols.min(max_cols);
                for c in 0..show_cols {
                    if c > 0 { write!(f, ", ")?; }
                    write!(f, "{:8.4}", data[r * cols + c])?;
                }
                if cols > max_cols { write!(f, ", ...")?; }
                writeln!(f, "]")?;
            }
            if rows > max_rows { writeln!(f, "  ...")?; }
            write!(f, "], shape={:?}, device={})", self.shape, self.device)
        } else {
            write!(f, "Tensor(shape={:?}, device={}, numel={})", self.shape, self.device, self.numel())
        }
    }
}
