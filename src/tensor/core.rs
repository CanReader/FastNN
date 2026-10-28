//! The [`Tensor`] type: shape, storage, and its link into the autograd graph.

use std::sync::Arc;

use crate::autograd::{self, Backward, GradSlot, Node};

use super::shape;
use super::storage::Storage;
use super::Device;

/// An n-dimensional `f32` array on the CPU or a GPU.
///
/// Immutable, and cloning only bumps a refcount. A tensor with a graph node
/// takes part in autograd, one without is plain data (see [`grad_fn`](Tensor::grad_fn)).
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    device: Device,
    node: Option<Node>,
}

impl Tensor {
    /// Wrap `data` in a tensor of `shape` on the CPU.
    ///
    /// Panics if `data.len()` does not match the shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        assert_eq!(
            data.len(),
            shape::numel(shape),
            "from_vec: {} values do not fill shape {shape:?}",
            data.len()
        );
        Tensor::raw(Storage::Cpu(data), shape.to_vec(), Device::Cpu)
    }

    /// A single-element tensor of shape `[1]`.
    pub fn scalar(value: f32) -> Tensor {
        Tensor::from_vec(vec![value], &[1])
    }

    /// Wrap storage an op already produced, possibly on the GPU.
    pub(crate) fn raw(storage: Storage, shape: Vec<usize>, device: Device) -> Tensor {
        Tensor::raw_shared(Arc::new(storage), shape, device)
    }

    /// Reuse an allocation under a new shape, which is what makes `reshape` free.
    pub(crate) fn raw_shared(storage: Arc<Storage>, shape: Vec<usize>, device: Device) -> Tensor {
        Tensor {
            storage,
            shape,
            device,
            node: None,
        }
    }

    /// A tensor whose gradient accumulates into `slot`. Backs [`Param`](crate::nn::Param).
    pub(crate) fn leaf(
        storage: Arc<Storage>,
        shape: Vec<usize>,
        device: Device,
        slot: GradSlot,
    ) -> Tensor {
        Tensor {
            storage,
            shape,
            device,
            node: Some(Node::Leaf(slot)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        shape::numel(&self.shape)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    /// The size of dimension `axis`. Panics if out of range.
    pub fn dim(&self, axis: usize) -> usize {
        assert!(
            axis < self.ndim(),
            "dim {axis} out of range for shape {:?}",
            self.shape
        );
        self.shape[axis]
    }

    /// The last dimension.
    pub fn last_dim(&self) -> usize {
        *self.shape.last().expect("tensor has no dimensions")
    }

    /// A host copy of the data.
    pub fn to_vec(&self) -> Vec<f32> {
        self.storage.to_vec()
    }

    /// The one value in a single-element tensor. Panics otherwise.
    pub fn item(&self) -> f32 {
        assert_eq!(
            self.numel(),
            1,
            "item() needs one element, got shape {:?}",
            self.shape
        );
        self.to_vec()[0]
    }

    /// The host slice, without copying. `None` if the tensor is on a GPU.
    pub fn as_slice(&self) -> Option<&[f32]> {
        self.storage.as_slice()
    }

    pub(crate) fn storage(&self) -> &Storage {
        &self.storage
    }

    pub(crate) fn storage_arc(&self) -> Arc<Storage> {
        self.storage.clone()
    }

    /// A copy of this tensor on `device`, differentiable in both directions.
    pub fn to(&self, device: Device) -> Tensor {
        if self.device == device {
            return self.clone();
        }
        let moved = Tensor::raw(self.storage.to_device(device), self.shape.clone(), device);
        let from = self.device;
        moved.with_grad(&[self], || ToDevice { from })
    }

    /// Shorthand for `to(Device::Cpu)`.
    pub fn cpu(&self) -> Tensor {
        self.to(Device::Cpu)
    }

    /// Shorthand for `to(Device::Cuda(0))`. Panics without a GPU, see [`Device::cuda`].
    pub fn cuda(&self) -> Tensor {
        self.to(Device::cuda(0).expect("cuda: no device"))
    }

    /// The graph node that produced this tensor. `None` for plain data, such as
    /// a literal or anything built inside [`no_grad`](crate::autograd::no_grad).
    pub fn grad_fn(&self) -> Option<&Node> {
        self.node.as_ref()
    }

    /// The same data with its graph link cut. Gradients stop here.
    pub fn detach(&self) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            device: self.device,
            node: None,
        }
    }

    /// Propagate gradients from this scalar loss back to every parameter that fed it.
    pub fn backward(&self) {
        autograd::backward(self);
    }

    /// Attach `rule` as the backward if tracking is on and any input needs it.
    /// It is a closure so inference never pays for saved values. Gradients must
    /// come back in the order of `inputs`.
    pub(crate) fn with_grad<R>(mut self, inputs: &[&Tensor], rule: impl FnOnce() -> R) -> Tensor
    where
        R: Backward + 'static,
    {
        if autograd::is_enabled() && inputs.iter().any(|t| t.node.is_some()) {
            let saved = inputs.iter().map(|t| (*t).clone()).collect();
            self.node = Some(Node::op(rule(), saved));
        }
        self
    }
}

/// The gradient of a device copy is a copy back.
struct ToDevice {
    from: Device,
}

impl Backward for ToDevice {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![grad.to(self.from)]
    }
    fn name(&self) -> &'static str {
        "ToDevice"
    }
}
