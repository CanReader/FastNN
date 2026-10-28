//! Shared handles to a model's tensors. Cloning a [`Param`] gives another handle
//! to the same weight and gradient, which is how an optimizer updates a model
//! it does not borrow. [`Buffer`] is the same without a gradient.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::autograd::GradSlot;
use crate::tensor::{Device, Tensor};

/// A learnable tensor plus its gradient.
#[derive(Clone)]
pub struct Param {
    value: Arc<RwLock<Tensor>>,
    grad: GradSlot,
    /// Shared with every clone, so freezing through one handle freezes them all.
    trainable: Arc<AtomicBool>,
}

impl Param {
    /// Wrap `value` as a learnable parameter.
    pub fn new(value: Tensor) -> Param {
        Param {
            value: Arc::new(RwLock::new(value.detach())),
            grad: GradSlot::new(),
            trainable: Arc::new(AtomicBool::new(true)),
        }
    }

    /// The parameter as a graph leaf for a forward pass. Detached while frozen,
    /// so backward never computes a gradient for it.
    pub fn tensor(&self) -> Tensor {
        let value = self.value.read().unwrap();
        if !self.is_trainable() {
            return value.detach();
        }
        Tensor::leaf(
            value.storage_arc(),
            value.shape().to_vec(),
            value.device(),
            self.grad.clone(),
        )
    }

    /// Stop training this parameter.
    pub fn freeze(&self) {
        self.trainable.store(false, Ordering::Relaxed);
        self.grad.clear();
    }

    /// Undo [`freeze`](Param::freeze).
    pub fn unfreeze(&self) {
        self.trainable.store(true, Ordering::Relaxed);
    }

    /// Whether optimizers should update this parameter.
    pub fn is_trainable(&self) -> bool {
        self.trainable.load(Ordering::Relaxed)
    }

    /// The current value, detached from the graph.
    pub fn value(&self) -> Tensor {
        self.value.read().unwrap().detach()
    }

    /// Replace the value for every handle. Shape must not change.
    pub fn set_value(&self, value: Tensor) {
        assert_eq!(
            value.shape(),
            self.value.read().unwrap().shape(),
            "set_value: shape must not change"
        );
        *self.value.write().unwrap() = value.detach();
    }

    /// The accumulated gradient, if `backward()` has reached this parameter.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.get()
    }

    /// Discard the accumulated gradient.
    pub fn zero_grad(&self) {
        self.grad.clear();
    }

    /// Replace the gradient, for clipping and custom optimizers.
    pub fn set_grad(&self, grad: Tensor) {
        self.grad.clear();
        self.grad.accumulate(grad.detach());
    }

    /// Move the value to `device`, dropping the gradient.
    pub fn to_device(&self, device: Device) {
        let mut value = self.value.write().unwrap();
        if value.device() != device {
            *value = value.to(device).detach();
            self.grad.clear();
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.value.read().unwrap().shape().to_vec()
    }

    pub fn numel(&self) -> usize {
        self.value.read().unwrap().numel()
    }

    pub fn device(&self) -> Device {
        self.value.read().unwrap().device()
    }
}

impl std::fmt::Debug for Param {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Param{:?} on {}", self.shape(), self.device())
    }
}

/// State that is checkpointed but not learned.
#[derive(Clone)]
pub struct Buffer {
    value: Arc<RwLock<Tensor>>,
}

impl Buffer {
    pub fn new(value: Tensor) -> Buffer {
        Buffer {
            value: Arc::new(RwLock::new(value.detach())),
        }
    }

    pub fn value(&self) -> Tensor {
        self.value.read().unwrap().detach()
    }

    pub fn set_value(&self, value: Tensor) {
        *self.value.write().unwrap() = value.detach();
    }

    pub fn to_device(&self, device: Device) {
        let mut value = self.value.write().unwrap();
        if value.device() != device {
            *value = value.to(device).detach();
        }
    }
}
