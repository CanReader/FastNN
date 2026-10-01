//! Shared handles to the tensors a model owns.
//!
//! A [`Param`] is a slot, not a value. Cloning one gives a second handle to the
//! same weight and the same gradient, which is why `Adam::new(model.parameters())`
//! works: the optimizer holds its own handles and updates them in place, with no
//! borrow of the model at step time.
//!
//! ```text
//!   Linear { weight: Param ─┐
//!                           ├─► Arc<RwLock<Tensor>>   the weight
//!   Adam   { params: [Param]┘   Arc<Mutex<Option<Tensor>>>   its gradient
//! ```
//!
//! [`Buffer`] is the same sharing without a gradient: state that persists and is
//! checkpointed but is not learned, such as batch-norm running statistics.

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

    /// This parameter as a tensor, for use in a forward pass.
    ///
    /// Normally a graph leaf, so gradients reaching it land in this parameter's
    /// slot. While frozen it comes back detached instead — the backward pass then
    /// stops at this point on its own, costing nothing to compute a gradient that
    /// would only be discarded.
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

    /// Stop training this parameter. Optimizers skip it and no gradient is
    /// computed for it at all.
    ///
    /// The usual reason is transfer learning: freeze a pretrained backbone and
    /// train only the new head on top.
    pub fn freeze(&self) {
        self.trainable.store(false, Ordering::Relaxed);
        self.grad.clear();
    }

    /// Resume training this parameter.
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

    /// Replace the value. Every handle sees the change; the gradient is untouched.
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

    /// Replace the gradient outright. Gradient clipping and custom optimizers
    /// use this; the backward pass itself accumulates instead.
    pub fn set_grad(&self, grad: Tensor) {
        self.grad.clear();
        self.grad.accumulate(grad.detach());
    }

    /// Move the value onto `device` in place. The gradient is dropped, since a
    /// stale gradient on the old device could never be applied.
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

/// Persistent state that is saved and restored but never learned.
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
