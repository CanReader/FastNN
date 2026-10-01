//! The trait every layer implements.

use crate::tensor::{Device, Tensor};

use super::param::{Buffer, Param};

/// A piece of a model.
///
/// Only [`forward`](Module::forward) is required. A layer with weights also
/// implements [`named_parameters`](Module::named_parameters); everything else —
/// parameter counting, device placement, gradient zeroing — falls out of that.
///
/// Every method takes `&self`. Parameters carry their own interior mutability,
/// so a container can recurse into its children without threading `&mut` through
/// the whole tree, and `model.eval()` needs no `mut` binding.
///
/// ```
/// use fastnn::prelude::*;
///
/// struct Residual(Linear);
///
/// impl Module for Residual {
///     fn forward(&self, x: &Tensor) -> Tensor {
///         x.add(&self.0.forward(x).relu())
///     }
///     fn named_parameters(&self) -> Vec<(String, Param)> {
///         self.0.named_parameters()
///     }
/// }
/// ```
pub trait Module: Send + Sync {
    /// Transform an input into an output.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// This module's learnable tensors, with names like `"weight"` or `"2.bias"`.
    ///
    /// A container prefixes its children's names; a leaf layer returns its own.
    fn named_parameters(&self) -> Vec<(String, Param)> {
        Vec::new()
    }

    /// Persistent non-learnable state, such as batch-norm running statistics.
    fn named_buffers(&self) -> Vec<(String, Buffer)> {
        Vec::new()
    }

    /// Switch between training and inference behaviour.
    ///
    /// Only [`Dropout`](super::Dropout) and [`BatchNorm2d`](super::BatchNorm2d)
    /// act on this; containers pass it down.
    fn set_training(&self, _training: bool) {}

    // ── Provided ─────────────────────────────────────────────────────────────

    /// Every learnable tensor. Hand this straight to an optimizer.
    fn parameters(&self) -> Vec<Param> {
        self.named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect()
    }

    /// Total number of learnable scalars.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Move every parameter and buffer to `device`.
    fn to_device(&self, device: Device) {
        for (_, p) in self.named_parameters() {
            p.to_device(device);
        }
        for (_, b) in self.named_buffers() {
            b.to_device(device);
        }
    }

    /// Stop training every parameter in this module.
    ///
    /// Freeze a pretrained backbone, leave the head trainable, and the optimizer
    /// updates only the head — the backward pass does not even compute the rest.
    fn freeze(&self) {
        for (_, p) in self.named_parameters() {
            p.freeze();
        }
    }

    /// Resume training every parameter in this module.
    fn unfreeze(&self) {
        for (_, p) in self.named_parameters() {
            p.unfreeze();
        }
    }

    /// The parameters an optimizer will actually update.
    fn trainable_parameters(&self) -> Vec<Param> {
        self.parameters()
            .into_iter()
            .filter(Param::is_trainable)
            .collect()
    }

    /// Discard all accumulated gradients.
    fn zero_grad(&self) {
        for (_, p) in self.named_parameters() {
            p.zero_grad();
        }
    }

    /// Enable training behaviour (dropout active, batch-norm using batch statistics).
    fn train(&self) {
        self.set_training(true);
    }

    /// Enable inference behaviour.
    fn eval(&self) {
        self.set_training(false);
    }
}

/// Prefix each name with `scope`, for a container building parameter paths.
///
/// ```
/// # use fastnn::prelude::*;
/// # use fastnn::nn::module::scoped;
/// # struct Net { encoder: Linear }
/// # impl Net {
/// fn named_parameters(&self) -> Vec<(String, Param)> {
///     scoped("encoder", self.encoder.named_parameters())  // -> "encoder.weight", ...
/// }
/// # }
/// ```
pub fn scoped<T>(scope: impl std::fmt::Display, items: Vec<(String, T)>) -> Vec<(String, T)> {
    items
        .into_iter()
        .map(|(name, item)| (format!("{scope}.{name}"), item))
        .collect()
}
