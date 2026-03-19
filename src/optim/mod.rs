pub mod sgd;
pub mod adam;
pub mod scheduler;

pub use sgd::SGD;
pub use adam::{Adam, AdamW};
pub use scheduler::{LRScheduler, StepLR, CosineAnnealingLR, LinearWarmup, OneCycleLR};

use crate::tensor::Tensor;

/// Trait for all optimizers.
pub trait Optimizer {
    /// Perform a single optimization step.
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]);

    /// Zero all accumulated gradients.
    fn zero_grad(&self, params: &[Tensor]) {
        for p in params {
            p.zero_grad();
        }
    }

    /// Get the current learning rate.
    fn get_lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}
