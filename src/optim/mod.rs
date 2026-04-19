pub mod sgd;
pub mod adam;
pub mod scheduler;

pub use sgd::SGD;
pub use adam::{Adam, AdamW};
pub use scheduler::{LRScheduler, StepLR, CosineAnnealingLR, LinearWarmup, OneCycleLR};

use crate::tensor::Tensor;

/// Trait for all optimizers.
///
/// Optimizers read `.grad()` from each parameter (populated by `loss.backward()`)
/// and update the parameter's data in place via `apply_sgd_update` or
/// `set_data_from_vec`. Pass `model.parameters_mut()` as `params`.
pub trait Optimizer {
    /// Perform a single optimization step. Reads `param.grad()` for each param
    /// and applies the update in place.
    fn step(&mut self, params: &mut [&mut Tensor]);

    /// Zero all parameter gradients (clears `.grad()` for each).
    fn zero_grad(&self, params: &mut [&mut Tensor]) {
        for p in params.iter() {
            p.zero_grad();
        }
    }

    /// Get the current learning rate.
    fn get_lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}
