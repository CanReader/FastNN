pub mod sgd;
pub mod adam;
pub mod scheduler;

pub use sgd::SGD;
pub use adam::{Adam, AdamW};
pub use scheduler::{LRScheduler, StepLR, CosineAnnealingLR, LinearWarmup, OneCycleLR};

use crate::tensor::Tensor;

/// Clips the global L2 norm of all parameter gradients to `max_norm`.
/// Returns the pre-clipping norm. Call this between `backward()` and `step()`.
pub fn clip_grad_norm(params: &[&mut Tensor], max_norm: f32) -> f32 {
    let total_norm: f32 = params.iter()
        .filter_map(|p| p.grad())
        .flat_map(|g| g.to_vec())
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in params.iter() {
            if let Some(g) = p.grad() {
                let shape = g.shape().to_vec();
                let scaled: Vec<f32> = g.to_vec().iter().map(|&x| x * scale).collect();
                p.set_grad(Tensor::from_vec(scaled, &shape));
            }
        }
    }
    total_norm
}

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
