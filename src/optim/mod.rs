//! Optimizers and learning-rate schedules.
//!
//! An optimizer owns [`Param`] handles, which are shared slots rather than
//! copies. That is what lets the training loop read:
//!
//! ```
//! use fastnn::prelude::*;
//! # let model = Sequential::new().add(Linear::new(4, 2));
//! # let x = Tensor::randn(&[3, 4]);
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//!
//! let loss = cross_entropy(&model.forward(&x), &[0, 1, 0]);
//! opt.zero_grad();
//! loss.backward();
//! opt.step();
//! ```
//!
//! with no borrow of the model at `step()` time and no parameter list threaded
//! through every call.
//!
//! Updates run on whichever device the parameters live on: they are written with
//! tensor ops, so a GPU model never round-trips to the host to take a step.

pub mod adagrad;
pub mod adam;
pub mod ema;
pub mod lion;
pub mod lookahead;
pub mod radam;
pub mod rmsprop;
pub mod schedule;
pub mod sgd;
pub mod state;

pub use adagrad::{Adadelta, Adagrad};
pub use adam::{Adam, AdamW};
pub use ema::Ema;
pub use lion::Lion;
pub use lookahead::Lookahead;
pub use radam::RAdam;
pub use rmsprop::RMSprop;
pub use schedule::{Constant, CosineAnnealing, LrSchedule, OneCycle, StepDecay, Warmup};
pub use sgd::SGD;
pub use state::OptimizerState;

use crate::error::Result;
use crate::nn::Param;

/// What every optimizer can do.
pub trait Optimizer {
    /// Apply one update using the gradients currently in each parameter.
    ///
    /// Parameters with no gradient are skipped, so a model with an unused branch
    /// still steps cleanly.
    fn step(&mut self);

    /// The parameters this optimizer updates.
    fn parameters(&self) -> &[Param];

    /// The current learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate — how a [`LrSchedule`] is applied.
    fn set_lr(&mut self, lr: f32);

    /// Momentum, moment estimates, and step count, ready to be written to disk.
    ///
    /// Use [`save_training`](crate::serialize::save_training) rather than calling
    /// this directly unless you are storing the state somewhere of your own.
    fn state(&self) -> OptimizerState;

    /// Restore what [`state`](Optimizer::state) captured.
    fn load_state(&mut self, state: OptimizerState) -> Result<()>;

    /// Clear every gradient. Call before `backward()`, or skip it to accumulate
    /// gradients across several micro-batches.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Whether every parameter's gradient is finite.
    ///
    /// A cheap guard before `step()`: one inf in one gradient becomes an inf
    /// weight, and from there every activation downstream is NaN. Skipping the
    /// step costs one batch; taking it costs the run.
    ///
    /// Reads every gradient, so on a GPU this downloads them. Worth it around a
    /// known-unstable phase; not worth it every step of a healthy run.
    fn gradients_are_finite(&self) -> bool {
        self.parameters()
            .iter()
            .filter_map(|p| p.grad())
            .all(|g| g.is_finite())
    }
}

/// Scale gradients down so their combined L2 norm is at most `max_norm`.
///
/// Returns the norm *before* clipping, which is worth logging: a sudden spike is
/// usually the first sign of divergence. Call between `backward()` and `step()`.
pub fn clip_grad_norm(params: &[Param], max_norm: f32) -> f32 {
    let total: f32 = params
        .iter()
        .filter_map(|p| p.grad())
        .map(|g| g.square().sum().item())
        .sum::<f32>()
        .sqrt();

    if total > max_norm {
        // The epsilon keeps the scale finite if the norm underflows to zero.
        let scale = max_norm / (total + 1e-6);
        for param in params {
            if let Some(grad) = param.grad() {
                param.set_grad(grad.mul_scalar(scale));
            }
        }
    }
    total
}

/// Clamp every gradient element into `[-max_value, max_value]`.
///
/// Cruder than [`clip_grad_norm`] — it changes the gradient's direction, not
/// just its length — but robust to a single exploded element. Call between
/// `backward()` and `step()`.
pub fn clip_grad_value(params: &[Param], max_value: f32) {
    for param in params {
        if let Some(grad) = param.grad() {
            param.set_grad(grad.clamp(-max_value, max_value));
        }
    }
}
