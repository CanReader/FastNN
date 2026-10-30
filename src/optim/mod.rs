//! Optimizers and learning-rate schedules.
//!
//! An optimizer owns shared [`Param`] handles, so `step()` needs no model borrow.
//! A [`LrSchedule`] maps the step to a rate, and [`Optimizer::set_lr`] applies it.
//!
//! ```
//! use fastnn::prelude::*;
//! # let model = Sequential::new().add(Linear::new(4, 2));
//! # let x = Tensor::randn(&[3, 4]);
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//! let schedule = Warmup::new(CosineAnnealing::new(1e-3, 10_000), 500);
//!
//! for step in 0..3 {
//!     opt.set_lr(schedule.lr_at(step));
//!     let loss = cross_entropy(&model.forward(&x), &[0, 1, 0]);
//!
//!     opt.zero_grad();
//!     loss.backward();
//!     clip_grad_norm(opt.parameters(), 1.0);
//!     opt.step();
//! }
//! ```
//!
//! - Optimizers: [`SGD`], [`Adam`], [`AdamW`], [`RMSprop`], [`Adagrad`],
//!   [`Adadelta`], [`RAdam`] and [`Lion`].
//! - Wrappers: [`Lookahead`] around any optimizer, and [`Ema`] for averaged
//!   weights to evaluate and ship.
//! - Schedules: [`Constant`], [`StepDecay`], [`CosineAnnealing`], [`OneCycle`],
//!   and [`Warmup`] around any of them.
//! - Clipping: [`clip_grad_norm`] and [`clip_grad_value`].
//! - [`OptimizerState`] is what [`save_training`](crate::serialize::save_training)
//!   stores, so a resumed run keeps its moments and step count.

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

const NORM_EPS: f32 = 1e-6;

/// What every optimizer can do.
pub trait Optimizer {
    /// Apply one update from the gradients currently in each parameter.
    /// Parameters with no gradient are skipped.
    fn step(&mut self);

    /// The parameters this optimizer updates.
    fn parameters(&self) -> &[Param];

    /// The current learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate. This is how a [`LrSchedule`] is applied.
    fn set_lr(&mut self, lr: f32);

    /// Momentum, moment estimates, and step count, for
    /// [`save_training`](crate::serialize::save_training).
    fn state(&self) -> OptimizerState;

    /// Restore what [`state`](Optimizer::state) returned.
    fn load_state(&mut self, state: OptimizerState) -> Result<()>;

    /// Clear every gradient. Skip it to accumulate across micro-batches.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Whether every parameter's gradient is finite.
    ///
    /// Guard for `step()`: one inf gradient becomes an inf weight and then NaN
    /// everywhere. Reads every gradient, so on a GPU it downloads them.
    fn gradients_are_finite(&self) -> bool {
        self.parameters()
            .iter()
            .filter_map(|p| p.grad())
            .all(|g| g.is_finite())
    }
}

/// Scale gradients down so their combined L2 norm is at most `max_norm`.
///
/// Returns the norm before clipping, which is worth logging.
pub fn clip_grad_norm(params: &[Param], max_norm: f32) -> f32 {
    let total: f32 = params
        .iter()
        .filter_map(|p| p.grad())
        .map(|g| g.square().sum().item())
        .sum::<f32>()
        .sqrt();

    if total > max_norm {
        let scale = max_norm / (total + NORM_EPS);
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
/// Unlike [`clip_grad_norm`] this changes the gradient's direction.
pub fn clip_grad_value(params: &[Param], max_value: f32) {
    for param in params {
        if let Some(grad) = param.grad() {
            param.set_grad(grad.clamp(-max_value, max_value));
        }
    }
}
