//! Adagrad and Adadelta: adaptive steps from accumulated gradient history.
//!
//! Adagrad divides by the square root of the *total* squared gradient ever
//! seen — coordinates that update often slow down, rare ones stay fast, which
//! is exactly right for sparse features and exactly wrong for long runs, where
//! the ever-growing accumulator drives every step toward zero.
//!
//! Adadelta is the repair: replace the total with an exponential average, and
//! replace the global learning rate with a running average of the updates
//! themselves, so the step size carries the *units* of the parameter — the
//! ratio `√E[Δx²]/√E[g²]` is dimensionally an x per g, where Adagrad's
//! `1/√E[g²]` is not.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::adam::blend;
use super::state::place;
use super::{Optimizer, OptimizerState};

/// Adagrad: per-coordinate steps that shrink with accumulated use.
pub struct Adagrad {
    params: Vec<Param>,
    accumulator: Vec<Option<Tensor>>,
    lr: f32,
    eps: f32,
    steps: u64,
}

impl Adagrad {
    pub fn new(params: Vec<Param>, lr: f32) -> Adagrad {
        let n = params.len();
        Adagrad {
            params,
            accumulator: vec![None; n],
            lr,
            eps: 1e-10,
            steps: 0,
        }
    }

    pub fn eps(mut self, eps: f32) -> Adagrad {
        self.eps = eps;
        self
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self) {
        self.steps += 1;
        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };

            // The accumulator is a *sum*, not an average — that monotone growth
            // is Adagrad's defining property and its known flaw.
            let accumulated = match &self.accumulator[index] {
                Some(prior) => prior.add(&grad.square()),
                None => grad.square(),
            };
            let update = grad.div(&accumulated.sqrt().add_scalar(self.eps));
            self.accumulator[index] = Some(accumulated);

            param.set_value(param.value().sub(&update.mul_scalar(self.lr)));
        }
    }

    fn parameters(&self) -> &[Param] {
        &self.params
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn state(&self) -> OptimizerState {
        let mut state = OptimizerState::new(self.steps);
        state.put("accumulator", &self.accumulator);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.accumulator = place(state.take("accumulator", self.params.len())?, &self.params);
        self.steps = state.steps;
        Ok(())
    }
}

/// Adadelta: Adagrad with a window, and no learning rate to tune.
///
/// `lr` survives as a final multiplier for the rare case it helps; the
/// canonical method uses 1.0, and that is the default.
pub struct Adadelta {
    params: Vec<Param>,
    square_avg: Vec<Option<Tensor>>,
    delta_avg: Vec<Option<Tensor>>,
    lr: f32,
    rho: f32,
    eps: f32,
    steps: u64,
}

impl Adadelta {
    pub fn new(params: Vec<Param>) -> Adadelta {
        let n = params.len();
        Adadelta {
            params,
            square_avg: vec![None; n],
            delta_avg: vec![None; n],
            lr: 1.0,
            rho: 0.9,
            eps: 1e-6,
            steps: 0,
        }
    }

    /// Decay of both running averages.
    pub fn rho(mut self, rho: f32) -> Adadelta {
        assert!(
            (0.0..1.0).contains(&rho),
            "rho must be in [0, 1), got {rho}"
        );
        self.rho = rho;
        self
    }

    pub fn eps(mut self, eps: f32) -> Adadelta {
        self.eps = eps;
        self
    }
}

impl Optimizer for Adadelta {
    fn step(&mut self) {
        self.steps += 1;
        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };

            let square = blend(self.square_avg[index].as_ref(), &grad.square(), self.rho);

            // The eps *inside* both roots doubles as the bootstrap: on the very
            // first step E[Δx²] is zero and the ratio would otherwise be too.
            let previous_delta = self.delta_avg[index].as_ref();
            let numerator = match previous_delta {
                Some(delta) => delta.add_scalar(self.eps).sqrt(),
                None => Tensor::zeros_like(&grad).add_scalar(self.eps.sqrt()),
            };
            let update = grad
                .mul(&numerator)
                .div(&square.add_scalar(self.eps).sqrt());

            self.delta_avg[index] = Some(blend(previous_delta, &update.square(), self.rho));
            self.square_avg[index] = Some(square);

            param.set_value(param.value().sub(&update.mul_scalar(self.lr)));
        }
    }

    fn parameters(&self) -> &[Param] {
        &self.params
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn state(&self) -> OptimizerState {
        let mut state = OptimizerState::new(self.steps);
        state.put("square_avg", &self.square_avg);
        state.put("delta_avg", &self.delta_avg);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.square_avg = place(state.take("square_avg", self.params.len())?, &self.params);
        self.delta_avg = place(state.take("delta_avg", self.params.len())?, &self.params);
        self.steps = state.steps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two hand-computed Adagrad steps with g = 2 each time and eps = 0:
    /// step 1 divides by √4 = 2, step 2 by √8, so the second step is smaller
    /// by exactly √2.
    #[test]
    fn adagrad_steps_shrink_by_the_accumulated_root() {
        let param = Param::new(Tensor::from_vec(vec![0.0], &[1]));
        let mut opt = Adagrad::new(vec![param.clone()], 1.0).eps(0.0);

        param.set_grad(Tensor::from_vec(vec![2.0], &[1]));
        opt.step();
        let first = -param.value().to_vec()[0];
        assert!(
            (first - 1.0).abs() < 1e-6,
            "first step should be lr·g/√g² = 1, got {first}"
        );

        param.set_grad(Tensor::from_vec(vec![2.0], &[1]));
        opt.step();
        let second = -param.value().to_vec()[0] - first;
        assert!(
            (second - 1.0 / 2.0f32.sqrt()).abs() < 1e-6,
            "second step should be 1/√2, got {second}"
        );
    }

    /// Adadelta's step size is self-tuning: with lr = 1 and a constant
    /// gradient it settles near √eps-scale moves that keep making progress.
    #[test]
    fn adadelta_makes_progress_without_a_learning_rate() {
        let param = Param::new(Tensor::from_vec(vec![5.0], &[1]));
        let mut opt = Adadelta::new(vec![param.clone()]);

        for _ in 0..200 {
            // d/dx (x²/2) = x: minimize a bowl centred at zero.
            param.set_grad(param.value());
            opt.step();
        }
        let final_value = param.value().to_vec()[0].abs();
        assert!(
            final_value < 5.0 * 0.9,
            "no progress made: still at {final_value}"
        );
    }
}
