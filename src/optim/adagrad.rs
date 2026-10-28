//! Adagrad and Adadelta.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::adam::blend;
use super::state::place;
use super::{Optimizer, OptimizerState};

/// Adagrad. Steps shrink as squared gradients accumulate.
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

/// Adadelta: Adagrad with an exponential window. `lr` defaults to 1.0.
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

            // eps inside the roots also bootstraps step one, where E[dx^2] is zero.
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

    #[test]
    fn adadelta_makes_progress_without_a_learning_rate() {
        let param = Param::new(Tensor::from_vec(vec![5.0], &[1]));
        let mut opt = Adadelta::new(vec![param.clone()]);

        for _ in 0..200 {
            // gradient of x^2 / 2
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
