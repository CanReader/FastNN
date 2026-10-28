//! Stochastic gradient descent.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::state::place;
use super::{Optimizer, OptimizerState};

/// SGD with optional momentum, Nesterov, and weight decay.
///
/// ```
/// # use fastnn::prelude::*;
/// # let model = Sequential::new().add(Linear::new(4, 2));
/// let mut opt = SGD::new(model.parameters(), 0.1)
///     .momentum(0.9)
///     .weight_decay(5e-4);
/// ```
pub struct SGD {
    params: Vec<Param>,
    velocity: Vec<Option<Tensor>>,
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    steps: u64,
}

impl SGD {
    pub fn new(params: Vec<Param>, lr: f32) -> SGD {
        SGD {
            velocity: vec![None; params.len()],
            params,
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            steps: 0,
        }
    }

    /// Fraction of the previous update carried forward.
    pub fn momentum(mut self, momentum: f32) -> SGD {
        self.momentum = momentum;
        self
    }

    /// Damping on each new gradient entering the velocity.
    pub fn dampening(mut self, dampening: f32) -> SGD {
        self.dampening = dampening;
        self
    }

    /// L2 penalty added to the gradient.
    pub fn weight_decay(mut self, decay: f32) -> SGD {
        self.weight_decay = decay;
        self
    }

    /// Use Nesterov momentum.
    pub fn nesterov(mut self, nesterov: bool) -> SGD {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        self.steps += 1;
        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };
            let value = param.value();

            let grad = if self.weight_decay == 0.0 {
                grad
            } else {
                grad.add(&value.mul_scalar(self.weight_decay))
            };

            let update = if self.momentum == 0.0 {
                grad
            } else {
                let velocity = match &self.velocity[index] {
                    Some(previous) => previous
                        .mul_scalar(self.momentum)
                        .add(&grad.mul_scalar(1.0 - self.dampening)),
                    None => grad.clone(),
                };
                let update = if self.nesterov {
                    grad.add(&velocity.mul_scalar(self.momentum))
                } else {
                    velocity.clone()
                };
                self.velocity[index] = Some(velocity);
                update
            };

            param.set_value(value.sub(&update.mul_scalar(self.lr)));
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
        state.put("velocity", &self.velocity);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.velocity = place(state.take("velocity", self.params.len())?, &self.params);
        self.steps = state.steps;
        Ok(())
    }
}
