//! Lion (Chen et al., 2023): updates by the sign of an interpolated momentum.
//!
//! Every coordinate moves by exactly lr, so lr and weight decay need retuning
//! compared to AdamW.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::adam::blend;
use super::state::place;
use super::{Optimizer, OptimizerState};

const SIGN_EPS: f32 = 1e-30;

/// Lion.
pub struct Lion {
    params: Vec<Param>,
    momentum: Vec<Option<Tensor>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    steps: u64,
}

impl Lion {
    pub fn new(params: Vec<Param>, lr: f32) -> Lion {
        let n = params.len();
        Lion {
            params,
            momentum: vec![None; n],
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
            steps: 0,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Lion {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Decoupled weight decay.
    pub fn weight_decay(mut self, decay: f32) -> Lion {
        self.weight_decay = decay;
        self
    }
}

/// `x/(|x|+eps)`, built from element-wise ops so it stays on the parameters' device.
fn sign(x: &Tensor) -> Tensor {
    x.div(&x.abs().add_scalar(SIGN_EPS))
}

impl Optimizer for Lion {
    fn step(&mut self) {
        self.steps += 1;
        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };
            let value = param.value();

            let previous = self.momentum[index].as_ref();
            let direction = sign(&blend(previous, &grad, self.beta1));
            // The step uses beta1 and the stored momentum uses beta2.
            self.momentum[index] = Some(blend(previous, &grad, self.beta2));

            let decayed = if self.weight_decay == 0.0 {
                value
            } else {
                value.mul_scalar(1.0 - self.lr * self.weight_decay)
            };
            param.set_value(decayed.sub(&direction.mul_scalar(self.lr)));
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
        state.put("momentum", &self.momentum);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.momentum = place(state.take("momentum", self.params.len())?, &self.params);
        self.steps = state.steps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_uniform_regardless_of_gradient_scale() {
        let param = Param::new(Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3]));
        param.set_grad(Tensor::from_vec(vec![1e-6, 42.0, -3.0], &[3]));

        let mut opt = Lion::new(vec![param.clone()], 0.01);
        opt.step();

        let values = param.value().to_vec();
        assert!(
            (values[0] - -0.01).abs() < 1e-6,
            "tiny gradient still steps full size"
        );
        assert!(
            (values[1] - -0.01).abs() < 1e-6,
            "huge gradient steps the same size"
        );
        assert!(
            (values[2] - 0.01).abs() < 1e-6,
            "negative gradient steps the other way"
        );
    }

    #[test]
    fn zero_gradient_takes_no_step() {
        let param = Param::new(Tensor::from_vec(vec![1.0], &[1]));
        param.set_grad(Tensor::zeros(&[1]));

        let mut opt = Lion::new(vec![param.clone()], 0.01);
        opt.step();
        assert_eq!(param.value().to_vec()[0], 1.0);
    }
}
