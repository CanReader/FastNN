//! RAdam (Liu et al., 2020): Adam with a rectified step, so it needs no warmup.
//!
//! While the variance estimate has too few samples (rho_t <= 4) it falls back
//! to momentum SGD.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::adam::blend;
use super::state::place;
use super::{Optimizer, OptimizerState};

/// Rectified Adam. A drop-in Adam that needs no warmup schedule.
pub struct RAdam {
    params: Vec<Param>,
    first: Vec<Option<Tensor>>,
    second: Vec<Option<Tensor>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    steps: u64,
}

impl RAdam {
    pub fn new(params: Vec<Param>, lr: f32) -> RAdam {
        let n = params.len();
        RAdam {
            params,
            first: vec![None; n],
            second: vec![None; n],
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            steps: 0,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> RAdam {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: f32) -> RAdam {
        self.eps = eps;
        self
    }

    /// Decoupled (AdamW-style) weight decay.
    pub fn weight_decay(mut self, decay: f32) -> RAdam {
        self.weight_decay = decay;
        self
    }

    /// The rectification factor for step `t`, or `None` while rho_t <= 4.
    fn rectifier(&self, t: u64) -> Option<f32> {
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let beta2_t = self.beta2.powi(t as i32);
        let rho_t = rho_inf - 2.0 * t as f32 * beta2_t / (1.0 - beta2_t);
        if rho_t <= 4.0 {
            return None;
        }
        let numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
        let denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
        Some((numerator / denominator).sqrt())
    }
}

impl Optimizer for RAdam {
    fn step(&mut self) {
        self.steps += 1;
        let t = self.steps;
        let rectifier = self.rectifier(t);

        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };
            let value = param.value();

            let first = blend(self.first[index].as_ref(), &grad, self.beta1);
            let second = blend(self.second[index].as_ref(), &grad.square(), self.beta2);

            let mean = first.div_scalar(1.0 - self.beta1.powi(t as i32));
            let update = match rectifier {
                Some(r) => {
                    let variance = second.div_scalar(1.0 - self.beta2.powi(t as i32));
                    mean.div(&variance.sqrt().add_scalar(self.eps))
                        .mul_scalar(r)
                }
                None => mean,
            };

            self.first[index] = Some(first);
            self.second[index] = Some(second);

            let decayed = if self.weight_decay == 0.0 {
                value
            } else {
                value.mul_scalar(1.0 - self.lr * self.weight_decay)
            };
            param.set_value(decayed.sub(&update.mul_scalar(self.lr)));
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
        state.put("moment1", &self.first);
        state.put("moment2", &self.second);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.first = place(state.take("moment1", self.params.len())?, &self.params);
        self.second = place(state.take("moment2", self.params.len())?, &self.params);
        self.steps = state.steps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With beta2 = 0.999 the first four steps fall back, step 5 rectifies.
    #[test]
    fn rectifier_switches_on_at_step_five_and_damps() {
        let opt = RAdam::new(vec![], 1e-3);
        for t in 1..=4 {
            assert!(
                opt.rectifier(t).is_none(),
                "step {t} should fall back to momentum"
            );
        }
        let r = opt.rectifier(5).expect("step 5 should rectify");
        assert!(r > 0.0 && r < 1.0, "early rectifier should damp, got {r}");
        let late = opt.rectifier(100_000).unwrap();
        assert!(
            (late - 1.0).abs() < 0.01,
            "late rectifier should approach 1, got {late}"
        );
    }

    #[test]
    fn first_step_is_exactly_momentum_sgd() {
        let param = Param::new(Tensor::from_vec(vec![1.0], &[1]));
        param.set_grad(Tensor::from_vec(vec![2.0], &[1]));

        let mut opt = RAdam::new(vec![param.clone()], 0.1);
        opt.step();

        let got = param.value().to_vec()[0];
        assert!(
            (got - 0.8).abs() < 1e-6,
            "expected 1 − 0.1·2 = 0.8, got {got}"
        );
    }
}
