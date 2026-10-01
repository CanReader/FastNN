//! RAdam: Adam with its warmup requirement solved in closed form.
//!
//! Adam's variance estimate `v` is built from very few samples early in
//! training, so `1/√v̂` is itself high-variance and the first updates can be
//! wildly mis-scaled — the practical reason Adam is run behind a warmup
//! schedule. Liu et al. (2020) computed that variance analytically: tracking
//! the effective sample size ρ_t of the exponential average, the adaptive step
//! can be *rectified* by
//!
//! ```text
//! r_t = √( (ρ_t−4)(ρ_t−2)ρ∞ / ((ρ∞−4)(ρ∞−2)ρ_t) ),   ρ∞ = 2/(1−β₂) − 1
//! ```
//!
//! which is exactly the factor that makes the adaptive step's variance match
//! its late-training value. While ρ_t ≤ 4 the variance is undefined — too few
//! samples — and RAdam falls back to plain momentum SGD for those steps.

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

    /// The rectification factor for step `t`, or `None` while the variance of
    /// the adaptive step is still undefined (ρ_t ≤ 4).
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
                // Too few samples to trust the variance: momentum SGD instead.
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

    /// With β₂ = 0.999: ρ∞ = 1999, and ρ_t ≤ 4 for the first four steps — the
    /// published fallback window. Step 5 must rectify, and with r < 1: the
    /// rectified step is *smaller* than plain Adam's would be.
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
        // Late in training the correction must vanish: r → 1.
        let late = opt.rectifier(100_000).unwrap();
        assert!(
            (late - 1.0).abs() < 0.01,
            "late rectifier should approach 1, got {late}"
        );
    }

    /// The first step is pure bias-corrected momentum: with β₁ = 0.9 and g = 2,
    /// m̂₁ = g exactly, so the parameter moves by lr·g.
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
