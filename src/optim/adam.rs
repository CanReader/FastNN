//! Adam and AdamW.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::state::place;
use super::{Optimizer, OptimizerState};

/// The moment estimates and update rule both optimizers share.
struct Moments {
    first: Vec<Option<Tensor>>,
    second: Vec<Option<Tensor>>,
    beta1: f32,
    beta2: f32,
    eps: f32,
    steps: u64,
}

impl Moments {
    fn new(count: usize) -> Moments {
        Moments {
            first: vec![None; count],
            second: vec![None; count],
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            steps: 0,
        }
    }

    /// How far to move parameter `index` given its gradient.
    ///
    /// Both moments start at zero and are therefore biased toward it early on;
    /// dividing by `1 - βᵗ` corrects for that, which is what makes the first
    /// few steps usable rather than vanishingly small.
    fn update(&mut self, index: usize, grad: &Tensor) -> Tensor {
        let first = blend(self.first[index].as_ref(), grad, self.beta1);
        let second = blend(self.second[index].as_ref(), &grad.square(), self.beta2);

        let correction1 = 1.0 - self.beta1.powi(self.steps as i32);
        let correction2 = 1.0 - self.beta2.powi(self.steps as i32);
        let mean = first.div_scalar(correction1);
        let scale = second.div_scalar(correction2).sqrt().add_scalar(self.eps);

        self.first[index] = Some(first);
        self.second[index] = Some(second);
        mean.div(&scale)
    }

    fn save(&self) -> OptimizerState {
        let mut state = OptimizerState::new(self.steps);
        state.put("moment1", &self.first);
        state.put("moment2", &self.second);
        state
    }

    fn restore(&mut self, state: OptimizerState, params: &[Param]) -> Result<()> {
        self.first = place(state.take("moment1", params.len())?, params);
        self.second = place(state.take("moment2", params.len())?, params);
        self.steps = state.steps;
        Ok(())
    }
}

/// `β·previous + (1-β)·value`, starting from zero on the first step.
///
/// The exponential moving average every adaptive optimizer is built on.
pub(crate) fn blend(previous: Option<&Tensor>, value: &Tensor, beta: f32) -> Tensor {
    let fresh = value.mul_scalar(1.0 - beta);
    match previous {
        Some(previous) => previous.mul_scalar(beta).add(&fresh),
        None => fresh,
    }
}

/// Adam: per-parameter step sizes from running gradient moments.
///
/// ```
/// # use fastnn::prelude::*;
/// # let model = Sequential::new().add(Linear::new(4, 2));
/// let mut opt = Adam::new(model.parameters(), 1e-3).betas(0.9, 0.95);
/// ```
pub struct Adam {
    params: Vec<Param>,
    moments: Moments,
    lr: f32,
    weight_decay: f32,
}

impl Adam {
    pub fn new(params: Vec<Param>, lr: f32) -> Adam {
        Adam {
            moments: Moments::new(params.len()),
            params,
            lr,
            weight_decay: 0.0,
        }
    }

    /// Decay rates for the first and second moment. Lower `beta2` reacts faster
    /// to changing gradient scale, which transformers often want.
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Adam {
        self.moments.beta1 = beta1;
        self.moments.beta2 = beta2;
        self
    }

    /// Floor on the denominator, guarding against division by a tiny variance.
    pub fn eps(mut self, eps: f32) -> Adam {
        self.moments.eps = eps;
        self
    }

    /// L2 penalty folded into the gradient. For true weight decay use [`AdamW`].
    pub fn weight_decay(mut self, decay: f32) -> Adam {
        self.weight_decay = decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.moments.steps += 1;
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

            let update = self.moments.update(index, &grad);
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
        self.moments.save()
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.moments.restore(state, &self.params)
    }
}

/// Adam with decoupled weight decay.
///
/// Adam's L2 penalty goes through the moment estimates, so a parameter with
/// consistently small gradients gets decayed *harder* than one with large ones —
/// the opposite of the intent. AdamW shrinks the weight directly instead, which
/// is why it is the default for transformers.
pub struct AdamW {
    params: Vec<Param>,
    moments: Moments,
    lr: f32,
    weight_decay: f32,
}

impl AdamW {
    pub fn new(params: Vec<Param>, lr: f32) -> AdamW {
        AdamW {
            moments: Moments::new(params.len()),
            params,
            lr,
            weight_decay: 0.01,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> AdamW {
        self.moments.beta1 = beta1;
        self.moments.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: f32) -> AdamW {
        self.moments.eps = eps;
        self
    }

    pub fn weight_decay(mut self, decay: f32) -> AdamW {
        self.weight_decay = decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.moments.steps += 1;
        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };

            // Shrink the weight first, untouched by the moment estimates.
            let value = param.value().mul_scalar(1.0 - self.lr * self.weight_decay);
            let update = self.moments.update(index, &grad);
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
        self.moments.save()
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.moments.restore(state, &self.params)
    }
}
