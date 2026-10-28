//! RMSprop: divide each gradient by a running estimate of its own scale.

use crate::error::Result;
use crate::nn::Param;
use crate::tensor::Tensor;

use super::adam::blend;
use super::state::place;
use super::{Optimizer, OptimizerState};

/// RMSprop, optionally centered and with classical momentum.
///
/// ```
/// # use fastnn::prelude::*;
/// # let model = Sequential::new().add(Linear::new(4, 2));
/// let mut opt = RMSprop::new(model.parameters(), 1e-3).momentum(0.9).centered();
/// ```
pub struct RMSprop {
    params: Vec<Param>,
    square_avg: Vec<Option<Tensor>>,
    grad_avg: Vec<Option<Tensor>>,
    momentum_buf: Vec<Option<Tensor>>,
    lr: f32,
    alpha: f32,
    eps: f32,
    momentum: f32,
    centered: bool,
    steps: u64,
}

impl RMSprop {
    pub fn new(params: Vec<Param>, lr: f32) -> RMSprop {
        let n = params.len();
        RMSprop {
            params,
            square_avg: vec![None; n],
            grad_avg: vec![None; n],
            momentum_buf: vec![None; n],
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            centered: false,
            steps: 0,
        }
    }

    /// Decay rate of the squared-gradient average.
    pub fn alpha(mut self, alpha: f32) -> RMSprop {
        assert!(
            (0.0..1.0).contains(&alpha),
            "alpha must be in [0, 1), got {alpha}"
        );
        self.alpha = alpha;
        self
    }

    pub fn eps(mut self, eps: f32) -> RMSprop {
        self.eps = eps;
        self
    }

    /// Momentum applied to the scaled update.
    pub fn momentum(mut self, momentum: f32) -> RMSprop {
        self.momentum = momentum;
        self
    }

    /// Normalise by the gradient's variance instead of its raw second moment.
    pub fn centered(mut self) -> RMSprop {
        self.centered = true;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        self.steps += 1;
        for (index, param) in self.params.iter().enumerate() {
            if !param.is_trainable() {
                continue;
            }
            let Some(grad) = param.grad() else { continue };

            let square = blend(self.square_avg[index].as_ref(), &grad.square(), self.alpha);

            // The variance can dip below zero in floating point, so eps goes inside the root.
            let denominator = if self.centered {
                let mean = blend(self.grad_avg[index].as_ref(), &grad, self.alpha);
                let variance = square.sub(&mean.square());
                self.grad_avg[index] = Some(mean);
                variance.add_scalar(self.eps).sqrt()
            } else {
                square.sqrt().add_scalar(self.eps)
            };
            self.square_avg[index] = Some(square);

            let scaled = grad.div(&denominator);
            let update = if self.momentum > 0.0 {
                let buf = match &self.momentum_buf[index] {
                    Some(buf) => buf.mul_scalar(self.momentum).add(&scaled),
                    None => scaled,
                };
                self.momentum_buf[index] = Some(buf.clone());
                buf
            } else {
                scaled
            };

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
        state.put("grad_avg", &self.grad_avg);
        state.put("momentum", &self.momentum_buf);
        state
    }

    fn load_state(&mut self, state: OptimizerState) -> Result<()> {
        self.square_avg = place(state.take("square_avg", self.params.len())?, &self.params);
        self.grad_avg = place(state.take("grad_avg", self.params.len())?, &self.params);
        self.momentum_buf = place(state.take("momentum", self.params.len())?, &self.params);
        self.steps = state.steps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_matches_the_formula_exactly() {
        let param = Param::new(Tensor::from_vec(vec![1.0], &[1]));
        param.set_grad(Tensor::from_vec(vec![3.0], &[1]));

        let mut opt = RMSprop::new(vec![param.clone()], 0.1).alpha(0.5).eps(0.0);
        opt.step();

        let expected = 1.0 - 0.1 * 3.0 / (0.5f32 * 9.0).sqrt();
        let got = param.value().to_vec()[0];
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }

    /// With a constant gradient the variance estimate shrinks, so late steps
    /// dwarf early ones. The plain form settles at |g| instead.
    #[test]
    fn centered_variant_accelerates_on_constant_gradients() {
        let param = Param::new(Tensor::from_vec(vec![0.0], &[1]));
        let mut opt = RMSprop::new(vec![param.clone()], 0.01)
            .alpha(0.9)
            .centered();

        let mut previous = 0.0f32;
        let mut first_step_size = 0.0f32;
        let mut step_size = 0.0f32;
        for iteration in 0..100 {
            param.set_grad(Tensor::from_vec(vec![1.0], &[1]));
            opt.step();
            let value = param.value().to_vec()[0];
            step_size = (previous - value).abs();
            if iteration == 0 {
                first_step_size = step_size;
            }
            previous = value;
        }
        assert!(
            step_size > 5.0 * first_step_size,
            "late steps ({step_size}) should dwarf the first ({first_step_size})"
        );
    }
}
