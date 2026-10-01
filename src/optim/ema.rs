//! An exponential moving average of the weights themselves.
//!
//! SGD's iterates orbit the minimum rather than landing on it; their running
//! average sits closer to the centre of that orbit (Polyak averaging). Kept as
//! a shadow copy with decay d, updated after each optimizer step, the average
//! is what you *evaluate and ship* while the raw weights keep training —
//! standard practice for diffusion models and GANs, cheap insurance elsewhere.

use crate::nn::Param;
use crate::tensor::Tensor;

/// Shadow weights: `shadow ← d·shadow + (1−d)·param` after every step.
///
/// ```
/// # use fastnn::prelude::*;
/// # use fastnn::optim::Ema;
/// # let model = Sequential::new().add(Linear::new(4, 2));
/// # let x = Tensor::randn(&[1, 4]);
/// let mut opt = SGD::new(model.parameters(), 0.1);
/// let mut ema = Ema::new(model.parameters(), 0.999);
///
/// // ... loss.backward(); opt.step(); ...
/// ema.update();
///
/// ema.swap();     // evaluate with the averaged weights
/// let logits = no_grad(|| model.forward(&x));
/// ema.swap();     // back to the raw weights; training continues
/// ```
pub struct Ema {
    params: Vec<Param>,
    shadow: Vec<Tensor>,
    decay: f32,
}

impl Ema {
    /// Typical decays are 0.99–0.9999; higher = smoother but slower to follow.
    pub fn new(params: Vec<Param>, decay: f32) -> Ema {
        assert!(
            (0.0..1.0).contains(&decay),
            "decay must be in [0, 1), got {decay}"
        );
        // Seeding the shadow at the current weights (not zero) avoids a long
        // bias toward the origin that 1/(1−d) steps would otherwise carry.
        let shadow = params.iter().map(|p| p.value().detach()).collect();
        Ema {
            params,
            shadow,
            decay,
        }
    }

    /// Fold the current weights into the average. Call after `opt.step()`.
    pub fn update(&mut self) {
        for (shadow, param) in self.shadow.iter_mut().zip(&self.params) {
            let value = param.value();
            *shadow = shadow
                .mul_scalar(self.decay)
                .add(&value.mul_scalar(1.0 - self.decay));
        }
    }

    /// Exchange model weights and averaged weights.
    ///
    /// Symmetric on purpose: one call to evaluate the average, a second to put
    /// the raw training weights back. Being a swap rather than a copy, nothing
    /// is lost if the calls are unbalanced — the other set is always held here.
    pub fn swap(&mut self) {
        for (shadow, param) in self.shadow.iter_mut().zip(&self.params) {
            let raw = param.value().detach();
            param.set_value(shadow.clone());
            *shadow = raw;
        }
    }

    /// The averaged tensors, in parameter order — for saving a shipped model.
    pub fn averaged(&self) -> &[Tensor] {
        &self.shadow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed: shadow seeds at 1, weight jumps to 3, one update with
    /// d = 0.9 gives 0.9·1 + 0.1·3 = 1.2.
    #[test]
    fn update_follows_the_definition_exactly() {
        let param = Param::new(Tensor::from_vec(vec![1.0], &[1]));
        let mut ema = Ema::new(vec![param.clone()], 0.9);

        param.set_value(Tensor::from_vec(vec![3.0], &[1]));
        ema.update();

        assert!((ema.averaged()[0].to_vec()[0] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn double_swap_is_identity() {
        let param = Param::new(Tensor::from_vec(vec![5.0], &[1]));
        let mut ema = Ema::new(vec![param.clone()], 0.99);
        param.set_value(Tensor::from_vec(vec![7.0], &[1]));

        ema.swap();
        assert_eq!(
            param.value().to_vec()[0],
            5.0,
            "first swap shows the average"
        );
        ema.swap();
        assert_eq!(
            param.value().to_vec()[0],
            7.0,
            "second swap restores training weights"
        );
    }
}
