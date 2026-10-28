//! Loss functions, as free functions since they hold no state.

use crate::autograd::ops::loss::{CrossEntropyBackward, WeightedCrossEntropyBackward};
use crate::tensor::Tensor;

/// Keeps probabilities off 0 and 1 so `log` stays finite.
const CLAMP_EPS: f32 = 1e-7;

/// Mean cross-entropy of raw `logits` `[batch, classes]` against integer class targets.
pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Tensor {
    assert_eq!(
        logits.ndim(),
        2,
        "cross_entropy expects [batch, classes], got {:?}",
        logits.shape()
    );
    let (batch, classes) = (logits.dim(0), logits.dim(1));
    assert_eq!(
        targets.len(),
        batch,
        "cross_entropy: {} targets for {batch} rows",
        targets.len()
    );

    let data = logits.to_vec();
    let mut softmax = vec![0.0f32; batch * classes];
    let mut total = 0.0f32;

    for (row, &target) in targets.iter().enumerate() {
        assert!(
            target < classes,
            "cross_entropy: target {target} outside 0..{classes}"
        );
        let base = row * classes;
        let values = &data[base..base + classes];

        // shift by the row max so exp can't overflow
        let shift = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = values.iter().map(|&v| (v - shift).exp()).sum();
        for c in 0..classes {
            softmax[base + c] = (values[c] - shift).exp() / sum;
        }
        total -= values[target] - shift - sum.ln();
    }

    let saved = Tensor::from_vec(softmax, &[batch, classes]);
    let targets = targets.to_vec();
    Tensor::scalar(total / batch as f32)
        .to(logits.device())
        .with_grad(&[logits], || CrossEntropyBackward {
            softmax: saved,
            targets,
            classes,
        })
}

/// Mean squared error.
pub fn mse(prediction: &Tensor, target: &Tensor) -> Tensor {
    prediction.sub(target).square().mean()
}

/// Mean absolute error.
pub fn mae(prediction: &Tensor, target: &Tensor) -> Tensor {
    prediction.sub(target).abs().mean()
}

/// Binary cross-entropy over probabilities. Prefer [`bce_with_logits`] for raw scores.
pub fn bce(prediction: &Tensor, target: &Tensor) -> Tensor {
    let p = prediction.clamp(CLAMP_EPS, 1.0 - CLAMP_EPS);
    let positive = target.mul(&p.log());
    let negative = target
        .neg()
        .add_scalar(1.0)
        .mul(&p.neg().add_scalar(1.0).log());
    positive.add(&negative).neg().mean()
}

/// Binary cross-entropy from logits, as `max(x,0) - x·t + log(1 + e^-|x|)` to avoid overflow.
pub fn bce_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    let floor = logits.clamp(0.0, f32::INFINITY);
    let stable_log = logits.abs().neg().exp().add_scalar(1.0).log();
    floor.sub(&logits.mul(target)).add(&stable_log).mean()
}

/// Huber loss: quadratic within `delta` of the target, linear beyond it.
pub fn huber(prediction: &Tensor, target: &Tensor, delta: f32) -> Tensor {
    assert!(delta > 0.0, "huber delta must be positive, got {delta}");
    let residual = prediction.sub(target).abs();
    let clipped = residual.clamp(0.0, delta);
    let quadratic = clipped.square().mul_scalar(0.5);
    let linear = residual.sub(&clipped).mul_scalar(delta);
    quadratic.add(&linear).mean()
}

/// Huber with `delta = 1`.
pub fn smooth_l1(prediction: &Tensor, target: &Tensor) -> Tensor {
    huber(prediction, target, 1.0)
}

/// Gaussian negative log-likelihood from a predicted mean and log-variance,
/// without the `½ln 2π` constant.
pub fn gaussian_nll(mean: &Tensor, log_variance: &Tensor, target: &Tensor) -> Tensor {
    let squared_error = target.sub(mean).square();
    let precision = log_variance.neg().exp();
    log_variance
        .add(&squared_error.mul(&precision))
        .mul_scalar(0.5)
        .mean()
}

/// Poisson negative log-likelihood from a predicted log-rate, without the `log k!` term.
pub fn poisson_nll(log_rate: &Tensor, target: &Tensor) -> Tensor {
    log_rate.exp().sub(&target.mul(log_rate)).mean()
}

/// `KL(target ‖ prediction)` over rows of probabilities, summed on the last axis.
/// Both sides are clamped so `0·log 0` can't become `NaN`.
pub fn kl_divergence(prediction: &Tensor, target: &Tensor) -> Tensor {
    let p = prediction.clamp(CLAMP_EPS, 1.0);
    let t = target.clamp(CLAMP_EPS, 1.0);
    let rows = (p.numel() / p.last_dim()) as f32;
    t.mul(&t.log().sub(&p.log())).sum().div_scalar(rows)
}

/// Negative log-likelihood over rows of log-probabilities. Equals [`cross_entropy`] after `log_softmax`.
pub fn nll(log_probs: &Tensor, targets: &[usize]) -> Tensor {
    let (batch, classes) = (log_probs.dim(0), log_probs.dim(1));
    assert_eq!(
        batch,
        targets.len(),
        "nll: {batch} rows but {} targets",
        targets.len()
    );

    // picking one entry per row as a dot with constant one-hot targets
    let mut one_hot = vec![0.0f32; batch * classes];
    for (row, &target) in targets.iter().enumerate() {
        assert!(
            target < classes,
            "nll: target {target} outside 0..{classes}"
        );
        one_hot[row * classes + target] = 1.0;
    }
    let selector = Tensor::from_vec(one_hot, &[batch, classes]).to(log_probs.device());
    log_probs
        .mul(&selector)
        .sum()
        .div_scalar(batch as f32)
        .neg()
}

/// Binary focal loss from logits: BCE scaled by `(1 − p_correct)^gamma`, with
/// `alpha` weighting the positive class. The paper uses `gamma = 2`, `alpha = 0.25`.
pub fn focal_bce_with_logits(logits: &Tensor, target: &Tensor, gamma: f32, alpha: f32) -> Tensor {
    let p = logits.sigmoid().clamp(CLAMP_EPS, 1.0 - CLAMP_EPS);
    let one_minus_p = p.neg().add_scalar(1.0);
    let one_minus_t = target.neg().add_scalar(1.0);

    let positive = target
        .mul(&one_minus_p.powf(gamma))
        .mul(&p.log())
        .mul_scalar(alpha);
    let negative = one_minus_t
        .mul(&p.powf(gamma))
        .mul(&one_minus_p.log())
        .mul_scalar(1.0 - alpha);
    positive.add(&negative).neg().mean()
}

/// Dice loss over probabilities: `1 − 2|P∩T| / (|P| + |T|)`, smoothed.
pub fn dice(prediction: &Tensor, target: &Tensor) -> Tensor {
    const SMOOTH: f32 = 1.0;
    let intersection = prediction
        .mul(target)
        .sum()
        .mul_scalar(2.0)
        .add_scalar(SMOOTH);
    let total = prediction.sum().add(&target.sum()).add_scalar(SMOOTH);
    intersection.div(&total).neg().add_scalar(1.0)
}

/// How per-sample losses become the reported value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reduction {
    /// Weighted mean.
    Mean,
    /// Plain sum; scales with batch size.
    Sum,
    /// One loss per row.
    None,
}

/// [`cross_entropy`] with label smoothing, class weights and an ignored index.
///
/// ```
/// # use fastnn::prelude::*;
/// let loss_fn = CrossEntropyLoss::new()
///     .label_smoothing(0.1)
///     .ignore_index(0);            // 0 is the padding token
/// # let logits = Tensor::randn(&[3, 5]);
/// let loss = loss_fn.compute(&logits, &[2, 0, 4]);
/// ```
///
/// Under `Mean` the loss is divided by the summed weights of the rows that
/// count, so weighting doesn't rescale the learning rate.
pub struct CrossEntropyLoss {
    label_smoothing: f32,
    class_weights: Option<Vec<f32>>,
    ignore_index: Option<usize>,
    reduction: Reduction,
}

impl CrossEntropyLoss {
    pub fn new() -> CrossEntropyLoss {
        CrossEntropyLoss {
            label_smoothing: 0.0,
            class_weights: None,
            ignore_index: None,
            reduction: Reduction::Mean,
        }
    }

    /// Spread `s` of the probability mass uniformly over the classes.
    pub fn label_smoothing(mut self, s: f32) -> CrossEntropyLoss {
        assert!(
            (0.0..1.0).contains(&s),
            "label smoothing must be in [0, 1), got {s}"
        );
        self.label_smoothing = s;
        self
    }

    /// Per-class weights.
    pub fn class_weights(mut self, weights: Vec<f32>) -> CrossEntropyLoss {
        self.class_weights = Some(weights);
        self
    }

    /// Rows with this target contribute nothing.
    pub fn ignore_index(mut self, index: usize) -> CrossEntropyLoss {
        self.ignore_index = Some(index);
        self
    }

    pub fn reduction(mut self, reduction: Reduction) -> CrossEntropyLoss {
        self.reduction = reduction;
        self
    }

    /// Loss for `[batch, classes]` logits, or `[batch]` under [`Reduction::None`].
    pub fn compute(&self, logits: &Tensor, targets: &[usize]) -> Tensor {
        assert_eq!(
            logits.ndim(),
            2,
            "cross entropy expects [batch, classes], got {:?}",
            logits.shape()
        );
        let (batch, classes) = (logits.dim(0), logits.dim(1));
        assert_eq!(
            batch,
            targets.len(),
            "{batch} rows but {} targets",
            targets.len()
        );
        if let Some(weights) = &self.class_weights {
            assert_eq!(
                weights.len(),
                classes,
                "{} class weights for {classes} classes",
                weights.len()
            );
        }

        let data = logits.to_vec();
        let uniform = self.label_smoothing / classes as f32;
        let on_target = 1.0 - self.label_smoothing + uniform;

        let mut per_row = vec![0.0f32; batch];
        let mut difference = vec![0.0f32; batch * classes];
        let mut weight_total = 0.0f32;

        for (row, &target) in targets.iter().enumerate() {
            assert!(target < classes, "target {target} outside 0..{classes}");
            let weight = if self.ignore_index == Some(target) {
                0.0
            } else {
                self.class_weights.as_ref().map_or(1.0, |w| w[target])
            };
            weight_total += weight;

            let values = &data[row * classes..(row + 1) * classes];
            // shift by the row max so every exponent is <= 0
            let shift = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum = values.iter().map(|&v| (v - shift).exp()).sum::<f32>().ln();

            let mut loss = 0.0;
            for (class, &value) in values.iter().enumerate() {
                let log_prob = value - shift - log_sum;
                let q = if class == target { on_target } else { uniform };
                loss -= q * log_prob;
                // d loss / d logit is p - q, scaled by the row weight
                difference[row * classes + class] = weight * ((log_prob).exp() - q);
            }
            per_row[row] = weight * loss;
        }

        let difference = Tensor::from_vec(difference, &[batch, classes]);
        let (value, normalizer, per_row_reduction) = match self.reduction {
            Reduction::Mean => {
                // all rows ignored: report 0 with zero gradient instead of dividing by 0
                let denominator = if weight_total > 0.0 {
                    weight_total
                } else {
                    1.0
                };
                let total: f32 = per_row.iter().sum();
                (Tensor::scalar(total / denominator), denominator, false)
            }
            Reduction::Sum => (Tensor::scalar(per_row.iter().sum()), 1.0, false),
            Reduction::None => (Tensor::from_vec(per_row, &[batch]), 1.0, true),
        };

        value
            .to(logits.device())
            .with_grad(&[logits], || WeightedCrossEntropyBackward {
                difference,
                normalizer,
                per_row: per_row_reduction,
            })
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> CrossEntropyLoss {
        CrossEntropyLoss::new()
    }
}
