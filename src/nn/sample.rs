//! Drawing a token from a model's output logits.
//!
//! Every generation loop ends the same way: one position's logits become one
//! token id. [`Sampler`] owns that step — temperature, top-k, and nucleus
//! (top-p) filtering — and draws through [`crate::rng`], so a run under
//! [`manual_seed`](crate::rng::manual_seed) generates reproducibly.

use crate::rng;
use crate::tensor::Tensor;

/// Turns one position's logits into a token id.
///
/// ```
/// use fastnn::nn::Sampler;
/// use fastnn::tensor::Tensor;
///
/// let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5], &[3]);
/// let sampler = Sampler::new().temperature(0.8).top_k(40).top_p(0.95);
/// let token = sampler.sample(&logits);
/// assert!(token < 3);
/// ```
pub struct Sampler {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: f32,
}

impl Sampler {
    /// Plain sampling from the softmax of the logits.
    pub fn new() -> Sampler {
        Sampler {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }

    /// Always the most likely token. Equivalent to `temperature(0.0)`.
    pub fn greedy() -> Sampler {
        Sampler::new().temperature(0.0)
    }

    /// Below 1.0 sharpens the distribution, above 1.0 flattens it; 0.0 is greedy.
    pub fn temperature(mut self, temperature: f32) -> Sampler {
        assert!(
            temperature >= 0.0,
            "temperature must be >= 0, got {temperature}"
        );
        self.temperature = temperature;
        self
    }

    /// Keep only the `k` most likely tokens before sampling.
    pub fn top_k(mut self, k: usize) -> Sampler {
        assert!(k > 0, "top_k must be at least 1");
        self.top_k = Some(k);
        self
    }

    /// Keep the smallest set of tokens whose probabilities sum to `p`.
    pub fn top_p(mut self, p: f32) -> Sampler {
        assert!(p > 0.0 && p <= 1.0, "top_p must be in (0, 1], got {p}");
        self.top_p = Some(p);
        self
    }

    /// Discourage tokens that already appeared. 1.0 is off; 1.1–1.3 is the
    /// usual range. Applied through [`sample_with_history`](Self::sample_with_history).
    pub fn repetition_penalty(mut self, penalty: f32) -> Sampler {
        assert!(
            penalty >= 1.0,
            "repetition_penalty must be >= 1, got {penalty}"
        );
        self.repetition_penalty = penalty;
        self
    }

    /// Draw a token id from `logits`, the unnormalized scores for one position.
    ///
    /// Any shape holding exactly the vocabulary — `[vocab]` or `[1, vocab]` —
    /// is fine; the elements are read in order.
    pub fn sample(&self, logits: &Tensor) -> usize {
        self.sample_with_history(logits, &[])
    }

    /// [`sample`](Self::sample), penalising tokens that appear in `history`.
    ///
    /// A positive logit is divided by the penalty and a negative one multiplied,
    /// so "already used" always means "less likely", whichever side of zero the
    /// score sits on.
    pub fn sample_with_history(&self, logits: &Tensor, history: &[usize]) -> usize {
        let mut logits = logits.to_vec();
        assert!(!logits.is_empty(), "cannot sample from empty logits");

        if self.repetition_penalty > 1.0 {
            for &id in history {
                if let Some(logit) = logits.get_mut(id) {
                    *logit = if *logit > 0.0 {
                        *logit / self.repetition_penalty
                    } else {
                        *logit * self.repetition_penalty
                    };
                }
            }
        }

        // Most likely first. Sorting the whole vocabulary is fine at the sizes
        // sampling sees, and gives top-k and top-p the same prefix structure.
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        if self.temperature == 0.0 {
            return order[0];
        }
        if let Some(k) = self.top_k {
            order.truncate(k);
        }

        // Softmax over the survivors, subtracting the max so exp cannot overflow.
        let max = logits[order[0]] / self.temperature;
        let mut probs: Vec<f32> = order
            .iter()
            .map(|&i| (logits[i] / self.temperature - max).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);

        if let Some(p) = self.top_p {
            let mut kept = 0.0;
            let cut = probs
                .iter()
                .position(|&prob| {
                    kept += prob;
                    kept >= p
                })
                .map_or(probs.len(), |i| i + 1);
            order.truncate(cut);
            probs.truncate(cut);
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|q| *q /= sum);
        }

        // Inverse-CDF draw; the fallback covers rounding leaving a sliver above 1.
        let mut draw = rng::uniform(1, 0.0, 1.0)[0];
        for (&index, &prob) in order.iter().zip(&probs) {
            draw -= prob;
            if draw <= 0.0 {
                return index;
            }
        }
        *order.last().unwrap()
    }
}

impl Default for Sampler {
    fn default() -> Sampler {
        Sampler::new()
    }
}
