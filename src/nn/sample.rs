//! Drawing a token from logits. Draws through [`crate::rng`], so
//! [`manual_seed`](crate::rng::manual_seed) makes generation reproducible.

use crate::rng;
use crate::tensor::Tensor;

/// Turns one position's logits into a token id: temperature, top-k, top-p.
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
    /// Plain softmax sampling.
    pub fn new() -> Sampler {
        Sampler {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
        }
    }

    /// Always the most likely token.
    pub fn greedy() -> Sampler {
        Sampler::new().temperature(0.0)
    }

    /// Below 1.0 sharpens, above flattens, 0.0 is greedy.
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

    /// Discourage tokens already seen. 1.0 is off. Needs [`sample_with_history`](Self::sample_with_history).
    pub fn repetition_penalty(mut self, penalty: f32) -> Sampler {
        assert!(
            penalty >= 1.0,
            "repetition_penalty must be >= 1, got {penalty}"
        );
        self.repetition_penalty = penalty;
        self
    }

    /// Draw a token id from one position's logits, any shape holding the vocabulary.
    pub fn sample(&self, logits: &Tensor) -> usize {
        self.sample_with_history(logits, &[])
    }

    /// [`sample`](Self::sample), penalising tokens in `history`. Positive logits are
    /// divided and negative ones multiplied, so both get less likely.
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

        // most likely first, so top-k and top-p are both prefixes
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        if self.temperature == 0.0 {
            return order[0];
        }
        if let Some(k) = self.top_k {
            order.truncate(k);
        }

        // subtract the max so exp can't overflow
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

        // the fallback covers rounding leaving the sum just under 1
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
