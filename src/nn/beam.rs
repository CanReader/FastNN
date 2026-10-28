//! Beam search decoding. Width 1 is greedy.

/// Beam search over any model that maps a sequence to next-token logits.
///
/// ```
/// use fastnn::nn::BeamSearch;
///
/// // A toy "model": after 0 prefer 1, otherwise prefer 0.
/// let step = |seq: &[usize]| match seq.last() {
///     Some(0) => vec![0.0, 2.0, 0.0],
///     _ => vec![2.0, 0.0, 0.0],
/// };
/// let out = BeamSearch::new(2).decode(&[0], 3, step);
/// assert_eq!(out, vec![1, 0, 1]);
/// ```
pub struct BeamSearch {
    width: usize,
    length_penalty: f32,
    eos: Option<usize>,
}

/// A candidate sequence and its total log-probability.
struct Beam {
    generated: Vec<usize>,
    log_prob: f32,
    finished: bool,
}

impl Beam {
    /// Length-normalised, since long sequences always have lower raw log-probability.
    fn score(&self, length_penalty: f32) -> f32 {
        self.log_prob / (self.generated.len().max(1) as f32).powf(length_penalty)
    }
}

impl BeamSearch {
    pub fn new(width: usize) -> BeamSearch {
        assert!(width > 0, "beam width must be at least 1");
        BeamSearch {
            width,
            length_penalty: 1.0,
            eos: None,
        }
    }

    /// Exponent on length when comparing beams. 1.0 is the per-token average.
    pub fn length_penalty(mut self, penalty: f32) -> BeamSearch {
        assert!(
            penalty > 0.0,
            "length_penalty must be positive, got {penalty}"
        );
        self.length_penalty = penalty;
        self
    }

    /// A beam that emits this token is finished and stops growing.
    pub fn eos(mut self, token: usize) -> BeamSearch {
        self.eos = Some(token);
        self
    }

    /// Generate up to `max_tokens` ids after `start`, without `start` or the eos token.
    /// `step` maps the full sequence so far to next-token logits.
    pub fn decode(
        &self,
        start: &[usize],
        max_tokens: usize,
        mut step: impl FnMut(&[usize]) -> Vec<f32>,
    ) -> Vec<usize> {
        let mut beams = vec![Beam {
            generated: Vec::new(),
            log_prob: 0.0,
            finished: false,
        }];

        for _ in 0..max_tokens {
            if beams.iter().all(|b| b.finished) {
                break;
            }

            let mut candidates = Vec::new();
            for beam in &beams {
                if beam.finished {
                    // finished beams keep competing on score
                    candidates.push(Beam {
                        generated: beam.generated.clone(),
                        log_prob: beam.log_prob,
                        finished: true,
                    });
                    continue;
                }

                let mut sequence = start.to_vec();
                sequence.extend_from_slice(&beam.generated);
                let log_probs = log_softmax(&step(&sequence));

                // only this beam's top `width` continuations can survive the global cut
                let mut order: Vec<usize> = (0..log_probs.len()).collect();
                order.sort_by(|&a, &b| log_probs[b].total_cmp(&log_probs[a]));
                for &token in order.iter().take(self.width) {
                    let mut generated = beam.generated.clone();
                    let finished = self.eos == Some(token);
                    if !finished {
                        generated.push(token);
                    }
                    candidates.push(Beam {
                        generated,
                        log_prob: beam.log_prob + log_probs[token],
                        finished,
                    });
                }
            }

            candidates.sort_by(|a, b| {
                b.score(self.length_penalty)
                    .total_cmp(&a.score(self.length_penalty))
            });
            candidates.truncate(self.width);
            beams = candidates;
        }

        beams
            .into_iter()
            .max_by(|a, b| {
                a.score(self.length_penalty)
                    .total_cmp(&b.score(self.length_penalty))
            })
            .map(|beam| beam.generated)
            .unwrap_or_default()
    }
}

/// Log-probabilities from logits, stable under large inputs.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&l| l - max - log_sum).collect()
}
