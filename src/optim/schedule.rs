//! Learning-rate schedules.
//!
//! A schedule is a pure function of the step number, not something that mutates
//! an optimizer. That makes it trivial to test, to plot, and to resume from a
//! checkpoint — and it composes, so [`Warmup`] can wrap any of the others.
//!
//! ```
//! use fastnn::prelude::*;
//! # let model = Sequential::new().add(Linear::new(4, 2));
//! let mut opt = Adam::new(model.parameters(), 3e-4);
//! let schedule = Warmup::new(CosineAnnealing::new(3e-4, 10_000), 200);
//!
//! for step in 0..10_000 {
//!     opt.set_lr(schedule.lr_at(step));
//!     // ... forward, backward, opt.step()
//! #   break;
//! }
//! ```

use std::f32::consts::PI;

/// A learning rate as a function of the training step, counting from 0.
pub trait LrSchedule: Send + Sync {
    fn lr_at(&self, step: usize) -> f32;
}

/// The same rate forever.
pub struct Constant(pub f32);

impl LrSchedule for Constant {
    fn lr_at(&self, _step: usize) -> f32 {
        self.0
    }
}

/// Multiply by `gamma` every `every` steps.
pub struct StepDecay {
    pub base: f32,
    /// The positive number of steps between rate changes.
    pub every: usize,
    pub gamma: f32,
}

impl StepDecay {
    /// Create a schedule that decays every `every` steps.
    ///
    /// # Panics
    ///
    /// Panics if `every` is zero.
    pub fn new(base: f32, every: usize, gamma: f32) -> StepDecay {
        assert!(every > 0, "every must be > 0");
        StepDecay { base, every, gamma }
    }
}

impl LrSchedule for StepDecay {
    /// # Panics
    ///
    /// Panics if the public `every` field is zero.
    fn lr_at(&self, step: usize) -> f32 {
        assert!(self.every > 0, "every must be > 0");
        self.base * self.gamma.powi((step / self.every) as i32)
    }
}

/// Follow half a cosine from `base` down to `floor` over `total` steps.
///
/// The slow start and slow finish are the point: most of the decay happens in
/// the middle, leaving a long low-rate tail to settle in.
pub struct CosineAnnealing {
    pub base: f32,
    pub floor: f32,
    pub total: usize,
}

impl CosineAnnealing {
    pub fn new(base: f32, total: usize) -> CosineAnnealing {
        CosineAnnealing {
            base,
            floor: 0.0,
            total,
        }
    }

    /// Stop decaying at `floor` instead of zero.
    pub fn floor(mut self, floor: f32) -> CosineAnnealing {
        self.floor = floor;
        self
    }
}

impl LrSchedule for CosineAnnealing {
    fn lr_at(&self, step: usize) -> f32 {
        let progress = (step as f32 / self.total.max(1) as f32).min(1.0);
        self.floor + 0.5 * (self.base - self.floor) * (1.0 + (PI * progress).cos())
    }
}

/// Ramp linearly from zero over the first `steps`, then hand off to `inner`.
///
/// Adaptive optimizers have unreliable variance estimates for their first few
/// hundred steps; warming up stops that turning into a large, badly aimed update.
pub struct Warmup<S> {
    inner: S,
    steps: usize,
}

impl<S: LrSchedule> Warmup<S> {
    pub fn new(inner: S, steps: usize) -> Warmup<S> {
        Warmup { inner, steps }
    }
}

impl<S: LrSchedule> LrSchedule for Warmup<S> {
    fn lr_at(&self, step: usize) -> f32 {
        let target = self.inner.lr_at(step);
        if step < self.steps {
            target * (step + 1) as f32 / self.steps as f32
        } else {
            target
        }
    }
}

/// One cycle: ramp up to `peak`, then cosine down well below the start.
pub struct OneCycle {
    pub peak: f32,
    pub total: usize,
    /// The finite fraction of steps used for warmup, in the inclusive range [0, 1].
    pub warmup_fraction: f32,
    pub start_divisor: f32,
    pub final_divisor: f32,
}

impl OneCycle {
    pub fn new(peak: f32, total: usize) -> OneCycle {
        OneCycle {
            peak,
            total,
            warmup_fraction: 0.3,
            start_divisor: 25.0,
            final_divisor: 10_000.0,
        }
    }
}

impl LrSchedule for OneCycle {
    /// # Panics
    ///
    /// Panics if `warmup_fraction` is not finite or is outside [0, 1].
    fn lr_at(&self, step: usize) -> f32 {
        assert!(
            (0.0..=1.0).contains(&self.warmup_fraction),
            "warmup_fraction must be in [0, 1], got {}",
            self.warmup_fraction
        );
        // Converting total to f32 can round warmup above total even at a fraction of 1.
        let warmup = ((self.warmup_fraction * self.total as f32) as usize).min(self.total);
        let start = self.peak / self.start_divisor;
        let end = self.peak / self.final_divisor;

        if step < warmup {
            start + (self.peak - start) * (step as f32 / warmup.max(1) as f32)
        } else {
            let remaining = (self.total - warmup).max(1);
            let progress = ((step - warmup) as f32 / remaining as f32).min(1.0);
            end + 0.5 * (self.peak - end) * (1.0 + (PI * progress).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_spans_base_to_floor() {
        let schedule = CosineAnnealing::new(1.0, 100);
        assert!((schedule.lr_at(0) - 1.0).abs() < 1e-6);
        assert!(schedule.lr_at(100).abs() < 1e-6);
        assert!((schedule.lr_at(50) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn warmup_ramps_then_defers() {
        let schedule = Warmup::new(Constant(1.0), 10);
        assert!((schedule.lr_at(0) - 0.1).abs() < 1e-6);
        assert!((schedule.lr_at(9) - 1.0).abs() < 1e-6);
        assert!((schedule.lr_at(50) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn step_decay_halves_on_schedule() {
        let schedule = StepDecay::new(1.0, 10, 0.5);
        assert_eq!(schedule.lr_at(9), 1.0);
        assert_eq!(schedule.lr_at(10), 0.5);
        assert_eq!(schedule.lr_at(20), 0.25);
    }
}
