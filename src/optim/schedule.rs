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

    #[test]
    fn step_decay_accepts_an_interval_of_one() {
        let schedule = StepDecay::new(1.0, 1, 0.5);
        assert_eq!(schedule.lr_at(1), 0.5);
        assert_eq!(schedule.lr_at(2), 0.25);
    }

    #[test]
    #[should_panic(expected = "every")]
    fn step_decay_rejects_zero_every() {
        StepDecay::new(1.0, 0, 0.5);
    }

    // The fields are public, so the check in lr_at has to catch this too.
    #[test]
    #[should_panic(expected = "every")]
    fn step_decay_rejects_zero_every_set_after_new() {
        let mut schedule = StepDecay::new(1.0, 1, 0.5);
        schedule.every = 0;
        schedule.lr_at(0);
    }

    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    fn one_cycle(total: usize, warmup_fraction: f32) -> OneCycle {
        OneCycle {
            warmup_fraction,
            ..OneCycle::new(1.0, total)
        }
    }

    #[test]
    fn one_cycle_goes_from_start_to_peak_to_floor() {
        let schedule = OneCycle::new(1.0, 100);
        close(schedule.lr_at(0), 0.04);
        close(schedule.lr_at(30), 1.0);
        close(schedule.lr_at(65), 0.50005);
        close(schedule.lr_at(100), 0.0001);
        close(schedule.lr_at(101), 0.0001);
    }

    #[test]
    fn one_cycle_accepts_both_ends_of_the_fraction_range() {
        for zero in [0.0, -0.0] {
            let schedule = one_cycle(100, zero);
            close(schedule.lr_at(0), 1.0);
            close(schedule.lr_at(100), 0.0001);
        }

        let schedule = one_cycle(100, 1.0);
        close(schedule.lr_at(0), 0.04);
        close(schedule.lr_at(50), 0.52);
        close(schedule.lr_at(100), 1.0);
        close(schedule.lr_at(101), 0.0001);
    }

    #[test]
    fn one_cycle_handles_tiny_totals() {
        for fraction in [0.0, 0.3, 1.0] {
            let schedule = one_cycle(0, fraction);
            close(schedule.lr_at(0), 1.0);
            close(schedule.lr_at(1), 0.0001);
        }

        let schedule = one_cycle(1, 1.0);
        close(schedule.lr_at(0), 0.04);
        close(schedule.lr_at(1), 1.0);
        close(schedule.lr_at(2), 0.0001);
    }

    // 2^24 + 3 rounds up in f32, so a fraction of exactly 1 used to put the
    // warmup past total and underflow the subtraction after it.
    #[test]
    fn one_cycle_survives_a_total_that_rounds_up_in_f32() {
        let total = 16_777_219;
        close(one_cycle(total, 1.0).lr_at(total + 1), 0.0001);
    }

    #[test]
    #[should_panic(expected = "warmup_fraction")]
    fn one_cycle_rejects_a_fraction_above_one() {
        one_cycle(10, 2.0).lr_at(0);
    }

    #[test]
    #[should_panic(expected = "warmup_fraction")]
    fn one_cycle_rejects_a_negative_fraction() {
        one_cycle(10, -0.1).lr_at(0);
    }

    #[test]
    #[should_panic(expected = "warmup_fraction")]
    fn one_cycle_rejects_nan() {
        one_cycle(10, f32::NAN).lr_at(0);
    }

    #[test]
    #[should_panic(expected = "warmup_fraction")]
    fn one_cycle_rejects_infinity() {
        one_cycle(10, f32::INFINITY).lr_at(0);
    }
}
