use fastnn::optim::{LrSchedule, OneCycle, StepDecay};

#[test]
#[should_panic(expected = "every")]
fn step_decay_rejects_zero_every_in_constructor() {
    StepDecay::new(1.0, 0, 0.5);
}

#[test]
#[should_panic(expected = "every")]
fn step_decay_rejects_zero_every_after_mutation() {
    let mut schedule = StepDecay::new(1.0, 1, 0.5);
    schedule.every = 0;
    schedule.lr_at(0);
}

#[test]
#[should_panic(expected = "every")]
fn step_decay_rejects_zero_every_in_struct_literal() {
    let schedule = StepDecay {
        base: 1.0,
        every: 0,
        gamma: 0.5,
    };
    schedule.lr_at(0);
}

#[test]
fn step_decay_accepts_an_interval_of_one() {
    let schedule = StepDecay::new(1.0, 1, 0.5);
    assert_eq!(schedule.lr_at(0), 1.0);
    assert_eq!(schedule.lr_at(1), 0.5);
    assert_eq!(schedule.lr_at(2), 0.25);
}

#[test]
#[should_panic(expected = "warmup_fraction")]
fn one_cycle_rejects_excessive_fraction_before_warmup() {
    let mut schedule = OneCycle::new(1.0, 10);
    schedule.warmup_fraction = 2.0;
    schedule.lr_at(0);
}

#[test]
#[should_panic(expected = "warmup_fraction")]
fn one_cycle_rejects_excessive_fraction_in_struct_literal() {
    let schedule = OneCycle {
        warmup_fraction: 2.0,
        ..OneCycle::new(1.0, 10)
    };
    schedule.lr_at(0);
}

#[test]
fn one_cycle_rejects_negative_and_nonfinite_fractions() {
    let mut failures = Vec::new();
    for (name, fraction) in [
        ("negative", -0.1),
        ("NaN", f32::NAN),
        ("positive infinity", f32::INFINITY),
        ("negative infinity", f32::NEG_INFINITY),
    ] {
        let result = std::panic::catch_unwind(|| {
            let mut schedule = OneCycle::new(1.0, 10);
            schedule.warmup_fraction = fraction;
            schedule.lr_at(0)
        });
        match result {
            Ok(_) => failures.push(name),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("");
                if !message.contains("warmup_fraction") {
                    failures.push(name);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fractions not clearly rejected: {failures:?}"
    );
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn one_cycle_default_reaches_start_peak_and_floor() {
    let schedule = OneCycle::new(1.0, 100);
    close(schedule.lr_at(0), 0.04);
    close(schedule.lr_at(30), 1.0);
    close(schedule.lr_at(65), 0.50005);
    close(schedule.lr_at(100), 0.0001);
    close(schedule.lr_at(101), 0.0001);
}

#[test]
fn one_cycle_zero_fraction_skips_warmup() {
    for fraction in [0.0, -0.0] {
        let mut schedule = OneCycle::new(1.0, 100);
        schedule.warmup_fraction = fraction;
        close(schedule.lr_at(0), 1.0);
        close(schedule.lr_at(50), 0.50005);
        close(schedule.lr_at(100), 0.0001);
    }
}

#[test]
fn one_cycle_full_fraction_preserves_the_full_warmup() {
    let mut schedule = OneCycle::new(1.0, 100);
    schedule.warmup_fraction = 1.0;
    close(schedule.lr_at(0), 0.04);
    close(schedule.lr_at(50), 0.52);
    close(schedule.lr_at(100), 1.0);
    close(schedule.lr_at(101), 0.0001);
}

#[test]
fn one_cycle_zero_total_retains_its_existing_curve() {
    for fraction in [0.0, 0.3, 1.0] {
        let mut schedule = OneCycle::new(1.0, 0);
        schedule.warmup_fraction = fraction;
        close(schedule.lr_at(0), 1.0);
        close(schedule.lr_at(1), 0.0001);
    }
}

#[test]
fn one_cycle_single_step_retains_its_existing_curves() {
    let default = OneCycle::new(1.0, 1);
    close(default.lr_at(0), 1.0);
    close(default.lr_at(1), 0.0001);

    let mut full_warmup = OneCycle::new(1.0, 1);
    full_warmup.warmup_fraction = 1.0;
    close(full_warmup.lr_at(0), 0.04);
    close(full_warmup.lr_at(1), 1.0);
    close(full_warmup.lr_at(2), 0.0001);
}

#[test]
fn one_cycle_full_fraction_reaches_floor_after_a_large_total() {
    let total = 16_777_219;
    let mut schedule = OneCycle::new(1.0, total);
    schedule.warmup_fraction = 1.0;
    close(schedule.lr_at(total + 1), 0.0001);
}
