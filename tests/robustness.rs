//! The guard rails a long unattended run depends on.
//!
//! Each of these covers a failure that is silent without it: an optimizer that
//! forgets its momentum, a NaN that only surfaces once the checkpoint is already
//! poisoned, a frozen layer that trains anyway.
//!
//!     cargo test --no-default-features --test robustness

use fastnn::autograd::detect_anomaly;
use fastnn::prelude::*;
use fastnn::serialize::{load_training, save_training};

/// A scratch path that cleans itself up.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(name: &str) -> TempFile {
        let mut path = std::env::temp_dir();
        path.push(format!("fastnn-test-{name}-{}.fdl", std::process::id()));
        TempFile(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn small_model() -> Sequential {
    Sequential::new()
        .add(Linear::new(4, 6))
        .add(ReLU)
        .add(Linear::new(6, 3))
}

fn batch() -> (Tensor, Vec<usize>) {
    (Tensor::randn(&[5, 4]), vec![0, 1, 2, 1, 0])
}

fn train_steps(
    model: &dyn Module,
    opt: &mut dyn Optimizer,
    inputs: &Tensor,
    targets: &[usize],
    steps: usize,
) {
    for _ in 0..steps {
        let loss = cross_entropy(&model.forward(inputs), targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
}

// ── Resuming a run ───────────────────────────────────────────────────────────

#[test]
fn resuming_matches_training_straight_through() {
    let file = TempFile::new("resume");
    let (inputs, targets) = batch();

    // Run twenty steps without interruption.
    manual_seed(9);
    let reference = small_model();
    let mut opt = Adam::new(reference.parameters(), 0.05);
    train_steps(&reference, &mut opt, &inputs, &targets, 20);
    let uninterrupted = cross_entropy(&reference.forward(&inputs), &targets).item();

    // The same twenty steps, saved and reloaded halfway.
    manual_seed(9);
    let resumed = small_model();
    let mut opt = Adam::new(resumed.parameters(), 0.05);
    train_steps(&resumed, &mut opt, &inputs, &targets, 10);
    save_training(&resumed, &opt, 10, file.path()).unwrap();

    let restored = small_model();
    let mut opt = Adam::new(restored.parameters(), 0.05);
    let step = load_training(&restored, &mut opt, file.path()).unwrap();
    assert_eq!(step, 10);
    train_steps(&restored, &mut opt, &inputs, &targets, 10);

    let after_resume = cross_entropy(&restored.forward(&inputs), &targets).item();
    assert!(
        (uninterrupted - after_resume).abs() < 1e-4,
        "resuming diverged: {uninterrupted} straight through vs {after_resume} resumed"
    );
}

#[test]
fn a_cold_optimizer_takes_a_different_path() {
    // Guards the test above. If the moments and step count did not really survive
    // the round trip, both paths here would be identical and
    // `resuming_matches_training_straight_through` could pass while restoring
    // nothing. Restoring the weights alone must visibly differ.
    let file = TempFile::new("cold");
    let (inputs, targets) = batch();

    manual_seed(9);
    let model = small_model();
    let mut opt = Adam::new(model.parameters(), 0.05);
    train_steps(&model, &mut opt, &inputs, &targets, 10);
    save_training(&model, &opt, 10, file.path()).unwrap();

    let continue_step = |restore_optimizer: bool| {
        let restored = small_model();
        let mut opt = Adam::new(restored.parameters(), 0.05);
        // Both paths load the same weights; only one also loads the optimizer.
        let mut loader = Adam::new(restored.parameters(), 0.05);
        load_training(&restored, &mut loader, file.path()).unwrap();
        if restore_optimizer {
            opt = loader;
        }
        train_steps(&restored, &mut opt, &inputs, &targets, 5);
        restored.parameters()[0].value().to_vec()
    };

    let warm = continue_step(true);
    let cold = continue_step(false);
    let gap: f32 = warm.iter().zip(&cold).map(|(a, b)| (a - b).abs()).sum();

    assert!(
        gap > 1e-4,
        "a cold optimizer produced the same weights as a restored one, so the \
         state is probably not being restored at all (total difference {gap})"
    );
}

#[test]
fn a_plain_checkpoint_is_rejected_for_resuming() {
    let file = TempFile::new("plain");
    let model = small_model();
    save(&model, file.path()).unwrap();

    let other = small_model();
    let mut opt = Adam::new(other.parameters(), 0.05);
    let err = load_training(&other, &mut opt, file.path()).unwrap_err();
    assert!(
        err.to_string().contains("optimizer state"),
        "expected a clear explanation, got: {err}"
    );
}

#[test]
fn sgd_momentum_survives_a_round_trip() {
    let file = TempFile::new("sgd");
    let (inputs, targets) = batch();

    manual_seed(4);
    let model = small_model();
    let mut opt = SGD::new(model.parameters(), 0.05).momentum(0.9);
    train_steps(&model, &mut opt, &inputs, &targets, 8);
    save_training(&model, &opt, 8, file.path()).unwrap();

    let restored = small_model();
    let mut opt = SGD::new(restored.parameters(), 0.05).momentum(0.9);
    assert_eq!(load_training(&restored, &mut opt, file.path()).unwrap(), 8);

    let velocity = opt.state();
    assert!(!velocity.tensors.is_empty(), "momentum was not restored");
    assert_eq!(velocity.steps, 8);
}

#[test]
fn a_mismatched_optimizer_is_reported() {
    let file = TempFile::new("mismatch");
    let (inputs, targets) = batch();

    let model = small_model();
    let mut opt = Adam::new(model.parameters(), 0.05);
    train_steps(&model, &mut opt, &inputs, &targets, 3);
    save_training(&model, &opt, 3, file.path()).unwrap();

    // Same architecture, but the optimizer covers only part of it.
    let other = small_model();
    let mut fewer = Adam::new(other.parameters()[..2].to_vec(), 0.05);
    assert!(load_training(&other, &mut fewer, file.path()).is_err());
}

// ── Anomaly detection ────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "Log")]
fn anomaly_detection_names_the_op_at_fault() {
    let x = Param::new(Tensor::zeros(&[2, 2]));
    detect_anomaly(|| x.tensor().log().sum().backward());
}

#[test]
fn anomaly_detection_leaves_healthy_gradients_alone() {
    let x = Param::new(Tensor::ones(&[2, 2]));
    detect_anomaly(|| x.tensor().mul_scalar(3.0).sum().backward());
    assert!(x.grad().unwrap().is_finite());
}

#[test]
fn anomaly_detection_resets_after_a_panic() {
    let x = Param::new(Tensor::zeros(&[2, 2]));
    let caught = std::panic::catch_unwind(|| {
        detect_anomaly(|| x.tensor().log().sum().backward());
    });
    assert!(caught.is_err());
    assert!(
        !fastnn::autograd::is_detecting(),
        "the guard did not unwind cleanly"
    );
}

#[test]
fn finiteness_guard_spots_a_poisoned_gradient() {
    let model = small_model();
    let opt = Adam::new(model.parameters(), 0.05);
    let (inputs, targets) = batch();

    cross_entropy(&model.forward(&inputs), &targets).backward();
    assert!(opt.gradients_are_finite());

    let poisoned = model.parameters()[0].clone();
    poisoned.set_grad(Tensor::full(&poisoned.shape(), f32::NAN));
    assert!(!opt.gradients_are_finite());
}

// ── Freezing ─────────────────────────────────────────────────────────────────

#[test]
fn a_frozen_layer_does_not_move() {
    let backbone = Linear::new(4, 6);
    let head = Linear::new(6, 3);
    backbone.freeze();

    let before = backbone.weight.value().to_vec();
    let params: Vec<Param> = backbone
        .parameters()
        .into_iter()
        .chain(head.parameters())
        .collect();
    let mut opt = Adam::new(params, 0.1);
    let (inputs, targets) = batch();

    for _ in 0..5 {
        let logits = head.forward(&backbone.forward(&inputs));
        let loss = cross_entropy(&logits, &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    assert_eq!(
        backbone.weight.value().to_vec(),
        before,
        "frozen weights moved"
    );
    assert_ne!(
        head.weight.value().to_vec(),
        head.weight.value().mul_scalar(0.0).to_vec()
    );
}

#[test]
fn freezing_skips_the_gradient_entirely() {
    // A frozen parameter hands out a detached tensor, so the backward pass stops
    // there — no gradient is computed only to be thrown away.
    let layer = Linear::new(4, 6);
    layer.freeze();

    layer.forward(&Tensor::randn(&[2, 4])).sum().backward();
    assert!(
        layer.weight.grad().is_none(),
        "a frozen parameter accumulated a gradient"
    );
}

#[test]
fn unfreezing_puts_a_layer_back_in_training() {
    let layer = Linear::new(4, 6);
    layer.freeze();
    assert!(layer.trainable_parameters().is_empty());

    layer.unfreeze();
    assert_eq!(layer.trainable_parameters().len(), 2);

    layer.forward(&Tensor::randn(&[2, 4])).sum().backward();
    assert!(layer.weight.grad().is_some());
}

#[test]
fn freezing_reaches_through_a_container() {
    let model = small_model();
    model.freeze();
    assert!(model.trainable_parameters().is_empty());
    assert_eq!(
        model.parameters().len(),
        4,
        "freezing should not hide parameters"
    );
}

// ── Fallible ops ─────────────────────────────────────────────────────────────

#[test]
fn shape_errors_are_reported_not_fatal() {
    let x = Tensor::zeros(&[4, 8]);

    let err = x.try_matmul(&Tensor::zeros(&[3, 3])).unwrap_err();
    assert!(
        err.to_string().contains("inner dimensions"),
        "unhelpful message: {err}"
    );

    assert!(x.try_add(&Tensor::zeros(&[5, 8])).is_err());
    assert!(x.try_reshape(&[7, 7]).is_err());
    assert!(x.try_index_select(&[99]).is_err());
}

#[test]
fn fallible_ops_agree_with_the_panicking_ones() {
    let a = Tensor::randn(&[4, 8]);
    let b = Tensor::randn(&[8, 2]);

    assert_eq!(a.try_matmul(&b).unwrap().to_vec(), a.matmul(&b).to_vec());
    assert_eq!(a.try_add(&a).unwrap().to_vec(), a.add(&a).to_vec());
    assert_eq!(
        a.try_reshape(&[2, 16]).unwrap().shape(),
        a.reshape(&[2, 16]).shape()
    );
}

#[test]
fn fallible_ops_still_build_the_graph() {
    let w = Param::new(Tensor::randn(&[8, 2]));
    let x = Tensor::randn(&[4, 8]);

    x.try_matmul(&w.tensor()).unwrap().sum().backward();
    assert!(w.grad().is_some(), "try_matmul dropped the graph");
}
