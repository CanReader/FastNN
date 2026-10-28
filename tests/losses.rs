//! Losses against values worked out by hand from their formulas.

use fastnn::nn::{
    contrastive, cosine_embedding, dice, focal_bce_with_logits, gaussian_nll, info_nce,
    margin_ranking, poisson_nll, smooth_l1, triplet_margin,
};
use fastnn::prelude::*;

const TOLERANCE: f32 = 1e-4;

#[track_caller]
fn assert_close(got: f32, want: f32, what: &str) {
    assert!(
        (got - want).abs() < TOLERANCE,
        "{what}: got {got}, want {want}"
    );
}

#[test]
fn huber_is_quadratic_inside_delta_and_linear_outside() {
    let prediction = Tensor::from_vec(vec![0.5, 3.0], &[2, 1]);
    let target = Tensor::zeros(&[2, 1]);
    // 0.5 * 0.5^2 = 0.125 inside, 1 * (3 - 0.5) = 2.5 outside, mean 1.3125.
    assert_close(huber(&prediction, &target, 1.0).item(), 1.3125, "huber");
    assert_close(
        smooth_l1(&prediction, &target).item(),
        1.3125,
        "smooth_l1 = huber(delta=1)",
    );
}

#[test]
fn kl_divergence_matches_the_definition() {
    let target = Tensor::from_vec(vec![0.5, 0.5], &[1, 2]);
    let prediction = Tensor::from_vec(vec![0.25, 0.75], &[1, 2]);
    // 0.5 * ln(0.5 / 0.25) + 0.5 * ln(0.5 / 0.75) = 0.14384
    assert_close(kl_divergence(&prediction, &target).item(), 0.14384, "kl");
    assert_close(
        kl_divergence(&target, &target).item(),
        0.0,
        "kl of identical",
    );
}

#[test]
fn nll_of_log_softmax_equals_fused_cross_entropy() {
    let logits = Tensor::from_vec(vec![1.0, -0.5, 2.0, 0.3, 0.1, -1.2], &[2, 3]);
    let targets = [2usize, 0];
    assert_close(
        nll(&logits.log_softmax(), &targets).item(),
        cross_entropy(&logits, &targets).item(),
        "nll ∘ log_softmax = cross_entropy",
    );
}

#[test]
fn gaussian_and_poisson_nll_match_their_formulas() {
    // 0.5 * (logvar + (x - mean)^2 / var) = 0.5 * 4
    let (mean, logvar) = (Tensor::zeros(&[1]), Tensor::zeros(&[1]));
    let x = Tensor::from_vec(vec![2.0], &[1]);
    assert_close(gaussian_nll(&mean, &logvar, &x).item(), 2.0, "gaussian nll");

    // exp(log_rate) - count * log_rate = 1
    let log_rate = Tensor::zeros(&[1]);
    let count = Tensor::from_vec(vec![3.0], &[1]);
    assert_close(poisson_nll(&log_rate, &count).item(), 1.0, "poisson nll");
}

#[test]
fn focal_with_gamma_zero_is_alpha_scaled_bce() {
    let logits = Tensor::from_vec(vec![0.7, -1.3, 0.2], &[3, 1]);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], &[3, 1]);
    // gamma 0 drops the modulation, alpha 0.5 scales both classes equally.
    assert_close(
        focal_bce_with_logits(&logits, &target, 0.0, 0.5).item(),
        0.5 * bce_with_logits(&logits, &target).item(),
        "focal(gamma=0, alpha=0.5) = 0.5 * bce",
    );
}

#[test]
fn dice_is_zero_on_perfect_binary_overlap_and_high_on_none() {
    let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0], &[4]);
    assert_close(dice(&mask, &mask).item(), 0.0, "perfect overlap");

    let inverse = Tensor::from_vec(vec![0.0, 1.0, 0.0, 0.0], &[4]);
    // 1 - (0 + 1) / (4 + 1)
    assert_close(dice(&mask, &inverse).item(), 0.8, "no overlap");
}

#[test]
fn plain_cross_entropy_loss_matches_the_fused_function() {
    let logits = Tensor::from_vec(vec![1.0, -0.5, 2.0, 0.3, 0.1, -1.2], &[2, 3]);
    let targets = [2usize, 0];
    assert_close(
        CrossEntropyLoss::new().compute(&logits, &targets).item(),
        cross_entropy(&logits, &targets).item(),
        "builder defaults = cross_entropy",
    );
}

#[test]
fn label_smoothing_matches_the_smoothed_target_distribution() {
    // p = [1/4, 3/4], target 1, smoothing 0.2 gives q = [0.1, 0.9]
    // loss = -(0.1 * ln(1/4) + 0.9 * ln(3/4)) = 0.39752
    let logits = Tensor::from_vec(vec![0.0, 3.0f32.ln()], &[1, 2]);
    let loss = CrossEntropyLoss::new()
        .label_smoothing(0.2)
        .compute(&logits, &[1]);
    assert_close(loss.item(), 0.39752, "smoothed ce");
}

#[test]
fn class_weights_take_a_weighted_mean() {
    let logits = Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]);
    let separate: Vec<f32> = (0..2)
        .map(|row| cross_entropy(&logits.narrow(0, row, 1), &[row]).item())
        .collect();

    let weighted = CrossEntropyLoss::new()
        .class_weights(vec![1.0, 3.0])
        .compute(&logits, &[0, 1]);
    let expected = (1.0 * separate[0] + 3.0 * separate[1]) / 4.0;
    assert_close(weighted.item(), expected, "mean over the weight sum");
}

#[test]
fn ignored_rows_contribute_neither_loss_nor_gradient() {
    let logits = fastnn::nn::Param::new(Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]));
    let loss_fn = CrossEntropyLoss::new().ignore_index(1);

    let loss = loss_fn.compute(&logits.tensor(), &[0, 1]);
    assert_close(
        loss.item(),
        cross_entropy(&logits.value().narrow(0, 0, 1), &[0]).item(),
        "ignored row excluded from the mean",
    );

    loss.backward();
    let grad = logits.grad().unwrap().to_vec();
    assert!(
        grad[2] == 0.0 && grad[3] == 0.0,
        "ignored row leaked gradient: {grad:?}"
    );
    assert!(grad[0] != 0.0, "participating row must receive gradient");

    // Every row ignored.
    let all_ignored = loss_fn.compute(&logits.tensor().detach(), &[1, 1]);
    assert_close(all_ignored.item(), 0.0, "all-ignored batch");
}

#[test]
fn reduction_modes_relate_as_mean_sum_and_rows() {
    let logits = Tensor::from_vec(vec![1.0, -0.5, 2.0, 0.3, 0.1, -1.2], &[2, 3]);
    let targets = [2usize, 0];

    let mean = CrossEntropyLoss::new().compute(&logits, &targets).item();
    let sum = CrossEntropyLoss::new()
        .reduction(Reduction::Sum)
        .compute(&logits, &targets)
        .item();
    let rows = CrossEntropyLoss::new()
        .reduction(Reduction::None)
        .compute(&logits, &targets);

    assert_eq!(rows.shape(), &[2], "None keeps one loss per row");
    assert_close(sum, mean * 2.0, "sum = mean * batch");
    assert_close(rows.to_vec().iter().sum::<f32>(), sum, "rows sum to Sum");
}

#[test]
fn metric_losses_match_hand_worked_geometry() {
    // d(a,p) = 1, d(a,n) = 1.5, margin 1, so the hinge is 0.5.
    let anchor = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]);
    let positive = Tensor::from_vec(vec![1.0, 0.0], &[1, 2]);
    let negative = Tensor::from_vec(vec![1.5, 0.0], &[1, 2]);
    assert_close(
        triplet_margin(&anchor, &positive, &negative, 1.0).item(),
        0.5,
        "triplet",
    );

    // Orthogonal vectors: the similar pair costs 1, the dissimilar one 0.
    let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]);
    let b = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[2, 2]);
    assert_close(
        cosine_embedding(&a, &b, &[1.0, -1.0], 0.0).item(),
        0.5,
        "cosine embedding",
    );

    // x1 already ahead costs 0, the wrong order costs 1 + margin.
    let x1 = Tensor::from_vec(vec![2.0, 2.0], &[2, 1]);
    let x2 = Tensor::from_vec(vec![1.0, 1.0], &[2, 1]);
    assert_close(
        margin_ranking(&x1, &x2, &[1.0, -1.0], 0.5).item(),
        0.75,
        "margin ranking",
    );

    // Matched at distance 0 costs 0, mismatched at distance 1 with margin 2 costs 1.
    let u = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let v = Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0], &[2, 2]);
    assert_close(
        contrastive(&u, &v, &[1.0, 0.0], 2.0).item(),
        0.5,
        "contrastive",
    );
}

#[test]
fn info_nce_on_an_identity_batch_matches_the_closed_form() {
    // Queries equal keys at temperature 1, so each row's logits are a row of I
    // and the loss is ln(1 + (n - 1) / e).
    let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let expected = (1.0f32 + 1.0 * (-1.0f32).exp()).ln();
    assert_close(
        info_nce(&identity, &identity, 1.0).item(),
        expected,
        "info_nce",
    );
}
