//! Loss functions against hand-computed references.
//!
//! Each case pins a loss to a value worked out on paper from its defining
//! formula, so a refactor that changes semantics — a swapped sign, a missing
//! normalizer, a wrong direction of KL — fails loudly with numbers a reader
//! can recompute.

use fastnn::nn::{
    contrastive, cosine_embedding, dice, focal_bce_with_logits, gaussian_nll, info_nce,
    margin_ranking, poisson_nll, smooth_l1, triplet_margin,
};
use fastnn::prelude::*;

#[track_caller]
fn assert_close(got: f32, want: f32, what: &str) {
    assert!((got - want).abs() < 1e-4, "{what}: got {got}, want {want}");
}

#[test]
fn huber_is_quadratic_inside_delta_and_linear_outside() {
    let prediction = Tensor::from_vec(vec![0.5, 3.0], &[2, 1]);
    let target = Tensor::zeros(&[2, 1]);
    // ½·0.5² = 0.125 inside; δ(|3| − ½δ) = 2.5 outside; mean = 1.3125.
    assert_close(huber(&prediction, &target, 1.0).item(), 1.3125, "huber");
    assert_close(
        smooth_l1(&prediction, &target).item(),
        1.3125,
        "smooth_l1 = huber(δ=1)",
    );
}

#[test]
fn kl_divergence_matches_the_definition() {
    let target = Tensor::from_vec(vec![0.5, 0.5], &[1, 2]);
    let prediction = Tensor::from_vec(vec![0.25, 0.75], &[1, 2]);
    // 0.5·ln(0.5/0.25) + 0.5·ln(0.5/0.75) = 0.5(ln 2 + ln ⅔) ≈ 0.14384.
    assert_close(kl_divergence(&prediction, &target).item(), 0.14384, "kl");
    // KL(p ‖ p) = 0.
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
    // ½(log σ² + (x−μ)²/σ²) with μ=0, log σ²=0, x=2 → ½·4 = 2.
    let (mean, logvar) = (Tensor::zeros(&[1]), Tensor::zeros(&[1]));
    let x = Tensor::from_vec(vec![2.0], &[1]);
    assert_close(gaussian_nll(&mean, &logvar, &x).item(), 2.0, "gaussian nll");

    // e^{log λ} − k·log λ with log λ = 0, k = 3 → 1.
    let log_rate = Tensor::zeros(&[1]);
    let count = Tensor::from_vec(vec![3.0], &[1]);
    assert_close(poisson_nll(&log_rate, &count).item(), 1.0, "poisson nll");
}

#[test]
fn focal_with_gamma_zero_is_alpha_scaled_bce() {
    let logits = Tensor::from_vec(vec![0.7, -1.3, 0.2], &[3, 1]);
    let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], &[3, 1]);
    // γ = 0 removes the modulation; α = ½ scales both classes equally.
    assert_close(
        focal_bce_with_logits(&logits, &target, 0.0, 0.5).item(),
        0.5 * bce_with_logits(&logits, &target).item(),
        "focal(γ=0, α=½) = ½·bce",
    );
}

#[test]
fn dice_is_zero_on_perfect_binary_overlap_and_high_on_none() {
    let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0], &[4]);
    assert_close(dice(&mask, &mask).item(), 0.0, "perfect overlap");

    let inverse = Tensor::from_vec(vec![0.0, 1.0, 0.0, 0.0], &[4]);
    // No overlap: 1 − (0+1)/(4+1) = 0.8.
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
    // p = [¼, ¾], target 1, s = 0.2 ⇒ q = [0.1, 0.9]:
    // loss = −(0.1·ln ¼ + 0.9·ln ¾) ≈ 0.39752.
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
    assert_close(
        weighted.item(),
        expected,
        "weighted mean over Σw, not batch",
    );
}

#[test]
fn ignored_rows_contribute_neither_loss_nor_gradient() {
    let logits = fastnn::nn::Param::new(Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]));
    let loss_fn = CrossEntropyLoss::new().ignore_index(1);

    let loss = loss_fn.compute(&logits.tensor(), &[0, 1]);
    // Only row 0 participates, so the mean is exactly row 0's loss.
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

    // Every row ignored: zero loss, and finite everywhere.
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
    assert_close(sum, mean * 2.0, "sum = mean·batch");
    assert_close(rows.to_vec().iter().sum::<f32>(), sum, "rows sum to Sum");
}

#[test]
fn metric_losses_match_hand_worked_geometry() {
    // Triplet: d(a,p) = 1, d(a,n) = 1.5, margin 1 → hinge at 0.5.
    let anchor = Tensor::from_vec(vec![0.0, 0.0], &[1, 2]);
    let positive = Tensor::from_vec(vec![1.0, 0.0], &[1, 2]);
    let negative = Tensor::from_vec(vec![1.5, 0.0], &[1, 2]);
    assert_close(
        triplet_margin(&anchor, &positive, &negative, 1.0).item(),
        0.5,
        "triplet",
    );

    // Cosine: orthogonal vectors. Similar pair costs 1 − 0 = 1; dissimilar
    // pair with margin 0 costs relu(0 − 0) = 0; mean = 0.5.
    let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]);
    let b = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[2, 2]);
    assert_close(
        cosine_embedding(&a, &b, &[1.0, -1.0], 0.0).item(),
        0.5,
        "cosine embedding",
    );

    // Ranking: x1 already margin ahead → 0; wrong order → 1 + margin.
    let x1 = Tensor::from_vec(vec![2.0, 2.0], &[2, 1]);
    let x2 = Tensor::from_vec(vec![1.0, 1.0], &[2, 1]);
    assert_close(
        margin_ranking(&x1, &x2, &[1.0, -1.0], 0.5).item(),
        0.75,
        "margin ranking",
    );

    // Contrastive: matched at distance 0 → 0; mismatched at distance 1 with
    // margin 2 → (2−1)² = 1; mean = 0.5.
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
    // Orthonormal queries equal to keys at τ = 1: each row's logits are its
    // own row of I, so loss = −ln(e/(e + (n−1))) = ln(1 + (n−1)e⁻¹).
    let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let expected = (1.0f32 + 1.0 * (-1.0f32).exp()).ln();
    assert_close(
        info_nce(&identity, &identity, 1.0).item(),
        expected,
        "info_nce",
    );
}
