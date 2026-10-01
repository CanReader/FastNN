//! Losses over embedding geometry: pull related points together, push
//! unrelated ones apart.
//!
//! Classification losses grade a prediction against a label; these grade the
//! *shape of the embedding space* itself — distances and angles between pairs
//! and triples — which is what retrieval, verification, and self-supervised
//! pretraining actually optimize. All of them compose from ordinary
//! differentiable ops, so their gradients come from the existing rules.

use crate::tensor::Tensor;

use super::loss::cross_entropy;

/// Guards divisions and square roots at zero distance, where the true
/// derivative of ‖·‖₂ does not exist.
const EPS: f32 = 1e-8;

/// Row-wise Euclidean distance between `[batch, dim]` tensors → `[batch, 1]`.
///
/// `√(Σd² + ε)` rather than `√Σd²`: at zero distance the exact root has an
/// undefined gradient, and the ε turns it into a smooth minimum instead.
fn row_distance(a: &Tensor, b: &Tensor) -> Tensor {
    a.sub(b).square().sum_axis_keep(1).add_scalar(EPS).sqrt()
}

/// Row-wise cosine similarity between `[batch, dim]` tensors → `[batch, 1]`.
fn row_cosine(a: &Tensor, b: &Tensor) -> Tensor {
    let dot = a.mul(b).sum_axis_keep(1);
    let norms = a
        .square()
        .sum_axis_keep(1)
        .sqrt()
        .mul(&b.square().sum_axis_keep(1).sqrt());
    dot.div(&norms.add_scalar(EPS))
}

/// A `[batch, 1]` selector holding 1.0 where `predicate` holds.
fn mask(targets: &[f32], predicate: impl Fn(f32) -> bool, device: crate::tensor::Device) -> Tensor {
    let values: Vec<f32> = targets
        .iter()
        .map(|&t| if predicate(t) { 1.0 } else { 0.0 })
        .collect();
    Tensor::from_vec(values, &[targets.len(), 1]).to(device)
}

/// Cosine embedding loss: for each row pair, `1 − cos` when `target = +1`,
/// `max(0, cos − margin)` when `target = −1`.
///
/// Similar pairs are pulled to angle zero; dissimilar ones are only pushed
/// until their similarity drops below the margin — beyond that they stop
/// mattering, which is what keeps the space from collapsing outward forever.
pub fn cosine_embedding(a: &Tensor, b: &Tensor, targets: &[f32], margin: f32) -> Tensor {
    assert_eq!(
        a.dim(0),
        targets.len(),
        "{} rows but {} targets",
        a.dim(0),
        targets.len()
    );
    let cosine = row_cosine(a, b);

    let similar = mask(targets, |t| t > 0.0, a.device());
    let dissimilar = mask(targets, |t| t <= 0.0, a.device());

    let pull = cosine.neg().add_scalar(1.0).mul(&similar);
    let push = cosine.add_scalar(-margin).relu().mul(&dissimilar);
    pull.add(&push).mean()
}

/// Triplet margin loss: `max(0, d(anchor, positive) − d(anchor, negative) + margin)`.
///
/// The hinge goes silent once the negative is `margin` farther than the
/// positive — only violating triplets produce gradient, so what you mine into
/// the batch is what the model learns from.
pub fn triplet_margin(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
) -> Tensor {
    let to_positive = row_distance(anchor, positive);
    let to_negative = row_distance(anchor, negative);
    to_positive
        .sub(&to_negative)
        .add_scalar(margin)
        .relu()
        .mean()
}

/// Margin ranking loss: `max(0, −t·(x₁ − x₂) + margin)` with `t = ±1` saying
/// which input should score higher. The SVM hinge, applied to rankings.
pub fn margin_ranking(x1: &Tensor, x2: &Tensor, targets: &[f32], margin: f32) -> Tensor {
    assert_eq!(
        x1.numel(),
        targets.len(),
        "{} scores but {} targets",
        x1.numel(),
        targets.len()
    );
    let sign = Tensor::from_vec(targets.to_vec(), &[targets.len(), 1]).to(x1.device());
    let difference = x1
        .reshape(&[targets.len() as i64, 1])
        .sub(&x2.reshape(&[targets.len() as i64, 1]));
    difference.mul(&sign).neg().add_scalar(margin).relu().mean()
}

/// Contrastive pair loss (Hadsell et al., 2006): `t·d² + (1−t)·max(0, m − d)²`
/// with `t = 1` for pairs that belong together.
///
/// Squared distance pulls matching pairs together; non-matching pairs are
/// pushed quadratically until they clear the margin `m`, then released.
pub fn contrastive(a: &Tensor, b: &Tensor, targets: &[f32], margin: f32) -> Tensor {
    assert_eq!(
        a.dim(0),
        targets.len(),
        "{} rows but {} targets",
        a.dim(0),
        targets.len()
    );
    let distance = row_distance(a, b);

    let together = mask(targets, |t| t > 0.0, a.device());
    let apart = mask(targets, |t| t <= 0.0, a.device());

    let pull = distance.square().mul(&together);
    let push = distance
        .neg()
        .add_scalar(margin)
        .relu()
        .square()
        .mul(&apart);
    pull.add(&push).mean()
}

/// InfoNCE, the loss behind CLIP and SimCLR-style pretraining.
///
/// Row `i` of `queries` and row `i` of `keys` are the matching pair. Both sets
/// are L2-normalized, all pairwise similarities become logits at temperature
/// `τ`, and each query must classify its own key out of the whole batch — so
/// every other row serves as a free negative, and the effective task gets
/// harder as the batch grows. Low `τ` sharpens the contrast; 0.07 is typical.
pub fn info_nce(queries: &Tensor, keys: &Tensor, temperature: f32) -> Tensor {
    assert!(
        temperature > 0.0,
        "temperature must be positive, got {temperature}"
    );
    let batch = queries.dim(0);

    let normalize = |x: &Tensor| x.div(&x.square().sum_axis_keep(1).sqrt().add_scalar(EPS));
    let logits = normalize(queries)
        .matmul_nt(&normalize(keys))
        .div_scalar(temperature);

    // The matching key for query i sits at column i: the targets are the diagonal.
    let diagonal: Vec<usize> = (0..batch).collect();
    cross_entropy(&logits, &diagonal)
}
