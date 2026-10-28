//! Losses on embedding geometry (pairs and triples), built from ordinary ops.

use crate::tensor::Tensor;

use super::loss::cross_entropy;

/// The derivative of a norm is undefined at zero, so keep sqrt and division off it.
const EPS: f32 = 1e-8;

/// Row-wise Euclidean distance, `[batch, dim]` to `[batch, 1]`.
fn row_distance(a: &Tensor, b: &Tensor) -> Tensor {
    a.sub(b).square().sum_axis_keep(1).add_scalar(EPS).sqrt()
}

/// Row-wise cosine similarity, `[batch, dim]` to `[batch, 1]`.
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

/// Cosine embedding loss: `1 − cos` when `target = +1`, `max(0, cos − margin)`
/// when `target = −1`.
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

/// Margin ranking loss: `max(0, −t·(x₁ − x₂) + margin)`, `t = ±1` picks the higher input.
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

/// Contrastive pair loss: `t·d² + (1−t)·max(0, m − d)²`, `t = 1` for matching pairs.
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

/// InfoNCE. Row `i` of `queries` matches row `i` of `keys`, every other row is a negative.
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

    // query i's match is column i
    let diagonal: Vec<usize> = (0..batch).collect();
    cross_entropy(&logits, &diagonal)
}
