//! Finite-difference gradient checks.
//!
//! Every backward rule is checked against central differences of its own
//! forward. This is the safety net for the whole autograd layer: a rule with the
//! wrong sign, a missing factor, or a forgotten broadcast reduction fails here
//! rather than showing up later as a model that mysteriously will not converge.
//!
//! **Any new `Backward` rule needs a test in this file.**
//!
//!     cargo test --no-default-features --test gradcheck

// Test inputs are always passed as a slice. `&[x.clone()]` for a single input
// costs one clone of a refcounted handle and keeps every call site reading the
// same way as the multi-input ones.
#![allow(clippy::cloned_ref_to_slice_refs)]

use fastnn::nn::Param;
use fastnn::prelude::*;

/// Step size for the numeric derivative.
///
/// `f32` carries about seven digits and central differences spend roughly half
/// of them. 1e-3 sits at the bottom of that trade-off: smaller drowns in
/// rounding, larger starts measuring curvature instead of slope.
const STEP: f32 = 1e-3;
const TOLERANCE: f32 = 2e-2;

/// Check that `f`'s analytic gradients match its numeric ones at `values`.
///
/// `f` must produce a scalar — compose with `.sum()` if it does not.
#[track_caller]
fn gradcheck(name: &str, values: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor) {
    let params: Vec<Param> = values.iter().map(|v| Param::new(v.clone())).collect();

    let loss = f(&params.iter().map(|p| p.tensor()).collect::<Vec<_>>());
    assert_eq!(
        loss.numel(),
        1,
        "{name}: gradcheck needs a scalar, got {:?}",
        loss.shape()
    );
    loss.backward();

    for (index, param) in params.iter().enumerate() {
        let analytic = param
            .grad()
            .unwrap_or_else(|| panic!("{name}: input {index} received no gradient"))
            .to_vec();
        let original = param.value().to_vec();
        let shape = param.shape();

        for slot in 0..original.len() {
            let at = |offset: f32| {
                let mut perturbed = original.clone();
                perturbed[slot] += offset;
                param.set_value(Tensor::from_vec(perturbed, &shape));
                no_grad(|| f(&params.iter().map(|p| p.value()).collect::<Vec<_>>())).item()
            };

            let numeric = (at(STEP) - at(-STEP)) / (2.0 * STEP);
            param.set_value(Tensor::from_vec(original.clone(), &shape));

            let error = (numeric - analytic[slot]).abs();
            let scale = 1.0f32.max(numeric.abs()).max(analytic[slot].abs());
            assert!(
                error / scale < TOLERANCE,
                "{name}: input {index}, element {slot}: analytic {} vs numeric {numeric}",
                analytic[slot]
            );
        }
    }
}

/// Deterministic, well-spread values in roughly `(-0.9, 0.9)`.
///
/// Bounded on purpose. Central differences subtract two nearby loss values, so
/// if the loss can grow into the thousands the difference loses most of `f32`'s
/// digits to cancellation and the check fails on its own arithmetic. The golden
/// ratio spreads the values without repeats, and none land on 0 or ±1, where a
/// wrong sign or exponent could still give the right answer.
fn spread(n: usize, low: f32, high: f32) -> Vec<f32> {
    const GOLDEN: f32 = 0.618_034;
    (0..n)
        .map(|i| low + (high - low) * ((i as f32 + 1.0) * GOLDEN).fract())
        .collect()
}

fn sample(shape: &[usize]) -> Tensor {
    Tensor::from_vec(spread(shape.iter().product(), -0.9, 0.9), shape)
}

/// Strictly positive values, for ops undefined at or below zero.
fn positive(shape: &[usize]) -> Tensor {
    Tensor::from_vec(spread(shape.iter().product(), 0.3, 1.4), shape)
}

// ── Arithmetic ───────────────────────────────────────────────────────────────

#[test]
fn arithmetic() {
    let a = sample(&[2, 3]);

    gradcheck("add", &[a.clone(), sample(&[2, 3])], |x| {
        x[0].add(&x[1]).sum()
    });
    gradcheck("sub", &[a.clone(), sample(&[2, 3])], |x| {
        x[0].sub(&x[1]).sum()
    });
    gradcheck("mul", &[a.clone(), sample(&[2, 3])], |x| {
        x[0].mul(&x[1]).sum()
    });
    gradcheck("div", &[a.clone(), positive(&[2, 3])], |x| {
        x[0].div(&x[1]).sum()
    });
    gradcheck("mul_scalar", &[a.clone()], |x| x[0].mul_scalar(-2.5).sum());
    gradcheck("add_scalar", &[a], |x| x[0].add_scalar(3.0).sum());
}

#[test]
fn broadcasting_reduces_gradients() {
    // The [1, 3] operand is stretched over 4 rows, so its gradient must be the
    // sum of all four contributions — the classic bias-gradient case.
    gradcheck("broadcast add", &[sample(&[4, 3]), sample(&[1, 3])], |x| {
        x[0].add(&x[1]).sum()
    });
    gradcheck("broadcast mul", &[sample(&[4, 3]), sample(&[1, 3])], |x| {
        x[0].mul(&x[1]).sum()
    });
    gradcheck("broadcast rank", &[sample(&[2, 3]), sample(&[3])], |x| {
        x[0].mul(&x[1]).sum()
    });
}

// ── Element-wise maths ───────────────────────────────────────────────────────

#[test]
fn unary_maths() {
    let x = sample(&[2, 3]);

    gradcheck("neg", &[x.clone()], |x| x[0].neg().sum());
    gradcheck("exp", &[x.clone()], |x| x[0].exp().sum());
    gradcheck("square", &[x.clone()], |x| x[0].square().sum());
    gradcheck("log", &[positive(&[2, 3])], |x| x[0].log().sum());
    gradcheck("sqrt", &[positive(&[2, 3])], |x| x[0].sqrt().sum());
    gradcheck("powf", &[positive(&[2, 3])], |x| x[0].powf(2.5).sum());
    gradcheck("abs", &[x.clone()], |x| x[0].abs().sum());
    // Bounds sit clear of every sample value: at a clamp boundary the true
    // derivative jumps, and central differences would straddle both sides.
    gradcheck("clamp", &[x], |x| x[0].clamp(-9.0, 9.0).sum());
}

// ── Activations ──────────────────────────────────────────────────────────────

#[test]
fn activations() {
    let x = sample(&[3, 4]);

    gradcheck("relu", &[x.clone()], |x| x[0].relu().sum());
    gradcheck("sigmoid", &[x.clone()], |x| x[0].sigmoid().sum());
    gradcheck("tanh", &[x.clone()], |x| x[0].tanh().sum());
    gradcheck("gelu", &[x.clone()], |x| x[0].gelu().sum());
    gradcheck("silu", &[x.clone()], |x| x[0].silu().sum());
    gradcheck("leaky_relu", &[x], |x| x[0].leaky_relu(0.1).sum());
}

#[test]
fn softmax_family() {
    let x = sample(&[3, 4]);
    let weights = sample(&[3, 4]);

    // Weighted, not a plain sum: softmax rows total 1 whatever the input, so an
    // unweighted sum has zero gradient and would pass against any rule at all.
    let w = weights.clone();
    gradcheck("softmax", &[x.clone()], move |x| {
        x[0].softmax().mul(&w).sum()
    });
    let w = weights;
    gradcheck("log_softmax", &[x], move |x| {
        x[0].log_softmax().mul(&w).sum()
    });
}

// ── Matrix multiplication ────────────────────────────────────────────────────

#[test]
fn matmul_layouts() {
    gradcheck("matmul", &[sample(&[2, 3]), sample(&[3, 4])], |x| {
        x[0].matmul(&x[1]).sum()
    });
    gradcheck("matmul_nt", &[sample(&[2, 3]), sample(&[4, 3])], |x| {
        x[0].matmul_nt(&x[1]).sum()
    });
    gradcheck("matmul_tn", &[sample(&[3, 2]), sample(&[3, 4])], |x| {
        x[0].matmul_tn(&x[1]).sum()
    });
}

#[test]
fn matmul_batched() {
    gradcheck("batched", &[sample(&[2, 3, 4]), sample(&[2, 4, 2])], |x| {
        x[0].matmul(&x[1]).sum()
    });
    // A weight shared across the batch: its gradient must sum over both items.
    gradcheck(
        "shared weight",
        &[sample(&[3, 4]), sample(&[2, 4, 2])],
        |x| x[0].matmul(&x[1]).sum(),
    );
}

// ── Reductions ───────────────────────────────────────────────────────────────

#[test]
fn reductions() {
    let x = sample(&[3, 4]);

    gradcheck("sum", &[x.clone()], |x| x[0].sum());
    gradcheck("mean", &[x.clone()], |x| x[0].mean());
    gradcheck("sum_axis 0", &[x.clone()], |x| x[0].sum_axis(0).sum());

    let w = sample(&[3]);
    gradcheck("sum_axis 1", &[x.clone()], move |x| {
        x[0].sum_axis(1).mul(&w).sum()
    });
    let w = sample(&[3]);
    gradcheck("mean_axis 1", &[x], move |x| {
        x[0].mean_axis(1).mul(&w).sum()
    });
}

// ── Views ────────────────────────────────────────────────────────────────────

#[test]
fn views() {
    let x = sample(&[2, 3, 4]);

    let w = sample(&[24]);
    gradcheck("reshape", &[x.clone()], move |x| {
        x[0].reshape(&[-1]).mul(&w).sum()
    });

    // Weighted so the reordering matters: a plain sum is invariant to any
    // permutation and would pass even if the inverse were computed wrongly.
    let w = sample(&[4, 3, 2]);
    gradcheck("permute", &[x.clone()], move |x| {
        x[0].permute(&[2, 1, 0]).mul(&w).sum()
    });

    let w = sample(&[2, 4, 3]);
    gradcheck("transpose", &[x.clone()], move |x| {
        x[0].transpose().mul(&w).sum()
    });

    let w = sample(&[4, 3]);
    gradcheck("expand", &[sample(&[1, 3])], move |x| {
        x[0].expand(&[4, 3]).mul(&w).sum()
    });

    gradcheck("squeeze/unsqueeze", &[x], |x| {
        x[0].unsqueeze(1).squeeze(1).sum()
    });
}

#[test]
fn narrow_and_join() {
    // Only part of the input reaches the loss; the rest must get exactly zero.
    let w = sample(&[2, 3]);
    gradcheck("narrow", &[sample(&[2, 6])], move |x| {
        x[0].narrow(1, 2, 3).mul(&w).sum()
    });

    let w = sample(&[2, 9]);
    gradcheck("cat", &[sample(&[2, 4]), sample(&[2, 5])], move |x| {
        Tensor::cat(&[&x[0], &x[1]], 1).mul(&w).sum()
    });

    let w = sample(&[2, 2, 3]);
    gradcheck("stack", &[sample(&[2, 3]), sample(&[2, 3])], move |x| {
        Tensor::stack(&[&x[0], &x[1]], 1).mul(&w).sum()
    });
}

// ── Indexing ─────────────────────────────────────────────────────────────────

#[test]
fn index_select_accumulates_repeats() {
    // Row 1 is picked twice, so its gradient must be the sum of both lookups.
    let w = sample(&[4, 3]);
    gradcheck("index_select", &[sample(&[5, 3])], move |x| {
        x[0].index_select(&[0, 1, 1, 4]).mul(&w).sum()
    });
}

// ── Structured ops ───────────────────────────────────────────────────────────

#[test]
fn convolution() {
    // [1, 1, 4, 4] with a 2x2 stride-2 window -> [1, 1*2*2 patch, 2*2 positions].
    let w = sample(&[1, 4, 4]);
    gradcheck("im2col", &[sample(&[1, 1, 4, 4])], move |x| {
        x[0].im2col(Window::square(2, 2, 0)).mul(&w).sum()
    });

    // The whole layer end to end: im2col and matmul must compose correctly, and
    // overlapping windows must accumulate into the same input pixels.
    gradcheck(
        "conv2d",
        &[sample(&[2, 2, 4, 4]), sample(&[3, 2, 3, 3]), sample(&[3])],
        |x| {
            let columns = x[0].im2col(Window::square(3, 1, 1));
            let out = x[1].reshape(&[3, -1]).matmul(&columns);
            out.add(&x[2].reshape(&[1, 3, 1])).sum()
        },
    );
}

#[test]
fn pooling() {
    let w = sample(&[1, 2, 2, 2]);
    gradcheck("max_pool2d", &[sample(&[1, 2, 4, 4])], move |x| {
        x[0].max_pool2d(Window::square(2, 2, 0)).mul(&w).sum()
    });

    let w = sample(&[1, 2, 2, 2]);
    gradcheck("avg_pool2d", &[sample(&[1, 2, 4, 4])], move |x| {
        x[0].avg_pool2d(Window::square(2, 2, 0)).mul(&w).sum()
    });

    // Uneven spans: 5 does not divide by 2, so the windows differ in size.
    let w = sample(&[1, 2, 2, 2]);
    gradcheck("adaptive_avg_pool2d", &[sample(&[1, 2, 5, 5])], move |x| {
        x[0].adaptive_avg_pool2d((2, 2)).mul(&w).sum()
    });
}

#[test]
fn normalization() {
    let w = sample(&[3, 4]);
    gradcheck(
        "layer_norm",
        &[sample(&[3, 4]), Tensor::ones(&[4]), Tensor::zeros(&[4])],
        move |x| x[0].layer_norm(&x[1], &x[2], 1e-5).mul(&w).sum(),
    );

    let w = sample(&[2, 2, 2, 2]);
    gradcheck(
        "batch_norm2d",
        &[
            sample(&[2, 2, 2, 2]),
            Tensor::ones(&[2]),
            Tensor::zeros(&[2]),
        ],
        move |x| x[0].batch_norm2d(&x[1], &x[2], 1e-5).0.mul(&w).sum(),
    );

    let w = sample(&[3, 4]);
    gradcheck("rms_norm", &[sample(&[3, 4])], move |x| {
        let scale = x[0].square().mean_axis_keep(1).add_scalar(1e-6).sqrt();
        x[0].div(&scale).mul(&w).sum()
    });
}

// ── Losses ───────────────────────────────────────────────────────────────────

#[test]
fn losses() {
    gradcheck("cross_entropy", &[sample(&[3, 4])], |x| {
        cross_entropy(&x[0], &[0, 2, 3])
    });

    let target = sample(&[3, 4]);
    let t = target.clone();
    gradcheck("mse", &[sample(&[3, 4])], move |x| mse(&x[0], &t));
    let t = target;
    gradcheck("mae", &[sample(&[3, 4])], move |x| mae(&x[0], &t));

    let labels = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[2, 3]);
    let t = labels.clone();
    gradcheck(
        "bce",
        &[Tensor::full(&[2, 3], 0.5).add(&sample(&[2, 3]).mul_scalar(0.1))],
        move |x| bce(&x[0], &t),
    );
    let t = labels;
    gradcheck("bce_with_logits", &[sample(&[2, 3])], move |x| {
        bce_with_logits(&x[0], &t)
    });
}

// ── Whole layers ─────────────────────────────────────────────────────────────

#[test]
fn a_full_model_trains() {
    manual_seed(7);
    let model = Sequential::new()
        .add(Linear::new(4, 6))
        .add(GELU)
        .add(LayerNorm::new(6))
        .add(Linear::new(6, 3));

    let inputs = sample(&[5, 4]);
    let targets = [0usize, 1, 2, 1, 0];
    let mut opt = Adam::new(model.parameters(), 0.05);

    let before = cross_entropy(&model.forward(&inputs), &targets).item();
    for _ in 0..80 {
        let loss = cross_entropy(&model.forward(&inputs), &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let after = cross_entropy(&model.forward(&inputs), &targets).item();

    assert!(
        after < before * 0.5,
        "loss went {before} -> {after}, expected a clear drop"
    );
}

#[test]
fn attention_gradients_reach_every_projection() {
    let attention = MultiHeadAttention::new(8, 2, 0.0);
    let x = sample(&[2, 3, 8]);

    attention.attend(&x, &x, &x, true).sum().backward();

    for (name, param) in attention.named_parameters() {
        assert!(param.grad().is_some(), "no gradient reached {name}");
    }
}

#[test]
fn recurrent_layers_are_differentiable() {
    let x = sample(&[2, 4, 3]);
    let cells: Vec<(&str, Box<dyn Module>)> = vec![
        ("LSTM", Box::new(LSTM::new(3, 5))),
        ("GRU", Box::new(GRU::new(3, 5))),
    ];

    for (name, cell) in cells {
        cell.forward(&x).sum().backward();
        for (param, handle) in cell.named_parameters() {
            assert!(
                handle.grad().is_some(),
                "{name}: no gradient reached {param}"
            );
        }
    }
}

#[test]
fn gradients_accumulate_until_zeroed() {
    let param = Param::new(sample(&[2, 2]));

    param.tensor().sum().backward();
    let once = param.grad().unwrap().to_vec();

    param.tensor().sum().backward();
    let twice = param.grad().unwrap().to_vec();
    assert_eq!(twice, once.iter().map(|v| v * 2.0).collect::<Vec<_>>());

    param.zero_grad();
    assert!(param.grad().is_none());
}

#[test]
fn a_parameter_used_twice_sums_both_paths() {
    // Weight tying: the same handle appears in two places, so its gradient must
    // be the sum of both. This works because both uses share one GradSlot.
    let shared = Param::new(sample(&[2, 2]));
    let x = sample(&[2, 2]);

    let out = x.matmul(&shared.tensor()).add(&shared.tensor());
    out.sum().backward();
    let both = shared.grad().unwrap().to_vec();

    shared.zero_grad();
    x.matmul(&shared.tensor()).sum().backward();
    let first = shared.grad().unwrap().to_vec();

    shared.zero_grad();
    shared.tensor().sum().backward();
    let second = shared.grad().unwrap().to_vec();

    for i in 0..both.len() {
        assert!((both[i] - (first[i] + second[i])).abs() < 1e-5);
    }
}

#[test]
fn no_grad_builds_no_graph() {
    let param = Param::new(sample(&[2, 2]));
    let out = no_grad(|| param.tensor().mul_scalar(3.0).sum());

    assert!(out.grad_fn().is_none());
    out.backward();
    assert!(
        param.grad().is_none(),
        "no_grad should leave nothing to differentiate"
    );
}

#[test]
fn decoder_gradients_reach_every_projection() {
    let transformer = Transformer::new(8, 2, 16, 1, 1, 0.0);
    let source = sample(&[2, 3, 8]);
    let target = sample(&[2, 4, 8]);

    transformer.run(&source, &target).sum().backward();

    for (name, param) in transformer.named_parameters() {
        assert!(param.grad().is_some(), "no gradient reached {name}");
    }
}

#[test]
fn masked_attention_matches_finite_differences() {
    let attention = MultiHeadAttention::new(4, 2, 0.0);
    let mask = Tensor::from_vec(vec![1.0, 1.0, 0.0], &[1, 3]);
    let weight = sample(&[1, 3, 4]);

    gradcheck("attend_masked", &[sample(&[1, 3, 4])], |v| {
        attention
            .attend_masked(&v[0], &v[0], &v[0], false, Some(&mask))
            .mul(&weight)
            .sum()
    });
}

/// The configurable cross-entropy has its own fused backward; check it under
/// every option at once — smoothing, class weights, and an ignored row.
#[test]
fn weighted_cross_entropy_matches_finite_differences() {
    let loss_fn = CrossEntropyLoss::new()
        .label_smoothing(0.1)
        .class_weights(vec![1.0, 2.0, 0.5, 1.5])
        .ignore_index(2);

    gradcheck("cross_entropy_loss", &[sample(&[3, 4])], |v| {
        loss_fn.compute(&v[0], &[1, 2, 3])
    });
}

/// The per-row reduction takes a `[batch]` upstream gradient — the other
/// branch of the backward rule, checked with a weighted combination.
#[test]
fn per_row_cross_entropy_matches_finite_differences() {
    let weight = Tensor::from_vec(vec![0.7, -0.3, 1.2], &[3]);
    let loss_fn = CrossEntropyLoss::new().reduction(Reduction::None);

    gradcheck("cross_entropy_rows", &[sample(&[3, 4])], |v| {
        loss_fn.compute(&v[0], &[0, 3, 1]).mul(&weight).sum()
    });
}

/// col2im is a new forward op; check it (and its Im2ColBackward) directly,
/// weighted so overlap accumulation cannot hide behind a symmetric sum.
#[test]
fn col2im_matches_finite_differences() {
    use fastnn::tensor::Window;
    let window = Window::square(2, 2, 0);
    // Columns for a 4×4 target image: patch = 1·2·2, positions = 2·2.
    let weight = sample(&[1, 1, 4, 4]);

    gradcheck("col2im", &[sample(&[1, 4, 4])], |v| {
        v[0].col2im(window, (4, 4)).mul(&weight).sum()
    });
}

/// Dilation changes the index map; the finite-difference check proves the
/// backward folds along the same stretched footprint the forward reads.
#[test]
fn dilated_im2col_matches_finite_differences() {
    use fastnn::tensor::Window;
    // Span (2−1)·2+1 = 3 on a padded 5×5 input → 5×5 = 25 output positions.
    let window = Window::square(2, 1, 1).dilated(2);
    let weight = sample(&[1, 4, 25]);

    gradcheck("dilated im2col", &[sample(&[1, 1, 5, 5])], |v| {
        v[0].im2col(window).mul(&weight).sum()
    });
}
