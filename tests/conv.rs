//! Convolution-variant semantics, checked against the definitions.

use fastnn::prelude::*;
use fastnn::tensor::Window;

#[track_caller]
fn assert_close(got: f32, want: f32, what: &str) {
    assert!((got - want).abs() < 1e-4, "{what}: got {got}, want {want}");
}

/// Transposed convolution IS the adjoint of convolution: with a shared weight
/// and no bias, ⟨conv(x), y⟩ = ⟨x, convᵀ(y)⟩ for every x and y. This is the
/// defining property — if it holds for random tensors, the layer computes the
/// transpose of the same linear map, output padding, stride, and all.
#[test]
fn conv_transpose_is_the_exact_adjoint_of_conv() {
    manual_seed(12);
    let window = Window::square(3, 2, 1);
    let conv = Conv2d::with_window(2, 3, window, false);
    let transpose = ConvTranspose2d::with_window(3, 2, window, false);

    // Same matrix in both layers. Conv flattens [out, in, kh, kw] to
    // [out, in·kh·kw]; the transpose flattens its [in_t=out, out_t=in, kh, kw]
    // to the same [out, in·kh·kw] and reads it transposed in the GEMM — so the
    // raw buffers coincide element for element.
    transpose.weight.set_value(conv.weight.value());

    // 7 satisfies (size + 2p − k) % s = 0, so conv's input size is recoverable
    // exactly; a size like 6 would need output_padding to disambiguate.
    let x = Tensor::randn(&[2, 2, 7, 7]);
    let forward = no_grad(|| conv.forward(&x)); // [2, 3, 4, 4]
    let y = Tensor::randn(&[2, 3, 4, 4]);
    let backward = no_grad(|| transpose.forward(&y)); // [2, 2, 7, 7]

    let lhs = forward.mul(&y).sum().item();
    let rhs = x.mul(&backward).sum().item();
    assert!(
        (lhs - rhs).abs() / lhs.abs().max(1.0) < 1e-4,
        "adjoint identity violated: ⟨conv x, y⟩ = {lhs} but ⟨x, convᵀ y⟩ = {rhs}"
    );
}

/// The smallest checkable upsample: a 1×1 input through a 2×2 all-ones kernel
/// at stride 2 paints the value into every cell of a 2×2 output.
#[test]
fn conv_transpose_upsamples_a_pixel_by_hand() {
    let layer = ConvTranspose2d::with_window(1, 1, Window::square(2, 2, 0), false);
    layer
        .weight
        .set_value(Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]));

    let out = no_grad(|| layer.forward(&Tensor::from_vec(vec![5.0], &[1, 1, 1, 1])));
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_eq!(out.to_vec(), vec![5.0; 4]);
}

/// Depthwise convolution with 1×1 kernels reduces to a per-channel scale —
/// and the channels must not mix.
#[test]
fn depthwise_keeps_channels_separate() {
    let layer = Conv2d::depthwise(2, 1, 1, 0);
    layer
        .weight
        .set_value(Tensor::from_vec(vec![2.0, 3.0], &[2, 1, 1, 1]));
    if let Some(bias) = &layer.bias {
        bias.set_value(Tensor::zeros(&[2]));
    }

    let input = Tensor::from_vec(vec![1.0, 10.0, 100.0, 1000.0], &[1, 2, 1, 2]);
    let out = no_grad(|| layer.forward(&input)).to_vec();
    assert_eq!(
        out,
        vec![2.0, 20.0, 300.0, 3000.0],
        "channel 0 ×2, channel 1 ×3, no mixing"
    );
}

/// A dilated kernel reads every d-th input: `out[t] = a·x[t] + b·x[t+2]` for
/// kernel `[a, b]` at dilation 2, worked by hand on a length-5 sequence.
#[test]
fn dilation_stretches_the_kernel_footprint() {
    let window = Window {
        kernel: (1, 2),
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 2),
    };
    let layer = Conv2d::with_window(1, 1, window, false);
    layer
        .weight
        .set_value(Tensor::from_vec(vec![10.0, 1.0], &[1, 1, 1, 2]));

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 1, 5]);
    let out = no_grad(|| layer.forward(&x)).to_vec();
    // Footprint spans 3, so 3 outputs: 10·x[t] + x[t+2].
    assert_eq!(out, vec![13.0, 24.0, 35.0]);
}

/// Causality, tested as a property: perturbing the future must not change the
/// past. Length is preserved, and output t sees only inputs ≤ t.
#[test]
fn causal_conv1d_cannot_see_the_future() {
    manual_seed(13);
    let layer = Conv1d::causal(1, 1, 3, 2); // footprint (3−1)·2+1 = 5, all left
    let x = Tensor::randn(&[1, 1, 8]);

    let mut poked = x.to_vec();
    poked[5] += 10.0;
    let poked = Tensor::from_vec(poked, &[1, 1, 8]);

    let out_a = no_grad(|| layer.forward(&x));
    let out_b = no_grad(|| layer.forward(&poked));
    assert_eq!(out_a.shape(), &[1, 1, 8], "causal padding preserves length");

    let (a, b) = (out_a.to_vec(), out_b.to_vec());
    for t in 0..5 {
        assert_eq!(a[t], b[t], "output {t} changed when input 5 was poked");
    }
    assert!(a[5] != b[5], "output 5 should see input 5");
}

/// Grouped convolution must equal running each group's slice through its own
/// small convolution — computed here directly from the weight slices.
#[test]
fn grouped_conv_equals_independent_group_convs() {
    manual_seed(14);
    let layer = Conv2d::grouped(4, 6, 3, 1, 1, 2);
    let x = Tensor::randn(&[2, 4, 5, 5]);
    let out = no_grad(|| layer.forward(&x));

    let window = Window::square(3, 1, 1);
    for group in 0..2 {
        let slice = x.narrow(1, group * 2, 2);
        let kernels = layer
            .weight
            .value()
            .narrow(0, group * 3, 3)
            .reshape(&[3, -1]);
        let reference = no_grad(|| {
            kernels.matmul(&slice.im2col(window)).add(
                &layer
                    .bias
                    .as_ref()
                    .unwrap()
                    .value()
                    .narrow(0, group * 3, 3)
                    .reshape(&[1, 3, 1]),
            )
        });

        let got = out.narrow(1, group * 3, 3).to_vec();
        for (i, (g, r)) in got.iter().zip(reference.to_vec()).enumerate() {
            assert_close(*g, r, &format!("group {group} element {i}"));
        }
    }
}

/// Gradients reach every parameter of every new layer.
#[test]
fn gradients_reach_all_conv_variant_parameters() {
    let layers: Vec<(&str, Box<dyn Module>, Tensor)> = vec![
        (
            "conv1d",
            Box::new(Conv1d::new(2, 3, 3, 1, 1)),
            Tensor::randn(&[1, 2, 6]),
        ),
        (
            "causal conv1d",
            Box::new(Conv1d::causal(2, 3, 3, 2)),
            Tensor::randn(&[1, 2, 6]),
        ),
        (
            "grouped",
            Box::new(Conv2d::grouped(4, 4, 3, 1, 1, 2)),
            Tensor::randn(&[1, 4, 5, 5]),
        ),
        (
            "transpose2d",
            Box::new(ConvTranspose2d::new(2, 3, 3, 2, 1)),
            Tensor::randn(&[1, 2, 4, 4]),
        ),
        (
            "transpose1d",
            Box::new(ConvTranspose1d::new(2, 3, 4, 2, 1)),
            Tensor::randn(&[1, 2, 5]),
        ),
    ];

    for (name, layer, input) in layers {
        layer.forward(&input).sum().backward();
        for (param_name, param) in layer.named_parameters() {
            assert!(
                param.grad().is_some(),
                "{name}: no gradient reached {param_name}"
            );
        }
    }
}

/// `Param::set_value` on shared handles must be visible through the layer —
/// the mechanism the hand-value tests above rely on.
#[test]
fn output_padding_hits_the_requested_size() {
    let plain = ConvTranspose2d::new(1, 1, 3, 2, 1);
    let padded = ConvTranspose2d::new(1, 1, 3, 2, 1).output_padding(1);
    let x = Tensor::randn(&[1, 1, 4, 4]);

    assert_eq!(no_grad(|| plain.forward(&x)).shape(), &[1, 1, 7, 7]);
    assert_eq!(no_grad(|| padded.forward(&x)).shape(), &[1, 1, 8, 8]);
}
