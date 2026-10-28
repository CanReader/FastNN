//! Linear initialization must consume only the random samples its parameters need.

use fastnn::prelude::*;

#[test]
fn no_bias_uses_only_weight_initialization_draws() {
    for seed in [0, 7, 42] {
        manual_seed(seed);
        let layer = Linear::no_bias(2, 3);
        let next = Tensor::rand(&[4]).to_vec();

        manual_seed(seed);
        let weight = Tensor::kaiming_uniform(&[3, 2], 2);
        let expected_next = Tensor::rand(&[4]).to_vec();

        assert!(layer.bias.is_none());
        assert_eq!(layer.weight.value().to_vec(), weight.to_vec());
        assert_eq!(
            next, expected_next,
            "no_bias consumed samples beyond its weight for seed {seed}"
        );
    }
}

#[test]
fn new_preserves_weight_then_bias_initialization() {
    for seed in [0, 7, 42] {
        manual_seed(seed);
        let layer = Linear::new(2, 3);
        let next = Tensor::rand(&[4]).to_vec();

        manual_seed(seed);
        let weight = Tensor::kaiming_uniform(&[3, 2], 2);
        let bound = 1.0 / 2.0_f32.sqrt();
        let bias = Tensor::uniform(&[3], -bound, bound);
        let expected_next = Tensor::rand(&[4]).to_vec();

        assert_eq!(layer.weight.value().to_vec(), weight.to_vec());
        assert_eq!(layer.bias.as_ref().unwrap().value().to_vec(), bias.to_vec());
        assert_eq!(next, expected_next);
    }
}

#[test]
fn no_bias_preserves_forward_shape_and_trainable_weight() {
    let layer = Linear::no_bias(2, 3);
    let input = Tensor::from_vec(vec![1.0, 2.0, -3.0, 4.0], &[2, 1, 2]);

    let output = layer.forward(&input);
    let expected = input
        .reshape(&[2, 2])
        .matmul_nt(&layer.weight.value())
        .reshape(&[2, 1, 3]);

    assert_eq!(layer.in_features(), 2);
    assert_eq!(layer.out_features(), 3);
    assert_eq!(output.shape(), &[2, 1, 3]);
    assert_eq!(output.to_vec(), expected.to_vec());
    assert_eq!(layer.named_parameters()[0].0, "weight");
    assert_eq!(layer.parameters().len(), 1);
    assert_eq!(layer.trainable_parameters().len(), 1);

    output.sum().backward();
    assert_eq!(layer.weight.grad().unwrap().shape(), &[3, 2]);
}
