//! End-to-end convergence: every major layer family, trained until it works.
//!
//! `gradcheck` proves each derivative in isolation; these prove the pieces
//! compose into models that actually learn. Each case is deliberately tiny so
//! the whole file runs in seconds even in a debug build.

use fastnn::prelude::*;

#[test]
fn mlp_converges_on_xor() {
    manual_seed(3);
    let inputs = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let targets = [0usize, 1, 1, 0];

    let model = Sequential::new()
        .add(Linear::new(2, 8))
        .add(Tanh)
        .add(Linear::new(8, 2));
    let mut opt = Adam::new(model.parameters(), 5e-2);

    let mut loss_value = f32::MAX;
    for _ in 0..1000 {
        let loss = cross_entropy(&model.forward(&inputs), &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
        loss_value = loss.item();
        if loss_value < 0.01 {
            break;
        }
    }
    assert!(
        loss_value < 0.01,
        "xor did not converge: final loss {loss_value}"
    );
}

#[test]
fn linear_regression_recovers_slope_and_intercept() {
    manual_seed(4);
    // y = 2x + 1, no noise: the model should recover both numbers nearly exactly.
    let xs: Vec<f32> = (0..32).map(|i| i as f32 / 16.0 - 1.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
    let inputs = Tensor::from_vec(xs, &[32, 1]);
    let targets = Tensor::from_vec(ys, &[32, 1]);

    let model = Linear::new(1, 1);
    let mut opt = SGD::new(model.parameters(), 0.5);

    for _ in 0..500 {
        let loss = mse(&model.forward(&inputs), &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let params = model.parameters();
    let (weight, bias) = (params[0].value().to_vec()[0], params[1].value().to_vec()[0]);
    assert!((weight - 2.0).abs() < 0.01, "slope {weight}, wanted 2");
    assert!((bias - 1.0).abs() < 0.01, "intercept {bias}, wanted 1");
}

#[test]
fn cnn_overfits_a_small_batch() {
    manual_seed(5);
    let inputs = Tensor::randn(&[10, 1, 8, 8]);
    let targets: Vec<usize> = (0..10).collect();

    let model = Sequential::new()
        .add(Conv2d::new(1, 4, 3, 1, 1))
        .add(ReLU)
        .add(MaxPool2d::new(2))
        .add(Flatten::new())
        .add(Linear::new(4 * 4 * 4, 10));
    let mut opt = Adam::new(model.parameters(), 1e-2);

    let mut loss_value = f32::MAX;
    for _ in 0..200 {
        let loss = cross_entropy(&model.forward(&inputs), &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
        loss_value = loss.item();
        if loss_value < 0.05 {
            break;
        }
    }
    assert!(
        loss_value < 0.05,
        "cnn did not overfit ten samples: loss {loss_value}"
    );
}

#[test]
fn lstm_memorizes_a_short_sequence() {
    manual_seed(6);
    // One fixed input sequence, one fixed target per step: pure memorization.
    let inputs = Tensor::randn(&[1, 6, 4]);
    let targets = [2usize, 0, 3, 1, 2, 0];

    let lstm = LSTM::new(4, 24);
    let head = Linear::new(24, 4);
    let mut params = lstm.parameters();
    params.extend(head.parameters());
    let mut opt = Adam::new(params, 1e-2);

    let mut loss_value = f32::MAX;
    for _ in 0..300 {
        let hidden = lstm.forward(&inputs);
        let logits = head.forward(&hidden).reshape(&[6, -1]);
        let loss = cross_entropy(&logits, &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
        loss_value = loss.item();
        if loss_value < 0.05 {
            break;
        }
    }
    assert!(
        loss_value < 0.05,
        "lstm did not memorize: loss {loss_value}"
    );
}

#[test]
fn transformer_overfits_a_small_batch() {
    manual_seed(7);
    let source = Tensor::randn(&[2, 3, 16]);
    let target_in = Tensor::randn(&[2, 4, 16]);
    let wanted = Tensor::randn(&[2, 4, 16]);

    // The full encoder-decoder path, driven to overfit a fixed pair.
    let model = Transformer::new(16, 2, 32, 1, 1, 0.0);
    let mut opt = Adam::new(model.parameters(), 3e-3);

    let mut loss_value = f32::MAX;
    for _ in 0..400 {
        let loss = mse(&model.run(&source, &target_in), &wanted);
        opt.zero_grad();
        loss.backward();
        opt.step();
        loss_value = loss.item();
        if loss_value < 0.05 {
            break;
        }
    }
    assert!(
        loss_value < 0.05,
        "transformer did not overfit: loss {loss_value}"
    );
}

#[test]
fn saved_model_predicts_identically_after_reload() {
    manual_seed(8);
    let inputs = Tensor::randn(&[4, 6]);
    let model = Sequential::new()
        .add(Linear::new(6, 12))
        .add(ReLU)
        .add(Linear::new(12, 3));

    // A few steps first, so the weights are not just their initialization.
    let targets = Tensor::zeros(&[4, 3]);
    let mut opt = SGD::new(model.parameters(), 0.01);
    for _ in 0..5 {
        let loss = mse(&model.forward(&inputs), &targets);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let before = no_grad(|| model.forward(&inputs)).to_vec();
    assert!(
        before.iter().all(|v| v.is_finite()),
        "training diverged before the roundtrip"
    );

    let path = std::env::temp_dir().join("fastnn_roundtrip_test.fdl");
    save(&model, &path).unwrap();
    let reloaded = Sequential::new()
        .add(Linear::new(6, 12))
        .add(ReLU)
        .add(Linear::new(12, 3));
    load(&reloaded, &path).unwrap();
    std::fs::remove_file(&path).ok();

    let after = no_grad(|| reloaded.forward(&inputs)).to_vec();
    assert_eq!(before, after, "reloaded model predicts differently");
}

#[test]
fn clip_grad_value_bounds_every_gradient_element() {
    let param = fastnn::nn::Param::new(Tensor::from_vec(vec![1.0, -2.0], &[2]));
    param.tensor().mul_scalar(100.0).sum().backward();

    clip_grad_value(std::slice::from_ref(&param), 1.5);
    let grad = param.grad().unwrap().to_vec();
    assert!(
        grad.iter().all(|g| g.abs() <= 1.5),
        "gradient escaped the clamp: {grad:?}"
    );
}

/// Every optimizer, one benchmark: recover a linear map by least squares.
/// Learning rates are per-optimizer (a shared one would test tuning, not
/// correctness); each must reach near-zero loss on the same data.
#[test]
fn every_optimizer_minimizes_least_squares() {
    use fastnn::nn::Param;
    type Recipe = fn(Vec<Param>) -> Box<dyn Optimizer>;

    let recipes: Vec<(&str, Recipe)> = vec![
        ("sgd", |p| Box::new(SGD::new(p, 0.1))),
        ("adam", |p| Box::new(Adam::new(p, 0.05))),
        ("adamw", |p| Box::new(AdamW::new(p, 0.05))),
        ("rmsprop", |p| Box::new(RMSprop::new(p, 0.01))),
        ("rmsprop-centered", |p| {
            Box::new(RMSprop::new(p, 0.01).momentum(0.9).centered())
        }),
        ("adagrad", |p| Box::new(Adagrad::new(p, 0.5))),
        ("adadelta", |p| Box::new(Adadelta::new(p))),
        ("radam", |p| Box::new(RAdam::new(p, 0.05))),
        ("lion", |p| Box::new(Lion::new(p, 0.005))),
        ("lookahead-adam", |p| {
            Box::new(Lookahead::new(Adam::new(p, 0.05), 5, 0.5))
        }),
    ];

    for (name, make) in recipes {
        manual_seed(9);
        let inputs = Tensor::randn(&[64, 3]);
        let true_map = Tensor::from_vec(vec![1.0, -2.0, 0.5], &[3, 1]);
        let targets = inputs.matmul(&true_map);

        let model = Linear::new(3, 1);
        let mut opt = make(model.parameters());

        let mut loss_value = f32::MAX;
        for _ in 0..1500 {
            let loss = mse(&model.forward(&inputs), &targets);
            opt.zero_grad();
            loss.backward();
            opt.step();
            loss_value = loss.item();
            if loss_value < 1e-3 {
                break;
            }
        }
        assert!(
            loss_value < 1e-3,
            "{name} failed to converge: loss {loss_value}"
        );
    }
}
