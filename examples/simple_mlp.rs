//! Simple MLP example — XOR problem.
//!
//! Demonstrates the full training loop: autograd, backward, optimizer step.

use fastnn::prelude::*;
use fastnn::autograd::graph;

fn main() {
    println!("FastNN — Simple MLP (XOR Problem)");
    println!("==================================\n");

    manual_seed(42);

    // XOR dataset: 4 rows of (input, target).
    let inputs = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
    );
    let targets = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // Build model.
    let mut model = Sequential::new()
        .add(Linear::new(2, 16))
        .add(ReLU)
        .add(Linear::new(16, 16))
        .add(ReLU)
        .add(Linear::new(16, 1))
        .add(Sigmoid);

    println!("Model parameters: {}", model.num_parameters());

    let loss_fn = MSELoss::new();
    let mut optimizer = SGD::new(0.5).momentum(0.9);
    let epochs = 2000;

    for epoch in 0..epochs {
        // Enable gradient tracking for this forward pass.
        graph::enable_grad();

        // Clear any leftover grads from prior iteration.
        {
            let mut params = model.parameters_mut();
            optimizer.zero_grad(&mut params);
        }

        // Forward.
        let output = model.forward(&inputs);
        let loss = loss_fn.forward(&output, &targets);

        // Backward — populates .grad() on every leaf tensor with requires_grad.
        loss.backward();

        // Optimizer step — reads .grad() and mutates params in place.
        {
            let mut params = model.parameters_mut();
            optimizer.step(&mut params);
        }

        // Disable grad tracking outside the training step.
        graph::disable_grad();

        if epoch % 100 == 0 || epoch == epochs - 1 {
            println!("Epoch {:4} | Loss: {:.6}", epoch, loss.item());
        }
    }

    // Final predictions (inference — grads disabled).
    let output = model.forward(&inputs);
    println!("\nFinal predictions:");
    let preds = output.to_vec();
    let input_data = inputs.to_vec();
    let target_data = targets.to_vec();
    for i in 0..4 {
        println!(
            "  [{:.0}, {:.0}] -> {:.4} (target: {:.0}, rounded: {})",
            input_data[i * 2], input_data[i * 2 + 1],
            preds[i],
            target_data[i],
            if preds[i] > 0.5 { 1 } else { 0 },
        );
    }
}
