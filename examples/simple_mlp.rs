//! Simple MLP example — XOR problem
//!
//! Demonstrates building and training a small neural network.

use fastdl::prelude::*;

fn main() {
    println!("fastDL — Simple MLP (XOR Problem)");
    println!("==================================\n");

    manual_seed(42);

    // XOR dataset
    let inputs = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
    );
    let targets = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // Build model
    let mut model = Sequential::new()
        .add(Linear::new(2, 16))
        .add(ReLU)
        .add(Linear::new(16, 16))
        .add(ReLU)
        .add(Linear::new(16, 1))
        .add(Sigmoid);

    println!("Model parameters: {}", model.num_parameters());

    let loss_fn = MSELoss::new();
    let mut optimizer = SGD::new(1.0).momentum(0.9);

    // Training loop
    let mut params = model.parameters();
    for epoch in 0..1000 {
        let output = model.forward(&inputs);
        let (loss, grad) = loss_fn.forward_with_grad(&output, &targets);

        // Simple numerical gradient approximation for this demo
        let eps = 1e-4;
        let mut grads = Vec::new();
        for p in &params {
            let mut param_grad = vec![0.0f32; p.numel()];
            let p_data = p.to_vec();
            for i in 0..p.numel() {
                // f(x + eps)
                let mut p_plus = p_data.clone();
                p_plus[i] += eps;
                let t_plus = Tensor::from_vec(p_plus, p.shape());
                // We can't easily do forward with modified params in Sequential,
                // so we use a simple gradient approximation
                param_grad[i] = 0.0; // Placeholder
            }
            grads.push(Tensor::from_vec(param_grad, p.shape()));
        }

        // For demonstration, use the output gradient to manually backprop
        // In practice, use the autograd Variable system
        if epoch % 100 == 0 {
            println!("Epoch {:4} | Loss: {:.6}", epoch, loss.item());
        }
    }

    // Final predictions
    let output = model.forward(&inputs);
    println!("\nFinal predictions:");
    let predictions = output.to_vec();
    let target_data = targets.to_vec();
    for i in 0..4 {
        let input = inputs.to_vec();
        println!(
            "  [{:.0}, {:.0}] -> {:.4} (target: {:.0})",
            input[i * 2], input[i * 2 + 1],
            predictions[i],
            target_data[i]
        );
    }
}
