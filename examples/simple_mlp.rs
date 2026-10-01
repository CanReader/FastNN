//! The smallest complete training loop: an MLP learning XOR.
//!
//!     cargo run --example simple_mlp --release
//!
//! XOR is not linearly separable, so a single linear layer cannot fit it at all.
//! Two layers with a non-linearity between them can, which makes this the
//! shortest honest test that a network is really learning.

use fastnn::prelude::*;

fn main() {
    manual_seed(42);

    let inputs = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let targets = [0usize, 1, 1, 0];

    let model = Sequential::new()
        .add(Linear::new(2, 8))
        .add(Tanh)
        .add(Linear::new(8, 2));

    let mut opt = Adam::new(model.parameters(), 0.05);

    println!(
        "training a {}-parameter MLP on XOR\n",
        model.num_parameters()
    );
    for step in 1..=400 {
        let loss = cross_entropy(&model.forward(&inputs), &targets);

        opt.zero_grad();
        loss.backward();
        opt.step();

        if step % 100 == 0 {
            println!("  step {step:3}  loss {:.4}", loss.item());
        }
    }

    // Inference builds no graph, so skip it.
    let predictions = no_grad(|| model.forward(&inputs)).argmax(1);
    let values = inputs.to_vec();

    println!("\n  a  b  →  predicted  expected");
    for (row, (&predicted, &expected)) in predictions.iter().zip(&targets).enumerate() {
        let mark = if predicted == expected {
            ""
        } else {
            "  <- wrong"
        };
        println!(
            "  {}  {}  →  {predicted}          {expected}{mark}",
            values[row * 2],
            values[row * 2 + 1]
        );
    }
}
