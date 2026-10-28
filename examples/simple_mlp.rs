//! The smallest training loop: an MLP learning XOR.
//!
//!     cargo run --example simple_mlp --release

use fastnn::prelude::*;

const SEED: u64 = 42;
const HIDDEN: usize = 8;
const LR: f32 = 0.05;
const STEPS: usize = 400;
const REPORT_EVERY: usize = 100;

fn main() {
    manual_seed(SEED);

    let inputs = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let targets = [0usize, 1, 1, 0];

    let model = Sequential::new()
        .add(Linear::new(2, HIDDEN))
        .add(Tanh)
        .add(Linear::new(HIDDEN, 2));

    let mut opt = Adam::new(model.parameters(), LR);

    println!(
        "training a {}-parameter MLP on XOR\n",
        model.num_parameters()
    );
    for step in 1..=STEPS {
        let loss = cross_entropy(&model.forward(&inputs), &targets);

        opt.zero_grad();
        loss.backward();
        opt.step();

        if step % REPORT_EVERY == 0 {
            println!("  step {step:3}  loss {:.4}", loss.item());
        }
    }

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
