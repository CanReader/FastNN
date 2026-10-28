//! Multi-task learning: a shared trunk with two heads and a summed loss.
//! One head predicts the sign of x, the other inside/outside the unit circle.
//!
//!     cargo run --example multi_task --release

use fastnn::nn::module::scoped;
use fastnn::prelude::*;

const SEED: u64 = 51;
const SAMPLES: usize = 512;
const HIDDEN: usize = 24;
const LR: f32 = 1e-2;
const STEPS: usize = 400;
const REPORT_EVERY: usize = 100;

struct MultiTask {
    trunk: Sequential,
    sign_head: Linear,
    circle_head: Linear,
}

impl MultiTask {
    fn new() -> MultiTask {
        MultiTask {
            trunk: Sequential::new()
                .add(Linear::new(2, HIDDEN))
                .add(Tanh)
                .add(Linear::new(HIDDEN, HIDDEN))
                .add(Tanh),
            sign_head: Linear::new(HIDDEN, 2),
            circle_head: Linear::new(HIDDEN, 2),
        }
    }

    fn heads(&self, x: &Tensor) -> (Tensor, Tensor) {
        let features = self.trunk.forward(x);
        (
            self.sign_head.forward(&features),
            self.circle_head.forward(&features),
        )
    }
}

impl Module for MultiTask {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.heads(x).0
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("trunk", self.trunk.named_parameters());
        params.extend(scoped("sign_head", self.sign_head.named_parameters()));
        params.extend(scoped("circle_head", self.circle_head.named_parameters()));
        params
    }
}

fn accuracy(logits: &Tensor, labels: &[usize]) -> f32 {
    let hits = logits
        .argmax(1)
        .iter()
        .zip(labels)
        .filter(|(a, b)| a == b)
        .count();
    hits as f32 / labels.len() as f32
}

fn main() {
    manual_seed(SEED);
    let coords = fastnn::rng::uniform(SAMPLES * 2, -1.5, 1.5);
    let sign_labels: Vec<usize> = coords.chunks(2).map(|p| (p[0] > 0.0) as usize).collect();
    let circle_labels: Vec<usize> = coords
        .chunks(2)
        .map(|p| (p[0] * p[0] + p[1] * p[1] < 1.0) as usize)
        .collect();
    let inputs = Tensor::from_vec(coords, &[SAMPLES, 2]);

    let model = MultiTask::new();
    let mut opt = Adam::new(model.parameters(), LR);

    for step in 0..STEPS {
        let (sign_logits, circle_logits) = model.heads(&inputs);
        let loss = cross_entropy(&sign_logits, &sign_labels)
            .add(&cross_entropy(&circle_logits, &circle_labels));

        opt.zero_grad();
        loss.backward();
        opt.step();

        if (step + 1) % REPORT_EVERY == 0 {
            println!(
                "step {:3}  loss {:.4}  sign {:.1}%  circle {:.1}%",
                step + 1,
                loss.item(),
                100.0 * accuracy(&sign_logits, &sign_labels),
                100.0 * accuracy(&circle_logits, &circle_labels),
            );
        }
    }
}
