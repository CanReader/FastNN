//! Fine-tuning: pretrain on one task, freeze the backbone, retrain a new head
//! on another.
//!
//!     cargo run --example finetune --release
//!
//! Task A classifies 2-D points by quadrant; task B by whether they fall inside
//! the unit circle. The backbone's features transfer, so the second task trains
//! a 34-parameter head instead of a whole network — and because frozen
//! parameters hand out detached tensors, the backward pass never even reaches
//! the backbone.

use fastnn::nn::module::scoped;
use fastnn::prelude::*;

struct Net {
    backbone: Sequential,
    head: Linear,
}

impl Net {
    fn new(classes: usize) -> Net {
        Net {
            backbone: Sequential::new()
                .add(Linear::new(2, 16))
                .add(Tanh)
                .add(Linear::new(16, 16))
                .add(Tanh),
            head: Linear::new(16, classes),
        }
    }
}

impl Module for Net {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.head.forward(&self.backbone.forward(x))
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("backbone", self.backbone.named_parameters());
        params.extend(scoped("head", self.head.named_parameters()));
        params
    }
}

/// 2-D points and their labels under `rule`.
fn dataset(n: usize, rule: impl Fn(f32, f32) -> usize) -> (Tensor, Vec<usize>) {
    let coords = fastnn::rng::uniform(n * 2, -1.5, 1.5);
    let labels = coords.chunks(2).map(|p| rule(p[0], p[1])).collect();
    (Tensor::from_vec(coords, &[n, 2]), labels)
}

fn accuracy(model: &impl Module, inputs: &Tensor, labels: &[usize]) -> f32 {
    let predicted = no_grad(|| model.forward(inputs)).argmax(1);
    let hits = predicted.iter().zip(labels).filter(|(a, b)| a == b).count();
    hits as f32 / labels.len() as f32
}

fn fit(model: &impl Module, inputs: &Tensor, labels: &[usize], steps: usize) {
    // trainable_parameters: after freezing, this is just the head.
    let mut opt = Adam::new(model.trainable_parameters(), 1e-2);
    for _ in 0..steps {
        let loss = cross_entropy(&model.forward(inputs), labels);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
}

fn main() -> fastnn::Result<()> {
    manual_seed(11);
    let checkpoint = std::env::temp_dir().join("finetune_backbone.fdl");

    // ── Pretrain on task A: which quadrant is the point in? ──────────────────
    let quadrant = |x: f32, y: f32| (x > 0.0) as usize + 2 * ((y > 0.0) as usize);
    let (inputs, labels) = dataset(512, quadrant);

    let pretrained = Net::new(4);
    fit(&pretrained, &inputs, &labels, 300);
    println!(
        "task A (quadrants):    {:.1}%",
        100.0 * accuracy(&pretrained, &inputs, &labels)
    );
    save(&pretrained, &checkpoint)?;

    // ── Fine-tune on task B: inside or outside the unit circle? ──────────────
    let circle = |x: f32, y: f32| (x * x + y * y < 1.0) as usize;
    let (inputs, labels) = dataset(512, circle);

    let mut model = Net::new(4);
    load(&model, &checkpoint)?;
    model.backbone.freeze();
    model.head = Linear::new(16, 2); // fresh head for the new label space

    let before = accuracy(&model, &inputs, &labels);
    fit(&model, &inputs, &labels, 300);
    println!("task B before head:    {:.1}%", 100.0 * before);
    println!(
        "task B after head:     {:.1}%",
        100.0 * accuracy(&model, &inputs, &labels)
    );
    println!(
        "trainable parameters:  {} of {}",
        model
            .trainable_parameters()
            .iter()
            .map(|p| p.numel())
            .sum::<usize>(),
        model.num_parameters()
    );

    std::fs::remove_file(&checkpoint).ok();
    Ok(())
}
