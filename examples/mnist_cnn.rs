//! MNIST classification with a CNN.
//!
//! Architecture: Conv(1→32,3,p=1) → BN → ReLU → MaxPool(2)
//!               Conv(32→64,3,p=1) → BN → ReLU → MaxPool(2)
//!               Flatten → Linear(3136→256) → ReLU → Linear(256→10)
//!
//! With 28×28 inputs:  after pool1 → 14×14, after pool2 → 7×7 → 64*7*7 = 3136

use fastnn::prelude::*;
use fastnn::data::{MnistDataset, MnistSplit};
use fastnn::autograd::graph;

fn main() -> std::io::Result<()> {
    println!("FastNN — MNIST CNN");
    println!("==================\n");

    manual_seed(42);

    println!("Loading MNIST...");
    let train = MnistDataset::load(MnistSplit::Train)?;
    let test  = MnistDataset::load(MnistSplit::Test)?;
    println!("  train: {} samples", train.len());
    println!("  test:  {} samples\n", test.len());

    // Build CNN model using Sequential.
    // Images are [N, 1, 28, 28].
    let mut model = Sequential::new()
        // Block 1
        .add(Conv2d::new(1, 32, 3, 1, 1))   // → [N, 32, 28, 28]
        .add(BatchNorm2d::new(32))
        .add(ReLU)
        .add(MaxPool2d::new(2))              // → [N, 32, 14, 14]
        // Block 2
        .add(Conv2d::new(32, 64, 3, 1, 1))  // → [N, 64, 14, 14]
        .add(BatchNorm2d::new(64))
        .add(ReLU)
        .add(MaxPool2d::new(2))              // → [N, 64,  7,  7]
        // Classifier head
        .add(Flatten::new())                 // → [N, 3136]
        .add(Linear::new(3136, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));

    println!("Parameters: {}\n", model.num_parameters());

    let loss_fn   = CrossEntropyLoss::new();
    let mut optim = Adam::new(1e-3);

    let batch_size = 64;
    let epochs     = 5;

    let n_train  = train.len();
    let n_batches = n_train / batch_size;

    // Keep images in [N, 1, 28, 28] format for the CNN.
    let train_images = train.images().clone();  // [N, 1, 28, 28]
    let train_labels: Vec<usize> = train.labels().to_vec().iter().map(|&v| v as usize).collect();
    let test_images  = test.images().clone();
    let test_labels: Vec<usize>  = test.labels().to_vec().iter().map(|&v| v as usize).collect();

    for epoch in 0..epochs {
        let mut running_loss    = 0.0f32;
        let mut running_correct = 0usize;

        let mut indices: Vec<usize> = (0..n_train).collect();
        shuffle(&mut indices);

        for b in 0..n_batches {
            let batch_idx = &indices[b * batch_size..(b + 1) * batch_size];
            let (xb, yb) = gather_batch(&train_images, &train_labels, batch_idx);

            graph::enable_grad();
            {
                let mut params = model.parameters_mut();
                optim.zero_grad(&mut params);
            }

            let logits = model.forward(&xb);
            let loss   = loss_fn.forward(&logits, &yb);
            loss.backward();

            {
                let mut params = model.parameters_mut();
                optim.step(&mut params);
            }
            graph::disable_grad();

            running_loss += loss.item();
            let preds = logits.argmax(1);
            running_correct += preds.iter().zip(yb.iter()).filter(|(&p, &t)| p == t).count();

            if (b + 1) % 100 == 0 {
                println!(
                    "  epoch {}/{} | batch {:4}/{} | loss {:.4} | acc {:.2}%",
                    epoch + 1, epochs, b + 1, n_batches,
                    running_loss / (b + 1) as f32,
                    100.0 * running_correct as f32 / ((b + 1) * batch_size) as f32,
                );
            }
        }

        let test_acc = evaluate(&model, &test_images, &test_labels);
        println!(
            "epoch {}/{} done | train loss {:.4} | train acc {:.2}% | test acc {:.2}%\n",
            epoch + 1, epochs,
            running_loss / n_batches as f32,
            100.0 * running_correct as f32 / (n_batches * batch_size) as f32,
            test_acc * 100.0,
        );
    }

    Ok(())
}

/// Gather a minibatch by index, preserving [N, 1, 28, 28] shape.
fn gather_batch(images: &Tensor, labels: &[usize], indices: &[usize]) -> (Tensor, Vec<usize>) {
    let bs      = indices.len();
    let img     = images.to_vec();
    let pixels  = 1 * 28 * 28; // C*H*W
    let mut xb  = vec![0.0f32; bs * pixels];
    let mut yb  = Vec::with_capacity(bs);

    for (i, &idx) in indices.iter().enumerate() {
        let src = idx * pixels;
        xb[i * pixels..(i + 1) * pixels].copy_from_slice(&img[src..src + pixels]);
        yb.push(labels[idx]);
    }

    (Tensor::from_vec(xb, &[bs, 1, 28, 28]), yb)
}

/// Evaluate accuracy on a split (runs in small chunks to keep memory sane).
fn evaluate(model: &Sequential, images: &Tensor, labels: &[usize]) -> f32 {
    let n      = labels.len();
    let chunk  = 200; // CNN uses more memory per sample than MLP
    let pixels = 1 * 28 * 28;
    let img    = images.to_vec();
    let mut correct = 0usize;
    let mut offset  = 0;

    while offset < n {
        let this = (n - offset).min(chunk);
        let xb = Tensor::from_vec(
            img[offset * pixels..(offset + this) * pixels].to_vec(),
            &[this, 1, 28, 28],
        );
        let logits = model.forward(&xb);
        let preds  = logits.argmax(1);
        for (i, &p) in preds.iter().enumerate() {
            if p == labels[offset + i] { correct += 1; }
        }
        offset += this;
    }

    correct as f32 / n as f32
}

fn shuffle<T>(xs: &mut [T]) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    xs.shuffle(&mut rng);
}
