//! MNIST classification with a 2-layer MLP.
//!
//! Downloads MNIST on first run, trains a 784→128→10 MLP with Adam +
//! CrossEntropyLoss, then reports test accuracy.

use fastnn::prelude::*;
use fastnn::data::{MnistDataset, MnistSplit};
use fastnn::autograd::graph;

fn main() -> std::io::Result<()> {
    println!("FastNN — MNIST MLP");
    println!("==================\n");

    manual_seed(42);

    println!("Loading MNIST...");
    let train = MnistDataset::load(MnistSplit::Train)?;
    let test = MnistDataset::load(MnistSplit::Test)?;
    println!("  train: {} samples", train.len());
    println!("  test:  {} samples", test.len());

    // Build a simple MLP classifier.
    let mut model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU)
        .add(Linear::new(128, 10));
    println!("\nModel parameters: {}\n", model.num_parameters());

    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(1e-3);

    let batch_size = 128;
    let epochs = 3;

    // Flatten images once to [N, 784] for the MLP.
    let train_images = train.images().reshape(&[train.len() as i64, 784]);
    let train_labels: Vec<usize> = train.labels().to_vec().iter().map(|&v| v as usize).collect();
    let test_images = test.images().reshape(&[test.len() as i64, 784]);
    let test_labels: Vec<usize> = test.labels().to_vec().iter().map(|&v| v as usize).collect();

    let n_train = train.len();
    let n_batches = n_train / batch_size;

    for epoch in 0..epochs {
        let mut running_loss = 0.0f32;
        let mut running_correct = 0usize;

        // Shuffle indices for this epoch.
        let mut indices: Vec<usize> = (0..n_train).collect();
        shuffle(&mut indices);

        for b in 0..n_batches {
            // Build the batch by gathering the shuffled indices.
            let batch_idx = &indices[b * batch_size..(b + 1) * batch_size];
            let (xb, yb) = gather_batch(&train_images, &train_labels, batch_idx);

            graph::enable_grad();
            {
                let mut params = model.parameters_mut();
                optimizer.zero_grad(&mut params);
            }

            let logits = model.forward(&xb);
            let loss = loss_fn.forward(&logits, &yb);
            loss.backward();

            {
                let mut params = model.parameters_mut();
                optimizer.step(&mut params);
            }
            graph::disable_grad();

            running_loss += loss.item();
            let preds = logits.argmax(1);
            running_correct += preds.iter().zip(yb.iter()).filter(|(&p, &t)| p == t).count();

            if (b + 1) % 50 == 0 {
                let seen = (b + 1) * batch_size;
                println!(
                    "  epoch {}/{} | batch {:4}/{} | loss {:.4} | acc {:.2}%",
                    epoch + 1, epochs, b + 1, n_batches,
                    running_loss / (b + 1) as f32,
                    100.0 * running_correct as f32 / seen as f32,
                );
            }
        }

        // Test evaluation (whole test set at once).
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

/// Gather a minibatch by index from flattened images + labels.
fn gather_batch(images_flat: &Tensor, labels: &[usize], indices: &[usize]) -> (Tensor, Vec<usize>) {
    let bs = indices.len();
    let img = images_flat.to_vec();
    let mut xb = vec![0.0f32; bs * 784];
    let mut yb = Vec::with_capacity(bs);
    for (i, &idx) in indices.iter().enumerate() {
        let src = idx * 784;
        xb[i * 784..(i + 1) * 784].copy_from_slice(&img[src..src + 784]);
        yb.push(labels[idx]);
    }
    (Tensor::from_vec(xb, &[bs, 784]), yb)
}

fn evaluate(model: &Sequential, images: &Tensor, labels: &[usize]) -> f32 {
    // Run in chunks to keep memory sane.
    let n = labels.len();
    let chunk = 1000;
    let img = images.to_vec();
    let mut correct = 0usize;
    let mut offset = 0;
    while offset < n {
        let this = (n - offset).min(chunk);
        let xb = Tensor::from_vec(
            img[offset * 784..(offset + this) * 784].to_vec(),
            &[this, 784],
        );
        let logits = model.forward(&xb);
        let preds = logits.argmax(1);
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
