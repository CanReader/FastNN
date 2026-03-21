//! MNIST classifier example
//!
//! Demonstrates a CNN for digit classification.
//! Note: This example requires MNIST data files. In a real setup, you'd
//! download and parse them. Here we generate synthetic data to demonstrate the API.

use fastnn::prelude::*;
use fastnn::data::dataset::TensorDataset;

fn main() {
    println!("FastNN — MNIST CNN Classifier");
    println!("==============================\n");

    manual_seed(42);

    // Synthetic MNIST-like data (in practice, load real MNIST)
    let num_train = 256;
    let num_test = 64;

    println!("Generating synthetic MNIST-like data...");
    let train_images = Tensor::randn(&[num_train, 1, 28, 28]);
    let train_labels = Tensor::from_vec(
        (0..num_train).map(|i| (i % 10) as f32).collect(),
        &[num_train],
    );

    let test_images = Tensor::randn(&[num_test, 1, 28, 28]);
    let test_labels = Tensor::from_vec(
        (0..num_test).map(|i| (i % 10) as f32).collect(),
        &[num_test],
    );

    // Build CNN model
    let conv1 = Conv2d::new(1, 32, 3, 1, 1);   // -> [N, 32, 28, 28]
    let pool1 = MaxPool2d::new(2);               // -> [N, 32, 14, 14]
    let conv2 = Conv2d::new(32, 64, 3, 1, 1);   // -> [N, 64, 14, 14]
    let pool2 = MaxPool2d::new(2);               // -> [N, 64, 7, 7]
    let fc1 = Linear::new(64 * 7 * 7, 128);
    let fc2 = Linear::new(128, 10);

    println!("Model architecture:");
    println!("  Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)");
    println!("  Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)");
    println!("  Linear(3136, 128) -> ReLU -> Linear(128, 10)");
    let total_params: usize = conv1.parameters().iter().chain(conv2.parameters().iter())
        .chain(fc1.parameters().iter()).chain(fc2.parameters().iter())
        .map(|p| p.numel()).sum();
    println!("  Total parameters: {}\n", total_params);

    let loss_fn = CrossEntropyLoss::new();
    let batch_size = 32;
    let epochs = 5;

    let dataset = TensorDataset::new(train_images, train_labels);
    let dataloader = DataLoader::new(&dataset, batch_size).shuffle(true);

    // Training loop
    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let mut num_batches = 0;

        for (batch_images, batch_labels) in dataloader.iter() {
            // Forward pass
            let x = conv1.forward(&batch_images);
            let x = x.relu();
            let x = pool1.forward(&x);
            let x = conv2.forward(&x);
            let x = x.relu();
            let x = pool2.forward(&x);

            // Flatten
            let batch = x.shape()[0];
            let x = x.reshape(&[batch as i64, -1]);
            let x = fc1.forward(&x);
            let x = x.relu();
            let logits = fc2.forward(&x);

            // Compute loss
            let label_data = batch_labels.to_vec();
            let targets: Vec<usize> = label_data.iter().map(|&v| v as usize).collect();
            let loss = loss_fn.forward(&logits, &targets);

            total_loss += loss.item();
            num_batches += 1;
        }

        println!("Epoch {}/{} | Avg Loss: {:.4}", epoch + 1, epochs, total_loss / num_batches as f32);
    }

    // Evaluation
    println!("\nEvaluating on test set...");
    let x = conv1.forward(&test_images);
    let x = x.relu();
    let x = pool1.forward(&x);
    let x = conv2.forward(&x);
    let x = x.relu();
    let x = pool2.forward(&x);
    let batch = x.shape()[0];
    let x = x.reshape(&[batch as i64, -1]);
    let x = fc1.forward(&x);
    let x = x.relu();
    let logits = fc2.forward(&x);

    let predictions = logits.argmax(1);
    let targets: Vec<usize> = test_labels.to_vec().iter().map(|&v| v as usize).collect();
    let correct = predictions.iter().zip(targets.iter()).filter(|(&p, &t)| p == t).count();
    println!("Test accuracy: {}/{} ({:.1}%)", correct, num_test, 100.0 * correct as f32 / num_test as f32);
}
