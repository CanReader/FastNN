//! Transformer example — sequence classification
//!
//! Demonstrates building a Transformer encoder for sequence classification.

use fastdl::prelude::*;
use fastdl::nn::embedding::PositionalEncoding;

fn main() {
    println!("fastDL — Transformer Encoder");
    println!("============================\n");

    manual_seed(42);

    // Hyperparameters
    let vocab_size = 1000;
    let d_model = 64;
    let nhead = 4;
    let d_ff = 256;
    let num_layers = 2;
    let num_classes = 5;
    let seq_len = 16;
    let batch_size = 8;
    let dropout = 0.1;

    // Build model components
    let embedding = Embedding::new(vocab_size, d_model);
    let pos_encoding = PositionalEncoding::new(d_model, 512);
    let encoder = TransformerEncoder::new(d_model, nhead, d_ff, num_layers, dropout);
    let classifier = Linear::new(d_model, num_classes);

    // Count parameters
    let total_params: usize = embedding.parameters().iter()
        .chain(encoder.parameters().iter())
        .chain(classifier.parameters().iter())
        .map(|p| p.numel()).sum();

    println!("Model: Embedding({}, {}) -> PositionalEncoding -> TransformerEncoder({}L) -> Linear({}, {})",
             vocab_size, d_model, num_layers, d_model, num_classes);
    println!("Total parameters: {}\n", total_params);

    // Synthetic input: random token IDs
    let input_ids: Vec<f32> = (0..batch_size * seq_len)
        .map(|_| (rand::random::<f32>() * vocab_size as f32).floor())
        .collect();
    let input = Tensor::from_vec(input_ids, &[batch_size, seq_len]);
    let targets: Vec<usize> = (0..batch_size).map(|i| i % num_classes).collect();

    let loss_fn = CrossEntropyLoss::new();

    // Forward pass
    println!("Running forward pass...");
    let embedded = embedding.forward(&input);
    println!("  After embedding: {:?}", embedded.shape());

    let encoded = pos_encoding.forward(&embedded);
    println!("  After positional encoding: {:?}", encoded.shape());

    let hidden = encoder.forward(&encoded);
    println!("  After transformer encoder: {:?}", hidden.shape());

    // Mean pooling over sequence dimension for classification
    let pooled = hidden.mean_axis(1);
    println!("  After mean pooling: {:?}", pooled.shape());

    let logits = classifier.forward(&pooled);
    println!("  Logits shape: {:?}", logits.shape());

    // Compute loss
    let loss = loss_fn.forward(&logits, &targets);
    println!("\n  Loss: {:.4}", loss.item());

    // Predictions
    let predictions = logits.argmax(1);
    println!("  Predictions: {:?}", predictions);
    println!("  Targets:     {:?}", targets);

    let correct = predictions.iter().zip(targets.iter()).filter(|(&p, &t)| p == t).count();
    println!("  Accuracy: {}/{} ({:.1}%)", correct, batch_size, 100.0 * correct as f32 / batch_size as f32);

    println!("\nDone!");
}
