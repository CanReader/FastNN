//! MNIST with a two-layer MLP. Downloads the dataset on first run.
//!
//!     cargo run --example mnist_mlp --release
//!     cargo run --example mnist_mlp --release --features cuda
//!
//! Reaches roughly 97% test accuracy in three epochs.

use fastnn::data::{Mnist, Split};
use fastnn::prelude::*;

const EPOCHS: usize = 3;
const BATCH: usize = 128;

fn main() -> fastnn::Result<()> {
    manual_seed(42);

    let train = Mnist::load(Split::Train)?;
    let test = Mnist::load(Split::Test)?;
    println!(
        "mnist: {} training, {} test images",
        train.len(),
        test.len()
    );

    let device = Device::best();
    let model = Sequential::new()
        .add(Flatten::new())
        .add(Linear::new(784, 128))
        .add(ReLU)
        .add(Linear::new(128, 10));
    model.to_device(device);

    let mut opt = Adam::new(model.parameters(), 1e-3);
    let train_loader = DataLoader::new(&train, BATCH)
        .shuffle(true)
        .to_device(device);
    let test_loader = DataLoader::new(&test, 1000).to_device(device);

    println!("model: {} parameters on {device}\n", model.num_parameters());

    for epoch in 1..=EPOCHS {
        model.train();
        let (mut total_loss, mut correct, mut seen) = (0.0f32, 0usize, 0usize);

        for (index, batch) in train_loader.iter().enumerate() {
            let labels = batch.labels();
            let logits = model.forward(&batch.inputs);
            let loss = cross_entropy(&logits, &labels);

            opt.zero_grad();
            loss.backward();
            opt.step();

            total_loss += loss.item();
            correct += count_correct(&logits, &labels);
            seen += batch.len();

            if (index + 1) % 100 == 0 {
                println!(
                    "  epoch {epoch}  batch {:4}/{}  loss {:.4}  acc {:.2}%",
                    index + 1,
                    train_loader.batches(),
                    total_loss / (index + 1) as f32,
                    100.0 * correct as f32 / seen as f32,
                );
            }
        }

        model.eval();
        println!(
            "epoch {epoch} done  train acc {:.2}%  test acc {:.2}%\n",
            100.0 * correct as f32 / seen as f32,
            100.0 * accuracy(&model, &test_loader),
        );
    }

    save(&model, "mnist_mlp.fdl")?;
    println!("saved mnist_mlp.fdl");
    Ok(())
}

/// Fraction of the loader's items the model classifies correctly.
fn accuracy(model: &dyn Module, loader: &DataLoader) -> f32 {
    let (mut correct, mut seen) = (0usize, 0usize);
    for batch in loader.iter() {
        let logits = no_grad(|| model.forward(&batch.inputs));
        correct += count_correct(&logits, &batch.labels());
        seen += batch.len();
    }
    correct as f32 / seen as f32
}

fn count_correct(logits: &Tensor, labels: &[usize]) -> usize {
    logits
        .argmax(1)
        .iter()
        .zip(labels)
        .filter(|(p, t)| p == t)
        .count()
}
