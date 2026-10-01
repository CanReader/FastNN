//! MNIST with a small convolutional network.
//!
//!     cargo run --example mnist_cnn --release
//!     cargo run --example mnist_cnn --release --features cuda
//!
//! Two conv blocks (convolution, batch norm, ReLU, max pool) then a linear head.
//! Slower per step than the MLP but reaches ~99% test accuracy, because the
//! convolutions reuse the same filters at every position instead of learning
//! each pixel independently.

use fastnn::data::{Mnist, Split};
use fastnn::prelude::*;

const EPOCHS: usize = 2;
const BATCH: usize = 64;

fn main() -> fastnn::Result<()> {
    manual_seed(42);

    let train = Mnist::load(Split::Train)?;
    let test = Mnist::load(Split::Test)?;

    let device = Device::best();
    let model = Sequential::new()
        // 28×28 → 14×14, 1 channel → 16
        .add(Conv2d::same(1, 16, 3))
        .add(BatchNorm2d::new(16))
        .add(ReLU)
        .add(MaxPool2d::new(2))
        // 14×14 → 7×7, 16 channels → 32
        .add(Conv2d::same(16, 32, 3))
        .add(BatchNorm2d::new(32))
        .add(ReLU)
        .add(MaxPool2d::new(2))
        .add(Flatten::new())
        .add(Linear::new(32 * 7 * 7, 10));
    model.to_device(device);

    let mut opt = Adam::new(model.parameters(), 1e-3);
    let schedule = CosineAnnealing::new(1e-3, EPOCHS * (train.len() / BATCH));
    let train_loader = DataLoader::new(&train, BATCH)
        .shuffle(true)
        .drop_last(true)
        .to_device(device);
    let test_loader = DataLoader::new(&test, 500).to_device(device);

    println!("model: {} parameters on {device}\n", model.num_parameters());

    let mut step = 0usize;
    for epoch in 1..=EPOCHS {
        model.train();
        let (mut total_loss, mut correct, mut seen) = (0.0f32, 0usize, 0usize);

        for (index, batch) in train_loader.iter().enumerate() {
            opt.set_lr(schedule.lr_at(step));
            step += 1;

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
                    "  epoch {epoch}  batch {:4}/{}  loss {:.4}  acc {:.2}%  lr {:.2e}",
                    index + 1,
                    train_loader.batches(),
                    total_loss / (index + 1) as f32,
                    100.0 * correct as f32 / seen as f32,
                    opt.lr(),
                );
            }
        }

        // eval() switches batch norm to its running statistics, so a single
        // image scores the same as it would inside a full batch.
        model.eval();
        println!(
            "epoch {epoch} done  test acc {:.2}%\n",
            100.0 * accuracy(&model, &test_loader)
        );
    }

    save(&model, "mnist_cnn.fdl")?;
    println!("saved mnist_cnn.fdl");
    Ok(())
}

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
