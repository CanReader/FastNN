//! A minimal GAN: two networks trained against each other.
//!
//!     cargo run --example gan --release
//!
//! The generator turns noise into numbers; the discriminator guesses which
//! numbers came from the real distribution, N(3, 0.5). Each is the other's
//! loss function. The detach on generated samples during the discriminator
//! step is the whole trick: it stops the discriminator's loss from reaching
//! back into the generator's weights.

use fastnn::prelude::*;

const NOISE: usize = 8;
const BATCH: usize = 64;

fn main() {
    manual_seed(21);
    let (target_mean, target_std) = (3.0, 0.5);

    let generator = Sequential::new()
        .add(Linear::new(NOISE, 32))
        .add(Tanh)
        .add(Linear::new(32, 1));
    let discriminator = Sequential::new()
        .add(Linear::new(1, 32))
        .add(LeakyReLU::new(0.2))
        .add(Linear::new(32, 1));

    let mut g_opt = Adam::new(generator.parameters(), 1e-3);
    let mut d_opt = Adam::new(discriminator.parameters(), 1e-3);

    let real_labels = Tensor::from_vec(vec![1.0; BATCH], &[BATCH, 1]);
    let fake_labels = Tensor::from_vec(vec![0.0; BATCH], &[BATCH, 1]);

    for step in 0..2000 {
        let real = Tensor::randn(&[BATCH, 1])
            .mul_scalar(target_std)
            .add_scalar(target_mean);
        let fake = generator.forward(&Tensor::randn(&[BATCH, NOISE]));

        // Discriminator: real → 1, fake → 0. The generator must not receive
        // this gradient, so its samples enter detached.
        let d_loss = bce_with_logits(&discriminator.forward(&real), &real_labels).add(
            &bce_with_logits(&discriminator.forward(&fake.detach()), &fake_labels),
        );
        d_opt.zero_grad();
        d_loss.backward();
        d_opt.step();

        // Generator: make the discriminator call the same fakes real.
        let g_loss = bce_with_logits(&discriminator.forward(&fake), &real_labels);
        g_opt.zero_grad();
        g_loss.backward();
        g_opt.step();

        if (step + 1) % 400 == 0 {
            let sample = no_grad(|| generator.forward(&Tensor::randn(&[256, NOISE]))).to_vec();
            let mean = sample.iter().sum::<f32>() / sample.len() as f32;
            let var = sample.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / sample.len() as f32;
            println!(
                "step {:4}  d_loss {:.3}  g_loss {:.3}  generated mean {mean:.2} std {:.2}  (target {target_mean} / {target_std})",
                step + 1, d_loss.item(), g_loss.item(), var.sqrt()
            );
        }
    }
}
