//! Minimal GAN that learns to sample from N(3, 0.5).
//!
//!     cargo run --example gan --release

use fastnn::prelude::*;

const NOISE: usize = 8;
const BATCH: usize = 64;
const HIDDEN: usize = 32;
const LR: f32 = 1e-3;
const STEPS: usize = 2000;
const REPORT_EVERY: usize = 400;
const SEED: u64 = 21;

fn main() {
    manual_seed(SEED);
    let (target_mean, target_std) = (3.0, 0.5);

    let generator = Sequential::new()
        .add(Linear::new(NOISE, HIDDEN))
        .add(Tanh)
        .add(Linear::new(HIDDEN, 1));
    let discriminator = Sequential::new()
        .add(Linear::new(1, HIDDEN))
        .add(LeakyReLU::new(0.2))
        .add(Linear::new(HIDDEN, 1));

    let mut g_opt = Adam::new(generator.parameters(), LR);
    let mut d_opt = Adam::new(discriminator.parameters(), LR);

    let real_labels = Tensor::from_vec(vec![1.0; BATCH], &[BATCH, 1]);
    let fake_labels = Tensor::from_vec(vec![0.0; BATCH], &[BATCH, 1]);

    for step in 0..STEPS {
        let real = Tensor::randn(&[BATCH, 1])
            .mul_scalar(target_std)
            .add_scalar(target_mean);
        let fake = generator.forward(&Tensor::randn(&[BATCH, NOISE]));

        // Detached so the discriminator loss doesn't reach the generator.
        let d_loss = bce_with_logits(&discriminator.forward(&real), &real_labels).add(
            &bce_with_logits(&discriminator.forward(&fake.detach()), &fake_labels),
        );
        d_opt.zero_grad();
        d_loss.backward();
        d_opt.step();

        let g_loss = bce_with_logits(&discriminator.forward(&fake), &real_labels);
        g_opt.zero_grad();
        g_loss.backward();
        g_opt.step();

        if (step + 1) % REPORT_EVERY == 0 {
            let sample = no_grad(|| generator.forward(&Tensor::randn(&[256, NOISE]))).to_vec();
            let mean = sample.iter().sum::<f32>() / sample.len() as f32;
            let var = sample.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / sample.len() as f32;
            println!(
                "step {:4}  d_loss {:.3}  g_loss {:.3}  generated mean {mean:.2} std {:.2}  (target {target_mean} / {target_std})",
                step + 1,
                d_loss.item(),
                g_loss.item(),
                var.sqrt()
            );
        }
    }
}
