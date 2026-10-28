//! VAE on noisy points around the unit circle.
//!
//!     cargo run --example vae --release

use fastnn::nn::module::scoped;
use fastnn::prelude::*;

const LATENT: usize = 2;
const HIDDEN: usize = 32;
const SEED: u64 = 31;
const BATCH: usize = 128;
const LR: f32 = 1e-3;
const STEPS: usize = 3000;
const REPORT_EVERY: usize = 600;
const KL_WEIGHT: f32 = 0.1;
const PRIOR_SAMPLES: usize = 256;

struct Vae {
    encoder: Sequential,
    to_mean: Linear,
    to_logvar: Linear,
    decoder: Sequential,
}

impl Vae {
    fn new() -> Vae {
        Vae {
            encoder: Sequential::new().add(Linear::new(2, HIDDEN)).add(Tanh),
            to_mean: Linear::new(HIDDEN, LATENT),
            to_logvar: Linear::new(HIDDEN, LATENT),
            decoder: Sequential::new()
                .add(Linear::new(LATENT, HIDDEN))
                .add(Tanh)
                .add(Linear::new(HIDDEN, 2)),
        }
    }

    /// Returns the reconstruction, mean and log-variance.
    fn forward_full(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let hidden = self.encoder.forward(x);
        let mean = self.to_mean.forward(&hidden);
        let logvar = self.to_logvar.forward(&hidden);

        // eps is sampled outside the graph so z stays differentiable.
        let eps = Tensor::randn(&[x.dim(0), LATENT]);
        let z = mean.add(&logvar.mul_scalar(0.5).exp().mul(&eps));
        (self.decoder.forward(&z), mean, logvar)
    }
}

impl Module for Vae {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_full(x).0
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("encoder", self.encoder.named_parameters());
        params.extend(scoped("to_mean", self.to_mean.named_parameters()));
        params.extend(scoped("to_logvar", self.to_logvar.named_parameters()));
        params.extend(scoped("decoder", self.decoder.named_parameters()));
        params
    }
}

fn circle_batch(n: usize) -> Tensor {
    let angles = fastnn::rng::uniform(n, 0.0, std::f32::consts::TAU);
    let noise = fastnn::rng::normal(n * 2);
    let data: Vec<f32> = angles
        .iter()
        .zip(noise.chunks(2))
        .flat_map(|(a, e)| [a.cos() + 0.05 * e[0], a.sin() + 0.05 * e[1]])
        .collect();
    Tensor::from_vec(data, &[n, 2])
}

fn main() {
    manual_seed(SEED);
    let model = Vae::new();
    let mut opt = Adam::new(model.parameters(), LR);

    for step in 0..STEPS {
        let batch = circle_batch(BATCH);
        let (reconstructed, mean, logvar) = model.forward_full(&batch);

        let reconstruction = mse(&reconstructed, &batch);
        // KL to N(0, 1), summed over latent dims and averaged over the batch.
        let kl = mean
            .square()
            .add(&logvar.exp())
            .sub(&logvar)
            .add_scalar(-1.0)
            .mul_scalar(0.5)
            .sum()
            .div_scalar(batch.dim(0) as f32);
        let loss = reconstruction.add(&kl.mul_scalar(KL_WEIGHT));

        opt.zero_grad();
        loss.backward();
        opt.step();

        if (step + 1) % REPORT_EVERY == 0 {
            println!(
                "step {:4}  reconstruction {:.4}  kl {:.3}",
                step + 1,
                reconstruction.item(),
                kl.item()
            );
        }
    }

    // Decode from the prior; a radius near 1 means it learned the circle.
    let generated = no_grad(|| {
        model
            .decoder
            .forward(&Tensor::randn(&[PRIOR_SAMPLES, LATENT]))
    })
    .to_vec();
    let mean_radius: f32 = generated
        .chunks(2)
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .sum::<f32>()
        / PRIOR_SAMPLES as f32;
    println!(
        "\nmean radius of {PRIOR_SAMPLES} prior samples: {mean_radius:.3}  (data lives at 1.0)"
    );
}
