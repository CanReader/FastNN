//! A variational autoencoder on 2-D data.
//!
//!     cargo run --example vae --release
//!
//! The data lives on a noisy circle. The encoder outputs a *distribution*
//! (mean and log-variance) rather than a point; the reparameterisation trick —
//! `z = mu + eps · exp(logvar/2)` with `eps` drawn outside the graph — is what
//! lets the sampling step stay differentiable. The KL term pulls the latent
//! space toward N(0, I), which is why sampling from the prior afterwards
//! produces data-like points.

use fastnn::nn::module::scoped;
use fastnn::prelude::*;

const LATENT: usize = 2;

struct Vae {
    encoder: Sequential,
    to_mean: Linear,
    to_logvar: Linear,
    decoder: Sequential,
}

impl Vae {
    fn new() -> Vae {
        Vae {
            encoder: Sequential::new().add(Linear::new(2, 32)).add(Tanh),
            to_mean: Linear::new(32, LATENT),
            to_logvar: Linear::new(32, LATENT),
            decoder: Sequential::new()
                .add(Linear::new(LATENT, 32))
                .add(Tanh)
                .add(Linear::new(32, 2)),
        }
    }

    /// Reconstruction plus the two pieces the loss needs.
    fn forward_full(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let hidden = self.encoder.forward(x);
        let mean = self.to_mean.forward(&hidden);
        let logvar = self.to_logvar.forward(&hidden);

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

/// Points on the unit circle plus noise.
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
    manual_seed(31);
    let model = Vae::new();
    let mut opt = Adam::new(model.parameters(), 1e-3);

    for step in 0..3000 {
        let batch = circle_batch(128);
        let (reconstructed, mean, logvar) = model.forward_full(&batch);

        let reconstruction = mse(&reconstructed, &batch);
        // KL(N(mu, sigma) ‖ N(0, 1)), summed over latent dims, averaged over the batch.
        let kl = mean
            .square()
            .add(&logvar.exp())
            .sub(&logvar)
            .add_scalar(-1.0)
            .mul_scalar(0.5)
            .sum()
            .div_scalar(batch.dim(0) as f32);
        let loss = reconstruction.add(&kl.mul_scalar(0.1));

        opt.zero_grad();
        loss.backward();
        opt.step();

        if (step + 1) % 600 == 0 {
            println!(
                "step {:4}  reconstruction {:.4}  kl {:.3}",
                step + 1,
                reconstruction.item(),
                kl.item()
            );
        }
    }

    // Decode from the prior: no encoder, no data — radius ≈ 1 means the latent
    // space really did learn the circle.
    let generated = no_grad(|| model.decoder.forward(&Tensor::randn(&[256, LATENT]))).to_vec();
    let mean_radius: f32 = generated
        .chunks(2)
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .sum::<f32>()
        / 256.0;
    println!("\nmean radius of 256 prior samples: {mean_radius:.3}  (data lives at 1.0)");
}
