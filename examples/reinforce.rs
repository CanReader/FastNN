//! REINFORCE on a five-armed bandit. The policy is a softmax over per-arm
//! preferences, trained on advantage over a running baseline.
//!
//!     cargo run --example reinforce --release

use fastnn::nn::Param;
use fastnn::prelude::*;

const SEED: u64 = 41;
const STEPS: usize = 2000;
const LR: f32 = 5e-2;
const REWARD_NOISE: f32 = 0.1;
const BASELINE_RATE: f32 = 0.01;
const REPORT_EVERY: usize = 400;

fn main() {
    manual_seed(SEED);
    // Mean payout per arm; arm 3 is the best.
    let arms = [0.2f32, 0.5, 0.4, 0.9, 0.6];

    let preferences = Param::new(Tensor::zeros(&[1, arms.len()]));
    let mut opt = Adam::new(vec![preferences.clone()], LR);
    let sampler = Sampler::new();

    let mut baseline = 0.0f32;
    for step in 0..STEPS {
        let logits = preferences.tensor();
        let arm = sampler.sample(&logits.detach());
        let reward = arms[arm] + REWARD_NOISE * fastnn::rng::normal(1)[0];

        // cross_entropy is the negative log-prob, so scaling by advantage
        // gives the policy gradient loss.
        let advantage = reward - baseline;
        baseline += BASELINE_RATE * (reward - baseline);

        let loss = cross_entropy(&logits, &[arm]).mul_scalar(advantage);
        opt.zero_grad();
        loss.backward();
        opt.step();

        if (step + 1) % REPORT_EVERY == 0 {
            let probs = no_grad(|| preferences.tensor().softmax()).to_vec();
            let shown: Vec<String> = probs.iter().map(|p| format!("{p:.2}")).collect();
            println!(
                "step {:4}  baseline {baseline:.2}  policy [{}]",
                step + 1,
                shown.join(" ")
            );
        }
    }

    let best = no_grad(|| preferences.tensor()).argmax(1)[0];
    println!("\nlearned best arm: {best} (true best: 3)");
    assert_eq!(best, 3, "policy failed to find the best arm");
}
