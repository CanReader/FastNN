//! Policy-gradient reinforcement learning on a multi-armed bandit.
//!
//!     cargo run --example reinforce --release
//!
//! REINFORCE in its smallest complete form. The policy is a softmax over five
//! arm preferences; each pull nudges the log-probability of the chosen arm by
//! the reward's advantage over a running baseline. `cross_entropy` already *is*
//! negative log-likelihood, so scaling it by the advantage is the entire loss.

use fastnn::nn::Param;
use fastnn::prelude::*;

fn main() {
    manual_seed(41);
    // Expected payout per arm; arm 3 is best, but only by observation.
    let arms = [0.2f32, 0.5, 0.4, 0.9, 0.6];

    let preferences = Param::new(Tensor::zeros(&[1, arms.len()]));
    let mut opt = Adam::new(vec![preferences.clone()], 5e-2);
    let sampler = Sampler::new();

    let mut baseline = 0.0f32;
    for step in 0..2000 {
        let logits = preferences.tensor();
        let arm = sampler.sample(&logits.detach());
        let reward = arms[arm] + 0.1 * fastnn::rng::normal(1)[0];

        // Better than the baseline → make this arm likelier; worse → less.
        let advantage = reward - baseline;
        baseline += 0.01 * (reward - baseline);

        let loss = cross_entropy(&logits, &[arm]).mul_scalar(advantage);
        opt.zero_grad();
        loss.backward();
        opt.step();

        if (step + 1) % 400 == 0 {
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
