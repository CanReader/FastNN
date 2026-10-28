//! Character-level GPT trained from scratch.
//!
//!     cargo run --example char_lm --release [-- corpus.txt]
//!
//! Defaults to a small embedded excerpt. Checkpoints to `char_lm.fdl` and
//! resumes from it on the next run.

use fastnn::nn::module::scoped;
use fastnn::prelude::*;

const CHECKPOINT: &str = "char_lm.fdl";
const SEED: u64 = 1337;
const DROPOUT: f32 = 0.1;
const CLIP_NORM: f32 = 1.0;
const SAMPLE_CHARS: usize = 400;
const TEMPERATURE: f32 = 0.8;

const CORPUS: &str = include_str!("data/tiny_corpus.txt");

fn main() -> fastnn::Result<()> {
    manual_seed(SEED);

    let text = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(&path)?,
        None => CORPUS.to_string(),
    };
    let vocab = Vocab::new(&text);
    let data = vocab.encode(&text);
    let config = Config::for_corpus(data.len());

    println!("corpus: {} chars, {} distinct", data.len(), vocab.len());
    println!("config: {config:?}");

    let device = Device::best();
    let model = CharGPT::new(vocab.len(), &config);
    model.to_device(device);
    println!(
        "model:  {} parameters on {device}\n",
        model.num_parameters()
    );

    let mut opt = AdamW::new(model.parameters(), config.lr).betas(0.9, 0.95);

    // Resuming needs the optimizer state too, not just the weights.
    let first_step = load_training(&model, &mut opt, CHECKPOINT).unwrap_or(0) as usize;
    if first_step > 0 {
        println!("resuming from step {first_step}\n");
    }
    // Adam's variance estimate is noisy early on, so ramp the lr up.
    let schedule = Warmup::new(CosineAnnealing::new(config.lr, config.steps), config.warmup);
    let params = model.parameters();

    model.train();
    for step in first_step..config.steps {
        opt.set_lr(schedule.lr_at(step));

        let (inputs, targets) = sample_batch(&data, &config);
        let loss = cross_entropy(&model.forward_ids(&inputs), &targets);

        opt.zero_grad();
        loss.backward();
        clip_grad_norm(&params, CLIP_NORM);
        opt.step();

        if (step + 1) % config.report_every == 0 {
            println!(
                "step {:5}/{}  loss {:.4}  lr {:.2e}",
                step + 1,
                config.steps,
                loss.item(),
                opt.lr()
            );
        }
        if (step + 1) % config.checkpoint_every == 0 {
            save_training(&model, &opt, step as u64 + 1, CHECKPOINT)?;
        }
    }

    model.eval();
    println!("\nsample:");
    println!(
        "{}",
        generate(&model, &vocab, &config, SAMPLE_CHARS, TEMPERATURE)
    );
    Ok(())
}

struct CharGPT {
    tokens: Embedding,
    positions: Embedding,
    blocks: TransformerStack,
    head: Linear,
    context: usize,
}

impl CharGPT {
    fn new(vocab: usize, config: &Config) -> CharGPT {
        CharGPT {
            tokens: Embedding::new(vocab, config.width),
            positions: Embedding::new(config.context, config.width),
            blocks: TransformerStack::causal(
                config.width,
                config.heads,
                4 * config.width,
                config.layers,
                DROPOUT,
            ),
            head: Linear::no_bias(config.width, vocab),
            context: config.context,
        }
    }

    /// Returns `[batch * length, vocab]` logits, flattened for `cross_entropy`.
    fn forward_ids(&self, ids: &[Vec<usize>]) -> Tensor {
        let (batch, length) = (ids.len(), ids[0].len());
        assert!(
            length <= self.context,
            "sequence {length} exceeds context {}",
            self.context
        );

        let embedded = self.tokens.lookup_batch(ids);
        let positions = self
            .positions
            .lookup(&(0..length).collect::<Vec<_>>())
            .reshape(&[1, length as i64, -1]);

        let hidden = self.blocks.forward(&embedded.add(&positions));
        self.head
            .forward(&hidden)
            .reshape(&[(batch * length) as i64, -1])
    }
}

impl Module for CharGPT {
    fn forward(&self, input: &Tensor) -> Tensor {
        let ids: Vec<Vec<usize>> = (0..input.dim(0))
            .map(|row| {
                input
                    .narrow(0, row, 1)
                    .to_vec()
                    .iter()
                    .map(|&v| v as usize)
                    .collect()
            })
            .collect();
        self.forward_ids(&ids)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("tokens", self.tokens.named_parameters());
        params.extend(scoped("positions", self.positions.named_parameters()));
        params.extend(scoped("blocks", self.blocks.named_parameters()));
        params.extend(scoped("head", self.head.named_parameters()));
        params
    }

    fn set_training(&self, training: bool) {
        self.blocks.set_training(training);
    }
}

struct Vocab {
    to_id: std::collections::HashMap<char, usize>,
    to_char: Vec<char>,
}

impl Vocab {
    fn new(text: &str) -> Vocab {
        let to_char: Vec<char> = text
            .chars()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        Vocab {
            to_id: to_char.iter().enumerate().map(|(i, &c)| (c, i)).collect(),
            to_char,
        }
    }

    fn len(&self) -> usize {
        self.to_char.len()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.to_id.get(&c).copied())
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.to_char[i]).collect()
    }
}

/// Random windows; targets are the inputs shifted by one character.
fn sample_batch(data: &[usize], config: &Config) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(config.batch);
    let mut targets = Vec::with_capacity(config.batch * config.context);

    for _ in 0..config.batch {
        let limit = data.len() - config.context - 1;
        let start = (fastnn::rng::uniform(1, 0.0, 1.0)[0] * limit as f32) as usize;
        inputs.push(data[start..start + config.context].to_vec());
        targets.extend_from_slice(&data[start + 1..start + config.context + 1]);
    }
    (inputs, targets)
}

fn generate(
    model: &CharGPT,
    vocab: &Vocab,
    config: &Config,
    count: usize,
    temperature: f32,
) -> String {
    let mut ids = vec![0usize];

    for _ in 0..count {
        let window = ids.len().saturating_sub(config.context);
        let context = vec![ids[window..].to_vec()];

        let logits = no_grad(|| model.forward_ids(&context));
        let last = logits.narrow(0, logits.dim(0) - 1, 1);
        ids.push(sample(&last.div_scalar(temperature).softmax().to_vec()));
    }

    vocab.decode(&ids)
}

fn sample(probabilities: &[f32]) -> usize {
    let mut draw = fastnn::rng::uniform(1, 0.0, 1.0)[0];
    for (index, &p) in probabilities.iter().enumerate() {
        draw -= p;
        if draw <= 0.0 {
            return index;
        }
    }
    probabilities.len() - 1
}

#[derive(Debug)]
struct Config {
    context: usize,
    batch: usize,
    width: usize,
    heads: usize,
    layers: usize,
    lr: f32,
    warmup: usize,
    steps: usize,
    report_every: usize,
    checkpoint_every: usize,
}

impl Config {
    fn for_corpus(chars: usize) -> Config {
        // A model much bigger than its corpus just memorises it.
        if chars > 200_000 {
            Config {
                context: 128,
                batch: 32,
                width: 256,
                heads: 8,
                layers: 6,
                lr: 3e-4,
                warmup: 400,
                steps: 5_000,
                report_every: 100,
                checkpoint_every: 500,
            }
        } else if chars > 20_000 {
            Config {
                context: 64,
                batch: 32,
                width: 128,
                heads: 4,
                layers: 4,
                lr: 3e-4,
                warmup: 200,
                steps: 3_000,
                report_every: 100,
                checkpoint_every: 500,
            }
        } else {
            Config {
                context: 32,
                batch: 16,
                width: 64,
                heads: 4,
                layers: 2,
                lr: 1e-3,
                warmup: 100,
                steps: 1_500,
                report_every: 100,
                checkpoint_every: 500,
            }
        }
    }
}
