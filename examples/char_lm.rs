//! Character-level language model (GPT-style).
//!
//! Trains a causal transformer on a text corpus and generates new text
//! one character at a time.  The model and training schedule scale
//! automatically with corpus size:
//!
//!   tiny   (<20K chars) — embedded Shakespeare excerpts, ~100K params,  CPU ok
//!   medium (20K–200K)   — your own file,                ~500K params
//!   large  (>200K chars)— tiny-shakespeare or bigger,   ~2M params, GPU needed
//!
//! Usage:
//!   # embedded corpus (for demo / quick test):
//!   cargo run --example char_lm --release
//!
//!   # external file (recommended for real quality):
//!   cargo run --example char_lm --release -- path/to/corpus.txt
//!
//!   # get tiny-shakespeare (~1 MB, public domain):
//!   curl -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
//!   cargo run --example char_lm --release -- shakespeare.txt

use fastnn::prelude::*;
use fastnn::autograd::graph;
use fastnn::nn::{MultiHeadAttention, LayerNorm};

// Fixed training knobs (not corpus-dependent).
const LR: f32 = 3e-4;
const GRAD_CLIP: f32 = 1.0;
const GENERATE_CHARS: usize = 200;
const TEMPERATURE: f32 = 0.8;

/// All hyperparameters in one place — chosen based on corpus size at runtime.
struct Config {
    block_size:   usize,
    batch_size:   usize,
    n_embd:       usize,
    n_head:       usize,
    n_layer:      usize,
    warmup_steps: usize,
    max_steps:    usize,
    eval_every:   usize,
}

impl Config {
    fn for_corpus(chars: usize) -> Self {
        // All tiers run on CPU via FastNN's Rust backend (rayon-parallelised).
        // The `cuda` feature compiles CUDA kernels but tensors must be explicitly
        // moved with .cuda() for GPU execution — that's a future improvement.
        if chars > 200_000 {
            // Large corpus (tiny-shakespeare ~1M chars).
            // ~200K params. Shows word patterns after ~500 steps (~10 min on CPU).
            Config { block_size: 128, batch_size: 32, n_embd: 256, n_head: 8,
                     n_layer: 6, warmup_steps: 400, max_steps: 5_000, eval_every: 250 }
        } else if chars > 20_000 {
            // Medium corpus.
            Config { block_size: 128, batch_size: 32, n_embd: 256, n_head: 8,
                     n_layer: 6, warmup_steps: 200, max_steps: 4_000, eval_every: 250 }
        } else {
            // Tiny embedded corpus. ~100K params, finishes in a few minutes.
            // Output will look rough — not enough data for coherent text.
            Config { block_size: 32, batch_size: 8, n_embd: 64, n_head: 2,
                     n_layer: 2, warmup_steps: 200, max_steps: 3_000, eval_every: 300 }
        }
    }
}

// ── Corpus ───────────────────────────────────────────────────────────────────

const CORPUS: &str = "\
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to. To die, to sleep;
To sleep, perchance to dream. Ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action.

ALL THE WORLD'S A STAGE

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slippered pantaloon,
With spectacles on nose and pouch on side,
His youthful hose, well saved, a world too wide
For his shrunk shank; and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

TOMORROW AND TOMORROW

Tomorrow and tomorrow and tomorrow
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

WHAT'S IN A NAME

What's in a name? That which we call a rose
By any other name would smell as sweet.
So Romeo would, were he not Romeo called,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.

THE QUALITY OF MERCY

The quality of mercy is not strained.
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest:
It blesseth him that gives and him that takes.
'Tis mightiest in the mightiest; it becomes
The throned monarch better than his crown.
His sceptre shows the force of temporal power,
The attribute to awe and majesty,
Wherein doth sit the dread and fear of kings;
But mercy is above this sceptred sway.
It is enthroned in the hearts of kings;
It is an attribute to God himself;
And earthly power doth then show likest God's
When mercy seasons justice.

WE ARE SUCH STUFF

We are such stuff as dreams are made on,
And our little life is rounded with a sleep.
Be cheerful, sir: our revels now are ended.
These our actors, as I foretold you, were all spirits,
And are melted into air, into thin air;
And like the baseless fabric of this vision,
The cloud-capped towers, the gorgeous palaces,
The solemn temples, the great globe itself,
Yea, all which it inherit, shall dissolve,
And like this insubstantial pageant faded,
Leave not a rack behind.

GOOD NAME

Good name in man and woman, dear my lord,
Is the immediate jewel of their souls.
Who steals my purse steals trash;
It was mine, it is his, and has been slave to thousands.
But he that filches from me my good name
Robs me of that which not enriches him
And makes me poor indeed.

FRIENDS ROMANS COUNTRYMEN

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answered it.
Here, under leave of Brutus and the rest,
For Brutus is an honourable man;
So are they all, all honourable men,
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.
";

// ── Vocabulary ───────────────────────────────────────────────────────────────

struct CharVocab {
    chars: Vec<char>,
    char_to_idx: std::collections::HashMap<char, usize>,
}

impl CharVocab {
    fn build(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        chars.sort();
        let char_to_idx = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharVocab { chars, char_to_idx }
    }

    fn size(&self) -> usize { self.chars.len() }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.char_to_idx[&c]).collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }
}

// ── GPT Block ────────────────────────────────────────────────────────────────

struct GPTBlock {
    attn: MultiHeadAttention,
    ffn1: Linear,
    ffn2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl GPTBlock {
    fn new(n_embd: usize, n_head: usize, d_ff: usize) -> Self {
        GPTBlock {
            attn: MultiHeadAttention::new(n_embd, n_head, 0.0),
            ffn1: Linear::new(n_embd, d_ff),
            ffn2: Linear::new(d_ff, n_embd),
            norm1: LayerNorm::new(&[n_embd]),
            norm2: LayerNorm::new(&[n_embd]),
        }
    }

    // Pre-LN causal transformer block.
    fn forward(&self, x: &Tensor) -> Tensor {
        let normed = self.norm1.forward(x);
        let attn_out = self.attn.forward_attn(&normed, &normed, &normed, true);
        let x = x.add(&attn_out);

        let normed2 = self.norm2.forward(&x);
        let ff = self.ffn2.forward(&self.ffn1.forward(&normed2).gelu());
        x.add(&ff)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = Vec::new();
        p.extend(self.attn.q_proj.parameters_mut());
        p.extend(self.attn.k_proj.parameters_mut());
        p.extend(self.attn.v_proj.parameters_mut());
        p.extend(self.attn.out_proj.parameters_mut());
        p.extend(self.ffn1.parameters_mut());
        p.extend(self.ffn2.parameters_mut());
        p.push(&mut self.norm1.gamma);
        p.push(&mut self.norm1.beta);
        p.push(&mut self.norm2.gamma);
        p.push(&mut self.norm2.beta);
        p
    }
}

// ── CharGPT ──────────────────────────────────────────────────────────────────

struct CharGPT {
    tok_emb: Embedding,
    pos_emb: Embedding,  // learned positional embedding
    blocks: Vec<GPTBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    block_size: usize,
    n_embd: usize,
}

impl CharGPT {
    fn new(vocab_size: usize, block_size: usize, n_embd: usize, n_head: usize, n_layer: usize, d_ff: usize) -> Self {
        CharGPT {
            tok_emb: Embedding::new(vocab_size, n_embd),
            pos_emb: Embedding::new(block_size, n_embd),
            blocks: (0..n_layer).map(|_| GPTBlock::new(n_embd, n_head, d_ff)).collect(),
            ln_f: LayerNorm::new(&[n_embd]),
            lm_head: Linear::no_bias(n_embd, vocab_size),
            block_size,
            n_embd,
        }
    }

    /// Forward pass. `tokens` is flat [batch * seq_len] of indices.
    /// Returns logits of shape [batch * seq_len, vocab_size].
    fn forward(&self, tokens: &[usize], seq_len: usize, batch: usize) -> Tensor {
        // Token embeddings: [batch*seq, n_embd]
        let tok = self.tok_emb.forward_indices(tokens);

        // Position embeddings: [seq, n_embd] broadcast-added
        let pos_indices: Vec<usize> = (0..seq_len).collect();
        let pos = self.pos_emb.forward_indices(&pos_indices);
        // Expand pos to [batch, seq, n_embd] then reshape to match tok
        let pos_exp = pos
            .reshape(&[1, seq_len as i64, self.n_embd as i64])
            .expand(&[batch, seq_len, self.n_embd])
            .reshape(&[(batch * seq_len) as i64, self.n_embd as i64]);

        // Combine and reshape to [batch, seq, n_embd] for attention
        let mut x = tok.add(&pos_exp)
            .reshape(&[batch as i64, seq_len as i64, self.n_embd as i64]);

        for block in &self.blocks {
            x = block.forward(&x);
        }

        let x = self.ln_f.forward(&x);
        // Flatten to [batch*seq, n_embd] for the linear head
        let x_flat = x.reshape(&[(batch * seq_len) as i64, self.n_embd as i64]);
        self.lm_head.forward(&x_flat)  // [batch*seq, vocab_size]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p = Vec::new();
        p.push(&mut self.tok_emb.weight);
        p.push(&mut self.pos_emb.weight);
        for block in &mut self.blocks {
            p.extend(block.parameters_mut());
        }
        p.push(&mut self.ln_f.gamma);
        p.push(&mut self.ln_f.beta);
        p.extend(self.lm_head.parameters_mut());
        p
    }

    fn num_params(&self) -> usize {
        self.parameters_mut_count()
    }

    fn to_cuda(&mut self) {
        let device = Device::Cuda(0);
        self.tok_emb.to_device(device);
        self.pos_emb.to_device(device);
        for block in &mut self.blocks {
            block.attn.q_proj.to_device(device);
            block.attn.k_proj.to_device(device);
            block.attn.v_proj.to_device(device);
            block.attn.out_proj.to_device(device);
            block.ffn1.to_device(device);
            block.ffn2.to_device(device);
            block.norm1.to_device(device);
            block.norm2.to_device(device);
        }
        self.ln_f.to_device(device);
        self.lm_head.to_device(device);
    }

    fn parameters_mut_count(&self) -> usize {
        // Count without borrowing mutably — just sum sizes
        let emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel();
        let blocks: usize = self.blocks.iter().map(|b| {
            b.attn.q_proj.weight.numel() + b.attn.q_proj.bias.as_ref().map_or(0, |x| x.numel())
            + b.attn.k_proj.weight.numel() + b.attn.k_proj.bias.as_ref().map_or(0, |x| x.numel())
            + b.attn.v_proj.weight.numel() + b.attn.v_proj.bias.as_ref().map_or(0, |x| x.numel())
            + b.attn.out_proj.weight.numel() + b.attn.out_proj.bias.as_ref().map_or(0, |x| x.numel())
            + b.ffn1.weight.numel() + b.ffn1.bias.as_ref().map_or(0, |x| x.numel())
            + b.ffn2.weight.numel() + b.ffn2.bias.as_ref().map_or(0, |x| x.numel())
            + b.norm1.gamma.numel() + b.norm1.beta.numel()
            + b.norm2.gamma.numel() + b.norm2.beta.numel()
        }).sum();
        let head = self.lm_head.weight.numel()
            + self.ln_f.gamma.numel() + self.ln_f.beta.numel();
        emb + blocks + head
    }
}

// ── Training helpers ─────────────────────────────────────────────────────────

/// Sample a random batch of (input, target) token sequences.
fn sample_batch(data: &[usize], block_size: usize, batch_size: usize) -> (Vec<usize>, Vec<usize>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let max_start = data.len() - block_size - 1;
    let mut inputs = Vec::with_capacity(batch_size * block_size);
    let mut targets = Vec::with_capacity(batch_size * block_size);
    for _ in 0..batch_size {
        let start = rng.gen_range(0..max_start);
        inputs.extend_from_slice(&data[start..start + block_size]);
        targets.extend_from_slice(&data[start + 1..start + block_size + 1]);
    }
    (inputs, targets)
}

/// Temperature-based character sampling from logits [vocab_size].
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    use rand::Rng;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits.iter().map(|&x| ((x - max) / temperature).exp()).collect();
    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|&x| x / sum).collect();

    let r: f32 = rand::thread_rng().gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum { return i; }
    }
    probs.len() - 1
}

/// Auto-regressive text generation (no grad).
fn generate(model: &CharGPT, vocab: &CharVocab, prompt: &str, max_new: usize, temp: f32) -> String {
    let mut ctx: Vec<usize> = vocab.encode(prompt);
    for _ in 0..max_new {
        let start = if ctx.len() > model.block_size { ctx.len() - model.block_size } else { 0 };
        let window = &ctx[start..];
        let seq_len = window.len();
        let logits = model.forward(window, seq_len, 1); // [seq*1, vocab]
        let logit_data = logits.to_vec();
        // Take logits at the last position: [(seq-1)*vocab .. seq*vocab]
        let last = &logit_data[(seq_len - 1) * vocab.size()..seq_len * vocab.size()];
        let next = sample_token(last, temp);
        ctx.push(next);
    }
    vocab.decode(&ctx[vocab.encode(prompt).len()..])
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("FastNN — Character Language Model (GPT-style)");
    println!("===============================================\n");

    // ── Load corpus ──
    let args: Vec<String> = std::env::args().collect();
    let (corpus_text, from_file) = if args.len() > 1 {
        let path = &args[1];
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("Cannot read {:?}: {}", path, e); std::process::exit(1); });
        (text, true)
    } else {
        (CORPUS.to_string(), false)
    };

    if !from_file {
        println!("Using embedded corpus ({} chars).", corpus_text.len());
        println!("For coherent output, use a larger text file:");
        println!("  curl -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
        println!("  cargo run --example char_lm --release -- shakespeare.txt\n");
    }

    let cfg = Config::for_corpus(corpus_text.len());

    manual_seed(42);

    let vocab = CharVocab::build(&corpus_text);
    let data = vocab.encode(&corpus_text);

    // Pick a seed prompt from the start of the corpus.
    let seed_end = corpus_text.char_indices().nth(40).map(|(i, _)| i).unwrap_or(corpus_text.len());
    let seed = corpus_text[..seed_end].lines().next().unwrap_or("To be or not to be");

    println!("Corpus : {} chars, {} unique characters", corpus_text.len(), vocab.size());
    println!("Config : block={}, embd={}, heads={}, layers={}, batch={}",
        cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.batch_size);

    let d_ff = cfg.n_embd * 4;
    let mut model = CharGPT::new(vocab.size(), cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer, d_ff);
    println!("Params : ~{}", model.num_params());

    // Move to GPU when CUDA is enabled (initialise cuBLAS context first).
    #[cfg(feature = "cuda")]
    {
        let _ctx = CudaContext::new(0).expect("Failed to initialise CUDA device 0");
        println!("GPU    : CUDA initialised, moving model to CUDA:0...");
        model.to_cuda();
        println!("GPU    : done\n");
    }
    #[cfg(not(feature = "cuda"))]
    println!("Device : CPU\n");

    println!("Steps  : {} (eval every {})", cfg.max_steps, cfg.eval_every);

    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = AdamW::new(LR).weight_decay(0.1).betas(0.9, 0.95);

    // ── Sample before training ──
    println!("--- Before training (random noise expected) ---");
    println!("Prompt : {:?}", seed);
    println!("Output : {}\n", generate(&model, &vocab, seed, GENERATE_CHARS, TEMPERATURE));

    // ── Training loop ──
    let t0 = std::time::Instant::now();

    for step in 0..=cfg.max_steps {
        // Linear LR warmup
        let lr = if step < cfg.warmup_steps {
            LR * (step + 1) as f32 / cfg.warmup_steps as f32
        } else {
            LR
        };
        optimizer.set_lr(lr);

        let (inputs, targets) = sample_batch(&data, cfg.block_size, cfg.batch_size);

        // Forward + backward
        graph::enable_grad();
        {
            let mut params = model.parameters_mut();
            optimizer.zero_grad(&mut params);
        }

        let logits = model.forward(&inputs, cfg.block_size, cfg.batch_size);
        let loss = loss_fn.forward(&logits, &targets);
        loss.backward();
        graph::disable_grad();

        // Clip gradients and step
        {
            let mut params = model.parameters_mut();
            let norm = clip_grad_norm(&params, GRAD_CLIP);
            optimizer.step(&mut params);

            if step % cfg.eval_every == 0 {
                let elapsed = t0.elapsed().as_secs_f32();
                let steps_per_sec = if elapsed > 0.0 { step as f32 / elapsed } else { 0.0 };
                println!("step {:5}/{} | loss {:.4} | grad_norm {:.3} | {:.1} steps/s",
                    step, cfg.max_steps, loss.item(), norm, steps_per_sec);
            }
        }

        // Generate a sample
        if step > 0 && step % cfg.eval_every == 0 {
            println!("--- Sample (step {step}) ---");
            println!("{}\n", generate(&model, &vocab, seed, GENERATE_CHARS, TEMPERATURE));
        }
    }

    // ── Save checkpoint ──
    println!("Saving checkpoint to char_lm.fdl...");
    let named: std::collections::HashMap<String, Tensor> = {
        let mut m = std::collections::HashMap::new();
        m.insert("tok_emb".to_string(), model.tok_emb.weight.clone());
        m.insert("pos_emb".to_string(), model.pos_emb.weight.clone());
        m
    };
    if let Err(e) = save_tensors(&named, "char_lm.fdl") {
        eprintln!("Checkpoint save failed: {e}");
    } else {
        println!("Checkpoint saved.");
    }

    // ── Final generation ──
    println!("\n--- Final generation ---");
    // Use prompts from the actual corpus so the tokens are in-vocabulary.
    let prompts: Vec<&str> = corpus_text.lines()
        .filter(|l| l.len() >= 6)
        .take(3)
        .collect();
    for prompt in &prompts {
        let p = if prompt.len() > 30 { &prompt[..30] } else { prompt };
        println!("\nPrompt : {:?}", p);
        println!("{}", generate(&model, &vocab, p, GENERATE_CHARS, TEMPERATURE));
    }
    println!("\nTraining complete. Total time: {:.1}s", t0.elapsed().as_secs_f32());
}
