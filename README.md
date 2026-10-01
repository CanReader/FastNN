<div align="center">

# FastNN

**A deep learning library in Rust, with CUDA kernels for the parts that matter.**

[![CI](https://github.com/CanReader/FastNN/actions/workflows/ci.yml/badge.svg)](https://github.com/CanReader/FastNN/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/fastnn.svg)](https://crates.io/crates/fastnn)
[![docs.rs](https://img.shields.io/docsrs/fastnn)](https://docs.rs/fastnn)
[![MSRV](https://img.shields.io/badge/rustc-1.87+-blue.svg)](https://github.com/CanReader/FastNN/blob/master/Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quick start](#quick-start) ·
[Examples](#examples) ·
[API docs](https://docs.rs/fastnn) ·
[Releases](https://github.com/CanReader/FastNN/releases) ·
[Contributing](CONTRIBUTING.md)

</div>

FastNN is a from-scratch deep learning library: tensors with reverse-mode
autograd, the usual layers, transformers with KV-cached generation, modern
optimizers, data loading, resumable checkpoints, and safetensors interchange.
It runs on the CPU out of the box and on NVIDIA GPUs with one feature flag.

It is built to be read. Every file is one idea and none of them are long, so you
can follow a `loss.backward()` call from the training loop down to the kernel
and back without getting lost.

* **Readable.** 86 source files, about 12k lines. Forward ops and their backward
  rules live in mirrored files, one per category.
* **Verified.** Every backward rule is checked against finite differences of its
  own forward, and every CUDA kernel against the CPU path.
* **Built for real runs.** Exact resume including optimizer state, NaN tracing
  to the op that produced it, frozen layers that skip their gradient entirely.

## Contents

* [Features](#features)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Examples](#examples)
* [Platform support](#platform-support)
* [How it fits together](#how-it-fits-together)
* [Long training runs](#long-training-runs)
* [Extending FastNN](#extending-fastnn)
* [Error handling](#error-handling)
* [Testing](#testing)
* [CUDA](#cuda)
* [Project layout](#project-layout)
* [Limitations](#limitations)
* [Contributing](#contributing)
* [License](#license)

## Features

| Area | What is in it |
|---|---|
| **Tensors** | Broadcasting arithmetic, matmul (batched, transposed forms), reductions, indexing, views, `cat`/`stack`/`narrow`, CPU or CUDA storage |
| **Autograd** | Reverse mode with no tape and no global state, `no_grad`, `detect_anomaly`, gradient clipping by norm or value |
| **Layers** | `Linear`, `Conv1d`, `Conv2d` (dilation, groups, depthwise), `ConvTranspose1d/2d`, pooling, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `Dropout`, `Embedding`, `LSTM`, `GRU` |
| **Transformers** | `MultiHeadAttention` with causal and padding masks, encoder and decoder stacks, full encoder-decoder `Transformer` |
| **Generation** | `KvCache` for incremental decoding, `Sampler` (temperature, top-k, top-p, repetition penalty), `BeamSearch` |
| **Optimizers** | `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, `RAdam`, `Lion`, plus `Lookahead` and `Ema` wrappers |
| **Schedules** | `Warmup`, `CosineAnnealing`, `OneCycle`, `StepDecay`, `Constant` |
| **Losses** | Cross-entropy (label smoothing, class weights, ignore index), MSE, MAE, BCE, Huber, KL, NLL, focal, dice, Gaussian and Poisson NLL, triplet, contrastive, InfoNCE |
| **Data** | `Dataset` trait, shuffling `DataLoader` with device placement, built-in MNIST |
| **Checkpoints** | Native weights, full training state (weights + optimizer + step), safetensors read and write with F16/BF16 widening |

## Installation

```bash
cargo add fastnn                   # CPU only, no CUDA toolkit needed
cargo add fastnn --features cuda   # with the CUDA kernels
```

| Requirement | Version |
|---|---|
| Rust | 1.87 or newer |
| CUDA toolkit (only with `--features cuda`) | 11.8 or newer, for GPUs from Turing to Hopper (compute capability 7.5 to 9.0) |

## Quick start

A complete MNIST training loop:

```rust
use fastnn::prelude::*;
use fastnn::data::{Mnist, Split};

let train = Mnist::load(Split::Train)?;
let loader = DataLoader::new(&train, 128).shuffle(true);

let model = Sequential::new()
    .add(Flatten::new())
    .add(Linear::new(784, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

let mut opt = Adam::new(model.parameters(), 1e-3);

for batch in loader.iter() {
    let loss = cross_entropy(&model.forward(&batch.inputs), &batch.labels());

    opt.zero_grad();
    loss.backward();
    opt.step();
}

save(&model, "mnist.fdl")?;
```

No gradient-mode flags, no borrow dance around the optimizer. That is the whole
training loop. To train on a GPU, move the model and the loader:

```rust
let device = Device::best();       // GPU if there is one, else CPU
model.to_device(device);
let loader = DataLoader::new(&train, 128).to_device(device);
```

## Examples

```bash
cargo run --release --example simple_mlp
cargo run --release --example mnist_cnn --features cuda
```

Always use `--release`. Debug builds are roughly 50x slower.

| Example | What it shows |
|---|---|
| `simple_mlp` | The smallest complete training loop, an MLP learning XOR |
| `mnist_mlp` | Dataset download, batching, evaluation, checkpointing |
| `mnist_cnn` | Convolutions, batch norm, pooling, an LR schedule |
| `char_lm` | A small GPT trained from scratch that generates text, with resume on Ctrl-C |
| `finetune` | Pretrain, freeze the backbone, train a new head |
| `gan` | Two networks trained against each other |
| `vae` | A variational autoencoder on 2-D data |
| `reinforce` | Policy-gradient reinforcement learning on a bandit |
| `multi_task` | One shared trunk, two heads, one combined loss |

`char_lm` trains on a small embedded corpus by default, or on any text file:

```bash
curl -o shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cargo run --release --example char_lm -- shakespeare.txt
```

### Prebuilt binaries

Every [release](https://github.com/CanReader/FastNN/releases) ships all of the
examples prebuilt, so you can try FastNN without installing Rust. Download the
archive for your platform, extract it, and run any example directly.

| Platform | Archive |
|---|---|
| Linux x86_64 | `fastnn-<version>-x86_64-linux.tar.gz` |
| Linux ARM64 | `fastnn-<version>-aarch64-linux.tar.gz` |
| Linux x86_64 with CUDA | `fastnn-<version>-x86_64-linux-cuda.tar.gz` |
| macOS Apple Silicon | `fastnn-<version>-aarch64-macos.tar.gz` |
| macOS Intel | `fastnn-<version>-x86_64-macos.tar.gz` |
| Windows x86_64 | `fastnn-<version>-x86_64-windows.zip` |

Each release includes a `SHA256SUMS` file and a build provenance attestation
for every archive, which you can check with
`gh attestation verify <archive> --repo CanReader/FastNN`. The CUDA build links
against the CUDA 13 runtime, so it needs that installed.

## Platform support

| Platform | CPU | CUDA |
|---|---|---|
| Linux x86_64 | Tested in CI | Built in CI, parity tested on real hardware |
| Linux ARM64 | Built for releases | Not supported |
| macOS (Intel, Apple Silicon) | Tested in CI | Not available on macOS |
| Windows x86_64 | Tested in CI | Should build, not tested |

## How it fits together

```
  data      Dataset ─► DataLoader ─► Batch
                                       │
  nn        Module ──► forward ────────┤    layers hold Param handles
                                       ▼
  tensor    Tensor ops on CPU or CUDA ─┴─► loss
                                       │
  autograd  loss.backward() walks the graph the ops built,
            depositing gradients in each Param's slot
                                       │
  optim     opt.step() reads those slots and updates in place
```

Everything starts on the CPU. `model.to_device(device)` moves a model,
`DataLoader::to_device` moves its batches, and mixing devices panics rather
than copying behind your back.

Two design decisions shape the rest of the library.

**The graph is the tensors.** There is no tape and no global state. A tensor
produced by a differentiable op carries a node naming the rule that made it and
the tensors it consumed, so `loss.backward()` just walks that structure. Drop the
loss and the graph frees itself. A tensor with no node is plain data, so there is
no `requires_grad` flag to keep in sync. Wrap inference in `no_grad(|| ...)` to
skip building a graph at all.

**Parameters are shared slots.** `Param` is a handle, not a value. The model and
the optimizer hold handles to the same weight and the same gradient, which is why
`opt.step()` needs no borrow of the model. It is also why weight tying works
without special support: use the same handle twice and both paths' gradients add
up on their own.

## Long training runs

Four things a long, unattended run needs that the quick start does not show.

**Resume exactly where you stopped.** `save()` writes weights, which is right for
a finished model. Resuming also needs the optimizer: momentum, Adam's moment
estimates, and the step count that drives bias correction. Restore weights alone
and the optimizer restarts at step 1, so the first update after resuming lands
far harder than it should.

```rust
let mut step = load_training(&model, &mut opt, "run.fdl").unwrap_or(0);

while step < total {
    // ... train ...
    step += 1;
    if step % 500 == 0 { save_training(&model, &opt, step, "run.fdl")?; }
}
```

Loading a plain checkpoint with `load_training` is an error, not a silent
optimizer reset.

**Find the NaN at the op that made it.** One non-finite gradient becomes a
non-finite weight, and every activation downstream is NaN from then on. By the
time the loss prints NaN the checkpoint is poisoned and the culprit is long gone.

```rust
detect_anomaly(|| {
    loss.backward();     // panics: "anomaly: Log produced inf in the gradient for input 0"
});

if !opt.gradients_are_finite() { continue; }   // cheap guard: skip the batch
```

Both read every gradient, so they are debugging and guard-rail tools, not
something to leave on in a healthy loop.

**Freeze a backbone.** A frozen parameter hands out a detached tensor, so the
backward pass stops there and no gradient is computed only to be thrown away.

```rust
backbone.freeze();
let mut opt = Adam::new(model.parameters(), 1e-3);   // updates only the head
```

**Report a bad shape instead of crashing.** Ordinary ops panic, which is right
inside a model. At the edge of an application, where a shape comes from a config
file or an upload, use the `try_` variants.

```rust
let y = x.try_matmul(&w)?;      // Error::Shape, not a panic
x.try_reshape(&[2, -1])?;
table.try_index_select(&ids)?;
```

They validate and then delegate, so there is still exactly one implementation of
each operation, and they build the same graph.

## Extending FastNN

**A new layer.** Implement `forward`, plus `named_parameters` if it has weights.
Device placement, parameter counting, and gradient zeroing all come from those.

```rust
struct Residual { inner: Linear }

impl Module for Residual {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.add(&self.inner.forward(x).relu())
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        scoped("inner", self.inner.named_parameters())
    }
}
```

**A new op.** Write the forward in `tensor/ops/`, the backward rule in
`autograd/ops/`, attach it with `with_grad`, and add a case to
`tests/gradcheck.rs`.

```rust
// tensor/ops/unary.rs
pub fn softplus(&self) -> Tensor {
    unary_op(self, |x| x.exp().ln_1p(), ffi::fastnn_cuda_softplus)
        .with_grad(&[self], || SoftplusBackward { input: self.detach() })
}

// autograd/ops/unary.rs
impl Backward for SoftplusBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![elementwise(grad, &self.input, |x| 1.0 / (1.0 + (-x).exp()))]
    }
    fn name(&self) -> &'static str { "Softplus" }
}
```

The rule returns gradients in the same order as the inputs. Save detached
tensors: a saved value that still carries its own node would pin the graph that
produced it.

**A new dataset.** Implement `Dataset` and hand it to a `DataLoader`. Items are
written into caller-provided slices rather than returned as tensors, so batching
60,000 images does not build 60,000 throwaway ones.

[CONTRIBUTING.md](CONTRIBUTING.md) has the full checklist for adding an op.

## Error handling

Tensor maths panics on shape and device mistakes. Those are bugs in the calling
code, like indexing past the end of a slice, and threading `Result` through every
`add` would bury the model in `?` for no safety gained.

`Result` is for what genuinely fails at runtime:

```rust
let device = Device::cuda(0)?;             // no GPU
let data   = Mnist::load(Split::Train)?;   // download or parse failed
load(&model, "model.fdl")?;                // missing file, or a shape that moved
```

## Testing

```bash
cargo test                     # everything
cargo test --test gradcheck    # every backward rule vs finite differences
cargo test --test robustness   # resume, anomalies, freezing
cargo test --test generation   # kv-cached decoding, masks, beam search, sampling
cargo test --test convergence  # every layer family and optimizer actually learns
cargo test --test losses       # loss values against hand-computed references
cargo test --features cuda --test cuda_parity -- --test-threads=1   # CPU vs GPU
cargo bench                    # throughput
```

`gradcheck` compares every backward rule with central finite differences of its
own forward. It is why the autograd layer can be trusted, and a new rule without
a case there is not finished.

`cuda_parity` runs each op on both devices and compares the results. It skips
itself when there is no GPU, so a CPU-only machine still gets a green run. It has
already caught a real bug: the softmax kernel's block reduction assumed a
power-of-two thread count and silently dropped the tail of every row whose width
was not one, which included MNIST's ten classes.

Every pull request runs formatting, clippy, the full suite on Linux, macOS, and
Windows, an MSRV build, docs, packaging, and a CUDA compile. A nightly job reruns
everything in release mode on stable and beta and trains the examples end to end.

## CUDA

The `cuda` feature is opt-in, so a plain `cargo add fastnn` never needs the
toolkit. With the feature on, `build.rs` compiles `cuda/kernels.cu` with `nvcc`
and links `cudart`, `cublas`, and `curand`.

* Set `CUDA_PATH` or `CUDA_HOME` if the toolkit is not in the default location.
* If `nvcc` rejects your system compiler as too new, point `FASTNN_NVCC_CCBIN`
  at one it accepts.
* Without the feature, `cuda/stubs.c` supplies the symbols, `Device::cuda(0)`
  returns an error, and everything runs on the CPU.

Two things carry most of the GPU performance. Matrix multiplication goes through
cuBLAS, and the two transposed forms the backward pass needs (`matmul_nt`,
`matmul_tn`) use a transpose flag instead of building a transposed copy. GPU
allocations come from a size-keyed free list, so the many short-lived temporaries
a training step creates cost a hash lookup instead of a driver round trip.

## Project layout

```
src/
  tensor/          the array type and everything you can do to it
    core.rs          Tensor: shape, storage, graph link
    checked.rs       try_* variants that report instead of panicking
    shape.rs         strides, broadcasting, index math
    storage.rs       the bytes, on one device or the other
    device.rs        Device::cuda(0) -> Result
    init.rs          zeros, randn, kaiming, xavier, ...
    ops/             arith  unary  activation  matmul  reduce  view
                     index  conv  pool  norm

  autograd/        reverse-mode differentiation
    node.rs          Backward trait, graph nodes, gradient slots
    engine.rs        the reverse pass
    mode.rs          no_grad
    anomaly.rs       detect_anomaly
    ops/             one backward rule per forward op, same file names

  nn/              layers, all implementing Module
    linear  conv  conv1d  conv_transpose  pooling  norm  activation
    dropout  embedding  shape  attention  transformer  decoder  rnn
    cache (kv cache)  sample (sampling)  beam (beam search)
    loss  metric_losses  module  param  sequential

  optim/           sgd  adam  rmsprop  adagrad  radam  lion  lookahead
                   ema  schedule  state
  data/            dataset  loader  mnist
  serialize/       checkpoint  training (weights + optimizer)
                   safetensors  half (F16/BF16 conversion)
  cuda/            ffi (raw bindings)  kernels (safe wrappers)  buffer

cuda/kernels.cu    every GPU kernel
```

`tensor/ops/` and `autograd/ops/` mirror each other file for file: the forward
for `relu` is in `tensor/ops/activation.rs` and its derivative is in
`autograd/ops/activation.rs`.

## Limitations

* Compute is `f32`. Checkpoints stored as F16 or BF16 load fine and are widened
  on read, but the maths runs in single precision.
* `LSTM` and `GRU` are built from ordinary differentiable ops, one graph node per
  gate per timestep. Correct, and fine for short sequences; use a transformer for
  long ones.
* Tensors are always contiguous. `permute` and `expand` write a new buffer rather
  than returning a strided view.

## Contributing

Bug reports, ideas, and pull requests are all welcome. [CONTRIBUTING.md](CONTRIBUTING.md)
covers the setup, the conventions, what CI checks, and how to add an op. Issues
labelled [`good first issue`](https://github.com/CanReader/FastNN/labels/good%20first%20issue)
are a good place to start.

Please report security problems privately, as described in [SECURITY.md](SECURITY.md).

## License

FastNN is released under the [MIT License](LICENSE).
