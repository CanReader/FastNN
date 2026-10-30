# FastNN

A robust deep learning library in Rust with CUDA kernels.

[![CI](https://github.com/CanReader/FastNN/actions/workflows/ci.yml/badge.svg)](https://github.com/CanReader/FastNN/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/fastnn.svg)](https://crates.io/crates/fastnn)
[![docs.rs](https://img.shields.io/docsrs/fastnn)](https://docs.rs/fastnn)
[![MSRV](https://img.shields.io/badge/rustc-1.87+-blue.svg)](https://github.com/CanReader/FastNN/blob/master/Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Tensors with reverse-mode autograd, the usual layers, transformers with
KV-cached generation, optimizers, data loading, resumable checkpoints and
safetensors. It runs on the CPU by default and on NVIDIA GPUs with the `cuda`
feature.

The codebase is small on purpose: 86 source files, about 12k lines, one idea
per file. You can follow `loss.backward()` from the training loop down to the
kernel. Every backward rule is checked against finite differences of its own
forward, and every CUDA kernel against the CPU path.

## Features

| Area | What is in it |
|---|---|
| Tensors | Broadcasting arithmetic, matmul (batched, transposed forms), reductions, indexing, views, `cat`/`stack`/`narrow`, CPU or CUDA storage |
| Autograd | Reverse mode, no tape, no global state. `no_grad`, `detect_anomaly`, gradient clipping by norm or value |
| Layers | `Linear`, `Conv1d`, `Conv2d` (dilation, groups, depthwise), `ConvTranspose1d/2d`, pooling, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `Dropout`, `Embedding`, `LSTM`, `GRU` |
| Transformers | `MultiHeadAttention` with causal and padding masks, encoder and decoder stacks, encoder-decoder `Transformer` |
| Generation | `KvCache`, `Sampler` (temperature, top-k, top-p, repetition penalty), `BeamSearch` |
| Optimizers | `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, `RAdam`, `Lion`, plus `Lookahead` and `Ema` wrappers |
| Schedules | `Warmup`, `CosineAnnealing`, `OneCycle`, `StepDecay`, `Constant` |
| Losses | Cross-entropy (label smoothing, class weights, ignore index), MSE, MAE, BCE, Huber, KL, NLL, focal, dice, Gaussian and Poisson NLL, triplet, contrastive, InfoNCE |
| Data | `Dataset` trait, shuffling `DataLoader` with device placement, MNIST |
| Checkpoints | Weights, full training state (weights, optimizer, step), safetensors read and write with F16/BF16 widening |

## Installation

```bash
cargo add fastnn                   # CPU only, no CUDA toolkit needed
cargo add fastnn --features cuda   # with the CUDA kernels
```

Needs Rust 1.87 or newer. With `--features cuda` you also need the CUDA toolkit,
11.8 or newer, for GPUs from Turing to Hopper (compute capability 7.5 to 9.0).

## Quick start

An MNIST training loop:

```rust
use fastnn::data::{Mnist, Split};
use fastnn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}
```

To train on a GPU, move the model and the loader:

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

Use `--release`. Debug builds are about 50x slower.

| Example | What it shows |
|---|---|
| `simple_mlp` | MLP learning XOR, the smallest training loop |
| `mnist_mlp` | Dataset download, batching, evaluation, checkpointing |
| `mnist_cnn` | Convolutions, batch norm, pooling, an LR schedule |
| `char_lm` | Small GPT trained from scratch, resumes after Ctrl-C |
| `finetune` | Pretrain, freeze the backbone, train a new head |
| `gan` | Two networks trained against each other |
| `vae` | Variational autoencoder on 2-D data |
| `reinforce` | Policy gradient on a bandit |
| `multi_task` | Shared trunk, two heads, one combined loss |

`char_lm` uses a small embedded corpus by default, or any text file:

```bash
curl -o shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cargo run --release --example char_lm -- shakespeare.txt
```

Want to see it without writing any code? [fastnn-visualizer](https://github.com/CanReader/fastnn-visualizer)
trains an MLP on MNIST with FastNN and lets you draw a digit in the browser
and watch the network light up as it predicts.

### Prebuilt binaries

Each [release](https://github.com/CanReader/FastNN/releases) has the examples
prebuilt, so you can try them without installing Rust.

| Platform | Archive |
|---|---|
| Linux x86_64 | `fastnn-<version>-x86_64-linux.tar.gz` |
| Linux ARM64 | `fastnn-<version>-aarch64-linux.tar.gz` |
| Linux x86_64 with CUDA | `fastnn-<version>-x86_64-linux-cuda.tar.gz` |
| macOS Apple Silicon | `fastnn-<version>-aarch64-macos.tar.gz` |
| macOS Intel | `fastnn-<version>-x86_64-macos.tar.gz` |
| Windows x86_64 | `fastnn-<version>-x86_64-windows.zip` |

Releases include a `SHA256SUMS` file and a build provenance attestation for each
archive (`gh attestation verify <archive> --repo CanReader/FastNN`). The CUDA
build needs an NVIDIA driver new enough for the toolkit it was built with.

## Platform support

| Platform | CPU | CUDA |
|---|---|---|
| Linux x86_64 | Tested in CI | Built in CI, parity tested on real hardware |
| Linux ARM64 | Built for releases | Not supported |
| macOS (Intel, Apple Silicon) | Tested in CI | Not available |
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

Everything starts on the CPU. `model.to_device(device)` and
`DataLoader::to_device` move things over, and mixing devices panics instead of
copying behind your back.

The graph is the tensors. A tensor produced by a differentiable op carries a
node naming the rule that made it and the tensors it consumed, and
`loss.backward()` walks that. Drop the loss and the graph frees itself. A tensor
with no node is plain data, so there is no `requires_grad` flag. Wrap inference
in `no_grad(|| ...)` to skip building a graph.

`Param` is a handle, not a value. The model and the optimizer hold handles to
the same weight and gradient, so `opt.step()` doesn't borrow the model. Weight
tying works the same way: use the handle twice and the gradients add up.

## Long training runs

**Resume.** `save()` writes weights, which is enough for a finished model.
Resuming also needs the optimizer's moments and step count. Without them Adam
restarts at step 1 and the first update is far too large.

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

**NaNs.** One non-finite gradient poisons the weights, and by the time the loss
prints NaN the culprit is long gone. `detect_anomaly` panics at the op that
produced it.

```rust
detect_anomaly(|| {
    loss.backward();     // panics: "anomaly: Log produced inf in the gradient for input 0"
});

if !opt.gradients_are_finite() { continue; }   // cheap guard: skip the batch
```

Both read every gradient, so don't leave them on in a healthy loop.

**Freezing.** A frozen parameter hands out a detached tensor, so the backward
pass stops there and no gradient is computed for it.

```rust
backbone.freeze();
let mut opt = Adam::new(model.parameters(), 1e-3);   // updates only the head
```

**Fallible ops.** Ordinary ops panic on bad shapes. Where a shape comes from a
config file or an upload, use the `try_` variants.

```rust
let y = x.try_matmul(&w)?;      // Error::Shape, not a panic
x.try_reshape(&[2, -1])?;
table.try_index_select(&ids)?;
```

They validate and then call the panicking op, so there is one implementation of
each operation.

## Extending

**A layer.** Implement `forward`, plus `named_parameters` if it has weights.
Device placement, parameter counting and gradient zeroing follow from those.

```rust
use fastnn::nn::module::scoped;

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

**An op.** Forward in `tensor/ops/`, backward rule in `autograd/ops/`, attach it
with `with_grad`, add a case to `tests/gradcheck.rs`.

```rust
// tensor/ops/unary.rs
pub fn sqrt(&self) -> Tensor {
    let out = unary_op(self, f32::sqrt, ffi::fastnn_cuda_sqrt);
    let saved = out.clone();
    out.with_grad(&[self], || SqrtBackward { output: saved })
}

// autograd/ops/unary.rs
impl Backward for SqrtBackward {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor> {
        vec![grad.div(&self.output.mul_scalar(2.0))]
    }
    fn name(&self) -> &'static str { "Sqrt" }
}
```

The rule returns gradients in input order. Save detached tensors, otherwise the
saved value pins the graph that produced it. [CONTRIBUTING.md](CONTRIBUTING.md)
has the full checklist.

**A dataset.** Implement `Dataset` and pass it to a `DataLoader`. Items are
written into caller-provided slices instead of returned as tensors, so batching
60,000 images doesn't allocate 60,000 throwaway ones.

## Errors

Shape and device mistakes panic. They are bugs in the calling code, like
indexing past the end of a slice. `Result` is for things that fail at runtime:

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

`cuda_parity` runs each op on both devices and compares. It skips itself when
there is no GPU. It has already caught a real bug: the softmax kernel's block
reduction assumed a power-of-two thread count and dropped the tail of every row
whose width wasn't one, which included MNIST's ten classes.

CI runs formatting, clippy, the suite on Linux, macOS and Windows, an MSRV
build, docs, packaging and a CUDA compile. A nightly job reruns the suite in
release on stable and beta and trains the examples that don't need a download.

## CUDA

The `cuda` feature is opt-in. With it on, `build.rs` compiles `cuda/kernels.cu`
with `nvcc` and links `cudart`, `cublas` and `curand`.

* `nvcc` is found on `PATH`, or set the `NVCC` environment variable.
* Headers and libraries come from `CUDA_PATH`, then `CUDA_HOME`, then the
  default install location. Make sure that is the same toolkit as your `nvcc`.
* If `nvcc` rejects your system compiler as too new, point `FASTNN_NVCC_CCBIN`
  at one it accepts.
* Without the feature, `cuda/stubs.c` supplies the symbols, `Device::cuda(0)`
  returns an error, and everything runs on the CPU.

Matmul goes through cuBLAS. The transposed forms the backward pass needs
(`matmul_nt`, `matmul_tn`) use a transpose flag instead of a transposed copy.
GPU allocations come from a size-keyed free list, so the short-lived temporaries
in a training step don't each hit the driver.

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
cuda/stubs.c       CPU-only stand-ins for the same symbols
```

`tensor/ops/` and `autograd/ops/` mirror each other file for file. The forward
for `relu` is in `tensor/ops/activation.rs`, its derivative in
`autograd/ops/activation.rs`.

## Limitations

* Version 0.x, so the API can still change between minor releases.
* Compute is `f32`. F16 and BF16 checkpoints load and are widened on read.
* `LSTM` and `GRU` are built from ordinary ops, one graph node per gate per
  timestep. Fine for short sequences, use a transformer for long ones.
* Tensors are always contiguous. `permute` and `expand` write a new buffer
  instead of returning a strided view.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues labelled
[`good first issue`](https://github.com/CanReader/FastNN/labels/good%20first%20issue)
are a good place to start. Report security problems privately, see
[SECURITY.md](SECURITY.md).

If FastNN is useful to you, a star helps other people find it.

## License

[MIT](LICENSE)
