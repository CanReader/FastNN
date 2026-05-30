<div align="center">

# ⚡ FastNN

**A GPU-accelerated deep learning framework built from scratch in Rust and CUDA.**

Tensors, reverse-mode autograd, neural-network layers, optimizers, and data loading — no wrappers around PyTorch or TensorFlow, just hand-written math and CUDA kernels behind a clean, PyTorch-like API.

![Rust](https://img.shields.io/badge/Rust-2021-000000?logo=rust&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12%2F13-76B900?logo=nvidia&logoColor=white)
![Backend](https://img.shields.io/badge/backend-CPU%20%2B%20CUDA-blue)
![Tests](https://img.shields.io/badge/gradient--checked-41%20ops%20%E2%9C%93-success)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Highlights

- **Dual CPU / CUDA backend.** Every tensor op has a pure-Rust path and a CUDA path; dispatch is automatic based on where the tensor lives. Develop on CPU, train on GPU, no code changes.
- **Reverse-mode autograd.** A tape-based engine records every operation and replays it backward — the same model definition trains end to end.
- **Gradient-checked correctness.** Every backward pass is verified against finite differences (41 ops) and every CUDA kernel against its CPU reference (14 parity checks). See [Testing](#testing).
- **Real models train.** Ships with working MLP, CNN (Conv + BatchNorm + MaxPool), and a GPT-style character language model with multi-head causal attention.
- **cuBLAS-backed GEMM** with TF32 tensor cores, a stream-ordered buffer cache, and transpose-fused matmul so the backward pass allocates nothing extra.
- **No unsafe surprises.** GPU memory is RAII-managed (`CudaBuffer` frees on drop); CPU ops are Rayon-parallel.

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
fastnn = { path = "." }
```

```bash
# CPU-only — no CUDA toolkit required, ideal for development
cargo build --release --no-default-features

# Full GPU build
cargo build --release
```

```rust
use fastnn::prelude::*;
use fastnn::autograd::graph;

// Define a model by composing Modules.
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Dropout::new(0.2))
    .add(Linear::new(256, 10));

let loss_fn = CrossEntropyLoss::new();
let mut optimizer = Adam::new(1e-3);

// One training step.
graph::enable_grad();
{
    let mut params = model.parameters_mut();
    optimizer.zero_grad(&mut params);
}

let logits = model.forward(&inputs);          // [batch, 10]
let loss = loss_fn.forward(&logits, &targets); // scalar
loss.backward();                               // fills every .grad()

{
    let mut params = model.parameters_mut();
    optimizer.step(&mut params);               // updates weights in place
}
graph::disable_grad();
```

Move a model to the GPU by transferring its tensors — the same `forward`/`backward` code then runs on CUDA:

```rust
let x_gpu = x.cuda();      // upload to VRAM
let y = model.forward(&x_gpu);
let y_cpu = y.cpu();       // bring the result back
```

---

## Examples

```bash
# XOR with a small MLP — fastest way to see the training loop
cargo run --example simple_mlp --no-default-features --release

# MNIST digit classifier (MLP)
cargo run --example mnist_mlp  --no-default-features --release

# MNIST CNN: Conv2d + BatchNorm + ReLU + MaxPool + Linear head
cargo run --example mnist_cnn  --no-default-features --release

# GPT-style character language model: causal attention, gradient
# clipping, LR warmup, and autoregressive text generation
cargo run --example char_lm    --no-default-features --release
```

`char_lm` trains a ~4.8M-parameter Transformer (block 128, 256-dim, 8 heads, 6 layers) on the
tiny-Shakespeare corpus. On an RTX 4050 Laptop GPU it runs at **~9 training steps/sec** and drives
loss from 5.4 to ~1.3, producing recognizable Shakespearean prose. Pass a text file as the first
argument to train on your own corpus.

> `simple_mlp`, `mnist_mlp`, `mnist_cnn`, and `char_lm` are the maintained, end-to-end-trainable
> references. `mnist.rs` and `transformer.rs` are older forward-only sketches kept for reference.

---

## What's Included

### Layers (`Module` trait)

| Category | Layers |
|---|---|
| Core | `Linear`, `Conv2d`, `Flatten`, `Sequential` |
| Recurrent | `LSTM`, `GRU` |
| Attention | `MultiHeadAttention`, `TransformerEncoderLayer`, `TransformerEncoder` |
| Embeddings | `Embedding`, `PositionalEncoding` |
| Normalization | `BatchNorm2d`, `LayerNorm`, `RMSNorm` |
| Pooling | `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` |
| Regularization | `Dropout` |

### Activations, Losses, Optimizers, Schedulers

| | |
|---|---|
| **Activations** | `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Softmax`, `log_softmax` |
| **Losses** | `CrossEntropyLoss`, `MSELoss`, `BCELoss`, `BCEWithLogitsLoss` |
| **Optimizers** | `SGD` (+ momentum/Nesterov), `Adam`, `AdamW`; plus `clip_grad_norm` |
| **Schedulers** | `StepLR`, `CosineAnnealingLR`, `LinearWarmup`, `OneCycleLR` |

### Tensor operations

| Category | Operations |
|---|---|
| Constructors | `zeros`, `ones`, `full`, `rand`, `randn`, `arange`, `from_vec`, `kaiming_uniform`, `xavier_uniform` |
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `abs`, `pow_scalar`, `add_scalar`, `mul_scalar` |
| Linear algebra | `matmul`, `matmul_nt` (A·Bᵀ), `matmul_tn` (Aᵀ·B), batched 3D matmul — all cuBLAS |
| Math | `exp`, `log`, `sqrt`, `clamp` |
| Shape | `reshape`, `flatten`, `transpose`, `permute`, `expand` |
| Reductions | `sum`, `mean`, `max_val`, `min_val`, `var`, `sum_axis`, `mean_axis`, `argmax` |
| Device | `cuda()`, `cpu()`, `to_device()`, `item()` |

Autograd is wired through every differentiable op above, so models built from `Tensor`/`Module`
calls train without any manual gradient code.

---

## Architecture

```
fastnn/
├── cuda/
│   ├── kernels.cu          # all GPU kernels (single translation unit)
│   ├── include/kernels.h   # C FFI surface
│   └── stubs.c             # link-time no-ops for CPU-only builds
├── src/
│   ├── tensor/             # Tensor type, ops (CPU/CUDA dispatch), FFI bindings
│   ├── autograd/           # tape-based graph + Variable + GradFn backward ops
│   ├── nn/                 # Module trait and all layers
│   ├── optim/              # SGD / Adam / AdamW + LR schedulers
│   ├── data/               # Dataset trait + batching/shuffling DataLoader
│   ├── cuda/               # CUDA context + RAII CudaBuffer (with buffer cache)
│   └── serialize/          # .fdl checkpoint format
├── tests/                  # gradcheck.rs, cuda_parity.rs
└── examples/
```

**Design notes**

- **Dual-backend dispatch.** A `Tensor` is either `TensorStorage::Cpu(Vec<f32>)` or
  `TensorStorage::Cuda(Arc<CudaBuffer>)`; each op picks the matching path automatically.
- **cuBLAS for GEMM.** Matrix multiply uses `Sgemm`/`SgemmStridedBatched` with TF32 tensor cores.
  The backward pass uses transpose-fused variants (`CUBLAS_OP_T`) so it never allocates transpose buffers.
- **Convolution = im2col + GEMM**, reusing the optimized matmul path.
- **Pre-LN Transformer**, the stable variant used by modern GPT-style models.
- **Stream-ordered buffer cache.** `CudaBuffer` recycles freed allocations by size, eliminating
  per-step `cudaMalloc` overhead during training.
- **Tape autograd.** Operations append `GradFn` nodes to a thread-local graph; `backward()` walks
  it in reverse, seeding the loss gradient on the loss tensor's own device.

---

## Testing

Correctness is enforced by two suites, not by "the loss went down."

```bash
# Finite-difference gradient checks for every op (CPU, no GPU needed)
cargo test --no-default-features --test gradcheck

# CUDA-vs-CPU forward + backward equivalence for custom kernels
cargo test --test cuda_parity -- --test-threads=1
```

- **`gradcheck.rs`** verifies each backward against the central finite difference
  `(L(x+h) − L(x−h)) / 2h` using a random upstream gradient. 41 checks, including the fused
  `CrossEntropy`, `MSE`, and `LayerNorm` paths.
- **`cuda_parity.rs`** runs each custom-kernel op (matmul variants, permute, sum-axis, LayerNorm,
  activations) on both backends and asserts the forward and gradients agree.

CI (`.github/workflows/ci.yml`) runs the CPU build and gradient checks on every push.

---

## CUDA Requirements

| Requirement | Version |
|---|---|
| NVIDIA CUDA Toolkit | 12.x or 13.x |
| GPU compute capability | 7.5+ |
| Architectures built | Turing (7.5), Ampere (8.0 / 8.6), Ada Lovelace (8.9), Hopper (9.0) |

`build.rs` compiles `cuda/kernels.cu` with `nvcc` and links `cudart`, `cublas`, and `curand`. Set
`CUDA_PATH` or `CUDA_HOME` if the toolkit is not in the default location. CPU-only builds
(`--no-default-features`) need no toolkit at all — `cuda/stubs.c` supplies link-time symbols.

> **Note:** under a very new host compiler (e.g. GCC 16), pin `nvcc`'s host compiler with
> `-ccbin=g++-15`; the build script already does this.

---

## Build & Test Reference

```bash
cargo build --release                      # GPU build
cargo build --release --no-default-features # CPU-only
cargo check  --no-default-features          # fast type-check
cargo test   --no-default-features          # unit + integration tests
cargo bench  --no-default-features          # criterion benchmarks
```

Always build non-trivial models with `--release`; debug mode is roughly 50× slower.

---

## Current Limitations

FastNN is a focused, from-scratch implementation. Known gaps:

- `LayerNorm` γ/β are trained on the **CUDA path only**; the CPU path propagates dL/dx but not the parameter gradients.
- `Cat`/`Stack` and `Squeeze`/`Unsqueeze` have no backward pass yet.
- `AdaptiveAvgPool2d` has no backward pass.
- Single-GPU only; no mixed precision (f16/bf16 is a placeholder feature flag).
- Several public APIs `panic!` on misuse rather than returning `Result`.

---

## Roadmap

- [ ] `Result`-based error handling across the public API
- [ ] LayerNorm γ/β gradients on the CPU path
- [ ] `Cat`/`Stack`/`Squeeze`/`Unsqueeze` backward
- [ ] Mixed-precision training (FP16 / BF16)
- [ ] Gradient checkpointing for deeper models
- [ ] Multi-GPU data parallelism
- [ ] Built-in dataset downloaders (MNIST, CIFAR-10)
- [ ] Flash-Attention-style fused attention

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes first.
New ops must ship with a gradient check in `tests/gradcheck.rs` (and a parity test if they add a CUDA kernel).

```bash
git checkout -b feature/my-feature
cargo test --no-default-features --test gradcheck   # must pass
```

---

## License

MIT — see [LICENSE](LICENSE).
