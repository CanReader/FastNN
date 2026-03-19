<p align="center">
  <h1 align="center">⚡ fastDL</h1>
  <p align="center">
    <strong>A high-performance deep learning framework built from the ground up in Rust and CUDA.</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#benchmarks">Benchmarks</a> •
    <a href="#examples">Examples</a>
  </p>
</p>

---

fastDL is a GPU-accelerated deep learning library that provides a complete, production-oriented toolkit for building, training, and deploying neural networks. Written entirely from scratch in Rust with hand-tuned CUDA kernels, it combines the safety and expressiveness of Rust's type system with the raw computational power of NVIDIA GPUs.

Unlike wrapper libraries, fastDL owns the entire stack — from low-level GPU memory management and kernel dispatch to high-level abstractions like Transformer encoders and learning rate schedulers. Every operation supports both CPU and CUDA execution paths with automatic device-aware dispatch.

## Key Highlights

- **Zero-dependency GPU backend** — Custom CUDA kernels for every operation; no reliance on cuDNN or external neural network libraries
- **cuBLAS-accelerated linear algebra** — Matrix multiplications routed through cuBLAS SGEMM with TF32 tensor core acceleration
- **Dual-backend architecture** — Every tensor operation transparently dispatches to optimized CPU (pure Rust) or CUDA paths
- **Reverse-mode automatic differentiation** — Tape-based autograd engine with a dynamically constructed computation graph
- **Modern architecture support** — Pre-LN Transformer encoders, Multi-Head Attention with causal masking, RMSNorm, GELU/SiLU activations
- **Memory-safe GPU programming** — RAII-based GPU memory management through Rust's ownership model; no manual `cudaFree` calls
- **Compute compatibility 7.0–9.0** — Compiled for Volta, Turing, Ampere, Ada Lovelace, and Hopper architectures

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
  - [Tensor Engine](#tensor-engine)
  - [CUDA GPU Backend](#cuda-gpu-backend)
  - [Neural Network Layers](#neural-network-layers)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Optimizers](#optimizers)
  - [Learning Rate Schedulers](#learning-rate-schedulers)
  - [Data Loading](#data-loading)
  - [Model Serialization](#model-serialization)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [CUDA Requirements](#cuda-requirements)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Add fastDL to your `Cargo.toml`:

```toml
[dependencies]
fastdl = { path = "." }
```

### Build Modes

```bash
# CPU-only build (no CUDA toolkit required)
cargo build --release --no-default-features

# Full build with CUDA GPU support
cargo build --release

# Run the test suite
cargo test --no-default-features

# Run performance benchmarks
cargo bench --no-default-features
```

### CUDA Requirements

For GPU-accelerated builds:

| Requirement | Version |
|---|---|
| NVIDIA CUDA Toolkit | 12.x |
| GPU Compute Capability | 7.0+ |
| Supported Architectures | Volta, Turing, Ampere, Ada Lovelace, Hopper |

Set the `CUDA_PATH` or `CUDA_HOME` environment variable if CUDA is installed in a non-default location:

```bash
# Linux
export CUDA_PATH=/usr/local/cuda-12.0

# Windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
```

The build system automatically detects the platform and links against `cudart`, `cublas`, and `curand`.

---

## Quick Start

### Basic Tensor Operations

```rust
use fastdl::prelude::*;

// Create tensors
let a = Tensor::randn(&[3, 4]);           // Random normal [3x4]
let b = Tensor::ones(&[3, 4]);            // All ones [3x4]
let c = Tensor::kaiming_uniform(&[4, 2], 4); // He initialization

// Arithmetic (element-wise, with broadcasting)
let sum = a.add(&b);
let product = a.mul(&b);
let scaled = a.mul_scalar(0.5);
let result = &a + &b;                     // Operator overloading

// Matrix multiplication
let x = Tensor::randn(&[8, 128]);
let w = Tensor::randn(&[128, 64]);
let output = x.matmul(&w);                // [8, 64]

// Reductions
let mean = output.mean();                 // Scalar tensor
let sum_axis = output.sum_axis(1);        // Sum along columns
let variance = output.var();

// Shape manipulation
let reshaped = output.reshape(&[2, 4, 64]);
let transposed = output.transpose();       // [64, 8]
let flat = output.flatten();               // [512]
let unsqueezed = output.unsqueeze(0);      // [1, 8, 64]

// Device transfer
let gpu_tensor = x.cuda();                // Move to GPU
let cpu_tensor = gpu_tensor.cpu();         // Move back to CPU
```

### Building and Training a Neural Network

```rust
use fastdl::prelude::*;

// Define model architecture
let model = Sequential::new()
    .add(Linear::new(784, 512))
    .add(ReLU)
    .add(Dropout::new(0.3))
    .add(Linear::new(512, 256))
    .add(GELU)
    .add(Dropout::new(0.2))
    .add(Linear::new(256, 10));

println!("Parameters: {}", model.num_parameters());

// Loss function and optimizer
let loss_fn = CrossEntropyLoss::new();
let mut optimizer = AdamW::new(3e-4)
    .betas(0.9, 0.999)
    .weight_decay(0.01);

// Training loop
let mut params = model.parameters();
for epoch in 0..num_epochs {
    for (batch_input, batch_target) in &dataloader {
        // Forward pass
        let logits = model.forward(&batch_input);

        // Compute loss
        let targets: Vec<usize> = batch_target.to_vec()
            .iter().map(|&v| v as usize).collect();
        let (loss, grad) = loss_fn.forward_with_grad(&logits, &targets);

        // Backward pass + optimize
        optimizer.step(&mut params, &grads);
        optimizer.zero_grad(&params);

        println!("Loss: {:.4}", loss.item());
    }
}
```

### Transformer Encoder

```rust
use fastdl::prelude::*;
use fastdl::nn::embedding::PositionalEncoding;

let vocab_size = 32000;
let d_model = 512;
let num_heads = 8;
let d_ff = 2048;
let num_layers = 6;

// Embedding + positional encoding
let embedding = Embedding::new(vocab_size, d_model);
let pos_enc = PositionalEncoding::new(d_model, 4096);

// Transformer encoder stack
let encoder = TransformerEncoder::new(
    d_model, num_heads, d_ff, num_layers, 0.1
);

// Classification head
let classifier = Linear::new(d_model, num_classes);

// Forward pass
let token_ids = Tensor::from_vec(/* ... */, &[batch, seq_len]);
let embedded = pos_enc.forward(&embedding.forward(&token_ids));
let encoded = encoder.forward(&embedded);          // [batch, seq_len, d_model]
let pooled = encoded.mean_axis(1);                 // [batch, d_model]
let logits = classifier.forward(&pooled);          // [batch, num_classes]
```

### Convolutional Neural Network

```rust
use fastdl::prelude::*;

// LeNet-style CNN
let conv1 = Conv2d::new(1, 32, 3, 1, 1);    // [N, 32, 28, 28]
let conv2 = Conv2d::new(32, 64, 3, 1, 1);   // [N, 64, 28, 28]
let pool = MaxPool2d::new(2);                 // Spatial /2
let gap = AdaptiveAvgPool2d::global();        // -> [N, C, 1, 1]
let fc = Linear::new(64, 10);

let x = Tensor::randn(&[16, 1, 28, 28]);

// Forward
let x = pool.forward(&conv1.forward(&x).relu());   // [16, 32, 14, 14]
let x = pool.forward(&conv2.forward(&x).relu());   // [16, 64, 7, 7]
let x = gap.forward(&x);                            // [16, 64, 1, 1]
let x = x.reshape(&[16, 64]);
let logits = fc.forward(&x);                        // [16, 10]
```

### Recurrent Networks

```rust
use fastdl::prelude::*;

// LSTM for sequence modeling
let lstm = LSTM::new(128, 256, 2);   // input=128, hidden=256, layers=2
let input = Tensor::randn(&[4, 50, 128]); // [batch, seq_len, features]

let (output, h_n, c_n) = lstm.forward_seq(&input, None);
// output: [4, 50, 256] — hidden states at each timestep
// h_n:    [4, 256]     — final hidden state
// c_n:    [4, 256]     — final cell state

// GRU variant
let gru = GRU::new(128, 256);
let (output, h_n) = gru.forward_seq(&input, None);
```

---

## Features

### Tensor Engine

The tensor system is the foundation of fastDL. Tensors are N-dimensional arrays that can live on CPU or GPU memory, with seamless device transfer.

| Operation | Description |
|---|---|
| `Tensor::zeros`, `ones`, `full`, `rand`, `randn` | Standard constructors |
| `Tensor::arange`, `linspace`, `eye` | Sequence and identity constructors |
| `Tensor::kaiming_uniform`, `xavier_uniform` | Neural network weight initialization |
| `add`, `sub`, `mul`, `div` | Element-wise arithmetic with broadcasting |
| `matmul` | Matrix multiplication (2D and batched 3D via cuBLAS) |
| `transpose`, `permute`, `reshape`, `flatten` | Shape manipulation |
| `unsqueeze`, `squeeze`, `expand`, `repeat` | Dimension manipulation |
| `cat`, `stack` | Tensor concatenation and stacking |
| `sum`, `mean`, `max_val`, `min_val`, `var` | Global reductions |
| `sum_axis`, `mean_axis`, `argmax` | Per-axis reductions |
| `relu`, `sigmoid`, `tanh_act`, `gelu`, `silu` | In-place activations |
| `softmax`, `log_softmax` | Normalized probability distributions |
| `exp`, `log`, `sqrt`, `abs`, `neg`, `pow_scalar` | Unary math |
| `clamp` | Value clamping |
| `to_device`, `cuda`, `cpu` | Device transfer |
| `item` | Extract scalar from 1-element tensor |

Broadcasting follows NumPy semantics — dimensions are matched from the right, and size-1 dimensions are expanded.

### CUDA GPU Backend

All tensor operations dispatch to optimized CUDA kernels when the tensor resides on GPU memory. The GPU backend includes:

**Element-wise Kernels**
- Vectorized element-wise operations (add, sub, mul, div, pow, sqrt, exp, log)
- Fused scalar operations (add_scalar, mul_scalar)
- All activations with both forward and backward kernels

**Linear Algebra**
- cuBLAS SGEMM with TF32 tensor core math mode for matrix multiplication
- Batched strided GEMM for multi-head attention and batched operations
- Tiled transpose kernel with shared memory and bank-conflict avoidance

**Convolution**
- im2col transformation kernel for unrolling convolution into matrix multiplication
- col2im kernel for the backward pass
- Convolution computed as im2col + cuBLAS GEMM for optimal performance

**Reduction Kernels**
- Two-pass parallel reduction with warp-level primitives (no sync needed in final warp)
- Atomic operations for multi-block reductions
- Specialized argmax kernels for both global and per-axis computation

**Normalization**
- BatchNorm: parallel mean/variance computation per channel with running statistics
- LayerNorm: shared-memory reduction across the normalized dimension
- RMSNorm: single-pass RMS computation

**Attention**
- Scaled dot-product attention with causal mask support
- Fused with softmax normalization

**Optimizer Kernels**
- SGD with momentum, dampening, Nesterov, and weight decay — all in a single kernel launch
- Adam/AdamW with bias correction, AMSGrad support, and decoupled weight decay

### Neural Network Layers

Every layer implements the `Module` trait:

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn named_parameters(&self) -> HashMap<String, Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
    fn num_parameters(&self) -> usize;
    fn zero_grad(&self);
}
```

**Available Layers:**

| Layer | Description | Parameters |
|---|---|---|
| `Linear(in, out)` | Fully-connected / dense | Weight `[out, in]`, Bias `[out]` |
| `Conv2d(in_c, out_c, k, s, p)` | 2D convolution | Weight `[out_c, in_c, k, k]`, Bias `[out_c]` |
| `LSTM(in, hidden, layers)` | Long Short-Term Memory | W_ih, W_hh, b_ih, b_hh per gate |
| `GRU(in, hidden)` | Gated Recurrent Unit | W_ih, W_hh, b_ih, b_hh |
| `MultiHeadAttention(d, heads, drop)` | Multi-head scaled dot-product attention | Q/K/V/Out projections |
| `TransformerEncoderLayer(d, h, ff, drop)` | Pre-LN transformer block | Attention + FFN + 2x LayerNorm |
| `TransformerEncoder(d, h, ff, n, drop)` | Stacked encoder layers | n × EncoderLayer + final norm |
| `Embedding(vocab, dim)` | Token embedding lookup table | Weight `[vocab, dim]` |
| `PositionalEncoding(d, max_len)` | Sinusoidal position encoding | None (deterministic) |
| `BatchNorm2d(channels)` | Batch normalization (4D) | Gamma, Beta, Running stats |
| `LayerNorm(shape)` | Layer normalization | Gamma, Beta |
| `RMSNorm(size)` | RMS normalization (LLaMA-style) | Gamma |
| `Dropout(p)` | Inverted dropout | None |
| `MaxPool2d(k)` | Max pooling | None |
| `AvgPool2d(k)` | Average pooling | None |
| `AdaptiveAvgPool2d(h, w)` | Adaptive average pooling | None |
| `Sequential` | Layer container | Sum of children |

**Activation Functions:**

| Function | Formula | Backward |
|---|---|---|
| `ReLU` | max(0, x) | 1 if x > 0, else 0 |
| `LeakyReLU(α)` | max(αx, x) | α if x < 0, else 1 |
| `Sigmoid` | 1 / (1 + e^(-x)) | σ(1 - σ) |
| `Tanh` | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh²(x) |
| `GELU` | 0.5x(1 + erf(x/√2)) | Φ(x) + xφ(x) |
| `SiLU` / Swish | x · σ(x) | σ(x)(1 + x(1-σ(x))) |
| `Softmax` | e^(xi) / Σe^(xj) | Jacobian-vector product |

### Automatic Differentiation

fastDL implements reverse-mode automatic differentiation (backpropagation) through a tape-based computation graph.

```rust
use fastdl::prelude::*;

// Enable gradient tracking
fastdl::autograd::graph::enable_grad();

// Create differentiable variables
let x = Variable::new(Tensor::randn(&[4, 3])).requires_grad();
let w = Variable::new(Tensor::randn(&[3, 2])).requires_grad();

// Forward (recorded on tape)
let y = x.matmul(&w);
let loss = y.mean();

// Backward
let grads = loss.backward();

// Access gradients
let dx = grads.get(&x.id());   // ∂loss/∂x
let dw = grads.get(&w.id());   // ∂loss/∂w
```

**Supported differentiable operations:**
- Arithmetic: `add`, `sub`, `mul`, `matmul`, `mul_scalar`, `add_scalar`
- Activations: `relu`, `sigmoid`, `tanh_act`, `gelu`, `softmax`, `log_softmax`
- Reductions: `sum`, `mean`
- Shape: `reshape`
- Each operation records a `GradFn` that implements the chain rule for backward.

### Optimizers

| Optimizer | Key Features |
|---|---|
| `SGD` | Momentum, Nesterov acceleration, L2 weight decay |
| `Adam` | Adaptive learning rates, bias correction, L2 regularization |
| `AdamW` | Decoupled weight decay (Loshchilov & Hutter, 2019), bias correction |

```rust
// SGD with momentum and Nesterov
let mut sgd = SGD::new(0.01)
    .momentum(0.9)
    .weight_decay(1e-4)
    .nesterov(true);

// Adam
let mut adam = Adam::new(1e-3)
    .betas(0.9, 0.999)
    .epsilon(1e-8);

// AdamW (recommended for transformers)
let mut adamw = AdamW::new(3e-4)
    .betas(0.9, 0.95)
    .weight_decay(0.1);
```

### Learning Rate Schedulers

| Scheduler | Strategy |
|---|---|
| `StepLR(base_lr, step_size, gamma)` | Multiply LR by γ every N epochs |
| `CosineAnnealingLR(base_lr, total_steps)` | Cosine decay to minimum LR |
| `LinearWarmup(base_lr, warmup_steps)` | Linear warmup from 0 to base LR |
| `OneCycleLR(max_lr, total_steps)` | Warmup + cosine decay (Smith, 2018) |

```rust
let mut scheduler = CosineAnnealingLR::new(1e-3, 10000)
    .min_lr(1e-6);

for step in 0..10000 {
    // ... training step ...
    scheduler.step(&mut optimizer);
}
```

### Data Loading

```rust
use fastdl::prelude::*;
use fastdl::data::dataset::TensorDataset;

let inputs = Tensor::randn(&[10000, 784]);
let labels = Tensor::randn(&[10000, 1]);

let dataset = TensorDataset::new(inputs, labels);
let loader = DataLoader::new(&dataset, 64)
    .shuffle(true)
    .drop_last(true);

for (batch_x, batch_y) in loader.iter() {
    // batch_x: [64, 784]
    // batch_y: [64, 1]
}
```

The `Dataset` trait is generic — implement `len()` and `get(index)` for custom datasets:

```rust
struct MyDataset { /* ... */ }

impl Dataset for MyDataset {
    fn len(&self) -> usize { /* ... */ }
    fn get(&self, index: usize) -> (Tensor, Tensor) { /* ... */ }
}
```

### Model Serialization

fastDL uses a compact binary checkpoint format (`.fdl`):

```
Header:  magic(4B) + version(4B) + num_tensors(4B)
Tensor:  name_len(4B) + name(UTF-8) + ndim(4B) + shape(4B × ndim) + data(4B × numel)
```

```rust
use fastdl::prelude::*;

// Save
save_model(&model, "checkpoint.fdl").unwrap();

// Load
let params = load_tensors("checkpoint.fdl").unwrap();
```

### Loss Functions

| Loss | Use Case | Input |
|---|---|---|
| `CrossEntropyLoss` | Multi-class classification | Logits `[B, C]` + class indices |
| `MSELoss` | Regression | Predictions + targets |
| `BCELoss` | Binary classification (probabilities) | Sigmoid outputs + binary targets |
| `BCEWithLogitsLoss` | Binary classification (numerically stable) | Raw logits + binary targets |

All loss functions provide `forward()` for loss only, and `forward_with_grad()` for loss + gradient w.r.t. input.

---

## Architecture

```
fastDL/
├── cuda/
│   ├── kernels.cu                 # CUDA kernel implementations (~1400 lines)
│   └── include/
│       └── kernels.h              # C FFI interface declarations
├── src/
│   ├── lib.rs                     # Crate root, public API, prelude
│   ├── tensor/
│   │   ├── tensor.rs              # Tensor struct, constructors, shape ops, device transfer
│   │   ├── ops.rs                 # Tensor operations with CPU/CUDA dispatch
│   │   └── cuda_backend.rs        # FFI bindings + CUDA helper functions
│   ├── autograd/
│   │   ├── graph.rs               # Computation graph, backward pass, global state
│   │   └── variable.rs            # Differentiable variable + GradFn implementations
│   ├── nn/
│   │   ├── module.rs              # Module trait definition
│   │   ├── sequential.rs          # Sequential layer container
│   │   ├── linear.rs              # Fully-connected layer
│   │   ├── conv.rs                # 2D convolution
│   │   ├── rnn.rs                 # LSTM, GRU
│   │   ├── transformer.rs         # Multi-Head Attention, Transformer Encoder
│   │   ├── embedding.rs           # Embedding + Positional Encoding
│   │   ├── activation.rs          # Activation function layers
│   │   ├── normalization.rs       # BatchNorm2d, LayerNorm, RMSNorm
│   │   ├── dropout.rs             # Dropout
│   │   ├── pooling.rs             # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   │   └── loss.rs                # Loss functions
│   ├── optim/
│   │   ├── sgd.rs                 # SGD optimizer
│   │   ├── adam.rs                # Adam, AdamW optimizers
│   │   └── scheduler.rs           # Learning rate schedulers
│   ├── data/
│   │   ├── dataset.rs             # Dataset trait, TensorDataset, VecDataset
│   │   └── dataloader.rs          # DataLoader with batching and shuffling
│   ├── cuda/
│   │   ├── context.rs             # CUDA device initialization, memory info
│   │   └── memory.rs              # CudaBuffer (RAII GPU memory)
│   ├── serialize/
│   │   └── checkpoint.rs          # Binary checkpoint save/load
│   └── utils/
│       └── random.rs              # Global RNG seeding
├── examples/
│   ├── simple_mlp.rs              # XOR problem with MLP
│   ├── mnist.rs                   # CNN image classification
│   └── transformer.rs             # Transformer sequence classification
├── benches/
│   └── tensor_ops.rs              # Criterion benchmarks
├── build.rs                       # CUDA compilation via cc crate + nvcc
├── Cargo.toml
└── README.md
```

### Design Decisions

**Single CUDA kernel file** — All GPU kernels reside in `cuda/kernels.cu` rather than being split across files. This simplifies the build system (single `nvcc` invocation) and makes it straightforward to share constants, helper macros, and the cuBLAS handle across all kernels.

**im2col convolution** — Convolution is implemented via im2col + cuBLAS GEMM rather than direct convolution or Winograd. This approach reuses the highly optimized cuBLAS matrix multiplication and is simpler to implement correctly, while providing competitive performance for most kernel sizes.

**Pre-LN Transformer** — The transformer encoder uses the Pre-LayerNorm variant (norm before attention/FFN) rather than Post-LN. Pre-LN is more stable during training and has become the standard in modern architectures.

**Tape-based autograd** — Operations are recorded to a global tape in forward order, then replayed in reverse during backward. This is simpler than a DAG-based approach and sufficient for the supported operation set.

**RAII GPU memory** — `CudaBuffer` wraps raw CUDA allocations with Rust's `Drop` trait, ensuring GPU memory is freed when the buffer goes out of scope. This eliminates memory leaks that are common in C/C++ CUDA code.

---

## Examples

### Run Examples

```bash
# Simple MLP on XOR problem
cargo run --example simple_mlp --no-default-features

# CNN classifier (synthetic MNIST-like data)
cargo run --example mnist --no-default-features

# Transformer encoder for sequence classification
cargo run --example transformer --no-default-features
```

---

## Benchmarks

Run the benchmark suite:

```bash
cargo bench --no-default-features
```

Available benchmarks:

| Benchmark | Description |
|---|---|
| `matmul_128x256_x_256x128` | Rectangular matrix multiplication |
| `matmul_512x512` | Square matrix multiplication |
| `relu_1M_elements` | ReLU activation on 1M elements |
| `softmax_64x1000` | Softmax over 1000 classes |
| `add_1M_elements` | Element-wise addition (1M elements) |
| `mul_1M_elements` | Element-wise multiplication (1M elements) |

---

## Roadmap

- [ ] Mixed-precision training (FP16/BF16)
- [ ] Multi-GPU data parallelism
- [ ] 1D and 3D convolutions
- [ ] Deconvolution / transposed convolution
- [ ] Gradient checkpointing for memory-efficient training
- [ ] ONNX model export
- [ ] Flash Attention v2
- [ ] Custom CUDA kernel JIT compilation
- [ ] Distributed training across nodes
- [ ] Weight quantization (INT8, INT4)
- [ ] Dynamic batching and variable-length sequence support
- [ ] Built-in dataset downloaders (MNIST, CIFAR-10, ImageNet)

---

## Contributing

Contributions are welcome. Please open an issue to discuss any significant changes before submitting a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Ensure `cargo check --no-default-features` passes
4. Commit your changes
5. Push to the branch and open a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
