<p align="center">
  <h1 align="center">⚡ FastNN</h1>
  <p align="center">
    <strong>A deep learning framework built from scratch in Rust and CUDA — no black boxes, no wrappers, just raw math and metal.</strong>
  </p>
  <p align="center">
    <a href="#what-is-this">What Is This?</a> •
    <a href="#understanding-the-building-blocks">Building Blocks</a> •
    <a href="#how-training-works">How Training Works</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#benchmarks">Benchmarks</a>
  </p>
</p>

---

## What Is This?

FastNN is a **deep learning library**. Before understanding what FastNN does, it helps to understand what deep learning *is* — and why building one from scratch is hard.

### What Is Deep Learning?

Imagine you want to teach a computer to recognize a cat in a photo. You cannot write rules like "if there are pointy ears and whiskers, it's a cat" — real-world images are too complex and varied for hand-crafted rules.

Instead, you show a computer hundreds of thousands of cat photos, and let it **learn the rules by itself**. That process of learning from examples is called **machine learning**. Deep learning is a specific approach to machine learning that uses structures called **neural networks**.

### What Is a Neural Network?

A neural network is a mathematical structure loosely inspired by how neurons in the brain work. It is composed of **layers** — each layer takes numbers as input, does some math on them, and passes the result forward to the next layer.

```
Input Image (pixels)
     ↓
Layer 1: Detect edges and corners
     ↓
Layer 2: Combine edges into shapes
     ↓
Layer 3: Combine shapes into patterns (ears, eyes)
     ↓
Layer 4: Recognize the animal
     ↓
Output: "Cat" (98% confident)
```

The "deep" in deep learning refers to having many such layers stacked on top of each other. Modern systems have dozens to hundreds of layers.

### What Is FastNN's Role?

FastNN is the **engine** that makes this possible. It handles all the underlying mathematics, memory management, and GPU acceleration so that you can focus on designing neural networks rather than writing low-level code.

Think of FastNN the same way you think of an engine in a car — you do not need to understand every combustion cycle to drive, but without the engine, the car does not move.

---

## Understanding the Building Blocks

### Tensors — The Universal Data Container

At the heart of every deep learning framework is the **tensor**. A tensor is simply a multi-dimensional array of numbers.

- A **scalar** (single number) is a 0-dimensional tensor: `5.0`
- A **vector** (list of numbers) is a 1D tensor: `[1.0, 2.0, 3.0]`
- A **matrix** (table of numbers) is a 2D tensor: rows × columns
- A **3D tensor** can represent an image: height × width × color_channels
- A **4D tensor** represents a batch of images: num_images × height × width × channels

In FastNN, every piece of data — inputs, weights, activations, gradients — is a tensor. Every operation (addition, multiplication, matrix multiply) works on tensors.

```rust
use fastnn::prelude::*;

// A 3×4 matrix of random numbers
let a = Tensor::randn(&[3, 4]);

// A batch of 16 grayscale 28×28 images
let images = Tensor::zeros(&[16, 1, 28, 28]);

// Matrix multiplication: [8, 128] × [128, 64] → [8, 64]
let output = Tensor::randn(&[8, 128]).matmul(&Tensor::randn(&[128, 64]));
```

### GPU Acceleration — Why We Need CUDA

A modern CPU has 8–32 cores and can do a handful of things simultaneously. A modern GPU has **thousands of cores** and can perform millions of simple operations in parallel.

Deep learning is perfect for GPUs because neural networks are fundamentally enormous amounts of **matrix multiplication** — multiplying two giant grids of numbers together. A GPU can do these multiplications thousands of times faster than a CPU.

**CUDA** is NVIDIA's programming interface for writing code that runs on their GPUs. FastNN has custom CUDA code (written in a language called CUDA C++) that handles all GPU-accelerated operations.

When you move a tensor to the GPU in FastNN:

```rust
let x = Tensor::randn(&[1000, 1000]);  // Lives in CPU RAM
let x_gpu = x.cuda();                   // Copied to GPU VRAM
let result = x_gpu.matmul(&x_gpu);      // Runs on GPU — much faster
let back = result.cpu();                // Copy result back to CPU
```

FastNN uses **cuBLAS** (NVIDIA's highly optimized matrix math library) for the most performance-critical operations, and its own hand-written kernels for everything else.

### Weights — What the Network "Knows"

A neural network learns by adjusting its **weights** — numbers stored inside each layer that determine how the layer transforms its input. Before training, weights are random. After training on millions of examples, they encode everything the network has learned.

A `Linear` layer (also called a fully-connected or dense layer) has a weight matrix and a bias vector:

```
output = input × weight_matrix + bias
```

If the input is a 784-dimensional vector (a 28×28 image flattened) and the output needs 512 dimensions, then:
- `weight_matrix` is shape [512, 784] — that is 401,408 numbers to learn
- `bias` is shape [512] — another 512 numbers

For perspective, GPT-3 has 175 billion weights.

### Automatic Differentiation — How Learning Happens

Training a neural network means finding the weights that produce the best outputs. We do this with **gradient descent**:

1. Run an input through the network (forward pass)
2. Compare the output to the correct answer using a **loss function** (a number measuring how wrong we were)
3. Figure out how much each weight contributed to the error
4. Nudge each weight in the direction that reduces the error
5. Repeat millions of times

Step 3 requires computing **gradients** — derivatives of the loss with respect to every weight in the network. This is done with **backpropagation**, which is an application of the chain rule from calculus, applied backward through every layer.

Doing this by hand for complex networks is impractical. FastNN's **autograd engine** does it automatically:

```rust
// Enable gradient tracking
fastnn::autograd::graph::enable_grad();

// Variables track their history
let x = Variable::new(Tensor::randn(&[4, 3])).requires_grad();
let w = Variable::new(Tensor::randn(&[3, 2])).requires_grad();

// Forward pass — every operation is recorded
let y = x.matmul(&w);
let loss = y.mean();

// Backward pass — gradients computed automatically
let grads = loss.backward();

// Now grads contains ∂loss/∂x and ∂loss/∂w
```

Internally, FastNN records every operation in a **computation graph** (a chain of recorded operations called a "tape"). Calling `.backward()` replays this tape in reverse, computing how much each operation contributed to the final loss.

### Optimizers — Applying What We Learned

Once we have gradients, an **optimizer** uses them to update the weights. The simplest rule is:

```
new_weight = old_weight - learning_rate × gradient
```

A small `learning_rate` (like `0.001`) means small, careful steps. This basic rule is **Stochastic Gradient Descent (SGD)**. More sophisticated optimizers like **Adam** and **AdamW** maintain statistics about past gradients to take smarter steps.

FastNN includes SGD, Adam, and AdamW, all with CUDA kernels that update weights directly on the GPU without copying data back to the CPU.

---

## How Training Works — End to End

Here is the complete cycle of training a neural network, tied back to FastNN's components:

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                          │
│                                                             │
│  1. DATA LOADING                                            │
│     DataLoader → fetches a batch of (input, label) pairs    │
│     Tensors: input [batch_size, features], labels [batch]   │
│                                                             │
│  2. FORWARD PASS                                            │
│     model.forward(input) → runs input through all layers    │
│     Each layer: Linear, Conv2d, BatchNorm, ReLU, Dropout    │
│     Output: logits [batch_size, num_classes]                │
│                                                             │
│  3. LOSS COMPUTATION                                        │
│     CrossEntropyLoss(logits, labels) → single scalar        │
│     "How wrong was the network on this batch?"              │
│                                                             │
│  4. BACKWARD PASS (Backpropagation)                         │
│     loss.backward() → computes ∂loss/∂weight for all weights│
│     Traverses computation graph in reverse                  │
│                                                             │
│  5. OPTIMIZER STEP                                          │
│     optimizer.step(params, grads) → updates all weights     │
│     Adam: uses moment estimates for adaptive step sizes     │
│                                                             │
│  6. ZERO GRADIENTS                                          │
│     optimizer.zero_grad() → clears gradients for next step  │
│                                                             │
│  Repeat for thousands of batches over many epochs           │
└─────────────────────────────────────────────────────────────┘
```

---

## What FastNN Provides

### Neural Network Layers (`src/nn/`)

Every layer in FastNN implements the `Module` trait — a common interface with `forward()`, `parameters()`, `train()`, and `eval()` methods. Layers can be composed into any architecture using `Sequential`.

| Layer | What It Does |
|---|---|
| `Linear(in, out)` | Fully-connected layer. Multiplies input by a weight matrix and adds bias. The most basic building block. |
| `Conv2d(in_c, out_c, k, s, p)` | 2D convolution. Slides a small filter (kernel) across an image to detect local patterns. Used in image recognition. |
| `LSTM(in, hidden, layers)` | Long Short-Term Memory. Processes sequences one step at a time, maintaining a hidden state that carries context across steps. |
| `GRU(in, hidden)` | Gated Recurrent Unit. A simpler variant of LSTM with fewer parameters. |
| `MultiHeadAttention(d, heads, drop)` | The key mechanism in Transformers. Lets every position in a sequence attend to every other position, weighted by relevance. |
| `TransformerEncoder(d, h, ff, n, drop)` | Stack of Transformer encoder blocks. Foundation of models like BERT. |
| `Embedding(vocab, dim)` | Maps integer token IDs to dense vectors. First layer in any language model. |
| `BatchNorm2d(channels)` | Normalizes activations across the batch dimension. Stabilizes and accelerates training of CNNs. |
| `LayerNorm(shape)` | Normalizes activations across the feature dimension. Used in Transformers. |
| `RMSNorm(size)` | Simplified normalization using only RMS (no mean subtraction). Used in LLaMA. |
| `Dropout(p)` | Randomly zeroes activations during training to prevent overfitting. |
| `MaxPool2d(k)` | Downsamples spatial dimensions by keeping only the maximum value in each window. |

### Activation Functions

Activations introduce **non-linearity** — without them, stacking layers would be mathematically equivalent to a single layer, no matter how deep the network. Non-linearity is what allows networks to learn complex patterns.

| Activation | Characteristic |
|---|---|
| `ReLU` | Returns max(0, x). Fast, simple, widely used. |
| `GELU` | Smooth approximation to ReLU. Used in BERT, GPT. |
| `SiLU / Swish` | x × sigmoid(x). Used in modern efficient networks. |
| `Sigmoid` | Squashes output to (0, 1). Used in binary classification outputs. |
| `Tanh` | Squashes output to (-1, 1). Common in RNNs. |
| `Softmax` | Converts a vector of numbers into probabilities that sum to 1. Used for multi-class classification. |

### Loss Functions

A loss function measures how wrong the network's prediction is. The optimizer minimizes this number.

| Loss | When to Use |
|---|---|
| `CrossEntropyLoss` | Multi-class classification (e.g., "is this a cat, dog, or bird?") |
| `MSELoss` | Regression — predicting a continuous number (e.g., house price) |
| `BCELoss` | Binary classification — yes/no outputs where sigmoid was applied |
| `BCEWithLogitsLoss` | Binary classification with raw logits (numerically more stable) |

### Optimizers

| Optimizer | Notes |
|---|---|
| `SGD` | Classic. Works well with careful tuning. Supports momentum and Nesterov acceleration. |
| `Adam` | Adaptive per-parameter learning rates. Usually converges faster than SGD. |
| `AdamW` | Adam with decoupled weight decay. Standard choice for training Transformers. |

### Learning Rate Schedulers

The learning rate controls how large each update step is. A schedule changes this over training.

| Scheduler | Strategy |
|---|---|
| `StepLR` | Multiply learning rate by γ every N steps. Simple step decay. |
| `CosineAnnealingLR` | Smoothly decay learning rate following a cosine curve. Very popular. |
| `LinearWarmup` | Gradually increase LR from 0 at the start of training. Prevents instability. |
| `OneCycleLR` | Warmup to max LR then cosine decay. Fast convergence. |

---

## Architecture

### How FastNN Is Organized

```
fastnn/
├── cuda/
│   ├── kernels.cu              # All GPU code — ~1400 lines of CUDA C++
│   └── include/kernels.h       # C interface header (how Rust calls CUDA)
├── src/
│   ├── lib.rs                  # Library entry point, public API, prelude
│   ├── tensor/
│   │   ├── tensor.rs           # Tensor type — the central data structure
│   │   ├── ops.rs              # All mathematical operations (add, matmul, relu…)
│   │   └── cuda_backend.rs     # Rust ↔ CUDA bridge (FFI bindings)
│   ├── autograd/
│   │   ├── graph.rs            # Computation graph — records operations for backward
│   │   └── variable.rs         # Variable type — a Tensor that tracks gradients
│   ├── nn/
│   │   ├── module.rs           # Module trait — the interface all layers implement
│   │   ├── sequential.rs       # Sequential — chains multiple layers together
│   │   ├── linear.rs           # Fully-connected layer
│   │   ├── conv.rs             # 2D convolution
│   │   ├── rnn.rs              # LSTM and GRU
│   │   ├── transformer.rs      # Multi-Head Attention, TransformerEncoder
│   │   ├── embedding.rs        # Embedding + PositionalEncoding
│   │   ├── activation.rs       # ReLU, GELU, Sigmoid, Tanh, SiLU…
│   │   ├── normalization.rs    # BatchNorm2d, LayerNorm, RMSNorm
│   │   ├── dropout.rs          # Dropout
│   │   ├── pooling.rs          # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   │   └── loss.rs             # CrossEntropy, MSE, BCE loss functions
│   ├── optim/
│   │   ├── sgd.rs              # SGD optimizer
│   │   ├── adam.rs             # Adam and AdamW optimizers
│   │   └── scheduler.rs        # Learning rate schedulers
│   ├── data/
│   │   ├── dataset.rs          # Dataset trait + TensorDataset
│   │   └── dataloader.rs       # DataLoader — batching, shuffling, iteration
│   ├── cuda/
│   │   ├── context.rs          # CUDA device setup, memory info
│   │   └── memory.rs           # CudaBuffer — RAII GPU memory wrapper
│   ├── serialize/
│   │   └── checkpoint.rs       # Save/load model weights to .fdl files
│   └── utils/
│       └── random.rs           # Global random seed control
├── examples/
│   ├── simple_mlp.rs           # XOR problem solved with a small MLP
│   ├── mnist.rs                # Handwritten digit recognition with a CNN
│   └── transformer.rs          # Sequence classification with a Transformer
├── benches/
│   └── tensor_ops.rs           # Performance benchmarks
├── build.rs                    # Compiles CUDA code at build time
└── Cargo.toml                  # Rust project configuration and dependencies
```

### Key Technical Decisions

**Dual-backend dispatch — one API, two execution paths**

Every tensor operation in FastNN works on both CPU and GPU. Internally, a `Tensor` is either:
- `TensorStorage::Cpu(Vec<f32>)` — a plain Rust vector in RAM
- `TensorStorage::Cuda(Arc<CudaBuffer>)` — RAII-managed GPU memory

When you call `a.add(&b)`, FastNN checks which device the tensors live on and dispatches to either the CPU Rust implementation or the CUDA kernel automatically. You never write device-specific code.

**Single CUDA kernel file**

All ~1400 lines of GPU code live in `cuda/kernels.cu`. This keeps the build system simple (one `nvcc` compilation), makes it easy to share CUDA constants and helpers across kernels, and keeps the GPU code easy to navigate.

**cuBLAS for matrix multiplication**

Matrix multiplication is the most computationally expensive operation in deep learning. Rather than writing a custom GEMM kernel, FastNN delegates to cuBLAS — NVIDIA's hand-tuned BLAS library that uses Tensor Cores for TF32 acceleration. Custom kernels handle everything else.

**Convolution via im2col + GEMM**

2D convolution is implemented by first running an `im2col` transformation (unrolling the input into a matrix), then calling cuBLAS GEMM. This reuses the highly optimized matrix multiply path and is simpler to implement correctly than a direct convolution kernel.

**Pre-LayerNorm Transformer**

FastNN's Transformer uses the Pre-LN variant (layer normalization applied *before* the attention/FFN sublayer, not after). Pre-LN training is more stable and is the standard in modern architectures like GPT-2 and beyond.

**RAII GPU memory**

`CudaBuffer` wraps `cudaMalloc` and implements Rust's `Drop` trait, calling `cudaFree` automatically when the buffer goes out of scope. This means it is impossible to leak GPU memory — Rust's ownership system enforces cleanup.

**Tape-based autograd**

When gradient tracking is enabled, every `Variable` operation appends a `GradFn` node to a global tape. Calling `.backward()` reads the tape in reverse order, feeding upstream gradients into each node's gradient function. This is the same approach used by PyTorch and is well-suited for dynamic computation graphs.

**CPU parallelism with Rayon**

On the CPU path, tensor operations use Rayon for data parallelism — the work is split across all available CPU cores automatically.

---

## CUDA Requirements

For GPU-accelerated builds:

| Requirement | Version |
|---|---|
| NVIDIA CUDA Toolkit | 12.x |
| GPU Compute Capability | 7.0+ (Volta and newer) |
| Supported Architectures | Volta, Turing, Ampere, Ada Lovelace, Hopper |

Set `CUDA_PATH` if the toolkit is not in the default location:

```bash
# Linux
export CUDA_PATH=/usr/local/cuda-12.0

# Windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
```

The build system links against `cudart` (CUDA runtime), `cublas` (matrix math), and `curand` (random number generation).

---

## Installation

Add FastNN to your `Cargo.toml`:

```toml
[dependencies]
fastnn = { path = "." }
```

### Build Modes

```bash
# CPU-only build (no CUDA toolkit required — best for development)
cargo build --release --no-default-features

# Full GPU build
cargo build --release

# Verify compilation without building
cargo check --no-default-features

# Run tests
cargo test --no-default-features

# Run a single test by name
cargo test --no-default-features test_name

# Run tests in a specific module
cargo test --no-default-features tensor::ops

# Run benchmarks
cargo bench --no-default-features
```

---

## Quick Start

### Basic Tensor Operations

```rust
use fastnn::prelude::*;

// Create tensors
let a = Tensor::randn(&[3, 4]);              // Random normal [3×4]
let b = Tensor::ones(&[3, 4]);               // All ones [3×4]

// Arithmetic
let sum = a.add(&b);
let scaled = a.mul_scalar(0.5);
let result = &a + &b;                        // Operator overloading

// Matrix multiplication
let x = Tensor::randn(&[8, 128]);
let w = Tensor::randn(&[128, 64]);
let output = x.matmul(&w);                   // [8, 64]

// Reductions
let mean = output.mean();
let sum_axis = output.sum_axis(1);           // Sum along columns

// Shape manipulation
let reshaped = output.reshape(&[2, 4, 64]);
let transposed = output.transpose();          // [64, 8]
let flat = output.flatten();                  // [512]

// Device transfer (GPU must be available)
let gpu_tensor = x.cuda();
let cpu_tensor = gpu_tensor.cpu();
```

### Building a Neural Network

```rust
use fastnn::prelude::*;

// Compose layers into an architecture
let model = Sequential::new()
    .add(Linear::new(784, 512))
    .add(ReLU)
    .add(Dropout::new(0.3))
    .add(Linear::new(512, 256))
    .add(GELU)
    .add(Dropout::new(0.2))
    .add(Linear::new(256, 10));

println!("Parameters: {}", model.num_parameters());

// Forward pass
let input = Tensor::randn(&[32, 784]);       // Batch of 32 images
let logits = model.forward(&input);           // [32, 10] class scores
```

### Transformer Encoder

```rust
use fastnn::prelude::*;
use fastnn::nn::embedding::PositionalEncoding;

let embedding = Embedding::new(32000, 512);
let pos_enc = PositionalEncoding::new(512, 4096);
let encoder = TransformerEncoder::new(512, 8, 2048, 6, 0.1);
let classifier = Linear::new(512, 10);

let token_ids = Tensor::from_vec(vec![1.0; 4 * 64], &[4, 64]);
let embedded = pos_enc.forward(&embedding.forward(&token_ids));
let encoded = encoder.forward(&embedded);    // [4, 64, 512]
let pooled = encoded.mean_axis(1);           // [4, 512]
let logits = classifier.forward(&pooled);    // [4, 10]
```

### Convolutional Network (Image Classification)

```rust
use fastnn::prelude::*;

// LeNet-style CNN
let conv1 = Conv2d::new(1, 32, 3, 1, 1);
let conv2 = Conv2d::new(32, 64, 3, 1, 1);
let pool = MaxPool2d::new(2);
let gap = AdaptiveAvgPool2d::global();
let fc = Linear::new(64, 10);

let x = Tensor::randn(&[16, 1, 28, 28]);    // Batch of 16 grayscale images

let x = pool.forward(&conv1.forward(&x).relu());   // [16, 32, 14, 14]
let x = pool.forward(&conv2.forward(&x).relu());   // [16, 64,  7,  7]
let x = gap.forward(&x);                            // [16, 64,  1,  1]
let x = x.reshape(&[16, 64]);
let logits = fc.forward(&x);                        // [16, 10]
```

### Recurrent Networks

```rust
use fastnn::prelude::*;

// LSTM for sequence modeling
let lstm = LSTM::new(128, 256, 2);           // input_size=128, hidden=256, layers=2
let input = Tensor::randn(&[4, 50, 128]);    // [batch, seq_len, features]

let (output, h_n, c_n) = lstm.forward_seq(&input, None);
// output: [4, 50, 256] — hidden state at every timestep
// h_n:    [4, 256]     — final hidden state
// c_n:    [4, 256]     — final cell state
```

### Automatic Differentiation

```rust
use fastnn::prelude::*;

fastnn::autograd::graph::enable_grad();

let x = Variable::new(Tensor::randn(&[4, 3])).requires_grad();
let w = Variable::new(Tensor::randn(&[3, 2])).requires_grad();

let y = x.matmul(&w);
let loss = y.mean();

let grads = loss.backward();
let dx = grads.get(&x.id());                // ∂loss/∂x
let dw = grads.get(&w.id());                // ∂loss/∂w
```

### Data Loading

```rust
use fastnn::prelude::*;
use fastnn::data::dataset::TensorDataset;

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

Custom datasets implement two methods:

```rust
struct MyDataset { /* your data */ }

impl Dataset for MyDataset {
    fn len(&self) -> usize { /* total number of samples */ }
    fn get(&self, index: usize) -> (Tensor, Tensor) { /* return one (input, label) pair */ }
}
```

### Model Checkpointing

FastNN saves weights in a compact binary format (`.fdl` extension):

```
Header:  magic(4B) + version(4B) + num_tensors(4B)
Tensor:  name_len(4B) + name(UTF-8) + ndim(4B) + shape(4B × ndim) + data(4B × numel)
```

```rust
use fastnn::prelude::*;

save_model(&model, "checkpoint.fdl").unwrap();

let params = load_tensors("checkpoint.fdl").unwrap();
```

---

## API Reference

### Tensor Operations

| Category | Operations |
|---|---|
| Constructors | `zeros`, `ones`, `full`, `rand`, `randn`, `arange`, `linspace`, `eye`, `from_vec` |
| Initialization | `kaiming_uniform`, `xavier_uniform` |
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `abs`, `pow_scalar` |
| Linear algebra | `matmul` (2D and batched 3D via cuBLAS) |
| Shape | `reshape`, `flatten`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `expand`, `repeat` |
| Combining | `cat`, `stack` |
| Reductions | `sum`, `mean`, `max_val`, `min_val`, `var`, `sum_axis`, `mean_axis`, `argmax` |
| Activations | `relu`, `sigmoid`, `tanh_act`, `gelu`, `silu`, `softmax`, `log_softmax` |
| Math | `exp`, `log`, `sqrt`, `clamp` |
| Device | `cuda()`, `cpu()`, `to_device()` |
| Scalar extract | `item()` |

---

## Examples

```bash
# Small MLP solving the XOR problem
cargo run --example simple_mlp --no-default-features

# CNN for handwritten digit recognition
cargo run --example mnist --no-default-features

# Transformer encoder for sequence classification
cargo run --example transformer --no-default-features
```

---

## Benchmarks

```bash
cargo bench --no-default-features
```

| Benchmark | Description |
|---|---|
| `matmul_128x256_x_256x128` | Rectangular matrix multiplication |
| `matmul_512x512` | Square matrix multiplication |
| `relu_1M_elements` | ReLU activation over 1 million values |
| `softmax_64x1000` | Softmax over 1000 classes, batch of 64 |
| `add_1M_elements` | Element-wise addition, 1 million values |
| `mul_1M_elements` | Element-wise multiplication, 1 million values |

---

## Known Limitations (Current State)

FastNN is in active development. These are critical gaps that affect real-world usability:

- **Training loops are not yet functional** — The three examples compile and run, but they use placeholder zero gradients. No actual weight updates occur, so no learning happens.
- **Autograd and Module are not yet connected** — The `Variable` autograd system and the `Module` layer system operate independently. Layers do not yet participate in the computation graph.
- **Several backward passes are missing** — `Transpose`/`Permute`, `Expand`/`Repeat`, `Cat`/`Stack`, `Squeeze`/`Unsqueeze`, `Div`, and `Pow` do not yet have backward implementations.
- **Optimizer step is a stub** — Optimizers exist but do not yet receive gradients from the autograd engine.

The most impactful next step is bridging `Variable` autograd to `Module.forward()`.

---

## Roadmap

- [ ] Connect autograd engine to Module layers (critical path)
- [ ] Working end-to-end training loop
- [ ] Mixed-precision training (FP16/BF16)
- [ ] Flash Attention v2
- [ ] 1D and 3D convolutions
- [ ] Gradient checkpointing
- [ ] Multi-GPU data parallelism
- [ ] ONNX model export
- [ ] Built-in dataset downloaders (MNIST, CIFAR-10)
- [ ] Weight quantization (INT8, INT4)

---

## Contributing

Contributions are welcome. Open an issue to discuss significant changes before submitting a pull request.

1. Fork the repository
2. Create your branch: `git checkout -b feature/my-feature`
3. Verify: `cargo check --no-default-features`
4. Commit and open a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.
