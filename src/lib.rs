//! # FastNN
//!
//! A robust deep learning library in Rust with CUDA kernels.
//!
//! FastNN is small on purpose. Every file is one idea, so you can follow
//! `loss.backward()` from your training loop down to the kernel. It runs on the
//! CPU out of the box and on NVIDIA GPUs with the `cuda` feature.
//!
//! - **Tensors** with broadcasting, batched matmul, reductions, indexing and views.
//! - **Reverse-mode autograd** with no tape and no global state.
//! - **Layers**: linear, convolutions, normalization, recurrent, attention and transformers.
//! - **Generation**: KV cache, sampling and beam search.
//! - **Optimizers and schedules**: SGD, Adam, AdamW, RMSprop, Lion and more.
//! - **Checkpoints**: resumable training state and safetensors.
//!
//! ## Quick start
//!
//! ```no_run
//! use fastnn::prelude::*;
//! use fastnn::data::{Mnist, Split};
//!
//! # fn main() -> fastnn::Result<()> {
//! let train = Mnist::load(Split::Train)?;
//! let loader = DataLoader::new(&train, 128).shuffle(true);
//!
//! let model = Sequential::new()
//!     .add(Flatten::new())
//!     .add(Linear::new(784, 128))
//!     .add(ReLU)
//!     .add(Linear::new(128, 10));
//!
//! let mut opt = Adam::new(model.parameters(), 1e-3);
//!
//! for batch in loader.iter() {
//!     let loss = cross_entropy(&model.forward(&batch.inputs), &batch.labels());
//!
//!     opt.zero_grad();
//!     loss.backward();
//!     opt.step();
//! }
//!
//! save(&model, "mnist.fdl")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## How it fits together
//!
//! - [`tensor`] holds the array type and its ops.
//! - [`autograd`] holds the backward rules. A tensor made by a differentiable op
//!   carries a node naming the rule and its inputs, and `backward()` walks them.
//!   A tensor with no node is plain data, so there is no `requires_grad` flag.
//! - [`nn`] holds the layers. A layer's weights are [`Param`](nn::Param) handles,
//!   shared with the optimizer, so `opt.step()` never borrows the model.
//! - [`optim`] holds the optimizers and learning-rate schedules.
//! - [`data`] holds datasets and the batching [`DataLoader`](data::DataLoader).
//! - [`serialize`] holds checkpoints and safetensors.
//! - [`cuda`] holds the GPU memory and kernel bindings.
//!
//! ## Devices
//!
//! Everything starts on the CPU. Move the model with
//! [`Module::to_device`](nn::Module::to_device) and the batches with
//! [`DataLoader::to_device`](data::DataLoader::to_device). Mixing devices panics
//! instead of copying behind your back.
//!
//! ```no_run
//! # use fastnn::prelude::*;
//! # let model = Sequential::new();
//! # let dataset: fastnn::data::TensorDataset = unimplemented!();
//! let device = Device::best(); // a GPU if there is one, else the CPU
//! model.to_device(device);
//! let loader = DataLoader::new(&dataset, 64).to_device(device);
//! ```
//!
//! ## Errors
//!
//! A wrong shape or a device mismatch is a bug in the calling code, so those
//! panic with a message naming both shapes. [`Result`] is for things that fail at
//! runtime: file I/O, dataset downloads and CUDA. Where a shape comes from a
//! config file or an upload, the `try_` variants of the tensor ops return an
//! [`Error`] instead of panicking.
//!
//! ## Feature flags
//!
//! - `cuda` compiles the GPU kernels with `nvcc` and links the CUDA runtime and
//!   cuBLAS. It is off by default. Without it everything runs on the CPU, and
//!   [`Device::cuda`](tensor::Device::cuda) returns an error.
//!
//! These docs are built without `cuda`, since docs.rs has no CUDA toolkit.

pub mod autograd;
pub mod cuda;
pub mod data;
pub mod error;
pub mod nn;
pub mod optim;
pub mod rng;
pub mod serialize;
pub mod tensor;

pub use error::{Error, Result};

/// The usual imports for a training script.
///
/// ```
/// use fastnn::prelude::*;
/// ```
pub mod prelude {
    pub use crate::autograd::{detect_anomaly, no_grad};
    pub use crate::data::{Batch, DataLoader, Dataset, TensorDataset};
    pub use crate::nn::{
        bce, bce_with_logits, cross_entropy, huber, kl_divergence, mae, mse, nll,
        AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, BeamSearch, Buffer, Conv1d, Conv2d,
        ConvTranspose1d, ConvTranspose2d, CrossEntropyLoss, Dropout, Embedding, Flatten, KvCache,
        LayerNorm, LeakyReLU, Linear, MaxPool2d, Module, MultiHeadAttention, Param,
        PositionalEncoding, RMSNorm, ReLU, Reduction, Reshape, Sampler, Sequential, SiLU, Sigmoid,
        Softmax, StackCache, Tanh, Transformer, TransformerBlock, TransformerDecoder,
        TransformerDecoderBlock, TransformerStack, GELU, GRU, LSTM,
    };
    pub use crate::optim::{
        clip_grad_norm, clip_grad_value, Adadelta, Adagrad, Adam, AdamW, Constant, CosineAnnealing,
        Ema, Lion, Lookahead, LrSchedule, OneCycle, Optimizer, OptimizerState, RAdam, RMSprop,
        StepDecay, Warmup, SGD,
    };
    pub use crate::rng::manual_seed;
    pub use crate::serialize::{
        load, load_safetensors, load_training, save, save_safetensors, save_training,
    };
    pub use crate::tensor::{Device, Tensor, Window};
}
