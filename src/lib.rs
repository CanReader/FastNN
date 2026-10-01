//! # FastNN
//!
//! A deep learning library in Rust, with CUDA kernels for the parts that matter.
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
//! ```text
//!   data     Dataset ─► DataLoader ─► Batch
//!                                       │
//!   nn       Module ──► forward ────────┤   layers hold Param handles
//!                                       ▼
//!   tensor   Tensor ops on CPU or CUDA ─┴─► loss
//!                                       │
//!   autograd loss.backward() walks the graph the ops built,
//!            depositing gradients in each Param's slot
//!                                       │
//!   optim    opt.step() reads those slots and updates in place
//! ```
//!
//! Each layer is one module directory and is documented where it lives:
//!
//! - [`tensor`] — the array type, its ops, and CPU/CUDA dispatch
//! - [`autograd`] — the graph the ops build and the reverse pass over it
//! - [`nn`] — layers, [`Param`](nn::Param) handles, and losses
//! - [`optim`] — optimizers and learning-rate schedules
//! - [`data`] — datasets, batching, MNIST
//! - [`serialize`] — checkpoints
//! - [`cuda`] — GPU memory and the kernel bindings
//!
//! ## Errors
//!
//! Tensor maths panics on shape and device mistakes — those are bugs in the
//! calling code, like indexing past the end of a slice. [`Result`] is for things
//! that genuinely fail at runtime: I/O, dataset downloads, and CUDA.
//!
//! ## Devices
//!
//! Everything starts on the CPU. [`Module::to_device`](nn::Module::to_device)
//! moves a model, [`DataLoader::to_device`](data::DataLoader::to_device) moves
//! its batches, and mixing the two panics rather than copying silently.
//!
//! ```no_run
//! # use fastnn::prelude::*;
//! # let model = Sequential::new();
//! # let dataset: fastnn::data::TensorDataset = unimplemented!();
//! let device = Device::best();
//! model.to_device(device);
//! let loader = DataLoader::new(&dataset, 64).to_device(device);
//! ```

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

/// Everything you need for a training script.
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
