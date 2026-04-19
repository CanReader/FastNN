//! # FastNN — GPU-Accelerated Deep Learning Library
//!
//! A deep learning library built from scratch in Rust with CUDA GPU acceleration.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use fastnn::prelude::*;
//!
//! // Create tensors
//! let x = Tensor::randn(&[32, 784]);
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU)
//!     .add(Linear::new(256, 10));
//!
//! let output = model.forward(&x);
//! ```

pub mod tensor;
pub mod autograd;
pub mod nn;
pub mod optim;
pub mod data;
pub mod cuda;
pub mod serialize;
pub mod utils;

/// Prelude — import everything you need with `use fastnn::prelude::*`.
pub mod prelude {
    pub use crate::tensor::{Tensor, Device};
    pub use crate::nn::{
        Module, Sequential, Linear, Conv2d,
        ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, Softmax,
        BatchNorm2d, LayerNorm, RMSNorm,
        Dropout,
        MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
        LSTM, GRU,
        MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder,
        Embedding,
        CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss,
    };
    pub use crate::optim::{Optimizer, SGD, Adam, AdamW};
    pub use crate::optim::{LRScheduler, StepLR, CosineAnnealingLR, LinearWarmup, OneCycleLR};
    pub use crate::data::{Dataset, DataLoader};
    pub use crate::autograd::{Variable, BackwardGraph};
    pub use crate::autograd::graph::{enable_grad, disable_grad, is_grad_enabled};
    pub use crate::serialize::{save_model, load_model, save_tensors, load_tensors};
    pub use crate::cuda::CudaContext;
    pub use crate::utils::random::manual_seed;
}
