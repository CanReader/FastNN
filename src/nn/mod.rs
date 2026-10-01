//! Neural network layers.
//!
//! Everything implements [`Module`]. A layer's weights are [`Param`] handles —
//! shared slots, not owned values — which is why the optimizer can hold them at
//! the same time as the model:
//!
//! ```
//! use fastnn::prelude::*;
//!
//! let model = Sequential::new()
//!     .add(Linear::new(8, 16))
//!     .add(ReLU)
//!     .add(Linear::new(16, 2));
//!
//! let mut opt = SGD::new(model.parameters(), 0.1);
//!
//! let logits = model.forward(&Tensor::randn(&[4, 8]));
//! let loss = cross_entropy(&logits, &[0, 1, 1, 0]);
//!
//! opt.zero_grad();
//! loss.backward();
//! opt.step();
//! ```
//!
//! ## Writing a layer
//!
//! Implement [`forward`](Module::forward), and [`named_parameters`](Module::named_parameters)
//! if it has weights. Device placement, parameter counting, and gradient zeroing
//! come from those for free.

pub mod activation;
pub mod attention;
pub mod beam;
pub mod cache;
pub mod conv;
pub mod conv1d;
pub mod conv_transpose;
pub mod decoder;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod loss;
pub mod metric_losses;
pub mod module;
pub mod norm;
pub mod param;
pub mod pooling;
pub mod rnn;
pub mod sample;
pub mod sequential;
pub mod shape;
pub mod transformer;

pub use activation::{LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Tanh, GELU};
pub use attention::MultiHeadAttention;
pub use beam::BeamSearch;
pub use cache::{KvCache, StackCache};
pub use conv::Conv2d;
pub use conv1d::Conv1d;
pub use conv_transpose::{ConvTranspose1d, ConvTranspose2d};
pub use decoder::{Transformer, TransformerDecoder, TransformerDecoderBlock};
pub use dropout::Dropout;
pub use embedding::{Embedding, PositionalEncoding};
pub use linear::Linear;
pub use loss::{
    bce, bce_with_logits, cross_entropy, dice, focal_bce_with_logits, gaussian_nll, huber,
    kl_divergence, mae, mse, nll, poisson_nll, smooth_l1, CrossEntropyLoss, Reduction,
};
pub use metric_losses::{contrastive, cosine_embedding, info_nce, margin_ranking, triplet_margin};
pub use module::Module;
pub use norm::{BatchNorm2d, LayerNorm, RMSNorm};
pub use param::{Buffer, Param};
pub use pooling::{AdaptiveAvgPool2d, AvgPool2d, MaxPool2d};
pub use rnn::{GRU, LSTM};
pub use sample::Sampler;
pub use sequential::Sequential;
pub use shape::{Flatten, Reshape};
pub use transformer::{Activation, TransformerBlock, TransformerStack};
