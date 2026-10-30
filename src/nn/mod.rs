//! Layers, losses and decoding helpers. Everything implements [`Module`], and a
//! layer's weights are shared [`Param`] handles so an optimizer can hold them too.
//!
//! ```
//! use fastnn::prelude::*;
//!
//! let cnn = Sequential::new()
//!     .add(Conv2d::new(1, 8, 3, 1, 1))
//!     .add(ReLU)
//!     .add(MaxPool2d::new(2))
//!     .add(Flatten::new())
//!     .add(Linear::new(8 * 14 * 14, 10));
//!
//! let logits = cnn.forward(&Tensor::randn(&[4, 1, 28, 28]));
//! assert_eq!(logits.shape(), &[4, 10]);
//! ```
//!
//! - Layers: [`Linear`], [`Conv1d`], [`Conv2d`], [`ConvTranspose1d`],
//!   [`ConvTranspose2d`], [`MaxPool2d`], [`AvgPool2d`], [`AdaptiveAvgPool2d`],
//!   [`BatchNorm2d`], [`LayerNorm`], [`RMSNorm`], [`Dropout`], [`Embedding`],
//!   [`PositionalEncoding`], [`LSTM`] and [`GRU`].
//! - Transformers: [`MultiHeadAttention`], [`TransformerStack`] as an encoder or a
//!   causal decoder, [`TransformerDecoder`] with cross-attention, and the
//!   encoder-decoder [`Transformer`].
//! - Generation: [`KvCache`] and [`StackCache`] for incremental decoding,
//!   [`Sampler`] for temperature, top-k and top-p, and [`BeamSearch`].
//! - Losses: [`cross_entropy`], [`mse`], [`bce_with_logits`], [`huber`],
//!   [`kl_divergence`], [`dice`], [`triplet_margin`], [`info_nce`] and the rest of
//!   [`loss`] and [`metric_losses`]. [`CrossEntropyLoss`] adds label smoothing,
//!   class weights and an ignore index.
//! - Plumbing: [`Module`], [`Sequential`], [`Param`] and [`Buffer`].

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

pub use activation::{GELU, LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Tanh};
pub use attention::MultiHeadAttention;
pub use beam::BeamSearch;
pub use cache::{KvCache, StackCache};
pub use conv::Conv2d;
pub use conv_transpose::{ConvTranspose1d, ConvTranspose2d};
pub use conv1d::Conv1d;
pub use decoder::{Transformer, TransformerDecoder, TransformerDecoderBlock};
pub use dropout::Dropout;
pub use embedding::{Embedding, PositionalEncoding};
pub use linear::Linear;
pub use loss::{
    CrossEntropyLoss, Reduction, bce, bce_with_logits, cross_entropy, dice, focal_bce_with_logits,
    gaussian_nll, huber, kl_divergence, mae, mse, nll, poisson_nll, smooth_l1,
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
