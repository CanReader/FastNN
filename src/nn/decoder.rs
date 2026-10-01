//! Transformer decoder blocks and the full encoder-decoder Transformer.
//!
//! A decoder block extends the encoder block with one more sub-layer: after
//! causal self-attention, each target position attends over the encoder's
//! output (the *memory*). That is what lets translation, summarisation, and
//! any other sequence-to-sequence model read its input while writing its
//! output. Decoder-only models like GPT do not need this file —
//! [`TransformerStack::causal`] is the whole architecture.

use crate::tensor::Tensor;

use super::attention::MultiHeadAttention;
use super::dropout::Dropout;
use super::linear::Linear;
use super::module::{scoped, Module};
use super::norm::LayerNorm;
use super::param::Param;
use super::transformer::{Activation, TransformerStack};

/// One pre-norm decoder block: causal self-attention, cross-attention over the
/// encoder memory, then a feed-forward network, each wrapped in a residual.
pub struct TransformerDecoderBlock {
    pub self_attention: MultiHeadAttention,
    pub cross_attention: MultiHeadAttention,
    pub norm_self: LayerNorm,
    pub norm_cross: LayerNorm,
    pub norm_feedforward: LayerNorm,
    pub up: Linear,
    pub down: Linear,
    dropout: Dropout,
    activation: Activation,
}

impl TransformerDecoderBlock {
    pub fn new(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        dropout: f32,
    ) -> TransformerDecoderBlock {
        TransformerDecoderBlock {
            self_attention: MultiHeadAttention::new(model_dim, heads, dropout),
            cross_attention: MultiHeadAttention::new(model_dim, heads, dropout),
            norm_self: LayerNorm::new(model_dim),
            norm_cross: LayerNorm::new(model_dim),
            norm_feedforward: LayerNorm::new(model_dim),
            up: Linear::new(model_dim, hidden_dim),
            down: Linear::new(hidden_dim, model_dim),
            dropout: Dropout::new(dropout),
            activation: Activation::GELU,
        }
    }

    /// Decode `target` while attending over the encoder's `memory`.
    pub fn decode(&self, target: &Tensor, memory: &Tensor) -> Tensor {
        self.decode_masked(target, memory, None)
    }

    /// [`decode`](Self::decode) with a padding mask for the memory:
    /// `[batch, source_len]`, 1 for real encoder positions and 0 for padding.
    ///
    /// The query side is normalized per sub-layer as usual; the memory arrives
    /// already normalized by the encoder's final norm and is used as-is.
    pub fn decode_masked(
        &self,
        target: &Tensor,
        memory: &Tensor,
        memory_mask: Option<&Tensor>,
    ) -> Tensor {
        let normed = self.norm_self.forward(target);
        let attended = self.self_attention.attend(&normed, &normed, &normed, true);
        let x = target.add(&self.dropout.forward(&attended));

        let normed = self.norm_cross.forward(&x);
        let attended =
            self.cross_attention
                .attend_masked(&normed, memory, memory, false, memory_mask);
        let x = x.add(&self.dropout.forward(&attended));

        let normed = self.norm_feedforward.forward(&x);
        let hidden = self.activation.apply(&self.up.forward(&normed));
        x.add(&self.dropout.forward(&self.down.forward(&hidden)))
    }
}

impl Module for TransformerDecoderBlock {
    /// Degenerate single-input form: the target attends over itself as memory.
    /// Real sequence-to-sequence use goes through [`decode`](Self::decode).
    fn forward(&self, input: &Tensor) -> Tensor {
        self.decode(input, input)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("self_attention", self.self_attention.named_parameters());
        params.extend(scoped(
            "cross_attention",
            self.cross_attention.named_parameters(),
        ));
        params.extend(scoped("norm_self", self.norm_self.named_parameters()));
        params.extend(scoped("norm_cross", self.norm_cross.named_parameters()));
        params.extend(scoped(
            "norm_feedforward",
            self.norm_feedforward.named_parameters(),
        ));
        params.extend(scoped("up", self.up.named_parameters()));
        params.extend(scoped("down", self.down.named_parameters()));
        params
    }

    fn set_training(&self, training: bool) {
        self.self_attention.set_training(training);
        self.cross_attention.set_training(training);
        self.dropout.set_training(training);
    }
}

/// A stack of decoder blocks with a final normalization.
pub struct TransformerDecoder {
    pub blocks: Vec<TransformerDecoderBlock>,
    pub norm: LayerNorm,
}

impl TransformerDecoder {
    pub fn new(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        layers: usize,
        dropout: f32,
    ) -> TransformerDecoder {
        TransformerDecoder {
            blocks: (0..layers)
                .map(|_| TransformerDecoderBlock::new(model_dim, heads, hidden_dim, dropout))
                .collect(),
            norm: LayerNorm::new(model_dim),
        }
    }

    pub fn decode(&self, target: &Tensor, memory: &Tensor) -> Tensor {
        self.decode_masked(target, memory, None)
    }

    pub fn decode_masked(
        &self,
        target: &Tensor,
        memory: &Tensor,
        memory_mask: Option<&Tensor>,
    ) -> Tensor {
        let hidden = self.blocks.iter().fold(target.clone(), |x, block| {
            block.decode_masked(&x, memory, memory_mask)
        });
        self.norm.forward(&hidden)
    }
}

impl Module for TransformerDecoder {
    /// Degenerate single-input form; see [`TransformerDecoderBlock::forward`].
    fn forward(&self, input: &Tensor) -> Tensor {
        self.decode(input, input)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params: Vec<(String, Param)> = self
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(i, block)| scoped(i, block.named_parameters()))
            .collect();
        params.extend(scoped("norm", self.norm.named_parameters()));
        params
    }

    fn set_training(&self, training: bool) {
        for block in &self.blocks {
            block.set_training(training);
        }
    }
}

/// The full encoder-decoder Transformer: a bidirectional encoder over the
/// source, a causal decoder over the target with cross-attention into the
/// encoder's output.
///
/// Embeddings and the output head stay outside, as they do for
/// [`TransformerStack`] — this is the trunk, not the whole model.
pub struct Transformer {
    pub encoder: TransformerStack,
    pub decoder: TransformerDecoder,
}

impl Transformer {
    pub fn new(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        encoder_layers: usize,
        decoder_layers: usize,
        dropout: f32,
    ) -> Transformer {
        Transformer {
            encoder: TransformerStack::encoder(
                model_dim,
                heads,
                hidden_dim,
                encoder_layers,
                dropout,
            ),
            decoder: TransformerDecoder::new(model_dim, heads, hidden_dim, decoder_layers, dropout),
        }
    }

    /// Encode `source`, then decode `target` against it.
    pub fn run(&self, source: &Tensor, target: &Tensor) -> Tensor {
        self.run_masked(source, target, None)
    }

    /// [`run`](Self::run) with a source padding mask, applied both to the
    /// encoder's self-attention and to the decoder's cross-attention — the two
    /// places a padded source position could leak in.
    pub fn run_masked(
        &self,
        source: &Tensor,
        target: &Tensor,
        source_mask: Option<&Tensor>,
    ) -> Tensor {
        let memory = self.encoder.forward_masked(source, source_mask);
        self.decoder.decode_masked(target, &memory, source_mask)
    }
}

impl Module for Transformer {
    /// Degenerate single-input form: encode and decode the same sequence.
    /// Real sequence-to-sequence use goes through [`run`](Self::run).
    fn forward(&self, input: &Tensor) -> Tensor {
        self.run(input, input)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("encoder", self.encoder.named_parameters());
        params.extend(scoped("decoder", self.decoder.named_parameters()));
        params
    }

    fn set_training(&self, training: bool) {
        self.encoder.set_training(training);
        self.decoder.set_training(training);
    }
}
