//! Decoder blocks with cross-attention over encoder output (the memory), and the
//! encoder-decoder [`Transformer`]. Decoder-only models just use [`TransformerStack::causal`].

use crate::tensor::Tensor;

use super::attention::MultiHeadAttention;
use super::dropout::Dropout;
use super::linear::Linear;
use super::module::{scoped, Module};
use super::norm::LayerNorm;
use super::param::Param;
use super::transformer::{Activation, TransformerStack};

/// Pre-norm decoder block: causal self-attention, cross-attention, feed-forward.
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

    /// [`decode`](Self::decode) with a memory padding mask, `[batch, source_len]`,
    /// 1 for real positions. The memory is already normed by the encoder, so it is used as-is.
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
    /// Attends over the input as its own memory. Real use goes through [`decode`](Self::decode).
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
    /// Attends over the input as its own memory.
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

/// Encoder-decoder Transformer trunk. Embeddings and the output head stay outside.
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

    /// [`run`](Self::run) with a source padding mask, used by the encoder and the cross-attention.
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
    /// Encodes and decodes the same sequence. Real use goes through [`run`](Self::run).
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
