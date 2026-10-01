//! Transformer blocks.

use crate::tensor::Tensor;

use super::attention::MultiHeadAttention;
use super::cache::{KvCache, StackCache};
use super::dropout::Dropout;
use super::linear::Linear;
use super::module::{scoped, Module};
use super::norm::LayerNorm;
use super::param::Param;

/// Which non-linearity the feed-forward sub-layer uses.
#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    GELU,
}

impl Activation {
    pub(crate) fn apply(self, x: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => x.relu(),
            Activation::GELU => x.gelu(),
        }
    }
}

/// One pre-norm transformer block: attention, then a feed-forward network, each
/// wrapped in a residual connection.
///
/// Pre-norm — normalizing the *input* to each sub-layer rather than the output —
/// leaves the residual path unnormalized from embedding to logits, which is what
/// lets deep stacks train without a learning-rate warmup schedule tuned per depth.
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub norm_attention: LayerNorm,
    pub norm_feedforward: LayerNorm,
    pub up: Linear,
    pub down: Linear,
    dropout: Dropout,
    activation: Activation,
    causal: bool,
}

impl TransformerBlock {
    /// An encoder block: every position sees every other.
    pub fn encoder(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        dropout: f32,
    ) -> TransformerBlock {
        TransformerBlock::new(
            model_dim,
            heads,
            hidden_dim,
            dropout,
            Activation::GELU,
            false,
        )
    }

    /// A decoder block: each position sees only itself and what came before.
    pub fn causal(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        dropout: f32,
    ) -> TransformerBlock {
        TransformerBlock::new(
            model_dim,
            heads,
            hidden_dim,
            dropout,
            Activation::GELU,
            true,
        )
    }

    pub fn new(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        dropout: f32,
        activation: Activation,
        causal: bool,
    ) -> TransformerBlock {
        TransformerBlock {
            attention: MultiHeadAttention::new(model_dim, heads, dropout),
            norm_attention: LayerNorm::new(model_dim),
            norm_feedforward: LayerNorm::new(model_dim),
            up: Linear::new(model_dim, hidden_dim),
            down: Linear::new(hidden_dim, model_dim),
            dropout: Dropout::new(dropout),
            activation,
            causal,
        }
    }

    /// [`forward`](Module::forward) with a key padding mask: `[batch, len]`,
    /// 1 for real positions and 0 for padding, so a batch of unequal-length
    /// sequences attends only to its own content.
    pub fn forward_masked(&self, input: &Tensor, key_mask: Option<&Tensor>) -> Tensor {
        let normed = self.norm_attention.forward(input);
        let attended =
            self.attention
                .attend_masked(&normed, &normed, &normed, self.causal, key_mask);
        let residual = input.add(&self.dropout.forward(&attended));

        let normed = self.norm_feedforward.forward(&residual);
        let hidden = self.activation.apply(&self.up.forward(&normed));
        residual.add(&self.dropout.forward(&self.down.forward(&hidden)))
    }

    /// [`forward`](Module::forward) with the attention keys and values cached
    /// for incremental decoding.
    ///
    /// Inference-only — dropout is skipped — and causal by nature: a cache only
    /// makes sense when later tokens cannot change earlier ones.
    pub fn forward_cached(&self, input: &Tensor, cache: &mut KvCache) -> Tensor {
        assert!(
            self.causal,
            "kv-cached decoding needs a causal block; this one is an encoder"
        );

        let normed = self.norm_attention.forward(input);
        let residual = input.add(&self.attention.attend_cached(&normed, cache));

        let normed = self.norm_feedforward.forward(&residual);
        let hidden = self.activation.apply(&self.up.forward(&normed));
        residual.add(&self.down.forward(&hidden))
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_masked(input, None)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut params = scoped("attention", self.attention.named_parameters());
        params.extend(scoped(
            "norm_attention",
            self.norm_attention.named_parameters(),
        ));
        params.extend(scoped(
            "norm_feedforward",
            self.norm_feedforward.named_parameters(),
        ));
        params.extend(scoped("up", self.up.named_parameters()));
        params.extend(scoped("down", self.down.named_parameters()));
        params
    }

    fn set_training(&self, training: bool) {
        self.attention.set_training(training);
        self.dropout.set_training(training);
    }
}

/// A stack of blocks with a final normalization.
pub struct TransformerStack {
    pub blocks: Vec<TransformerBlock>,
    pub norm: LayerNorm,
}

impl TransformerStack {
    /// A bidirectional encoder stack.
    pub fn encoder(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        layers: usize,
        dropout: f32,
    ) -> TransformerStack {
        TransformerStack::build(layers, model_dim, || {
            TransformerBlock::encoder(model_dim, heads, hidden_dim, dropout)
        })
    }

    /// A causal decoder stack, as used by GPT-style language models.
    pub fn causal(
        model_dim: usize,
        heads: usize,
        hidden_dim: usize,
        layers: usize,
        dropout: f32,
    ) -> TransformerStack {
        TransformerStack::build(layers, model_dim, || {
            TransformerBlock::causal(model_dim, heads, hidden_dim, dropout)
        })
    }

    fn build(
        layers: usize,
        model_dim: usize,
        block: impl Fn() -> TransformerBlock,
    ) -> TransformerStack {
        TransformerStack {
            blocks: (0..layers).map(|_| block()).collect(),
            norm: LayerNorm::new(model_dim),
        }
    }

    /// A cache sized for this stack, ready for [`forward_cached`](Self::forward_cached).
    pub fn new_cache(&self) -> StackCache {
        StackCache::new(self.blocks.len())
    }

    /// [`forward`](Module::forward) with a key padding mask applied in every
    /// block. See [`TransformerBlock::forward_masked`].
    pub fn forward_masked(&self, input: &Tensor, key_mask: Option<&Tensor>) -> Tensor {
        let hidden = self
            .blocks
            .iter()
            .fold(input.clone(), |x, block| block.forward_masked(&x, key_mask));
        self.norm.forward(&hidden)
    }

    /// [`forward`](Module::forward) through per-block KV caches, for feeding a
    /// generation loop one token at a time. See [`super::cache`].
    pub fn forward_cached(&self, input: &Tensor, cache: &mut StackCache) -> Tensor {
        assert_eq!(
            cache.layers.len(),
            self.blocks.len(),
            "cache has {} layers but the stack has {} blocks",
            cache.layers.len(),
            self.blocks.len()
        );
        let hidden = self
            .blocks
            .iter()
            .zip(&mut cache.layers)
            .fold(input.clone(), |x, (block, layer)| {
                block.forward_cached(&x, layer)
            });
        self.norm.forward(&hidden)
    }
}

impl Module for TransformerStack {
    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = self
            .blocks
            .iter()
            .fold(input.clone(), |x, block| block.forward(&x));
        self.norm.forward(&hidden)
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
