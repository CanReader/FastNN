//! Cached keys and values for incremental decoding.
//!
//! Detached on purpose, so a cache never pins a graph. Run it under
//! [`no_grad`](crate::autograd::no_grad).

use crate::tensor::Tensor;

/// One attention layer's projected keys and values, each `[batch·heads, len, head_dim]`.
#[derive(Default)]
pub struct KvCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> KvCache {
        KvCache::default()
    }

    /// How many positions are cached.
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.dim(1))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything, ready for a new sequence.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
    }

    /// Append new keys and values, returning the full cached tensors.
    pub fn append(&mut self, keys: &Tensor, values: &Tensor) -> (&Tensor, &Tensor) {
        self.keys = Some(match &self.keys {
            Some(prior) => Tensor::cat(&[prior, &keys.detach()], 1),
            None => keys.detach(),
        });
        self.values = Some(match &self.values {
            Some(prior) => Tensor::cat(&[prior, &values.detach()], 1),
            None => values.detach(),
        });
        (self.keys.as_ref().unwrap(), self.values.as_ref().unwrap())
    }
}

/// One [`KvCache`] per block of a [`TransformerStack`](super::TransformerStack).
pub struct StackCache {
    pub layers: Vec<KvCache>,
}

impl StackCache {
    pub fn new(layers: usize) -> StackCache {
        StackCache {
            layers: (0..layers).map(|_| KvCache::new()).collect(),
        }
    }

    /// How many positions are cached, which is the next token's position index.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, KvCache::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything, ready for a new sequence.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}
