//! Cached keys and values for incremental decoding.
//!
//! Generating token by token through [`Module::forward`](super::Module::forward)
//! recomputes attention over the whole prefix for every new token. A cache keeps
//! each layer's projected keys and values, so a step only computes projections
//! for the tokens it has not seen — O(n) attention per token instead of O(n²).
//!
//! One [`KvCache`] serves one attention layer; a [`StackCache`] holds one per
//! block of a [`TransformerStack`](super::TransformerStack). The cached tensors
//! are detached, so holding a cache never pins an autograd graph — this is an
//! inference path, and callers should run it under
//! [`no_grad`](crate::autograd::no_grad).

use crate::tensor::Tensor;

/// The keys and values one attention layer has already projected, each
/// `[batch·heads, cached_len, head_dim]`.
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

    /// Extend the cache with newly projected keys and values, returning the
    /// full cached tensors to attend over. Stores detached copies, so the graph
    /// that produced the projections is free to go.
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

/// One [`KvCache`] per block of a transformer stack.
pub struct StackCache {
    pub layers: Vec<KvCache>,
}

impl StackCache {
    pub fn new(layers: usize) -> StackCache {
        StackCache {
            layers: (0..layers).map(|_| KvCache::new()).collect(),
        }
    }

    /// How many positions are cached — the next token's position index.
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
