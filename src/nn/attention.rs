//! Multi-head scaled dot-product attention.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::tensor::{Device, Tensor};

use super::cache::KvCache;
use super::dropout::Dropout;
use super::linear::Linear;
use super::module::{scoped, Module};
use super::param::Param;

/// Attention over `[batch, sequence, embed_dim]`.
///
/// Heads are not separate tensors: one projection produces all of them, and the
/// head axis is folded into the batch so a single batched GEMM covers every head.
pub struct MultiHeadAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    dropout: Dropout,
    heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, heads: usize, dropout: f32) -> MultiHeadAttention {
        assert_eq!(
            embed_dim % heads,
            0,
            "embed_dim {embed_dim} must divide into {heads} heads"
        );
        let head_dim = embed_dim / heads;
        MultiHeadAttention {
            query: Linear::new(embed_dim, embed_dim),
            key: Linear::new(embed_dim, embed_dim),
            value: Linear::new(embed_dim, embed_dim),
            output: Linear::new(embed_dim, embed_dim),
            dropout: Dropout::new(dropout),
            heads,
            head_dim,
            // Without this, dot products grow with head_dim and softmax saturates.
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Attend from `query` over `key`/`value`.
    ///
    /// With `causal` set, position `i` can only see positions `≤ i` — what a
    /// language model needs so it cannot read ahead.
    pub fn attend(&self, query: &Tensor, key: &Tensor, value: &Tensor, causal: bool) -> Tensor {
        self.attend_masked(query, key, value, causal, None)
    }

    /// [`attend`](Self::attend) with an optional key padding mask.
    ///
    /// `key_mask` is `[batch, kv_len]`: 1 where a key position is real, 0 where
    /// it is padding. Padded positions get the same additive `-1e9` treatment as
    /// the causal mask, so softmax gives them zero weight and batches of
    /// unequal-length sequences attend only to their own content.
    pub fn attend_masked(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
        key_mask: Option<&Tensor>,
    ) -> Tensor {
        let (batch, q_len, embed_dim) = (query.dim(0), query.dim(1), query.dim(2));
        let kv_len = key.dim(1);
        let lanes = batch * self.heads;

        let q = self.split_heads(&self.query.forward(query), batch, q_len);
        let k = self.split_heads(&self.key.forward(key), batch, kv_len);
        let v = self.split_heads(&self.value.forward(value), batch, kv_len);

        // matmul_nt gives q · kᵀ without ever materialising kᵀ.
        let mut scores = q.matmul_nt(&k).mul_scalar(self.scale);
        if causal {
            // An additive -inf bias rather than an in-place overwrite: the mask
            // stays part of the graph, so gradients still flow through softmax.
            scores = scores.add(&causal_mask(lanes, q_len, kv_len, scores.device()));
        }
        if let Some(mask) = key_mask {
            assert_eq!(
                mask.shape(),
                &[batch, kv_len],
                "key mask must be [batch, kv_len] = [{batch}, {kv_len}]"
            );
            // 1/0 keep/pad → 0/-1e9, repeated per head so it broadcasts over
            // the [batch·heads, q_len, kv_len] scores.
            let bias = mask
                .add_scalar(-1.0)
                .mul_scalar(1e9)
                .reshape(&[batch as i64, 1, kv_len as i64])
                .expand(&[batch, self.heads, kv_len])
                .reshape(&[lanes as i64, 1, kv_len as i64]);
            scores = scores.add(&bias);
        }

        let weights = self.dropout.forward(&scores.softmax());
        let attended = weights.matmul(&v);

        self.output
            .forward(&self.merge_heads(&attended, batch, q_len, embed_dim))
    }

    /// Causal self-attention over `input`'s new tokens plus everything in
    /// `cache`, extending the cache as it goes.
    ///
    /// Inference-only: dropout is skipped and the cache stores detached
    /// tensors, so run it under [`no_grad`](crate::autograd::no_grad). An empty
    /// cache makes this a plain causal forward (the prefill); after that, feed
    /// one token at a time and only the new projections are computed.
    pub fn attend_cached(&self, input: &Tensor, cache: &mut KvCache) -> Tensor {
        let (batch, new_len, embed_dim) = (input.dim(0), input.dim(1), input.dim(2));

        let q = self.split_heads(&self.query.forward(input), batch, new_len);
        let k = self.split_heads(&self.key.forward(input), batch, new_len);
        let v = self.split_heads(&self.value.forward(input), batch, new_len);
        let (k, v) = cache.append(&k, &v);
        let kv_len = k.dim(1);

        let mut scores = q.matmul_nt(k).mul_scalar(self.scale);
        // A single new token sits at the end of the sequence and may see all of
        // it, so only multi-token steps need masking.
        if new_len > 1 {
            scores = scores.add(&causal_mask(
                batch * self.heads,
                new_len,
                kv_len,
                scores.device(),
            ));
        }

        let attended = scores.softmax().matmul(v);
        self.output
            .forward(&self.merge_heads(&attended, batch, new_len, embed_dim))
    }

    /// `[batch, len, embed]` → `[batch·heads, len, head_dim]`.
    fn split_heads(&self, x: &Tensor, batch: usize, len: usize) -> Tensor {
        x.reshape(&[
            batch as i64,
            len as i64,
            self.heads as i64,
            self.head_dim as i64,
        ])
        .permute(&[0, 2, 1, 3])
        .reshape(&[
            (batch * self.heads) as i64,
            len as i64,
            self.head_dim as i64,
        ])
    }

    /// `[batch·heads, len, head_dim]` → `[batch, len, embed]`.
    fn merge_heads(&self, x: &Tensor, batch: usize, len: usize, embed_dim: usize) -> Tensor {
        x.reshape(&[
            batch as i64,
            self.heads as i64,
            len as i64,
            self.head_dim as i64,
        ])
        .permute(&[0, 2, 1, 3])
        .reshape(&[batch as i64, len as i64, embed_dim as i64])
    }
}

impl Module for MultiHeadAttention {
    /// Self-attention with no mask. Use [`attend`](MultiHeadAttention::attend)
    /// for cross-attention or causal masking.
    fn forward(&self, input: &Tensor) -> Tensor {
        self.attend(input, input, input, false)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        [
            ("query", &self.query),
            ("key", &self.key),
            ("value", &self.value),
            ("output", &self.output),
        ]
        .into_iter()
        .flat_map(|(name, layer)| scoped(name, layer.named_parameters()))
        .collect()
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

thread_local! {
    /// Causal masks depend only on shape and device, and every layer of every
    /// step wants the same one — so build each exactly once per thread.
    static MASKS: RefCell<HashMap<(usize, usize, usize, Device), Tensor>> =
        RefCell::new(HashMap::new());
}

/// `[lanes, q_len, kv_len]`, zero where attention is allowed and a large
/// negative where it is not, so softmax drives those weights to zero.
///
/// When `kv_len > q_len` the queries are the *last* `q_len` positions of the
/// sequence — a cached decoding step — so each sees the whole prefix plus
/// itself. With equal lengths this is the ordinary causal triangle.
fn causal_mask(lanes: usize, q_len: usize, kv_len: usize, device: Device) -> Tensor {
    MASKS.with(|cache| {
        cache
            .borrow_mut()
            .entry((lanes, q_len, kv_len, device))
            .or_insert_with(|| {
                let offset = kv_len.saturating_sub(q_len);
                let data: Vec<f32> = (0..lanes * q_len * kv_len)
                    .map(|i| {
                        let query_pos = (i / kv_len) % q_len;
                        let key_pos = i % kv_len;
                        if key_pos > query_pos + offset {
                            -1e9
                        } else {
                            0.0
                        }
                    })
                    .collect();
                Tensor::from_vec(data, &[lanes, q_len, kv_len]).to(device)
            })
            .clone()
    })
}
