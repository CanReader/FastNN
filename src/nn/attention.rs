//! Multi-head scaled dot-product attention.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::tensor::{Device, Tensor};

use super::cache::KvCache;
use super::dropout::Dropout;
use super::linear::Linear;
use super::module::{scoped, Module};
use super::param::Param;

/// Large negative added to masked scores, so softmax gives them zero weight.
const MASKED: f32 = -1e9;

/// Attention over `[batch, sequence, embed_dim]`. The head axis is folded into the
/// batch, so one batched GEMM covers every head.
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
            // dot products grow with head_dim and would saturate the softmax
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Attend from `query` over `key`/`value`. With `causal`, position `i` sees only `≤ i`.
    pub fn attend(&self, query: &Tensor, key: &Tensor, value: &Tensor, causal: bool) -> Tensor {
        self.attend_masked(query, key, value, causal, None)
    }

    /// [`attend`](Self::attend) with an optional padding mask, `[batch, kv_len]`,
    /// 1 for real keys and 0 for padding.
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

        let mut scores = q.matmul_nt(&k).mul_scalar(self.scale);
        if causal {
            // additive rather than an overwrite, so it stays in the graph
            scores = scores.add(&causal_mask(lanes, q_len, kv_len, scores.device()));
        }
        if let Some(mask) = key_mask {
            assert_eq!(
                mask.shape(),
                &[batch, kv_len],
                "key mask must be [batch, kv_len] = [{batch}, {kv_len}]"
            );
            // 1/0 becomes 0/MASKED, repeated per head to match the folded batch
            let bias = mask
                .add_scalar(-1.0)
                .mul_scalar(-MASKED)
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

    /// Causal self-attention over the new tokens plus `cache`, which it extends.
    /// Inference only: no dropout, run under [`no_grad`](crate::autograd::no_grad).
    pub fn attend_cached(&self, input: &Tensor, cache: &mut KvCache) -> Tensor {
        let (batch, new_len, embed_dim) = (input.dim(0), input.dim(1), input.dim(2));

        let q = self.split_heads(&self.query.forward(input), batch, new_len);
        let k = self.split_heads(&self.key.forward(input), batch, new_len);
        let v = self.split_heads(&self.value.forward(input), batch, new_len);
        let (k, v) = cache.append(&k, &v);
        let kv_len = k.dim(1);

        let mut scores = q.matmul_nt(k).mul_scalar(self.scale);
        // one new token may see everything, so only multi-token steps need a mask
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
    /// Unmasked self-attention. Use [`attend`](MultiHeadAttention::attend) for the rest.
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
    /// Every layer wants the same mask for a given shape, so build each once.
    static MASKS: RefCell<HashMap<(usize, usize, usize, Device), Tensor>> =
        RefCell::new(HashMap::new());
}

/// `[lanes, q_len, kv_len]`, zero where attention is allowed and [`MASKED`] elsewhere.
/// When `kv_len > q_len` the queries are the last `q_len` positions (a cached step).
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
                            MASKED
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
