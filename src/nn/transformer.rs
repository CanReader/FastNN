use std::collections::HashMap;
use std::cell::RefCell;
use crate::tensor::{Tensor, Device};
use crate::nn::module::Module;
use crate::nn::{Linear, LayerNorm, Dropout};

// Per-thread cache for causal masks — keyed by (n, seq_q, seq_k, device).
// Masks are identical across steps for fixed batch/heads/seq, so we compute
// once and reuse. Using thread-local avoids any locking overhead.
thread_local! {
    static CAUSAL_MASK_CACHE: RefCell<HashMap<(usize, usize, usize, usize), Tensor>> =
        RefCell::new(HashMap::new());
}

fn get_causal_mask(n: usize, seq_q: usize, seq_k: usize, device: Device) -> Tensor {
    // Device as usize key: CPU=0, Cuda(id)=id+1
    let dev_key = match device { Device::Cpu => 0, Device::Cuda(id) => id + 1 };
    CAUSAL_MASK_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        map.entry((n, seq_q, seq_k, dev_key))
            .or_insert_with(|| {
                let mask_data: Vec<f32> = (0..(n * seq_q * seq_k))
                    .map(|idx| {
                        let qi = (idx / seq_k) % seq_q;
                        let ki = idx % seq_k;
                        if ki > qi { -1e9 } else { 0.0 }
                    })
                    .collect();
                Tensor::from_vec(mask_data, &[n, seq_q, seq_k]).to_device(device)
            })
            .clone()
    })
}

/// Multi-Head Attention mechanism.
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;

        MultiHeadAttention {
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            dropout: Dropout::new(dropout),
        }
    }

    /// Forward pass with optional causal mask.
    /// input shape: [batch, seq_len, embed_dim]
    /// Returns: [batch, seq_len, embed_dim]
    pub fn forward_attn(&self, query: &Tensor, key: &Tensor, value: &Tensor, causal: bool) -> Tensor {
        let shape = query.shape();
        let (batch, seq_len_q, _embed_dim) = (shape[0], shape[1], shape[2]);
        let seq_len_k = key.shape()[1];

        // Project Q, K, V
        let q = self.q_proj.forward(query);  // [batch, seq_q, embed_dim]
        let k = self.k_proj.forward(key);    // [batch, seq_k, embed_dim]
        let v = self.v_proj.forward(value);  // [batch, seq_k, embed_dim]

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q.reshape(&[batch as i64, seq_len_q as i64, self.num_heads as i64, self.head_dim as i64])
                 .permute(&[0, 2, 1, 3]);
        let k = k.reshape(&[batch as i64, seq_len_k as i64, self.num_heads as i64, self.head_dim as i64])
                 .permute(&[0, 2, 1, 3]);
        let v = v.reshape(&[batch as i64, seq_len_k as i64, self.num_heads as i64, self.head_dim as i64])
                 .permute(&[0, 2, 1, 3]);

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // q: [batch, heads, seq_q, head_dim], k^T: [batch, heads, head_dim, seq_k]
        let k_t = k.permute(&[0, 1, 3, 2]);

        // Flatten batch and heads for matmul
        let q_flat = q.reshape(&[(batch * self.num_heads) as i64, seq_len_q as i64, self.head_dim as i64]);
        let k_t_flat = k_t.reshape(&[(batch * self.num_heads) as i64, self.head_dim as i64, seq_len_k as i64]);
        let v_flat = v.reshape(&[(batch * self.num_heads) as i64, seq_len_k as i64, self.head_dim as i64]);

        let scores = q_flat.matmul(&k_t_flat).mul_scalar(self.scale);
        // scores: [batch*heads, seq_q, seq_k]

        // Apply causal mask via additive bias (preserves gradient graph).
        // Mask is identical for fixed (batch, heads, seq) so we cache it.
        let scores = if causal {
            let n = batch * self.num_heads;
            let mask = get_causal_mask(n, seq_len_q, seq_len_k, scores.device());
            scores.add(&mask)
        } else {
            scores
        };

        // Softmax over key dimension
        let attn_weights = scores.reshape(&[(batch * self.num_heads * seq_len_q) as i64, seq_len_k as i64])
                                 .softmax()
                                 .reshape(&[(batch * self.num_heads) as i64, seq_len_q as i64, seq_len_k as i64]);

        let attn_weights = self.dropout.forward(&attn_weights);

        // Attention output: attn_weights @ V
        let attn_output = attn_weights.matmul(&v_flat);
        // [batch*heads, seq_q, head_dim]

        // Reshape back: [batch, heads, seq_q, head_dim] -> [batch, seq_q, embed_dim]
        let attn_output = attn_output
            .reshape(&[batch as i64, self.num_heads as i64, seq_len_q as i64, self.head_dim as i64])
            .permute(&[0, 2, 1, 3])
            .reshape(&[batch as i64, seq_len_q as i64, (self.num_heads * self.head_dim) as i64]);

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_attn(input, input, input, false)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters_mut());
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        params
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.q_proj.to_device(device);
        self.k_proj.to_device(device);
        self.v_proj.to_device(device);
        self.out_proj.to_device(device);
    }
}

/// Transformer Encoder Layer (Pre-LN variant).
pub struct TransformerEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout: Dropout,
    pub activation: ActivationType,
}

#[derive(Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, nhead: usize, d_ff: usize, dropout: f32) -> Self {
        Self::with_activation(d_model, nhead, d_ff, dropout, ActivationType::GELU)
    }

    pub fn with_activation(d_model: usize, nhead: usize, d_ff: usize, dropout: f32, activation: ActivationType) -> Self {
        TransformerEncoderLayer {
            self_attn: MultiHeadAttention::new(d_model, nhead, dropout),
            linear1: Linear::new(d_model, d_ff),
            linear2: Linear::new(d_ff, d_model),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(dropout),
            activation,
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let normed = self.norm1.forward(input);
        let attn_out = self.self_attn.forward_attn(&normed, &normed, &normed, false);
        let x = input.add(&self.dropout.forward(&attn_out));

        let normed2 = self.norm2.forward(&x);
        let ff = self.linear1.forward(&normed2);
        let ff = match self.activation {
            ActivationType::ReLU => ff.relu(),
            ActivationType::GELU => ff.gelu(),
        };
        let ff = self.linear2.forward(&ff);
        x.add(&self.dropout.forward(&ff))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.self_attn.to_device(device);
        self.linear1.to_device(device);
        self.linear2.to_device(device);
        self.norm1.to_device(device);
        self.norm2.to_device(device);
    }
}

/// Stack of Transformer Encoder Layers.
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub norm: LayerNorm,
}

impl TransformerEncoder {
    pub fn new(d_model: usize, nhead: usize, d_ff: usize, num_layers: usize, dropout: f32) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, nhead, d_ff, dropout))
            .collect();
        TransformerEncoder {
            layers,
            norm: LayerNorm::new(&[d_model]),
        }
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.norm.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.norm.parameters_mut());
        params
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        for layer in &mut self.layers {
            layer.to_device(device);
        }
        self.norm.to_device(device);
    }
}
