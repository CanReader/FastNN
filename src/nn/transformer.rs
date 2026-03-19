use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::nn::{Linear, LayerNorm, Dropout};

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

        // Apply causal mask
        let scores = if causal {
            let mut score_data = scores.to_vec();
            for bh in 0..(batch * self.num_heads) {
                for qi in 0..seq_len_q {
                    for ki in 0..seq_len_k {
                        if ki > qi {
                            score_data[(bh * seq_len_q + qi) * seq_len_k + ki] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            Tensor::from_vec(score_data, scores.shape())
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
        // Self-attention: Q=K=V=input
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
        // Pre-LN Transformer: norm -> attn -> residual -> norm -> ffn -> residual
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
}
