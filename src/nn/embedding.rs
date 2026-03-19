use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::nn::module::Module;

/// Lookup table for token embeddings.
pub struct Embedding {
    pub weight: Tensor, // [num_embeddings, embedding_dim]
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut weight = Tensor::randn(&[num_embeddings, embedding_dim]);
        weight.set_requires_grad(true);

        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Look up embeddings for integer indices.
    /// indices: flat slice of token IDs.
    /// Returns: Tensor of shape [indices.len(), embedding_dim]
    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        let weight_data = self.weight.data();
        let mut output = vec![0.0f32; indices.len() * self.embedding_dim];

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.num_embeddings, "Index {} out of range (vocab size {})", idx, self.num_embeddings);
            let offset = idx * self.embedding_dim;
            output[i * self.embedding_dim..(i + 1) * self.embedding_dim]
                .copy_from_slice(&weight_data[offset..offset + self.embedding_dim]);
        }

        Tensor::from_vec(output, &[indices.len(), self.embedding_dim])
    }

    /// Forward pass with a tensor of indices.
    /// input: integer tensor (values interpreted as indices).
    /// Returns: Tensor with embedding_dim appended.
    pub fn forward_tensor(&self, input: &Tensor) -> Tensor {
        let indices: Vec<usize> = input.to_vec().iter().map(|&x| x as usize).collect();
        let input_shape = input.shape().to_vec();
        let result = self.forward_indices(&indices);

        // Reshape to input_shape + [embedding_dim]
        let mut out_shape: Vec<i64> = input_shape.iter().map(|&s| s as i64).collect();
        out_shape.push(self.embedding_dim as i64);
        result.reshape(&out_shape)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_tensor(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params
    }
}

/// Positional encoding using sinusoidal functions.
pub struct PositionalEncoding {
    encoding: Tensor,
    max_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut pe = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in (0..d_model).step_by(2) {
                let angle = pos as f32 / (10000.0f32).powf(i as f32 / d_model as f32);
                pe[pos * d_model + i] = angle.sin();
                if i + 1 < d_model {
                    pe[pos * d_model + i + 1] = angle.cos();
                }
            }
        }

        PositionalEncoding {
            encoding: Tensor::from_vec(pe, &[max_len, d_model]),
            max_len,
            d_model,
        }
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, seq_len, d_model]
        let seq_len = input.shape()[1];
        assert!(seq_len <= self.max_len, "Sequence length exceeds max_len");

        // Extract pe[:seq_len, :] and broadcast-add
        let pe_data: Vec<f32> = self.encoding.to_vec()[..seq_len * self.d_model].to_vec();
        let pe = Tensor::from_vec(pe_data, &[1, seq_len, self.d_model]);
        let pe_expanded = pe.expand(&[input.shape()[0], seq_len, self.d_model]);
        input.add(&pe_expanded)
    }
}
