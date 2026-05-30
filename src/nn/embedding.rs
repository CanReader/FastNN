use std::collections::HashMap;
use std::sync::Arc;
use crate::tensor::{Tensor, TensorStorage};
use crate::nn::module::Module;
use crate::autograd::graph;
use crate::autograd::backward_ops::EmbeddingBackward;

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
        let device = self.weight.device();
        let indices_i32: Vec<i32> = indices.iter().map(|&x| x as i32).collect();

        // ── CUDA fast path ──────────────────────────────────────────────────
        #[cfg(feature = "cuda")]
        if let TensorStorage::Cuda(w_buf) = &self.weight.storage {
            use crate::tensor::cuda_backend;
            let idx_buf = cuda_backend::cuda_upload_indices(&indices_i32)
                .expect("Failed to upload embedding indices to GPU");
            let out_buf = cuda_backend::cuda_embedding_forward(
                &idx_buf, w_buf, indices.len(), self.embedding_dim,
            ).expect("CUDA embedding_forward failed");
            let mut out = Tensor::from_cuda_buffer(out_buf, vec![indices.len(), self.embedding_dim], false);

            if graph::is_grad_enabled() && self.weight.requires_grad() {
                out.set_requires_grad(true);
                let grad_fn = Arc::new(EmbeddingBackward {
                    input_ids: vec![self.weight.id()],
                    indices: indices.to_vec(),
                    indices_buf: Some(idx_buf),
                    num_embeddings: self.num_embeddings,
                    embedding_dim: self.embedding_dim,
                    device,
                });
                graph::record_op_with_cells(
                    grad_fn, out.id(),
                    vec![(self.weight.id(), self.weight.grad_cell())],
                );
            }
            return out;
        }

        // ── CPU path ────────────────────────────────────────────────────────
        let weight_data = self.weight.to_vec();
        let mut output = vec![0.0f32; indices.len() * self.embedding_dim];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.num_embeddings, "Index {} out of range (vocab size {})", idx, self.num_embeddings);
            let offset = idx * self.embedding_dim;
            output[i * self.embedding_dim..(i + 1) * self.embedding_dim]
                .copy_from_slice(&weight_data[offset..offset + self.embedding_dim]);
        }

        let mut out = Tensor::from_vec(output, &[indices.len(), self.embedding_dim])
            .to_device(device);

        if graph::is_grad_enabled() && self.weight.requires_grad() {
            out.set_requires_grad(true);
            let grad_fn = Arc::new(EmbeddingBackward {
                input_ids: vec![self.weight.id()],
                indices: indices.to_vec(),
                indices_buf: None,
                num_embeddings: self.num_embeddings,
                embedding_dim: self.embedding_dim,
                device,
            });
            graph::record_op_with_cells(
                grad_fn, out.id(),
                vec![(self.weight.id(), self.weight.grad_cell())],
            );
        }

        out
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

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }

    fn named_parameters(&self) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params
    }

    fn to_device(&mut self, device: crate::tensor::Device) {
        self.weight = self.weight.to_device(device);
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
