//! Embedding table and positional encoding.

use crate::tensor::Tensor;

use super::module::Module;
use super::param::Param;

/// A learnable `[vocab, dim]` table, looked up by token id.
///
/// The lookup is [`index_select`](Tensor::index_select), so the gradient
/// scatter-adds back into the rows that were used and leaves the rest untouched.
pub struct Embedding {
    pub weight: Param,
    vocab: usize,
    dim: usize,
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize) -> Embedding {
        // Small normal init: large embeddings would dominate the residual stream
        // before the model has learned anything.
        Embedding {
            weight: Param::new(Tensor::randn(&[vocab, dim]).mul_scalar(0.02)),
            vocab,
            dim,
        }
    }

    /// Look up a flat list of ids, giving `[ids.len(), dim]`.
    pub fn lookup(&self, ids: &[usize]) -> Tensor {
        self.weight.tensor().index_select(ids)
    }

    /// Look up a `[batch, sequence]` id grid, giving `[batch, sequence, dim]`.
    pub fn lookup_batch(&self, ids: &[Vec<usize>]) -> Tensor {
        let (batch, sequence) = (ids.len(), ids.first().map_or(0, |row| row.len()));
        let flat: Vec<usize> = ids
            .iter()
            .inspect(|row| assert_eq!(row.len(), sequence, "lookup_batch: ragged id grid"))
            .flatten()
            .copied()
            .collect();
        self.lookup(&flat)
            .reshape(&[batch as i64, sequence as i64, self.dim as i64])
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Module for Embedding {
    /// Treats the input's values as ids. Prefer [`lookup`](Embedding::lookup) —
    /// it takes `usize` and cannot silently truncate a float.
    fn forward(&self, input: &Tensor) -> Tensor {
        let ids: Vec<usize> = input.to_vec().iter().map(|&v| v as usize).collect();
        let mut shape: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        shape.push(self.dim as i64);
        self.lookup(&ids).reshape(&shape)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("weight".into(), self.weight.clone())]
    }
}

/// Fixed sinusoidal position signal, added to token embeddings.
///
/// Not learned: the sin/cos pattern lets the model read relative offsets as a
/// linear function of position, and generalises past the longest training length.
pub struct PositionalEncoding {
    table: Tensor,
    dim: usize,
}

impl PositionalEncoding {
    pub fn new(dim: usize, max_len: usize) -> PositionalEncoding {
        let mut table = vec![0.0f32; max_len * dim];
        for position in 0..max_len {
            for i in (0..dim).step_by(2) {
                let angle = position as f32 / 10000f32.powf(i as f32 / dim as f32);
                table[position * dim + i] = angle.sin();
                if i + 1 < dim {
                    table[position * dim + i + 1] = angle.cos();
                }
            }
        }
        PositionalEncoding {
            table: Tensor::from_vec(table, &[max_len, dim]),
            dim,
        }
    }
}

impl Module for PositionalEncoding {
    /// Adds positions to a `[batch, sequence, dim]` input.
    fn forward(&self, input: &Tensor) -> Tensor {
        let sequence = input.dim(1);
        assert!(
            sequence <= self.table.dim(0),
            "PositionalEncoding: sequence {sequence} exceeds max_len {}",
            self.table.dim(0)
        );
        let positions: Vec<usize> = (0..sequence).collect();
        let slice = self
            .table
            .to(input.device())
            .index_select(&positions)
            .reshape(&[1, sequence as i64, self.dim as i64]);
        input.add(&slice)
    }
}
