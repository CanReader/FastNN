//! Fallible variants of the ops that can disagree about shape.
//!
//! The ordinary ops panic, because inside a model a shape mismatch is a bug and
//! threading `Result` through every `add` would bury the maths in `?`. But at the
//! edge of an application — a shape from a config file, a batch from a user
//! upload, a checkpoint from elsewhere — a mismatch is *data*, and an app should
//! report it rather than die.
//!
//! Every function here validates, then delegates to the panicking op. There is
//! one implementation of each operation, and this file only decides whether
//! reaching it is safe.
//!
//! ```
//! # use fastnn::prelude::*;
//! let x = Tensor::zeros(&[4, 8]);
//! let w = Tensor::zeros(&[3, 3]);
//!
//! assert!(x.try_matmul(&w).is_err());   // reported
//! // x.matmul(&w)                       // would panic
//! ```

use crate::error::{Error, Result};

use super::shape;
use super::Tensor;

impl Tensor {
    /// [`matmul`](Tensor::matmul), reporting a mismatch instead of panicking.
    pub fn try_matmul(&self, other: &Tensor) -> Result<Tensor> {
        self.check_same_device(other, "matmul")?;
        // Rank first: `ndim() - 2` underflows on a 1-D operand.
        self.check_matmul_rank(other)?;
        self.check_matmul_dims(other, self.last_dim(), other.dim(other.ndim() - 2))?;
        Ok(self.matmul(other))
    }

    /// [`matmul_nt`](Tensor::matmul_nt), reporting a mismatch instead of panicking.
    pub fn try_matmul_nt(&self, other: &Tensor) -> Result<Tensor> {
        self.check_same_device(other, "matmul_nt")?;
        self.check_matmul_rank(other)?;
        self.check_matmul_dims(other, self.last_dim(), other.last_dim())?;
        Ok(self.matmul_nt(other))
    }

    /// [`matmul_tn`](Tensor::matmul_tn), reporting a mismatch instead of panicking.
    pub fn try_matmul_tn(&self, other: &Tensor) -> Result<Tensor> {
        self.check_same_device(other, "matmul_tn")?;
        self.check_matmul_rank(other)?;
        self.check_matmul_dims(
            other,
            self.dim(self.ndim() - 2),
            other.dim(other.ndim() - 2),
        )?;
        Ok(self.matmul_tn(other))
    }

    /// [`add`](Tensor::add), reporting a broadcast failure instead of panicking.
    pub fn try_add(&self, other: &Tensor) -> Result<Tensor> {
        self.check_broadcast(other, "add")?;
        Ok(self.add(other))
    }

    /// [`sub`](Tensor::sub), reporting a broadcast failure instead of panicking.
    pub fn try_sub(&self, other: &Tensor) -> Result<Tensor> {
        self.check_broadcast(other, "sub")?;
        Ok(self.sub(other))
    }

    /// [`mul`](Tensor::mul), reporting a broadcast failure instead of panicking.
    pub fn try_mul(&self, other: &Tensor) -> Result<Tensor> {
        self.check_broadcast(other, "mul")?;
        Ok(self.mul(other))
    }

    /// [`div`](Tensor::div), reporting a broadcast failure instead of panicking.
    pub fn try_div(&self, other: &Tensor) -> Result<Tensor> {
        self.check_broadcast(other, "div")?;
        Ok(self.div(other))
    }

    /// [`reshape`](Tensor::reshape), reporting an impossible shape instead of panicking.
    pub fn try_reshape(&self, dims: &[i64]) -> Result<Tensor> {
        let placeholders = dims.iter().filter(|&&d| d == -1).count();
        if placeholders > 1 {
            return Err(Error::Shape(format!(
                "reshape: more than one -1 in {dims:?}"
            )));
        }
        if dims.iter().any(|&d| d == 0 || d < -1) {
            return Err(Error::Shape(format!(
                "reshape: dimensions must be positive or -1, got {dims:?}"
            )));
        }

        let known: usize = dims
            .iter()
            .filter(|&&d| d != -1)
            .map(|&d| d as usize)
            .product();
        let fits = if placeholders == 1 {
            known > 0 && self.numel().is_multiple_of(known)
        } else {
            known == self.numel()
        };
        if !fits {
            return Err(Error::Shape(format!(
                "reshape: {} elements do not fit shape {dims:?}",
                self.numel()
            )));
        }
        Ok(self.reshape(dims))
    }

    /// [`index_select`](Tensor::index_select), reporting an out-of-range id.
    pub fn try_index_select(&self, ids: &[usize]) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(Error::Shape(format!(
                "index_select: needs a 2-D table, got {:?}",
                self.shape()
            )));
        }
        let rows = self.dim(0);
        if let Some(&bad) = ids.iter().find(|&&id| id >= rows) {
            return Err(Error::Shape(format!(
                "index_select: row {bad} outside 0..{rows}"
            )));
        }
        Ok(self.index_select(ids))
    }

    // ── Validation ───────────────────────────────────────────────────────────

    fn check_same_device(&self, other: &Tensor, op: &str) -> Result<()> {
        if self.device() != other.device() {
            return Err(Error::Shape(format!(
                "{op}: {} and {} are on different devices",
                self.device(),
                other.device()
            )));
        }
        Ok(())
    }

    /// Matmul needs rank 2+ on both sides before any inner axis even exists.
    fn check_matmul_rank(&self, other: &Tensor) -> Result<()> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(Error::Shape(format!(
                "matmul: needs 2+ dimensions, got {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }
        Ok(())
    }

    /// Matching inner dimensions and compatible batches; rank is already checked.
    fn check_matmul_dims(&self, other: &Tensor, inner: usize, other_inner: usize) -> Result<()> {
        if inner != other_inner {
            return Err(Error::Shape(format!(
                "matmul: inner dimensions {inner} and {other_inner} disagree for {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }

        let batch = shape::numel(&self.shape()[..self.ndim() - 2]);
        let other_batch = shape::numel(&other.shape()[..other.ndim() - 2]);
        if batch != other_batch && batch != 1 && other_batch != 1 {
            return Err(Error::Shape(format!(
                "matmul: batch counts {batch} and {other_batch} are not broadcastable"
            )));
        }
        Ok(())
    }

    fn check_broadcast(&self, other: &Tensor, op: &str) -> Result<()> {
        self.check_same_device(other, op)?;

        let ndim = self.ndim().max(other.ndim());
        let a = shape::pad_left(self.shape(), ndim);
        let b = shape::pad_left(other.shape(), ndim);
        if a.iter().zip(&b).any(|(&x, &y)| x != y && x != 1 && y != 1) {
            return Err(Error::Shape(format!(
                "{op}: cannot broadcast {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn matmul_reports_instead_of_panicking() {
        let x = Tensor::zeros(&[4, 8]);
        assert!(x.try_matmul(&Tensor::zeros(&[8, 2])).is_ok());
        assert!(x.try_matmul(&Tensor::zeros(&[3, 3])).is_err());
        assert!(x.try_matmul(&Tensor::zeros(&[8])).is_err());
        assert!(Tensor::zeros(&[8]).try_matmul(&x).is_err());
        assert!(x.try_matmul_tn(&Tensor::zeros(&[4])).is_err());
        assert!(x.try_matmul_nt(&Tensor::zeros(&[8])).is_err());
    }

    #[test]
    fn broadcast_rules_match_the_panicking_ops() {
        let x = Tensor::zeros(&[4, 3]);
        assert!(x.try_add(&Tensor::zeros(&[1, 3])).is_ok());
        assert!(x.try_add(&Tensor::zeros(&[3])).is_ok());
        assert!(x.try_add(&Tensor::zeros(&[2, 3])).is_err());
    }

    #[test]
    fn reshape_rejects_what_cannot_fit() {
        let x = Tensor::zeros(&[4, 3]);
        assert!(x.try_reshape(&[2, 6]).is_ok());
        assert!(x.try_reshape(&[2, -1]).is_ok());
        assert!(x.try_reshape(&[5, 5]).is_err());
        assert!(x.try_reshape(&[-1, -1]).is_err());
        assert!(x.try_reshape(&[0, 12]).is_err());
    }

    #[test]
    fn index_select_rejects_out_of_range_rows() {
        let table = Tensor::zeros(&[5, 3]);
        assert!(table.try_index_select(&[0, 4]).is_ok());
        assert!(table.try_index_select(&[0, 5]).is_err());
    }
}
