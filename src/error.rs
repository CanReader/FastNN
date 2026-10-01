//! Error type for the fallible boundaries of the library.
//!
//! Tensor math does *not* return `Result`. A shape or device mismatch is a bug in
//! the calling code, like indexing past the end of a slice, so it panics with a
//! message naming both shapes. `Result` is reserved for things that can fail at
//! runtime through no fault of the code: I/O, downloads, and CUDA.

use std::fmt;

/// Everything that can go wrong at a fallible boundary.
#[derive(Debug)]
pub enum Error {
    /// A CUDA driver, runtime, or kernel call failed.
    Cuda(String),
    /// CUDA support was requested but this build or machine has no GPU.
    NoCuda,
    /// Filesystem or network I/O failed.
    Io(std::io::Error),
    /// A checkpoint file was unreadable or did not match the model.
    Checkpoint(String),
    /// A dataset could not be downloaded or parsed.
    Dataset(String),
    /// Shapes or devices did not line up.
    ///
    /// Only the [`try_*`](crate::tensor::Tensor::try_matmul) ops produce this.
    /// The ordinary ops panic instead — see the module docs.
    Shape(String),
}

/// Shorthand for results carrying a fastnn [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda(msg) => write!(f, "cuda: {msg}"),
            Error::NoCuda => write!(
                f,
                "cuda: no device available (build with the `cuda` feature and a working GPU)"
            ),
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Checkpoint(msg) => write!(f, "checkpoint: {msg}"),
            Error::Dataset(msg) => write!(f, "dataset: {msg}"),
            Error::Shape(msg) => write!(f, "shape: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
