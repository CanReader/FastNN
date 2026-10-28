//! Datasets, batching, and MNIST.

pub mod dataset;
pub mod loader;
pub mod mnist;

pub use dataset::{Dataset, TensorDataset};
pub use loader::{Batch, Batches, DataLoader};
pub use mnist::{Mnist, Split};
