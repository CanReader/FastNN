//! Datasets, batching, and MNIST.
//!
//! A [`Dataset`] writes one item into a slice. A [`DataLoader`] fills one buffer
//! per batch from it, shuffles, and can place batches on a device.
//! [`TensorDataset`] wraps two tensors, and [`Mnist`] downloads the digits once
//! and caches them.
//!
//! ```
//! use fastnn::data::{DataLoader, Dataset};
//!
//! struct Squares;
//!
//! impl Dataset for Squares {
//!     fn len(&self) -> usize { 1000 }
//!     fn input_shape(&self) -> Vec<usize> { vec![1] }
//!     fn target_shape(&self) -> Vec<usize> { vec![1] }
//!     fn write(&self, i: usize, input: &mut [f32], target: &mut [f32]) {
//!         let x = i as f32 / 1000.0;
//!         input[0] = x;
//!         target[0] = x * x;
//!     }
//! }
//!
//! let loader = DataLoader::new(&Squares, 100).shuffle(true);
//! assert_eq!(loader.batches(), 10);
//! for batch in loader.iter() {
//!     // A [1] item batches to [batch], like a label vector, not [batch, 1].
//!     assert_eq!(batch.inputs.shape(), &[100]);
//!     assert_eq!(batch.targets.shape(), &[100]);
//! }
//! ```

pub mod dataset;
pub mod loader;
pub mod mnist;

pub use dataset::{Dataset, TensorDataset};
pub use loader::{Batch, Batches, DataLoader};
pub use mnist::{Mnist, Split};
