//! Tensors: shape, storage and the link into the autograd graph.
//!
//! A [`Tensor`] is an immutable `f32` array on the CPU or a GPU. Ops return new
//! tensors and broadcast the way NumPy does.
//!
//! ```
//! use fastnn::tensor::Tensor;
//!
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
//! let bias = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
//!
//! let shifted = a.add(&bias); // [2, 3] + [3] broadcasts over the rows
//! assert_eq!(shifted.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
//!
//! let product = a.matmul(&a.transpose()); // [2, 3] x [3, 2]
//! assert_eq!(product.shape(), &[2, 2]);
//! ```

mod checked;
mod core;
mod device;
mod display;
mod init;
pub mod ops;
pub mod shape;
pub mod storage;

pub use core::Tensor;
pub use device::Device;
pub use ops::conv::Window;
pub use ops::norm::BatchStats;
