//! Tensors: shape, storage and the link into the autograd graph.

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
