//! Where a tensor lives.

use std::fmt;

use crate::error::{Error, Result};

/// Where a tensor's storage is allocated. Ops panic if operands disagree, so
/// move data with [`Tensor::to`](crate::tensor::Tensor::to).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Device {
    /// The GPU at `index`, if this build has CUDA and the device exists.
    pub fn cuda(index: usize) -> Result<Device> {
        if !cfg!(feature = "cuda") {
            return Err(Error::NoCuda);
        }
        let count = crate::cuda::device_count();
        if index >= count {
            return Err(Error::Cuda(format!(
                "requested cuda:{index} but only {count} device(s) present"
            )));
        }
        crate::cuda::init(index)?;
        Ok(Device::Cuda(index))
    }

    /// The first GPU if one is usable, otherwise the CPU. Never fails.
    pub fn best() -> Device {
        Device::cuda(0).unwrap_or(Device::Cpu)
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}
