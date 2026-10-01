//! The bytes behind a tensor, on one device or the other.
//!
//! `Storage` is never cloned during normal use — a [`Tensor`](super::Tensor)
//! shares it through an `Arc`, so passing tensors around is a refcount bump and
//! a device copy only happens when you ask for one.

use crate::cuda::CudaBuffer;

use super::Device;

#[derive(Debug)]
pub enum Storage {
    Cpu(Vec<f32>),
    Cuda(CudaBuffer),
}

impl Storage {
    /// Allocate `data` on `device`, uploading if it is a GPU.
    pub fn new(data: Vec<f32>, device: Device) -> Self {
        match device {
            Device::Cpu => Storage::Cpu(data),
            Device::Cuda(_) => {
                Storage::Cuda(CudaBuffer::from_slice(&data).expect("cuda: upload failed"))
            }
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Cuda(_) => Device::Cuda(0),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Cuda(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A host copy of the data. Downloads from the GPU when needed.
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(v) => v.clone(),
            Storage::Cuda(b) => b.to_vec().expect("cuda: download failed"),
        }
    }

    /// The host slice, if this storage is already on the CPU.
    pub fn as_slice(&self) -> Option<&[f32]> {
        match self {
            Storage::Cpu(v) => Some(v),
            Storage::Cuda(_) => None,
        }
    }

    /// The GPU buffer, if this storage lives on a device.
    pub fn as_cuda(&self) -> Option<&CudaBuffer> {
        match self {
            Storage::Cuda(b) => Some(b),
            Storage::Cpu(_) => None,
        }
    }

    /// A copy of this storage on `device`.
    pub fn to_device(&self, device: Device) -> Storage {
        match (self, device) {
            (Storage::Cpu(v), Device::Cpu) => Storage::Cpu(v.clone()),
            (Storage::Cuda(b), Device::Cuda(_)) => Storage::Cuda(b.clone()),
            (_, target) => Storage::new(self.to_vec(), target),
        }
    }
}
