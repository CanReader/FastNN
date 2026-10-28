//! Saving and loading model weights in the `.fdl` format.
//!
//! ```text
//! magic "FDL\0"  u32
//! version        u32
//! count          u32
//! repeated count times:
//!     name_len   u32
//!     name       name_len bytes, UTF-8
//!     rank       u32
//!     shape      rank x u32
//!     data       numel x f32, little-endian
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::tensor::Tensor;

const MAGIC: u32 = 0x4644_4C00;
const VERSION: u32 = 1;

/// Named tensors. A `BTreeMap` keeps the file order stable.
pub type Checkpoint = BTreeMap<String, Tensor>;

/// Write a module's parameters and buffers to `path`.
///
/// ```no_run
/// # use fastnn::prelude::*;
/// # let model = Sequential::new().add(Linear::new(4, 2));
/// save(&model, "model.fdl")?;
/// # Ok::<(), fastnn::Error>(())
/// ```
pub fn save(module: &dyn Module, path: impl AsRef<Path>) -> Result<()> {
    let mut tensors = Checkpoint::new();
    for (name, param) in module.named_parameters() {
        tensors.insert(name, param.value().cpu());
    }
    for (name, buffer) in module.named_buffers() {
        tensors.insert(name, buffer.value().cpu());
    }
    save_tensors(&tensors, path)
}

/// Read `path` into a module's parameters and buffers.
///
/// Every parameter must be present with a matching shape. Extra tensors in the
/// file are ignored.
pub fn load(module: &dyn Module, path: impl AsRef<Path>) -> Result<()> {
    let saved = load_tensors(path)?;

    for (name, param) in module.named_parameters() {
        let value = take(&saved, &name, &param.shape())?;
        param.set_value(value.to(param.device()));
    }
    for (name, buffer) in module.named_buffers() {
        let device = buffer.value().device();
        let value = take(&saved, &name, buffer.value().shape())?;
        buffer.set_value(value.to(device));
    }
    Ok(())
}

/// Write named tensors to `path`.
pub fn save_tensors(tensors: &Checkpoint, path: impl AsRef<Path>) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);

    write_u32(&mut out, MAGIC)?;
    write_u32(&mut out, VERSION)?;
    write_u32(&mut out, tensors.len() as u32)?;

    for (name, tensor) in tensors {
        write_u32(&mut out, name.len() as u32)?;
        out.write_all(name.as_bytes())?;

        write_u32(&mut out, tensor.ndim() as u32)?;
        for &dim in tensor.shape() {
            write_u32(&mut out, dim as u32)?;
        }
        for value in tensor.to_vec() {
            out.write_all(&value.to_le_bytes())?;
        }
    }

    out.flush()?;
    Ok(())
}

/// Read named tensors from `path`.
pub fn load_tensors(path: impl AsRef<Path>) -> Result<Checkpoint> {
    let mut input = BufReader::new(File::open(path)?);

    if read_u32(&mut input)? != MAGIC {
        return Err(Error::Checkpoint("not a fastnn checkpoint".into()));
    }
    let version = read_u32(&mut input)?;
    if version != VERSION {
        return Err(Error::Checkpoint(format!(
            "file is version {version}, this build reads version {VERSION}"
        )));
    }

    let count = read_u32(&mut input)?;
    let mut tensors = Checkpoint::new();
    for _ in 0..count {
        let name_len = read_u32(&mut input)? as usize;
        let mut name = vec![0u8; name_len];
        input.read_exact(&mut name)?;
        let name = String::from_utf8(name)
            .map_err(|_| Error::Checkpoint("tensor name is not valid UTF-8".into()))?;

        let rank = read_u32(&mut input)? as usize;
        let shape: Vec<usize> = (0..rank)
            .map(|_| read_u32(&mut input).map(|d| d as usize))
            .collect::<Result<_>>()?;

        let mut data = vec![0.0f32; shape.iter().product()];
        for value in &mut data {
            let mut bytes = [0u8; 4];
            input.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
        }

        tensors.insert(name, Tensor::from_vec(data, &shape));
    }

    Ok(tensors)
}

/// Look up `name` and check its shape.
fn take(saved: &Checkpoint, name: &str, expected: &[usize]) -> Result<Tensor> {
    let tensor = saved
        .get(name)
        .ok_or_else(|| Error::Checkpoint(format!("checkpoint has no tensor named '{name}'")))?;
    if tensor.shape() != expected {
        return Err(Error::Checkpoint(format!(
            "'{name}' is {:?} in the file but {expected:?} in the model",
            tensor.shape()
        )));
    }
    Ok(tensor.clone())
}

fn write_u32(out: &mut impl Write, value: u32) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(input: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
