use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, BufWriter, BufReader};
use std::path::Path;

use crate::tensor::Tensor;
use crate::nn::Module;

/// Serialized tensor format: name_len(u32) + name(bytes) + ndim(u32) + shape(u32*ndim) + data(f32*numel)
/// File format: magic(u32) + version(u32) + num_tensors(u32) + tensors...

const MAGIC: u32 = 0x46444C00; // "FDL\0"
const VERSION: u32 = 1;

/// Save named tensors to a binary file.
pub fn save_tensors(tensors: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);

    write_u32(&mut writer, MAGIC)?;
    write_u32(&mut writer, VERSION)?;
    write_u32(&mut writer, tensors.len() as u32)?;

    for (name, tensor) in tensors {
        let name_bytes = name.as_bytes();
        write_u32(&mut writer, name_bytes.len() as u32)?;
        writer.write_all(name_bytes).map_err(|e| e.to_string())?;

        let shape = tensor.shape();
        write_u32(&mut writer, shape.len() as u32)?;
        for &dim in shape {
            write_u32(&mut writer, dim as u32)?;
        }

        let data = tensor.to_vec();
        for &val in &data {
            writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Load named tensors from a binary file.
pub fn load_tensors(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    let magic = read_u32(&mut reader)?;
    if magic != MAGIC {
        return Err("Invalid file format (bad magic number)".to_string());
    }

    let version = read_u32(&mut reader)?;
    if version != VERSION {
        return Err(format!("Unsupported version: {}", version));
    }

    let num_tensors = read_u32(&mut reader)? as usize;
    let mut tensors = HashMap::new();

    for _ in 0..num_tensors {
        let name_len = read_u32(&mut reader)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes).map_err(|e| e.to_string())?;
        let name = String::from_utf8(name_bytes).map_err(|e| e.to_string())?;

        let ndim = read_u32(&mut reader)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(&mut reader)? as usize);
        }

        let numel: usize = shape.iter().product();
        let mut data = vec![0.0f32; numel];
        for val in &mut data {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
            *val = f32::from_le_bytes(buf);
        }

        tensors.insert(name, Tensor::from_vec(data, &shape));
    }

    Ok(tensors)
}

/// Save a module's parameters to a file.
pub fn save_model(module: &dyn Module, path: impl AsRef<Path>) -> Result<(), String> {
    let params = module.named_parameters();
    save_tensors(&params, path)
}

/// Load parameters from a file into a module's named parameters.
pub fn load_model(params: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, String> {
    let loaded = load_tensors(path)?;

    // Verify shapes match
    for (name, tensor) in params {
        if let Some(loaded_tensor) = loaded.get(name) {
            if tensor.shape() != loaded_tensor.shape() {
                return Err(format!(
                    "Shape mismatch for '{}': expected {:?}, got {:?}",
                    name, tensor.shape(), loaded_tensor.shape()
                ));
            }
        }
    }

    Ok(loaded)
}

fn write_u32(writer: &mut impl Write, val: u32) -> Result<(), String> {
    writer.write_all(&val.to_le_bytes()).map_err(|e| e.to_string())
}

fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}
