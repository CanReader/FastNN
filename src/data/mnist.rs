//! MNIST handwritten digits.
//!
//! Downloads the four IDX files on first use and caches them under
//! `$FASTNN_DATA_DIR/mnist` (or `~/.fastnn/datasets/mnist`).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::tensor::Tensor;

use super::dataset::Dataset;

const MIRROR: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
const IMAGE_MAGIC: u32 = 0x0000_0803;
const LABEL_MAGIC: u32 = 0x0000_0801;

/// Which half of the dataset to load.
#[derive(Clone, Copy, Debug)]
pub enum Split {
    Train,
    Test,
}

impl Split {
    fn files(self) -> (&'static str, &'static str) {
        match self {
            Split::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            Split::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        }
    }
}

/// 28×28 grayscale digits with labels 0–9.
///
/// Images are `[1, 28, 28]` scaled to `[0, 1]`; targets are class indices.
///
/// ```no_run
/// use fastnn::prelude::*;
/// use fastnn::data::{Mnist, Split};
///
/// let train = Mnist::load(Split::Train)?;
/// let loader = DataLoader::new(&train, 128).shuffle(true);
/// # Ok::<(), fastnn::Error>(())
/// ```
pub struct Mnist {
    images: Vec<f32>,
    labels: Vec<f32>,
    len: usize,
}

impl Mnist {
    /// Load a split from the default cache, downloading if it is not there.
    pub fn load(split: Split) -> Result<Mnist> {
        Mnist::load_from(&default_cache_dir(), split)
    }

    /// Load a split from an explicit directory.
    pub fn load_from(cache: &Path, split: Split) -> Result<Mnist> {
        let (image_file, label_file) = split.files();
        download_missing(cache, &[image_file, label_file])?;

        let (images, image_count) = parse_images(&fs::read(cache.join(image_file))?)?;
        let (labels, label_count) = parse_labels(&fs::read(cache.join(label_file))?)?;
        if image_count != label_count {
            return Err(Error::Dataset(format!(
                "mnist: {image_count} images but {label_count} labels"
            )));
        }

        Ok(Mnist {
            images,
            labels,
            len: image_count,
        })
    }

    /// Every image as one `[N, 1, 28, 28]` tensor.
    pub fn images(&self) -> Tensor {
        Tensor::from_vec(self.images.clone(), &[self.len, 1, 28, 28])
    }

    /// Every label as a class index.
    pub fn labels(&self) -> Vec<usize> {
        self.labels.iter().map(|&v| v as usize).collect()
    }
}

impl Dataset for Mnist {
    fn len(&self) -> usize {
        self.len
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![1, 28, 28]
    }

    fn target_shape(&self) -> Vec<usize> {
        vec![1]
    }

    fn write(&self, index: usize, input: &mut [f32], target: &mut [f32]) {
        let start = index * 28 * 28;
        input.copy_from_slice(&self.images[start..start + 28 * 28]);
        target[0] = self.labels[index];
    }
}

/// Where cached IDX files live.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("FASTNN_DATA_DIR") {
        return PathBuf::from(dir).join("mnist");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".fastnn")
        .join("datasets")
        .join("mnist")
}

/// Fetch and decompress any of `files` that are not already cached.
fn download_missing(cache: &Path, files: &[&str]) -> Result<()> {
    fs::create_dir_all(cache)?;
    for name in files {
        let target = cache.join(name);
        if target.exists() {
            continue;
        }

        let url = format!("{MIRROR}/{name}.gz");
        eprintln!("mnist: downloading {url}");
        let response = ureq::get(&url)
            .call()
            .map_err(|e| Error::Dataset(format!("could not fetch {url}: {e}")))?;

        let mut compressed = Vec::new();
        response.into_reader().read_to_end(&mut compressed)?;
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(&compressed[..]).read_to_end(&mut raw)?;

        fs::File::create(&target)?.write_all(&raw)?;
    }
    Ok(())
}

/// IDX images: magic, count, rows, cols, then one byte per pixel.
fn parse_images(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let header = read_header(bytes, 4, IMAGE_MAGIC, "images")?;
    let (count, rows, cols) = (header[1], header[2], header[3]);

    let expected = 16 + count * rows * cols;
    if bytes.len() != expected {
        return Err(Error::Dataset(format!(
            "mnist images: {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    // Scale to [0, 1]; unnormalized 0–255 inputs would saturate the first layer.
    Ok((
        bytes[16..].iter().map(|&b| b as f32 / 255.0).collect(),
        count,
    ))
}

/// IDX labels: magic, count, then one byte per label.
fn parse_labels(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let header = read_header(bytes, 2, LABEL_MAGIC, "labels")?;
    let count = header[1];

    if bytes.len() != 8 + count {
        return Err(Error::Dataset(format!(
            "mnist labels: {} bytes, expected {}",
            bytes.len(),
            8 + count
        )));
    }
    Ok((bytes[8..].iter().map(|&b| b as f32).collect(), count))
}

/// Read `words` big-endian u32s and check the magic number.
fn read_header(bytes: &[u8], words: usize, magic: u32, what: &str) -> Result<Vec<usize>> {
    if bytes.len() < words * 4 {
        return Err(Error::Dataset(format!("mnist {what}: header truncated")));
    }
    let values: Vec<usize> = bytes[..words * 4]
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();

    if values[0] as u32 != magic {
        return Err(Error::Dataset(format!(
            "mnist {what}: magic {:#010x}, expected {magic:#010x}",
            values[0]
        )));
    }
    Ok(values)
}
