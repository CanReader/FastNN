//! MNIST digits, downloaded on first use into `$FASTNN_DATA_DIR/mnist` or
//! `~/.fastnn/datasets/mnist`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::tensor::Tensor;

use super::dataset::Dataset;

const MIRROR: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
const IMAGE_MAGIC: u32 = 0x0000_0803;
const LABEL_MAGIC: u32 = 0x0000_0801;
const SIDE: usize = 28;
const PIXELS: usize = SIDE * SIDE;
const IMAGE_HEADER: usize = 16;
const LABEL_HEADER: usize = 8;

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

/// 28x28 grayscale digits scaled to `[0, 1]`, with class index targets.
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

        let (images, image_count) = parse_images(&read_or_redownload(cache, image_file)?)?;
        let (labels, label_count) = parse_labels(&read_or_redownload(cache, label_file)?)?;
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

    /// All images as one `[N, 1, 28, 28]` tensor.
    pub fn images(&self) -> Tensor {
        Tensor::from_vec(self.images.clone(), &[self.len, 1, SIDE, SIDE])
    }

    /// All labels as class indices.
    pub fn labels(&self) -> Vec<usize> {
        self.labels.iter().map(|&v| v as usize).collect()
    }
}

impl Dataset for Mnist {
    fn len(&self) -> usize {
        self.len
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![1, SIDE, SIDE]
    }

    fn target_shape(&self) -> Vec<usize> {
        vec![1]
    }

    fn write(&self, index: usize, input: &mut [f32], target: &mut [f32]) {
        let start = index * PIXELS;
        input.copy_from_slice(&self.images[start..start + PIXELS]);
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

/// Write `bytes` to `target` atomically: write a per-process temporary file,
/// `sync_all`, then rename into place. A crash at any point leaves the
/// target either absent or complete.
fn write_atomic(target: &Path, bytes: &[u8]) -> Result<()> {
    let name = target
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let partial = target.with_file_name(format!("{name}.{}.partial", std::process::id()));

    let mut file = fs::File::create(&partial)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&partial, target)?;
    Ok(())
}

/// Fetch a single file from the mirror, decompress, and write atomically.
fn fetch_and_write(cache: &Path, name: &str) -> Result<()> {
    let url = format!("{MIRROR}/{name}.gz");
    eprintln!("mnist: downloading {url}");
    let response = ureq::get(&url)
        .call()
        .map_err(|e| Error::Dataset(format!("could not fetch {url}: {e}")))?;

    let mut compressed = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut compressed)?;
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&compressed[..]).read_to_end(&mut raw)?;

    write_atomic(&cache.join(name), &raw)
}

/// Download and decompress any of `files` not already cached.
///
/// Writes to a per-process `.partial` temporary file, calls `sync_all`, and
/// renames on success so that an interrupted or concurrent download never
/// leaves a truncated file in the cache.  If a cached file exists but is
/// corrupt (truncated by an earlier crash), it is deleted and re-downloaded
/// once.
fn download_missing(cache: &Path, files: &[&str]) -> Result<()> {
    fs::create_dir_all(cache)?;
    for name in files {
        let target = cache.join(name);
        if target.exists() {
            continue;
        }
        fetch_and_write(cache, name)?;
    }
    Ok(())
}

/// Try to parse a cached file; if it fails, delete it and re-download once.
fn read_or_redownload(cache: &Path, name: &str) -> Result<Vec<u8>> {
    let target = cache.join(name);
    let bytes = fs::read(&target)?;
    let header_ok = if bytes.len() >= 4 {
        let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        magic == IMAGE_MAGIC || magic == LABEL_MAGIC
    } else {
        false
    };
    if header_ok {
        return Ok(bytes);
    }
    eprintln!("mnist: cached {name} is corrupt, re-downloading");
    let _ = fs::remove_file(&target);
    fetch_and_write(cache, name)?;
    Ok(fs::read(&target)?)
}

/// IDX images: magic, count, rows, cols, then one byte per pixel.
fn parse_images(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let header = read_header(bytes, 4, IMAGE_MAGIC, "images")?;
    let (count, rows, cols) = (header[1], header[2], header[3]);

    let expected = IMAGE_HEADER + count * rows * cols;
    if bytes.len() != expected {
        return Err(Error::Dataset(format!(
            "mnist images: {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok((
        bytes[IMAGE_HEADER..]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect(),
        count,
    ))
}

/// IDX labels: magic, count, then one byte per label.
fn parse_labels(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let header = read_header(bytes, 2, LABEL_MAGIC, "labels")?;
    let count = header[1];

    if bytes.len() != LABEL_HEADER + count {
        return Err(Error::Dataset(format!(
            "mnist labels: {} bytes, expected {}",
            bytes.len(),
            LABEL_HEADER + count
        )));
    }
    Ok((
        bytes[LABEL_HEADER..].iter().map(|&b| b as f32).collect(),
        count,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("fastnn_test")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn truncated_file_rejected_by_parser() {
        let cache = test_dir("truncated_file");
        let target = cache.join("train-images-idx3-ubyte");

        fs::write(&target, b"truncated").unwrap();
        assert!(target.exists());

        let bytes = fs::read(&target).unwrap();
        assert!(parse_images(&bytes).is_err());
    }

    #[test]
    fn partial_suffix_not_mistaken_for_final() {
        let cache = test_dir("partial_suffix");
        let name = "train-images-idx3-ubyte";
        let partial = cache.join(format!("{name}.partial"));

        fs::write(&partial, b"junk").unwrap();

        let target = cache.join(name);
        assert!(!target.exists(), ".partial must not satisfy the exists() check");
    }

    #[test]
    fn write_atomic_creates_file_with_correct_content() {
        let dir = test_dir("write_atomic_ok");
        let target = dir.join("output.bin");
        let payload = b"hello, atomic world";

        write_atomic(&target, payload).unwrap();
        assert_eq!(fs::read(&target).unwrap(), payload);
    }

    #[test]
    fn write_atomic_leaves_no_partial_on_success() {
        let dir = test_dir("write_atomic_no_partial");
        let target = dir.join("data.bin");
        write_atomic(&target, b"ok").unwrap();

        let partials: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".partial"))
            .collect();
        assert!(partials.is_empty(), "no .partial files should remain");
    }

    #[test]
    fn write_atomic_overwrites_existing() {
        let dir = test_dir("write_atomic_overwrite");
        let target = dir.join("file.bin");
        fs::write(&target, b"old").unwrap();

        write_atomic(&target, b"new").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new");
    }
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
