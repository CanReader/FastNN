//! MNIST digits, downloaded on first use into `$FASTNN_DATA_DIR/mnist` or
//! `~/.fastnn/datasets/mnist`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::serialize::atomic::write_atomic_bytes;
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
        let (images, image_count) = load_cached(cache, image_file, parse_images, fetch)?;
        let (labels, label_count) = load_cached(cache, label_file, parse_labels, fetch)?;
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

/// Parse a cached file, downloading it first when it is missing.
///
/// A cached file that does not parse is deleted and fetched once more. Without
/// that, a file truncated by an interrupted write would be trusted forever,
/// since the cache check only asks whether the file exists.
fn load_cached<T>(
    cache: &Path,
    name: &str,
    parse: fn(&[u8]) -> Result<T>,
    fetch: impl Fn(&Path, &str) -> Result<Vec<u8>>,
) -> Result<T> {
    let target = cache.join(name);
    if target.exists() {
        match parse(&fs::read(&target)?) {
            Ok(parsed) => return Ok(parsed),
            Err(e) => {
                eprintln!(
                    "mnist: {} is not usable ({e}), downloading it again",
                    target.display()
                );
                let _ = fs::remove_file(&target);
            }
        }
    }
    fs::create_dir_all(cache)?;
    let bytes = fetch(cache, name)?;
    parse(&bytes).map_err(|e| Error::Dataset(format!("mnist: {}: {e}", target.display())))
}

/// Download `name` from the mirror, decompress it, and write it into the cache.
fn fetch(cache: &Path, name: &str) -> Result<Vec<u8>> {
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

    write_atomic_bytes(&cache.join(name), &raw)?;
    Ok(raw)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("fastnn-mnist-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A valid IDX image file holding `count` one-pixel images.
    fn image_file(count: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        for word in [IMAGE_MAGIC, count, 1, 1] {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        bytes.resize(IMAGE_HEADER + count as usize, 7);
        bytes
    }

    #[test]
    fn a_valid_cached_file_is_used_without_fetching() {
        let dir = TempDir::new("valid");
        fs::write(dir.0.join("images"), image_file(3)).unwrap();
        let (_, count) = load_cached(&dir.0, "images", parse_images, |_, _| {
            panic!("a valid cached file must not be fetched")
        })
        .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn a_missing_file_is_fetched() {
        let dir = TempDir::new("missing");
        let (_, count) = load_cached(&dir.0, "images", parse_images, |cache, name| {
            let bytes = image_file(2);
            fs::write(cache.join(name), &bytes)?;
            Ok(bytes)
        })
        .unwrap();
        assert_eq!(count, 2);
        assert!(parse_images(&fs::read(dir.0.join("images")).unwrap()).is_ok());
    }

    #[test]
    fn a_truncated_cached_file_is_fetched_again() {
        let dir = TempDir::new("truncated");
        // Valid magic and header, body cut short: what an interrupted write leaves.
        let mut truncated = image_file(5);
        truncated.truncate(IMAGE_HEADER + 2);
        fs::write(dir.0.join("images"), &truncated).unwrap();

        let fetches = std::cell::Cell::new(0);
        let (_, count) = load_cached(&dir.0, "images", parse_images, |cache, name| {
            fetches.set(fetches.get() + 1);
            let bytes = image_file(5);
            fs::write(cache.join(name), &bytes)?;
            Ok(bytes)
        })
        .unwrap();
        assert_eq!(count, 5);
        assert_eq!(fetches.get(), 1);
    }

    #[test]
    fn a_bad_download_names_the_file() {
        let dir = TempDir::new("bad-download");
        fs::write(dir.0.join("images"), b"junk").unwrap();
        let err = load_cached(&dir.0, "images", parse_images, |_, _| {
            Ok(b"still junk".to_vec())
        })
        .unwrap_err();
        assert!(err.to_string().contains("images"), "{err}");
    }
}
