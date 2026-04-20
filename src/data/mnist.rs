//! MNIST dataset loader.
//!
//! On first use, downloads the 4 MNIST IDX files from a public mirror into
//! `~/.fastnn/datasets/mnist/` (or `$FASTNN_DATA_DIR/mnist/` if set), then
//! parses them into `[N, 1, 28, 28]` image tensors and `[N]` label tensors.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::tensor::Tensor;
use crate::data::dataset::Dataset;

const MIRROR: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
const FILES: &[&str] = &[
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

/// MNIST dataset split.
pub enum MnistSplit {
    Train,
    Test,
}

/// Tensors for one MNIST split.
pub struct MnistData {
    /// `[N, 1, 28, 28]` image tensor, normalized to [0, 1].
    pub images: Tensor,
    /// `[N]` label tensor with values 0..=9.
    pub labels: Tensor,
    /// Number of samples.
    pub len: usize,
}

/// Default cache dir: `$FASTNN_DATA_DIR/mnist/` or `~/.fastnn/datasets/mnist/`.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("FASTNN_DATA_DIR") {
        return PathBuf::from(dir).join("mnist");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".fastnn").join("datasets").join("mnist")
}

/// Ensure all 4 IDX files exist at `cache_dir`, downloading if needed.
pub fn ensure_downloaded(cache_dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(cache_dir)?;
    for name in FILES {
        let out = cache_dir.join(name);
        if out.exists() { continue; }

        let url = format!("{}/{}.gz", MIRROR, name);
        eprintln!("[MNIST] downloading {} ...", url);
        let resp = ureq::get(&url).call()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("download failed: {}", e)))?;
        let mut gz = Vec::new();
        resp.into_reader().read_to_end(&mut gz)?;

        let mut decoder = flate2::read::GzDecoder::new(&gz[..]);
        let mut raw = Vec::new();
        decoder.read_to_end(&mut raw)?;

        let mut f = fs::File::create(&out)?;
        f.write_all(&raw)?;
        eprintln!("[MNIST] wrote {} ({} bytes)", out.display(), raw.len());
    }
    Ok(())
}

/// Load one split from the default cache directory.
pub fn load(split: MnistSplit) -> std::io::Result<MnistData> {
    load_from(&default_cache_dir(), split)
}

/// Load one split from an explicit cache directory.
pub fn load_from(cache_dir: &Path, split: MnistSplit) -> std::io::Result<MnistData> {
    ensure_downloaded(cache_dir)?;
    let (img_name, lbl_name) = match split {
        MnistSplit::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        MnistSplit::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    };
    let img_bytes = fs::read(cache_dir.join(img_name))?;
    let lbl_bytes = fs::read(cache_dir.join(lbl_name))?;

    let (images, n_img) = parse_images(&img_bytes)?;
    let (labels, n_lbl) = parse_labels(&lbl_bytes)?;
    assert_eq!(n_img, n_lbl, "MNIST image/label count mismatch: {} vs {}", n_img, n_lbl);

    Ok(MnistData { images, labels, len: n_img })
}

fn parse_images(bytes: &[u8]) -> std::io::Result<(Tensor, usize)> {
    if bytes.len() < 16 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "IDX images header too short"));
    }
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x00000803 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                                       format!("bad IDX images magic: {:#010x}", magic)));
    }
    let n = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let rows = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let cols = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

    let expected = 16 + n * rows * cols;
    if bytes.len() != expected {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                                       format!("IDX images size mismatch: got {} expected {}", bytes.len(), expected)));
    }

    // Normalize to [0, 1].
    let data: Vec<f32> = bytes[16..].iter().map(|&b| b as f32 / 255.0).collect();
    let t = Tensor::from_vec(data, &[n, 1, rows, cols]);
    Ok((t, n))
}

fn parse_labels(bytes: &[u8]) -> std::io::Result<(Tensor, usize)> {
    if bytes.len() < 8 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "IDX labels header too short"));
    }
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x00000801 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                                       format!("bad IDX labels magic: {:#010x}", magic)));
    }
    let n = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if bytes.len() != 8 + n {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                                       format!("IDX labels size mismatch: got {} expected {}", bytes.len(), 8 + n)));
    }

    let data: Vec<f32> = bytes[8..].iter().map(|&b| b as f32).collect();
    let t = Tensor::from_vec(data, &[n]);
    Ok((t, n))
}

/// `Dataset` wrapper over MNIST so `DataLoader` can shuffle and batch it.
pub struct MnistDataset {
    images: Tensor,
    labels: Tensor,
    len: usize,
}

impl MnistDataset {
    pub fn new(data: MnistData) -> Self {
        MnistDataset { images: data.images, labels: data.labels, len: data.len }
    }

    /// Convenience: load + wrap in one call.
    pub fn load(split: MnistSplit) -> std::io::Result<Self> {
        Ok(Self::new(load(split)?))
    }

    /// Raw tensor access (useful for whole-batch evaluation).
    pub fn images(&self) -> &Tensor { &self.images }
    pub fn labels(&self) -> &Tensor { &self.labels }
}

impl Dataset for MnistDataset {
    fn len(&self) -> usize { self.len }

    fn get(&self, index: usize) -> (Tensor, Tensor) {
        assert!(index < self.len, "MNIST index {} out of range (len {})", index, self.len);
        let img_data = self.images.to_vec();
        let lbl_data = self.labels.to_vec();
        let img_start = index * 28 * 28;
        let img = Tensor::from_vec(
            img_data[img_start..img_start + 28 * 28].to_vec(),
            &[1, 28, 28],
        );
        let lbl = Tensor::from_vec(vec![lbl_data[index]], &[1]);
        (img, lbl)
    }
}
