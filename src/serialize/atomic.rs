//! One atomic file write for checkpoints and the dataset cache: a temporary
//! file next to the target, `sync_all`, then a rename into place.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::Result;

/// Write to `path` so a crash leaves it either absent or complete.
///
/// `write` fills a temporary file next to `path`, which is synced and renamed
/// into place only when `write` succeeds, and removed when anything fails. The
/// previous file is never touched until the rename.
pub(crate) fn write_atomic(path: &Path, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
    let partial = partial_path(path);
    let written: Result<()> = (|| {
        let mut file = File::create(&partial)?;
        write(&mut file)?;
        file.sync_all()?;
        fs::rename(&partial, path)?;
        Ok(())
    })();
    if written.is_err() {
        let _ = fs::remove_file(&partial);
    }
    written
}

/// [`write_atomic`] for bytes already in memory.
pub(crate) fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic(path, |file| Ok(file.write_all(bytes)?))
}

/// `{path}.{pid}.{serial}.partial`. The pid and the counter keep concurrent
/// writers, in and across processes, off each other's temporary file.
fn partial_path(path: &Path) -> PathBuf {
    static SERIAL: AtomicU32 = AtomicU32::new(0);
    let mut name = OsString::from(path.as_os_str());
    name.push(format!(
        ".{}.{}.partial",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// A fresh directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("fastnn-atomic-{}-{name}", std::process::id()));
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

    fn partials_in(dir: &Path) -> usize {
        fs::read_dir(dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".partial")
            })
            .count()
    }

    #[test]
    fn writes_the_bytes_and_leaves_no_partial() {
        let dir = TempDir::new("write");
        let target = dir.0.join("data.bin");
        write_atomic_bytes(&target, b"hello").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"hello");
        assert_eq!(partials_in(&dir.0), 0);
    }

    #[test]
    fn replaces_an_existing_file() {
        let dir = TempDir::new("replace");
        let target = dir.0.join("data.bin");
        fs::write(&target, b"old").unwrap();
        write_atomic_bytes(&target, b"new").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new");
    }

    #[test]
    fn a_failed_write_leaves_the_old_file_and_no_partial() {
        let dir = TempDir::new("write-fails");
        let target = dir.0.join("data.bin");
        fs::write(&target, b"old").unwrap();

        let result = write_atomic(&target, |file| {
            file.write_all(b"half")?;
            Err(Error::Checkpoint("disk full".into()))
        });

        assert!(result.is_err());
        assert_eq!(fs::read(&target).unwrap(), b"old");
        assert_eq!(partials_in(&dir.0), 0);
    }

    #[test]
    fn a_failed_rename_leaves_no_partial() {
        let dir = TempDir::new("rename-fails");
        // A file cannot replace a directory, so the rename fails after the write.
        let target = dir.0.join("taken");
        fs::create_dir(&target).unwrap();
        assert!(write_atomic_bytes(&target, b"x").is_err());
        assert_eq!(partials_in(&dir.0), 0);
    }
}
