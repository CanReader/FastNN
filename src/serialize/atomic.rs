//! Atomic file writes: write to a per-process temp, `sync_all`, then rename.

use std::ffi::OsString;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::Result;

/// Write to `path` atomically.
///
/// Writes into `{path}.partial.{pid}`, calls [`File::sync_all`], then renames
/// the temp to `path`. A crash mid-write leaves the previous file intact. On
/// error the temp is removed so it never accumulates.
pub(crate) fn write_atomic(
    path: &Path,
    write_fn: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let tmp = tmp_path(path);
    let result = write_and_sync(&tmp, write_fn);
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
        return result;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn write_and_sync(
    path: &Path,
    write_fn: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let mut file = File::create(path)?;
    write_fn(&mut file)?;
    file.sync_all()?;
    Ok(())
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = OsString::from(path.as_os_str());
    name.push(format!(".partial.{}", std::process::id()));
    PathBuf::from(name)
}
