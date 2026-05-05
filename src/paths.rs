//! Cache directory resolution and the (gated) SMPL-X non-commercial download.
//!
//! Mirrors `anny/src/anny/paths.py`. The cache dir defaults to
//! `~/.cache/anny`, overridable via the `ANNY_CACHE_DIR` env var.

use std::path::PathBuf;

const ANNY2SMPLX_RELATIVE: &str = "noncommercial/anny2smplx.pth";

/// Returns the resolved Anny cache directory:
/// `$ANNY_CACHE_DIR` if set, else `$HOME/.cache/anny`. Falls back to `./.anny`
/// in the unlikely event neither is available.
pub fn cache_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("ANNY_CACHE_DIR") {
        return PathBuf::from(custom);
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".cache").join("anny");
    }
    PathBuf::from(".anny")
}

/// Path to the SMPL-X retopology weights once downloaded.
pub fn anny2smplx_path() -> PathBuf {
    cache_dir().join(ANNY2SMPLX_RELATIVE)
}

#[cfg(feature = "smplx-download")]
pub mod download {
    //! Downloads the non-commercial SMPL-X retopology data and prints its
    //! bundled LICENSE/NOTICE before returning. Mirrors
    //! `download_noncommercial_data` in `anny/src/anny/paths.py`.

    use super::cache_dir;
    use std::fs;
    use std::io::Read;
    use std::path::Path;

    const URL: &str = "https://download.europe.naverlabs.com/humans/Anny/noncommercial.zip";

    #[derive(Debug, thiserror::Error)]
    pub enum DownloadError {
        #[error("io: {0}")]
        Io(#[from] std::io::Error),
        #[error("http: {0}")]
        Http(#[from] reqwest::Error),
        #[error("zip: {0}")]
        Zip(#[from] zip::result::ZipError),
    }

    pub fn fetch_noncommercial() -> Result<(), DownloadError> {
        let cache = cache_dir();
        let dest = cache.join("noncommercial");
        fs::create_dir_all(&dest)?;
        let zip_path = cache.join("noncommercial.zip");

        let bytes = reqwest::blocking::get(URL)?.bytes()?;
        fs::write(&zip_path, &bytes)?;

        let file = fs::File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(&dest)?;
        fs::remove_file(&zip_path)?;

        emit_doc(&dest, "LICENSE.txt");
        emit_doc(&dest, "NOTICE.txt");
        Ok(())
    }

    fn emit_doc(dest: &Path, name: &str) {
        let path = dest.join(name);
        if let Ok(mut f) = fs::File::open(&path) {
            let mut s = String::new();
            if f.read_to_string(&mut s).is_ok() {
                eprintln!("--- {name} ---\n{s}\n--------------");
            }
        } else {
            eprintln!("--- {name} not found ---");
        }
    }
}
