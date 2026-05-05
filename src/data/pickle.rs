//! Thin wrapper over `candle_core::pickle` for reading PyTorch `.pth` files.
//!
//! All `.pth` files in the Anny data tree (`boys.pth`, `girls.pth`,
//! `anny2smplx.pth`) are saved with `weights_only=True` semantics — the
//! payload is just nested dicts of tensors, no arbitrary Python classes —
//! so candle's reader is sufficient.

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PickleError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("missing tensor {0}")]
    Missing(String),
}

/// Returns the tensor keys present under the given sub-dict (or the root dict
/// when `sub_key` is `None`), sorted.
///
/// For the calibration files (`boys.pth`, `girls.pth`), tensors live under
/// keys like `"conditional_height_distribution"`; passing `None` yields no
/// tensors because candle's pickle reader does not auto-flatten nested dicts.
pub fn list_keys(
    path: impl AsRef<Path>,
    sub_key: Option<&str>,
) -> Result<Vec<String>, PickleError> {
    let pth = PthTensors::new(path.as_ref(), sub_key)?;
    let mut keys: Vec<String> = pth.tensor_infos().keys().cloned().collect();
    keys.sort();
    Ok(keys)
}

/// Loads every tensor in the (optionally nested) state-dict into a flat
/// `BTreeMap<name, Tensor>`.
pub fn load_all(
    path: impl AsRef<Path>,
    sub_key: Option<&str>,
    device: &Device,
) -> Result<BTreeMap<String, Tensor>, PickleError> {
    let pth = PthTensors::new(path.as_ref(), sub_key)?;
    let mut out = BTreeMap::new();
    for name in pth.tensor_infos().keys() {
        let tensor = pth
            .get(name)?
            .ok_or_else(|| PickleError::Missing(name.clone()))?;
        out.insert(name.clone(), tensor.to_device(device)?);
    }
    Ok(out)
}

/// Loads a single tensor by its dotted key, casting to the requested dtype on the
/// requested device. Returns `Missing` if the key is not present.
pub fn load_tensor(
    path: impl AsRef<Path>,
    key: &str,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, PickleError> {
    let pth = PthTensors::new(path.as_ref(), None)?;
    let t = pth
        .get(key)?
        .ok_or_else(|| PickleError::Missing(key.to_string()))?;
    Ok(t.to_dtype(dtype)?.to_device(device)?)
}
