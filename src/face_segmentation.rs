//! Per-face body-part segmentation, computed by sampling the bundled
//! `body_parts_segmentation.png` at each face's centre UV. Mirrors
//! `anny/src/anny/face_segmentation.py:10–38`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use crate::models::full_model::Model;

#[derive(Debug, Error)]
pub enum SegmentationError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("unknown body-part label: {0}")]
    UnknownLabel(String),
    #[error("face has no texture coordinate indices")]
    MissingTextureCoords,
}

#[derive(Debug, Deserialize)]
struct SegmentationMetadata {
    colors: HashMap<String, [u8; 3]>,
}

/// Loads + processes the bundled segmentation. Hold this around to avoid
/// re-reading the image on every call to [`Self::face_mask`].
pub struct FaceSegmentation {
    label_to_color: HashMap<String, [u8; 3]>,
    /// `[F]` of `[u8; 3]` — RGB color sampled at each face's centre UV.
    face_colors: Vec<[u8; 3]>,
}

impl FaceSegmentation {
    pub fn new(model: &Model, data_root: &Path) -> Result<Self, SegmentationError> {
        let png_path: PathBuf = data_root.join("segmentation/body_parts_segmentation.png");
        let yaml_path: PathBuf = data_root.join("segmentation/body_parts_segmentation.yaml");
        let metadata: SegmentationMetadata =
            serde_yaml::from_str(&std::fs::read_to_string(&yaml_path)?)?;

        let img = image::open(&png_path)?.to_rgb8();
        let (img_w, img_h) = (img.width() as usize, img.height() as usize);
        let img_buf = img.into_raw(); // row-major, [r, g, b, r, g, b, ...]

        // Per-face centre UV: average of the texture coords used by each face.
        let face_colors = sample_face_colors(model, &img_buf, img_w, img_h)?;

        Ok(Self {
            label_to_color: metadata.colors,
            face_colors,
        })
    }

    /// Returns a `[F]` boolean mask: `true` iff the face's body part matches
    /// any of the requested labels.
    pub fn face_mask(&self, labels: &[&str]) -> Result<Vec<bool>, SegmentationError> {
        let mut targets: Vec<[u8; 3]> = Vec::with_capacity(labels.len());
        for label in labels {
            let color = self
                .label_to_color
                .get(*label)
                .ok_or_else(|| SegmentationError::UnknownLabel((*label).to_string()))?;
            targets.push(*color);
        }
        let mut mask = vec![false; self.face_colors.len()];
        for (i, fc) in self.face_colors.iter().enumerate() {
            for t in &targets {
                if fc == t {
                    mask[i] = true;
                    break;
                }
            }
        }
        Ok(mask)
    }
}

fn sample_face_colors(
    model: &Model,
    img_buf: &[u8],
    img_w: usize,
    img_h: usize,
) -> Result<Vec<[u8; 3]>, SegmentationError> {
    let mut out = Vec::with_capacity(model.face_texture_coordinate_indices.len());
    let tex = &model.texture_coordinates;
    for face_uv_indices in &model.face_texture_coordinate_indices {
        if face_uv_indices.is_empty() {
            return Err(SegmentationError::MissingTextureCoords);
        }
        let mut su = 0.0_f64;
        let mut sv = 0.0_f64;
        for &uv_idx in face_uv_indices {
            let uv = &tex[uv_idx as usize];
            su += uv[0];
            sv += uv[1];
        }
        let n = face_uv_indices.len() as f64;
        let u = su / n;
        let v_ = sv / n;
        // Match Python's pixel-coordinate mapping (with its idiosyncratic
        // shape[0]/shape[1] choices — they cancel for a square image).
        let pixel_u = (u * img_w as f64).round() as i64;
        let pixel_v = ((1.0 - v_) * img_h as f64).round() as i64;
        let px = pixel_u.clamp(0, img_h as i64 - 1) as usize;
        let py = pixel_v.clamp(0, img_w as i64 - 1) as usize;
        let off = (py * img_w + px) * 3;
        out.push([img_buf[off], img_buf[off + 1], img_buf[off + 2]]);
    }
    Ok(out)
}
