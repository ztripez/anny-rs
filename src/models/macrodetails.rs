//! Macrodetails (and local-change) blend-shape loader.
//!
//! Mirrors `load_macrodetails` (`full_model.py:35–115`) plus the local-change
//! loader at lines 297–315. Walks the `data/mpfb2/targets/` tree, parses each
//! `.target.gz` into a dense `[V, 3]` delta buffer, applies the world
//! transformation, and stacks every blend shape into a single
//! `[C, V, 3]` tensor along with a `[C, 26]` mask telling the masked-product
//! reduction in [`crate::phenotype`] which variations gate which blend shape.
//!
//! `n_macrodetails` is the count of macrodetails-only entries (universal +
//! race + height + proportions + breast) — local changes follow these but
//! have all-zero mask rows and instead get appended via the local-change
//! coefficient path in [`crate::phenotype`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use thiserror::Error;

use crate::data::target_gz;
use crate::phenotype::{PHENOTYPE_VARIATION_COUNT, PHENOTYPE_VARIATIONS};

#[derive(Debug, Error)]
pub enum MacrodetailsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("target.gz: {0}")]
    TargetGz(#[from] target_gz::TargetGzError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("missing required file: {0}")]
    Missing(PathBuf),
}

/// World transformation as a 3×3 row-major matrix. Applied via `delta @ Mᵀ`,
/// matching `roma.Linear.apply` semantics.
pub type WorldTransform3x3 = [[f64; 3]; 3];

/// Default world transformation: 0.1 (decimeters → metres) × R_x(90°).
/// Equivalent to `roma.Linear(0.1 * roma.euler_to_rotmat("X", [90], degrees=True))`.
pub fn default_world_transform() -> WorldTransform3x3 {
    let s = 0.1_f64;
    // R_x(90°): rows are [1, 0, 0], [0, 0, -1], [0, 1, 0]
    [[s, 0.0, 0.0], [0.0, 0.0, -s], [0.0, s, 0.0]]
}

/// Stacked phenotype blend shapes plus the per-blend-shape variation mask.
pub struct StackedBlendShapes {
    /// `[C, V, 3]`: every macrodetails entry plus every local-change pair.
    pub blendshapes: Tensor,
    /// `[C_macro, 26]`: variation mask used by the masked-product phenotype
    /// coefficient computation. Local-change rows are not included here.
    pub mask: Tensor,
    /// Number of macrodetails entries (i.e. rows of `mask`). Local-change
    /// blend shapes occupy `blendshapes[n_macrodetails..]`.
    pub n_macrodetails: usize,
    /// Names of local changes appended after macrodetails, in pair order
    /// (positive then negative — they appear interleaved in `blendshapes`).
    pub local_change_labels: Vec<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// Loader
// ────────────────────────────────────────────────────────────────────────────

const NEWBORN_SCALING: [f64; 3] = [0.922, 0.922, 0.75];
const NEWBORN_NORMALISING_FACTOR: f64 = 3.0;

/// Loads every macrodetails and local-change blend shape under
/// `<data_root>/mpfb2/targets/`.
///
/// `template_vertices_world`: post-transform `[V, 3]` template, needed for
/// newborn-blend-shape scaling.
/// `world_transform`: 3×3 linear transform applied to every loaded delta.
pub fn load_all(
    data_root: &Path,
    template_vertices_world: &[[f64; 3]],
    world_transform: &WorldTransform3x3,
    dtype: DType,
    device: &Device,
) -> std::result::Result<StackedBlendShapes, MacrodetailsError> {
    let v = template_vertices_world.len();

    // Variation-name → flat-mask-index lookup.
    let label_idx: HashMap<&'static str, usize> = {
        let mut map = HashMap::new();
        let mut i = 0usize;
        for (_, vars) in PHENOTYPE_VARIATIONS {
            for var in *vars {
                map.insert(*var, i);
                i += 1;
            }
        }
        map
    };
    debug_assert_eq!(label_idx.len(), PHENOTYPE_VARIATION_COUNT);

    let macrodetails_dir = data_root.join("mpfb2/targets/macrodetails");
    let mut blend_buf: Vec<Vec<f64>> = Vec::new();
    let mut masks: Vec<[f64; PHENOTYPE_VARIATION_COUNT]> = Vec::new();

    let load = |relative: &str| -> std::result::Result<Vec<f64>, MacrodetailsError> {
        let path = macrodetails_dir.join(relative);
        let raw = target_gz::load(&path, v)?;
        Ok(apply_world_transform(&raw, world_transform))
    };

    // ── universal: gender × age × muscle × weight ─────────────────────────
    for gender in variations("gender") {
        for age in variations("age") {
            let age_to_load = if *age == "newborn" { "baby" } else { *age };
            for muscle in variations("muscle") {
                for weight in variations("weight") {
                    let path =
                        format!("universal-{gender}-{age_to_load}-{muscle}-{weight}.target.gz");
                    let mut bs = load(&path)?;
                    if *age == "newborn" {
                        apply_newborn_scaling(&mut bs, template_vertices_world);
                    }
                    let mut mask = [0.0_f64; PHENOTYPE_VARIATION_COUNT];
                    set_mask(&mut mask, &label_idx, &[gender, age, muscle, weight]);
                    blend_buf.push(bs);
                    masks.push(mask);
                }
            }
        }
    }

    // ── race: race × gender × age ────────────────────────────────────────
    for race in variations("race") {
        for gender in variations("gender") {
            for age in variations("age") {
                let age_to_load = if *age == "newborn" { "baby" } else { *age };
                let path = format!("{race}-{gender}-{age_to_load}.target.gz");
                let mut bs = load(&path)?;
                if *age == "newborn" {
                    apply_newborn_scaling(&mut bs, template_vertices_world);
                }
                let mut mask = [0.0_f64; PHENOTYPE_VARIATION_COUNT];
                set_mask(&mut mask, &label_idx, &[race, gender, age]);
                blend_buf.push(bs);
                masks.push(mask);
            }
        }
    }

    // ── height: gender × age × muscle × weight × height ──────────────────
    for gender in variations("gender") {
        for age in variations("age") {
            let age_to_load = if *age == "newborn" { "baby" } else { *age };
            for muscle in variations("muscle") {
                for weight in variations("weight") {
                    for height in variations("height") {
                        let path = format!(
                            "height/{gender}-{age_to_load}-{muscle}-{weight}-{height}.target.gz"
                        );
                        let mut bs = load(&path)?;
                        if *age == "newborn" {
                            apply_newborn_scaling(&mut bs, template_vertices_world);
                        }
                        let mut mask = [0.0_f64; PHENOTYPE_VARIATION_COUNT];
                        set_mask(
                            &mut mask,
                            &label_idx,
                            &[gender, age, muscle, weight, height],
                        );
                        blend_buf.push(bs);
                        masks.push(mask);
                    }
                }
            }
        }
    }

    // ── proportions: gender × age × muscle × weight × proportions, skipping newborn/baby ─
    for gender in variations("gender") {
        for age in variations("age") {
            if *age == "newborn" || *age == "baby" {
                continue;
            }
            for muscle in variations("muscle") {
                for weight in variations("weight") {
                    for proportions in variations("proportions") {
                        let path = format!(
                            "proportions/{gender}-{age}-{muscle}-{weight}-{proportions}.target.gz"
                        );
                        let bs = load(&path)?;
                        let mut mask = [0.0_f64; PHENOTYPE_VARIATION_COUNT];
                        set_mask(
                            &mut mask,
                            &label_idx,
                            &[gender, age, muscle, weight, proportions],
                        );
                        blend_buf.push(bs);
                        masks.push(mask);
                    }
                }
            }
        }
    }

    // ── breast: female × age × muscle × weight × cupsize × firmness, skipping newborn/baby; missing files allowed ─
    let breast_dir = data_root.join("mpfb2/targets/breast");
    let gender = "female";
    for age in variations("age") {
        if *age == "newborn" || *age == "baby" {
            continue;
        }
        for muscle in variations("muscle") {
            for weight in variations("weight") {
                for cupsize in variations("cupsize") {
                    for firmness in variations("firmness") {
                        let filename = format!(
                            "{gender}-{age}-{muscle}-{weight}-{cupsize}-{firmness}.target.gz"
                        );
                        let path = breast_dir.join(&filename);
                        if !path.exists() {
                            continue;
                        }
                        let raw = target_gz::load(&path, v)?;
                        let bs = apply_world_transform(&raw, world_transform);
                        let mut mask = [0.0_f64; PHENOTYPE_VARIATION_COUNT];
                        set_mask(
                            &mut mask,
                            &label_idx,
                            &[gender, age, muscle, weight, cupsize, firmness],
                        );
                        blend_buf.push(bs);
                        masks.push(mask);
                    }
                }
            }
        }
    }

    let n_macrodetails = blend_buf.len();

    // ── local changes ────────────────────────────────────────────────────
    let local_change_labels = load_local_changes(data_root, v, world_transform, &mut blend_buf)?;

    // Stack into final tensors.
    let mut blend_flat: Vec<f64> = Vec::with_capacity(blend_buf.len() * v * 3);
    for bs in &blend_buf {
        blend_flat.extend_from_slice(bs);
    }
    let blendshapes =
        Tensor::from_vec(blend_flat, (blend_buf.len(), v, 3), device)?.to_dtype(dtype)?;

    let mut mask_flat: Vec<f64> = Vec::with_capacity(masks.len() * PHENOTYPE_VARIATION_COUNT);
    for row in &masks {
        mask_flat.extend_from_slice(row);
    }
    let mask = Tensor::from_vec(mask_flat, (masks.len(), PHENOTYPE_VARIATION_COUNT), device)?
        .to_dtype(dtype)?;

    Ok(StackedBlendShapes {
        blendshapes,
        mask,
        n_macrodetails,
        local_change_labels,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn variations(key: &str) -> &'static [&'static str] {
    for (k, vars) in PHENOTYPE_VARIATIONS {
        if *k == key {
            return vars;
        }
    }
    panic!("unknown phenotype variation key: {key}");
}

fn set_mask(
    mask: &mut [f64; PHENOTYPE_VARIATION_COUNT],
    label_idx: &HashMap<&'static str, usize>,
    components: &[&str],
) {
    for c in components {
        let idx = *label_idx
            .get(c)
            .unwrap_or_else(|| panic!("unknown label {c}"));
        mask[idx] = 1.0;
    }
}

fn apply_world_transform(raw: &[f64], m: &WorldTransform3x3) -> Vec<f64> {
    // raw is [V*3] row-major: each delta is (x, y, z). Apply v' = M · v.
    let mut out = vec![0.0_f64; raw.len()];
    let n = raw.len() / 3;
    for i in 0..n {
        let x = raw[i * 3];
        let y = raw[i * 3 + 1];
        let z = raw[i * 3 + 2];
        out[i * 3] = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        out[i * 3 + 1] = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        out[i * 3 + 2] = m[2][0] * x + m[2][1] * y + m[2][2] * z;
    }
    out
}

fn apply_newborn_scaling(bs: &mut [f64], template: &[[f64; 3]]) {
    // bs *= NEWBORN_SCALING[None,:]
    // bs += (NEWBORN_SCALING - 1)/3 * template
    let s = NEWBORN_SCALING;
    let factor = NEWBORN_NORMALISING_FACTOR;
    let n = template.len();
    debug_assert_eq!(bs.len(), n * 3);
    for i in 0..n {
        for j in 0..3 {
            let scaled = s[j] * bs[i * 3 + j];
            let extra = (s[j] - 1.0) / factor * template[i][j];
            bs[i * 3 + j] = scaled + extra;
        }
    }
}

// ── Local changes ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct TargetCategory {
    #[serde(default)]
    opposites: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct TargetMetadata {
    #[serde(default)]
    categories: Vec<TargetCategory>,
}

/// Walks the local-change opposites declared in `targets/target.json`. For
/// each category with `negative-<side>` and `positive-<side>` slots, loads
/// both `.target.gz` files and appends `(positive, negative)` to `blend_buf`.
/// Returns the list of positive-pole labels in the same pair order.
fn load_local_changes(
    data_root: &Path,
    vertex_count: usize,
    world_transform: &WorldTransform3x3,
    blend_buf: &mut Vec<Vec<f64>>,
) -> std::result::Result<Vec<String>, MacrodetailsError> {
    let json_path = data_root.join("mpfb2/targets/target.json");
    let text = std::fs::read_to_string(&json_path)?;
    let meta: HashMap<String, TargetMetadata> = serde_json::from_str(&text)?;

    let targets_dir = data_root.join("mpfb2/targets");
    let mut positive_labels = Vec::new();
    let mut keys: Vec<&String> = meta.keys().collect();
    // The Python relies on dict iteration order (insertion). serde_json doesn't
    // preserve that; sort alphabetically for a deterministic order. (The model
    // forward pass doesn't depend on order beyond consistent indexing.)
    keys.sort();
    for key in keys {
        if key == "genitals" {
            continue;
        }
        let category_list = &meta[key].categories;
        for category in category_list {
            let opposites = match &category.opposites {
                Some(o) => o,
                None => continue,
            };
            for side in &["left", "right", "unsided"] {
                let neg_key = format!("negative-{side}");
                let pos_key = format!("positive-{side}");
                let neg_label = match opposites.get(&neg_key) {
                    Some(s) if !s.is_empty() => s,
                    _ => continue,
                };
                let pos_label = match opposites.get(&pos_key) {
                    Some(s) if !s.is_empty() => s,
                    _ => continue,
                };
                let neg_path = targets_dir.join(key).join(format!("{neg_label}.target.gz"));
                let pos_path = targets_dir.join(key).join(format!("{pos_label}.target.gz"));
                let neg_raw = target_gz::load(&neg_path, vertex_count)?;
                let pos_raw = target_gz::load(&pos_path, vertex_count)?;
                let neg_bs = apply_world_transform(&neg_raw, world_transform);
                let pos_bs = apply_world_transform(&pos_raw, world_transform);
                positive_labels.push(pos_label.clone());
                blend_buf.push(pos_bs);
                blend_buf.push(neg_bs);
            }
        }
    }
    Ok(positive_labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("anny")
            .join("src")
            .join("anny")
            .join("data")
    }

    #[test]
    fn world_transform_rotates_y_to_z() {
        let m = default_world_transform();
        // Apply to (0, 1, 0): expect (0, 0, 0.1)
        let v = vec![0.0_f64, 1.0, 0.0];
        let out = apply_world_transform(&v, &m);
        assert!((out[0]).abs() < 1e-15);
        assert!((out[1]).abs() < 1e-15);
        assert!((out[2] - 0.1).abs() < 1e-15);

        // (0, 0, 1) → (0, -0.1, 0)
        let v2 = vec![0.0_f64, 0.0, 1.0];
        let out2 = apply_world_transform(&v2, &m);
        assert!((out2[0]).abs() < 1e-15);
        assert!((out2[1] - -0.1).abs() < 1e-15);
        assert!((out2[2]).abs() < 1e-15);
    }

    #[test]
    #[ignore = "loads ~650 .target.gz files; slow but green when run"]
    fn loads_full_macrodetails() {
        let device = Device::Cpu;
        // Use the real base mesh's vertex count.
        let mesh = crate::data::obj::load(data_root().join("mpfb2/3dobjs/base.obj")).unwrap();
        let template_vertices: Vec<[f64; 3]> = mesh
            .vertices
            .iter()
            .map(|v| {
                let m = default_world_transform();
                [
                    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
                ]
            })
            .collect();

        let result = load_all(
            &data_root(),
            &template_vertices,
            &default_world_transform(),
            DType::F64,
            &device,
        )
        .unwrap();
        // Expect ~564 macrodetails entries (the explore agent reported 564).
        assert!(result.n_macrodetails > 500, "got {}", result.n_macrodetails);
        assert_eq!(
            result.mask.dims(),
            &[result.n_macrodetails, PHENOTYPE_VARIATION_COUNT]
        );
        // blendshapes has macrodetails + 2 * local_changes rows.
        let expected = result.n_macrodetails + 2 * result.local_change_labels.len();
        assert_eq!(result.blendshapes.dim(0).unwrap(), expected);
    }
}
