//! Conditional Beta priors over phenotype scalars, calibrated against
//! WHO height-for-age data. Direct port of
//! [`anny/src/anny/shape_distribution.py`](../../../anny/src/anny/shape_distribution.py).
//!
//! The on-disk layout (in `data/shape_calibration/{boys,girls}.pth`) is:
//! a top-level dict whose keys are `morphological_age_mapping`,
//! `conditional_height_distribution`, `conditional_weight_distribution`,
//! `conditional_muscle_distribution`, `conditional_proportions_distribution`.
//! Each conditional dict carries `age_anchors`, `alpha_anchors`,
//! `beta_anchors` (all 1-D `f64`).
//!
//! ## Caveat
//!
//! The weight/muscle/proportions distributions store `age_anchors` in
//! *morphological years* (0–110), but the Python `sample()` passes the
//! Anny-scale age (0–1) to all four. We reproduce that behaviour faithfully
//! for parity — we are not "fixing" the upstream bug.

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use rand::Rng;
use rand_distr::{Beta, Distribution, Uniform};
use thiserror::Error;

use crate::data::pickle::{self, PickleError};
use crate::phenotype::PhenotypeValues;
use crate::utils::interpolation::linear_interpolation_coefficients;

#[derive(Debug, Error)]
pub enum ShapeDistributionError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("pickle: {0}")]
    Pickle(#[from] PickleError),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("rand_distr: {0}")]
    Rand(#[from] rand_distr::BetaError),
}

// ────────────────────────────────────────────────────────────────────────────
// Bidirectional age mapping.
// ────────────────────────────────────────────────────────────────────────────

/// Linear interpolation between Anny's internal age scalar and an external
/// "morphological age" expressed in years. Mirrors `MorphologicalAgeMapping`
/// (lines 15–34 of `shape_distribution.py`).
#[derive(Debug, Clone)]
pub struct MorphologicalAgeMapping {
    pub anny_age_anchors: Tensor,
    pub morphological_age_anchors: Tensor,
}

impl MorphologicalAgeMapping {
    pub fn from_state_dict(
        sd: &BTreeMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, ShapeDistributionError> {
        let aa = sd
            .get("anny_age_anchors")
            .ok_or_else(|| ShapeDistributionError::Missing("anny_age_anchors".into()))?
            .to_dtype(dtype)?
            .to_device(device)?;
        let ma = sd
            .get("morphological_age_anchors")
            .ok_or_else(|| ShapeDistributionError::Missing("morphological_age_anchors".into()))?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok(Self {
            anny_age_anchors: aa,
            morphological_age_anchors: ma,
        })
    }

    /// `morphological_age` (years) → Anny-scale age. Extrapolates linearly.
    pub fn morphological_to_anny_age(&self, morphological_age: &Tensor) -> Result<Tensor> {
        let coeffs = linear_interpolation_coefficients(
            morphological_age,
            &self.morphological_age_anchors,
            true,
        )?;
        // einsum('bk, k -> b') ≡ matmul([B, K], [K, 1]).squeeze.
        let aa_col = self.anny_age_anchors.unsqueeze(1)?;
        let result = coeffs.matmul(&aa_col)?.squeeze(1)?;
        Ok(result)
    }

    /// Inverse direction: Anny-scale age → morphological years.
    pub fn anny_to_morphological_age(&self, anny_age: &Tensor) -> Result<Tensor> {
        let coeffs = linear_interpolation_coefficients(anny_age, &self.anny_age_anchors, true)?;
        let ma_col = self.morphological_age_anchors.unsqueeze(1)?;
        let result = coeffs.matmul(&ma_col)?.squeeze(1)?;
        Ok(result)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Conditional Beta distribution.
// ────────────────────────────────────────────────────────────────────────────

/// A `Beta(α(age), β(age))` distribution where α and β are linearly
/// interpolated against `age_anchors`. Mirrors `ConditionalBetaDistribution`
/// (lines 36–67 of `shape_distribution.py`).
#[derive(Debug, Clone)]
pub struct ConditionalBetaDistribution {
    pub age_anchors: Tensor,
    pub alpha_anchors: Tensor,
    pub beta_anchors: Tensor,
}

impl ConditionalBetaDistribution {
    pub fn from_state_dict(
        sd: &BTreeMap<String, Tensor>,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, ShapeDistributionError> {
        let pull = |k: &str| -> std::result::Result<Tensor, ShapeDistributionError> {
            sd.get(k)
                .ok_or_else(|| ShapeDistributionError::Missing(k.into()))?
                .to_dtype(dtype)
                .map_err(Into::into)
                .and_then(|t| t.to_device(device).map_err(Into::into))
        };
        Ok(Self {
            age_anchors: pull("age_anchors")?,
            alpha_anchors: pull("alpha_anchors")?,
            beta_anchors: pull("beta_anchors")?,
        })
    }

    /// Returns `(alpha, beta)` per batch element for the supplied age tensor.
    /// `age` is `[B]`. Output is two `[B]` tensors.
    pub fn distribution_params(&self, age: &Tensor) -> Result<(Tensor, Tensor)> {
        let coefs = linear_interpolation_coefficients(age, &self.age_anchors, false)?;
        let alpha = coefs
            .matmul(&self.alpha_anchors.unsqueeze(1)?)?
            .squeeze(1)?;
        let beta = coefs.matmul(&self.beta_anchors.unsqueeze(1)?)?.squeeze(1)?;
        Ok((alpha, beta))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Bundle: one set of four distributions per gender.
// ────────────────────────────────────────────────────────────────────────────

/// Beta priors over height/weight/muscle/proportions for one gender.
#[derive(Debug, Clone)]
pub struct GenderBetaSet {
    pub height: ConditionalBetaDistribution,
    pub weight: ConditionalBetaDistribution,
    pub muscle: ConditionalBetaDistribution,
    pub proportions: ConditionalBetaDistribution,
}

impl GenderBetaSet {
    fn load(
        path: &Path,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, ShapeDistributionError> {
        let load_sub = |key: &str| -> std::result::Result<
            ConditionalBetaDistribution,
            ShapeDistributionError,
        > {
            let sd = pickle::load_all(path, Some(key), device)?;
            ConditionalBetaDistribution::from_state_dict(&sd, dtype, device)
        };
        Ok(Self {
            height: load_sub("conditional_height_distribution")?,
            weight: load_sub("conditional_weight_distribution")?,
            muscle: load_sub("conditional_muscle_distribution")?,
            proportions: load_sub("conditional_proportions_distribution")?,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Top-level distribution.
// ────────────────────────────────────────────────────────────────────────────

/// Full phenotype prior: morphological-age mapping + Beta priors per gender.
/// Mirrors `SimpleShapeDistribution` (lines 87–186 of `shape_distribution.py`).
#[derive(Debug, Clone)]
pub struct SimpleShapeDistribution {
    pub mapping: MorphologicalAgeMapping,
    pub boys: GenderBetaSet,
    pub girls: GenderBetaSet,
    /// Phenotype labels expected by the consumer model. Sampled values will
    /// fill these labels; non-derived ones get uniform `[0, 1]` samples.
    pub phenotype_labels: Vec<String>,
    pub dtype: DType,
    pub device: Device,
}

impl SimpleShapeDistribution {
    /// Loads from the canonical
    /// `data/shape_calibration/{boys,girls}.pth` files.
    pub fn load_default(
        data_root: &Path,
        phenotype_labels: Vec<String>,
        dtype: DType,
        device: &Device,
    ) -> std::result::Result<Self, ShapeDistributionError> {
        let boys_path = data_root.join("shape_calibration/boys.pth");
        let girls_path = data_root.join("shape_calibration/girls.pth");
        let boys = GenderBetaSet::load(&boys_path, dtype, device)?;
        let girls = GenderBetaSet::load(&girls_path, dtype, device)?;
        let mapping_sd = pickle::load_all(&boys_path, Some("morphological_age_mapping"), device)?;
        let mapping = MorphologicalAgeMapping::from_state_dict(&mapping_sd, dtype, device)?;
        Ok(Self {
            mapping,
            boys,
            girls,
            phenotype_labels,
            dtype,
            device: device.clone(),
        })
    }

    /// Draws `batch_size` samples and returns `(morphological_age, phenotype)`.
    /// `morphological_age` is in years; `phenotype` is an Anny-ready
    /// [`PhenotypeValues`] with `gender`/`age`/`height`/`weight`/`muscle`/
    /// `proportions` set from the prior and the rest uniform-random in `[0, 1]`.
    pub fn sample<R: Rng>(
        &self,
        batch_size: usize,
        rng: &mut R,
    ) -> std::result::Result<(Tensor, PhenotypeValues), ShapeDistributionError> {
        // Morphological age ∈ Uniform(0, 90).
        let uniform_age = Uniform::new(0.0_f64, 90.0);
        let morph: Vec<f64> = (0..batch_size).map(|_| uniform_age.sample(rng)).collect();
        let morph_t =
            Tensor::from_vec(morph.clone(), batch_size, &self.device)?.to_dtype(self.dtype)?;
        let anny_age = self.mapping.morphological_to_anny_age(&morph_t)?;

        // Gender ∈ Uniform(0, 1).
        let uniform = Uniform::new(0.0_f64, 1.0);
        let gender_h: Vec<f64> = (0..batch_size).map(|_| uniform.sample(rng)).collect();
        let gender_t =
            Tensor::from_vec(gender_h.clone(), batch_size, &self.device)?.to_dtype(self.dtype)?;

        // Beta-distributed phenotype scalars, switching by gender.
        let height = self.sample_per_gender_beta(&anny_age, &gender_h, |g| &g.height, rng)?;
        let weight = self.sample_per_gender_beta(&anny_age, &gender_h, |g| &g.weight, rng)?;
        let muscle = self.sample_per_gender_beta(&anny_age, &gender_h, |g| &g.muscle, rng)?;
        let proportions =
            self.sample_per_gender_beta(&anny_age, &gender_h, |g| &g.proportions, rng)?;

        // Other phenotype labels: uniform [0, 1].
        let mut phen = PhenotypeValues::defaults(self.dtype, &self.device)?;
        // Fill all named scalars first with uniform draws so any label
        // present in `phenotype_labels` but not in the canonical set is
        // ignored — every label below is from the canonical 11.
        let make_uniform = |rng: &mut R| -> std::result::Result<Tensor, ShapeDistributionError> {
            let v: Vec<f64> = (0..batch_size).map(|_| uniform.sample(rng)).collect();
            Ok(Tensor::from_vec(v, batch_size, &self.device)?.to_dtype(self.dtype)?)
        };
        let known = ["height", "weight", "muscle", "proportions", "age", "gender"];
        for label in &self.phenotype_labels {
            if known.contains(&label.as_str()) {
                continue;
            }
            let t = make_uniform(rng)?;
            assign_phen(&mut phen, label, t);
        }
        // Pinned values from the prior.
        phen.gender = gender_t;
        phen.age = anny_age;
        phen.height = height;
        phen.weight = weight;
        phen.muscle = muscle;
        phen.proportions = proportions;
        Ok((morph_t, phen))
    }

    /// Helper: sample Beta(α, β) per batch element, picking the boys' or
    /// girls' parameters by `gender_h[i] <= 0.5`.
    fn sample_per_gender_beta<R: Rng, F>(
        &self,
        anny_age: &Tensor,
        gender_h: &[f64],
        select: F,
        rng: &mut R,
    ) -> std::result::Result<Tensor, ShapeDistributionError>
    where
        F: Fn(&GenderBetaSet) -> &ConditionalBetaDistribution,
    {
        let (alpha_b, beta_b) = select(&self.boys).distribution_params(anny_age)?;
        let (alpha_g, beta_g) = select(&self.girls).distribution_params(anny_age)?;
        let ab: Vec<f64> = alpha_b.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let bb: Vec<f64> = beta_b.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let ag: Vec<f64> = alpha_g.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let bg: Vec<f64> = beta_g.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let mut out = Vec::with_capacity(gender_h.len());
        for (i, g) in gender_h.iter().enumerate() {
            let (alpha, beta) = if *g <= 0.5 {
                (ab[i], bb[i])
            } else {
                (ag[i], bg[i])
            };
            // Beta requires α, β > 0. If either is non-positive (a malformed
            // calibration), fall back to a uniform sample to avoid panicking.
            let v = if alpha > 0.0 && beta > 0.0 {
                let dist = Beta::new(alpha, beta)?;
                dist.sample(rng)
            } else {
                Uniform::new(0.0_f64, 1.0).sample(rng)
            };
            out.push(v);
        }
        Ok(Tensor::from_vec(out, gender_h.len(), &self.device)?.to_dtype(self.dtype)?)
    }
}

fn assign_phen(p: &mut PhenotypeValues, label: &str, t: Tensor) {
    match label {
        "age" => p.age = t,
        "gender" => p.gender = t,
        "muscle" => p.muscle = t,
        "weight" => p.weight = t,
        "height" => p.height = t,
        "proportions" => p.proportions = t,
        "cupsize" => p.cupsize = t,
        "firmness" => p.firmness = t,
        "african" => p.african = t,
        "asian" => p.asian = t,
        "caucasian" => p.caucasian = t,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
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
    fn loads_boys_anchors() {
        let dist = SimpleShapeDistribution::load_default(
            &data_root(),
            vec!["age".into(), "height".into()],
            DType::F64,
            &Device::Cpu,
        )
        .expect("load");
        // From the Python inspection: anny_age_anchors = [0.0, 0.05, 0.215, 0.415, 0.67, 0.77, 0.83, 1.0]
        let anny_anchors: Vec<f64> = dist.mapping.anny_age_anchors.to_vec1().unwrap();
        let expected = [0.0, 0.05, 0.215, 0.415, 0.67, 0.77, 0.83, 1.0];
        assert_eq!(anny_anchors.len(), expected.len());
        for (g, e) in anny_anchors.iter().zip(expected.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn morphological_age_round_trip() {
        let dist = SimpleShapeDistribution::load_default(
            &data_root(),
            vec!["age".into()],
            DType::F64,
            &Device::Cpu,
        )
        .unwrap();
        // Pick a few morphological ages, map to Anny age, map back, expect round-trip.
        let ages: Vec<f64> = vec![0.0, 1.0, 4.0, 11.0, 18.0, 50.0, 80.0, 110.0];
        let morph = Tensor::from_vec(ages.clone(), ages.len(), &Device::Cpu).unwrap();
        let anny = dist.mapping.morphological_to_anny_age(&morph).unwrap();
        let back = dist.mapping.anny_to_morphological_age(&anny).unwrap();
        let back_v: Vec<f64> = back.to_vec1().unwrap();
        for (in_, out) in ages.iter().zip(back_v.iter()) {
            assert_relative_eq!(in_, out, epsilon = 1e-9);
        }
    }

    #[test]
    fn sample_outputs_are_in_range() {
        let dist = SimpleShapeDistribution::load_default(
            &data_root(),
            vec![
                "age".into(),
                "gender".into(),
                "height".into(),
                "weight".into(),
                "muscle".into(),
                "proportions".into(),
                "cupsize".into(),
                "firmness".into(),
                "african".into(),
                "asian".into(),
                "caucasian".into(),
            ],
            DType::F64,
            &Device::Cpu,
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(0xfade);
        let (morph, phen) = dist.sample(64, &mut rng).unwrap();
        let mv: Vec<f64> = morph.to_vec1().unwrap();
        for v in &mv {
            assert!((0.0..=90.0).contains(v));
        }
        for label in &[
            "height",
            "weight",
            "muscle",
            "proportions",
            "gender",
            "cupsize",
            "firmness",
            "african",
            "asian",
            "caucasian",
        ] {
            let t = match *label {
                "height" => &phen.height,
                "weight" => &phen.weight,
                "muscle" => &phen.muscle,
                "proportions" => &phen.proportions,
                "gender" => &phen.gender,
                "cupsize" => &phen.cupsize,
                "firmness" => &phen.firmness,
                "african" => &phen.african,
                "asian" => &phen.asian,
                "caucasian" => &phen.caucasian,
                _ => unreachable!(),
            };
            let v: Vec<f64> = t.to_vec1().unwrap();
            for x in &v {
                assert!((0.0..=1.0).contains(x), "{label} sample {x} outside [0, 1]");
            }
        }
    }
}
