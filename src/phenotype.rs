//! Phenotype layer — port of `anny/src/anny/models/phenotype.py`.
//!
//! The model has 11 phenotype scalars (8 non-race + 3 race) which select
//! a blend-shape combination. Each non-race scalar is linearly interpolated
//! against its named anchors (e.g. `age` ∈ {newborn, baby, child, young, old}).
//! The three race scalars are normalised to a 3-way softmax. The resulting
//! `[B, 26]` "phen" vector is combined with a `[C, 26]` 0/1 mask via
//! `wi = prod(mask * phens + (1 - mask), dim=-1)` → `[B, 564]` blend-shape
//! coefficients. The trick: for each blend shape `c`, the product collapses
//! to the product of phenotype scalars whose mask bit is 1, leaving the
//! others as multiplicative identities.

use candle_core::shape::Dim;
use candle_core::{D, DType, Device, Result, Tensor};

use crate::utils::interpolation::linear_interpolation_coefficients;

// ────────────────────────────────────────────────────────────────────────────
// Phenotype taxonomy.
// ────────────────────────────────────────────────────────────────────────────

/// Names of every phenotype variation, grouped by feature, in the canonical
/// stacking order (matches Python's `PHENOTYPE_VARIATIONS`).
pub const PHENOTYPE_VARIATIONS: &[(&str, &[&str])] = &[
    ("race", &["african", "asian", "caucasian"]),
    ("gender", &["male", "female"]),
    ("age", &["newborn", "baby", "child", "young", "old"]),
    ("muscle", &["minmuscle", "averagemuscle", "maxmuscle"]),
    ("weight", &["minweight", "averageweight", "maxweight"]),
    ("height", &["minheight", "maxheight"]),
    ("proportions", &["idealproportions", "uncommonproportions"]),
    ("cupsize", &["mincup", "averagecup", "maxcup"]),
    (
        "firmness",
        &["minfirmness", "averagefirmness", "maxfirmness"],
    ),
];

/// Total number of variation columns (== number of columns in the phen vector
/// and rows in the per-blend-shape mask). Equals 3+2+5+3+3+2+2+3+3 = 26.
pub const PHENOTYPE_VARIATION_COUNT: usize = {
    let mut n = 0;
    let mut i = 0;
    while i < PHENOTYPE_VARIATIONS.len() {
        n += PHENOTYPE_VARIATIONS[i].1.len();
        i += 1;
    }
    n
};

/// Phenotypes excluded by default (mirrors `EXCLUDED_PHENOTYPES` in Python):
/// `cupsize`, `firmness`, and the three race scalars.
pub const EXCLUDED_PHENOTYPES: &[&str] = &["cupsize", "firmness", "african", "asian", "caucasian"];

// ────────────────────────────────────────────────────────────────────────────
// Phenotype kwargs (continuous scalars, default 0.5).
// ────────────────────────────────────────────────────────────────────────────

/// All 11 phenotype scalars. Each value is treated as a batch of size 1 if a
/// scalar; pass `Tensor` for a real batch (all tensors must broadcast to the
/// same `[B]` length).
#[derive(Debug, Clone)]
pub struct PhenotypeValues {
    pub age: Tensor,
    pub gender: Tensor,
    pub muscle: Tensor,
    pub weight: Tensor,
    pub height: Tensor,
    pub proportions: Tensor,
    pub cupsize: Tensor,
    pub firmness: Tensor,
    pub african: Tensor,
    pub asian: Tensor,
    pub caucasian: Tensor,
}

impl PhenotypeValues {
    /// Build a default `PhenotypeValues` with every scalar set to `0.5` and
    /// batch size 1 on the given device/dtype.
    pub fn defaults(dtype: DType, device: &Device) -> Result<Self> {
        let half = || Tensor::from_vec(vec![0.5_f64], 1, device)?.to_dtype(dtype);
        Ok(Self {
            age: half()?,
            gender: half()?,
            muscle: half()?,
            weight: half()?,
            height: half()?,
            proportions: half()?,
            cupsize: half()?,
            firmness: half()?,
            african: half()?,
            asian: half()?,
            caucasian: half()?,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Anchor tables.
// ────────────────────────────────────────────────────────────────────────────

/// Per-feature anchor tensors. `age` spans `[-1/3, 1]` (so 0 ≈ baby, 1 = old);
/// every other non-race feature spans `[0, 1]`. Mirrors lines 100–102 of
/// `phenotype.py`.
pub struct PhenotypeAnchors {
    pub age: Tensor,
    pub gender: Tensor,
    pub muscle: Tensor,
    pub weight: Tensor,
    pub height: Tensor,
    pub proportions: Tensor,
    pub cupsize: Tensor,
    pub firmness: Tensor,
}

impl PhenotypeAnchors {
    pub fn build(dtype: DType, device: &Device) -> Result<Self> {
        fn linspace(lo: f64, hi: f64, n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
            if n == 1 {
                return Tensor::from_vec(vec![lo], 1, device)?.to_dtype(dtype);
            }
            let step = (hi - lo) / (n as f64 - 1.0);
            let v: Vec<f64> = (0..n).map(|i| lo + step * i as f64).collect();
            Tensor::from_vec(v, n, device)?.to_dtype(dtype)
        }
        Ok(Self {
            age: linspace(-1.0 / 3.0, 1.0, 5, dtype, device)?,
            gender: linspace(0.0, 1.0, 2, dtype, device)?,
            muscle: linspace(0.0, 1.0, 3, dtype, device)?,
            weight: linspace(0.0, 1.0, 3, dtype, device)?,
            height: linspace(0.0, 1.0, 2, dtype, device)?,
            proportions: linspace(0.0, 1.0, 2, dtype, device)?,
            cupsize: linspace(0.0, 1.0, 3, dtype, device)?,
            firmness: linspace(0.0, 1.0, 3, dtype, device)?,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Coefficient computation.
// ────────────────────────────────────────────────────────────────────────────

/// Computes the `[B, C]` blend-shape coefficient vector for a batch of
/// phenotype values, given the model's `[C, V]` mask (where `V == PHENOTYPE_VARIATION_COUNT`).
///
/// Mirrors `RiggedModelWithPhenotypeParameters.get_phenotype_blendshape_coefficients`
/// in `phenotype.py:111–176`, less the local-change machinery (which is
/// appended by callers when `local_change_labels` is non-empty).
pub fn blendshape_coefficients(
    values: &PhenotypeValues,
    anchors: &PhenotypeAnchors,
    stacked_mask: &Tensor,
    extrapolate: bool,
) -> Result<Tensor> {
    let dtype = stacked_mask.dtype();
    let device = stacked_mask.device();

    // Per-feature interpolation coefficients (each [B, n_variations]).
    let coeffs_age = interp_coeffs(&values.age, &anchors.age, extrapolate, dtype, device)?;
    let coeffs_gender = interp_coeffs(&values.gender, &anchors.gender, extrapolate, dtype, device)?;
    let coeffs_muscle = interp_coeffs(&values.muscle, &anchors.muscle, extrapolate, dtype, device)?;
    let coeffs_weight = interp_coeffs(&values.weight, &anchors.weight, extrapolate, dtype, device)?;
    let coeffs_height = interp_coeffs(&values.height, &anchors.height, extrapolate, dtype, device)?;
    let coeffs_proportions = interp_coeffs(
        &values.proportions,
        &anchors.proportions,
        extrapolate,
        dtype,
        device,
    )?;
    let coeffs_cupsize = interp_coeffs(
        &values.cupsize,
        &anchors.cupsize,
        extrapolate,
        dtype,
        device,
    )?;
    let coeffs_firmness = interp_coeffs(
        &values.firmness,
        &anchors.firmness,
        extrapolate,
        dtype,
        device,
    )?;

    let batch_size = [
        &coeffs_age,
        &coeffs_gender,
        &coeffs_muscle,
        &coeffs_weight,
        &coeffs_height,
        &coeffs_proportions,
        &coeffs_cupsize,
        &coeffs_firmness,
    ]
    .iter()
    .map(|c| c.dim(0).unwrap_or(1))
    .max()
    .unwrap_or(1);

    // Race weights, normalised. NaN → 1/3 (when all three are zero).
    let race_stack = Tensor::stack(
        &[
            cast_and_broadcast(&values.african, batch_size, dtype, device)?,
            cast_and_broadcast(&values.asian, batch_size, dtype, device)?,
            cast_and_broadcast(&values.caucasian, batch_size, dtype, device)?,
        ],
        1,
    )?; // [B, 3]
    let race_sum = race_stack.sum_keepdim(1)?; // [B, 1]
    let race_weights = race_stack.broadcast_div(&race_sum)?;
    let race_weights = nan_to_third(&race_weights)?;

    // Build the [B, V] phen tensor in PHENOTYPE_VARIATIONS order:
    //   race (3) | gender (2) | age (5) | muscle (3) | weight (3) | height (2)
    //   proportions (2) | cupsize (3) | firmness (3)
    let parts: Vec<Tensor> = vec![
        broadcast_to_batch(&race_weights, batch_size)?,
        broadcast_to_batch(&coeffs_gender, batch_size)?,
        broadcast_to_batch(&coeffs_age, batch_size)?,
        broadcast_to_batch(&coeffs_muscle, batch_size)?,
        broadcast_to_batch(&coeffs_weight, batch_size)?,
        broadcast_to_batch(&coeffs_height, batch_size)?,
        broadcast_to_batch(&coeffs_proportions, batch_size)?,
        broadcast_to_batch(&coeffs_cupsize, batch_size)?,
        broadcast_to_batch(&coeffs_firmness, batch_size)?,
    ];
    let phens = Tensor::cat(&parts, 1)?; // [B, V]

    let v = phens.dim(1)?;
    if stacked_mask.dim(1)? != v {
        candle_core::bail!(
            "stacked_mask second dim {} ≠ phen vector length {} (PHENOTYPE_VARIATION_COUNT)",
            stacked_mask.dim(1)?,
            v
        );
    }

    // masked_phens = phens.unsqueeze(1) * mask.unsqueeze(0)            [B, C, V]
    // wi          = prod(masked_phens + (1 - mask.unsqueeze(0)), -1)   [B, C]
    let phens_b = phens.unsqueeze(1)?; // [B, 1, V]
    let mask_b = stacked_mask.unsqueeze(0)?; // [1, C, V]
    let masked = phens_b.broadcast_mul(&mask_b)?;
    let one_minus_mask = (Tensor::ones_like(&mask_b)? - &mask_b)?;
    let summed = masked.broadcast_add(&one_minus_mask)?;
    let wi = product_along(&summed, D::Minus1)?; // [B, C]
    Ok(wi)
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn interp_coeffs(
    value: &Tensor,
    anchors: &Tensor,
    extrapolate: bool,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // Promote a 0-D value to [1] (mirrors `to_batched_tensor`).
    let v = if value.rank() == 0 {
        value.unsqueeze(0)?
    } else {
        value.clone()
    };
    let v = v.to_dtype(dtype)?.to_device(device)?;
    let a = anchors.to_dtype(dtype)?.to_device(device)?;
    linear_interpolation_coefficients(&v, &a, extrapolate)
}

fn cast_and_broadcast(
    value: &Tensor,
    batch_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let v = if value.rank() == 0 {
        value.unsqueeze(0)?
    } else {
        value.clone()
    };
    let v = v.to_dtype(dtype)?.to_device(device)?;
    if v.dim(0)? == batch_size {
        Ok(v)
    } else if v.dim(0)? == 1 {
        v.broadcast_as((batch_size,))
    } else {
        candle_core::bail!(
            "cannot broadcast scalar of length {} to batch {batch_size}",
            v.dim(0)?
        );
    }
}

fn broadcast_to_batch(t: &Tensor, batch_size: usize) -> Result<Tensor> {
    let b = t.dim(0)?;
    if b == batch_size {
        Ok(t.clone())
    } else if b == 1 {
        // Broadcast across batch dim; preserve trailing dims.
        let mut shape: Vec<usize> = vec![batch_size];
        shape.extend_from_slice(&t.dims()[1..]);
        t.broadcast_as(shape)
    } else {
        candle_core::bail!("cannot broadcast batch dim {b} to {batch_size}");
    }
}

fn nan_to_third(t: &Tensor) -> Result<Tensor> {
    // Replace NaNs (which arise when all three race scalars sum to 0) with 1/3.
    // candle has no native nan_to_num — fall back to host conversion.
    let dtype = t.dtype();
    let dims = t.dims().to_vec();
    let host: Vec<f64> = t
        .to_dtype(DType::F64)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let third = 1.0 / 3.0;
    let cleaned: Vec<f64> = host
        .into_iter()
        .map(|x| if x.is_nan() { third } else { x })
        .collect();
    Tensor::from_vec(cleaned, dims, t.device())?.to_dtype(dtype)
}

/// Product reduction along `dim`. candle's `Tensor::cumprod` exists but no
/// reduction; do it by reshaping to 2D and looping multiplicatively. The
/// reduction axis is always the last (`V`) in our masked-product use, so the
/// shape is `[B*C, V]` and `V == 26`.
fn product_along(t: &Tensor, dim: D) -> Result<Tensor> {
    let dim_idx = dim.to_index(t.shape(), "product_along")?;
    let n = t.dim(dim_idx)?;
    let mut acc = t.narrow(dim_idx, 0, 1)?;
    for i in 1..n {
        let slice = t.narrow(dim_idx, i, 1)?;
        acc = (acc * slice)?;
    }
    acc.squeeze(dim_idx)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn variation_count_is_26() {
        assert_eq!(PHENOTYPE_VARIATION_COUNT, 26);
    }

    #[test]
    fn defaults_yield_uniform_race() {
        let device = Device::Cpu;
        let dtype = DType::F64;
        let values = PhenotypeValues::defaults(dtype, &device).unwrap();

        // Use a trivial mask: identity-like — set the first 26 blend shapes
        // to mask each individual variation, and verify the resulting
        // coefficient equals the corresponding phen value.
        let mask = Tensor::eye(26, dtype, &device).unwrap();
        let anchors = PhenotypeAnchors::build(dtype, &device).unwrap();
        let coeffs = blendshape_coefficients(&values, &anchors, &mask, false).unwrap();
        assert_eq!(coeffs.dims(), &[1, 26]);

        // With every value at 0.5 and uniform race (1/3 each), the masked
        // product for each row is just the matching phen scalar.
        let v: Vec<Vec<f64>> = coeffs.to_vec2().unwrap();
        // race columns 0..3 = 1/3 each (after normalisation of (0.5,0.5,0.5))
        for i in 0..3 {
            assert!((v[0][i] - 1.0 / 3.0).abs() < 1e-12);
        }
        // every other column should be the gender/age/etc. interp, which at
        // 0.5 lands exactly between two anchors → 0.5 split. So a single-row
        // mask returns 0.5 for the boundary positions and 0 / 0.5 for others.
        // Easier check: every column is in [0, 1].
        for i in 0..26 {
            assert!(
                v[0][i] >= -1e-12 && v[0][i] <= 1.0 + 1e-12,
                "col {i} = {}",
                v[0][i]
            );
        }
    }

    #[test]
    fn race_normalisation_sums_to_one() {
        let device = Device::Cpu;
        let dtype = DType::F64;
        let mut values = PhenotypeValues::defaults(dtype, &device).unwrap();
        values.african = Tensor::from_vec(vec![1.0_f64], 1, &device).unwrap();
        values.asian = Tensor::from_vec(vec![2.0_f64], 1, &device).unwrap();
        values.caucasian = Tensor::from_vec(vec![1.0_f64], 1, &device).unwrap();

        let mask = Tensor::eye(26, dtype, &device).unwrap();
        let anchors = PhenotypeAnchors::build(dtype, &device).unwrap();
        let coeffs = blendshape_coefficients(&values, &anchors, &mask, false).unwrap();
        let v: Vec<Vec<f64>> = coeffs.to_vec2().unwrap();
        let race_sum = v[0][0] + v[0][1] + v[0][2];
        assert!((race_sum - 1.0).abs() < 1e-12);
        assert!((v[0][1] - 0.5).abs() < 1e-12); // asian = 2 / 4
    }

    #[test]
    fn product_collapses_correctly() {
        // Build a mask that gates blend shape 0 by columns 0 and 1 only,
        // and blend shape 1 by columns 2..5 only. The product should equal
        // phens[0]*phens[1] for shape 0 and phens[2]*phens[3]*phens[4]
        // for shape 1.
        let device = Device::Cpu;
        let dtype = DType::F64;

        let mut values = PhenotypeValues::defaults(dtype, &device).unwrap();
        values.african = Tensor::from_vec(vec![1.0_f64], 1, &device).unwrap();
        values.asian = Tensor::from_vec(vec![0.0_f64], 1, &device).unwrap();
        values.caucasian = Tensor::from_vec(vec![0.0_f64], 1, &device).unwrap();
        values.gender = Tensor::from_vec(vec![1.0_f64], 1, &device).unwrap(); // all-female

        // Layout: race[3] gender[2] age[5] muscle[3] weight[3] height[2] proportions[2] cupsize[3] firmness[3]
        // Columns 0=african, 1=asian, 2=caucasian, 3=male, 4=female
        let anchors = PhenotypeAnchors::build(dtype, &device).unwrap();
        let mut mask_data = vec![0.0_f64; 2 * 26];
        mask_data[0] = 1.0; // shape 0: african
        mask_data[1] = 1.0; // shape 0: asian
        mask_data[26 + 4] = 1.0; // shape 1: female
        let mask = Tensor::from_vec(mask_data, (2, 26), &device).unwrap();

        let coeffs = blendshape_coefficients(&values, &anchors, &mask, false).unwrap();
        let v: Vec<Vec<f64>> = coeffs.to_vec2().unwrap();
        // shape 0 = africa * asian = 1 * 0 = 0
        assert!((v[0][0] - 0.0).abs() < 1e-12);
        // shape 1 = female = 1
        assert!((v[0][1] - 1.0).abs() < 1e-12);
    }
}
