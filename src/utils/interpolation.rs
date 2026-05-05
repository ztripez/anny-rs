//! Linear interpolation between monotonic anchor points.
//!
//! Port of `linear_interpolation_coefficients` in
//! `anny/src/anny/utils/interpolation.py`. Operates on contiguous batches via
//! candle tensors; produces a `[batch, n_anchors]` coefficient matrix where
//! exactly two adjacent columns per row are non-zero (or one, at the boundary).

use candle_core::{DType, Tensor};

/// Returns the per-anchor weights used to express each `value` as a convex
/// combination of two adjacent `anchors`. With `extrapolate = false` the
/// coefficients are clamped to `[0, 1]`.
///
/// Shapes: `value` is `[B]`, `anchors` is `[N]` (monotonically increasing),
/// output is `[B, N]`. The returned tensor has the dtype of `anchors`.
pub fn linear_interpolation_coefficients(
    value: &Tensor,
    anchors: &Tensor,
    extrapolate: bool,
) -> candle_core::Result<Tensor> {
    let device = anchors.device();
    let dtype = anchors.dtype();
    let batch_size = value.dim(0)?;
    let n = anchors.dim(0)?;

    // searchsorted side="left" — find the first anchor >= value. Candle has no
    // built-in searchsorted, so do it manually on host f64 for correctness;
    // the anchor count is tiny (≤ ~10) and this isn't on a hot path.
    let value_h: Vec<f64> = value.to_dtype(DType::F64)?.to_vec1()?;
    let anchors_h: Vec<f64> = anchors.to_dtype(DType::F64)?.to_vec1()?;
    let mut indices: Vec<i64> = Vec::with_capacity(batch_size);
    for v in &value_h {
        let mut lo = 0usize;
        let mut hi = anchors_h.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if anchors_h[mid] < *v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Clamp to [1, n-1] to mirror the Python clamp.
        let clamped = lo.clamp(1, n - 1);
        indices.push(clamped as i64);
    }

    let lower: Vec<f64> = indices
        .iter()
        .map(|&i| anchors_h[(i - 1) as usize])
        .collect();
    let upper: Vec<f64> = indices.iter().map(|&i| anchors_h[i as usize]).collect();

    let mut alpha: Vec<f64> = value_h
        .iter()
        .zip(lower.iter().zip(upper.iter()))
        .map(|(v, (lo, hi))| (v - lo) / (hi - lo))
        .collect();
    if !extrapolate {
        for a in alpha.iter_mut() {
            *a = a.clamp(0.0, 1.0);
        }
    }

    // Scatter (1 - alpha) to column idx-1 and alpha to column idx for each row.
    let mut flat = vec![0.0_f64; batch_size * n];
    for (b, (idx, a)) in indices.iter().zip(alpha.iter()).enumerate() {
        let i = *idx as usize;
        flat[b * n + (i - 1)] = 1.0 - a;
        flat[b * n + i] = *a;
    }

    let weights = Tensor::from_vec(flat, (batch_size, n), device)?.to_dtype(dtype)?;
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::Device;

    #[test]
    fn matches_python_example() {
        // Mirrors the __main__ block in interpolation.py, with extrapolate=true.
        let device = Device::Cpu;
        let anchors = Tensor::from_vec(vec![0.0_f64, 0.5, 1.0], 3, &device).unwrap();
        let xs = Tensor::from_vec(vec![-0.1_f64, 0.25, 0.5, 0.75, 1.1], 5, &device).unwrap();
        let coeffs = linear_interpolation_coefficients(&xs, &anchors, true).unwrap();
        let got: Vec<Vec<f64>> = coeffs.to_vec2().unwrap();

        // Hand-checked rows: at x=-0.1 with extrapolate→ idx clamped to 1, lower=0, upper=0.5,
        // alpha = -0.1/0.5 = -0.2 → row [1.2, -0.2, 0.0]
        let expected = [
            [1.2, -0.2, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, -0.2, 1.2],
        ];
        for (row, exp) in got.iter().zip(expected.iter()) {
            for (g, e) in row.iter().zip(exp.iter()) {
                assert_relative_eq!(g, e, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn no_extrapolate_clamps() {
        let device = Device::Cpu;
        let anchors = Tensor::from_vec(vec![0.0_f64, 0.5, 1.0], 3, &device).unwrap();
        let xs = Tensor::from_vec(vec![-0.5_f64, 1.5], 2, &device).unwrap();
        let coeffs = linear_interpolation_coefficients(&xs, &anchors, false).unwrap();
        let got: Vec<Vec<f64>> = coeffs.to_vec2().unwrap();
        assert_relative_eq!(got[0][0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(got[0][1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(got[1][1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(got[1][2], 1.0, epsilon = 1e-12);
    }
}
