//! Replacements for the [`roma`](https://github.com/naver/roma) functions Anny
//! actually uses. Quaternions follow roma's **XYZW** convention (vector part
//! first, scalar last), and rigid transforms are represented as homogeneous
//! `[..., 4, 4]` matrices — accessors `linear()`/`translation()` slice them
//! directly, avoiding a separate `Rigid` struct for now.
//!
//! All public functions accept arbitrary leading batch dimensions: a `[..., 3]`
//! rotation vector becomes `[..., 3, 3]` rotation matrices, etc.

use candle_core::{D, Result, Tensor};

// ────────────────────────────────────────────────────────────────────────────
// Quaternion algebra (XYZW convention).
// ────────────────────────────────────────────────────────────────────────────

/// Hamilton product of two quaternions stored in XYZW order.
///
/// `(q1 * q2)` rotates a vector first by `q2` then by `q1`.
pub fn quat_product(q1: &Tensor, q2: &Tensor) -> Result<Tensor> {
    let last = q1.rank() - 1;
    let x1 = q1.narrow(last, 0, 1)?;
    let y1 = q1.narrow(last, 1, 1)?;
    let z1 = q1.narrow(last, 2, 1)?;
    let w1 = q1.narrow(last, 3, 1)?;

    let x2 = q2.narrow(last, 0, 1)?;
    let y2 = q2.narrow(last, 1, 1)?;
    let z2 = q2.narrow(last, 2, 1)?;
    let w2 = q2.narrow(last, 3, 1)?;

    let x = (w1.mul(&x2)? + x1.mul(&w2)? + y1.mul(&z2)? - z1.mul(&y2)?)?;
    let y = (w1.mul(&y2)? - x1.mul(&z2)? + y1.mul(&w2)? + z1.mul(&x2)?)?;
    let z = (w1.mul(&z2)? + x1.mul(&y2)? - y1.mul(&x2)? + z1.mul(&w2)?)?;
    let w = (w1.mul(&w2)? - x1.mul(&x2)? - y1.mul(&y2)? - z1.mul(&z2)?)?;

    Tensor::cat(&[&x, &y, &z, &w], last)
}

/// Conjugate `(x, y, z, w) -> (-x, -y, -z, w)`. For unit quaternions this is
/// the inverse rotation.
pub fn quat_conjugation(q: &Tensor) -> Result<Tensor> {
    let last = q.rank() - 1;
    let v = q.narrow(last, 0, 3)?.neg()?;
    let w = q.narrow(last, 3, 1)?;
    Tensor::cat(&[&v, &w], last)
}

/// Apply a unit-quaternion rotation to 3-vectors using the sandwich product
/// `q * (v, 0) * q⁻¹`. Inputs must broadcast on the leading dims:
/// `q: [..., 4]`, `v: [..., 3]` → `[..., 3]`.
pub fn quat_action(q: &Tensor, v: &Tensor) -> Result<Tensor> {
    let last = v.rank() - 1;
    // Treat v as a pure quaternion (x, y, z, 0).
    let zero = Tensor::zeros_like(&v.narrow(last, 0, 1)?)?;
    let v_quat = Tensor::cat(&[v, &zero], last)?;
    let qv = quat_product(q, &v_quat)?;
    let result = quat_product(&qv, &quat_conjugation(q)?)?;
    result.narrow(last, 0, 3)
}

// ────────────────────────────────────────────────────────────────────────────
// Rotation conversions.
// ────────────────────────────────────────────────────────────────────────────

/// Rodrigues' formula: rotation vector (axis × angle in radians) → 3×3 matrix.
///
/// Input: `[..., 3]`. Output: `[..., 3, 3]`. Uses an angle threshold to keep
/// the small-angle case stable (`I + [k]_×` Taylor approximation).
pub fn rotvec_to_rotmat(rotvec: &Tensor) -> Result<Tensor> {
    let last = rotvec.rank() - 1;
    let theta = rotvec.sqr()?.sum_keepdim(last)?.sqrt()?; // [..., 1]

    // Avoid division by zero: where theta ≈ 0 we'll fall back to 1.
    let safe_theta = theta.broadcast_maximum(&Tensor::new(1e-30_f64, theta.device())?.to_dtype(theta.dtype())?)?;
    let unit = rotvec.broadcast_div(&safe_theta)?; // [..., 3]

    let cos_t = theta.cos()?;
    let sin_t = theta.sin()?;
    let one_minus_cos = (Tensor::ones_like(&cos_t)? - &cos_t)?;

    let kx = unit.narrow(last, 0, 1)?;
    let ky = unit.narrow(last, 1, 1)?;
    let kz = unit.narrow(last, 2, 1)?;

    // Skew-symmetric pieces, broadcast-compatible.
    let zero = Tensor::zeros_like(&kx)?;
    let neg_kz = kz.neg()?;
    let neg_ky = ky.neg()?;
    let neg_kx = kx.neg()?;

    let row0 = Tensor::cat(&[&zero, &neg_kz, &ky], last)?;
    let row1 = Tensor::cat(&[&kz, &zero, &neg_kx], last)?;
    let row2 = Tensor::cat(&[&neg_ky, &kx, &zero], last)?;
    let k_cross = Tensor::stack(&[&row0, &row1, &row2], last)?; // [..., 3, 3]

    let outer = unit
        .unsqueeze(last + 1)?
        .matmul(&unit.unsqueeze(last)?)?; // [..., 3, 3]

    let eye = identity_3x3_like(rotvec)?;

    // R = cos·I + sin·[k]_× + (1-cos)·k k^T
    let cos_term = eye.broadcast_mul(&cos_t.unsqueeze(last + 1)?)?;
    let sin_term = k_cross.broadcast_mul(&sin_t.unsqueeze(last + 1)?)?;
    let outer_term = outer.broadcast_mul(&one_minus_cos.unsqueeze(last + 1)?)?;

    cos_term.add(&sin_term)?.add(&outer_term)
}

/// Convert a 3×3 rotation matrix to a unit quaternion (XYZW), using Shepperd's
/// method (numerically stable across all four cases).
///
/// Input: `[..., 3, 3]`. Output: `[..., 4]`. We do this on the host because
/// the per-element branching is awkward to vectorise in candle and it isn't
/// on the hot forward path.
pub fn rotmat_to_unitquat(rotmat: &Tensor) -> Result<Tensor> {
    let dtype = rotmat.dtype();
    let device = rotmat.device().clone();
    let dims = rotmat.dims().to_vec();
    let n_last_two = dims.len() - 2;
    let batch_dims = &dims[..n_last_two];
    let batch: usize = batch_dims.iter().product();

    let flat = rotmat
        .to_dtype(candle_core::DType::F64)?
        .reshape((batch, 3, 3))?;
    let cpu = flat.to_device(&candle_core::Device::Cpu)?;
    let host: Vec<Vec<Vec<f64>>> = cpu.to_vec3()?;

    let mut out = Vec::with_capacity(batch * 4);
    for m in &host {
        let r00 = m[0][0]; let r01 = m[0][1]; let r02 = m[0][2];
        let r10 = m[1][0]; let r11 = m[1][1]; let r12 = m[1][2];
        let r20 = m[2][0]; let r21 = m[2][1]; let r22 = m[2][2];
        let trace = r00 + r11 + r22;
        let (x, y, z, w) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            (
                (r21 - r12) / s,
                (r02 - r20) / s,
                (r10 - r01) / s,
                0.25 * s,
            )
        } else if r00 > r11 && r00 > r22 {
            let s = (1.0 + r00 - r11 - r22).sqrt() * 2.0;
            (
                0.25 * s,
                (r01 + r10) / s,
                (r02 + r20) / s,
                (r21 - r12) / s,
            )
        } else if r11 > r22 {
            let s = (1.0 + r11 - r00 - r22).sqrt() * 2.0;
            (
                (r01 + r10) / s,
                0.25 * s,
                (r12 + r21) / s,
                (r02 - r20) / s,
            )
        } else {
            let s = (1.0 + r22 - r00 - r11).sqrt() * 2.0;
            (
                (r02 + r20) / s,
                (r12 + r21) / s,
                0.25 * s,
                (r10 - r01) / s,
            )
        };
        out.extend([x, y, z, w]);
    }

    let mut shape = batch_dims.to_vec();
    shape.push(4);
    Tensor::from_vec(out, shape, &device)?.to_dtype(dtype)
}

/// Convert single-axis Euler angles to 3×3 rotation matrices.
///
/// `axis` is `'x' | 'y' | 'z'` (case-insensitive). `angles` has shape `[...]`
/// (any rank); output is `[..., 3, 3]`. When `degrees == true` the input is
/// converted to radians first.
pub fn euler_to_rotmat(axis: char, angles: &Tensor, degrees: bool) -> Result<Tensor> {
    let radians = if degrees {
        let factor = std::f64::consts::PI / 180.0;
        angles.affine(factor, 0.0)?
    } else {
        angles.clone()
    };
    let c = radians.cos()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // [..., 1, 1]
    let s = radians.sin()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;

    // Build the matrix from constant 0/1 plus c, s, -s placements.
    let zero = Tensor::zeros_like(&c)?;
    let one = Tensor::ones_like(&c)?;
    let neg_s = s.neg()?;

    let m = match axis.to_ascii_lowercase() {
        'x' => stack_3x3(&one, &zero, &zero, &zero, &c, &neg_s, &zero, &s, &c)?,
        'y' => stack_3x3(&c, &zero, &s, &zero, &one, &zero, &neg_s, &zero, &c)?,
        'z' => stack_3x3(&c, &neg_s, &zero, &s, &c, &zero, &zero, &zero, &one)?,
        other => candle_core::bail!("euler_to_rotmat: axis must be x/y/z, got {other:?}"),
    };
    Ok(m)
}

// ────────────────────────────────────────────────────────────────────────────
// Rigid (homogeneous 4×4) transforms.
// ────────────────────────────────────────────────────────────────────────────

/// Pack a `[..., 3, 3]` rotation and `[..., 3]` translation into a `[..., 4, 4]`
/// homogeneous matrix.
pub fn rigid_to_homogeneous(linear: &Tensor, translation: &Tensor) -> Result<Tensor> {
    let dims = linear.dims();
    let n = dims.len();
    if n < 2 || dims[n - 2] != 3 || dims[n - 1] != 3 {
        candle_core::bail!(
            "rigid_to_homogeneous: linear must end in [3, 3], got {:?}",
            dims
        );
    }
    let last = n - 1; // axis for `cat`-as-rows
    // Build the top three rows: linear concatenated with translation column.
    let t_col = translation.unsqueeze(D::Minus1)?; // [..., 3, 1]
    let top = Tensor::cat(&[linear, &t_col], last)?; // [..., 3, 4]

    // Bottom row = [0, 0, 0, 1]
    let mut bottom_shape: Vec<usize> = dims.iter().take(n - 2).copied().collect();
    bottom_shape.push(1);
    bottom_shape.push(4);
    let bottom_data = vec![0.0_f64, 0.0, 0.0, 1.0];
    let bottom_unbroadcast = Tensor::from_vec(bottom_data, (1, 4), linear.device())?
        .to_dtype(linear.dtype())?;
    let bottom = bottom_unbroadcast.broadcast_as(bottom_shape)?.contiguous()?;
    Tensor::cat(&[&top, &bottom], last - 1)?.contiguous()
}

/// Pull `(linear, translation)` out of a `[..., 4, 4]` homogeneous matrix.
/// Bottom row is ignored — mirroring roma's behaviour.
pub fn rigid_from_homogeneous(h: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = h.dims();
    let n = dims.len();
    if n < 2 || dims[n - 2] != 4 || dims[n - 1] != 4 {
        candle_core::bail!(
            "rigid_from_homogeneous: tensor must end in [4, 4], got {:?}",
            dims
        );
    }
    let linear = h.narrow(n - 2, 0, 3)?.narrow(n - 1, 0, 3)?;
    let translation = h.narrow(n - 2, 0, 3)?.narrow(n - 1, 3, 1)?.squeeze(n - 1)?;
    Ok((linear, translation))
}

/// Inverse of a rigid transform stored as homogeneous `[..., 4, 4]`.
/// `inv(R, t) = (Rᵀ, -Rᵀ t)`.
pub fn rigid_inverse_homogeneous(h: &Tensor) -> Result<Tensor> {
    let (r, t) = rigid_from_homogeneous(h)?;
    let r_t = r.transpose(D::Minus1, D::Minus2)?.contiguous()?;
    // -Rᵀ t : matmul Rᵀ with t as a column vector.
    let t_col = t.unsqueeze(D::Minus1)?.contiguous()?;
    let new_t = r_t.matmul(&t_col)?.squeeze(D::Minus1)?.neg()?;
    rigid_to_homogeneous(&r_t, &new_t)
}

// ────────────────────────────────────────────────────────────────────────────
// Weighted rigid points registration (SVD-based).
// ────────────────────────────────────────────────────────────────────────────

/// Solves `min Σ wᵢ ‖R xᵢ + t − yᵢ‖²` over rotations `R ∈ SO(3)` and
/// translations `t ∈ ℝ³` for every batch element.
///
/// Inputs:
///   - `src`, `tgt`: `[B, N, 3]` source and target point clouds.
///   - `weights`: optional `[B, N]` non-negative weights. `None` means uniform.
///
/// Returns:
///   - `R`: `[B, 3, 3]`
///   - `t`: `[B, 3]`
///
/// Implementation: per-batch weighted Kabsch. SVD is done with `nalgebra` on
/// the host because candle's CPU SVD is not exposed for arbitrary 3×3 inputs
/// in the public API at 0.10, and the per-batch SVD is on a tiny matrix.
pub fn rigid_points_registration(
    src: &Tensor,
    tgt: &Tensor,
    weights: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let dtype = src.dtype();
    let device = src.device().clone();

    let src_h = src.to_dtype(candle_core::DType::F64)?.to_device(&candle_core::Device::Cpu)?;
    let tgt_h = tgt.to_dtype(candle_core::DType::F64)?.to_device(&candle_core::Device::Cpu)?;

    let dims = src_h.dims();
    if dims.len() != 3 || dims[2] != 3 {
        candle_core::bail!("rigid_points_registration: src must be [B, N, 3], got {dims:?}");
    }
    let (b, n) = (dims[0], dims[1]);
    let src_v: Vec<Vec<Vec<f64>>> = src_h.to_vec3()?;
    let tgt_v: Vec<Vec<Vec<f64>>> = tgt_h.to_vec3()?;
    let weights_v: Option<Vec<Vec<f64>>> = match weights {
        Some(w) => Some(
            w.to_dtype(candle_core::DType::F64)?
                .to_device(&candle_core::Device::Cpu)?
                .to_vec2()?,
        ),
        None => None,
    };

    let mut r_out = Vec::with_capacity(b * 9);
    let mut t_out = Vec::with_capacity(b * 3);

    for bi in 0..b {
        let w_row: Vec<f64> = match weights_v.as_ref() {
            Some(w) => w[bi].clone(),
            None => vec![1.0; n],
        };
        let total_w: f64 = w_row.iter().sum();
        let inv_total = if total_w > 0.0 { 1.0 / total_w } else { 1.0 };

        // Weighted centroids.
        let mut cs = [0.0_f64; 3];
        let mut ct = [0.0_f64; 3];
        for i in 0..n {
            for j in 0..3 {
                cs[j] += w_row[i] * src_v[bi][i][j];
                ct[j] += w_row[i] * tgt_v[bi][i][j];
            }
        }
        for j in 0..3 {
            cs[j] *= inv_total;
            ct[j] *= inv_total;
        }

        // Cross-covariance H = Σ wᵢ (xᵢ − cs)(yᵢ − ct)ᵀ.
        let mut h = [[0.0_f64; 3]; 3];
        for i in 0..n {
            let dx = [
                src_v[bi][i][0] - cs[0],
                src_v[bi][i][1] - cs[1],
                src_v[bi][i][2] - cs[2],
            ];
            let dy = [
                tgt_v[bi][i][0] - ct[0],
                tgt_v[bi][i][1] - ct[1],
                tgt_v[bi][i][2] - ct[2],
            ];
            let w = w_row[i];
            for r in 0..3 {
                for c in 0..3 {
                    h[r][c] += w * dx[r] * dy[c];
                }
            }
        }

        // SVD via nalgebra.
        let h_mat = nalgebra::Matrix3::new(
            h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
        );
        let svd = h_mat.svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        // R = V · diag(1, 1, det(V Uᵀ)) · Uᵀ — fix sign to ensure SO(3).
        let v = vt.transpose();
        let ut = u.transpose();
        let det_vu = (v * ut).determinant();
        let mut d = nalgebra::Matrix3::identity();
        d[(2, 2)] = det_vu.signum();
        let r_mat = v * d * ut;

        for r in 0..3 {
            for c in 0..3 {
                r_out.push(r_mat[(r, c)]);
            }
        }
        // t = ct − R · cs
        let r_cs = [
            r_mat[(0, 0)] * cs[0] + r_mat[(0, 1)] * cs[1] + r_mat[(0, 2)] * cs[2],
            r_mat[(1, 0)] * cs[0] + r_mat[(1, 1)] * cs[1] + r_mat[(1, 2)] * cs[2],
            r_mat[(2, 0)] * cs[0] + r_mat[(2, 1)] * cs[1] + r_mat[(2, 2)] * cs[2],
        ];
        for j in 0..3 {
            t_out.push(ct[j] - r_cs[j]);
        }
    }

    let r_tensor = Tensor::from_vec(r_out, (b, 3, 3), &device)?.to_dtype(dtype)?;
    let t_tensor = Tensor::from_vec(t_out, (b, 3), &device)?.to_dtype(dtype)?;
    Ok((r_tensor, t_tensor))
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn stack_3x3(
    m00: &Tensor, m01: &Tensor, m02: &Tensor,
    m10: &Tensor, m11: &Tensor, m12: &Tensor,
    m20: &Tensor, m21: &Tensor, m22: &Tensor,
) -> Result<Tensor> {
    // Each input is `[..., 1, 1]`. Concatenate columns then rows.
    let last = m00.rank() - 1;
    let row0 = Tensor::cat(&[m00, m01, m02], last)?; // [..., 1, 3]
    let row1 = Tensor::cat(&[m10, m11, m12], last)?;
    let row2 = Tensor::cat(&[m20, m21, m22], last)?;
    Tensor::cat(&[&row0, &row1, &row2], last - 1)
}

/// Identity `[..., 3, 3]` broadcastable to the leading dims of `like` (a
/// `[..., 3]` tensor). Returns shape `[..., 3, 3]` with same dtype/device.
fn identity_3x3_like(like: &Tensor) -> Result<Tensor> {
    let dims = like.dims();
    let n = dims.len();
    let leading: Vec<usize> = dims.iter().take(n - 1).copied().collect();
    let mut shape = leading.clone();
    shape.push(3);
    shape.push(3);
    let i3 = Tensor::eye(3, like.dtype(), like.device())?;
    // Reshape to (1,1,...,3,3) then expand to full leading shape.
    let mut view_shape: Vec<usize> = leading.iter().map(|_| 1).collect();
    view_shape.push(3);
    view_shape.push(3);
    i3.reshape(view_shape)?.broadcast_as(shape)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::{Device, Tensor};

    fn cpu() -> Device { Device::Cpu }

    #[test]
    fn rotvec_zero_is_identity() {
        let v = Tensor::from_vec(vec![0.0_f64, 0.0, 0.0], (1, 3), &cpu()).unwrap();
        let r = rotvec_to_rotmat(&v).unwrap();
        let r: Vec<Vec<Vec<f64>>> = r.to_vec3().unwrap();
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r[0][i][j], eye[i][j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rotvec_z_pi_over_two() {
        let theta = std::f64::consts::FRAC_PI_2;
        let v = Tensor::from_vec(vec![0.0_f64, 0.0, theta], (1, 3), &cpu()).unwrap();
        let r = rotvec_to_rotmat(&v).unwrap();
        let r: Vec<Vec<Vec<f64>>> = r.to_vec3().unwrap();
        // 90° around z: rotates (1,0,0) to (0,1,0). Matrix is
        // [[0,-1,0],[1,0,0],[0,0,1]].
        let expected = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r[0][i][j], expected[i][j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn euler_x_90deg_matches_rotvec() {
        let angle = Tensor::from_vec(vec![std::f64::consts::FRAC_PI_2], 1, &cpu()).unwrap();
        let m_euler = euler_to_rotmat('x', &angle, false).unwrap();
        let m_rotvec = rotvec_to_rotmat(
            &Tensor::from_vec(
                vec![std::f64::consts::FRAC_PI_2, 0.0, 0.0],
                (1, 3),
                &cpu(),
            )
            .unwrap(),
        )
        .unwrap();
        let a: Vec<Vec<Vec<f64>>> = m_euler.to_vec3().unwrap();
        let b: Vec<Vec<Vec<f64>>> = m_rotvec.to_vec3().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[0][i][j], b[0][i][j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn euler_degrees_flag() {
        let radians_pi_4 = Tensor::from_vec(vec![std::f64::consts::FRAC_PI_4], 1, &cpu()).unwrap();
        let degrees_45 = Tensor::from_vec(vec![45.0_f64], 1, &cpu()).unwrap();
        let m_rad = euler_to_rotmat('y', &radians_pi_4, false).unwrap();
        let m_deg = euler_to_rotmat('y', &degrees_45, true).unwrap();
        let a: Vec<Vec<Vec<f64>>> = m_rad.to_vec3().unwrap();
        let b: Vec<Vec<Vec<f64>>> = m_deg.to_vec3().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[0][i][j], b[0][i][j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rotmat_unitquat_roundtrip() {
        // Build rotations from a few rotation vectors, then verify
        // rotmat → quat → quat_action gives the same rotated vector as the
        // matrix multiply.
        let rotvec = Tensor::from_vec(
            vec![0.3_f64, -0.7, 0.2, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0],
            (3, 3),
            &cpu(),
        )
        .unwrap();
        let r = rotvec_to_rotmat(&rotvec).unwrap();
        let q = rotmat_to_unitquat(&r).unwrap();

        let v = Tensor::from_vec(
            vec![1.0_f64, 0.5, -0.3, 0.1, 0.2, 0.3, -1.0, 2.0, 0.5],
            (3, 3),
            &cpu(),
        )
        .unwrap();

        let by_quat = quat_action(&q, &v).unwrap();
        let v_col = v.unsqueeze(2).unwrap();
        let by_mat = r.matmul(&v_col).unwrap().squeeze(2).unwrap();

        let a: Vec<Vec<f64>> = by_quat.to_vec2().unwrap();
        let b: Vec<Vec<f64>> = by_mat.to_vec2().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[i][j], b[i][j], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn quat_product_identity() {
        // Identity quaternion is (0,0,0,1).
        let q = Tensor::from_vec(vec![0.0_f64, 0.5, 0.0, 0.5_f64.sqrt()], 4, &cpu())
            .unwrap();
        // Normalize for safety.
        let norm: f64 = q.sqr().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap().sqrt();
        let q = q.affine(1.0 / norm, 0.0).unwrap();

        let id = Tensor::from_vec(vec![0.0_f64, 0.0, 0.0, 1.0], 4, &cpu()).unwrap();
        let prod = quat_product(&q, &id).unwrap();
        let q_v: Vec<f64> = q.to_vec1().unwrap();
        let p_v: Vec<f64> = prod.to_vec1().unwrap();
        for i in 0..4 {
            assert_relative_eq!(q_v[i], p_v[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn rigid_homogeneous_roundtrip() {
        // Build a rotation + translation, pack to homogeneous, unpack, repack.
        let r = rotvec_to_rotmat(&Tensor::from_vec(vec![0.3_f64, 0.4, 0.1], (1, 3), &cpu()).unwrap())
            .unwrap();
        let t = Tensor::from_vec(vec![0.7_f64, -1.2, 0.3], (1, 3), &cpu()).unwrap();
        let h = rigid_to_homogeneous(&r, &t).unwrap();
        assert_eq!(h.dims(), &[1, 4, 4]);
        let (r2, t2) = rigid_from_homogeneous(&h).unwrap();
        let r_v: Vec<Vec<Vec<f64>>> = r.to_vec3().unwrap();
        let r2_v: Vec<Vec<Vec<f64>>> = r2.to_vec3().unwrap();
        let t_v: Vec<Vec<f64>> = t.to_vec2().unwrap();
        let t2_v: Vec<Vec<f64>> = t2.to_vec2().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_v[0][i][j], r2_v[0][i][j], epsilon = 1e-12);
            }
            assert_relative_eq!(t_v[0][i], t2_v[0][i], epsilon = 1e-12);
        }
    }

    #[test]
    fn rigid_inverse_undoes_transform() {
        let r = rotvec_to_rotmat(&Tensor::from_vec(vec![0.3_f64, -0.4, 0.7], (1, 3), &cpu()).unwrap())
            .unwrap();
        let t = Tensor::from_vec(vec![0.5_f64, -0.2, 1.1], (1, 3), &cpu()).unwrap();
        let h = rigid_to_homogeneous(&r, &t).unwrap();
        let h_inv = rigid_inverse_homogeneous(&h).unwrap();
        let prod = h.matmul(&h_inv).unwrap();
        let p: Vec<Vec<Vec<f64>>> = prod.to_vec3().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(p[0][i][j], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn registration_recovers_known_rigid() {
        // Take src, apply a known R + t, then recover them.
        let src_data: Vec<f64> = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            0.5, 0.5, 0.5,
        ];
        let src = Tensor::from_vec(src_data.clone(), (1, 6, 3), &cpu()).unwrap();

        // Known rotation: 30° around z.
        let theta = 30.0_f64.to_radians();
        let r = rotvec_to_rotmat(&Tensor::from_vec(vec![0.0_f64, 0.0, theta], (1, 3), &cpu()).unwrap())
            .unwrap();
        let t = Tensor::from_vec(vec![0.4_f64, -0.7, 1.2], (1, 3), &cpu()).unwrap();

        // tgt = R @ src + t
        let tgt = r.matmul(&src.transpose(1, 2).unwrap()).unwrap()
            .transpose(1, 2).unwrap()
            .broadcast_add(&t.unsqueeze(1).unwrap()).unwrap();

        let (r_hat, t_hat) = rigid_points_registration(&src, &tgt, None).unwrap();
        let r_v: Vec<Vec<Vec<f64>>> = r.to_vec3().unwrap();
        let r_hat_v: Vec<Vec<Vec<f64>>> = r_hat.to_vec3().unwrap();
        let t_v: Vec<Vec<f64>> = t.to_vec2().unwrap();
        let t_hat_v: Vec<Vec<f64>> = t_hat.to_vec2().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(r_v[0][i][j], r_hat_v[0][i][j], epsilon = 1e-9);
            }
            assert_relative_eq!(t_v[0][i], t_hat_v[0][i], epsilon = 1e-9);
        }
    }
}
