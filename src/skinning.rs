//! Skinning — port of `anny/src/anny/skinning/skinning.py`. We provide:
//!
//! * [`apply_linear_blendshape`] — `T + Σ_c coeff_c · blendshape_c`
//! * [`linear_blend_skinning`] (LBS) — sparse-weight weighted average of
//!   per-bone homogeneous transforms applied to each vertex.
//! * [`dual_quaternion_skinning`] (DQS) — same input shape as LBS, but blends
//!   the rigid transforms in dual-quaternion space (volume-preserving).
//!
//! The Warp-accelerated path from the Python source is **not** ported — the
//! pure-tensor versions here match the Python reference's correctness tests.
//!
//! Inputs (consistent across LBS/DQS):
//! * `vertices`             — `[bs, V, 3]`
//! * `bone_weights`         — `[bs, V, M]`, weights per vertex per slot, summing to 1
//! * `bone_indices`         — `[bs, V, M]` of `u32`, bone index per slot
//! * `bone_transforms`      — `[bs, K, 4, 4]` per-bone homogeneous transforms
//!
//! Output: skinned vertices `[bs, V, 3]`.

use candle_core::{D, DType, Device, Result, Tensor};

use crate::rotation::{
    quat_action, quat_conjugation, quat_product, rigid_from_homogeneous, rotmat_to_unitquat,
};

#[cfg(test)]
use crate::rotation::rigid_to_homogeneous;

/// `T_v = template_vertices + Σ_c coeff_{b,c} · blendshape_c`.
///
/// `template_vertices`: `[V, 3]` (or `[1, V, 3]`)
/// `blendshapes`:        `[C, V, 3]`
/// `blendshape_coeffs`:  `[B, C]`
/// returns:              `[B, V, 3]`
pub fn apply_linear_blendshape(
    template_vertices: &Tensor,
    blendshapes: &Tensor,
    blendshape_coeffs: &Tensor,
) -> Result<Tensor> {
    let template = if template_vertices.rank() == 2 {
        template_vertices.unsqueeze(0)?
    } else {
        template_vertices.clone()
    };
    // Python: torch.einsum("cpd, bc -> bpd", blendshapes, coeffs)
    // Equivalent: coeffs @ blendshapes.reshape(C, V*3) then reshape
    let dims = blendshapes.dims();
    if dims.len() != 3 {
        candle_core::bail!("blendshapes must be [C, V, 3], got {dims:?}");
    }
    let c = dims[0];
    let v = dims[1];
    let three = dims[2];
    let bs_flat = blendshapes.reshape((c, v * three))?;
    let bs_size = blendshape_coeffs.dim(0)?;
    let mixed = blendshape_coeffs.matmul(&bs_flat)?.reshape((bs_size, v, three))?; // [B, V, 3]
    let template_b = template.broadcast_as(mixed.shape())?;
    template_b.add(&mixed)
}

/// Linear blend skinning. Mirrors `linear_blend_skinning` (lines 8–47).
pub fn linear_blend_skinning(
    vertices: &Tensor,
    bone_weights: &Tensor,
    bone_indices: &Tensor,
    bone_transforms: &Tensor,
) -> Result<Tensor> {
    let bs = max_batch(&[vertices, bone_weights, bone_indices, bone_transforms])?;
    let v = vertices.dim(1)?;
    let m = bone_indices.dim(2)?;
    let k = bone_transforms.dim(1)?;
    let device = vertices.device().clone();
    let dtype = vertices.dtype();

    let v_b = broadcast_first(vertices, bs)?;
    let w_b = broadcast_first(bone_weights, bs)?;
    let idx_b = broadcast_first(bone_indices, bs)?;
    let bt_b = broadcast_first(bone_transforms, bs)?;

    // Selected per-(b, v, m) bone transforms via flat index_select.
    let selected = gather_bone_transforms(&idx_b, &bt_b, bs, v, m, k, &device)?; // [bs, V, M, 4, 4]

    // Weighted sum over the M axis.
    let w_expanded = w_b.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // [bs, V, M, 1, 1]
    let weighted = w_expanded.broadcast_mul(&selected)?;
    let transforms = weighted.sum(2)?; // [bs, V, 4, 4]

    // Apply: vertices_h = cat(v, 1); out = transforms · v_h → take xyz.
    let ones = Tensor::ones((bs, v, 1), dtype, &device)?;
    let v_h = Tensor::cat(&[&v_b, &ones], D::Minus1)?.unsqueeze(D::Minus1)?; // [bs, V, 4, 1]
    let out_h = transforms.matmul(&v_h)?.squeeze(D::Minus1)?; // [bs, V, 4]
    out_h.narrow(D::Minus1, 0, 3)?.contiguous()
}

/// Dual-quaternion skinning. Mirrors `dual_quaternion_skinning` (lines 67–119).
pub fn dual_quaternion_skinning(
    vertices: &Tensor,
    bone_weights: &Tensor,
    bone_indices: &Tensor,
    bone_transforms: &Tensor,
) -> Result<Tensor> {
    let bs = max_batch(&[vertices, bone_weights, bone_indices, bone_transforms])?;
    let v = vertices.dim(1)?;
    let m = bone_indices.dim(2)?;
    let k = bone_transforms.dim(1)?;
    let device = vertices.device().clone();
    let dtype = vertices.dtype();

    let v_b = broadcast_first(vertices, bs)?;
    let w_b = broadcast_first(bone_weights, bs)?;
    let idx_b = broadcast_first(bone_indices, bs)?;
    let bt_b = broadcast_first(bone_transforms, bs)?;

    // Convert each bone transform into a (q_rotation, q_translation) pair.
    let (rot_part, tr_part) = homogeneous_to_dual_quaternion(&bt_b)?; // both [bs, K, 4]

    // Gather per-vertex per-slot dual quaternions.
    // We can reuse the index machinery by treating each quaternion as 4 scalars.
    let rot_sel = gather_quat(&idx_b, &rot_part, bs, v, m, k, &device)?; // [bs, V, M, 4]
    let tr_sel = gather_quat(&idx_b, &tr_part, bs, v, m, k, &device)?; // [bs, V, M, 4]

    // Antipodal sign correction relative to slot 0.
    let rot_ref = rot_sel.narrow(2, 0, 1)?;
    let tr_ref = tr_sel.narrow(2, 0, 1)?;
    let dot_rot = rot_sel.broadcast_mul(&rot_ref)?.sum_keepdim(D::Minus1)?;
    let dot_tr = tr_sel.broadcast_mul(&tr_ref)?.sum_keepdim(D::Minus1)?;
    let dot = (dot_rot + dot_tr)?; // [bs, V, M, 1]
    // sign = (dot >= 0) ? 1 : -1
    let zero = Tensor::zeros_like(&dot)?;
    let positive = dot.ge(&zero)?.to_dtype(dtype)?;
    let sign = positive.affine(2.0, -1.0)?;
    let rot_sel = rot_sel.broadcast_mul(&sign)?;
    let tr_sel = tr_sel.broadcast_mul(&sign)?;

    // Linear blend in DQ space.
    let w_exp = w_b.unsqueeze(D::Minus1)?; // [bs, V, M, 1]
    let mean_rot = w_exp.broadcast_mul(&rot_sel)?.sum(2)?; // [bs, V, 4]
    let mean_tr = w_exp.broadcast_mul(&tr_sel)?.sum(2)?; // [bs, V, 4]

    // Normalise.
    let norm = mean_rot.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let mean_rot = mean_rot.broadcast_div(&norm)?;
    let mean_tr = mean_tr.broadcast_div(&norm)?;

    // tr = (2 · q_tr · conj(q_rot))[..., :3]
    let q_conj = quat_conjugation(&mean_rot)?;
    let prod = quat_product(&mean_tr, &q_conj)?;
    let tr = prod.narrow(D::Minus1, 0, 3)?.affine(2.0, 0.0)?;

    // Apply rotation to vertices, then add translation.
    let rotated = quat_action(&mean_rot, &v_b)?;
    rotated.add(&tr)?.contiguous()
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn max_batch(tensors: &[&Tensor]) -> Result<usize> {
    let mut max = 1;
    for t in tensors {
        let b = t.dim(0)?;
        if b > max {
            max = b;
        }
    }
    Ok(max)
}

fn broadcast_first(t: &Tensor, target_batch: usize) -> Result<Tensor> {
    let b = t.dim(0)?;
    if b == target_batch {
        return t.contiguous();
    }
    if b == 1 {
        let mut shape = vec![target_batch];
        shape.extend_from_slice(&t.dims()[1..]);
        return t.broadcast_as(shape)?.contiguous();
    }
    candle_core::bail!("cannot broadcast batch dim {b} to {target_batch}");
}

/// `bone_indices: [bs, V, M] u32` selects rows from `bone_transforms: [bs, K, 4, 4]`,
/// returning `[bs, V, M, 4, 4]`. Implemented via a flat index_select on a
/// 2D view of the transforms (shape `[bs*K, 16]`), with global offsets added
/// to the indices.
fn gather_bone_transforms(
    bone_indices: &Tensor,
    bone_transforms: &Tensor,
    bs: usize,
    v: usize,
    m: usize,
    k: usize,
    device: &Device,
) -> Result<Tensor> {
    let bt_flat = bone_transforms.reshape((bs * k, 4 * 4))?;
    let abs_idx = absolute_indices(bone_indices, bs, k, device)?; // [bs*V*M] u32
    let selected = bt_flat.index_select(&abs_idx, 0)?; // [bs*V*M, 16]
    selected.reshape((bs, v, m, 4, 4))
}

fn gather_quat(
    bone_indices: &Tensor,
    bone_quats: &Tensor,
    bs: usize,
    v: usize,
    m: usize,
    k: usize,
    device: &Device,
) -> Result<Tensor> {
    let q_flat = bone_quats.reshape((bs * k, 4))?;
    let abs_idx = absolute_indices(bone_indices, bs, k, device)?;
    let selected = q_flat.index_select(&abs_idx, 0)?;
    selected.reshape((bs, v, m, 4))
}

fn absolute_indices(bone_indices: &Tensor, bs: usize, k: usize, device: &Device) -> Result<Tensor> {
    // bone_indices: [bs, V, M], typically u32. Convert to absolute indices into
    // the flattened [bs * K, ...] tensor.
    let idx_u32 = bone_indices.to_dtype(DType::U32)?.to_device(device)?;
    let offsets = Tensor::arange(0u32, bs as u32, device)?
        .reshape((bs, 1, 1))?
        .affine(k as f64, 0.0)?
        .to_dtype(DType::U32)?;
    let abs = idx_u32.broadcast_add(&offsets)?;
    abs.flatten_all()
}

/// `[..., 4, 4] → ([..., 4], [..., 4])` (rotation quat, translation quat)
/// using XYZW convention. The translation quaternion is `(½t, 1) · q_rot`.
fn homogeneous_to_dual_quaternion(homogeneous: &Tensor) -> Result<(Tensor, Tensor)> {
    let (linear, translation) = rigid_from_homogeneous(homogeneous)?;
    let q = rotmat_to_unitquat(&linear)?;
    // q_tr = (0.5·t, 1) — pure-translation quaternion.
    let half_t = translation.affine(0.5, 0.0)?;
    let dims = q.dims().to_vec();
    let mut ones_shape = dims.clone();
    *ones_shape.last_mut().unwrap() = 1;
    let ones = Tensor::ones(ones_shape, q.dtype(), q.device())?;
    let q_tr_initial = Tensor::cat(&[&half_t, &ones], D::Minus1)?;
    let q_tr = quat_product(&q_tr_initial, &q)?;
    Ok((q, q_tr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu() -> Device { Device::Cpu }

    #[test]
    fn lbs_identity_transforms_passes_vertices_through() {
        // 1 batch, 4 vertices, 2 bones, 1 bone per vertex.
        let device = cpu();
        let bs = 1; let v = 4; let m = 1; let k = 2;
        let vertices = Tensor::from_vec(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            (bs, v, 3), &device).unwrap();
        let weights = Tensor::ones((bs, v, m), DType::F64, &device).unwrap();
        let indices = Tensor::from_vec(vec![0u32, 1, 0, 1], (bs, v, m), &device).unwrap();
        let identity = Tensor::eye(4, DType::F64, &device).unwrap();
        let transforms = identity
            .reshape((1, 1, 4, 4)).unwrap()
            .broadcast_as((bs, k, 4, 4)).unwrap()
            .contiguous().unwrap();

        let out = linear_blend_skinning(&vertices, &weights, &indices, &transforms).unwrap();
        let in_v: Vec<f64> = vertices.flatten_all().unwrap().to_vec1().unwrap();
        let out_v: Vec<f64> = out.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in in_v.iter().zip(out_v.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn lbs_translation_only() {
        // 1 batch, 2 vertices, 1 bone slot, 1 bone (just a translation).
        let device = cpu();
        let bs = 1; let v = 2; let m = 1; let k = 1;
        let vertices = Tensor::from_vec(
            vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0],
            (bs, v, 3), &device).unwrap();
        let weights = Tensor::ones((bs, v, m), DType::F64, &device).unwrap();
        let indices = Tensor::from_vec(vec![0u32, 0], (bs, v, m), &device).unwrap();

        // Translation by (10, 20, 30).
        let mat = vec![
            1.0_f64, 0.0, 0.0, 10.0,
            0.0, 1.0, 0.0, 20.0,
            0.0, 0.0, 1.0, 30.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let transforms = Tensor::from_vec(mat, (bs, k, 4, 4), &device).unwrap();

        let out = linear_blend_skinning(&vertices, &weights, &indices, &transforms).unwrap();
        let v: Vec<f64> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!((v[0] - 10.0).abs() < 1e-12);
        assert!((v[1] - 20.0).abs() < 1e-12);
        assert!((v[2] - 30.0).abs() < 1e-12);
        assert!((v[3] - 11.0).abs() < 1e-12);
        assert!((v[4] - 21.0).abs() < 1e-12);
        assert!((v[5] - 31.0).abs() < 1e-12);
    }

    #[test]
    fn dqs_matches_lbs_for_pure_rigid_transforms() {
        // For a single bone (M=1), DQS and LBS should produce identical
        // results — both apply the same rigid transform.
        let device = cpu();
        let bs = 1; let v = 4; let m = 1; let k = 1;

        let r = crate::rotation::rotvec_to_rotmat(
            &Tensor::from_vec(vec![0.3_f64, -0.2, 0.5], (1, 3), &device).unwrap()
        ).unwrap();
        let t = Tensor::from_vec(vec![0.5_f64, -1.0, 2.5], (1, 3), &device).unwrap();
        let transforms = rigid_to_homogeneous(&r, &t).unwrap()
            .unsqueeze(0).unwrap();
        assert_eq!(transforms.dims(), &[bs, k, 4, 4]);

        let vertices = Tensor::from_vec(
            vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
            (bs, v, 3), &device).unwrap();
        let weights = Tensor::ones((bs, v, m), DType::F64, &device).unwrap();
        let indices = Tensor::from_vec(vec![0u32; v * m], (bs, v, m), &device).unwrap();

        let out_lbs = linear_blend_skinning(&vertices, &weights, &indices, &transforms).unwrap();
        let out_dqs = dual_quaternion_skinning(&vertices, &weights, &indices, &transforms).unwrap();
        let lbs_v: Vec<f64> = out_lbs.flatten_all().unwrap().to_vec1().unwrap();
        let dqs_v: Vec<f64> = out_dqs.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in lbs_v.iter().zip(dqs_v.iter()) {
            assert!((a - b).abs() < 1e-9, "LBS={a} DQS={b}");
        }
    }

    #[test]
    fn blendshape_application() {
        let device = cpu();
        // V=2 vertices, C=3 blend shapes
        let template = Tensor::from_vec(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], (2, 3), &device).unwrap();
        let blendshapes = Tensor::from_vec(
            vec![
                // shape 0: translate v0 by (1, 0, 0)
                1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0,
                // shape 1: translate v1 by (0, 1, 0)
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                // shape 2: translate v0 by (0, 0, -1)
                0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
            ],
            (3, 2, 3), &device).unwrap();
        // coeffs: batch 1, [0.5, 0.0, 1.0]
        let coeffs = Tensor::from_vec(vec![0.5_f64, 0.0, 1.0], (1, 3), &device).unwrap();
        let out = apply_linear_blendshape(&template, &blendshapes, &coeffs).unwrap();
        let v: Vec<f64> = out.flatten_all().unwrap().to_vec1().unwrap();
        // v0 = (0,0,0) + 0.5*(1,0,0) + 0*(0,0,0) + 1*(0,0,-1) = (0.5, 0, -1)
        // v1 = (1,1,1) + 0.5*(0,0,0) + 0*(0,1,0) + 1*(0,0,0) = (1, 1, 1)
        let expected = [0.5_f64, 0.0, -1.0, 1.0, 1.0, 1.0];
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "got {a}, want {b}");
        }
    }
}
