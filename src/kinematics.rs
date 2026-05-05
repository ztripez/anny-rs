//! Forward kinematics — port of `anny/src/anny/utils/kinematics.py`.
//!
//! The bone tree is laid out as a parent-index list of length `B`; root bones
//! have parent `-1`. Forward kinematics is propagated level-by-level (a
//! "propagation front" is the set of bones at the same depth) so each level
//! can be processed in parallel. The resulting `[bs, B, 4, 4]` `poses` and
//! `transforms` tensors are the homogeneous-matrix sequence each LBS pass
//! consumes.

use candle_core::{D, Result, Tensor};

use crate::rotation::{rigid_inverse_homogeneous, rotvec_to_rotmat};

// ────────────────────────────────────────────────────────────────────────────
// Propagation fronts.
// ────────────────────────────────────────────────────────────────────────────

/// Topologically groups joints into "fronts": each front is a set of bones
/// whose parents are all already resolved, so the front itself can be batched.
///
/// Returns a `Vec<(indices, parents)>` where `indices[i]` is the bones at
/// depth `i` and `parents[i]` is each bone's parent index (or `-1` for roots).
pub fn propagation_fronts(parent_indices: &[i64]) -> Vec<(Vec<usize>, Vec<i64>)> {
    let n = parent_indices.len();
    let mut assigned = vec![false; n];
    let mut fronts = Vec::new();

    let mut current: Vec<usize> = (0..n).filter(|i| parent_indices[*i] < 0).collect();
    while !current.is_empty() {
        let parents: Vec<i64> = current.iter().map(|i| parent_indices[*i]).collect();
        let current_set: std::collections::HashSet<usize> = current.iter().copied().collect();
        for &j in &current {
            assigned[j] = true;
        }
        fronts.push((current.clone(), parents));
        current = (0..n)
            .filter(|i| !assigned[*i] && parent_indices[*i] >= 0)
            .filter(|i| current_set.contains(&(parent_indices[*i] as usize)))
            .collect();
    }
    debug_assert!(assigned.iter().all(|x| *x));
    fronts
}

// ────────────────────────────────────────────────────────────────────────────
// Forward kinematic driver.
// ────────────────────────────────────────────────────────────────────────────

/// Output of `parallel_forward_kinematic`. Both tensors are `[bs, B, 4, 4]`.
pub struct FkOutput {
    /// Absolute world-space pose of each bone.
    pub poses: Tensor,
    /// Per-bone transform from rest pose to current pose: `pose @ rest⁻¹`.
    pub transforms: Tensor,
}

/// Level-batched forward kinematics.
///
/// * `fronts` — output of [`propagation_fronts`] (passed in pre-built so the
///   model can cache it).
/// * `rest_bone_poses` — `[bs, B, 4, 4]` rest-pose homogeneous matrices.
/// * `delta_transforms` — `[bs, B, 4, 4]` per-bone delta transforms applied
///   in the rest-pose frame (`T = rest @ delta`).
/// * `base_transform` — optional `[bs, 4, 4]` to left-multiply onto the root
///   bones (used when callers want a global transform, e.g. world placement).
///
/// Mirrors `parallel_forward_kinematic` (lines 156–203 of `kinematics.py`).
pub fn parallel_forward_kinematic(
    fronts: &[(Vec<usize>, Vec<i64>)],
    rest_bone_poses: &Tensor,
    delta_transforms: &Tensor,
    base_transform: Option<&Tensor>,
) -> Result<FkOutput> {
    let dims = rest_bone_poses.dims();
    if dims.len() != 4 || dims[2] != 4 || dims[3] != 4 {
        candle_core::bail!(
            "rest_bone_poses must be [bs, B, 4, 4], got {dims:?}"
        );
    }
    let bs = dims[0];
    let n_bones = dims[1];
    let device = rest_bone_poses.device().clone();
    let dtype = rest_bone_poses.dtype();

    let rest_bone_poses_c = rest_bone_poses.contiguous()?;
    let delta_transforms_c = delta_transforms.contiguous()?;
    let t_all = rest_bone_poses_c.matmul(&delta_transforms_c)?.contiguous()?; // [bs, B, 4, 4]
    let rest_inv_all = rigid_inverse_homogeneous(&rest_bone_poses_c)?.contiguous()?; // [bs, B, 4, 4]

    // Per-bone results, populated in topological order.
    let mut poses_per_bone: Vec<Option<Tensor>> = vec![None; n_bones];
    let mut transforms_per_bone: Vec<Option<Tensor>> = vec![None; n_bones];

    for (indices, parents) in fronts {
        // ── Roots in this front (parent == -1).
        for (k, &bone_id) in indices.iter().enumerate() {
            let parent_id = parents[k];
            if parent_id != -1 {
                continue;
            }
            let t = t_all.narrow(1, bone_id, 1)?.squeeze(1)?; // [bs, 4, 4]
            let pose = match base_transform {
                Some(base) => base.matmul(&t)?,
                None => t,
            };
            let rest_inv = rest_inv_all.narrow(1, bone_id, 1)?.squeeze(1)?;
            let transform = pose.matmul(&rest_inv)?;
            poses_per_bone[bone_id] = Some(pose);
            transforms_per_bone[bone_id] = Some(transform);
        }

        // ── Non-roots in this front, batched as one matmul.
        let mut child_ids = Vec::new();
        let mut parent_ids = Vec::new();
        for (k, &bone_id) in indices.iter().enumerate() {
            if parents[k] >= 0 {
                child_ids.push(bone_id);
                parent_ids.push(parents[k] as usize);
            }
        }
        if !child_ids.is_empty() {
            // Gather the parent transforms and child Ts in their level order.
            let mut parent_stack: Vec<Tensor> = Vec::with_capacity(parent_ids.len());
            for &p in &parent_ids {
                parent_stack.push(
                    transforms_per_bone[p]
                        .as_ref()
                        .expect("parent transform must be computed first")
                        .clone(),
                );
            }
            let parent_block = Tensor::stack(&parent_stack, 1)?.contiguous()?; // [bs, L, 4, 4]
            let child_indices = Tensor::from_vec(
                child_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                child_ids.len(),
                &device,
            )?;
            let child_t = t_all.index_select(&child_indices, 1)?.contiguous()?; // [bs, L, 4, 4]
            let rest_inv_block = rest_inv_all.index_select(&child_indices, 1)?.contiguous()?;

            let child_poses = parent_block.matmul(&child_t)?.contiguous()?;
            let child_transforms = child_poses.matmul(&rest_inv_block)?.contiguous()?;

            for (k, &bone_id) in child_ids.iter().enumerate() {
                let p = child_poses.narrow(1, k, 1)?.squeeze(1)?;
                let tr = child_transforms.narrow(1, k, 1)?.squeeze(1)?;
                poses_per_bone[bone_id] = Some(p);
                transforms_per_bone[bone_id] = Some(tr);
            }
        }
    }

    // Stack into [bs, B, 4, 4] in bone_id order.
    let pose_refs: Vec<Tensor> = poses_per_bone
        .into_iter()
        .map(|o| o.expect("every bone must be assigned"))
        .collect();
    let trans_refs: Vec<Tensor> = transforms_per_bone
        .into_iter()
        .map(|o| o.expect("every bone must be assigned"))
        .collect();
    let pose_views: Vec<&Tensor> = pose_refs.iter().collect();
    let trans_views: Vec<&Tensor> = trans_refs.iter().collect();
    let poses = Tensor::stack(&pose_views, 1)?.contiguous()?;
    let transforms = Tensor::stack(&trans_views, 1)?.contiguous()?;

    let _ = (bs, dtype, &device); // suppress unused
    Ok(FkOutput { poses, transforms })
}

/// Sequential forward kinematics. One bone at a time, in topological order.
/// Slower than [`parallel_forward_kinematic`] but uses a different code path,
/// so it serves as a parity reference. Mirrors `forward_kinematic` in
/// `kinematics.py:55–86`.
///
/// Requires bone parents to be in a topologically valid order: every parent
/// index must be `< i`.
pub fn sequential_forward_kinematic(
    bone_parents: &[i64],
    rest_bone_poses: &Tensor,
    delta_transforms: &Tensor,
) -> Result<FkOutput> {
    let dims = rest_bone_poses.dims();
    if dims.len() != 4 || dims[2] != 4 || dims[3] != 4 {
        candle_core::bail!("rest_bone_poses must be [bs, B, 4, 4], got {dims:?}");
    }
    let n_bones = dims[1];
    if bone_parents.len() != n_bones {
        candle_core::bail!("bone_parents length {} ≠ B {}", bone_parents.len(), n_bones);
    }

    let rest_c = rest_bone_poses.contiguous()?;
    let delta_c = delta_transforms.contiguous()?;

    let mut poses: Vec<Option<Tensor>> = vec![None; n_bones];
    let mut transforms: Vec<Option<Tensor>> = vec![None; n_bones];

    for i in 0..n_bones {
        let rest = rest_c.narrow(1, i, 1)?.squeeze(1)?.contiguous()?; // [bs, 4, 4]
        let delta = delta_c.narrow(1, i, 1)?.squeeze(1)?.contiguous()?;
        let t = rest.matmul(&delta)?;
        let pose = if bone_parents[i] >= 0 {
            let p = bone_parents[i] as usize;
            transforms[p]
                .as_ref()
                .expect("parent must be processed before child")
                .matmul(&t)?
        } else {
            t
        };
        let rest_inv = rigid_inverse_homogeneous(&rest)?;
        let transform = pose.matmul(&rest_inv)?.contiguous()?;
        poses[i] = Some(pose.contiguous()?);
        transforms[i] = Some(transform);
    }

    let pose_refs: Vec<Tensor> = poses.into_iter().map(|o| o.unwrap()).collect();
    let trans_refs: Vec<Tensor> = transforms.into_iter().map(|o| o.unwrap()).collect();
    let pose_views: Vec<&Tensor> = pose_refs.iter().collect();
    let trans_views: Vec<&Tensor> = trans_refs.iter().collect();
    let stacked_poses = Tensor::stack(&pose_views, 1)?.contiguous()?;
    let stacked_transforms = Tensor::stack(&trans_views, 1)?.contiguous()?;
    Ok(FkOutput {
        poses: stacked_poses,
        transforms: stacked_transforms,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Bone-pose construction from head/tail/roll.
// ────────────────────────────────────────────────────────────────────────────

/// Computes rest-pose homogeneous matrices from head + tail bone endpoints
/// and per-bone roll rotations. The bone's local Y axis aligns with the
/// head→tail direction; degenerate cases (tail very close to head, or exactly
/// opposite) fall back to `degenerate_rotation`.
///
/// Mirrors `get_bone_poses` (lines 255–295 of `kinematics.py`).
///
/// * `bone_heads`, `bone_tails`: `[bs, B, 3]`
/// * `bone_rolls_rotmat`: `[bs_or_1, B, 3, 3]` — broadcasts on dim 0
/// * `y_axis`: `[3]`
/// * `degenerate_rotation`: `[3, 3]`
pub fn get_bone_poses(
    bone_heads: &Tensor,
    bone_tails: &Tensor,
    bone_rolls_rotmat: &Tensor,
    y_axis: &Tensor,
    degenerate_rotation: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    let dtype = bone_heads.dtype();
    let device = bone_heads.device();

    let vectors = (bone_tails - bone_heads)?; // [bs, B, 3]
    let norms = vectors.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [bs, B, 1]
    let y = vectors.broadcast_div(&norms)?; // [bs, B, 3]

    let y_axis_view = y_axis
        .reshape((1, 1, 3))?
        .to_dtype(dtype)?
        .to_device(device)?;

    // cross_p = y × y_axis  ; dot_p = y · y_axis
    let cross_p = cross_3d(&y, &y_axis_view)?; // [bs, B, 3]
    let dot_p = y.broadcast_mul(&y_axis_view)?.sum(D::Minus1)?; // [bs, B]

    let cross_p_norm = cross_p.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [bs, B, 1]
    let cross_p_norm_2 = cross_p_norm.squeeze(D::Minus1)?; // [bs, B]
    let angle = atan2_host(&cross_p_norm_2, &dot_p)?; // [bs, B]

    // axis = cross_p / cross_p_norm. Where the norm is zero this produces NaN,
    // detected below by `axis · axis ≠ 1`.
    let safe_norm = cross_p_norm.broadcast_maximum(
        &Tensor::new(1e-30_f64, device)?.to_dtype(dtype)?,
    )?;
    let axis = cross_p.broadcast_div(&safe_norm)?; // [bs, B, 3]

    let rotvec = axis.broadcast_mul(&angle.unsqueeze(D::Minus1)?)?.neg()?; // [bs, B, 3]
    let r = rotvec_to_rotmat(&rotvec)?; // [bs, B, 3, 3]

    // is_valid = | axis · axis - 1 | < epsilon
    let axis_norm_sq = axis.sqr()?.sum(D::Minus1)?; // [bs, B]
    let validity = (axis_norm_sq.affine(1.0, -1.0)?.abs()?)
        .lt(epsilon)?; // [bs, B]
    let validity_4d = validity
        .unsqueeze(D::Minus1)?
        .unsqueeze(D::Minus1)?
        .broadcast_as(r.shape())?; // [bs, B, 3, 3]

    let degenerate = degenerate_rotation
        .reshape((1, 1, 3, 3))?
        .to_dtype(dtype)?
        .to_device(device)?
        .broadcast_as(r.shape())?
        .contiguous()?;

    let r = validity_4d.where_cond(&r, &degenerate)?;

    // R = R · bone_rolls_rotmat (broadcast over batch if needed).
    let r = r.broadcast_matmul(&bone_rolls_rotmat.to_dtype(dtype)?.to_device(device)?)?;

    // Pack into [bs, B, 4, 4] with translation = bone_heads.
    pack_homogeneous(&r, bone_heads)
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn atan2_host(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    // candle has no native atan2; do it on the host. Both inputs share dtype/device.
    let dtype = y.dtype();
    let device = y.device().clone();
    let dims = y.dims().to_vec();
    let y_h: Vec<f64> = y
        .to_dtype(candle_core::DType::F64)?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let x_h: Vec<f64> = x
        .to_dtype(candle_core::DType::F64)?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let result: Vec<f64> = y_h.iter().zip(x_h.iter()).map(|(&yi, &xi)| yi.atan2(xi)).collect();
    Tensor::from_vec(result, dims, &device)?.to_dtype(dtype)
}

fn cross_3d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // 3D cross product on the last dim. Uses broadcasting on the leading dims.
    let last = a.rank() - 1;
    let ax = a.narrow(last, 0, 1)?;
    let ay = a.narrow(last, 1, 1)?;
    let az = a.narrow(last, 2, 1)?;
    let bx = b.narrow(last, 0, 1)?;
    let by = b.narrow(last, 1, 1)?;
    let bz = b.narrow(last, 2, 1)?;
    let cx = (ay.broadcast_mul(&bz)? - az.broadcast_mul(&by)?)?;
    let cy = (az.broadcast_mul(&bx)? - ax.broadcast_mul(&bz)?)?;
    let cz = (ax.broadcast_mul(&by)? - ay.broadcast_mul(&bx)?)?;
    Tensor::cat(&[&cx, &cy, &cz], last)
}

fn pack_homogeneous(linear: &Tensor, translation: &Tensor) -> Result<Tensor> {
    // linear: [..., 3, 3], translation: [..., 3] → [..., 4, 4]
    let device = linear.device();
    let dtype = linear.dtype();
    let dims = linear.dims();
    let n = dims.len();
    let leading: Vec<usize> = dims.iter().take(n - 2).copied().collect();

    let t_col = translation.unsqueeze(D::Minus1)?; // [..., 3, 1]
    let top = Tensor::cat(&[linear, &t_col], n - 1)?; // [..., 3, 4]

    let mut bottom_shape = leading.clone();
    bottom_shape.push(1);
    bottom_shape.push(4);
    let bottom_unbroadcast = Tensor::from_vec(vec![0.0_f64, 0.0, 0.0, 1.0], (1, 4), device)?
        .to_dtype(dtype)?;
    let bottom = bottom_unbroadcast.broadcast_as(bottom_shape)?;
    Tensor::cat(&[&top, &bottom], n - 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn cpu() -> Device { Device::Cpu }

    #[test]
    fn fronts_simple_chain() {
        // 0 → 1 → 2 → 3
        let parents = vec![-1_i64, 0, 1, 2];
        let fronts = propagation_fronts(&parents);
        assert_eq!(fronts.len(), 4);
        assert_eq!(fronts[0].0, vec![0]);
        assert_eq!(fronts[1].0, vec![1]);
        assert_eq!(fronts[2].0, vec![2]);
        assert_eq!(fronts[3].0, vec![3]);
    }

    #[test]
    fn fronts_branching() {
        //          0
        //        / | \
        //       1  2  3
        //      /
        //     4
        let parents = vec![-1_i64, 0, 0, 0, 1];
        let fronts = propagation_fronts(&parents);
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0].0, vec![0]);
        assert_eq!(fronts[1].0, vec![1, 2, 3]);
        assert_eq!(fronts[2].0, vec![4]);
    }

    #[test]
    fn fk_identity_delta_recovers_rest() {
        // 3 bones in a chain with arbitrary rest poses; delta = identity →
        // pose should equal accumulated rest poses, transform should be identity.
        let parents = vec![-1_i64, 0, 1];
        let fronts = propagation_fronts(&parents);

        let rest = arbitrary_rest_chain();
        let bs = rest.dim(0).unwrap();
        let n = rest.dim(1).unwrap();
        let identity = Tensor::eye(4, DType::F64, &cpu()).unwrap();
        let delta = identity
            .reshape((1, 1, 4, 4))
            .unwrap()
            .broadcast_as((bs, n, 4, 4))
            .unwrap()
            .contiguous()
            .unwrap();

        let out = parallel_forward_kinematic(&fronts, &rest, &delta, None).unwrap();
        assert_eq!(out.poses.dims(), &[bs, n, 4, 4]);

        // transforms should all be identity. Flatten to a 1D vec for indexing.
        let flat: Vec<f64> = out.transforms.flatten_all().unwrap().to_vec1().unwrap();
        for bi in 0..bs {
            for k in 0..n {
                for i in 0..4 {
                    for j in 0..4 {
                        let want = if i == j { 1.0 } else { 0.0 };
                        let idx = ((bi * n + k) * 4 + i) * 4 + j;
                        let got = flat[idx];
                        assert!(
                            (got - want).abs() < 1e-9,
                            "transforms[{bi}][{k}][{i}][{j}] = {got}, want {want}"
                        );
                    }
                }
            }
        }
    }

    fn arbitrary_rest_chain() -> Tensor {
        // Construct rest poses for a 3-bone chain at arbitrary positions/orientations.
        // Each rest pose is the world-pose of the bone's tail-coordinate frame.
        let device = cpu();
        let r1 = crate::rotation::rotvec_to_rotmat(
            &Tensor::from_vec(vec![0.1_f64, 0.2, 0.3], (1, 3), &device).unwrap(),
        )
        .unwrap();
        let t1 = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], (1, 3), &device).unwrap();
        let h1 = crate::rotation::rigid_to_homogeneous(&r1, &t1).unwrap();

        let r2 = crate::rotation::rotvec_to_rotmat(
            &Tensor::from_vec(vec![0.4_f64, -0.1, 0.05], (1, 3), &device).unwrap(),
        )
        .unwrap();
        let t2 = Tensor::from_vec(vec![1.5_f64, 2.5, 3.5], (1, 3), &device).unwrap();
        let h2 = crate::rotation::rigid_to_homogeneous(&r2, &t2).unwrap();

        let r3 = crate::rotation::rotvec_to_rotmat(
            &Tensor::from_vec(vec![-0.2_f64, 0.0, 0.4], (1, 3), &device).unwrap(),
        )
        .unwrap();
        let t3 = Tensor::from_vec(vec![1.7_f64, 2.8, 4.0], (1, 3), &device).unwrap();
        let h3 = crate::rotation::rigid_to_homogeneous(&r3, &t3).unwrap();

        // Stack to [1, 3, 4, 4]
        Tensor::stack(&[&h1, &h2, &h3], 1).unwrap()
    }

}
