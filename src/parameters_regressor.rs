//! Inverse-fitting / parameter regressor.
//!
//! Port of `anny/src/anny/parameters_regressor.py`. Given a target mesh,
//! recovers `(pose, phenotype)` via an alternating fit:
//!
//! 1. Build per-bone correspondence sets from the model's skinning weights.
//! 2. Per iteration:
//!    - **Pose** = joint-wise weighted SVD-based rigid registration, refined
//!      by a global rigid alignment of the root.
//!    - **Phenotype** = finite-difference Jacobian + Tikhonov-regularised
//!      normal-equations solve `(AᵀA + λI) δ = Aᵀ b`.
//!
//! No autograd is used — Python's regressor is `@torch.no_grad()` end-to-end.

use std::collections::HashMap;

use candle_core::{D, DType, Device, Result, Tensor};
use thiserror::Error;

use crate::models::full_model::{Model, PoseParameterization};
use crate::phenotype::{PHENOTYPE_VARIATIONS, PhenotypeValues};
use crate::rotation::{rigid_points_registration, rigid_to_homogeneous};

#[derive(Debug, Error)]
pub enum RegressorError {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("config: {0}")]
    Config(String),
}

#[derive(Debug, Clone)]
pub struct RegressorOptions {
    /// Finite-difference step for the phenotype Jacobian.
    pub eps: f64,
    /// Number of vertices sub-sampled for the Jacobian (cost saver).
    pub n_points: usize,
    /// Outer iterations of (pose → phenotype) updates.
    pub max_n_iters: usize,
    /// Per-iteration cap on the absolute change of any phenotype scalar.
    pub max_delta: f64,
    /// Tikhonov regularisation weights, indexed by phenotype label. Defaults
    /// match the Python reference values.
    pub reg_weights: HashMap<String, f64>,
    /// Print per-iteration PVE to stderr.
    pub verbose: bool,
}

impl Default for RegressorOptions {
    fn default() -> Self {
        Self {
            eps: 0.1,
            n_points: 5000,
            max_n_iters: 5,
            max_delta: 0.2,
            reg_weights: default_reg_weights(),
            verbose: false,
        }
    }
}

fn default_reg_weights() -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("gender".to_string(), 1.0);
    m.insert("age".to_string(), 10.0);
    m.insert("muscle".to_string(), 1.0);
    m.insert("weight".to_string(), 1.0);
    m.insert("height".to_string(), 1e-3);
    m.insert("proportions".to_string(), 1.0);
    m.insert("cupsize".to_string(), 2.0);
    m.insert("firmness".to_string(), 2.0);
    m.insert("african".to_string(), 100.0);
    m.insert("asian".to_string(), 100.0);
    m.insert("caucasian".to_string(), 100.0);
    m
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub pose_parameters: Tensor,
    pub phenotype: PhenotypeValues,
    /// Final fitted vertices in the model's full-vertex layout `[B, V, 3]`.
    pub vertices: Tensor,
}

/// Per-bone vertex partitioning derived from the model's skinning weights.
struct Partition {
    /// `unique_ids[k]` is the vertex index in `template_vertices` for the kᵗʰ
    /// vertex used by any face. Built by the regressor as the unique vertex
    /// indices referenced by `model.faces`.
    unique_ids: Vec<u32>,
    /// `joint_vertex_sets[j]` = indices into `unique_ids` of vertices the jᵗʰ
    /// bone influences with weight ≥ 0.01.
    joint_vertex_sets: Vec<Vec<u32>>,
    /// `vertex_joint_weights[j]` = weights, normalised to sum to 1 within the
    /// joint's set. Same length as `joint_vertex_sets[j]`.
    vertex_joint_weights: Vec<Vec<f64>>,
}

pub struct Regressor<'a> {
    model: &'a Model,
    opts: RegressorOptions,
    partition: Partition,
    /// Sub-sample of `unique_ids` used for the Jacobian. `[n_points]` u32.
    sample_idx: Tensor,
    /// `[n_optim_phen]` reg-weight diag — one scalar per optimisable phenotype.
    /// Reordered at fit-time to match `optim_keys`.
    reg_weights_full: HashMap<String, f64>,
}

impl<'a> Regressor<'a> {
    pub fn new(model: &'a Model, opts: RegressorOptions) -> Result<Self> {
        // unique_ids = sorted(unique(model.faces.flatten()))
        let mut set = std::collections::BTreeSet::new();
        for f in &model.faces {
            for v in f {
                set.insert(*v);
            }
        }
        let unique_ids: Vec<u32> = set.into_iter().collect();

        let partition = build_partition(model, &unique_ids)?;

        // Sub-sample n_points points from unique_ids.
        let n_unique = unique_ids.len();
        let n = opts.n_points.min(n_unique);
        let mut sampled = Vec::with_capacity(n);
        if n > 0 {
            for i in 0..n {
                let frac = i as f64 / (n.max(1) - 1).max(1) as f64;
                let pos = (frac * (n_unique - 1) as f64).round() as usize;
                sampled.push(unique_ids[pos.min(n_unique - 1)]);
            }
        }
        let sample_idx = Tensor::from_vec(sampled, (n,), &model.device)?;

        let reg_weights_full = opts.reg_weights.clone();

        Ok(Self {
            model,
            opts,
            partition,
            sample_idx,
            reg_weights_full,
        })
    }

    /// Run the alternating fit. `target_vertices` is `[B, V, 3]` (or `[V, 3]`
    /// — promoted to a singleton batch). Phenotype keys in
    /// `excluded_phenotypes` are held constant.
    pub fn fit(&self, target_vertices: &Tensor, excluded_phenotypes: &[&str]) -> Result<FitResult> {
        self.fit_inner(target_vertices, excluded_phenotypes, None, None)
    }

    /// Like [`Self::fit`], but with an explicit initial phenotype + a per-call
    /// override of [`RegressorOptions::max_delta`]. Used by
    /// [`Self::fit_with_age_anchor_search`].
    pub fn fit_inner(
        &self,
        target_vertices: &Tensor,
        excluded_phenotypes: &[&str],
        initial_phenotype: Option<&PhenotypeValues>,
        max_delta_override: Option<f64>,
    ) -> Result<FitResult> {
        let target = if target_vertices.rank() == 2 {
            target_vertices.unsqueeze(0)?
        } else {
            target_vertices.clone()
        };
        let bs = target.dim(0)?;
        let dtype = self.model.dtype;
        let device = self.model.device.clone();
        let max_delta = max_delta_override.unwrap_or(self.opts.max_delta);

        // Phenotype init: caller-supplied, or 0.5 everywhere with age 0.7
        // (the Python default). When the caller supplies a `PhenotypeValues`
        // we promote any 1-element tensor to the requested batch size.
        let mut phen = match initial_phenotype {
            Some(p) => broadcast_phenotype(p, bs, dtype, &device)?,
            None => {
                let mut p = PhenotypeValues::defaults(dtype, &device)?;
                p.age = Tensor::from_vec(vec![0.7_f64; bs], bs, &device)?.to_dtype(dtype)?;
                broadcast_phenotype(&p, bs, dtype, &device)?
            }
        };

        // Identity pose params [bs, K, 4, 4] in root_relative_world frame.
        let n_bones = self.model.bone_count();
        let identity = Tensor::eye(4, dtype, &device)?
            .reshape((1, 1, 4, 4))?
            .broadcast_as((bs, n_bones, 4, 4))?
            .contiguous()?;

        let mut pose_params = identity.clone();

        // Initial forward.
        let mut output = self.model.forward(
            Some(&pose_params),
            &phen,
            Some(PoseParameterization::RootRelativeWorld),
        )?;
        let unique_idx_t = self.unique_idx_tensor()?;
        let mut v_ref = output.vertices.index_select(&unique_idx_t, 1)?;

        // Global root rigid alignment.
        let (r0, t0) = rigid_points_registration(&v_ref, &target, None)?;
        if self.opts.verbose {
            let initial_pve = mean_pve_mm(&v_ref, &target.index_select(&unique_idx_t, 1)?)?;
            eprintln!("initial PVE (before alignment): {initial_pve:.2} mm");
        }
        let new_root = rigid_to_homogeneous(&r0, &t0)?.unsqueeze(1)?;
        pose_params = self.replace_root(&pose_params, &new_root)?;
        output = self.model.forward(
            Some(&pose_params),
            &phen,
            Some(PoseParameterization::RootRelativeWorld),
        )?;
        v_ref = output.vertices.index_select(&unique_idx_t, 1)?;
        if self.opts.verbose {
            let after_pve = mean_pve_mm(&v_ref, &target.index_select(&unique_idx_t, 1)?)?;
            eprintln!("after global alignment: PVE = {after_pve:.2} mm");
        }

        let optim_keys: Vec<String> = self
            .model
            .phenotype_labels()
            .iter()
            .filter(|k| !excluded_phenotypes.contains(k))
            .map(|s| s.to_string())
            .collect();

        let mut b_ref = output.bone_poses.clone();

        for iter in 0..self.opts.max_n_iters {
            // 1. Pose update via joint-wise registration.
            let (new_pose, v_hat) = self.jointwise_registration(&v_ref, &target, &b_ref, &phen)?;
            pose_params = new_pose;

            if self.opts.verbose {
                let pve = mean_pve_mm(&v_hat, &target.index_select(&unique_idx_t, 1)?)?;
                eprintln!("iter {iter} after pose: PVE = {pve:.2} mm");
            }

            // 2. Phenotype update.
            if !optim_keys.is_empty() {
                let jacobian = self.compute_macro_jacobian(&pose_params, &phen)?; // [B, V'*3, n_optim]
                let b_resid = (target.index_select(&self.sample_idx, 1)?
                    - v_hat.index_select(&self.sample_idx, 1)?)?
                .reshape((bs, ()))?;
                let delta = self.tikhonov_solve(&jacobian, &b_resid, &optim_keys)?;
                self.apply_phenotype_delta(&mut phen, &delta, &optim_keys, bs, max_delta)?;
            }

            // Refresh v_ref / b_ref for next iter.
            output = self.model.forward(
                Some(&pose_params),
                &phen,
                Some(PoseParameterization::RootRelativeWorld),
            )?;
            v_ref = output.vertices.index_select(&unique_idx_t, 1)?;
            b_ref = output.bone_poses.clone();
        }

        // Convert pose to the model's default parameterization.
        let final_pose = self
            .model
            .pose_parameterization(&output, self.model.default_pose_parameterization)?;

        Ok(FitResult {
            pose_parameters: final_pose,
            phenotype: phen,
            vertices: output.vertices,
        })
    }

    // ── Internals ──────────────────────────────────────────────────────────

    fn unique_idx_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(
            self.partition.unique_ids.clone(),
            self.partition.unique_ids.len(),
            &self.model.device,
        )
    }

    fn replace_root(&self, pose_params: &Tensor, new_root: &Tensor) -> Result<Tensor> {
        let tail = pose_params.narrow(1, 1, pose_params.dim(1)? - 1)?;
        Tensor::cat(&[new_root, &tail], 1)?.contiguous()
    }

    fn jointwise_registration(
        &self,
        v_ref: &Tensor,
        v_tar: &Tensor,
        b_ref: &Tensor,
        phen: &PhenotypeValues,
    ) -> Result<(Tensor, Tensor)> {
        let bs = v_ref.dim(0)?;
        let n_bones = self.model.bone_count();
        let dtype = self.model.dtype;
        let device = self.model.device.clone();

        // For each bone with a non-empty vertex set, compute weighted Kabsch
        // on (its vertices in v_ref, in v_tar). We do this on the host
        // because the per-bone batch sizes vary.
        let v_ref_flat: Vec<f64> = v_ref.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let v_tar_flat: Vec<f64> = v_tar.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let n_unique = self.partition.unique_ids.len();

        let mut r_per_bone = vec![[[0.0_f64; 3]; 3]; bs * n_bones];
        let mut t_per_bone = vec![[0.0_f64; 3]; bs * n_bones];
        // Identity defaults.
        for slot in &mut r_per_bone {
            slot[0][0] = 1.0;
            slot[1][1] = 1.0;
            slot[2][2] = 1.0;
        }

        for j in 0..n_bones {
            let idxs = &self.partition.joint_vertex_sets[j];
            let ws = &self.partition.vertex_joint_weights[j];
            if idxs.is_empty() {
                continue;
            }
            // Compute joint position from skinned vertices, weighted.
            let mut sum_w = 0.0_f64;
            for w in ws {
                sum_w += *w;
            }
            let inv_w = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
            for bi in 0..bs {
                let mut joint_r = [0.0_f64; 3];
                let mut joint_t = [0.0_f64; 3];
                let mut xs_r = Vec::with_capacity(idxs.len() + 1);
                let mut xs_t = Vec::with_capacity(idxs.len() + 1);
                let mut weights = Vec::with_capacity(idxs.len() + 1);
                let mut max_w = 0.0_f64;
                for (i, &k) in idxs.iter().enumerate() {
                    let r_off = bi * n_unique * 3 + (k as usize) * 3;
                    let t_off = bi * n_unique * 3 + (k as usize) * 3;
                    let xr = [
                        v_ref_flat[r_off],
                        v_ref_flat[r_off + 1],
                        v_ref_flat[r_off + 2],
                    ];
                    let xt = [
                        v_tar_flat[t_off],
                        v_tar_flat[t_off + 1],
                        v_tar_flat[t_off + 2],
                    ];
                    let w = ws[i];
                    if w > max_w {
                        max_w = w;
                    }
                    for c in 0..3 {
                        joint_r[c] += w * xr[c];
                        joint_t[c] += w * xt[c];
                    }
                    xs_r.push(xr);
                    xs_t.push(xt);
                    weights.push(w);
                }
                for c in 0..3 {
                    joint_r[c] *= inv_w;
                    joint_t[c] *= inv_w;
                }
                xs_r.push(joint_r);
                xs_t.push(joint_t);
                weights.push(2.0 * max_w);

                // Solve weighted Kabsch on host. Reuse the rotation crate's
                // helper by rebuilding tensors for one bone at a time.
                let n = xs_r.len();
                let xr_flat: Vec<f64> = xs_r.iter().flatten().copied().collect();
                let xt_flat: Vec<f64> = xs_t.iter().flatten().copied().collect();
                let xr_t = Tensor::from_vec(xr_flat, (1, n, 3), &device)?.to_dtype(dtype)?;
                let xt_t = Tensor::from_vec(xt_flat, (1, n, 3), &device)?.to_dtype(dtype)?;
                let w_t = Tensor::from_vec(weights, (1, n), &device)?.to_dtype(dtype)?;
                let (r_b, t_b) = rigid_points_registration(&xr_t, &xt_t, Some(&w_t))?;
                let r_v: Vec<f64> = r_b.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
                let t_v: Vec<f64> = t_b.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
                let slot = bi * n_bones + j;
                for r in 0..3 {
                    for c in 0..3 {
                        r_per_bone[slot][r][c] = r_v[r * 3 + c];
                    }
                    t_per_bone[slot][r] = t_v[r];
                }
            }
        }

        // Pack into [B, J, 4, 4] and apply: b_tar = rigid @ b_ref.
        let mut h_flat = vec![0.0_f64; bs * n_bones * 16];
        for bi in 0..bs {
            for j in 0..n_bones {
                let slot = bi * n_bones + j;
                let off = (bi * n_bones + j) * 16;
                for r in 0..3 {
                    for c in 0..3 {
                        h_flat[off + r * 4 + c] = r_per_bone[slot][r][c];
                    }
                    h_flat[off + r * 4 + 3] = t_per_bone[slot][r];
                }
                h_flat[off + 12] = 0.0;
                h_flat[off + 13] = 0.0;
                h_flat[off + 14] = 0.0;
                h_flat[off + 15] = 1.0;
            }
        }
        let rigid_h = Tensor::from_vec(h_flat, (bs, n_bones, 4, 4), &device)?.to_dtype(dtype)?;
        let b_tar = rigid_h.matmul(b_ref)?;

        // Run model with absolute parameterization on b_tar, then convert to
        // root_relative_world.
        let abs_out =
            self.model
                .forward(Some(&b_tar), phen, Some(PoseParameterization::Absolute))?;
        let pose_root = self
            .model
            .pose_parameterization(&abs_out, PoseParameterization::RootRelativeWorld)?;
        let pose_root = sanitize_pose_parameters(&pose_root)?;

        // Reset bones with no support and the indices_identity bones — but
        // Python's `face_joints = {}` makes that set empty, so we just zero the
        // root translation slot and the per-bone translations of supported
        // joints (matching Python lines 301–307 with the empty set).
        let pose_root = self.zero_translation_for_supported_joints(&pose_root)?;

        // Re-render in root_relative_world to compute the alignment-only output.
        let neutral_out = self.model.forward(
            Some(&pose_root),
            phen,
            Some(PoseParameterization::RootRelativeWorld),
        )?;
        let (r_root, t_root) =
            rigid_points_registration(&neutral_out.vertices, &abs_out.vertices, None)?;
        let new_root = rigid_to_homogeneous(&r_root, &t_root)?.unsqueeze(1)?;
        let pose_root = self.replace_root(&pose_root, &new_root)?;

        // v_hat = neutral_vertices @ R_rootᵀ + t_root, restricted to unique_ids.
        let unique_idx_t = self.unique_idx_tensor()?;
        let neutral_unique = neutral_out.vertices.index_select(&unique_idx_t, 1)?;
        let r_root_t = r_root.transpose(D::Minus1, D::Minus2)?.contiguous()?;
        let v_hat = neutral_unique
            .matmul(&r_root_t)?
            .broadcast_add(&t_root.unsqueeze(1)?)?
            .contiguous()?;

        Ok((pose_root, v_hat))
    }

    fn zero_translation_for_supported_joints(&self, pose_root: &Tensor) -> Result<Tensor> {
        // Set translation = 0 for any bone with a non-empty vertex set, and
        // root entry = identity, mirroring Python lines 301–306.
        let bs = pose_root.dim(0)?;
        let n_bones = self.model.bone_count();
        let mut flat: Vec<f64> = pose_root.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        for bi in 0..bs {
            // Root → identity.
            let off_root = (bi * n_bones) * 16;
            for r in 0..4 {
                for c in 0..4 {
                    flat[off_root + r * 4 + c] = if r == c { 1.0 } else { 0.0 };
                }
            }
            for j in 1..n_bones {
                let off = (bi * n_bones + j) * 16;
                if self.partition.joint_vertex_sets[j].is_empty() {
                    for r in 0..4 {
                        for c in 0..4 {
                            flat[off + r * 4 + c] = if r == c { 1.0 } else { 0.0 };
                        }
                    }
                } else {
                    flat[off + 3] = 0.0;
                    flat[off + 7] = 0.0;
                    flat[off + 11] = 0.0;
                }
            }
        }
        Tensor::from_vec(flat, (bs, n_bones, 4, 4), &self.model.device)?.to_dtype(self.model.dtype)
    }

    fn compute_macro_jacobian(
        &self,
        pose_parameters: &Tensor,
        phen: &PhenotypeValues,
    ) -> Result<Tensor> {
        let bs = pose_parameters.dim(0)?;
        let n_bones = self.model.bone_count();
        let dtype = self.model.dtype;
        let device = self.model.device.clone();

        // We perturb each phenotype label in `optim_keys` (== model.phenotype_labels for
        // now — caller filters by selecting columns afterwards). The Jacobian
        // includes ALL phenotype_labels and the caller picks columns.
        let labels = self.model.phenotype_labels();
        let n_phen = labels.len();
        let n_samples = self.sample_idx.dim(0)?;

        // Baseline + per-phen perturbed forward passes, all stacked into one
        // batch of size bs * (n_phen + 1).
        let total = bs * (n_phen + 1);
        let pose_repeated = pose_parameters
            .unsqueeze(1)?
            .broadcast_as((bs, n_phen + 1, n_bones, 4, 4))?
            .reshape((total, n_bones, 4, 4))?
            .contiguous()?;
        let phen_repeated = repeat_phenotype(phen, bs, n_phen + 1)?;
        // Add eps to the i-th phenotype scalar in the i-th interleaved slot.
        let phen_perturbed = perturb_phenotype(phen_repeated, &labels, self.opts.eps)?;

        let out = self.model.forward(
            Some(&pose_repeated),
            &phen_perturbed,
            Some(PoseParameterization::RootRelativeWorld),
        )?;
        let unique_idx_t = self.unique_idx_tensor()?;
        let v_unique = out.vertices.index_select(&unique_idx_t, 1)?; // [total, V', 3]
        let v_unique = v_unique.reshape((bs, n_phen + 1, v_unique.dim(1)?, 3))?;
        // err[bi, p, v, c] = v_unique[bi, p+1, v, c] - v_unique[bi, 0, v, c]
        let baseline = v_unique.narrow(1, 0, 1)?; // [bs, 1, V', 3]
        let perturbed = v_unique.narrow(1, 1, n_phen)?; // [bs, n_phen, V', 3]
        let err = perturbed.broadcast_sub(&baseline)?;

        // Subsample.
        let err_sampled = err.index_select(&self.sample_idx, 2)?; // [bs, n_phen, n_samples, 3]
        // Reshape to [bs, n_phen, n_samples * 3] and divide by eps.
        let err_flat = err_sampled
            .reshape((bs, n_phen, n_samples * 3))?
            .affine(1.0 / self.opts.eps, 0.0)?;
        // Permute to [bs, V'*3, n_phen].
        let _ = (dtype, &device);
        err_flat.transpose(1, 2)?.contiguous()
    }

    fn tikhonov_solve(
        &self,
        jacobian: &Tensor,
        residual: &Tensor,
        optim_keys: &[String],
    ) -> Result<Tensor> {
        // Solve (AᵀA + λI) δ = Aᵀ r per batch. Use nalgebra on the host —
        // matrices are small (n_optim ≤ ~12).
        let bs = jacobian.dim(0)?;
        let _v3 = jacobian.dim(1)?;
        let labels = self.model.phenotype_labels();
        let label_to_col: HashMap<&str, usize> =
            labels.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let cols: Vec<usize> = optim_keys
            .iter()
            .map(|k| label_to_col[k.as_str()])
            .collect();

        let cols_t = Tensor::from_vec(
            cols.iter().map(|&c| c as u32).collect::<Vec<_>>(),
            cols.len(),
            &self.model.device,
        )?;
        let a = jacobian.index_select(&cols_t, 2)?; // [bs, V'*3, n_optim]
        let n_optim = a.dim(2)?;

        let a_host: Vec<f64> = a.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let r_host: Vec<f64> = residual.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        let v3 = a.dim(1)?;

        let mut delta = vec![0.0_f64; bs * n_optim];
        for bi in 0..bs {
            // A: v3 × n_optim
            let mut mat = nalgebra::DMatrix::<f64>::zeros(v3, n_optim);
            let a_off = bi * v3 * n_optim;
            for r in 0..v3 {
                for c in 0..n_optim {
                    mat[(r, c)] = a_host[a_off + r * n_optim + c];
                }
            }
            // r: v3
            let r_off = bi * v3;
            let r_vec = nalgebra::DVector::<f64>::from_iterator(
                v3,
                r_host[r_off..r_off + v3].iter().copied(),
            );
            let ata = mat.transpose() * &mat;
            let mut reg = nalgebra::DMatrix::<f64>::zeros(n_optim, n_optim);
            for (i, key) in optim_keys.iter().enumerate() {
                reg[(i, i)] = *self.reg_weights_full.get(key).unwrap_or(&1.0);
            }
            let lhs = ata + reg;
            let rhs = mat.transpose() * r_vec;
            let solved = lhs.lu().solve(&rhs);
            if let Some(d) = solved {
                for i in 0..n_optim {
                    let v = d[i];
                    delta[bi * n_optim + i] = if v.is_finite() { v } else { 0.0 };
                }
            }
        }
        Tensor::from_vec(delta, (bs, n_optim), &self.model.device)?.to_dtype(self.model.dtype)
    }

    fn apply_phenotype_delta(
        &self,
        phen: &mut PhenotypeValues,
        delta: &Tensor,
        optim_keys: &[String],
        bs: usize,
        max_delta: f64,
    ) -> Result<()> {
        let delta_host: Vec<f64> = delta.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        for (i, key) in optim_keys.iter().enumerate() {
            let mut current: Vec<f64> = phenotype_get(phen, key)?
                .to_dtype(DType::F64)?
                .flatten_all()?
                .to_vec1()?;
            for bi in 0..bs {
                let d = delta_host[bi * optim_keys.len() + i].clamp(-max_delta, max_delta);
                current[bi] = (current[bi] + d).clamp(0.01, 0.99);
            }
            let new_t =
                Tensor::from_vec(current, bs, &self.model.device)?.to_dtype(self.model.dtype)?;
            phenotype_set(phen, key, new_t)?;
        }
        Ok(())
    }

    /// Grid-searches over age anchors, picks the best per-batch-element age
    /// (and the corresponding height that the regressor settled on), then
    /// runs a final fit with both pinned. Mirrors
    /// `fit_with_age_anchor_search` (lines 453–520 of `parameters_regressor.py`).
    pub fn fit_with_age_anchor_search(
        &self,
        target_vertices: &Tensor,
        anchors: &[f64],
    ) -> Result<FitResult> {
        let target = if target_vertices.rank() == 2 {
            target_vertices.unsqueeze(0)?
        } else {
            target_vertices.clone()
        };
        let bs = target.dim(0)?;
        let dtype = self.model.dtype;
        let device = self.model.device.clone();

        let mut best_pve = vec![f64::INFINITY; bs];
        let mut best_age = vec![0.0_f64; bs];
        let mut best_height = vec![0.5_f64; bs];

        // 1. Per-anchor sweeps with age held constant at the anchor.
        for &anchor in anchors {
            let mut init = PhenotypeValues::defaults(dtype, &device)?;
            init.age = Tensor::from_vec(vec![anchor; bs], bs, &device)?.to_dtype(dtype)?;
            let result = self.fit_inner(&target, &["age"], Some(&init), None)?;
            let unique_idx_t = self.unique_idx_tensor()?;
            let pve = per_element_pve_mm(
                &result.vertices.index_select(&unique_idx_t, 1)?,
                &target.index_select(&unique_idx_t, 1)?,
            )?;
            let heights: Vec<f64> = result
                .phenotype
                .height
                .to_dtype(DType::F64)?
                .flatten_all()?
                .to_vec1()?;
            for bi in 0..bs {
                if pve[bi] < best_pve[bi] {
                    best_pve[bi] = pve[bi];
                    best_age[bi] = anchor;
                    best_height[bi] = heights[bi];
                }
            }
        }

        // 2. Final fit with age + height pinned per element, max_delta = 0.1.
        let mut final_init = PhenotypeValues::defaults(dtype, &device)?;
        final_init.age = Tensor::from_vec(best_age, bs, &device)?.to_dtype(dtype)?;
        final_init.height = Tensor::from_vec(best_height, bs, &device)?.to_dtype(dtype)?;
        self.fit_inner(&target, &[], Some(&final_init), Some(0.1))
    }
}

fn per_element_pve_mm(a: &Tensor, b: &Tensor) -> Result<Vec<f64>> {
    // a, b shape [B, V, 3]. Returns [B] of mean PVE in mm per batch element.
    let av: Vec<f64> = a.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let bv: Vec<f64> = b.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let bs = a.dim(0)?;
    let v = a.dim(1)?;
    let mut out = Vec::with_capacity(bs);
    for bi in 0..bs {
        let mut sum = 0.0_f64;
        for i in 0..v {
            let off = (bi * v + i) * 3;
            let dx = av[off] - bv[off];
            let dy = av[off + 1] - bv[off + 1];
            let dz = av[off + 2] - bv[off + 2];
            sum += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        out.push(sum / v as f64 * 1000.0);
    }
    Ok(out)
}

fn broadcast_phenotype(
    src: &PhenotypeValues,
    bs: usize,
    dtype: DType,
    device: &Device,
) -> Result<PhenotypeValues> {
    let go = |t: &Tensor| -> Result<Tensor> {
        let cur = if t.rank() == 0 {
            t.unsqueeze(0)?
        } else {
            t.clone()
        };
        let cur_b = cur.dim(0)?;
        let promoted = if cur_b == bs {
            cur
        } else if cur_b == 1 {
            cur.broadcast_as((bs,))?.contiguous()?
        } else {
            candle_core::bail!("phenotype field has incompatible batch dim {cur_b} (target {bs})");
        };
        promoted.to_dtype(dtype)?.to_device(device)
    };
    Ok(PhenotypeValues {
        age: go(&src.age)?,
        gender: go(&src.gender)?,
        muscle: go(&src.muscle)?,
        weight: go(&src.weight)?,
        height: go(&src.height)?,
        proportions: go(&src.proportions)?,
        cupsize: go(&src.cupsize)?,
        firmness: go(&src.firmness)?,
        african: go(&src.african)?,
        asian: go(&src.asian)?,
        caucasian: go(&src.caucasian)?,
    })
}

fn mean_pve_mm(a: &Tensor, b: &Tensor) -> Result<f64> {
    let av: Vec<f64> = a.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let bv: Vec<f64> = b.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let n = av.len() / 3;
    let mut sum = 0.0_f64;
    for i in 0..n {
        let dx = av[i * 3] - bv[i * 3];
        let dy = av[i * 3 + 1] - bv[i * 3 + 1];
        let dz = av[i * 3 + 2] - bv[i * 3 + 2];
        sum += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    Ok(sum / n.max(1) as f64 * 1000.0)
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_partition(model: &Model, unique_ids: &[u32]) -> Result<Partition> {
    let n_unique = unique_ids.len();
    let n_bones = model.bone_count();
    let m = model.vertex_bone_indices.dim(1)?;
    let device = &model.device;

    // Index_select bone_indices and bone_weights at unique_ids, host them.
    let unique_t = Tensor::from_vec(unique_ids.to_vec(), n_unique, device)?;
    let i_unique = model.vertex_bone_indices.index_select(&unique_t, 0)?;
    let w_unique = model.vertex_bone_weights.index_select(&unique_t, 0)?;
    let i_host: Vec<u32> = i_unique.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?;
    let w_host: Vec<f64> = w_unique.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;

    let mut joint_vertex_sets: Vec<Vec<u32>> = vec![Vec::new(); n_bones];
    let mut vertex_joint_weights: Vec<Vec<f64>> = vec![Vec::new(); n_bones];

    for v in 0..n_unique {
        for s in 0..m {
            let off = v * m + s;
            let j = i_host[off] as usize;
            let w = w_host[off];
            if w >= 0.01 && j < n_bones {
                joint_vertex_sets[j].push(v as u32);
                vertex_joint_weights[j].push(w);
            }
        }
    }
    // Normalise per-joint weights to sum to 1.
    for ws in vertex_joint_weights.iter_mut() {
        let s: f64 = ws.iter().sum();
        if s > 0.0 {
            for w in ws.iter_mut() {
                *w /= s;
            }
        }
    }

    Ok(Partition {
        unique_ids: unique_ids.to_vec(),
        joint_vertex_sets,
        vertex_joint_weights,
    })
}

fn sanitize_pose_parameters(pose: &Tensor) -> Result<Tensor> {
    // Project the 3×3 rotation block of every [B, J, 4, 4] entry back onto
    // SO(3) via SVD with a determinant-sign correction. Done on the host
    // (per-3×3 SVD via nalgebra).
    let dims = pose.dims();
    if dims.len() != 4 || dims[2] != 4 || dims[3] != 4 {
        candle_core::bail!("sanitize_pose_parameters: expected [B, J, 4, 4]");
    }
    let bs = dims[0];
    let nj = dims[1];
    let dtype = pose.dtype();
    let mut flat: Vec<f64> = pose.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    for bi in 0..bs {
        for j in 0..nj {
            let off = (bi * nj + j) * 16;
            // Read 3×3.
            let m = nalgebra::Matrix3::<f64>::new(
                flat[off],
                flat[off + 1],
                flat[off + 2],
                flat[off + 4],
                flat[off + 5],
                flat[off + 6],
                flat[off + 8],
                flat[off + 9],
                flat[off + 10],
            );
            let svd = m.svd(true, true);
            let u = svd.u.unwrap();
            let vt = svd.v_t.unwrap();
            let det = (u * vt).determinant();
            let mut corr = nalgebra::Matrix3::<f64>::identity();
            corr[(2, 2)] = det.signum();
            let r_clean = u * corr * vt;
            for r in 0..3 {
                for c in 0..3 {
                    flat[off + r * 4 + c] = r_clean[(r, c)];
                }
            }
        }
    }
    Tensor::from_vec(flat, (bs, nj, 4, 4), pose.device())?.to_dtype(dtype)
}

fn repeat_phenotype(phen: &PhenotypeValues, bs: usize, repeats: usize) -> Result<PhenotypeValues> {
    // For each scalar field, broadcast `[bs] → [bs, repeats] → [bs * repeats]`
    // (interleaved: bi=0,p=0; bi=0,p=1; ...; bi=0,p=repeats-1; bi=1,p=0; ...).
    let go = |t: &Tensor| -> Result<Tensor> {
        let cur = if t.rank() == 0 {
            t.unsqueeze(0)?
        } else {
            t.clone()
        };
        let cur_b = cur.dim(0)?;
        let expanded = if cur_b == bs {
            cur
        } else if cur_b == 1 {
            cur.broadcast_as((bs,))?.contiguous()?
        } else {
            candle_core::bail!("phenotype field has incompatible batch dim {cur_b}");
        };
        // Repeat along a new axis.
        let r = expanded
            .unsqueeze(1)?
            .broadcast_as((bs, repeats))?
            .reshape((bs * repeats,))?;
        r.contiguous()
    };
    Ok(PhenotypeValues {
        age: go(&phen.age)?,
        gender: go(&phen.gender)?,
        muscle: go(&phen.muscle)?,
        weight: go(&phen.weight)?,
        height: go(&phen.height)?,
        proportions: go(&phen.proportions)?,
        cupsize: go(&phen.cupsize)?,
        firmness: go(&phen.firmness)?,
        african: go(&phen.african)?,
        asian: go(&phen.asian)?,
        caucasian: go(&phen.caucasian)?,
    })
}

fn perturb_phenotype(
    mut phen: PhenotypeValues,
    labels: &[&'static str],
    eps: f64,
) -> Result<PhenotypeValues> {
    // Each label l at index i: add eps to row p == i + 1 (slot 0 is baseline).
    // We need the original batch (n_phen + 1)-multiple.
    let total = phen.age.dim(0)?;
    let n_phen = labels.len();
    let repeats = n_phen + 1;
    debug_assert!(total % repeats == 0);
    let bs = total / repeats;
    for (i, label) in labels.iter().enumerate() {
        let mut current: Vec<f64> = phenotype_get(&phen, label)?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1()?;
        for bi in 0..bs {
            let idx = bi * repeats + (i + 1);
            current[idx] += eps;
        }
        let new_t =
            Tensor::from_vec(current, total, phen.age.device())?.to_dtype(phen.age.dtype())?;
        phenotype_set(&mut phen, label, new_t)?;
    }
    Ok(phen)
}

fn phenotype_get<'a>(p: &'a PhenotypeValues, label: &str) -> Result<&'a Tensor> {
    let t = match label {
        "age" => &p.age,
        "gender" => &p.gender,
        "muscle" => &p.muscle,
        "weight" => &p.weight,
        "height" => &p.height,
        "proportions" => &p.proportions,
        "cupsize" => &p.cupsize,
        "firmness" => &p.firmness,
        "african" => &p.african,
        "asian" => &p.asian,
        "caucasian" => &p.caucasian,
        other => candle_core::bail!("unknown phenotype label: {other}"),
    };
    Ok(t)
}

fn phenotype_set(p: &mut PhenotypeValues, label: &str, t: Tensor) -> Result<()> {
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
        other => candle_core::bail!("unknown phenotype label: {other}"),
    }
    Ok(())
}

// suppress unused `PHENOTYPE_VARIATIONS` import — retain for future use.
#[allow(dead_code)]
const _PHENOTYPE_VARIATIONS_REFERENCED: &[(&str, &[&str])] = PHENOTYPE_VARIATIONS;
