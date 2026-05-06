//! Top-level Anny model: loads the rig, weights, base mesh and macrodetails
//! into a single struct whose `.forward()` produces posed/skinned vertices
//! from `(pose, phenotype)` inputs.
//!
//! Mirrors the union of `full_model.py` (data loading) and `rigged_model.py`
//! (forward pass). This is the central entry point most consumers will use.

use std::collections::HashSet;
use std::path::PathBuf;

use candle_core::{D, DType, Device, Result, Tensor};
use thiserror::Error;

use crate::data::obj::{self, ObjGroup};
use crate::kinematics::{self, propagation_fronts};
use crate::models::macrodetails::{self, StackedBlendShapes, default_world_transform};
use crate::models::rig::{
    self, BoneHierarchy, RigError, build_hierarchy, build_vertex_bone_matrix,
};
use crate::phenotype::{self, PhenotypeAnchors, PhenotypeValues};
use crate::rotation::{euler_to_rotmat, rigid_from_homogeneous, rigid_inverse_homogeneous};
use crate::skinning::{self, apply_linear_blendshape};

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("obj: {0}")]
    Obj(#[from] obj::ObjError),
    #[error("rig: {0}")]
    Rig(#[from] RigError),
    #[error("macrodetails: {0}")]
    Macrodetails(#[from] macrodetails::MacrodetailsError),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid configuration: {0}")]
    Config(String),
}

#[derive(Debug, Clone, Copy)]
pub enum SkinningMethod {
    /// Linear blend skinning.
    Lbs,
    /// Dual-quaternion skinning.
    Dqs,
}

#[derive(Debug, Clone, Copy)]
pub enum PoseParameterization {
    /// All `delta_transforms` are relative to rest pose; no base transform.
    RestRelative,
    /// First parameter is the absolute root pose; others are rest-relative.
    RootRelative,
    /// Root parameter's translation is absolute world position; rotation is
    /// left-multiplied onto the rest root orientation.
    RootRelativeWorld,
    /// All parameters are absolute world poses; we recover delta transforms.
    Absolute,
}

#[derive(Debug, Clone, Copy)]
pub enum Rig {
    /// `data/mpfb2/rigs/standard/{rig,weights}.default.json`. The full
    /// MakeHuman default rig.
    Default,
}

/// Mesh-topology variant. `Default` triggers `get_edited_mesh_faces` —
/// drops a small region of faces (genitals) and stitches in 14 cap quads.
/// `Makehuman` keeps the raw MakeHuman topology unchanged. Mirrors the
/// `topology="default" | "makehuman"` argument in
/// `anny/src/anny/models/full_model.py:create_model`.
///
/// Note: under `Topology::Default` the genital geometry is replaced with
/// stitched cap quads, so any morph targets under `targets/genitals/` would
/// be no-ops on the rendered mesh. Pair
/// [`ModelOptions::include_genital_morphs`] with [`Topology::Makehuman`] if
/// you actually want those morphs to drive vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Default,
    Makehuman,
}

impl Rig {
    fn rig_filename(self) -> &'static str {
        match self {
            Rig::Default => "mpfb2/rigs/standard/rig.default.json",
        }
    }
    fn weights_filename(self) -> &'static str {
        match self {
            Rig::Default => "mpfb2/rigs/standard/weights.default.json",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Build options.
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub data_root: PathBuf,
    pub rig: Rig,
    pub include_eyes: bool,
    pub include_tongue: bool,
    /// Bones to drop from the hierarchy (their children are reparented and
    /// their skinning weights are merged into the parent's).
    pub bones_to_remove: HashSet<String>,
    /// Mesh topology variant. Defaults to [`Topology::Makehuman`] (raw mesh)
    /// for backward compatibility with the existing golden fixtures. Set to
    /// [`Topology::Default`] to match Python's `create_fullbody_model()`
    /// which applies the nudity face edits.
    pub topology: Topology,
    /// Optional `[F]` boolean mask: only faces whose bit is `true` are kept.
    /// Length must match the number of faces collected from the OBJ.
    pub faces_to_keep: Option<Vec<bool>>,
    /// When true, drop vertices that no kept face references and reindex the
    /// face / blend-shape / bone-weight tensors accordingly. Mirrors
    /// `remove_unattached_vertices` in
    /// `anny/src/anny/models/full_model.py:create_model`.
    pub remove_unattached_vertices: bool,
    pub default_pose_parameterization: PoseParameterization,
    pub skinning_method: SkinningMethod,
    pub extrapolate_phenotypes: bool,
    pub all_phenotypes: bool,
    /// When `true`, include the `genitals` category from
    /// `targets/target.json` in `local_change_labels`. Defaults to `false`
    /// for parity with Python's `create_fullbody_model()`, which excludes
    /// genitals from `local_changes="all"`. Pair with
    /// [`Topology::Makehuman`] to keep the genital geometry the morphs
    /// target — under [`Topology::Default`] the cap-quad replacement makes
    /// these morphs no-ops on the rendered mesh.
    pub include_genital_morphs: bool,
    pub dtype: DType,
    pub device: Device,
}

impl ModelOptions {
    pub fn new(data_root: impl Into<PathBuf>) -> Self {
        Self {
            data_root: data_root.into(),
            rig: Rig::Default,
            include_eyes: false,
            include_tongue: false,
            bones_to_remove: HashSet::new(),
            topology: Topology::Makehuman,
            faces_to_keep: None,
            remove_unattached_vertices: false,
            default_pose_parameterization: PoseParameterization::RootRelativeWorld,
            skinning_method: SkinningMethod::Lbs,
            extrapolate_phenotypes: false,
            all_phenotypes: false,
            include_genital_morphs: false,
            dtype: DType::F64,
            device: Device::Cpu,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// The model.
// ────────────────────────────────────────────────────────────────────────────

pub struct Model {
    // ── mesh / skinning ──
    pub template_vertices: Tensor,   // [V, 3]
    pub blendshapes: Tensor,         // [C, V, 3]
    pub vertex_bone_indices: Tensor, // [V, M] u32
    pub vertex_bone_weights: Tensor, // [V, M]
    pub faces: Vec<Vec<u32>>,
    pub texture_coordinates: Vec<[f64; 2]>,
    pub face_texture_coordinate_indices: Vec<Vec<u32>>,
    /// When `remove_unattached_vertices` was set, this maps current vertex
    /// indices to their original (un-pruned) base-mesh indices. `None` when
    /// the model still uses the full base-mesh vertex array.
    pub base_mesh_vertex_indices: Option<Vec<u32>>,

    // ── bone hierarchy ──
    pub bone_labels: Vec<String>,
    pub bone_parents: Vec<i64>,
    pub propagation_fronts: Vec<(Vec<usize>, Vec<i64>)>,

    pub template_bone_heads: Tensor,    // [K, 3]
    pub template_bone_tails: Tensor,    // [K, 3]
    pub bone_heads_blendshapes: Tensor, // [C, K, 3]
    pub bone_tails_blendshapes: Tensor, // [C, K, 3]
    pub bone_rolls_rotmat: Tensor,      // [1, K, 3, 3]

    // ── phenotype machinery ──
    pub stacked_phenotype_blend_shapes_mask: Tensor, // [C_macro, 26]
    pub anchors: PhenotypeAnchors,
    pub local_change_labels: Vec<String>,
    pub extrapolate_phenotypes: bool,
    pub all_phenotypes: bool,

    // ── settings ──
    pub default_pose_parameterization: PoseParameterization,
    pub skinning_method: SkinningMethod,

    // ── constants ──
    pub(crate) y_axis: Tensor,              // [3]
    pub(crate) degenerate_rotation: Tensor, // [3, 3]
    pub dtype: DType,
    pub device: Device,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    pub vertices: Tensor,
    pub rest_vertices: Tensor,
    pub bone_poses: Tensor,
    pub bone_transforms: Tensor,
    pub rest_bone_poses: Tensor,
    pub blendshape_coeffs: Tensor,
    /// Delta transforms actually used during forward (after parameterization).
    pub delta_transforms: Tensor,
    /// Base transform (left-multiplied onto root); `None` for rest-relative
    /// and absolute parameterizations.
    pub base_transform: Option<Tensor>,
    /// The parameterization that produced this output.
    pub parameterization: PoseParameterization,
}

impl Model {
    /// Convenience: number of bones.
    pub fn bone_count(&self) -> usize {
        self.bone_labels.len()
    }

    /// Convenience: number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.template_vertices.dim(0).unwrap_or(0)
    }

    pub fn build(opts: &ModelOptions) -> std::result::Result<Self, ModelError> {
        let dtype = opts.dtype;
        let device = opts.device.clone();

        // ── Load base mesh and apply world transformation. ───────────────
        let mesh = obj::load(opts.data_root.join("mpfb2/3dobjs/base.obj"))?;
        let world_transform = default_world_transform();
        let template_vertices_world: Vec<[f64; 3]> =
            apply_world_to_vertices(&mesh.vertices, &world_transform);

        // ── Body + optional eyes/tongue faces. ───────────────────────────
        let (mut faces, mut face_tex_indices) =
            collect_body_faces(&mesh.groups, opts.include_eyes, opts.include_tongue)?;

        // Apply nudity face edit when topology = Default. Mirrors
        // get_edited_mesh_faces in full_model.py:360–420.
        if opts.topology == Topology::Default {
            (faces, face_tex_indices) = get_edited_mesh_faces(&faces, &face_tex_indices)?;
        }

        if let Some(mask) = &opts.faces_to_keep {
            if mask.len() != faces.len() {
                return Err(ModelError::Config(format!(
                    "faces_to_keep length {} ≠ face count {}",
                    mask.len(),
                    faces.len()
                )));
            }
            let mut kept = Vec::new();
            let mut kept_tex = Vec::new();
            for (i, keep) in mask.iter().enumerate() {
                if *keep {
                    kept.push(faces[i].clone());
                    kept_tex.push(face_tex_indices[i].clone());
                }
            }
            faces = kept;
            face_tex_indices = kept_tex;
        }

        // ── Rig + weights + hierarchy + vertex/bone matrix. ──────────────
        let rig_path = opts.data_root.join(opts.rig.rig_filename());
        let weights_path = opts.data_root.join(opts.rig.weights_filename());
        let rig_data = rig::load_rig_json(&rig_path)?;
        let mut weights = rig::load_weights_json(&weights_path)?;
        let hierarchy = build_hierarchy(&rig_data, &mesh, &opts.bones_to_remove, &mut weights)?;
        let vbm = build_vertex_bone_matrix(&weights, &hierarchy.labels, mesh.vertices.len());

        // ── Macrodetails + local changes. ────────────────────────────────
        let stacked = macrodetails::load_all(
            &opts.data_root,
            &template_vertices_world,
            &world_transform,
            opts.include_genital_morphs,
            dtype,
            &device,
        )?;
        let StackedBlendShapes {
            mut blendshapes,
            mask,
            n_macrodetails: _,
            local_change_labels,
        } = stacked;

        // ── Bone heads / tails (template + per-blendshape regressors). ───
        let (template_bone_heads, template_bone_tails, bone_heads_bs, bone_tails_bs) =
            build_bone_endpoints(
                &template_vertices_world,
                &blendshapes,
                &hierarchy,
                dtype,
                &device,
            )?;

        // ── Bone rolls → [1, K, 3, 3] rotation matrices via Y-axis Euler. ─
        let rolls_tensor =
            Tensor::from_vec(hierarchy.rolls.clone(), hierarchy.rolls.len(), &device)?
                .to_dtype(dtype)?
                .unsqueeze(0)?; // [1, K]
        let bone_rolls_rotmat = euler_to_rotmat('y', &rolls_tensor, false)?; // [1, K, 3, 3]

        // ── Optional vertex pruning. Done after bone heads/tails so the rig
        //    regressor indices (which reference the full 19158-vertex space)
        //    can still be resolved. ──────────────────────────────────────
        let mut template_vertices_world = template_vertices_world;
        let mut vbm = vbm;
        let mut base_mesh_vertex_indices: Option<Vec<u32>> = None;
        if opts.remove_unattached_vertices {
            let (new_t, new_bs, new_vbm, new_faces, new_indices) = prune_unattached_vertices(
                template_vertices_world,
                blendshapes,
                vbm,
                faces,
                dtype,
                &device,
            )?;
            template_vertices_world = new_t;
            blendshapes = new_bs;
            vbm = new_vbm;
            faces = new_faces;
            base_mesh_vertex_indices = Some(new_indices);
        }

        // ── Tensor views of mesh + bone matrix ───────────────────────────
        let template_vertices = vertices_to_tensor(&template_vertices_world, dtype, &device)?;
        let vertex_bone_indices =
            indices_to_tensor(&vbm.indices, vbm.max_bones_per_vertex, &device)?;
        let vertex_bone_weights =
            weights_to_tensor(&vbm.weights, vbm.max_bones_per_vertex, dtype, &device)?;

        // ── Pre-compute kinematic propagation fronts. ────────────────────
        let propagation_fronts = propagation_fronts(&hierarchy.parents);

        // ── Constants. ──
        let y_axis = Tensor::from_vec(vec![0.0_f64, 1.0, 0.0], 3, &device)?.to_dtype(dtype)?;
        let degenerate_rotation = Tensor::from_vec(
            vec![1.0_f64, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            (3, 3),
            &device,
        )?
        .to_dtype(dtype)?;

        let anchors = PhenotypeAnchors::build(dtype, &device)?;

        Ok(Model {
            template_vertices,
            blendshapes,
            vertex_bone_indices,
            vertex_bone_weights,
            faces,
            texture_coordinates: mesh.texture_coordinates,
            face_texture_coordinate_indices: face_tex_indices,
            base_mesh_vertex_indices,
            bone_labels: hierarchy.labels,
            bone_parents: hierarchy.parents,
            propagation_fronts,
            template_bone_heads,
            template_bone_tails,
            bone_heads_blendshapes: bone_heads_bs,
            bone_tails_blendshapes: bone_tails_bs,
            bone_rolls_rotmat,
            stacked_phenotype_blend_shapes_mask: mask,
            anchors,
            local_change_labels,
            extrapolate_phenotypes: opts.extrapolate_phenotypes,
            all_phenotypes: opts.all_phenotypes,
            default_pose_parameterization: opts.default_pose_parameterization,
            skinning_method: opts.skinning_method,
            y_axis,
            degenerate_rotation,
            dtype,
            device,
        })
    }

    /// Computes per-blend-shape coefficients from a [`PhenotypeValues`] using
    /// the model's mask + anchors. Local-change coefficients are zero
    /// (no fine-detail morphs applied). Use
    /// [`Self::phenotype_coefficients_with_local_changes`] to drive them.
    pub fn phenotype_coefficients(&self, values: &PhenotypeValues) -> Result<Tensor> {
        self.phenotype_coefficients_with_local_changes(values, &std::collections::HashMap::new())
    }

    /// Computes per-blend-shape coefficients including the local-change
    /// (fine-detail) morphs. `local_changes` maps a label from
    /// [`Self::local_change_labels`] to a `[B]` tensor of activation values
    /// in roughly `[-1, 1]`. Mirrors the `local_changes` kwarg of Python's
    /// `RiggedModelWithPhenotypeParameters.get_phenotype_blendshape_coefficients`.
    ///
    /// For each label `L` with value `v`:
    /// - the positive-direction blend-shape activates with weight `max(v, 0)`
    /// - the negative-direction blend-shape activates with weight `max(-v, 0)`
    ///
    /// Labels not present in the map default to `0` (no effect). Unknown
    /// labels (i.e. labels not in `self.local_change_labels`) are silently
    /// ignored, mirroring Python's `try / except KeyError` behaviour.
    pub fn phenotype_coefficients_with_local_changes(
        &self,
        values: &PhenotypeValues,
        local_changes: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let macro_coeffs = phenotype::blendshape_coefficients(
            values,
            &self.anchors,
            &self.stacked_phenotype_blend_shapes_mask,
            self.extrapolate_phenotypes,
        )?; // [B, C_macro]
        let n_local_pairs = self.local_change_labels.len();
        if n_local_pairs == 0 {
            return Ok(macro_coeffs);
        }
        let bs = macro_coeffs.dim(0)?;
        // Build a [B, 2 * n_local_pairs] activation tensor on the host.
        // For each label slot i: column 2i = max(v, 0); column 2i+1 = max(-v, 0).
        let mut local_data = vec![0.0_f64; bs * 2 * n_local_pairs];
        for (i, label) in self.local_change_labels.iter().enumerate() {
            let Some(value) = local_changes.get(label) else {
                continue;
            };
            let value = value.to_dtype(DType::F64)?.to_device(&Device::Cpu)?;
            let value_h: Vec<f64> = value.flatten_all()?.to_vec1()?;
            if value_h.len() == 1 {
                let v = value_h[0];
                let pos = v.max(0.0);
                let neg = (-v).max(0.0);
                for bi in 0..bs {
                    local_data[bi * (2 * n_local_pairs) + 2 * i] = pos;
                    local_data[bi * (2 * n_local_pairs) + 2 * i + 1] = neg;
                }
            } else if value_h.len() == bs {
                for bi in 0..bs {
                    let v = value_h[bi];
                    local_data[bi * (2 * n_local_pairs) + 2 * i] = v.max(0.0);
                    local_data[bi * (2 * n_local_pairs) + 2 * i + 1] = (-v).max(0.0);
                }
            } else {
                candle_core::bail!(
                    "local_changes['{label}'] has length {}, expected 1 or {bs}",
                    value_h.len()
                );
            }
        }
        let local_t = Tensor::from_vec(local_data, (bs, 2 * n_local_pairs), &self.device)?
            .to_dtype(self.dtype)?;
        Tensor::cat(&[&macro_coeffs, &local_t], 1)
    }

    /// Run the model. `pose_parameters` is `[B, K, 4, 4]` of delta transforms
    /// (or `None` for the rest pose). The interpretation of the root entry is
    /// controlled by `parameterization` (defaults to the model's setting).
    pub fn forward(
        &self,
        pose_parameters: Option<&Tensor>,
        phenotype: &PhenotypeValues,
        parameterization: Option<PoseParameterization>,
    ) -> Result<ForwardOutput> {
        self.forward_with_local_changes(
            pose_parameters,
            phenotype,
            &std::collections::HashMap::new(),
            parameterization,
        )
    }

    /// Same as [`Self::forward`] but additionally drives the named
    /// local-change blend-shapes. See
    /// [`Self::phenotype_coefficients_with_local_changes`] for the
    /// `local_changes` map semantics.
    pub fn forward_with_local_changes(
        &self,
        pose_parameters: Option<&Tensor>,
        phenotype: &PhenotypeValues,
        local_changes: &std::collections::HashMap<String, Tensor>,
        parameterization: Option<PoseParameterization>,
    ) -> Result<ForwardOutput> {
        let coeffs = self.phenotype_coefficients_with_local_changes(phenotype, local_changes)?;
        let bs = coeffs.dim(0)?;
        let n_bones = self.bone_count();

        // Rest bone heads / tails / poses.
        let rest_heads = apply_linear_blendshape(
            &self.template_bone_heads,
            &self.bone_heads_blendshapes,
            &coeffs,
        )?;
        let rest_tails = apply_linear_blendshape(
            &self.template_bone_tails,
            &self.bone_tails_blendshapes,
            &coeffs,
        )?;
        let rest_bone_poses = kinematics::get_bone_poses(
            &rest_heads,
            &rest_tails,
            &self.bone_rolls_rotmat,
            &self.y_axis,
            &self.degenerate_rotation,
            0.1,
        )?;

        // Delta transforms from pose_parameters; identity if None.
        let delta_transforms_raw = match pose_parameters {
            Some(p) => p.clone(),
            None => identity_pose(bs, n_bones, self.dtype, &self.device)?,
        };

        // Pose parameterization → (delta_transforms, base_transform).
        let resolved = self.resolve_parameterization(
            &rest_bone_poses,
            &delta_transforms_raw,
            parameterization.unwrap_or(self.default_pose_parameterization),
        )?;

        // Forward kinematics, unless absolute already gave us bone_transforms.
        let (bone_poses, bone_transforms) = if let Some((bt, bp)) = resolved.bone_overrides.clone()
        {
            (bp, bt)
        } else {
            let fk = kinematics::parallel_forward_kinematic(
                &self.propagation_fronts,
                &rest_bone_poses,
                &resolved.delta_transforms,
                resolved.base_transform.as_ref(),
            )?;
            (fk.poses, fk.transforms)
        };
        let delta_transforms = resolved.delta_transforms;
        let base_transform = resolved.base_transform;

        // Skin.
        let rest_vertices =
            apply_linear_blendshape(&self.template_vertices, &self.blendshapes, &coeffs)?;
        let weights = self.vertex_bone_weights.unsqueeze(0)?;
        let indices = self.vertex_bone_indices.unsqueeze(0)?;
        let vertices = match self.skinning_method {
            SkinningMethod::Lbs => skinning::linear_blend_skinning(
                &rest_vertices,
                &weights,
                &indices,
                &bone_transforms,
            )?,
            SkinningMethod::Dqs => skinning::dual_quaternion_skinning(
                &rest_vertices,
                &weights,
                &indices,
                &bone_transforms,
            )?,
        };

        Ok(ForwardOutput {
            vertices,
            rest_vertices,
            bone_poses,
            bone_transforms,
            rest_bone_poses,
            blendshape_coeffs: coeffs,
            delta_transforms,
            base_transform,
            parameterization: parameterization.unwrap_or(self.default_pose_parameterization),
        })
    }

    /// Phenotype labels in canonical order. Mirrors
    /// `RiggedModelWithPhenotypeParameters.phenotype_labels` in
    /// `phenotype.py:97`. When `all_phenotypes=false`, the race scalars and
    /// `cupsize`/`firmness` are omitted.
    pub fn phenotype_labels(&self) -> Vec<&'static str> {
        // Order: PHENOTYPE_LABELS = (every non-race feature) + race variations.
        let non_race = [
            "gender",
            "age",
            "muscle",
            "weight",
            "height",
            "proportions",
            "cupsize",
            "firmness",
        ];
        let race = ["african", "asian", "caucasian"];
        let mut out: Vec<&'static str> = non_race.to_vec();
        out.extend(race.iter().copied());
        if !self.all_phenotypes {
            // EXCLUDED_PHENOTYPES = ['cupsize', 'firmness'] + race
            out.retain(|n| {
                !matches!(
                    *n,
                    "cupsize" | "firmness" | "african" | "asian" | "caucasian"
                )
            });
        }
        out
    }

    /// Inverse pose parameterization: takes a [`ForwardOutput`] and produces
    /// the pose parameter tensor in `target` parameterization. Mirrors
    /// `RiggedModelWithLinearBlendShapes.get_pose_parameterization` in
    /// `rigged_model.py:285–310`.
    pub fn pose_parameterization(
        &self,
        out: &ForwardOutput,
        target: PoseParameterization,
    ) -> Result<Tensor> {
        match target {
            PoseParameterization::RestRelative => {
                if let Some(base) = out.base_transform.as_ref() {
                    // delta_new = inv(rest_root) @ base @ rest_root @ delta_root
                    let rest_root = out.rest_bone_poses.narrow(1, 0, 1)?.squeeze(1)?;
                    let rest_root_inv = rigid_inverse_homogeneous(&rest_root)?;
                    let root_delta = out.delta_transforms.narrow(1, 0, 1)?.squeeze(1)?;
                    let new_root = rest_root_inv
                        .matmul(base)?
                        .matmul(&rest_root)?
                        .matmul(&root_delta)?
                        .unsqueeze(1)?;
                    let tail =
                        out.delta_transforms
                            .narrow(1, 1, out.delta_transforms.dim(1)? - 1)?;
                    Tensor::cat(&[&new_root, &tail], 1)?.contiguous()
                } else {
                    Ok(out.delta_transforms.clone())
                }
            }
            PoseParameterization::RootRelative => {
                // Replace the root delta with the absolute root pose.
                let abs_root = out.bone_poses.narrow(1, 0, 1)?;
                let tail = out
                    .delta_transforms
                    .narrow(1, 1, out.delta_transforms.dim(1)? - 1)?;
                Tensor::cat(&[&abs_root, &tail], 1)?.contiguous()
            }
            PoseParameterization::RootRelativeWorld => {
                // Root entry: linear part = bone_poses[0].linear @ inv(rest_root.linear);
                //             translation  = bone_poses[0].translation.
                let abs_root = out.bone_poses.narrow(1, 0, 1)?.squeeze(1)?; // [B, 4, 4]
                let (abs_linear, abs_t) = rigid_from_homogeneous(&abs_root)?;
                let rest_root = out.rest_bone_poses.narrow(1, 0, 1)?.squeeze(1)?;
                let (rest_linear, _) = rigid_from_homogeneous(&rest_root)?;
                let rest_linear_inv = rest_linear.transpose(D::Minus1, D::Minus2)?.contiguous()?;
                let new_linear = abs_linear.matmul(&rest_linear_inv)?;
                let new_root =
                    crate::rotation::rigid_to_homogeneous(&new_linear, &abs_t)?.unsqueeze(1)?;
                let tail = out
                    .delta_transforms
                    .narrow(1, 1, out.delta_transforms.dim(1)? - 1)?;
                Tensor::cat(&[&new_root, &tail], 1)?.contiguous()
            }
            PoseParameterization::Absolute => Ok(out.bone_poses.clone()),
        }
    }

    fn resolve_parameterization(
        &self,
        rest_bone_poses: &Tensor,
        delta_transforms: &Tensor,
        mode: PoseParameterization,
    ) -> Result<ResolvedParameterization> {
        match mode {
            PoseParameterization::RestRelative => Ok(ResolvedParameterization {
                delta_transforms: delta_transforms.clone(),
                base_transform: None,
                bone_overrides: None,
            }),
            PoseParameterization::RootRelative => {
                let rest_root = rest_bone_poses.narrow(1, 0, 1)?.squeeze(1)?; // [B, 4, 4]
                let base = rigid_inverse_homogeneous(&rest_root)?;
                Ok(ResolvedParameterization {
                    delta_transforms: delta_transforms.clone(),
                    base_transform: Some(base),
                    bone_overrides: None,
                })
            }
            PoseParameterization::RootRelativeWorld => {
                let rest_root = rest_bone_poses.narrow(1, 0, 1)?.squeeze(1)?; // [B, 4, 4]
                let base = rigid_inverse_homogeneous(&rest_root)?;
                let root_param = delta_transforms.narrow(1, 0, 1)?.squeeze(1)?; // [B, 4, 4]
                let (rest_linear, _) = rigid_from_homogeneous(&rest_root)?;
                let rest_rot_only = embed_rotation_only(&rest_linear)?;
                let new_root = root_param.matmul(&rest_rot_only)?;
                let new_root_4d = new_root.unsqueeze(1)?; // [B, 1, 4, 4]
                let tail = delta_transforms.narrow(1, 1, delta_transforms.dim(1)? - 1)?;
                let delta_new = Tensor::cat(&[&new_root_4d, &tail], 1)?;
                Ok(ResolvedParameterization {
                    delta_transforms: delta_new.contiguous()?,
                    base_transform: Some(base),
                    bone_overrides: None,
                })
            }
            PoseParameterization::Absolute => {
                // bone_poses = input (already absolute).
                // bone_transforms = bone_poses @ inv(rest_bone_poses).
                // delta_transforms[i] = inv(rest[i]) @ inv(parent_transform[i]) @ bone_poses[i]
                //                       — recomputed so downstream parameterization
                //                       conversions see proper rest-relative deltas.
                let rest_inv = rigid_inverse_homogeneous(rest_bone_poses)?;
                let bone_transforms = delta_transforms.matmul(&rest_inv)?.contiguous()?;
                let device = delta_transforms.device();
                let dtype = delta_transforms.dtype();
                let parent_bone_transforms =
                    build_parent_transforms(&bone_transforms, &self.bone_parents, dtype, device)?;
                let parent_inv = rigid_inverse_homogeneous(&parent_bone_transforms)?;
                let recomputed_delta = rest_inv
                    .matmul(&parent_inv)?
                    .matmul(delta_transforms)?
                    .contiguous()?;
                Ok(ResolvedParameterization {
                    delta_transforms: recomputed_delta,
                    base_transform: None,
                    bone_overrides: Some((bone_transforms, delta_transforms.clone())),
                })
            }
        }
    }
}

/// Output of `resolve_parameterization`. `bone_overrides`, when present, is
/// `(bone_transforms, bone_poses)` and signals that the FK chain has already
/// produced these — used by the absolute parameterization, which interprets
/// its input as already-resolved bone poses.
struct ResolvedParameterization {
    delta_transforms: Tensor,
    base_transform: Option<Tensor>,
    bone_overrides: Option<(Tensor, Tensor)>,
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn apply_world_to_vertices(
    verts: &[[f64; 3]],
    m: &macrodetails::WorldTransform3x3,
) -> Vec<[f64; 3]> {
    verts
        .iter()
        .map(|v| {
            [
                m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
            ]
        })
        .collect()
}

fn collect_body_faces(
    groups: &std::collections::BTreeMap<String, ObjGroup>,
    eyes: bool,
    tongue: bool,
) -> std::result::Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), ModelError> {
    let body = groups
        .get("body")
        .ok_or_else(|| ModelError::Config("base mesh has no 'body' group".into()))?;
    let mut faces = body.face_vertex_indices.clone();
    let mut tex_indices = body.face_texture_coordinate_indices.clone();
    if eyes {
        for name in &["helper-l-eye", "helper-r-eye"] {
            if let Some(g) = groups.get(*name) {
                faces.extend(g.face_vertex_indices.clone());
                tex_indices.extend(g.face_texture_coordinate_indices.clone());
            }
        }
    }
    if tongue && let Some(g) = groups.get("helper-tongue") {
        faces.extend(g.face_vertex_indices.clone());
        tex_indices.extend(g.face_texture_coordinate_indices.clone());
    }
    Ok((faces, tex_indices))
}

fn build_bone_endpoints(
    template: &[[f64; 3]],
    blendshapes: &Tensor,
    h: &BoneHierarchy,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let k = h.labels.len();
    let bs_dim = blendshapes.dim(0)?;
    let v_dim = blendshapes.dim(1)?;

    // Pull blendshapes to host for averaging — host iteration is friendlier
    // than candle's index_select on awkward index lists.
    let bs_flat: Vec<f64> = blendshapes
        .to_dtype(DType::F64)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let mut head_template = vec![[0.0_f64; 3]; k];
    let mut tail_template = vec![[0.0_f64; 3]; k];
    let mut head_bs: Vec<f64> = vec![0.0_f64; bs_dim * k * 3];
    let mut tail_bs: Vec<f64> = vec![0.0_f64; bs_dim * k * 3];

    for (bone_id, idxs) in h.head_regressors.iter().enumerate() {
        let mean = mean_vertices(template, idxs);
        head_template[bone_id] = mean;
        for c in 0..bs_dim {
            let avg = mean_blendshape(&bs_flat, c, v_dim, idxs);
            for j in 0..3 {
                head_bs[(c * k + bone_id) * 3 + j] = avg[j];
            }
        }
    }
    for (bone_id, idxs) in h.tail_regressors.iter().enumerate() {
        let mean = mean_vertices(template, idxs);
        tail_template[bone_id] = mean;
        for c in 0..bs_dim {
            let avg = mean_blendshape(&bs_flat, c, v_dim, idxs);
            for j in 0..3 {
                tail_bs[(c * k + bone_id) * 3 + j] = avg[j];
            }
        }
    }

    let head_t = endpoints_to_tensor(&head_template, dtype, device)?;
    let tail_t = endpoints_to_tensor(&tail_template, dtype, device)?;
    let head_b = Tensor::from_vec(head_bs, (bs_dim, k, 3), device)?.to_dtype(dtype)?;
    let tail_b = Tensor::from_vec(tail_bs, (bs_dim, k, 3), device)?.to_dtype(dtype)?;
    Ok((head_t, tail_t, head_b, tail_b))
}

fn mean_vertices(verts: &[[f64; 3]], idxs: &[u32]) -> [f64; 3] {
    if idxs.is_empty() {
        return [0.0; 3];
    }
    let mut sum = [0.0_f64; 3];
    for &i in idxs {
        let v = &verts[i as usize];
        for j in 0..3 {
            sum[j] += v[j];
        }
    }
    let n = idxs.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

fn mean_blendshape(bs_flat: &[f64], c: usize, v: usize, idxs: &[u32]) -> [f64; 3] {
    if idxs.is_empty() {
        return [0.0; 3];
    }
    let stride = v * 3;
    let base = c * stride;
    let mut sum = [0.0_f64; 3];
    for &i in idxs {
        let off = base + (i as usize) * 3;
        sum[0] += bs_flat[off];
        sum[1] += bs_flat[off + 1];
        sum[2] += bs_flat[off + 2];
    }
    let n = idxs.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

fn endpoints_to_tensor(verts: &[[f64; 3]], dtype: DType, device: &Device) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(verts.len() * 3);
    for v in verts {
        flat.extend_from_slice(v);
    }
    Tensor::from_vec(flat, (verts.len(), 3), device)?.to_dtype(dtype)
}

fn vertices_to_tensor(verts: &[[f64; 3]], dtype: DType, device: &Device) -> Result<Tensor> {
    endpoints_to_tensor(verts, dtype, device)
}

fn indices_to_tensor(rows: &[Vec<u32>], width: usize, device: &Device) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(rows.len() * width);
    for r in rows {
        flat.extend_from_slice(r);
    }
    Tensor::from_vec(flat, (rows.len(), width), device)
}

fn weights_to_tensor(
    rows: &[Vec<f64>],
    width: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut flat = Vec::with_capacity(rows.len() * width);
    for r in rows {
        flat.extend_from_slice(r);
    }
    Tensor::from_vec(flat, (rows.len(), width), device)?.to_dtype(dtype)
}

fn identity_pose(bs: usize, n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let eye = Tensor::eye(4, dtype, device)?;
    eye.reshape((1, 1, 4, 4))?
        .broadcast_as((bs, n, 4, 4))?
        .contiguous()
}

/// Drops vertices not referenced by any kept face, then remaps face vertex
/// indices, the per-vertex bone matrix, and the blend-shape vertex axis to
/// the new compact indexing. Returns the pruned `(template, blendshapes,
/// vbm, faces, base_mesh_vertex_indices)` where the last item is the list of
/// original indices kept (sorted).
fn prune_unattached_vertices(
    template_vertices: Vec<[f64; 3]>,
    blendshapes: Tensor,
    vbm: crate::models::rig::VertexBoneMatrix,
    faces: Vec<Vec<u32>>,
    dtype: DType,
    device: &Device,
) -> Result<(
    Vec<[f64; 3]>,
    Tensor,
    crate::models::rig::VertexBoneMatrix,
    Vec<Vec<u32>>,
    Vec<u32>,
)> {
    let n_orig = template_vertices.len();

    let mut kept_set = std::collections::BTreeSet::<u32>::new();
    for face in &faces {
        for v in face {
            kept_set.insert(*v);
        }
    }
    let kept: Vec<u32> = kept_set.into_iter().collect();
    let mut old_to_new = vec![-1_i64; n_orig];
    for (new_i, &old) in kept.iter().enumerate() {
        old_to_new[old as usize] = new_i as i64;
    }

    let new_template: Vec<[f64; 3]> = kept
        .iter()
        .map(|&i| template_vertices[i as usize])
        .collect();
    let new_indices: Vec<Vec<u32>> = kept
        .iter()
        .map(|&i| vbm.indices[i as usize].clone())
        .collect();
    let new_weights: Vec<Vec<f64>> = kept
        .iter()
        .map(|&i| vbm.weights[i as usize].clone())
        .collect();
    let new_vbm = crate::models::rig::VertexBoneMatrix {
        indices: new_indices,
        weights: new_weights,
        max_bones_per_vertex: vbm.max_bones_per_vertex,
    };

    let new_faces: Vec<Vec<u32>> = faces
        .iter()
        .map(|face| {
            face.iter()
                .map(|&v| {
                    let mapped = old_to_new[v as usize];
                    debug_assert!(mapped >= 0, "face references pruned vertex {v}");
                    mapped as u32
                })
                .collect()
        })
        .collect();

    // Subset the [C, V, 3] blend-shape tensor on the V axis.
    let kept_t = Tensor::from_vec(kept.clone(), kept.len(), device)?;
    let new_blendshapes = blendshapes
        .index_select(&kept_t, 1)?
        .to_dtype(dtype)?
        .contiguous()?;

    Ok((new_template, new_blendshapes, new_vbm, new_faces, kept))
}

/// Drops the 32 vertices forming the genital region of the MakeHuman base
/// mesh, then stitches in 14 cap quads (7 per side). Mirrors
/// `get_edited_mesh_faces` in `full_model.py:360–420`.
///
/// Returns `(filtered_faces, filtered_texture_indices)` in face order:
/// kept-faces, then 7 left caps, then 7 right caps.
pub fn get_edited_mesh_faces(
    faces: &[Vec<u32>],
    face_tex_indices: &[Vec<u32>],
) -> std::result::Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), ModelError> {
    use std::collections::HashMap;

    let discard_l: std::ops::Range<u32> = 1778..1794;
    let discard_r: std::ops::Range<u32> = 8450..8466;
    let is_discarded = |v: u32| -> bool { discard_l.contains(&v) || discard_r.contains(&v) };

    let mut faces_kept: Vec<Vec<u32>> = Vec::with_capacity(faces.len());
    let mut tex_kept: Vec<Vec<u32>> = Vec::with_capacity(faces.len());
    let mut ignored_faces: Vec<usize> = Vec::new();
    for (i, face) in faces.iter().enumerate() {
        if face.iter().any(|v| is_discarded(*v)) {
            ignored_faces.push(i);
        } else {
            faces_kept.push(face.clone());
            tex_kept.push(face_tex_indices[i].clone());
        }
    }

    // vertex_id → uv_id, harvested from the discarded faces.
    let mut vertex_to_uv: HashMap<u32, u32> = HashMap::new();
    for &face_id in &ignored_faces {
        let face = &faces[face_id];
        let tex = &face_tex_indices[face_id];
        for (vid, uvid) in face.iter().zip(tex.iter()) {
            match vertex_to_uv.get(vid) {
                Some(&existing) if existing != *uvid => {
                    return Err(ModelError::Config(format!(
                        "vertex {vid} has inconsistent texture coordinates {existing} vs {uvid}",
                    )));
                }
                _ => {
                    vertex_to_uv.insert(*vid, *uvid);
                }
            }
        }
    }

    // 7 left + 7 right cap quads, indices verbatim from full_model.py:390–409.
    const CAP_L: [[u32; 4]; 7] = [
        [8437, 8438, 8439, 8440],
        [8436, 8437, 8440, 8441],
        [8435, 8436, 8441, 8442],
        [8434, 8435, 8442, 8443],
        [8449, 8434, 8443, 8444],
        [8448, 8449, 8444, 8445],
        [8447, 8448, 8445, 8446],
    ];
    const CAP_R: [[u32; 4]; 7] = [
        [1762, 1771, 1770, 1763],
        [1763, 1770, 1769, 1764],
        [1764, 1769, 1768, 1765],
        [1765, 1768, 1767, 1766],
        [1762, 1777, 1772, 1771],
        [1777, 1776, 1773, 1772],
        [1776, 1775, 1774, 1773],
    ];

    for cap_set in [&CAP_L[..], &CAP_R[..]] {
        for cap in cap_set {
            if cap.iter().any(|v| is_discarded(*v)) {
                return Err(ModelError::Config(
                    "cap face references a discarded vertex — index table is wrong".into(),
                ));
            }
            let face_vec: Vec<u32> = cap.to_vec();
            let tex_vec: Vec<u32> = cap
                .iter()
                .map(|v| {
                    *vertex_to_uv
                        .get(v)
                        .expect("cap vertex must appear in a discarded face")
                })
                .collect();
            faces_kept.push(face_vec);
            tex_kept.push(tex_vec);
        }
    }

    Ok((faces_kept, tex_kept))
}

fn build_parent_transforms(
    bone_transforms: &Tensor,
    bone_parents: &[i64],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // For each bone i with parent p ∈ [0, K), gather bone_transforms[:, p, :, :].
    // Use identity for the root (parent == -1).
    let bs = bone_transforms.dim(0)?;
    let n_bones = bone_parents.len();
    // Build a parent-index lookup; for root, gather slot 0 then overwrite with identity.
    let idx: Vec<u32> = bone_parents
        .iter()
        .map(|p| if *p < 0 { 0 } else { *p as u32 })
        .collect();
    let idx_t = Tensor::from_vec(idx, n_bones, device)?;
    let gathered = bone_transforms.index_select(&idx_t, 1)?.contiguous()?; // [bs, K, 4, 4]
    // Overwrite root entry with identity for every batch element.
    let mut flat: Vec<f64> = gathered.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    for bi in 0..bs {
        let off = (bi * n_bones) * 16;
        for r in 0..4 {
            for c in 0..4 {
                flat[off + r * 4 + c] = if r == c { 1.0 } else { 0.0 };
            }
        }
    }
    Tensor::from_vec(flat, (bs, n_bones, 4, 4), device)?.to_dtype(dtype)
}

fn embed_rotation_only(linear: &Tensor) -> Result<Tensor> {
    // [B, 3, 3] rotation → [B, 4, 4] homogeneous with translation = 0.
    let dims = linear.dims();
    let n = dims.len();
    let leading: Vec<usize> = dims.iter().take(n - 2).copied().collect();
    let mut t_shape = leading.clone();
    t_shape.push(3);
    let zero_t = Tensor::zeros(t_shape, linear.dtype(), linear.device())?;
    crate::rotation::rigid_to_homogeneous(linear, &zero_t)
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
    #[ignore = "loads ~650 .target.gz files; release-only"]
    fn full_model_builds_and_runs_forward() {
        let opts = ModelOptions::new(data_root());
        let model = Model::build(&opts).expect("model build");
        assert!(model.bone_count() > 100, "expected many bones");
        assert!(model.vertex_count() > 18_000, "expected many vertices");

        let phenotype = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
        let out = model.forward(None, &phenotype, None).unwrap();
        assert_eq!(out.vertices.dim(0).unwrap(), 1);
        assert_eq!(out.vertices.dim(1).unwrap(), model.vertex_count());
        assert_eq!(out.vertices.dim(2).unwrap(), 3);

        let v: Vec<f64> = out.vertices.flatten_all().unwrap().to_vec1().unwrap();
        // No NaN, no Inf.
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite vertex coords");
        // Reasonable human-sized bounds (metres).
        let (min_v, max_v) = v
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        assert!(
            min_v > -3.0 && max_v < 3.0,
            "vertices out of plausible human range: [{min_v}, {max_v}]"
        );
        // The body is taller than it is wide; the Z-extent should exceed the
        // X-extent (Z is up after world transformation).
        let stride = 3;
        let n_verts = v.len() / stride;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        for i in 0..n_verts {
            let x = v[i * stride];
            let z = v[i * stride + 2];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }
        let height = z_max - z_min;
        let width = x_max - x_min;
        assert!(height > 1.0, "height {height} unexpectedly small");
        assert!(
            height > width,
            "height {height} should exceed width {width}"
        );
    }

    #[test]
    #[ignore = "loads ~650 .target.gz files; release-only"]
    fn rest_pose_matches_rest_vertices() {
        // With identity pose parameters and rest_relative parameterization,
        // forward() should return rest_vertices unchanged (within FK rounding).
        let mut opts = ModelOptions::new(data_root());
        opts.default_pose_parameterization = PoseParameterization::RestRelative;
        let model = Model::build(&opts).expect("model build");
        let phenotype = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
        let out = model.forward(None, &phenotype, None).unwrap();
        let rest: Vec<f64> = out.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();
        let posed: Vec<f64> = out.vertices.flatten_all().unwrap().to_vec1().unwrap();
        let max_err = rest
            .iter()
            .zip(posed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-9, "rest pose max err = {max_err}");
    }
}
