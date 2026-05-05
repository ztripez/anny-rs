//! Retopology: rebuild a [`Model`] over a different mesh topology by
//! barycentrically interpolating the reference Anny model's vertices,
//! blendshapes, and skinning weights. Mirrors
//! `anny/src/anny/models/retopology.py`.
//!
//! Two paths:
//!
//! * **SMPL-X** ([`create_smplx_topology_model`]) reads the
//!   `(face_id, u, v)` per-target-vertex map from a pre-computed
//!   safetensors file (Python's pickle list-of-tensors layout is hostile
//!   to candle's reader, hence the conversion step in
//!   [`tests/fixtures/convert_smplx.py`](../../../tests/fixtures/convert_smplx.py)).
//!   The data file itself is non-commercial and downloaded via the
//!   `smplx-download` cargo feature.
//! * **Alternative topology** ([`create_alternative_topology_model`])
//!   computes the per-target-vertex map on the fly using a
//!   closest-point-on-triangle search ([`crate::utils::mesh`]). Targets
//!   are the bundled `data/topology/*.obj` reference meshes; no network
//!   download required.

use std::path::Path;

use candle_core::{DType, Tensor};
use thiserror::Error;

use crate::data::obj;
use crate::models::full_model::{Model, ModelError, ModelOptions};
use crate::utils::mesh::{point_to_mesh_distance_and_face_uvs, triangulate_faces};

#[derive(Debug, Error)]
pub enum RetopologyError {
    #[error("model: {0}")]
    Model(#[from] ModelError),
    #[error("obj: {0}")]
    Obj(#[from] obj::ObjError),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error(
        "missing converted SMPL-X data at {0}; run `tests/fixtures/convert_smplx.py` after `download-smplx`"
    )]
    MissingConverted(std::path::PathBuf),
    #[error(
        "alternative topology mesh has vertices too far from the reference (max {0} m); the topology .obj is misaligned"
    )]
    AltTopologyTooFar(f64),
    #[error("topology .obj has no `noname` group: {0}")]
    AltTopologyMissingGroup(std::path::PathBuf),
}

/// Returns the on-disk path of the converted (safetensors) SMPL-X retopology
/// weights. The Python conversion script writes here.
pub fn smplx_safetensors_path() -> std::path::PathBuf {
    crate::paths::cache_dir().join("noncommercial/anny2smplx.safetensors")
}

/// Builds an SMPL-X-topology Anny model. Reads `anny2smplx.safetensors` from
/// the cache; downloads + asks the user to convert if missing.
///
/// `opts` is used for the underlying fullbody model build. `eyes` is forced
/// to `true` and `tongue` to `false` to match Python's semantics.
pub fn create_smplx_topology_model(opts: &ModelOptions) -> Result<Model, RetopologyError> {
    let safetensors = smplx_safetensors_path();
    if !safetensors.exists() {
        return Err(RetopologyError::MissingConverted(safetensors));
    }

    let mut ref_opts = opts.clone();
    ref_opts.include_eyes = true;
    ref_opts.include_tongue = false;
    let reference = Model::build(&ref_opts)?;

    let (vertices, faces, vertex_bone_weights, vertex_bone_indices, blendshapes) =
        load_and_remap(&safetensors, &reference)?;

    Ok(build_retopologised(
        reference,
        vertices,
        faces,
        vertex_bone_weights,
        vertex_bone_indices,
        blendshapes,
    ))
}

// ── Core remap ───────────────────────────────────────────────────────────

fn load_and_remap(
    safetensors_path: &Path,
    reference: &Model,
) -> Result<(Tensor, Vec<Vec<u32>>, Tensor, Tensor, Tensor), RetopologyError> {
    let device = reference.device.clone();

    let store = candle_core::safetensors::load(safetensors_path, &device)?;
    // bary: [3, V_smplx], indices: [V_smplx, 3], dst_faces: [F, 3]
    let bary_h: Vec<f64> = store
        .get("barycentric")
        .ok_or_else(|| candle_core::Error::Msg("safetensors missing 'barycentric'".to_string()))?
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1()?;
    let indices_h: Vec<u32> = store
        .get("vertex_indices")
        .ok_or_else(|| candle_core::Error::Msg("safetensors missing 'vertex_indices'".to_string()))?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1()?;
    let dst_faces_h: Vec<u32> = store
        .get("dst_faces")
        .ok_or_else(|| candle_core::Error::Msg("safetensors missing 'dst_faces'".to_string()))?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1()?;

    let v_target = indices_h.len() / 3;
    let n_faces = dst_faces_h.len() / 3;
    let mut faces_out: Vec<[u32; 3]> = Vec::with_capacity(n_faces);
    for i in 0..n_faces {
        faces_out.push([
            dst_faces_h[i * 3],
            dst_faces_h[i * 3 + 1],
            dst_faces_h[i * 3 + 2],
        ]);
    }

    interpolate_topology(reference, &bary_h, &indices_h, v_target, &faces_out)
}

/// Per-target-vertex barycentric remap. `bary` is `[3, V_target]` row-major,
/// `indices` is `[V_target, 3]` of source-vertex indices into the reference,
/// `faces` is the triangulated face list of the target topology.
fn interpolate_topology(
    reference: &Model,
    bary_h: &[f64],
    indices_h: &[u32],
    v_smplx: usize,
    dst_faces: &[[u32; 3]],
) -> Result<(Tensor, Vec<Vec<u32>>, Tensor, Tensor, Tensor), RetopologyError> {
    let device = reference.device.clone();
    let dtype = reference.dtype;
    debug_assert_eq!(bary_h.len(), 3 * v_smplx);
    debug_assert_eq!(indices_h.len(), v_smplx * 3);
    let template_h: Vec<f64> = reference
        .template_vertices
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1()?;
    let blendshapes_h: Vec<f64> = reference
        .blendshapes
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1()?;
    let n_blends = reference.blendshapes.dim(0)?;
    let v_anny = reference.template_vertices.dim(0)?;
    let m = reference.vertex_bone_indices.dim(1)?;
    let n_bones = reference.bone_count();
    let weights_h: Vec<f64> = reference
        .vertex_bone_weights
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1()?;
    let bone_idx_h: Vec<u32> = reference
        .vertex_bone_indices
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1()?;

    // 1. Interpolated vertices: [V_smplx, 3].
    let mut new_vertices = vec![0.0_f64; v_smplx * 3];
    for i in 0..v_smplx {
        for k in 0..3 {
            let src = indices_h[i * 3 + k] as usize;
            let w = bary_h[k * v_smplx + i];
            for c in 0..3 {
                new_vertices[i * 3 + c] += w * template_h[src * 3 + c];
            }
        }
    }

    // 2. Interpolated blendshapes: [C, V_smplx, 3].
    let mut new_blends = vec![0.0_f64; n_blends * v_smplx * 3];
    let stride_c = v_anny * 3;
    let new_stride_c = v_smplx * 3;
    for i in 0..v_smplx {
        for k in 0..3 {
            let src = indices_h[i * 3 + k] as usize;
            let w = bary_h[k * v_smplx + i];
            for c in 0..n_blends {
                let base_old = c * stride_c + src * 3;
                let base_new = c * new_stride_c + i * 3;
                new_blends[base_new] += w * blendshapes_h[base_old];
                new_blends[base_new + 1] += w * blendshapes_h[base_old + 1];
                new_blends[base_new + 2] += w * blendshapes_h[base_old + 2];
            }
        }
    }

    // 3. Interpolated bone weights, then renormalise.
    // For each new vertex: aggregate weights from each of the three source
    // vertices; sum contributions per bone via a Vec<f64> of length n_bones.
    let mut new_vw_rows: Vec<Vec<u32>> = Vec::with_capacity(v_smplx);
    let mut new_w_rows: Vec<Vec<f64>> = Vec::with_capacity(v_smplx);
    let mut max_bones_per_vertex = 0;
    let mut acc = vec![0.0_f64; n_bones];
    for i in 0..v_smplx {
        acc.fill(0.0);
        for k in 0..3 {
            let src = indices_h[i * 3 + k] as usize;
            let coeff = bary_h[k * v_smplx + i];
            for s in 0..m {
                let off = src * m + s;
                let bone = bone_idx_h[off] as usize;
                let w = weights_h[off];
                if bone < n_bones {
                    acc[bone] += coeff * w;
                }
            }
        }
        let mut indices_row = Vec::new();
        let mut weights_row = Vec::new();
        for (bone, w) in acc.iter().enumerate() {
            if *w > 0.0 {
                indices_row.push(bone as u32);
                weights_row.push(*w);
            }
        }
        if indices_row.len() > max_bones_per_vertex {
            max_bones_per_vertex = indices_row.len();
        }
        new_vw_rows.push(indices_row);
        new_w_rows.push(weights_row);
    }
    // Pad + normalise rows.
    for (idx_row, w_row) in new_vw_rows.iter_mut().zip(new_w_rows.iter_mut()) {
        let s: f64 = w_row.iter().sum();
        if s > 0.0 {
            for w in w_row.iter_mut() {
                *w /= s;
            }
        }
        while idx_row.len() < max_bones_per_vertex {
            idx_row.push(0);
            w_row.push(0.0);
        }
    }

    // Pack into tensors.
    let new_vertices_t = Tensor::from_vec(new_vertices, (v_smplx, 3), &device)?.to_dtype(dtype)?;
    let new_blends_t =
        Tensor::from_vec(new_blends, (n_blends, v_smplx, 3), &device)?.to_dtype(dtype)?;
    let mut idx_flat = Vec::with_capacity(v_smplx * max_bones_per_vertex);
    for r in &new_vw_rows {
        idx_flat.extend_from_slice(r);
    }
    let mut w_flat = Vec::with_capacity(v_smplx * max_bones_per_vertex);
    for r in &new_w_rows {
        w_flat.extend_from_slice(r);
    }
    let new_indices_t = Tensor::from_vec(idx_flat, (v_smplx, max_bones_per_vertex), &device)?;
    let new_weights_t =
        Tensor::from_vec(w_flat, (v_smplx, max_bones_per_vertex), &device)?.to_dtype(dtype)?;

    // Faces: convert [F, 3] tri layout to Vec<Vec<u32>> for Model storage.
    let faces_out: Vec<Vec<u32>> = dst_faces
        .iter()
        .map(|tri| vec![tri[0], tri[1], tri[2]])
        .collect();

    Ok((
        new_vertices_t,
        faces_out,
        new_weights_t,
        new_indices_t,
        new_blends_t,
    ))
}

// ── Alternative topology (custom decimated meshes via closest-triangle). ─

/// Builds a model on a non-SMPL-X target topology by closest-triangle
/// barycentric remap. Mirrors `create_alternative_topology_model` in
/// `anny/src/anny/models/retopology.py:121–185`.
///
/// `topology_name` is the filename stem under
/// `<data_root>/mpfb2/../topology/`, e.g. `"notoes"`,
/// `"notoes_collapse10pc"`. The reference mesh is `topology/default.obj`
/// (vendored).
///
/// The reference Anny model is built with `eyes=false, tongue=false`,
/// `remove_unattached_vertices=false` (the closest-point search uses the
/// full reference vertex array).
pub fn create_alternative_topology_model(
    opts: &ModelOptions,
    topology_name: &str,
) -> Result<Model, RetopologyError> {
    let mut ref_opts = opts.clone();
    ref_opts.include_eyes = false;
    ref_opts.include_tongue = false;
    ref_opts.remove_unattached_vertices = false;
    let reference = Model::build(&ref_opts)?;

    // Reference + target meshes from the topology directory. Apply a 90°
    // X-axis rotation to bring them into the same world frame as the
    // reference Anny model (whose template was rotated identically by
    // `default_world_transform()`, but with an additional 0.1× scaling
    // because the macrodetails .obj uses decimeters).
    let topo_dir = opts.data_root.join("topology");
    let ref_mesh = obj::load(topo_dir.join("default.obj"))?;
    let tgt_mesh = obj::load(topo_dir.join(format!("{topology_name}.obj")))?;

    let rotate_y_to_z = |v: &[f64; 3]| -> [f64; 3] { [v[0], -v[2], v[1]] };
    let ref_vertices: Vec<[f64; 3]> = ref_mesh.vertices.iter().map(rotate_y_to_z).collect();
    let tgt_vertices: Vec<[f64; 3]> = tgt_mesh.vertices.iter().map(rotate_y_to_z).collect();

    // The OBJ files have a single unnamed group; the loader stores it under
    // the literal `"noname"` key (see data/obj.rs).
    let ref_group_name = ref_mesh
        .group_order
        .first()
        .cloned()
        .or_else(|| ref_mesh.groups.keys().next().cloned())
        .ok_or_else(|| RetopologyError::AltTopologyMissingGroup(topo_dir.join("default.obj")))?;
    let tgt_group_name = tgt_mesh
        .group_order
        .first()
        .cloned()
        .or_else(|| tgt_mesh.groups.keys().next().cloned())
        .ok_or_else(|| {
            RetopologyError::AltTopologyMissingGroup(topo_dir.join(format!("{topology_name}.obj")))
        })?;
    let ref_faces = ref_mesh.groups[&ref_group_name].face_vertex_indices.clone();
    let tgt_faces = tgt_mesh.groups[&tgt_group_name].face_vertex_indices.clone();

    let ref_tri = triangulate_faces(&ref_vertices, &ref_faces);
    let tgt_tri = triangulate_faces(&tgt_vertices, &tgt_faces);

    // Closest-point search: per target vertex → (face_id, [u, v, w]) on the
    // reference mesh. Brute force; meshes are tiny.
    let (distances, face_ids, baries) =
        point_to_mesh_distance_and_face_uvs(&tgt_vertices, &ref_vertices, &ref_tri);
    let max_dist = distances.iter().cloned().fold(0.0_f64, f64::max);
    if max_dist > 1.5e-2 {
        return Err(RetopologyError::AltTopologyTooFar(max_dist));
    }

    // Pack bary as [3, V_target] (row-major: bary[k][i] at offset k*V + i)
    // and indices as [V_target, 3] (offset i*3 + k).
    let v_target = tgt_vertices.len();
    let mut bary_flat = vec![0.0_f64; 3 * v_target];
    let mut indices_flat = vec![0_u32; v_target * 3];
    for i in 0..v_target {
        let face = ref_tri[face_ids[i] as usize];
        for k in 0..3 {
            bary_flat[k * v_target + i] = baries[i][k];
            indices_flat[i * 3 + k] = face[k];
        }
    }

    let (vertices, faces, vertex_bone_weights, vertex_bone_indices, blendshapes) =
        interpolate_topology(&reference, &bary_flat, &indices_flat, v_target, &tgt_tri)?;

    Ok(build_retopologised(
        reference,
        vertices,
        faces,
        vertex_bone_weights,
        vertex_bone_indices,
        blendshapes,
    ))
}

// ── Construction of the retopologised model from a reference. ────────────

fn build_retopologised(
    reference: Model,
    vertices: Tensor,
    faces: Vec<Vec<u32>>,
    vertex_bone_weights: Tensor,
    vertex_bone_indices: Tensor,
    blendshapes: Tensor,
) -> Model {
    Model {
        template_vertices: vertices,
        blendshapes,
        vertex_bone_indices,
        vertex_bone_weights,
        faces,
        // SMPL-X retopology drops textures.
        texture_coordinates: Vec::new(),
        face_texture_coordinate_indices: Vec::new(),
        // SMPL-X starts from a fresh vertex array; no remap to record.
        base_mesh_vertex_indices: None,
        // Bone hierarchy + rest-pose data is inherited unchanged.
        bone_labels: reference.bone_labels,
        bone_parents: reference.bone_parents,
        propagation_fronts: reference.propagation_fronts,
        template_bone_heads: reference.template_bone_heads,
        template_bone_tails: reference.template_bone_tails,
        bone_heads_blendshapes: reference.bone_heads_blendshapes,
        bone_tails_blendshapes: reference.bone_tails_blendshapes,
        bone_rolls_rotmat: reference.bone_rolls_rotmat,
        stacked_phenotype_blend_shapes_mask: reference.stacked_phenotype_blend_shapes_mask,
        anchors: reference.anchors,
        local_change_labels: reference.local_change_labels,
        extrapolate_phenotypes: reference.extrapolate_phenotypes,
        all_phenotypes: reference.all_phenotypes,
        default_pose_parameterization: reference.default_pose_parameterization,
        skinning_method: reference.skinning_method,
        y_axis: reference.y_axis,
        degenerate_rotation: reference.degenerate_rotation,
        dtype: reference.dtype,
        device: reference.device,
    }
}
