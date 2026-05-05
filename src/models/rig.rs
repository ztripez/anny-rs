//! Rig + skinning-weights JSON parsing and the per-vertex bone matrix builder.
//!
//! Mirrors the rig-loading section of `anny/src/anny/models/full_model.py:170–268`.
//! The rig JSON describes a bone hierarchy: each entry has a `head`/`tail`
//! coordinate-regressor (which vertices to average to compute the joint
//! position), a `roll` angle (radians, used for bone roll correction), and a
//! string `parent` (`""` for the root). The weights JSON is a sparse list of
//! `(vertex_id, weight)` pairs per bone.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

use crate::data::obj::ObjMesh;

#[derive(Debug, Error)]
pub enum RigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("expected exactly one root bone (parent == \"\"), found {found}")]
    BadRootCount { found: usize },
    #[error("unknown coordinate strategy: {0}")]
    UnknownStrategy(String),
    #[error("missing OBJ group {0} required by rig regressor")]
    MissingGroup(String),
    #[error("bone {0} referenced as parent but not present")]
    UnknownParent(String),
    #[error("bone to remove not found: {0}")]
    UnknownBoneToRemove(String),
    #[error("bone {0} has no skinning weights")]
    NoWeights(String),
}

// ── Rig JSON schema ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RigBoneJson {
    pub head: CoordinateRegressorJson,
    pub tail: CoordinateRegressorJson,
    pub roll: f64,
    pub parent: String,
}

#[derive(Debug, Deserialize)]
pub struct CoordinateRegressorJson {
    pub strategy: String,
    pub cube_name: Option<String>,
    pub vertex_index: Option<u32>,
    pub vertex_indices: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
pub struct WeightsJson {
    pub weights: BTreeMap<String, Vec<(u32, f64)>>,
}

/// Parses a rig JSON file. The result preserves the file's iteration order via
/// a `Vec<(name, body)>` so we can apply the same DFS the Python loader does.
pub fn load_rig_json(path: impl AsRef<Path>) -> Result<Vec<(String, RigBoneJson)>, RigError> {
    let text = std::fs::read_to_string(path)?;
    let raw: serde_json::Value = serde_json::from_str(&text)?;
    let map = raw.as_object().ok_or_else(|| {
        RigError::Json(serde_json::Error::io(std::io::Error::other(
            "rig JSON root not an object",
        )))
    })?;
    let mut out = Vec::with_capacity(map.len());
    for (k, v) in map {
        let bone: RigBoneJson = serde_json::from_value(v.clone())?;
        out.push((k.clone(), bone));
    }
    Ok(out)
}

pub fn load_weights_json(path: impl AsRef<Path>) -> Result<WeightsJson, RigError> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

// ── Coordinate regressor: vertex indices to average for a joint ──────────

/// Resolves a head/tail regressor to the list of vertex indices that should be
/// averaged to obtain the joint position. Mirrors `_get_coordinates_regressor`
/// in `full_model.py:117–130`.
pub fn resolve_regressor(
    regressor: &CoordinateRegressorJson,
    mesh: &ObjMesh,
) -> Result<Vec<u32>, RigError> {
    match regressor.strategy.as_str() {
        "VERTEX" => Ok(vec![regressor.vertex_index.unwrap_or(0)]),
        "CUBE" => {
            let name = regressor
                .cube_name
                .as_ref()
                .ok_or_else(|| RigError::UnknownStrategy("CUBE missing cube_name".into()))?;
            let group = mesh
                .groups
                .get(name)
                .ok_or_else(|| RigError::MissingGroup(name.clone()))?;
            // Unique vertex indices used by any face in this group.
            let mut set: HashSet<u32> = HashSet::new();
            for face in &group.face_vertex_indices {
                for v in face {
                    set.insert(*v);
                }
            }
            let mut out: Vec<u32> = set.into_iter().collect();
            out.sort_unstable();
            Ok(out)
        }
        "MEAN" => Ok(regressor.vertex_indices.clone().unwrap_or_default()),
        other => Err(RigError::UnknownStrategy(other.to_string())),
    }
}

// ── Bone hierarchy in topological (DFS) order ────────────────────────────

/// Topologically-sorted bone hierarchy. `parents[i]` is the parent's index in
/// this list (`-1` for roots). After construction, `i > parents[i]` always.
#[derive(Debug, Clone)]
pub struct BoneHierarchy {
    pub labels: Vec<String>,
    pub parents: Vec<i64>,
    pub head_regressors: Vec<Vec<u32>>,
    pub tail_regressors: Vec<Vec<u32>>,
    pub rolls: Vec<f64>,
}

/// Walks the rig in DFS order starting from the unique root, mirroring
/// `parse_recursively` in `full_model.py:208–215`. Order matters: children
/// are encountered in the order they appear in the JSON map.
fn dfs_topological_order(bones: &[(String, RigBoneJson)]) -> Result<Vec<usize>, RigError> {
    // Map from bone name → index in the input list.
    let name_to_idx: HashMap<&str, usize> = bones
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.as_str(), i))
        .collect();

    // Find the unique root.
    let roots: Vec<usize> = bones
        .iter()
        .enumerate()
        .filter(|(_, (_, b))| b.parent.is_empty())
        .map(|(i, _)| i)
        .collect();
    if roots.len() != 1 {
        return Err(RigError::BadRootCount { found: roots.len() });
    }
    let root = roots[0];

    // Pre-compute children-in-source-order for each bone, so the DFS uses the
    // same insertion order Python's dict iteration would.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); bones.len()];
    for (i, (_, b)) in bones.iter().enumerate() {
        if b.parent.is_empty() {
            continue;
        }
        let p_idx = *name_to_idx
            .get(b.parent.as_str())
            .ok_or_else(|| RigError::UnknownParent(b.parent.clone()))?;
        children[p_idx].push(i);
    }

    let mut order: Vec<usize> = Vec::with_capacity(bones.len());
    let mut stack: Vec<usize> = vec![root];
    while let Some(node) = stack.pop() {
        order.push(node);
        // Push children in reverse so the first child is processed first.
        for &c in children[node].iter().rev() {
            stack.push(c);
        }
    }
    Ok(order)
}

/// Builds the topologically-sorted bone hierarchy from rig JSON + base mesh.
/// Optionally drops bones whose names appear in `bones_to_remove` (their
/// children are reparented to the grandparent and their skinning weights are
/// merged into the parent's).
///
/// `weights` is mutated in place to reflect the bone removals.
pub fn build_hierarchy(
    bones: &[(String, RigBoneJson)],
    mesh: &ObjMesh,
    bones_to_remove: &HashSet<String>,
    weights: &mut WeightsJson,
) -> Result<BoneHierarchy, RigError> {
    // Topological order via DFS.
    let order = dfs_topological_order(bones)?;
    let name_to_topo: HashMap<String, usize> = order
        .iter()
        .enumerate()
        .map(|(topo_i, &src_i)| (bones[src_i].0.clone(), topo_i))
        .collect();

    let mut labels: Vec<String> = order.iter().map(|&i| bones[i].0.clone()).collect();
    let mut parents: Vec<i64> = order
        .iter()
        .map(|&src_i| {
            let parent_name = &bones[src_i].1.parent;
            if parent_name.is_empty() {
                -1
            } else {
                name_to_topo[parent_name] as i64
            }
        })
        .collect();
    let mut head_regressors: Vec<Vec<u32>> = order
        .iter()
        .map(|&i| resolve_regressor(&bones[i].1.head, mesh))
        .collect::<Result<_, _>>()?;
    let mut tail_regressors: Vec<Vec<u32>> = order
        .iter()
        .map(|&i| resolve_regressor(&bones[i].1.tail, mesh))
        .collect::<Result<_, _>>()?;
    let mut rolls: Vec<f64> = order.iter().map(|&i| bones[i].1.roll).collect();

    // Apply bone removals one at a time, mirroring the Python while-loop.
    for to_remove in bones_to_remove {
        let idx = match labels.iter().position(|n| n == to_remove) {
            Some(i) => i,
            None => return Err(RigError::UnknownBoneToRemove(to_remove.clone())),
        };
        let parent_idx = parents[idx];
        // Move skinning weights from removed bone → its parent.
        if let Some(child_weights) = weights.weights.remove(to_remove)
            && parent_idx >= 0
        {
            let parent_name = labels[parent_idx as usize].clone();
            weights
                .weights
                .entry(parent_name)
                .or_default()
                .extend(child_weights);
        }
        // Reparent children of the removed bone to the grandparent. Indices
        // > idx need to be decremented after the pop.
        for p in parents.iter_mut() {
            if *p == idx as i64 {
                *p = parent_idx;
            } else if *p > idx as i64 {
                *p -= 1;
            }
        }
        labels.remove(idx);
        parents.remove(idx);
        head_regressors.remove(idx);
        tail_regressors.remove(idx);
        rolls.remove(idx);
    }

    Ok(BoneHierarchy {
        labels,
        parents,
        head_regressors,
        tail_regressors,
        rolls,
    })
}

// ── Per-vertex bone matrix (sparse → padded dense, normalised) ───────────

/// Per-vertex bone influence matrix. Both tensors are `[V, M]` where `M` is
/// the maximum number of influencing bones across all vertices. Padding uses
/// bone index `0` and weight `0.0`. Weights are L1-normalised per vertex.
#[derive(Debug, Clone)]
pub struct VertexBoneMatrix {
    pub indices: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f64>>,
    pub max_bones_per_vertex: usize,
}

/// Builds the padded per-vertex bone matrix from a bone-keyed weight list and
/// the topologically-sorted bone labels. After this call, the rows are
/// normalised to sum to 1 and padded to a uniform width.
pub fn build_vertex_bone_matrix(
    weights: &WeightsJson,
    labels: &[String],
    vertex_count: usize,
) -> VertexBoneMatrix {
    let mut indices: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
    let mut row_weights: Vec<Vec<f64>> = vec![Vec::new(); vertex_count];

    for (bone_id, label) in labels.iter().enumerate() {
        if let Some(entries) = weights.weights.get(label) {
            for &(vertex_id, w) in entries {
                let v = vertex_id as usize;
                if v >= vertex_count {
                    continue; // ignore stray indices
                }
                indices[v].push(bone_id as u32);
                row_weights[v].push(w);
            }
        }
    }

    let max_bones_per_vertex = indices.iter().map(|r| r.len()).max().unwrap_or(0);

    // Pad and normalise.
    for (idx_row, w_row) in indices.iter_mut().zip(row_weights.iter_mut()) {
        while idx_row.len() < max_bones_per_vertex {
            idx_row.push(0);
            w_row.push(0.0);
        }
        let sum: f64 = w_row.iter().sum();
        if sum > 0.0 {
            for w in w_row.iter_mut() {
                *w /= sum;
            }
        }
    }

    VertexBoneMatrix {
        indices,
        weights: row_weights,
        max_bones_per_vertex,
    }
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
    fn loads_default_rig() {
        let rig = load_rig_json(data_root().join("mpfb2/rigs/standard/rig.default.json")).unwrap();
        assert!(!rig.is_empty(), "expected at least one bone");
        let roots: Vec<&String> = rig
            .iter()
            .filter(|(_, b)| b.parent.is_empty())
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            roots.len(),
            1,
            "expected exactly one root, found {:?}",
            roots
        );
    }

    #[test]
    fn loads_default_weights() {
        let weights =
            load_weights_json(data_root().join("mpfb2/rigs/standard/weights.default.json"))
                .unwrap();
        assert!(
            weights.weights.len() > 50,
            "expected many bones to have weights"
        );
    }

    #[test]
    fn builds_hierarchy_for_default_rig() {
        let mesh = crate::data::obj::load(data_root().join("mpfb2/3dobjs/base.obj")).unwrap();
        let rig = load_rig_json(data_root().join("mpfb2/rigs/standard/rig.default.json")).unwrap();
        let mut weights =
            load_weights_json(data_root().join("mpfb2/rigs/standard/weights.default.json"))
                .unwrap();
        let h = build_hierarchy(&rig, &mesh, &HashSet::new(), &mut weights).unwrap();
        assert_eq!(h.labels.len(), rig.len());
        assert_eq!(h.parents.iter().filter(|&&p| p == -1).count(), 1);
        // Topological invariant: every parent index is < its child index.
        for (i, &p) in h.parents.iter().enumerate() {
            if p >= 0 {
                assert!((p as usize) < i, "bone {} parent {p} not before child", i);
            }
        }
    }

    #[test]
    fn vertex_bone_matrix_normalises_weights() {
        let mesh = crate::data::obj::load(data_root().join("mpfb2/3dobjs/base.obj")).unwrap();
        let rig = load_rig_json(data_root().join("mpfb2/rigs/standard/rig.default.json")).unwrap();
        let mut weights =
            load_weights_json(data_root().join("mpfb2/rigs/standard/weights.default.json"))
                .unwrap();
        let h = build_hierarchy(&rig, &mesh, &HashSet::new(), &mut weights).unwrap();
        let vbm = build_vertex_bone_matrix(&weights, &h.labels, mesh.vertices.len());
        assert_eq!(vbm.indices.len(), mesh.vertices.len());
        // Every vertex should sum to ~1 (or 0 if completely unattached).
        let mut weighted = 0;
        for row in &vbm.weights {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                assert!((sum - 1.0).abs() < 1e-9, "row sum {sum} ≠ 1");
                weighted += 1;
            }
        }
        assert!(
            weighted > 18000,
            "expected most vertices to have weights, got {weighted}"
        );
        assert!(
            vbm.max_bones_per_vertex >= 4 && vbm.max_bones_per_vertex <= 16,
            "max bones per vertex {} outside expected range",
            vbm.max_bones_per_vertex
        );
    }

    #[test]
    fn bone_removal_reparents_children() {
        let mesh = crate::data::obj::load(data_root().join("mpfb2/3dobjs/base.obj")).unwrap();
        let rig = load_rig_json(data_root().join("mpfb2/rigs/standard/rig.default.json")).unwrap();
        let mut weights =
            load_weights_json(data_root().join("mpfb2/rigs/standard/weights.default.json"))
                .unwrap();
        // Remove the eye bones (a known safe removal — they're leaves).
        let mut to_remove = HashSet::new();
        to_remove.insert("eye.L".to_string());
        to_remove.insert("eye.R".to_string());
        let h = build_hierarchy(&rig, &mesh, &to_remove, &mut weights).unwrap();
        assert!(!h.labels.iter().any(|n| n == "eye.L" || n == "eye.R"));
        assert_eq!(h.labels.len(), rig.len() - 2);
    }
}
