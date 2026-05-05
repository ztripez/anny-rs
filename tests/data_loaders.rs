//! Smoke tests against the actual MakeHuman/MPFB2 data files vendored in the
//! adjacent `anny/` Python crate. Confirms the OBJ + `.target.gz` loaders parse
//! the real assets without errors and produce the expected counts.

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
fn loads_makehuman_base_mesh() {
    let path = data_root().join("mpfb2/3dobjs/base.obj");
    let mesh = anny_rs::data::obj::load(&path).unwrap_or_else(|e| {
        panic!("failed to load {path:?}: {e}");
    });
    assert_eq!(mesh.vertices.len(), 19158, "vertex count");
    assert_eq!(mesh.texture_coordinates.len(), 21334, "texture coord count");
    let total_faces: usize = mesh
        .groups
        .values()
        .map(|g| g.face_vertex_indices.len())
        .sum();
    assert_eq!(total_faces, 18486, "face count");
    assert!(mesh.groups.len() >= 100, "expected many named groups");
}

#[test]
fn loads_macrodetails_target() {
    let path = data_root()
        .join("mpfb2/targets/macrodetails/universal-female-young-maxmuscle-maxweight.target.gz");
    // 19,158 vertices in base mesh.
    let buf = anny_rs::data::target_gz::load(&path, 19158).unwrap_or_else(|e| {
        panic!("failed to load {path:?}: {e}");
    });
    assert_eq!(buf.len(), 19158 * 3);
    // The file mutates many but not all vertices; sanity-check that *some*
    // entries are non-zero and the buffer is the right size.
    let nonzero = buf.iter().filter(|x| **x != 0.0).count();
    assert!(
        nonzero > 100,
        "expected many non-zero deltas, got {nonzero}"
    );
}

#[test]
fn empty_target_is_all_zeros() {
    // The "average" macrodetails file is the identity target — content is empty.
    let path = data_root().join(
        "mpfb2/targets/macrodetails/universal-female-young-averagemuscle-averageweight.target.gz",
    );
    let buf = anny_rs::data::target_gz::load(&path, 19158).unwrap();
    assert!(buf.iter().all(|x| *x == 0.0));
}

#[test]
fn loads_shape_calibration_pth() {
    use candle_core::Device;
    let path = data_root().join("shape_calibration/girls.pth");
    // Each girls.pth top-level key holds a sub-dict with three tensors:
    // age_anchors, alpha_anchors, beta_anchors.
    for sub in [
        "conditional_height_distribution",
        "conditional_weight_distribution",
        "conditional_muscle_distribution",
        "conditional_proportions_distribution",
    ] {
        let keys = anny_rs::data::pickle::list_keys(&path, Some(sub))
            .unwrap_or_else(|e| panic!("list {sub}: {e}"));
        assert_eq!(keys.len(), 3, "{sub} expected 3 tensors, got {keys:?}");
        let map = anny_rs::data::pickle::load_all(&path, Some(sub), &Device::Cpu)
            .unwrap_or_else(|e| panic!("load {sub}: {e}"));
        for name in &["age_anchors", "alpha_anchors", "beta_anchors"] {
            assert!(map.contains_key(*name), "{sub} missing {name}");
        }
    }
}
