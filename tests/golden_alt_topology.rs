//! Alternative-topology smoke test. Builds a model on the smallest decimated
//! mesh (`notoes_collapse3pc`, ~369 verts) and verifies vertex/face counts
//! match the input topology, plus a basic sanity check on the forward pass.
//!
//! Cannot do a numerical golden comparison against Python — Python's
//! `create_alternative_topology_model` depends on NVIDIA Warp's
//! `point_to_mesh_distance_and_face_uvs`, which we did not port (we use a
//! pure-Rust closest-triangle search instead). The two implementations
//! should converge to the same correspondences but the Python path is not
//! easily callable from this venv.

use std::path::PathBuf;

use anny_rs::models::full_model::{ModelOptions, PoseParameterization};
use anny_rs::models::retopology::create_alternative_topology_model;
use anny_rs::phenotype::PhenotypeValues;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("anny")
        .join("src")
        .join("anny")
        .join("data")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn read_counts() -> std::collections::HashMap<String, String> {
    std::fs::read_to_string(fixtures_dir().join("alt_topology_counts.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap().to_string();
            let val = parts.next().unwrap().to_string();
            (key, val)
        })
        .collect()
}

#[test]
#[ignore = "release-only — full reference model build"]
fn alt_topology_3pct_builds_and_runs() {
    let counts = read_counts();
    let topo: &str = &counts["name"];
    let expected_verts: usize = counts["vertices"].parse().unwrap();
    let expected_faces: usize = counts["faces"].parse().unwrap();

    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = create_alternative_topology_model(&opts, topo).expect("alt topology model");
    eprintln!(
        "alt topology {topo}: V={} F={}",
        model.vertex_count(),
        model.faces.len()
    );
    assert_eq!(model.vertex_count(), expected_verts);
    // Faces are triangulated; the input is already triangles in this case.
    assert_eq!(model.faces.len(), expected_faces);

    // Forward pass produces finite, plausibly-shaped output.
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let v: Vec<f64> = out.vertices.flatten_all().unwrap().to_vec1().unwrap();
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "{bad} non-finite vertex coords");
    let z_min = v.iter().skip(2).step_by(3).cloned().fold(f64::INFINITY, f64::min);
    let z_max = v.iter().skip(2).step_by(3).cloned().fold(f64::NEG_INFINITY, f64::max);
    let height = z_max - z_min;
    eprintln!("alt topology height = {height:.3} m");
    assert!(height > 1.0 && height < 2.5, "height {height} out of range");
}
