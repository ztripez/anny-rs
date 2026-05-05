//! Verifies the `Topology::Default` face-edit and `remove_unattached_vertices`
//! pruning paths produce the same counts Python does. Mirrors the implicit
//! contract in `anny/src/anny/models/full_model.py`.

use std::path::PathBuf;

use anny_rs::anthropometry::Anthropometry;
use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization, Topology};
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

fn read_counts(name: &str) -> std::collections::HashMap<String, usize> {
    std::fs::read_to_string(fixtures_dir().join(name))
        .unwrap_or_else(|e| panic!("read {name}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap().to_string();
            let val: usize = parts.next().unwrap().parse().unwrap();
            (key, val)
        })
        .collect()
}

#[test]
#[ignore = "release-only — full model build"]
fn topology_default_face_count_matches_python() {
    let mut opts = ModelOptions::new(data_root());
    opts.topology = Topology::Default;
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");
    let counts = read_counts("topology_default_counts.txt");
    eprintln!("rust topology=default faces: {}; python: {}", model.faces.len(), counts["faces"]);
    assert_eq!(model.faces.len(), counts["faces"]);
}

#[test]
#[ignore = "release-only — full model build"]
fn pruned_counts_match_python() {
    let mut opts = ModelOptions::new(data_root());
    opts.topology = Topology::Default;
    opts.remove_unattached_vertices = true;
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");
    let counts = read_counts("pruned_counts.txt");
    eprintln!(
        "rust pruned: V={} F={}; python: V={} F={}",
        model.vertex_count(),
        model.faces.len(),
        counts["vertices"],
        counts["faces"]
    );
    assert_eq!(model.vertex_count(), counts["vertices"]);
    assert_eq!(model.faces.len(), counts["faces"]);
    // base_mesh_vertex_indices should be set and have exactly V entries.
    let remap = model
        .base_mesh_vertex_indices
        .as_ref()
        .expect("base_mesh_vertex_indices should be set when pruning");
    assert_eq!(remap.len(), model.vertex_count());
}

#[test]
#[ignore = "release-only — full model build"]
fn anthropometry_works_after_pruning() {
    let mut opts = ModelOptions::new(data_root());
    opts.topology = Topology::Default;
    opts.remove_unattached_vertices = true;
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let anth = Anthropometry::new(&model).expect("anthropometry available after prune");
    let m = anth.measurements(&out.rest_vertices).unwrap();
    let height: Vec<f64> = m.height.flatten_all().unwrap().to_vec1().unwrap();
    let waist: Vec<f64> = m.waist_circumference.flatten_all().unwrap().to_vec1().unwrap();
    eprintln!("pruned anthropometry: height={:.4}m waist={:.4}m", height[0], waist[0]);
    // Plausible human range.
    assert!(height[0] > 1.0 && height[0] < 2.5);
    assert!(waist[0] > 0.4 && waist[0] < 1.5);
}
