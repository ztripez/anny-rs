//! SMPL-X retopology golden tests. The retopology module itself is no
//! longer feature-gated, but these tests need the converted safetensors
//! file produced by `tests/fixtures/convert_smplx.py` after
//! `download-smplx` (which is feature-gated).
//!
//!   cargo run --features smplx-download --bin anny-cli -- download-smplx
//!   .venv/bin/python tests/fixtures/convert_smplx.py
//!   cargo test --release --test golden_smplx -- --include-ignored

use std::path::PathBuf;

use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization};
use anny_rs::models::retopology::{create_smplx_topology_model, smplx_safetensors_path};
use anny_rs::phenotype::PhenotypeValues;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("anny")
        .join("src")
        .join("anny")
        .join("data")
}

fn read_golden(name: &str) -> (Vec<f64>, Vec<usize>) {
    let bin = fixtures_dir().join(format!("{name}.bin"));
    let shape = fixtures_dir().join(format!("{name}.shape"));
    let bytes = std::fs::read(&bin).unwrap_or_else(|e| panic!("read {bin:?}: {e}"));
    let shape_text = std::fs::read_to_string(&shape).unwrap();
    let dims: Vec<usize> = shape_text
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let n: usize = dims.iter().product();
    assert_eq!(bytes.len(), n * 8);
    let mut out = Vec::with_capacity(n);
    let mut buf = [0_u8; 8];
    for i in 0..n {
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        out.push(f64::from_le_bytes(buf));
    }
    (out, dims)
}

fn read_counts() -> std::collections::HashMap<String, usize> {
    std::fs::read_to_string(fixtures_dir().join("smplx_counts.txt"))
        .unwrap()
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

fn build_model() -> Model {
    let path = smplx_safetensors_path();
    if !path.exists() {
        panic!(
            "missing {} — run tests/fixtures/convert_smplx.py after `download-smplx`",
            path.display()
        );
    }
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    create_smplx_topology_model(&opts).expect("smplx model")
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
#[ignore = "release-only — full model build + smplx data download"]
fn smplx_counts_match_python() {
    let model = build_model();
    let counts = read_counts();
    assert_eq!(model.vertex_count(), counts["vertices"]);
    assert_eq!(model.faces.len(), counts["faces"]);
    assert_eq!(
        model.vertex_bone_indices.dim(1).unwrap(),
        counts["max_bones_per_vertex"]
    );
}

#[test]
#[ignore = "release-only — full model build + smplx data download"]
fn smplx_template_vertices_match_python() {
    let model = build_model();
    let (golden, dims) = read_golden("smplx_template_vertices");
    assert_eq!(dims, vec![model.vertex_count(), 3]);
    let got: Vec<f64> = model
        .template_vertices
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max = max_abs(&got, &golden);
    eprintln!("smplx template_vertices: max abs = {max:.3e}");
    // The barycentric data ships at f32 precision (per the .pth dtype). After
    // the up-cast the most we can hope for is ~3e-7 absolute on a 1m body.
    assert!(max < 1e-6, "max abs err {max} exceeds 1e-6");
}

#[test]
#[ignore = "release-only — full model build + smplx data download"]
fn smplx_rest_vertices_match_python() {
    let model = build_model();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let (golden, dims) = read_golden("smplx_rest_vertices");
    assert_eq!(dims, out.rest_vertices.dims().to_vec());
    let got: Vec<f64> = out.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();
    let max = max_abs(&got, &golden);
    eprintln!("smplx rest_vertices: max abs = {max:.3e}");
    assert!(max < 1e-6, "max abs err {max} exceeds 1e-6");
}
