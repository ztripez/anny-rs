//! Golden test: build the Rust model with config matching `tests/fixtures/export_golden.py`
//! and compare every published intermediate to the Python output bit-for-bit
//! (within a tight tolerance).
//!
//! Heavy: loads ~650 .target.gz files. Run only in `--release`:
//!   cargo test --release --test golden_forward -- --include-ignored

use std::path::PathBuf;

use anny_rs::anthropometry::Anthropometry;
use anny_rs::face_segmentation::FaceSegmentation;
use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization, SkinningMethod};
use anny_rs::models::presets::{create_hand_model, create_head_model, HandSide};
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
    assert_eq!(bytes.len(), n * 8, "{name}: byte count mismatch");
    let mut out = Vec::with_capacity(n);
    let mut buf = [0_u8; 8];
    for i in 0..n {
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        out.push(f64::from_le_bytes(buf));
    }
    (out, dims)
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn rms_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

fn build_model() -> Model {
    let mut opts = ModelOptions::new(data_root());
    opts.default_pose_parameterization = PoseParameterization::RestRelative;
    opts.skinning_method = SkinningMethod::Lbs;
    opts.all_phenotypes = true;
    Model::build(&opts).expect("model build")
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_template_vertices_match_python() {
    let model = build_model();
    let (golden, dims) = read_golden("template_vertices");
    assert_eq!(dims, vec![model.vertex_count(), 3]);
    let got: Vec<f64> = model.template_vertices.flatten_all().unwrap().to_vec1().unwrap();
    let max = max_abs_err(&got, &golden);
    let rms = rms_err(&got, &golden);
    eprintln!("template_vertices: max_abs={max:.3e} rms={rms:.3e}");
    assert!(max < 1e-12, "template_vertices max abs err = {max}");
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_blendshape_coeffs_match_python() {
    let model = build_model();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let coeffs = model.phenotype_coefficients(&phen).unwrap();
    let (golden, dims) = read_golden("blendshape_coeffs");
    assert_eq!(dims, coeffs.dims().to_vec());
    let got: Vec<f64> = coeffs.flatten_all().unwrap().to_vec1().unwrap();
    let max = max_abs_err(&got, &golden);
    let rms = rms_err(&got, &golden);
    eprintln!("blendshape_coeffs: max_abs={max:.3e} rms={rms:.3e}");
    assert!(max < 1e-12, "blendshape_coeffs max abs err = {max}");
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_rest_vertices_match_python() {
    let model = build_model();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let (golden, dims) = read_golden("rest_vertices");
    assert_eq!(dims, out.rest_vertices.dims().to_vec());
    let got: Vec<f64> = out.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();
    let max = max_abs_err(&got, &golden);
    let rms = rms_err(&got, &golden);
    eprintln!("rest_vertices: max_abs={max:.3e} rms={rms:.3e}");
    assert!(max < 1e-9, "rest_vertices max abs err = {max} (rms {rms})");
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_rest_bone_poses_match_python() {
    let model = build_model();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let (golden, dims) = read_golden("rest_bone_poses");
    assert_eq!(dims, out.rest_bone_poses.dims().to_vec());
    let got: Vec<f64> = out.rest_bone_poses.flatten_all().unwrap().to_vec1().unwrap();
    let max = max_abs_err(&got, &golden);
    let rms = rms_err(&got, &golden);
    eprintln!("rest_bone_poses: max_abs={max:.3e} rms={rms:.3e}");
    // Bone-pose construction goes through atan2, cross-product, and rotvec→
    // rotmat, where Python (roma) uses a different numerical path than our
    // host-side conversions. ~1e-7 abs (100 nanometers on a 1 m body) is the
    // expected precision.
    assert!(max < 1e-6, "rest_bone_poses max abs err = {max} (rms {rms})");
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_full_forward_matches_python() {
    let model = build_model();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let (golden, dims) = read_golden("vertices_rest_relative_identity");
    assert_eq!(dims, out.vertices.dims().to_vec());
    let got: Vec<f64> = out.vertices.flatten_all().unwrap().to_vec1().unwrap();
    let max = max_abs_err(&got, &golden);
    let rms = rms_err(&got, &golden);
    eprintln!("posed vertices: max_abs={max:.3e} rms={rms:.3e}");
    assert!(max < 1e-6, "vertices max abs err = {max} (rms {rms})");
}

fn read_preset_counts() -> std::collections::HashMap<String, usize> {
    let text = std::fs::read_to_string(fixtures_dir().join("preset_counts.txt")).unwrap();
    text.lines()
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
#[ignore = "release-only preset construction (rebuilds the full model twice)"]
fn golden_hand_model_matches_python() {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = create_hand_model(&opts, HandSide::Right).expect("hand model build");
    let counts = read_preset_counts();
    eprintln!(
        "rust hand.R: bones={} faces={}",
        model.bone_count(),
        model.faces.len()
    );
    assert_eq!(model.bone_count(), counts["hand_R_bones"]);
    assert_eq!(model.faces.len(), counts["hand_R_faces"]);
}

#[test]
#[ignore = "release-only preset construction (rebuilds the full model twice)"]
fn golden_head_model_matches_python() {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = create_head_model(&opts, true, true).expect("head model build");
    let counts = read_preset_counts();
    eprintln!(
        "rust head: bones={} faces={}",
        model.bone_count(),
        model.faces.len()
    );
    assert_eq!(model.bone_count(), counts["head_bones"]);
    assert_eq!(model.faces.len(), counts["head_faces"]);
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_face_segmentation_matches_python() {
    let model = build_model();
    let seg = FaceSegmentation::new(&model, &data_root()).unwrap();
    for (label, fixture) in &[
        ("head", "face_segmentation_head"),
        ("hand.R", "face_segmentation_hand_R"),
    ] {
        let mask = seg.face_mask(&[*label]).unwrap();
        let (golden, _dims) = read_golden(fixture);
        assert_eq!(mask.len(), golden.len(), "{label}: length");
        let mut diff = 0usize;
        for (m, g) in mask.iter().zip(golden.iter()) {
            let g_bool = *g != 0.0;
            if *m != g_bool {
                diff += 1;
            }
        }
        let count_rust: usize = mask.iter().filter(|x| **x).count();
        let count_py: usize = golden.iter().filter(|x| **x != 0.0).count();
        eprintln!("{label}: rust={count_rust} python={count_py} diff={diff}");
        assert_eq!(diff, 0, "{label}: {diff} face-segmentation differences");
    }
}

#[test]
#[ignore = "release-only golden test against Python anny output"]
fn golden_anthropometry_matches_python() {
    let model = build_model();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let anth = Anthropometry::new(&model).expect("anthropometry available");
    let m = anth.measurements(&out.rest_vertices).unwrap();

    for (name, t) in &[
        ("height", &m.height),
        ("waist_circumference", &m.waist_circumference),
        ("volume", &m.volume),
        ("mass", &m.mass),
        ("bmi", &m.bmi),
    ] {
        let (golden, _dims) = read_golden(&format!("anthropometry_{name}"));
        let got: Vec<f64> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), golden.len(), "{name}: length mismatch");
        let max = max_abs_err(&got, &golden);
        let scale = golden[0].abs().max(1.0);
        let rel = max / scale;
        eprintln!("{name}: got={:.6} golden={:.6} abs_err={:.3e} rel_err={:.3e}",
                  got[0], golden[0], max, rel);
        assert!(rel < 1e-6, "{name} relative err {rel} exceeds 1e-6");
    }
}

