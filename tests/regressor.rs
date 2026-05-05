//! Smoke test for the parameters regressor: build a target mesh by running
//! the model with a known non-default phenotype + identity pose, then run
//! the regressor and verify it converges to a small per-vertex error.
//!
//! Heavy: full model build + several forward passes per regressor iteration.
//! Run only in `--release`:
//!   cargo test --release --test regressor -- --include-ignored

use std::path::PathBuf;

use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization};
use anny_rs::parameters_regressor::{Regressor, RegressorOptions};
use anny_rs::phenotype::PhenotypeValues;
use candle_core::Tensor;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("anny")
        .join("src")
        .join("anny")
        .join("data")
}

fn build_model() -> Model {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    Model::build(&opts).expect("model build")
}

#[test]
#[ignore = "release-only regressor smoke test"]
fn fit_recovers_known_phenotype() {
    let model = build_model();

    // Build a target mesh from a known phenotype (different from defaults).
    let mut target_phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    target_phen.height = Tensor::from_vec(vec![0.8_f64], 1, &model.device).unwrap()
        .to_dtype(model.dtype).unwrap();
    target_phen.weight = Tensor::from_vec(vec![0.7_f64], 1, &model.device).unwrap()
        .to_dtype(model.dtype).unwrap();
    target_phen.muscle = Tensor::from_vec(vec![0.6_f64], 1, &model.device).unwrap()
        .to_dtype(model.dtype).unwrap();

    let target_out = model
        .forward(None, &target_phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let target_vertices = target_out.vertices.clone();

    // Run regressor.
    let opts = RegressorOptions {
        max_n_iters: 5,
        n_points: 3000,
        verbose: true,
        ..Default::default()
    };
    let reg = Regressor::new(&model, opts).unwrap();
    let result = reg
        .fit(&target_vertices, &["cupsize", "firmness", "african", "asian", "caucasian"])
        .expect("fit succeeded");

    // PVE in millimetres.
    let target_v: Vec<f64> = target_vertices.flatten_all().unwrap().to_vec1().unwrap();
    let got_v: Vec<f64> = result.vertices.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(target_v.len(), got_v.len());
    let n_verts = target_v.len() / 3;
    let mut sum_err = 0.0_f64;
    let mut max_err = 0.0_f64;
    for i in 0..n_verts {
        let dx = target_v[i * 3] - got_v[i * 3];
        let dy = target_v[i * 3 + 1] - got_v[i * 3 + 1];
        let dz = target_v[i * 3 + 2] - got_v[i * 3 + 2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        sum_err += d;
        if d > max_err {
            max_err = d;
        }
    }
    let mean_pve_mm = (sum_err / n_verts as f64) * 1000.0;
    let max_pve_mm = max_err * 1000.0;
    eprintln!("regressor PVE: mean={mean_pve_mm:.2}mm max={max_pve_mm:.2}mm");

    // Python's tolerance is 5 mm (test_fixed_shape) / 10 mm (with phenotype)
    // / 15 mm (out-of-distribution). 25 mm is a lenient first-pass smoke
    // check — if we're nowhere close to that, something is seriously broken.
    assert!(mean_pve_mm < 25.0, "mean PVE too high: {mean_pve_mm} mm");
}

#[test]
#[ignore = "release-only regressor age-anchor sweep"]
fn fit_with_age_anchor_search_at_least_as_good_as_single_call() {
    // Build a target with a known phenotype; verify that the age-anchor
    // sweep finds a fit at least as tight as a single-call fit (the sweep
    // exercises 4 anchors + a final pass, so it has more chances to land
    // close to the true age).
    let model = build_model();
    let mut target_phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    target_phen.age = Tensor::from_vec(vec![0.4_f64], 1, &model.device).unwrap()
        .to_dtype(model.dtype).unwrap();
    target_phen.height = Tensor::from_vec(vec![0.7_f64], 1, &model.device).unwrap()
        .to_dtype(model.dtype).unwrap();
    target_phen.muscle = Tensor::from_vec(vec![0.55_f64], 1, &model.device).unwrap()
        .to_dtype(model.dtype).unwrap();
    let target_out = model
        .forward(None, &target_phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let target_vertices = target_out.vertices.clone();

    let opts = RegressorOptions {
        max_n_iters: 5,
        n_points: 3000,
        verbose: false,
        ..Default::default()
    };
    let reg = Regressor::new(&model, opts).unwrap();
    let single = reg.fit(&target_vertices, &[]).unwrap();
    let swept = reg
        .fit_with_age_anchor_search(&target_vertices, &[0.0, 0.33, 0.67, 1.0])
        .unwrap();

    let target_v: Vec<f64> = target_vertices.flatten_all().unwrap().to_vec1().unwrap();
    let pve = |verts: &Tensor| -> f64 {
        let v: Vec<f64> = verts.flatten_all().unwrap().to_vec1().unwrap();
        let n = v.len() / 3;
        let mut s = 0.0;
        for i in 0..n {
            let dx = target_v[i * 3] - v[i * 3];
            let dy = target_v[i * 3 + 1] - v[i * 3 + 1];
            let dz = target_v[i * 3 + 2] - v[i * 3 + 2];
            s += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        s / n as f64 * 1000.0
    };
    let single_pve = pve(&single.vertices);
    let swept_pve = pve(&swept.vertices);
    eprintln!("single fit PVE = {single_pve:.2} mm; swept PVE = {swept_pve:.2} mm");
    // Sweep should be no worse than the single-call fit by a meaningful margin.
    // Allow 1mm slack since the final pass uses a tighter max_delta.
    assert!(
        swept_pve <= single_pve + 1.0,
        "swept fit ({swept_pve}mm) substantially worse than single ({single_pve}mm)"
    );
}
