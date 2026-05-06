//! Verifies that the local-change blend-shape API actually drives the
//! output. Mirrors the spirit of `anny/test/test_local_changes.py` plus
//! the additional check that *non-default* `local_changes` produce
//! different vertices.
//!
//! Heavy: full model build. Run only in `--release`:
//!   cargo test --release --test local_changes -- --include-ignored

use std::collections::HashMap;
use std::path::PathBuf;

use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization, Topology};
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

fn build() -> Model {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    Model::build(&opts).expect("model build")
}

#[test]
#[ignore = "release-only — full model build"]
fn local_changes_modify_rest_vertices() {
    let model = build();
    assert!(
        !model.local_change_labels.is_empty(),
        "model should expose at least one local change label"
    );
    let label = model.local_change_labels[0].clone();
    eprintln!(
        "exercising local change '{label}' (of {} total)",
        model.local_change_labels.len()
    );

    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();

    // Baseline: empty local-change map (zero-pad path).
    let baseline = model
        .forward_with_local_changes(
            None,
            &phen,
            &HashMap::new(),
            Some(PoseParameterization::RestRelative),
        )
        .unwrap();
    let base_v: Vec<f64> = baseline
        .rest_vertices
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // Driven positive: v = 1.0 → activates `relu(1) = 1` on the positive slot.
    let pos_value = Tensor::from_vec(vec![1.0_f64], 1, &model.device)
        .unwrap()
        .to_dtype(model.dtype)
        .unwrap();
    let mut driven = HashMap::new();
    driven.insert(label.clone(), pos_value);
    let pos = model
        .forward_with_local_changes(
            None,
            &phen,
            &driven,
            Some(PoseParameterization::RestRelative),
        )
        .unwrap();
    let pos_v: Vec<f64> = pos.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();

    let max_pos_diff = base_v
        .iter()
        .zip(pos_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("positive activation max abs diff vs baseline: {max_pos_diff:.3e}");
    assert!(
        max_pos_diff > 1e-6,
        "local-change '{label}' had no effect at v=1 (max diff {max_pos_diff})"
    );

    // Driven negative: v = -1.0 → activates the negative slot.
    let neg_value = Tensor::from_vec(vec![-1.0_f64], 1, &model.device)
        .unwrap()
        .to_dtype(model.dtype)
        .unwrap();
    let mut driven_neg = HashMap::new();
    driven_neg.insert(label.clone(), neg_value);
    let neg = model
        .forward_with_local_changes(
            None,
            &phen,
            &driven_neg,
            Some(PoseParameterization::RestRelative),
        )
        .unwrap();
    let neg_v: Vec<f64> = neg.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();

    let max_neg_vs_base = base_v
        .iter()
        .zip(neg_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_neg_vs_pos = pos_v
        .iter()
        .zip(neg_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("negative activation max abs diff vs baseline: {max_neg_vs_base:.3e}");
    eprintln!("negative activation max abs diff vs positive: {max_neg_vs_pos:.3e}");
    assert!(
        max_neg_vs_base > 1e-6,
        "negative activation had no effect (max diff {max_neg_vs_base})"
    );
    assert!(
        max_neg_vs_pos > 1e-6,
        "positive and negative activations gave the same output"
    );

    // Default forward (no map) must equal the empty-map path bit-for-bit.
    let default_out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let default_v: Vec<f64> = default_out
        .rest_vertices
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_default_vs_base = base_v
        .iter()
        .zip(default_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_default_vs_base < 1e-12,
        "forward(...) must match forward_with_local_changes(empty map) (diff {max_default_vs_base})"
    );
}

#[test]
#[ignore = "release-only — full model build"]
fn smooth_normals_on_full_body_are_unit_length() {
    use anny_rs::utils::mesh::smooth_vertex_normals;

    let model = build();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    // [1, V, 3] → flatten to [V, 3].
    let v_flat: Vec<f64> = out.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();
    let n_verts = model.vertex_count();
    let vertices: Vec<[f64; 3]> = (0..n_verts)
        .map(|i| [v_flat[i * 3], v_flat[i * 3 + 1], v_flat[i * 3 + 2]])
        .collect();

    let normals = smooth_vertex_normals(&vertices, &model.faces);
    assert_eq!(normals.len(), n_verts);

    let mut zero_count = 0usize;
    let mut bad_count = 0usize;
    for n in &normals {
        if !n[0].is_finite() || !n[1].is_finite() || !n[2].is_finite() {
            bad_count += 1;
            continue;
        }
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-9 {
            zero_count += 1;
        } else {
            assert!(
                (len - 1.0).abs() < 1e-9,
                "non-unit normal {n:?} (len {len})"
            );
        }
    }
    eprintln!(
        "smooth normals: V={n_verts} unit={} unreferenced={zero_count} non-finite={bad_count}",
        n_verts - zero_count - bad_count
    );
    assert_eq!(bad_count, 0, "{bad_count} non-finite normals");
    // Most vertices participate in at least one face — a few thousand are
    // unattached scalp/eyeball internals.
    assert!(
        zero_count < n_verts / 2,
        "too many unreferenced vertices: {zero_count} of {n_verts}"
    );
}

#[test]
#[ignore = "release-only — full model build"]
fn genital_morphs_are_gated_by_opt_in_flag() {
    // Default build: parity with Python's create_fullbody_model() — no
    // entries from the `genitals/` category should be exposed.
    let default_model = build();
    let default_genital: Vec<&String> = default_model
        .local_change_labels
        .iter()
        .filter(|s| s.starts_with("penis-"))
        .collect();
    assert!(
        default_genital.is_empty(),
        "default build leaked genital morphs: {default_genital:?}"
    );

    // Opt-in build with Makehuman topology: the 6 baseline genital morphs
    // shipped under data/mpfb2/targets/genitals/ should now be drivable.
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    opts.topology = Topology::Makehuman;
    opts.include_genital_morphs = true;
    let opt_in = Model::build(&opts).expect("opt-in model build");

    let expected = [
        "penis-circ-incr",
        "penis-length-incr",
        "penis-testicles-incr",
    ];
    for label in expected {
        assert!(
            opt_in.local_change_labels.iter().any(|l| l == label),
            "expected '{label}' in local_change_labels with include_genital_morphs=true; \
             got {:?}",
            opt_in.local_change_labels
        );
    }

    // Counts: opt-in adds exactly the genital pairs on top of the default set.
    assert!(
        opt_in.local_change_labels.len() > default_model.local_change_labels.len(),
        "opt-in build did not add any extra labels"
    );
}

#[test]
#[ignore = "release-only — full model build"]
fn unknown_local_change_labels_are_ignored() {
    let model = build();
    let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();

    let mut bogus = HashMap::new();
    let v = Tensor::from_vec(vec![0.5_f64], 1, &model.device)
        .unwrap()
        .to_dtype(model.dtype)
        .unwrap();
    bogus.insert("not-a-real-label".to_string(), v);

    // Should not panic / error — Python's behaviour is silent ignore.
    let out = model
        .forward_with_local_changes(
            None,
            &phen,
            &bogus,
            Some(PoseParameterization::RestRelative),
        )
        .unwrap();
    let v: Vec<f64> = out.rest_vertices.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}
