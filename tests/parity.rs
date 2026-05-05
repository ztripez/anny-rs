//! Self-consistency tests mirroring Python's `test_kinematics.py`,
//! `test_pose_parameterization.py`, and `test_various.py::test_batch_consistency`.
//!
//! These check internal mathematical invariants (sequential FK == parallel FK,
//! pose-parameterization round-trip, batched forward == per-element forward),
//! not parity with Python — so they run fast in debug too.

use std::path::PathBuf;

use anny_rs::kinematics::{
    parallel_forward_kinematic, propagation_fronts, sequential_forward_kinematic,
};
use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization};
use anny_rs::phenotype::PhenotypeValues;
use anny_rs::rotation::{rigid_to_homogeneous, rotvec_to_rotmat};
use candle_core::{DType, Device, Tensor};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("anny")
        .join("src")
        .join("anny")
        .join("data")
}

/// Tiny deterministic pseudo-random sequence — splitmix64 keyed by a counter.
struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform `[0, 1)` f64.
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / (1u64 << 53) as f64
    }
    /// Roughly N(0, 1) via Box-Muller, called twice on each invocation but we
    /// only consume one result.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-30);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn random_rotvecs(rng: &mut Rng, shape: &[usize]) -> Vec<f64> {
    let total: usize = shape.iter().product();
    (0..total).map(|_| rng.next_normal() * 0.5).collect()
}

fn random_translations(rng: &mut Rng, shape: &[usize]) -> Vec<f64> {
    let total: usize = shape.iter().product();
    (0..total).map(|_| rng.next_normal() * 0.5).collect()
}

/// Builds `[B, K, 4, 4]` of random rigid transforms by combining a random
/// rotvec → rotmat with a random translation.
fn random_rigid_batch(rng: &mut Rng, bs: usize, k: usize, dtype: DType, device: &Device) -> Tensor {
    let rotvecs = random_rotvecs(rng, &[bs * k, 3]);
    let translations = random_translations(rng, &[bs * k, 3]);
    let rotvec_t = Tensor::from_vec(rotvecs, (bs * k, 3), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let translation_t = Tensor::from_vec(translations, (bs * k, 3), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let r = rotvec_to_rotmat(&rotvec_t).unwrap(); // [B*K, 3, 3]
    let h = rigid_to_homogeneous(&r, &translation_t).unwrap(); // [B*K, 4, 4]
    h.reshape((bs, k, 4, 4)).unwrap().contiguous().unwrap()
}

// ────────────────────────────────────────────────────────────────────────────
// 1. test_kinematics: sequential FK == parallel FK.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn parity_sequential_vs_parallel_fk_synthetic() {
    // Tree: 0 → {1, 2}; 1 → 3; 2 → 4; 1 → 5; 2 → 6.
    let parents = vec![-1_i64, 0, 0, 1, 2, 1, 2];
    let bs = 4;
    let n = parents.len();
    let mut rng = Rng::new(0xa1ce);
    let device = Device::Cpu;
    let dtype = DType::F64;
    let rest = random_rigid_batch(&mut rng, bs, n, dtype, &device);
    let delta = random_rigid_batch(&mut rng, bs, n, dtype, &device);
    let fronts = propagation_fronts(&parents);
    let parallel = parallel_forward_kinematic(&fronts, &rest, &delta, None).unwrap();
    let sequential = sequential_forward_kinematic(&parents, &rest, &delta).unwrap();

    let p_par: Vec<f64> = parallel.poses.flatten_all().unwrap().to_vec1().unwrap();
    let p_seq: Vec<f64> = sequential.poses.flatten_all().unwrap().to_vec1().unwrap();
    let t_par: Vec<f64> = parallel
        .transforms
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let t_seq: Vec<f64> = sequential
        .transforms
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_pose = max_diff(&p_par, &p_seq);
    let max_transform = max_diff(&t_par, &t_seq);
    assert!(max_pose < 1e-12, "poses diverge: max abs err = {max_pose}");
    assert!(
        max_transform < 1e-12,
        "transforms diverge: max abs err = {max_transform}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// 2. test_pose_parameterization: round-trip every pair of modes.
// ────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "release-only — full model build required"]
fn parity_pose_parameterization_roundtrip() {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");

    let bs = 2;
    let mut rng = Rng::new(0xb007);
    let device = model.device.clone();
    let dtype = model.dtype;
    let n_bones = model.bone_count();

    // Random phenotype.
    let mut phen = PhenotypeValues::defaults(dtype, &device).unwrap();
    let labels = [
        "age",
        "gender",
        "muscle",
        "weight",
        "height",
        "proportions",
        "cupsize",
        "firmness",
        "african",
        "asian",
        "caucasian",
    ];
    for label in &labels {
        let v: Vec<f64> = (0..bs).map(|_| 0.3 + 0.4 * rng.next_f64()).collect();
        let t = Tensor::from_vec(v, bs, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        match *label {
            "age" => phen.age = t,
            "gender" => phen.gender = t,
            "muscle" => phen.muscle = t,
            "weight" => phen.weight = t,
            "height" => phen.height = t,
            "proportions" => phen.proportions = t,
            "cupsize" => phen.cupsize = t,
            "firmness" => phen.firmness = t,
            "african" => phen.african = t,
            "asian" => phen.asian = t,
            "caucasian" => phen.caucasian = t,
            _ => unreachable!(),
        }
    }

    let source_pose = random_rigid_batch(&mut rng, bs, n_bones, dtype, &device);
    let modes = [
        PoseParameterization::RestRelative,
        PoseParameterization::RootRelative,
        PoseParameterization::RootRelativeWorld,
        PoseParameterization::Absolute,
    ];

    let mut failures = Vec::new();
    for &source_mode in &modes {
        let source_out = model
            .forward(Some(&source_pose), &phen, Some(source_mode))
            .unwrap();
        let source_v: Vec<f64> = source_out
            .vertices
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        for &target_mode in &modes {
            let target_pose = model
                .pose_parameterization(&source_out, target_mode)
                .unwrap();
            let target_out = model
                .forward(Some(&target_pose), &phen, Some(target_mode))
                .unwrap();
            let target_v: Vec<f64> = target_out
                .vertices
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max = max_diff(&source_v, &target_v);
            let label = format!("{:?} -> {:?}", source_mode, target_mode);
            eprintln!("  [{label}] max abs err = {max:.3e}");
            if max >= 1e-5 {
                failures.push(label);
            }
        }
    }
    assert!(failures.is_empty(), "failed pairs: {failures:?}");
}

// ────────────────────────────────────────────────────────────────────────────
// 3. test_batch_consistency: forward(B=N) ≡ [forward(B=1) × N].
// ────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "release-only — full model build required"]
fn parity_batch_consistency() {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");

    let bs = 4;
    let mut rng = Rng::new(0xc0de);
    let device = model.device.clone();
    let dtype = model.dtype;
    let n_bones = model.bone_count();

    // Per-batch phenotype: [B] random in [0.3, 0.7].
    let make_field = |rng: &mut Rng| -> Tensor {
        let v: Vec<f64> = (0..bs).map(|_| 0.3 + 0.4 * rng.next_f64()).collect();
        Tensor::from_vec(v, bs, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    };
    let phen_batched = PhenotypeValues {
        age: make_field(&mut rng),
        gender: make_field(&mut rng),
        muscle: make_field(&mut rng),
        weight: make_field(&mut rng),
        height: make_field(&mut rng),
        proportions: make_field(&mut rng),
        cupsize: make_field(&mut rng),
        firmness: make_field(&mut rng),
        african: make_field(&mut rng),
        asian: make_field(&mut rng),
        caucasian: make_field(&mut rng),
    };
    let pose_batched = random_rigid_batch(&mut rng, bs, n_bones, dtype, &device);

    let batched = model
        .forward(
            Some(&pose_batched),
            &phen_batched,
            Some(PoseParameterization::RestRelative),
        )
        .unwrap();
    let batched_v: Vec<f64> = batched.vertices.flatten_all().unwrap().to_vec1().unwrap();
    let v_per_sample = model.vertex_count() * 3;

    for i in 0..bs {
        // Single-batch slice.
        let single_phen = PhenotypeValues {
            age: phen_batched.age.narrow(0, i, 1).unwrap(),
            gender: phen_batched.gender.narrow(0, i, 1).unwrap(),
            muscle: phen_batched.muscle.narrow(0, i, 1).unwrap(),
            weight: phen_batched.weight.narrow(0, i, 1).unwrap(),
            height: phen_batched.height.narrow(0, i, 1).unwrap(),
            proportions: phen_batched.proportions.narrow(0, i, 1).unwrap(),
            cupsize: phen_batched.cupsize.narrow(0, i, 1).unwrap(),
            firmness: phen_batched.firmness.narrow(0, i, 1).unwrap(),
            african: phen_batched.african.narrow(0, i, 1).unwrap(),
            asian: phen_batched.asian.narrow(0, i, 1).unwrap(),
            caucasian: phen_batched.caucasian.narrow(0, i, 1).unwrap(),
        };
        let single_pose = pose_batched.narrow(0, i, 1).unwrap().contiguous().unwrap();
        let single = model
            .forward(
                Some(&single_pose),
                &single_phen,
                Some(PoseParameterization::RestRelative),
            )
            .unwrap();
        let single_v: Vec<f64> = single.vertices.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(single_v.len(), v_per_sample);

        let slice_start = i * v_per_sample;
        let slice = &batched_v[slice_start..slice_start + v_per_sample];
        let max = max_diff(slice, &single_v);
        assert!(max < 1e-9, "batch slice {i} differs by {max}");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 4. test_degenerate_configuration: the tongue02 bone is aligned with the Y
//    axis under one specific shape, hitting the degenerate-axis fallback
//    in `get_bone_poses`. Pin that shape, apply small random perturbations,
//    verify the resulting tongue02 rest pose stays continuous.
// ────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "release-only — full model build required"]
fn parity_degenerate_tongue02_continuity() {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");

    // Pinned "naughty" shape that aligns tongue02 with the Y axis.
    let naughty: &[(&str, f64)] = &[
        ("gender", 0.4645),
        ("age", 0.6078),
        ("muscle", 0.2637),
        ("weight", 0.7545),
        ("height", 0.5872),
        ("proportions", 0.7788),
        ("cupsize", 0.4095),
        ("firmness", 0.8335),
        ("african", 0.3333),
        ("asian", 0.3333),
        ("caucasian", 0.3333),
    ];
    let device = model.device.clone();
    let dtype = model.dtype;

    let make_phen = |values: &[(&str, f64)]| -> PhenotypeValues {
        let mut p = PhenotypeValues::defaults(dtype, &device).unwrap();
        for (label, v) in values {
            let t = Tensor::from_vec(vec![*v], 1, &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            match *label {
                "age" => p.age = t,
                "gender" => p.gender = t,
                "muscle" => p.muscle = t,
                "weight" => p.weight = t,
                "height" => p.height = t,
                "proportions" => p.proportions = t,
                "cupsize" => p.cupsize = t,
                "firmness" => p.firmness = t,
                "african" => p.african = t,
                "asian" => p.asian = t,
                "caucasian" => p.caucasian = t,
                _ => unreachable!(),
            }
        }
        p
    };

    let tongue02 = model
        .bone_labels
        .iter()
        .position(|n| n == "tongue02")
        .expect("default rig must have tongue02");

    let pinned_phen = make_phen(naughty);
    let pinned = model
        .forward(None, &pinned_phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let pinned_pose = extract_bone_pose(&pinned.rest_bone_poses, 0, tongue02);

    let mut rng = Rng::new(0xd0a1);
    let mut max_dev = 0.0_f64;
    for _ in 0..500 {
        let perturbed: Vec<(&str, f64)> = naughty
            .iter()
            .map(|(k, v)| (*k, v + (rng.next_f64() - 0.5) * 0.001))
            .collect();
        let phen = make_phen(&perturbed);
        let out = model
            .forward(None, &phen, Some(PoseParameterization::RestRelative))
            .unwrap();
        let pose = extract_bone_pose(&out.rest_bone_poses, 0, tongue02);
        let dev: f64 = pinned_pose
            .iter()
            .zip(pose.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    eprintln!("tongue02 max deviation across 500 perturbations: {max_dev:.6}");
    // Python uses 2e-3 absolute tolerance.
    assert!(max_dev < 2e-3, "tongue02 deviation {max_dev} exceeds 2e-3");
}

fn extract_bone_pose(rest_bone_poses: &Tensor, bi: usize, k: usize) -> Vec<f64> {
    // [B, K, 4, 4] → [16] for one element.
    let n_bones = rest_bone_poses.dim(1).unwrap();
    let flat: Vec<f64> = rest_bone_poses
        .to_dtype(DType::F64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let off = (bi * n_bones + k) * 16;
    flat[off..off + 16].to_vec()
}

// ────────────────────────────────────────────────────────────────────────────
// 5. test_local_changes: when no `local_changes` are passed, the local-change
//    coefficients are zero, so they should not affect the resulting vertices
//    or bone positions vs. a model loaded without local changes. We always
//    load local changes in our port (they're just zero-coeffed by default),
//    so this reduces to a sanity check that the local-change rows of
//    `blendshapes` don't bleed into the output.
// ────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "release-only — full model build required"]
fn parity_local_changes_zero_default() {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    let model = Model::build(&opts).expect("model");

    let mut rng = Rng::new(0xfee1);
    let device = model.device.clone();
    let dtype = model.dtype;
    let bs = 4;
    let make_field = |rng: &mut Rng| -> Tensor {
        let v: Vec<f64> = (0..bs).map(|_| 0.3 + 0.4 * rng.next_f64()).collect();
        Tensor::from_vec(v, bs, &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    };
    let phen = PhenotypeValues {
        age: make_field(&mut rng),
        gender: make_field(&mut rng),
        muscle: make_field(&mut rng),
        weight: make_field(&mut rng),
        height: make_field(&mut rng),
        proportions: make_field(&mut rng),
        cupsize: make_field(&mut rng),
        firmness: make_field(&mut rng),
        african: make_field(&mut rng),
        asian: make_field(&mut rng),
        caucasian: make_field(&mut rng),
    };

    // Get phenotype coefficients at full size (incl. zero-padded local change pairs).
    let coeffs_full = model.phenotype_coefficients(&phen).unwrap();
    let n_macro = model.stacked_phenotype_blend_shapes_mask.dim(0).unwrap();
    assert!(
        !model.local_change_labels.is_empty(),
        "expected local changes loaded"
    );

    // The trailing 2*n_local columns must all be zero.
    let coeffs_v: Vec<f64> = coeffs_full.flatten_all().unwrap().to_vec1().unwrap();
    let cols = coeffs_full.dim(1).unwrap();
    let mut max_local = 0.0_f64;
    for b in 0..bs {
        for c in n_macro..cols {
            max_local = max_local.max(coeffs_v[b * cols + c].abs());
        }
    }
    assert_eq!(
        max_local, 0.0,
        "local-change coeffs must be zero by default"
    );

    // Forward should also work and produce finite output.
    let out = model
        .forward(None, &phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let v: Vec<f64> = out.vertices.flatten_all().unwrap().to_vec1().unwrap();
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0);
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}
