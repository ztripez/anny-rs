//! Numerical parity between the CPU and CUDA forward passes. Gated behind
//! the `cuda` cargo feature; further `#[ignore]`d so the test suite runs
//! green on machines without a GPU.
//!
//!   cargo test --release --features cuda --test cuda_parity -- --include-ignored

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization};
use anny_rs::phenotype::PhenotypeValues;
use candle_core::Device;

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("anny")
        .join("src")
        .join("anny")
        .join("data")
}

fn build(device: Device) -> Model {
    let mut opts = ModelOptions::new(data_root());
    opts.all_phenotypes = true;
    opts.device = device;
    Model::build(&opts).expect("model build")
}

#[test]
#[ignore = "release-only — requires CUDA-capable GPU"]
fn cuda_forward_matches_cpu() {
    let cpu = build(Device::Cpu);
    let gpu = build(Device::new_cuda(0).expect("cuda init"));

    let cpu_phen = PhenotypeValues::defaults(cpu.dtype, &cpu.device).unwrap();
    let gpu_phen = PhenotypeValues::defaults(gpu.dtype, &gpu.device).unwrap();

    let cpu_out = cpu
        .forward(None, &cpu_phen, Some(PoseParameterization::RestRelative))
        .unwrap();
    let gpu_out = gpu
        .forward(None, &gpu_phen, Some(PoseParameterization::RestRelative))
        .unwrap();

    let cpu_v: Vec<f64> = cpu_out.vertices.flatten_all().unwrap().to_vec1().unwrap();
    let gpu_v: Vec<f64> = gpu_out
        .vertices
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(cpu_v.len(), gpu_v.len());

    let max = cpu_v
        .iter()
        .zip(gpu_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("CPU vs CUDA max abs err: {max:.3e}");
    // Same f64 input data, deterministic ops, host-side trig fallbacks
    // — agreement should be at f64 ε.
    assert!(max < 1e-12, "CPU vs CUDA divergence: {max}");
}
