use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};

use anny_rs::data::obj as obj_io;
use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization, SkinningMethod};
use anny_rs::parameters_regressor::{Regressor, RegressorOptions};
use anny_rs::phenotype::PhenotypeValues;
use candle_core::{DType, Device, Tensor};

#[derive(Parser)]
#[command(name = "anny-cli", about = "Anny body model CLI", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the model with the supplied phenotype + pose, write the posed mesh as an OBJ.
    Pose {
        /// Path to the bundled `data/` directory (e.g. anny/src/anny/data).
        #[arg(long)]
        data_root: PathBuf,
        /// JSON file mapping phenotype label → scalar in [0, 1] (or [-1/3, 1] for age).
        /// Missing entries default to 0.5. Example:
        /// `{"age": 0.5, "height": 0.5, "muscle": 0.6}`.
        #[arg(long)]
        phenotype: Option<PathBuf>,
        /// Output topology. `default` is the MakeHuman base mesh; `smplx`
        /// retopologises to SMPL-X and requires the `smplx-download` feature
        /// + a one-time `download-smplx` + Python conversion.
        #[arg(long, default_value = "default")]
        topology: String,
        /// Compute device. `cpu` (default) or `cuda:N` (requires
        /// `--features cuda`). Example: `--device cuda:0`.
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Where to write the posed mesh.
        #[arg(long)]
        out: PathBuf,
    },
    /// Fit phenotype + pose to a target OBJ via the parameters regressor.
    Fit {
        #[arg(long)]
        data_root: PathBuf,
        /// Target mesh (OBJ).
        #[arg(long)]
        target: PathBuf,
        /// Where to write the fitted mesh.
        #[arg(long)]
        out: PathBuf,
        /// Number of regressor iterations (default 5).
        #[arg(long, default_value_t = 5)]
        iters: usize,
        /// Compute device. `cpu` (default) or `cuda:N`.
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// Download the non-commercial SMPL-X retopology data into the cache.
    DownloadSmplx,
}

fn parse_device(s: &str) -> anyhow::Result<Device> {
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let id: usize = rest.parse().context("cuda device index")?;
        #[cfg(feature = "cuda")]
        {
            return Device::new_cuda(id).map_err(|e| anyhow!("cuda init: {e}"));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = id;
            return Err(anyhow!(
                "cuda support not compiled in; rebuild with `--features cuda`"
            ));
        }
    }
    Err(anyhow!("unknown device '{s}'; expected `cpu` or `cuda:N`"))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Pose {
            data_root,
            phenotype,
            topology,
            device,
            out,
        } => {
            let device = parse_device(&device)?;
            run_pose(&data_root, phenotype.as_deref(), &topology, &device, &out)
        }
        Cmd::Fit {
            data_root,
            target,
            out,
            iters,
            device,
        } => {
            let device = parse_device(&device)?;
            run_fit(&data_root, &target, &out, iters, &device)
        }
        Cmd::DownloadSmplx => run_download_smplx(),
    }
}

fn run_pose(
    data_root: &std::path::Path,
    phenotype_path: Option<&std::path::Path>,
    topology: &str,
    device: &Device,
    out: &std::path::Path,
) -> anyhow::Result<()> {
    let mut opts = ModelOptions::new(data_root);
    opts.all_phenotypes = true;
    opts.skinning_method = SkinningMethod::Lbs;
    opts.default_pose_parameterization = PoseParameterization::RestRelative;
    opts.device = device.clone();

    eprintln!("loading model...");
    let model = build_model_for_topology(&opts, topology)?;
    eprintln!(
        "model: {} bones, {} vertices, {} faces",
        model.bone_count(),
        model.vertex_count(),
        model.faces.len()
    );

    let phen = read_phenotype(phenotype_path, model.dtype, &model.device)?;
    let forward = model
        .forward(None, &phen, None)
        .map_err(|e| anyhow!("forward: {e}"))?;
    write_obj(out, &forward.vertices, &model.faces)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn build_model_for_topology(opts: &ModelOptions, topology: &str) -> anyhow::Result<Model> {
    match topology {
        "default" => Model::build(opts).map_err(|e| anyhow!("model build: {e}")),
        "smplx" => anny_rs::models::retopology::create_smplx_topology_model(opts)
            .map_err(|e| anyhow!("smplx model: {e}")),
        // Alternative topologies live in `<data_root>/topology/<name>.obj` —
        // closest-triangle barycentric remap from the reference Anny mesh.
        other => anny_rs::models::retopology::create_alternative_topology_model(opts, other)
            .map_err(|e| anyhow!("alt topology '{other}': {e}")),
    }
}

fn run_fit(
    data_root: &std::path::Path,
    target_path: &std::path::Path,
    out: &std::path::Path,
    iters: usize,
    device: &Device,
) -> anyhow::Result<()> {
    let mut opts = ModelOptions::new(data_root);
    opts.all_phenotypes = true;
    opts.skinning_method = SkinningMethod::Lbs;
    opts.device = device.clone();
    eprintln!("loading model...");
    let model = Model::build(&opts).map_err(|e| anyhow!("model build: {e}"))?;
    eprintln!(
        "model: {} bones, {} vertices",
        model.bone_count(),
        model.vertex_count()
    );

    eprintln!("loading target {} ...", target_path.display());
    let target_mesh = obj_io::load(target_path).context("load target")?;
    if target_mesh.vertices.len() != model.vertex_count() {
        return Err(anyhow!(
            "target has {} vertices, model expects {}",
            target_mesh.vertices.len(),
            model.vertex_count()
        ));
    }
    let target_flat: Vec<f64> = target_mesh
        .vertices
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    let target = Tensor::from_vec(
        target_flat,
        (1, target_mesh.vertices.len(), 3),
        &model.device,
    )?
    .to_dtype(model.dtype)?;

    let reg_opts = RegressorOptions {
        max_n_iters: iters,
        verbose: true,
        ..Default::default()
    };
    let reg = Regressor::new(&model, reg_opts).map_err(|e| anyhow!("regressor: {e}"))?;
    let result = reg
        .fit(
            &target,
            &["cupsize", "firmness", "african", "asian", "caucasian"],
        )
        .map_err(|e| anyhow!("fit: {e}"))?;
    write_obj(out, &result.vertices, &model.faces)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn run_download_smplx() -> anyhow::Result<()> {
    #[cfg(feature = "smplx-download")]
    {
        anny_rs::paths::download::fetch_noncommercial()
            .map_err(|e| anyhow!("smplx download: {e}"))?;
        eprintln!(
            "downloaded to {}",
            anny_rs::paths::anny2smplx_path().display()
        );
        Ok(())
    }
    #[cfg(not(feature = "smplx-download"))]
    {
        Err(anyhow!(
            "rebuild with `--features smplx-download` to enable this subcommand"
        ))
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

fn read_phenotype(
    path: Option<&std::path::Path>,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<PhenotypeValues> {
    let mut phen = PhenotypeValues::defaults(dtype, device).map_err(|e| anyhow!("{e}"))?;
    let Some(path) = path else { return Ok(phen) };
    let text = std::fs::read_to_string(path).context("read phenotype JSON")?;
    let map: HashMap<String, f64> = serde_json::from_str(&text).context("parse phenotype JSON")?;
    for (k, v) in map {
        let t = Tensor::from_vec(vec![v], 1, device)?.to_dtype(dtype)?;
        match k.as_str() {
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
            other => return Err(anyhow!("unknown phenotype label: {other}")),
        }
    }
    Ok(phen)
}

fn write_obj(path: &std::path::Path, vertices: &Tensor, faces: &[Vec<u32>]) -> anyhow::Result<()> {
    let dims = vertices.dims();
    let n_verts = if dims.len() == 3 { dims[1] } else { dims[0] };
    let flat: Vec<f64> = vertices.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
    let mut verts: Vec<[f64; 3]> = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        verts.push([flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]);
    }
    obj_io::save(path, &verts, faces).map_err(|e| anyhow!("write OBJ: {e}"))?;
    Ok(())
}
