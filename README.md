# anny-rs

A Rust port of NAVER Labs' [Anny](https://github.com/naver/anny) — a parametric
human-body mesh model originally written in Python + PyTorch
(arXiv [2511.03589](https://arxiv.org/abs/2511.03589)). Built on
[candle](https://github.com/huggingface/candle); no Python or PyTorch runtime
required at use-time.

## Status

| Layer | Parity vs Python |
|---|---|
| Data loaders (OBJ, `.target.gz`, `.pth`) | bit-perfect |
| Phenotype coefficients | f64 ε (1.11e-16) |
| Forward pass (rest vertices) | f64 ε (2.22e-16) |
| Forward pass (posed vertices) | 2.05e-7 (atan2/cross path) |
| Anthropometry (height / waist / volume / mass / BMI) | machine precision |
| Face segmentation | 0 differences |
| Hand / head / SMPL-X presets | exact bone & face counts |
| Pose-parameterization round-trips (all 16 mode pairs) | f64 ε |
| Parameters regressor PVE on a known target | 4.55 mm (Python tolerance: 5 mm) |
| Age-anchor sweep PVE | 0.93 mm on the same target |
| `topology="default"` face edit | exact face count (13346) |
| `remove_unattached_vertices` | exact vertex / face counts (13348 / 13346) |
| Tongue02 degenerate-axis stability | < 0.0012 (Python tolerance: 0.002) |

**70+ tests pass** across debug + release; release runs the heavy ones (full
model build is ~7 s for ~650 `.target.gz` files + ~565 stacked blend shapes).

## What's not ported

- **Differentiability.** The port uses no autograd. Forward + regressor work,
  but you cannot embed Anny inside a larger PyTorch-style training loop with
  gradients flowing through the body. (Python's `ParametersRegressor` is
  `@torch.no_grad()` end-to-end, so fitting is unaffected.)
- **NVIDIA Warp acceleration paths.** Pure LBS / DQS skinning + a brute-force
  closest-triangle search cover the same correctness contract; GPU
  acceleration is a separate effort.

## Quick start

```rust
use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization};
use anny_rs::phenotype::PhenotypeValues;

// Path to the bundled MakeHuman/MPFB2 data shipped with upstream Anny.
// Cloning the upstream Python repo gives you `anny/src/anny/data/`.
let data_root = "path/to/anny/src/anny/data";

let mut opts = ModelOptions::new(data_root);
opts.all_phenotypes = true;
let model = Model::build(&opts).expect("model build");

let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
let out = model
    .forward(None, &phen, Some(PoseParameterization::RestRelative))
    .unwrap();
// out.vertices is [1, V, 3] in metres.
```

### Fitting

```rust
use anny_rs::parameters_regressor::{Regressor, RegressorOptions};

let reg = Regressor::new(&model, RegressorOptions::default()).unwrap();
let result = reg.fit(&target_vertices, &[]).unwrap();
// result.phenotype: recovered phenotype scalars.
// result.pose_parameters: [B, K, 4, 4] in the model's default mode.
// result.vertices: [B, V, 3] fitted vertices.

// For better age recovery (when the target's age differs significantly
// from the regressor's defaults), use the anchor sweep:
let result = reg.fit_with_age_anchor_search(&target_vertices, &[0.0, 0.33, 0.67, 1.0]).unwrap();
```

### Anthropometry, segmentation, sampling

```rust
let anth = anny_rs::anthropometry::Anthropometry::new(&model).unwrap();
let m = anth.measurements(&out.rest_vertices).unwrap();
// m.height, m.waist_circumference, m.volume, m.mass, m.bmi (all [B] tensors).

let seg = anny_rs::face_segmentation::FaceSegmentation::new(&model, &data_root.into()).unwrap();
let head_mask = seg.face_mask(&["head"]).unwrap(); // Vec<bool> per face.

let mut rng = rand::rngs::StdRng::seed_from_u64(0xfeed);
let dist = anny_rs::shape_distribution::SimpleShapeDistribution::load_default(
    data_root.as_ref(), model.phenotype_labels().iter().map(|s| s.to_string()).collect(),
    model.dtype, &model.device,
).unwrap();
let (morph_age, sampled_phen) = dist.sample(64, &mut rng).unwrap();
```

### SMPL-X retopology

The 10,475-vertex SMPL-X mesh layout is non-commercial and downloaded
separately. Once on disk, the Rust path is feature-free:

```rust
let smplx_model = anny_rs::models::retopology::create_smplx_topology_model(&opts).unwrap();
```

### Alternative topology

For decimated reference meshes shipped under `data/topology/*.obj`:

```rust
let model = anny_rs::models::retopology::create_alternative_topology_model(
    &opts, "notoes_collapse10pc"
).unwrap();
```

## CLI

A thin binary wraps the library:

```bash
# Build (with optional SMPL-X auto-downloader).
cargo build --release --features smplx-download --bin anny-cli

# Pose: phenotype JSON in, posed-mesh OBJ out.
echo '{"height": 0.8, "weight": 0.7}' > phen.json
./target/release/anny-cli pose \
    --data-root path/to/anny/src/anny/data \
    --phenotype phen.json \
    --out posed.obj

# SMPL-X (after one-time download + Python conversion, see below).
./target/release/anny-cli pose \
    --data-root path/to/anny/src/anny/data \
    --topology smplx \
    --out smplx.obj

# Alternative decimated topology.
./target/release/anny-cli pose \
    --data-root path/to/anny/src/anny/data \
    --topology notoes_collapse10pc \
    --out decimated.obj

# Fit phenotype + pose to an external mesh.
./target/release/anny-cli fit \
    --data-root path/to/anny/src/anny/data \
    --target subject.obj \
    --out fitted.obj
```

## Feature flags

| Flag | Effect |
|---|---|
| `smplx-download` | Enables `paths::download::fetch_noncommercial` (HTTPS download via reqwest+rustls) and the `download-smplx` CLI subcommand. The SMPL-X retopology *consumer* code is always available — this flag only enables auto-download. |

## Data dependencies

The Rust crate ships zero data assets. You point it at a `data_root` that
matches the upstream Python package's `anny/src/anny/data/` layout:

```
data_root/
├── mpfb2/
│   ├── 3dobjs/base.obj                # 19,158-vertex MakeHuman base mesh
│   ├── rigs/standard/{rig,weights}.*.json
│   └── targets/                       # ~650 .target.gz blend shapes + target.json
├── topology/                          # alternative-topology .obj files
├── segmentation/                      # body-part PNG + label YAML
└── shape_calibration/                 # boys.pth, girls.pth (sampling priors)
```

The simplest way to get this is to clone the
[upstream Python repo](https://github.com/naver/anny) — the data is
CC0-licensed (MakeHuman/MPFB2 origins) and ships in-tree.

### SMPL-X (optional)

The SMPL-X retopology weights are non-commercial and not shipped in-tree:

```bash
# 1. Auto-download the non-commercial bundle (gated cargo feature).
cargo run --features smplx-download --bin anny-cli -- download-smplx

# 2. One-time Python conversion: candle's pickle reader doesn't handle
#    Python lists, so we reshape into a candle-friendly safetensors file.
.venv/bin/python tests/fixtures/convert_smplx.py
```

After this, `--topology smplx` works and the auto-downloader feature flag
is no longer needed.

## Verifying parity

A Python venv at `anny/.venv/` (sibling crate) holds upstream Anny in editable
mode. The fixture export script regenerates every golden tensor:

```bash
.venv/bin/python tests/fixtures/export_golden.py
cargo test --release -- --include-ignored
```

## Architecture (brief)

- `data::{obj, target_gz, pickle}` — loaders for the on-disk MakeHuman /
  PyTorch state-dict / image-segmentation files.
- `rotation` — hand-port of the seven `roma` functions Anny uses (rotvec
  ↔ rotmat ↔ unit quat, rigid transforms, weighted Kabsch).
- `kinematics` — propagation-front forward kinematics and bone-pose
  construction with degenerate-axis fallback.
- `skinning` — LBS + DQS, both candle-tensor-native.
- `phenotype` — `PHENOTYPE_VARIATIONS` taxonomy and the masked-product
  coefficient computation that turns 11 phenotype scalars into ~624 blend
  shape weights.
- `shape_distribution` — Beta priors per gender, calibrated against WHO
  height-for-age data.
- `models::full_model` — central `Model` struct + `Model::build` that ties
  the loaders and forward chain together.
- `models::{presets, retopology}` — fullbody/hand/head presets and
  SMPL-X / alternative-topology retopologies.
- `parameters_regressor` — alternating fit (per-bone Kabsch + Tikhonov
  finite-difference Jacobian solve) plus an age-anchor sweep.
- `anthropometry` — height / waist / volume / mass / BMI readouts.
- `face_segmentation` — UV-barycenter PNG sampling for body-part masks.

## Licensing

- Code: Apache-2.0 (matches upstream Python).
- MakeHuman/MPFB2 assets (the `data/mpfb2/` tree): CC0 — see the upstream
  repository's `data/mpfb2/LICENSE.md`.
- SMPL-X retopology weights: **non-commercial only** — see the
  `LICENSE.txt` / `NOTICE.txt` extracted alongside `anny2smplx.pth`.

## Acknowledgements

This port mirrors NAVER's open release. Original authors:
Romain Brégier, Guénolé Fiche, Laura Bravo-Sánchez, Thomas Lucas,
Matthieu Armando, Philippe Weinzaepfel, Grégory Rogez, Fabien Baradel.

```
@misc{bregier2025humanmeshmodelinganny,
    title  = {Human Mesh Modeling for Anny Body},
    author = {Romain Br\'egier and Gu\'enol\'e Fiche and Laura Bravo-S\'anchez and
              Thomas Lucas and Matthieu Armando and Philippe Weinzaepfel and
              Gr\'egory Rogez and Fabien Baradel},
    year   = {2025},
    eprint = {2511.03589},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
}
```
