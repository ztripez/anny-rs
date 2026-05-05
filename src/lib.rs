//! Rust port of NAVER Labs' [Anny](https://github.com/naver/anny), a
//! parametric human body mesh model. Mirrors the public Python entry points
//! in `anny/__init__.py`.
//!
//! # Quick start
//!
//! ```no_run
//! use anny_rs::models::full_model::{Model, ModelOptions, PoseParameterization};
//! use anny_rs::phenotype::PhenotypeValues;
//!
//! // `data_root` points at the MakeHuman/MPFB2 data shipped with upstream
//! // Anny — clone https://github.com/naver/anny and use
//! // `anny/src/anny/data/`.
//! let data_root = std::path::PathBuf::from("/path/to/anny/src/anny/data");
//!
//! let mut opts = ModelOptions::new(data_root);
//! opts.all_phenotypes = true;
//! let model = Model::build(&opts).expect("model build");
//!
//! let phen = PhenotypeValues::defaults(model.dtype, &model.device).unwrap();
//! let out = model
//!     .forward(None, &phen, Some(PoseParameterization::RestRelative))
//!     .unwrap();
//! // `out.vertices` is `[1, V, 3]` in metres.
//! ```
//!
//! # Modules
//!
//! - [`models::full_model`] — the central [`Model`](models::full_model::Model)
//!   struct + [`Model::build`](models::full_model::Model::build) factory.
//! - [`models::presets`] — `create_fullbody/hand/head_model` convenience
//!   constructors.
//! - [`models::retopology`] — SMPL-X retopology + alternative-topology
//!   barycentric remap.
//! - [`parameters_regressor`] — inverse fit `(target mesh) → (pose, phenotype)`.
//! - [`anthropometry`] — height/waist/volume/mass/BMI readouts.
//! - [`face_segmentation`] — body-part face masks from the bundled UV PNG.
//! - [`shape_distribution`] — Beta priors over phenotype scalars (sampling).
//! - [`phenotype`] — phenotype taxonomy + masked-product coefficient computation.
//! - [`kinematics`], [`skinning`], [`rotation`] — the differentiable forward
//!   chain primitives.
//! - [`data`] — `.obj`, `.target.gz`, and PyTorch `.pth` loaders.
//!
//! # Feature flags
//!
//! - `smplx-download` — enables `paths::download::fetch_noncommercial` and
//!   the `download-smplx` CLI subcommand. The SMPL-X retopology *consumer*
//!   code in [`models::retopology`] is always available.
//!
//! # Parity status
//!
//! See the crate-root [`README.md`](https://github.com/naver/anny) for the
//! verified-parity table. Forward pass matches Python at f64 ε on the data
//! layer and ~2e-7 on the geometric chain; the regressor hits 4.55 mm PVE
//! on the same fixed-shape test Python clears at 5 mm.

// Internal helpers regularly take many tensors at once and return tuples of
// tensor types — both styles trip these clippy lints, neither is worth
// abstracting under a `type` alias.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod anthropometry;
pub mod data;
pub mod face_segmentation;
pub mod kinematics;
pub mod models;
pub mod parameters_regressor;
pub mod paths;
pub mod phenotype;
pub mod rotation;
pub mod shape_distribution;
pub mod skinning;
pub mod utils;
