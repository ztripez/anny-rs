//! Anthropometric readouts: height, waist circumference, volume, mass, BMI.
//!
//! Direct port of `anny/src/anny/anthropometry.py`. All measurements are
//! computed from rest-pose vertices in metres (post-world-transformation).

use candle_core::{D, Result, Tensor};

use crate::models::full_model::Model;
use crate::utils::mesh::triangulate_faces;

/// Hardcoded loop of waist vertices in the MakeHuman base mesh, in
/// circumferential order. Lifted verbatim from `anny/src/anny/anthropometry.py:6`.
const BASE_MESH_WAIST_VERTICES: &[u32] = &[
    4121, 10763, 10760, 10757, 10777, 10776, 10779, 10780, 10778, 10781, 10771, 10773, 10772,
    10775, 10774, 10814, 10834, 10816, 10817, 10818, 10819, 10820, 10821, 4181, 4180, 4179, 4178,
    4177, 4176, 4175, 4196, 4173, 4131, 4132, 4129, 4130, 4128, 4138, 4135, 4137, 4136, 4133, 4134,
    4108, 4113, 4118,
];

/// Density assumed by `mass()`. Mirrors Python's hard-coded 980 kg/m³
/// (slightly less than water; reasonable average for a human body).
const DEFAULT_DENSITY: f64 = 980.0;

pub struct Anthropometry {
    /// Indices into `model.template_vertices` for the waist loop.
    waist_vertex_indices: Vec<u32>,
    /// Pre-triangulated face indices, `[F, 3]`.
    triangular_faces: Vec<[u32; 3]>,
}

#[derive(Debug, Clone)]
pub struct Measurements {
    /// Body height (Z extent) in metres.
    pub height: Tensor,
    /// Waist circumference (perimeter of the waist loop) in metres.
    pub waist_circumference: Tensor,
    /// Body volume in cubic metres (signed shoelace, absolute value).
    pub volume: Tensor,
    /// Body mass = volume × density (kg).
    pub mass: Tensor,
    /// BMI = mass / height².
    pub bmi: Tensor,
}

impl Anthropometry {
    /// Builds the readout helper for a given [`Model`]. We don't take a
    /// reference to the model — we just snapshot the triangulation and the
    /// waist-vertex map at construction time.
    ///
    /// Returns `None` if `model.base_mesh_vertex_indices` is set (i.e.
    /// `remove_unattached_vertices` was used) but at least one waist vertex
    /// was pruned. In that case anthropometry over the waist loop is
    /// undefined for the model and the helper cannot be constructed.
    pub fn new(model: &Model) -> Option<Self> {
        // Snapshot the template vertices to compute triangulation diagonals.
        let n_verts = model.vertex_count();
        let template_flat: Vec<f64> = model
            .template_vertices
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let template: Vec<[f64; 3]> = (0..n_verts)
            .map(|i| {
                [
                    template_flat[i * 3],
                    template_flat[i * 3 + 1],
                    template_flat[i * 3 + 2],
                ]
            })
            .collect();

        // Map the hardcoded waist vertices through the base-mesh remap if any.
        let waist_vertex_indices = match &model.base_mesh_vertex_indices {
            None => BASE_MESH_WAIST_VERTICES.to_vec(),
            Some(remap) => {
                // remap[new] = old. Invert: build old → new.
                let max_old = *remap.iter().max().unwrap_or(&0) as usize;
                let mut old_to_new = vec![-1_i64; max_old + 1];
                for (new_i, &old) in remap.iter().enumerate() {
                    old_to_new[old as usize] = new_i as i64;
                }
                let mut out = Vec::with_capacity(BASE_MESH_WAIST_VERTICES.len());
                for &v in BASE_MESH_WAIST_VERTICES {
                    let new_v = old_to_new.get(v as usize).copied().unwrap_or(-1);
                    if new_v < 0 {
                        return None;
                    }
                    out.push(new_v as u32);
                }
                out
            }
        };

        let triangular_faces = triangulate_faces(&template, &model.faces);
        Some(Self {
            waist_vertex_indices,
            triangular_faces,
        })
    }

    /// Z-extent of the input vertices: `max(z) - min(z)` per batch.
    pub fn height(&self, rest_vertices: &Tensor) -> Result<Tensor> {
        let z = rest_vertices.narrow(D::Minus1, 2, 1)?.squeeze(D::Minus1)?; // [B, V]
        let max_z = z.max(D::Minus1)?;
        let min_z = z.min(D::Minus1)?;
        max_z.sub(&min_z)
    }

    /// Perimeter of the waist loop: `Σ_i ‖v[i+1] - v[i]‖`.
    pub fn waist_circumference(&self, rest_vertices: &Tensor) -> Result<Tensor> {
        let device = rest_vertices.device().clone();
        let idx_tensor = Tensor::from_vec(
            self.waist_vertex_indices.clone(),
            self.waist_vertex_indices.len(),
            &device,
        )?;
        let waist = rest_vertices.index_select(&idx_tensor, 1)?; // [B, W, 3]
        // Roll along W axis by 1, then take element-wise differences.
        let n = waist.dim(1)?;
        let last = waist.narrow(1, n - 1, 1)?;
        let head = waist.narrow(1, 0, n - 1)?;
        let rolled = Tensor::cat(&[&last, &head], 1)?; // shifted
        let edges = (rolled - waist)?; // [B, W, 3]
        let lens = edges.sqr()?.sum(D::Minus1)?.sqrt()?; // [B, W]
        lens.sum(D::Minus1)
    }

    /// Closed-mesh signed volume via the shoelace formula.
    /// `V = |Σ_f ((v0 × v1) · v2) / 6|`
    pub fn volume(&self, rest_vertices: &Tensor) -> Result<Tensor> {
        let device = rest_vertices.device().clone();
        let f = self.triangular_faces.len();
        let f0: Vec<u32> = self.triangular_faces.iter().map(|t| t[0]).collect();
        let f1: Vec<u32> = self.triangular_faces.iter().map(|t| t[1]).collect();
        let f2: Vec<u32> = self.triangular_faces.iter().map(|t| t[2]).collect();
        let idx0 = Tensor::from_vec(f0, f, &device)?;
        let idx1 = Tensor::from_vec(f1, f, &device)?;
        let idx2 = Tensor::from_vec(f2, f, &device)?;
        let v0 = rest_vertices.index_select(&idx0, 1)?; // [B, F, 3]
        let v1 = rest_vertices.index_select(&idx1, 1)?;
        let v2 = rest_vertices.index_select(&idx2, 1)?;
        let cross = cross_3d_last(&v0, &v1)?; // [B, F, 3]
        let signed = cross
            .broadcast_mul(&v2)?
            .sum(D::Minus1)?
            .affine(1.0 / 6.0, 0.0)?; // [B, F]
        let volume_signed = signed.sum(D::Minus1)?; // [B]
        volume_signed.abs()
    }

    pub fn mass(&self, rest_vertices: &Tensor) -> Result<Tensor> {
        self.volume(rest_vertices)?.affine(DEFAULT_DENSITY, 0.0)
    }

    pub fn bmi(&self, rest_vertices: &Tensor) -> Result<Tensor> {
        let h = self.height(rest_vertices)?;
        let m = self.mass(rest_vertices)?;
        m.div(&h.sqr()?)
    }

    pub fn measurements(&self, rest_vertices: &Tensor) -> Result<Measurements> {
        Ok(Measurements {
            height: self.height(rest_vertices)?,
            waist_circumference: self.waist_circumference(rest_vertices)?,
            volume: self.volume(rest_vertices)?,
            mass: self.mass(rest_vertices)?,
            bmi: self.bmi(rest_vertices)?,
        })
    }
}

fn cross_3d_last(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let last = a.rank() - 1;
    let ax = a.narrow(last, 0, 1)?;
    let ay = a.narrow(last, 1, 1)?;
    let az = a.narrow(last, 2, 1)?;
    let bx = b.narrow(last, 0, 1)?;
    let by = b.narrow(last, 1, 1)?;
    let bz = b.narrow(last, 2, 1)?;
    let cx = (ay.broadcast_mul(&bz)? - az.broadcast_mul(&by)?)?;
    let cy = (az.broadcast_mul(&bx)? - ax.broadcast_mul(&bz)?)?;
    let cz = (ax.broadcast_mul(&by)? - ay.broadcast_mul(&bx)?)?;
    Tensor::cat(&[&cx, &cy, &cz], last)
}
