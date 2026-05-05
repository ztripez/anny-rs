//! Mesh utilities: quad → triangle splitting on the shorter diagonal.
//!
//! Mirrors the relevant subset of `anny/src/anny/utils/mesh_utils.py:7–44`.

/// Triangulates a face list using the shorter diagonal of each quad. Triangle
/// faces are passed through unchanged. Variable-arity faces with > 4 vertices
/// are not supported (they don't appear in the MakeHuman base mesh).
pub fn triangulate_faces(vertices: &[[f64; 3]], faces: &[Vec<u32>]) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(faces.len() * 2);
    for face in faces {
        match face.len() {
            3 => out.push([face[0], face[1], face[2]]),
            4 => {
                let a = vertices[face[0] as usize];
                let b = vertices[face[1] as usize];
                let c = vertices[face[2] as usize];
                let d = vertices[face[3] as usize];
                let diag1 = dist(&a, &c);
                let diag2 = dist(&b, &d);
                if diag1 < diag2 {
                    // split a-c
                    out.push([face[0], face[1], face[2]]);
                    out.push([face[2], face[3], face[0]]);
                } else {
                    // split b-d
                    out.push([face[0], face[1], face[3]]);
                    out.push([face[3], face[1], face[2]]);
                }
            }
            n => panic!("unexpected face arity {n}"),
        }
    }
    out
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Closest point on a triangle `(a, b, c)` to a query point `p`.
/// Returns `(closest_point, [u, v, w])` where `u + v + w == 1` and the
/// closest point equals `u·a + v·b + w·c`. Uses Ericson's reference
/// algorithm ("Real-Time Collision Detection" §5.1.5) — handles all 7
/// regions (3 vertex, 3 edge, 1 interior) without numerical traps.
pub fn closest_point_on_triangle(
    p: &[f64; 3],
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let sub = |x: &[f64; 3], y: &[f64; 3]| -> [f64; 3] { [x[0] - y[0], x[1] - y[1], x[2] - y[2]] };
    let dot = |x: &[f64; 3], y: &[f64; 3]| -> f64 { x[0] * y[0] + x[1] * y[1] + x[2] * y[2] };

    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(&ab, &ap);
    let d2 = dot(&ac, &ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (*a, [1.0, 0.0, 0.0]);
    }

    let bp = sub(p, b);
    let d3 = dot(&ab, &bp);
    let d4 = dot(&ac, &bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (*b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        return (q, [1.0 - v, v, 0.0]);
    }

    let cp = sub(p, c);
    let d5 = dot(&ab, &cp);
    let d6 = dot(&ac, &cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (*c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        return (q, [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub(c, b);
        let q = [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
        return (q, [0.0, 1.0 - w, w]);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    (q, [1.0 - v - w, v, w])
}

/// For each query point, find the closest triangle in the mesh
/// `(vertices, faces)` and the barycentric coordinates of the closest
/// point inside that triangle. Brute force `O(P × F)` — meant for the
/// small `data/topology/*.obj` reference meshes.
///
/// Returns `(distances, face_ids, barycentric)` of length `points.len()`.
pub fn point_to_mesh_distance_and_face_uvs(
    points: &[[f64; 3]],
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
) -> (Vec<f64>, Vec<u32>, Vec<[f64; 3]>) {
    let mut distances = Vec::with_capacity(points.len());
    let mut face_ids = Vec::with_capacity(points.len());
    let mut bary = Vec::with_capacity(points.len());
    for p in points {
        let mut best_d2 = f64::INFINITY;
        let mut best_face: u32 = 0;
        let mut best_bary = [1.0_f64, 0.0, 0.0];
        for (fi, tri) in faces.iter().enumerate() {
            let a = &vertices[tri[0] as usize];
            let b = &vertices[tri[1] as usize];
            let c = &vertices[tri[2] as usize];
            let (q, b_uvw) = closest_point_on_triangle(p, a, b, c);
            let dx = p[0] - q[0];
            let dy = p[1] - q[1];
            let dz = p[2] - q[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_face = fi as u32;
                best_bary = b_uvw;
            }
        }
        distances.push(best_d2.sqrt());
        face_ids.push(best_face);
        bary.push(best_bary);
    }
    (distances, face_ids, bary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_triangles_through() {
        let verts = vec![[0.0; 3]; 3];
        let faces = vec![vec![0u32, 1, 2]];
        let tri = triangulate_faces(&verts, &faces);
        assert_eq!(tri, vec![[0, 1, 2]]);
    }

    #[test]
    fn splits_quad_on_shorter_diagonal() {
        // Long-thin quad: |a-c| < |b-d| → split along a-c.
        let verts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [11.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
        ];
        let faces = vec![vec![0u32, 1, 2, 3]];
        let tri = triangulate_faces(&verts, &faces);
        assert_eq!(tri, vec![[0, 1, 2], [2, 3, 0]]);
    }

    #[test]
    fn closest_point_at_vertex_a() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [-1.0, -1.0, 0.5];
        let (q, bary) = closest_point_on_triangle(&p, &a, &b, &c);
        assert!((q[0] - a[0]).abs() < 1e-12);
        assert!((bary[0] - 1.0).abs() < 1e-12);
        assert_eq!(bary[1], 0.0);
        assert_eq!(bary[2], 0.0);
    }

    #[test]
    fn closest_point_inside_triangle() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.25, 0.25, 5.0]; // directly above the centroid-ish point
        let (q, bary) = closest_point_on_triangle(&p, &a, &b, &c);
        assert!((q[0] - 0.25).abs() < 1e-12);
        assert!((q[1] - 0.25).abs() < 1e-12);
        assert!(q[2].abs() < 1e-12);
        let bary_sum: f64 = bary.iter().sum();
        assert!((bary_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn closest_point_on_edge_ab() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.5, -1.0, 0.0];
        let (q, bary) = closest_point_on_triangle(&p, &a, &b, &c);
        assert!((q[0] - 0.5).abs() < 1e-12);
        assert!(q[1].abs() < 1e-12);
        assert_eq!(bary[2], 0.0);
        assert!((bary[0] + bary[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn splits_quad_on_other_diagonal() {
        // Make b-d shorter than a-c.
        let verts = vec![
            [-10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ];
        let faces = vec![vec![0u32, 1, 2, 3]];
        let tri = triangulate_faces(&verts, &faces);
        // |a-c|=20, |b-d|≈0.1 → split b-d → triangles (a,b,d), (d,b,c).
        assert_eq!(tri, vec![[0, 1, 3], [3, 1, 2]]);
    }
}
