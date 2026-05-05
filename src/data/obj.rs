//! OBJ mesh load/save. Faithful port of `anny/utils/obj_utils.py`.
//!
//! Faces preserve their original arity (triangles or quads) — the MakeHuman
//! base mesh contains both, and the consumer decides whether to triangulate.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObjError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed obj at line {line}: {msg}")]
    Malformed { line: usize, msg: String },
}

/// One named group's face index lists. Faces are variable-arity.
#[derive(Debug, Default, Clone)]
pub struct ObjGroup {
    pub face_vertex_indices: Vec<Vec<u32>>,
    pub face_texture_coordinate_indices: Vec<Vec<u32>>,
}

#[derive(Debug, Default, Clone)]
pub struct ObjMesh {
    pub vertices: Vec<[f64; 3]>,
    pub texture_coordinates: Vec<[f64; 2]>,
    /// Insertion-ordered group map. The first (default) group is named `"noname"`,
    /// matching the Python loader.
    pub groups: BTreeMap<String, ObjGroup>,
    /// Group order as encountered in the file — `BTreeMap` is alphabetic, so we
    /// keep this so callers (and round-trips) can preserve order.
    pub group_order: Vec<String>,
}

pub fn load(path: impl AsRef<Path>) -> Result<ObjMesh, ObjError> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);

    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut tex: Vec<[f64; 2]> = Vec::new();
    let mut groups: BTreeMap<String, ObjGroup> = BTreeMap::new();
    let mut group_order: Vec<String> = Vec::new();

    let mut current = String::from("noname");
    let mut current_group = ObjGroup::default();
    let mut current_known = false;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let starts_mtllib = line.starts_with("mtllib");
        if line.starts_with('#') && !starts_mtllib {
            continue;
        }
        let mut parts = line.split_whitespace();
        let kw = match parts.next() {
            Some(k) => k,
            None => continue,
        };
        match kw {
            "o" => {
                // Python: stop at second `o` once vertices have started.
                if !vertices.is_empty() {
                    break;
                }
            }
            "v" => {
                let coords: Result<Vec<f64>, _> = parts.map(|x| x.parse::<f64>()).collect();
                let coords = coords.map_err(|e| ObjError::Malformed {
                    line: lineno + 1,
                    msg: format!("vertex parse: {e}"),
                })?;
                if coords.len() != 3 {
                    return Err(ObjError::Malformed {
                        line: lineno + 1,
                        msg: format!("vertex needs 3 coords, got {}", coords.len()),
                    });
                }
                vertices.push([coords[0], coords[1], coords[2]]);
            }
            "vt" => {
                let coords: Result<Vec<f64>, _> = parts.map(|x| x.parse::<f64>()).collect();
                let coords = coords.map_err(|e| ObjError::Malformed {
                    line: lineno + 1,
                    msg: format!("texture coord parse: {e}"),
                })?;
                if coords.len() != 2 {
                    return Err(ObjError::Malformed {
                        line: lineno + 1,
                        msg: format!("vt needs 2 coords, got {}", coords.len()),
                    });
                }
                tex.push([coords[0], coords[1]]);
            }
            "g" => {
                if !current_group.face_vertex_indices.is_empty() {
                    groups.insert(current.clone(), std::mem::take(&mut current_group));
                    if !group_order.contains(&current) {
                        group_order.push(current.clone());
                    }
                }
                let name = parts.next().unwrap_or("noname").to_string();
                current = name.clone();
                if let Some(existing) = groups.remove(&name) {
                    current_group = existing;
                    current_known = true;
                } else {
                    current_group = ObjGroup::default();
                    current_known = false;
                }
            }
            "f" => {
                let mut vids: Vec<u32> = Vec::new();
                let mut vtids: Vec<u32> = Vec::new();
                for spec in parts {
                    let mut iter = spec.split('/');
                    let v = iter
                        .next()
                        .and_then(|s| s.parse::<i64>().ok())
                        .ok_or_else(|| ObjError::Malformed {
                            line: lineno + 1,
                            msg: format!("face vertex index: {spec}"),
                        })?;
                    vids.push((v - 1) as u32);
                    if let Some(vt) = iter.next()
                        && !vt.is_empty()
                            && let Ok(vti) = vt.parse::<i64>()
                        {
                            vtids.push((vti - 1) as u32);
                        }
                }
                current_group.face_vertex_indices.push(vids);
                current_group.face_texture_coordinate_indices.push(vtids);
                let _ = current_known; // suppress unused-write warning
            }
            _ => {} // ignore other prefixes
        }
    }

    if !current_group.face_vertex_indices.is_empty() {
        groups.insert(current.clone(), current_group);
        if !group_order.contains(&current) {
            group_order.push(current);
        }
    }

    Ok(ObjMesh {
        vertices,
        texture_coordinates: tex,
        groups,
        group_order,
    })
}

/// Minimal writer matching the Python `save_obj_file` — vertices then faces.
pub fn save(
    path: impl AsRef<Path>,
    vertices: &[[f64; 3]],
    faces: &[Vec<u32>],
) -> Result<(), ObjError> {
    let file = File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    for v in vertices {
        writeln!(w, "v {} {} {}", v[0], v[1], v[2])?;
    }
    for f in faces {
        let parts: Vec<String> = f.iter().map(|i| (i + 1).to_string()).collect();
        writeln!(w, "f {}", parts.join(" "))?;
    }
    w.flush()?;
    Ok(())
}
