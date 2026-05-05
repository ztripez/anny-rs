//! Reader for MakeHuman `.target.gz` blend-shape files.
//!
//! Mirrors `load_blend_shape` in `anny/src/anny/models/full_model.py:20–32`.
//! Each non-empty line has the form `<vertex_id> <dx> <dy> <dz>` where the
//! id is 0-based and the offsets are floats expressed in decimeters.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use flate2::read::GzDecoder;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TargetGzError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed line {line}: {msg}")]
    Malformed { line: usize, msg: String },
    #[error("vertex id {id} out of bounds (vertices_count={count})")]
    OutOfBounds { id: usize, count: usize },
}

/// Loads a `.target.gz` file into a dense `[vertices_count, 3]` row-major buffer
/// of `f64` deltas. Vertices not mentioned in the file are zero. The caller is
/// expected to apply the world transformation (decimeter → meter scaling, etc.).
pub fn load(path: impl AsRef<Path>, vertices_count: usize) -> Result<Vec<f64>, TargetGzError> {
    let file = File::open(path.as_ref())?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut buf = vec![0.0_f64; vertices_count * 3];

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let id_token = match parts.next() {
            Some(t) => t,
            None => continue,
        };
        let id: usize = id_token.parse().map_err(|e| TargetGzError::Malformed {
            line: lineno + 1,
            msg: format!("vertex id parse: {e}"),
        })?;
        if id >= vertices_count {
            return Err(TargetGzError::OutOfBounds {
                id,
                count: vertices_count,
            });
        }
        let coords: Result<Vec<f64>, _> = parts.map(|s| s.parse::<f64>()).collect();
        let coords = coords.map_err(|e| TargetGzError::Malformed {
            line: lineno + 1,
            msg: format!("offset parse: {e}"),
        })?;
        if coords.len() != 3 {
            return Err(TargetGzError::Malformed {
                line: lineno + 1,
                msg: format!("offset needs 3 floats, got {}", coords.len()),
            });
        }
        let base = id * 3;
        buf[base] = coords[0];
        buf[base + 1] = coords[1];
        buf[base + 2] = coords[2];
    }

    Ok(buf)
}
