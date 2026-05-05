//! Convenience model constructors mirroring `anny.create_fullbody_model`,
//! `create_hand_model`, and `create_head_model` (`anny/src/anny/models/__init__.py`).
//!
//! Hand and head models are derived from a fullbody model by:
//! 1. Building a fullbody model (just to enumerate `bone_labels` and to feed
//!    the face-segmentation sampler).
//! 2. Computing a `bones_to_remove` set (everything except the kept bones).
//! 3. Computing a `faces_to_keep` mask from the body-part segmentation.
//! 4. Re-building the model with those filters.
//!
//! The two-build pattern matches the Python implementation. It is wasteful
//! (~doubles model construction time) but keeps the data-loading pipeline
//! single-pass and easy to maintain.

use std::collections::HashSet;

use crate::face_segmentation::{FaceSegmentation, SegmentationError};
use crate::models::full_model::{Model, ModelError, ModelOptions};

/// Hand side designator. Used by [`create_hand_model`].
#[derive(Debug, Clone, Copy)]
pub enum HandSide {
    Left,
    Right,
}

impl HandSide {
    fn suffix(self) -> &'static str {
        match self {
            HandSide::Left => "L",
            HandSide::Right => "R",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PresetError {
    #[error("model: {0}")]
    Model(#[from] ModelError),
    #[error("segmentation: {0}")]
    Segmentation(#[from] SegmentationError),
}

/// Builds a fullbody model. Equivalent to `anny.create_fullbody_model()` with
/// `topology="default"` (modulo the nudity face edits, which are not ported)
/// and `local_changes="all"` semantics built in.
pub fn create_fullbody_model(opts: &ModelOptions) -> Result<Model, PresetError> {
    Ok(Model::build(opts)?)
}

/// Builds a hand-only model. Mirrors `create_hand_model` in
/// `anny/src/anny/models/__init__.py:129–172`.
pub fn create_hand_model(opts: &ModelOptions, side: HandSide) -> Result<Model, PresetError> {
    let s = side.suffix();

    // 1. Build a fullbody model purely to inspect its bones + sample the
    //    body-part segmentation.
    let full = Model::build(opts)?;

    // 2. Compute the keep-set of bones for this hand.
    let hand_bones: HashSet<String> = [
        format!("wrist.{s}"),
        format!("finger1-1.{s}"),
        format!("finger1-2.{s}"),
        format!("finger1-3.{s}"),
        format!("metacarpal1.{s}"),
        format!("finger2-1.{s}"),
        format!("finger2-2.{s}"),
        format!("finger2-3.{s}"),
        format!("metacarpal2.{s}"),
        format!("finger3-1.{s}"),
        format!("finger3-2.{s}"),
        format!("finger3-3.{s}"),
        format!("metacarpal3.{s}"),
        format!("finger4-1.{s}"),
        format!("finger4-2.{s}"),
        format!("finger4-3.{s}"),
        format!("metacarpal4.{s}"),
        format!("finger5-1.{s}"),
        format!("finger5-2.{s}"),
        format!("finger5-3.{s}"),
    ]
    .into_iter()
    .collect();
    let bones_to_remove: HashSet<String> = full
        .bone_labels
        .iter()
        .filter(|b| !hand_bones.contains(*b))
        .cloned()
        .collect();

    // 3. Faces to keep = those tagged `hand.{side}` in the body-part PNG.
    let seg = FaceSegmentation::new(&full, &opts.data_root)?;
    let label = format!("hand.{s}");
    let faces_to_keep = seg.face_mask(&[&label])?;

    // 4. Rebuild with the filters applied.
    let mut hand_opts = opts.clone();
    hand_opts.bones_to_remove = bones_to_remove;
    hand_opts.faces_to_keep = Some(faces_to_keep);
    Ok(Model::build(&hand_opts)?)
}

/// Builds a head-only model. Mirrors `create_head_model` in
/// `anny/src/anny/models/__init__.py:174–219`.
///
/// `eyes` / `tongue` toggle inclusion of the corresponding bones and faces.
pub fn create_head_model(
    opts: &ModelOptions,
    eyes: bool,
    tongue: bool,
) -> Result<Model, PresetError> {
    let mut head_opts = opts.clone();
    head_opts.include_eyes = eyes;
    head_opts.include_tongue = tongue;
    let full = Model::build(&head_opts)?;

    let mut keep: HashSet<&str> = ["neck01", "neck02", "neck03", "head"].into_iter().collect();
    keep.extend(FACIAL_EXPRESSION_BONES.iter().copied());
    if eyes {
        keep.extend(EYE_BONES.iter().copied());
    }
    if tongue {
        keep.extend(TONGUE_BONES.iter().copied());
    }
    let bones_to_remove: HashSet<String> = full
        .bone_labels
        .iter()
        .filter(|b| !keep.contains(b.as_str()))
        .cloned()
        .collect();

    let seg = FaceSegmentation::new(&full, &opts.data_root)?;
    let mut labels: Vec<&str> = vec!["head", "eye_cavity.R", "eye_cavity.L", "mouth_cavity"];
    if eyes {
        // NOTE: this faithfully mirrors `anny.create_head_model` in
        // `anny/src/anny/models/__init__.py:198–207`, which has a typo:
        // it lists `eye_back.L` twice, so the right-side back-of-eye faces
        // are not included. Kept for face-count parity.
        labels.extend(["eye_front.L", "eye_back.L", "eye_front.R", "eye_back.L"]);
    }
    if tongue {
        labels.push("tongue");
    }
    let faces_to_keep = seg.face_mask(&labels)?;

    head_opts.bones_to_remove = bones_to_remove;
    head_opts.faces_to_keep = Some(faces_to_keep);
    Ok(Model::build(&head_opts)?)
}

const EYE_BONES: &[&str] = &["eye.L", "eye.R"];

const TONGUE_BONES: &[&str] = &[
    "tongue00",
    "tongue01",
    "tongue02",
    "tongue03",
    "tongue04",
    "tongue05.L",
    "tongue05.R",
    "tongue06.L",
    "tongue06.R",
    "tongue07.L",
    "tongue07.R",
];

const FACIAL_EXPRESSION_BONES: &[&str] = &[
    "jaw",
    "special04",
    "oris02",
    "oris01",
    "oris06.L",
    "oris07.L",
    "oris06.R",
    "oris07.R",
    "levator02.L",
    "levator03.L",
    "levator04.L",
    "levator05.L",
    "levator02.R",
    "levator03.R",
    "levator04.R",
    "levator05.R",
    "special01",
    "oris04.L",
    "oris03.L",
    "oris04.R",
    "oris03.R",
    "oris06",
    "oris05",
    "special03",
    "levator06.L",
    "levator06.R",
    "special06.L",
    "special05.L",
    "orbicularis03.L",
    "orbicularis04.L",
    "special06.R",
    "special05.R",
    "orbicularis03.R",
    "orbicularis04.R",
    "temporalis01.L",
    "oculi02.L",
    "oculi01.L",
    "temporalis01.R",
    "oculi02.R",
    "oculi01.R",
    "temporalis02.L",
    "risorius02.L",
    "risorius03.L",
    "temporalis02.R",
    "risorius02.R",
    "risorius03.R",
];
