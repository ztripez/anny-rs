"""Export reference forward-pass output from Python anny for the Rust port to
compare against. Runs in the uv venv at `anny/.venv` (sibling crate).

Outputs raw little-endian f64 binary files (no header) into this directory.
Each file has a sibling `.shape` text file with one line: `dim0 dim1 ...`.
"""
from __future__ import annotations

import os
import struct
import sys

import anny
import numpy as np
import torch

from anny.models.full_model import create_model

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(name: str, t: torch.Tensor) -> None:
    arr = t.detach().cpu().to(torch.float64).contiguous().numpy()
    arr.tofile(os.path.join(OUT_DIR, f"{name}.bin"))
    with open(os.path.join(OUT_DIR, f"{name}.shape"), "w") as f:
        f.write(" ".join(str(d) for d in arr.shape))
    print(f"wrote {name}: shape={list(arr.shape)} mean={float(arr.mean()):.6e} std={float(arr.std()):.6e}")


def main() -> None:
    torch.set_default_dtype(torch.float64)
    model = create_model(
        rig="default",
        topology="makehuman",          # skip the nudity face edits
        local_changes="all",            # match Rust's always-load behaviour
        remove_unattached_vertices=False,
        skinning_method="lbs",
        triangulate_faces=False,
        all_phenotypes=True,            # use full label set
        extrapolate_phenotypes=False,
    )

    # Default phenotype = all 0.5 except race (which would be ill-conditioned at 0).
    # Use the parse_phenotype_kwargs path with explicit defaults.
    bs = 1
    dtype = torch.float64
    phenotype_kwargs = dict(
        gender=torch.full((bs,), 0.5, dtype=dtype),
        age=torch.full((bs,), 0.5, dtype=dtype),
        muscle=torch.full((bs,), 0.5, dtype=dtype),
        weight=torch.full((bs,), 0.5, dtype=dtype),
        height=torch.full((bs,), 0.5, dtype=dtype),
        proportions=torch.full((bs,), 0.5, dtype=dtype),
        cupsize=torch.full((bs,), 0.5, dtype=dtype),
        firmness=torch.full((bs,), 0.5, dtype=dtype),
        african=torch.full((bs,), 0.5, dtype=dtype),
        asian=torch.full((bs,), 0.5, dtype=dtype),
        caucasian=torch.full((bs,), 0.5, dtype=dtype),
    )

    # Identity pose (None → identity per Python defaults).
    out = model.forward(
        pose_parameters=None,
        phenotype_kwargs=phenotype_kwargs,
        local_changes_kwargs={},
        pose_parameterization="rest_relative",
    )

    save("rest_vertices", out["rest_vertices"])
    save("vertices_rest_relative_identity", out["vertices"])
    save("rest_bone_poses", out["rest_bone_poses"])
    save("bone_poses_rest_relative_identity", out["bone_poses"])
    save("blendshape_coeffs", out["blendshape_coeffs"])
    save("template_vertices", model.template_vertices)
    save("blendshapes_first10", model.blendshapes[:10])
    print(f"\nbone_count={len(model.bone_labels)} vertex_count={model.template_vertices.shape[0]}")
    print(f"blendshape_count={model.blendshapes.shape[0]} mask_shape={list(model.stacked_phenotype_blend_shapes_mask.shape)}")
    print(f"local_change_count={len(model.local_change_labels)}")

    # ── Anthropometry readouts on the rest-pose vertices. ────────────────
    # The Python class needs `model.base_mesh_vertex_indices`. Build it
    # ourselves since `remove_unattached_vertices=False` skipped the path
    # that records it.
    if not hasattr(model, "base_mesh_vertex_indices"):
        model.base_mesh_vertex_indices = torch.arange(model.template_vertices.shape[0], dtype=torch.int64)
    from anny.anthropometry import Anthropometry
    anth = Anthropometry(model)
    measurements = anth(out["rest_vertices"])
    for name, value in measurements.items():
        save(f"anthropometry_{name}", value)
        print(f"  {name}: {float(value):.6f}")

    # ── Face segmentation reference for the head + R hand. ───────────────
    from anny.face_segmentation import get_face_segmentation_mask
    head_mask = get_face_segmentation_mask(model, ["head"])
    rhand_mask = get_face_segmentation_mask(model, ["hand.R"])
    save("face_segmentation_head", head_mask.to(torch.float64))
    save("face_segmentation_hand_R", rhand_mask.to(torch.float64))
    print(f"  face seg head count: {int(head_mask.sum())}")
    print(f"  face seg hand.R count: {int(rhand_mask.sum())}")

    # ── Alternative-topology reference. ──────────────────────────────────
    # Use the smallest available variant (notoes_collapse3pc, ~369 verts)
    # so the closest-point search is cheap. We can't easily call Python's
    # create_alternative_topology_model because that path requires NVIDIA
    # Warp; instead we emit only the *target topology* counts so the Rust
    # build can verify shapes. (Numerical comparison would require Warp.)
    import os
    topo = "notoes_collapse3pc"
    obj_path = os.path.join("src/anny/data/topology", f"{topo}.obj")
    n_verts = 0
    n_faces = 0
    if os.path.exists(obj_path):
        with open(obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    n_verts += 1
                elif line.startswith("f "):
                    arity = len(line.split()) - 1
                    if arity == 3:
                        n_faces += 1
                    elif arity == 4:
                        # quad → 2 triangles after triangulation
                        n_faces += 2
                    else:
                        raise AssertionError(f"unexpected face arity {arity}")
        with open(os.path.join(OUT_DIR, "alt_topology_counts.txt"), "w") as f:
            f.write(f"name {topo}\n")
            f.write(f"vertices {n_verts}\n")
            f.write(f"faces {n_faces}\n")
        print(f"  alt topology {topo}: V={n_verts} F={n_faces} (post-triangulation)")

    # ── remove_unattached_vertices reference (vertex/face counts). ───────
    pruned = create_model(
        rig="default",
        topology="default",
        local_changes="all",
        remove_unattached_vertices=True,
        skinning_method="lbs",
        all_phenotypes=True,
        extrapolate_phenotypes=False,
    )
    with open(os.path.join(OUT_DIR, "pruned_counts.txt"), "w") as f:
        f.write(f"vertices {pruned.template_vertices.shape[0]}\n")
        f.write(f"faces {pruned.faces.shape[0]}\n")
    print(f"  pruned model: V={pruned.template_vertices.shape[0]} F={pruned.faces.shape[0]}")

    # ── Topology=default face-count reference. ───────────────────────────
    # Build a model with topology="default" (the face-edit path) and capture
    # the resulting face count — Rust's get_edited_mesh_faces should produce
    # the same number of faces.
    model_default_topo = create_model(
        rig="default",
        topology="default",
        local_changes="all",
        remove_unattached_vertices=False,
        skinning_method="lbs",
        all_phenotypes=True,
        extrapolate_phenotypes=False,
    )
    with open(os.path.join(OUT_DIR, "topology_default_counts.txt"), "w") as f:
        f.write(f"faces {model_default_topo.faces.shape[0]}\n")
    print(f"  topology=default faces: {model_default_topo.faces.shape[0]}")

    # ── SMPL-X retopology reference (only if data is on disk). ───────────
    try:
        from anny.paths import ANNY2SMPLX_DATA_PATH
        if os.path.exists(str(ANNY2SMPLX_DATA_PATH)):
            from anny.models.retopology import create_smplx_topology_model
            smplx_model = create_smplx_topology_model(all_phenotypes=True)
            smplx_phen = {k: torch.full((1,), 0.5, dtype=torch.float64) for k in [
                "gender", "age", "muscle", "weight", "height", "proportions",
                "cupsize", "firmness", "african", "asian", "caucasian"]}
            smplx_out = smplx_model(pose_parameters=None,
                                    phenotype_kwargs=smplx_phen,
                                    local_changes_kwargs={},
                                    pose_parameterization="rest_relative")
            save("smplx_template_vertices", smplx_model.template_vertices)
            save("smplx_rest_vertices", smplx_out["rest_vertices"])
            with open(os.path.join(OUT_DIR, "smplx_counts.txt"), "w") as f:
                f.write(f"vertices {smplx_model.template_vertices.shape[0]}\n")
                f.write(f"faces {smplx_model.faces.shape[0]}\n")
                f.write(f"max_bones_per_vertex {smplx_model.vertex_bone_indices.shape[1]}\n")
            print(f"  smplx model: V={smplx_model.template_vertices.shape[0]} F={smplx_model.faces.shape[0]} M={smplx_model.vertex_bone_indices.shape[1]}")
        else:
            print("  smplx data not on disk; skipping smplx fixtures")
    except Exception as e:
        print(f"  smplx fixture export failed: {e}")

    # ── Hand-model and head-model construction. ──────────────────────────
    # Force remove_unattached_vertices=False so both Python and Rust keep
    # all 19158 vertices and we can compare face counts directly.
    hand_model = anny.create_hand_model(side="R", remove_unattached_vertices=False, all_phenotypes=True)
    head_model = anny.create_head_model(eyes=True, tongue=True, remove_unattached_vertices=False, all_phenotypes=True)
    print(f"  hand.R model: bones={len(hand_model.bone_labels)} faces={hand_model.faces.shape}")
    print(f"  head model:   bones={len(head_model.bone_labels)} faces={head_model.faces.shape}")
    with open(os.path.join(OUT_DIR, "preset_counts.txt"), "w") as f:
        f.write(f"hand_R_bones {len(hand_model.bone_labels)}\n")
        f.write(f"hand_R_faces {hand_model.faces.shape[0]}\n")
        f.write(f"head_bones {len(head_model.bone_labels)}\n")
        f.write(f"head_faces {head_model.faces.shape[0]}\n")


if __name__ == "__main__":
    main()
