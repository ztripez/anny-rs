"""One-shot conversion of `anny2smplx.pth` (which contains a Python list of
tensors that candle's pickle reader skips) into a flat safetensors file.

Run after `cargo run --features smplx-download --bin anny-cli download-smplx`
(or equivalent), e.g.:

    .venv/bin/python tests/fixtures/convert_smplx.py

The output goes to the same directory as the input, named `anny2smplx.safetensors`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    cache_root = Path(os.environ.get("ANNY_CACHE_DIR", str(Path.home() / ".cache" / "anny")))
    src = cache_root / "noncommercial" / "anny2smplx.pth"
    if not src.exists():
        print(f"missing: {src}", file=sys.stderr)
        sys.exit(1)
    state = torch.load(src, map_location="cpu", weights_only=True)

    bary_list = state["anny2dst_barycentric_coordinates"]
    bary = torch.stack(bary_list).contiguous().to(torch.float64)  # [3, V_smplx]

    out_state = {
        "barycentric": bary,
        "vertex_indices": state["anny2dst_vertex_indices"].contiguous().to(torch.int64),
        "dst_faces": state["dst_faces"].contiguous().to(torch.int64),
    }
    dst = src.with_suffix(".safetensors")
    save_file(out_state, str(dst))
    sizes = {k: list(v.shape) for k, v in out_state.items()}
    print(f"wrote {dst} with {sizes}")


if __name__ == "__main__":
    main()
