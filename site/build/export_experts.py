"""Write site/build/experts.json: per-layer routing frequency (normalised) and kept expert ids of the shipped build.

uv run python site/build/export_experts.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "artifacts/glm/calib"
BUILD = ROOT / "models/glm-reap153-76-lsq/build_summary.json"

layers = json.loads(BUILD.read_text())["prune"]["layers"]
out = {}
for i in range(3, 45):
    freq = np.load(CALIB / f"layer_{i:03d}.npz")["expert_freq"].astype(float)
    freq /= freq.sum()
    kept = layers[str(i)]["kept_ids"]
    out[i] = {"freq": [round(x, 6) for x in freq], "kept": kept, "coverage": round(float(freq[kept].sum()), 4)}
(Path(__file__).parent / "experts.json").write_text(json.dumps(out))
