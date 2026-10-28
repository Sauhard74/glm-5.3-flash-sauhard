"""Graft the BF16 vision tower back onto a build made with --drop-vision, as a new model directory.

uv run python -m build.add_vision --src models/src-glm-5.3-flash --base models/glm-reap153-76-lsq \
    --out models/glm-5.3-flash-sauhard-vl

The base directory is never written. Everything in it is symlinked into --out except the files that change:
config.json (explicit ``false`` quantization entries for every vision module), model.safetensors.index.json
(+ the vision shard), build_summary.json and README.md. The vision tensors go through the same Target.transform
(model / vision / language sanitize) the full build uses, so they are stored pre-sanitized as ``vision_tower.*``
like the language shards; mlx-vlm re-sanitizes on load, which is a no-op for them (checked below).
"""

import argparse
import copy
import json
import os
import re
from pathlib import Path

import mlx.core as mx

from .families import family_for
from .pipeline import SKIP_COPY, Item, main_target, materialize, save_json
from .source import Source

SHARD = "model-vision.safetensors"
CHANGED = SKIP_COPY | {"README.md", SHARD}


def vision_weights(src: Source, family, cfg: dict):
    """Sanitized vision tensors + the vision_tower params/modules the mlx-vlm model expects."""
    target = main_target(family, cfg)
    keys = [k for k in src.keys() if k.startswith(family.vision_prefix)]
    if not keys:
        raise ValueError(f"no {family.vision_prefix}* tensors in {src.path}")
    weights = target.transform(materialize(src, [Item(k, [k]) for k in keys], real=True))
    expected = {k: v for k, v in target.params().items() if k.startswith("vision_tower.")}
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    bad = [(k, tuple(weights[k].shape), expected[k][0]) for k in weights if k in expected and tuple(weights[k].shape) != expected[k][0]]
    if missing or extra or bad:
        raise ValueError(f"vision weights do not match the mlx-vlm model: missing={missing[:10]} extra={extra[:10]} shape_mismatch={bad[:10]}")
    if again := [k for k, v in target.transform(dict(weights)).items() if v is not weights[k]]:
        raise ValueError(f"load-time sanitize would change stored vision tensors: {again[:10]}")
    linears = sorted(p for p, m in target.modules.items() if p.startswith("vision_tower.") and hasattr(m, "to_quantized"))
    return weights, linears


def link_base(base: Path, out: Path):
    for f in sorted(base.iterdir()):
        if f.name in CHANGED:
            continue
        dst = out / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(os.path.relpath(f.resolve(), out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, help="HF snapshot with the original vision tensors")
    p.add_argument("--base", required=True, help="existing --drop-vision build (left untouched)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    base, out = Path(args.base).resolve(), Path(args.out)
    if out.resolve() == base:
        raise ValueError("--out must differ from --base")

    cfg = json.loads((base / "config.json").read_text())
    src = Source(args.src)
    family = family_for(cfg["model_type"])
    if src.config.get("vision_config") != cfg.get("vision_config"):
        raise ValueError("vision_config differs between --src and --base")

    weights, linears = vision_weights(src, family, cfg)
    dtypes = {str(v.dtype) for v in weights.values()}
    if dtypes != {"mlx.core.bfloat16"}:
        raise ValueError(f"expected BF16 vision tensors, got {dtypes}")

    out.mkdir(parents=True, exist_ok=True)
    link_base(base, out)
    mx.eval(list(weights.values()))
    nbytes = sum(v.nbytes for v in weights.values())
    mx.save_safetensors(str(out / SHARD), weights, metadata={"format": "mlx"})

    # mlx-vlm's class_predicate honours `path: false` as "leave unquantized" (build/pipeline quant_config style).
    out_cfg = copy.deepcopy(cfg)
    for key in ("quantization", "quantization_config"):
        if isinstance(out_cfg.get(key), dict):
            out_cfg[key].update({path: False for path in linears})
    save_json(out / "config.json", out_cfg)

    index = json.loads((base / "model.safetensors.index.json").read_text())
    if clash := set(index["weight_map"]) & set(weights):
        raise ValueError(f"base index already has vision keys: {sorted(clash)[:5]}")
    index["weight_map"].update({k: SHARD for k in weights})
    index["weight_map"] = dict(sorted(index["weight_map"].items()))
    index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + nbytes
    save_json(out / "model.safetensors.index.json", index)

    summary_path = base / "build_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["drop_vision"] = False
        summary["vision"] = {
            "added_by": "build.add_vision",
            "base": str(Path(args.base)),
            "source": str(Path(args.src)),
            "dtype": "bfloat16",
            "tensors": len(weights),
            "params": sum(v.size for v in weights.values()),
            "bytes": nbytes,
            "shard": SHARD,
        }
        summary.setdefault("written_bytes", {})["vision"] = nbytes
        save_json(out / "build_summary.json", summary)

    readme = base / "README.md"
    if readme.exists():
        text = readme.read_text()
        text = re.sub(r"pipeline_tag: text-generation", "pipeline_tag: image-text-to-text", text)
        text = re.sub(
            r"- \*\*Text-only:\*\* the vision tower is dropped\.",
            f"- **Vision:** the original vision tower is included unquantized (BF16, {nbytes / 1e9:.2f} GB).",
            text,
        )
        (out / "README.md").write_text(text)

    print(f"[add_vision] {len(weights)} tensors, {nbytes / 1e9:.3f} GB -> {out / SHARD}; "
          f"{len(linears)} vision linears marked unquantized")


if __name__ == "__main__":
    main()
