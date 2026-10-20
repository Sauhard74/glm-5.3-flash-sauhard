"""Validate the checkpoint -> mlx layer mapping for every layer without reading weights.

Lazy zero tensors with the checkpoint's shapes/dtypes go through the same sanitize path
as a real load; nothing is evaluated, so this is safe on the full-size models.

uv run python -m calib.check --model {glm,qwen} --src <hf snapshot dir>
"""

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten

from .arch import MLX_LAYER_PREFIX, make_arch

_MX = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "I64": mx.int64,
       "I32": mx.int32, "U8": mx.uint8, "F8_E4M3": mx.uint8, "BOOL": mx.bool_}


def _lazy(snap, keys):
    return {k: mx.zeros(snap.shape(k), dtype=_MX[snap.dtype(k)]) for k in keys}


def check_layer(arch, i):
    from mlx_vlm.fp8 import _dequantize_fp8_weight

    layer = arch.build_layer(i)
    keys = arch.layer_source_keys(i)
    src = _lazy(arch.snap, keys)
    for k in [k for k in src if k.endswith(".weight_scale_inv")]:
        w = k[: -len("_scale_inv")]
        src[w] = _dequantize_fp8_weight(src[w], src.pop(k)).astype(mx.bfloat16)
    got = arch.sanitize(src, layer)
    prefix = MLX_LAYER_PREFIX.format(i)
    got = {k[len(prefix):]: v for k, v in got.items()}
    want = dict(tree_flatten(layer.parameters()))
    problems = [f"missing {k}" for k in sorted(set(want) - set(got))]
    problems += [f"unexpected {k}" for k in sorted(set(got) - set(want))]
    problems += [f"shape {k}: ckpt {tuple(got[k].shape)} != model {tuple(want[k].shape)}"
                 for k in sorted(set(want) & set(got)) if got[k].shape != want[k].shape]
    return layer, problems


def check(model, src, layers=None):
    arch = make_arch(model, src)
    report = {}
    for i in layers if layers is not None else range(arch.n_layers):
        layer, problems = check_layer(arch, i)
        report[i] = {"type": getattr(layer, "block_type", None) or layer.layer_type,
                     "moe": arch.moe(layer) is not None,
                     "targets": sorted(arch.quant_targets(layer, i)),
                     "problems": problems}
    for key in ["model.language_model.embed_tokens.weight", *arch.final_norm_keys()]:
        if key not in arch.snap:
            report.setdefault("head", {"problems": []})["problems"].append(f"missing {key}")
    return arch, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["glm", "qwen"], required=True)
    ap.add_argument("--src", required=True)
    args = ap.parse_args(argv)
    _, report = check(args.model, args.src)
    bad = 0
    for i, r in report.items():
        if r["problems"]:
            bad += 1
            print(f"layer {i}: " + "; ".join(r["problems"][:10]))
        elif i != "head":
            print(f"layer {i:3d} {r['type']:<26} moe={r['moe']!s:<5} {len(r['targets'])} dense targets ok")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
