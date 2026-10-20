"""uv run python -m build.make --model {glm,qwen} --src <snap> --calib <dir> --prune <ratio|keep|auto>
--budget-gb 68 --out models/<name>"""

import argparse
import json
import logging
import sys

from . import alloc
from .pipeline import GB, BuildOptions, Builder


def _ints(s: str):
    return tuple(int(x) for x in s.split(",") if x)


def _pin(s: str):
    pattern, _, spec = s.rpartition("=")
    if not pattern:
        raise argparse.ArgumentTypeError(f"--pin expects REGEX=BITS[:GROUP], got {s!r}")
    bits, _, group = spec.partition(":")
    return pattern, int(bits), int(group) if group else None


def _floor(s: str):
    pattern, _, bits = s.rpartition("=")
    if not pattern:
        raise argparse.ArgumentTypeError(f"--floor expects REGEX=BITS, got {s!r}")
    return pattern, int(bits)


def _weights(s: str):
    out = {}
    for part in s.split(","):
        if part:
            k, _, v = part.partition("=")
            out[k] = float(v or 1)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="REAP-prune + mixed-precision quantize a MoE checkpoint for mlx-vlm.")
    p.add_argument("--model", choices=["glm", "qwen"], help="model family (default: from source config)")
    p.add_argument("--src", required=True, help="HF snapshot dir (config.json + safetensors)")
    p.add_argument("--calib", help="calib/ output dir (layer_XXX.{npz,json})")
    p.add_argument("--out", required=True)
    p.add_argument("--prune", default="none", help="'none', fraction pruned (0.375), experts kept per layer (320), or 'auto'")
    p.add_argument("--prune-mode", choices=["uniform", "global"], default="uniform",
                   help="uniform top-K per layer, or global ranking of per-layer-normalized scores with min-keep")
    p.add_argument("--min-keep", type=float, default=0.25, help="global/auto: min fraction of experts kept per layer")
    p.add_argument("--score", choices=["reap", "freq", "mass"], default="reap")
    p.add_argument("--domain-weights", type=_weights, help="e.g. code=2,agentic=2,reasoning=1,longctx=1")
    p.add_argument("--auto-step", type=int, help="auto: expert-count grid step (default max(8, E/32))")
    p.add_argument("--prune-lambda", type=float, default=1.0, help="auto: weight of prune loss vs quant sensitivity")
    p.add_argument("--budget-gb", type=float, default=68.0, help="total weight bytes (1 GB = 1e9) incl. MTP + vision")
    p.add_argument("--drop-vision", action="store_true")
    p.add_argument("--mtp", choices=["prune", "intact", "drop"], default="prune")
    p.add_argument("--mtp-sens-weight", type=float, default=0.25)
    p.add_argument("--bits", type=_ints, default=alloc.DEFAULT_BITS)
    p.add_argument("--groups", type=_ints, default=alloc.DEFAULT_GROUPS)
    p.add_argument("--no-bf16", dest="allow_bf16", action="store_false", help="never leave a quantizable tensor in bf16")
    p.add_argument("--solver", choices=["dp", "greedy"], default="dp")
    p.add_argument("--pin", type=_pin, action="append", default=[], help="REGEX=BITS[:GROUP] (16 = bf16)")
    p.add_argument("--floor", type=_floor, action="append", default=[], help="REGEX=MIN_BITS")
    p.add_argument("--no-default-floors", dest="default_floors", action="store_false")
    p.add_argument("--min-bits", type=int, default=3, help="lowest bit width for any non-pinned tensor")
    p.add_argument("--sens-power", type=float, default=1.0,
                   help="objective = sum(sens**p) (+ lambda * sum(prune_loss**p)); sens is NMSE, so 1 = additive error energy")
    p.add_argument("--ple-bits", type=int, default=4, help="Qwen PLE n-gram table bits (4 keeps mlx-vlm mmap PLE storage usable)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--shard-gb", type=float, default=2.0)
    p.add_argument("--rounding", choices=["rtn", "lsq"], default="rtn",
                   help="lsq: imatrix-weighted least-squares scale/bias refit with range search (docs/plan/05 §2.6a)")
    p.add_argument("--lsq-iters", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="plan + validate + print allocation, write nothing")
    p.add_argument("--summary-json", help="also write the summary JSON here")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    opts = BuildOptions(
        model=args.model,
        src=args.src,
        out=args.out,
        calib=args.calib,
        prune=args.prune,
        prune_mode=args.prune_mode,
        budget_gb=args.budget_gb,
        drop_vision=args.drop_vision,
        mtp=args.mtp,
        bits=args.bits,
        groups=args.groups,
        allow_bf16=args.allow_bf16,
        solver=args.solver,
        pins=args.pin,
        floors=args.floor,
        default_floors=args.default_floors,
        ple_bits=args.ple_bits,
        min_bits=args.min_bits,
        sens_power=args.sens_power,
        score=args.score,
        domain_weights=args.domain_weights,
        min_keep_frac=args.min_keep,
        auto_step=args.auto_step,
        prune_lambda=args.prune_lambda,
        mtp_sens_weight=args.mtp_sens_weight,
        dtype=args.dtype,
        shard_gb=args.shard_gb,
        dry_run=args.dry_run,
        rounding=args.rounding,
        lsq_iters=args.lsq_iters,
    )
    summary = Builder(opts).run()
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
    print_summary(summary)
    return summary


def print_summary(s: dict, file=sys.stdout):
    pr = s["prune"]
    counts = sorted(set(pr["experts_per_layer"].values()))
    print(f"[build] {s['model']} -> {'(dry run)' if s.get('dry_run') else 'written'}", file=file)
    print(f"  experts kept/layer: {counts}  prune loss: {pr['total_prune_loss']:.4f}", file=file)
    for ns, b in s["predicted_bytes"].items():
        print(f"  {ns:5s} {b / GB:8.3f} GB", file=file)
    print(
        f"  total {s['predicted_total_bytes'] / GB:.3f} / {s['budget_bytes'] / GB:.3f} GB budget, "
        f"avg {s['avg_bits_per_weight']} bpw over quantizable weights, objective {s['objective']:.5g}",
        file=file,
    )
    print(f"  bits by params: { {k: round(v / max(s['quantized_params'], 1), 4) for k, v in s['bits_histogram_params'].items()} }", file=file)


if __name__ == "__main__":
    main()
