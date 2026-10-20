"""KL divergence and top-1 agreement vs the full-precision reference top-k
logprobs written by calib (``<calib>/ref/index.json`` + per-sample npz).

uv run python -m evals.kl --model models/<name> [--ref artifacts/<family>/calib/ref] [--limit N]

KL(P_ref || Q) is computed on the reference top-k support plus one tail bucket
(1 - sum of top-k mass), i.e. the KL between the coarse-grained distributions:
a tight lower bound on the full-vocab KL when k=64.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .common import base_parser, load_model, mean, model_family, read_config, resolve_model_path, write_result


def load_ref_index(ref_dir: Path) -> Dict[str, Any]:
    with open(ref_dir / "index.json") as f:
        return json.load(f)


def iter_ref(ref_dir: Path, limit_per_domain: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    index = load_ref_index(ref_dir)
    counts: Dict[str, int] = {}
    for entry in index["samples"]:
        dom = entry.get("domain", "all")
        if limit_per_domain is not None and counts.get(dom, 0) >= limit_per_domain:
            continue
        counts[dom] = counts.get(dom, 0) + 1
        with np.load(ref_dir / entry["file"]) as z:
            yield {
                "id": entry["id"],
                "domain": dom,
                "tokens": z["tokens"].astype(np.int32),
                "top_idx": z["top_idx"].astype(np.int32),
                "top_logprob": z["top_logprob"].astype(np.float32),
                "target_logprob": z["target_logprob"].astype(np.float32) if "target_logprob" in z else None,
            }


def chunk_bounds(n: int, chunk: int):
    for lo in range(0, n, chunk):
        yield lo, min(n, lo + chunk)


def model_logprob_chunks(model, tokens: np.ndarray, chunk: int):
    """Yield (lo, hi, log-softmax [hi-lo, V] float32) over a chunked prefill."""
    import mlx.core as mx
    from mlx_vlm.models import cache as cache_mod

    prompt_cache = cache_mod.make_prompt_cache(model.language_model)
    for lo, hi in chunk_bounds(len(tokens), chunk):
        out = model(mx.array(tokens[lo:hi])[None], cache=prompt_cache)
        logits = getattr(out, "logits", out)[0].astype(mx.float32)
        yield lo, hi, logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval([c.state for c in prompt_cache])


def kl_terms(lq, top_idx, top_lp):
    """Per-position coarse-grained KL, top-1 agreement. mlx arrays in, mlx out."""
    import mlx.core as mx

    q_top = mx.take_along_axis(lq, top_idx, axis=-1)
    p_top = mx.exp(top_lp)
    kl = (p_top * (top_lp - q_top)).sum(-1)
    p_tail = mx.clip(1.0 - p_top.sum(-1), 0.0, 1.0)
    q_tail = mx.clip(1.0 - mx.exp(q_top).sum(-1), 1e-10, 1.0)
    tail = mx.where(p_tail > 1e-8, p_tail * (mx.log(mx.maximum(p_tail, 1e-30)) - mx.log(q_tail)), 0.0)
    kl = mx.maximum(kl + tail, 0.0)
    ref_top1 = mx.take_along_axis(top_idx, mx.argmax(top_lp, axis=-1, keepdims=True), axis=-1)[:, 0]
    agree = mx.argmax(lq, axis=-1) == ref_top1
    return kl, agree


def eval_sample(model, s: Dict[str, Any], chunk: int) -> Dict[str, np.ndarray]:
    import mlx.core as mx

    toks = s["tokens"]
    L = min(len(toks), len(s["top_idx"]))
    kls, agrees, nlls = [], [], []
    for lo, hi, lq in model_logprob_chunks(model, toks[:L], chunk):
        kl, agree = kl_terms(lq, mx.array(s["top_idx"][lo:hi]), mx.array(s["top_logprob"][lo:hi]))
        nxt = toks[lo + 1 : hi + 1]
        nll = -mx.take_along_axis(lq[: len(nxt)], mx.array(nxt)[:, None], axis=-1)[:, 0]
        mx.eval(kl, agree, nll)
        kls.append(np.array(kl))
        agrees.append(np.array(agree))
        nlls.append(np.array(nll))
    return {"kl": np.concatenate(kls), "agree": np.concatenate(agrees), "nll": np.concatenate(nlls)}


ROLE_MARKERS = {  # GLM-style single-token role markers
    "<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant", "<|observation|>": "tool",
}
ROLES = ("system", "user", "assistant", "tool")


def position_roles(tokens: np.ndarray, tokenizer) -> np.ndarray:
    """Chat role in effect at each position (predicting token i+1 from position i), from the last
    role marker: GLM ``<|user|>``-style tokens, or ChatML ``<|im_start|>role``. "none" before any."""
    vocab = tokenizer.get_vocab() if tokenizer is not None else {}
    single = {vocab[t]: r for t, r in ROLE_MARKERS.items() if t in vocab}
    im_start = vocab.get("<|im_start|>")
    roles = np.full(len(tokens), "none", dtype=object)
    cur = "none"
    for i, t in enumerate(tokens):
        t = int(t)
        if t in single:
            cur = single[t]
        elif im_start is not None and i > 0 and int(tokens[i - 1]) == im_start:
            name = tokenizer.decode([t]).strip()
            cur = {"observation": "tool"}.get(name, name if name in ROLES else "none")
        roles[i] = cur
    return roles


def summarize(kl: np.ndarray, agree: np.ndarray, nll: np.ndarray, ref_nll: np.ndarray) -> Dict[str, Any]:
    out = {
        "tokens": int(kl.size),
        "mean_kl": float(kl.mean()) if kl.size else None,
        "median_kl": float(np.median(kl)) if kl.size else None,
        "p90_kl": float(np.percentile(kl, 90)) if kl.size else None,
        "p99_kl": float(np.percentile(kl, 99)) if kl.size else None,
        "top1_agree": float(agree.mean()) if agree.size else None,
        "nll": float(nll.mean()) if nll.size else None,
    }
    ref = ref_nll[~np.isnan(ref_nll)] if ref_nll.size else ref_nll
    out["ref_nll"] = float(ref.mean()) if ref.size else None
    if out["nll"] is not None and out["ref_nll"] is not None:
        out["ppl_ratio"] = float(np.exp(out["nll"] - out["ref_nll"]))
    return out


def write_self_ref(model, samples: List[Dict[str, Any]], out_dir: Path, topk: int = 64) -> Path:
    """Reference from the model itself (single full forward) in calib's ref format;
    used by --smoke and tests: KL vs this should be ~0 with top-1 agreement ~1."""
    import mlx.core as mx

    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for k, s in enumerate(samples):
        toks = np.asarray(s["tokens"], dtype=np.int32)
        lq = next(model_logprob_chunks(model, toks, len(toks)))[2]
        idx = mx.argpartition(-lq, kth=topk - 1, axis=-1)[:, :topk]
        vals = mx.take_along_axis(lq, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx, vals = mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(vals, order, axis=-1)
        tgt = np.full(len(toks), np.nan, dtype=np.float32)
        t = mx.take_along_axis(lq[:-1], mx.array(toks[1:])[:, None], axis=-1)[:, 0]
        mx.eval(idx, vals, t)
        tgt[:-1] = np.array(t)
        fname = f"{k:05d}.npz"
        np.savez(out_dir / fname, tokens=toks, top_idx=np.array(idx).astype(np.int32),
                 top_logprob=np.array(vals).astype(np.float32), target_logprob=tgt)
        index.append({"id": s["id"], "domain": s["domain"], "file": fname, "n_tokens": int(len(toks))})
    with open(out_dir / "index.json", "w") as f:
        json.dump({"model": "self", "topk": topk, "samples": index}, f)
    return out_dir


def smoke_samples(tokenizer) -> List[Dict[str, Any]]:
    from .longctx import filler_sentences
    import random

    text = " ".join(filler_sentences(random.Random(0), 200))
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        {"id": "smoke-code", "domain": "code", "tokens": ids[:64]},
        {"id": "smoke-reasoning", "domain": "reasoning", "tokens": ids[100:400]},
        {"id": "smoke-longctx", "domain": "longctx", "tokens": ids[400:1100]},
    ]


def main(argv=None):
    p = base_parser("kl", __doc__)
    p.add_argument("--ref", help="reference dir (default artifacts/<family>/calib/ref)")
    p.add_argument("--chunk", type=int, default=1024, help="prefill chunk (tokens); logits are chunk x vocab fp32")
    args = p.parse_args(argv)
    started = time.time()

    model_path = resolve_model_path(args.model)
    family = model_family(read_config(model_path))
    model, processor = load_model(args)
    ref_dir = Path(args.ref or f"artifacts/{family}/calib/ref")
    tmp = None
    # Smoke runs use a self-reference: a real calib ref for this family belongs to the full model, not the tiny one.
    if args.smoke and not args.ref:
        tmp = tempfile.TemporaryDirectory(prefix="evals-kl-smoke-")
        ref_dir = write_self_ref(model, smoke_samples(getattr(processor, "tokenizer", processor)), Path(tmp.name))
        args.chunk = min(args.chunk, 128)
    if not (ref_dir / "index.json").exists():
        raise SystemExit(f"no reference at {ref_dir}/index.json (run calib, or pass --ref)")
    index_meta = {k: v for k, v in load_ref_index(ref_dir).items() if k != "samples"}

    tokenizer = getattr(processor, "tokenizer", processor)
    per_domain: Dict[str, Dict[str, List[np.ndarray]]] = {}
    per_role: Dict[str, Dict[str, List[np.ndarray]]] = {}
    samples = []
    for n, s in enumerate(iter_ref(ref_dir, args.limit)):
        r = eval_sample(model, s, args.chunk)
        roles = position_roles(s["tokens"][: len(r["kl"])], tokenizer)
        for role in set(roles):
            m = roles == role
            pr = per_role.setdefault(role, {"kl": [], "agree": []})
            pr["kl"].append(r["kl"][m])
            pr["agree"].append(r["agree"][m])
        d = per_domain.setdefault(s["domain"], {"kl": [], "agree": [], "nll": [], "ref_nll": []})
        d["kl"].append(r["kl"])
        d["agree"].append(r["agree"])
        d["nll"].append(r["nll"])
        ref_nll = -s["target_logprob"][: len(r["nll"])] if s["target_logprob"] is not None else np.array([])
        d["ref_nll"].append(ref_nll)
        rec = {"id": s["id"], "domain": s["domain"], **summarize(r["kl"], r["agree"], r["nll"], ref_nll)}
        samples.append(rec)
        print(f"[kl] {n + 1} {s['id']} ({s['domain']}, {rec['tokens']} tok) kl={rec['mean_kl']:.5f} top1={rec['top1_agree']:.4f}", flush=True)

    def cat(xs):
        return np.concatenate(xs) if xs else np.array([])

    breakdown = {
        dom: {"samples": len(v["kl"]), **summarize(cat(v["kl"]), cat(v["agree"]), cat(v["nll"]), cat(v["ref_nll"]))}
        for dom, v in per_domain.items()
    }
    allv = {k: cat([x for v in per_domain.values() for x in v[k]]) for k in ("kl", "agree", "nll", "ref_nll")}
    summary = summarize(allv["kl"], allv["agree"], allv["nll"], allv["ref_nll"])
    summary["macro_mean_kl"] = mean(b["mean_kl"] for b in breakdown.values() if b["mean_kl"] is not None)
    summary["macro_top1_agree"] = mean(b["top1_agree"] for b in breakdown.values() if b["top1_agree"] is not None)
    for dom, b in breakdown.items():
        summary[f"{dom}_kl"] = b["mean_kl"]
    # Assistant positions are what the model generates; prompt/tool positions measure how it reads context.
    for role, v in per_role.items():
        kl, agree = cat(v["kl"]), cat(v["agree"])
        if kl.size:
            summary[f"{role}_tokens"] = int(kl.size)
            summary[f"{role}_kl"] = float(kl.mean())
            summary[f"{role}_top1"] = float(agree.mean())
    summary["samples"] = len(samples)
    summary["headline"] = "mean_kl"
    settings = {"ref": str(ref_dir) if tmp is None else "self (smoke)", "chunk": args.chunk, "ref_meta": index_meta}
    write_result(args, "kl", summary=summary, breakdown=breakdown, samples=samples, settings=settings, started=started)
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
