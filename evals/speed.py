"""Prefill / decode throughput, TTFT and memory at several context lengths,
with the native MTP drafter off and on.

uv run python -m evals.speed --model models/<name> [--contexts 1024,8192,32768,131072] [--mtp-modes off,on]
"""

import random
import threading
import time
from typing import Any, Dict, List, Optional

from .common import GB, base_parser, load_model, resolve_model_path, weights_gb, write_result
from .longctx import ContextBuilder, filler_sentences

DEFAULT_CONTEXTS = "1024,8192,32768,131072"
SMOKE_CONTEXTS = "128,512"


class RssSampler:
    def __init__(self, interval: float = 0.05):
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def build_prompt(tokenizer, n_tokens: int, seed: int):
    rng = random.Random(f"speed-{seed}-{n_tokens}")
    filler = filler_sentences(rng, max(200, n_tokens // 12))

    def render(content: str):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if text.rstrip().endswith("<think>"):
            text += "</think>"
        return text, tokenizer.encode(text, add_special_tokens=False)

    def make_prompt(context: str) -> str:
        return f"{context}\n\nContinue this chronicle with several new paragraphs in the same style."

    return ContextBuilder(tokenizer, render).build(n_tokens, make_prompt, [], filler)


def measure(model, processor, prompt: str, ids: List[int], decode_tokens: int, prefill_step: int, draft=None, kind=None):
    import mlx.core as mx
    from mlx_vlm import stream_generate
    from mlx_vlm.models import cache as cache_mod

    tokenizer = getattr(processor, "tokenizer", processor)
    eos = model.config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
    tokenizer.stopping_criteria.reset(eos)
    prompt_cache = cache_mod.make_prompt_cache(model.language_model)
    kwargs: Dict[str, Any] = dict(
        input_ids=mx.array([ids]),
        max_tokens=decode_tokens,
        temperature=0.0,
        prefill_step_size=prefill_step,
        prompt_cache=prompt_cache,
        logit_bias={int(e): -1e9 for e in eos},
    )
    if draft is not None:
        kwargs.update(draft_model=draft, draft_kind=kind)
    mx.clear_cache()
    mx.reset_peak_memory()
    t_first = t_last = None
    n, last = 0, None
    with RssSampler() as rss:
        start = time.perf_counter()
        for r in stream_generate(model, processor, prompt, **kwargs):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            last = r
        n = last.generation_tokens if last else 0
    ttft = (t_first or time.perf_counter()) - start
    decode_s = (t_last - t_first) if t_first is not None and t_last is not None else 0.0
    cache_bytes = 0
    for c in prompt_cache:
        try:
            cache_bytes += c.nbytes
        except Exception:
            pass
    res = {
        "prompt_tokens": len(ids),
        "ttft_s": round(ttft, 4),
        "prefill_tps": round(len(ids) / ttft, 2) if ttft > 0 else None,
        "decode_tokens": n,
        "decode_tps": round((n - 1) / decode_s, 2) if n > 1 and decode_s > 0 else None,
        "decode_tps_reported": round(last.generation_tps, 2) if last else None,
        "peak_mlx_gb": round(mx.get_peak_memory() / GB, 3),
        "active_mlx_gb": round(mx.get_active_memory() / GB, 3),
        "peak_rss_gb": round(rss.peak / GB, 3),
        "cache_gb": round(cache_bytes / GB, 4),
    }
    del prompt_cache
    mx.clear_cache()
    return res


def main(argv=None):
    import mlx.core as mx

    from .generation import load_mtp

    p = base_parser("speed", __doc__)
    p.add_argument("--contexts", help=f"prompt lengths (default {DEFAULT_CONTEXTS})")
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--mtp-modes", default="off,on")
    p.add_argument("--draft-model")
    p.add_argument("--prefill-step-size", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args(argv)
    started = time.time()

    contexts = [int(x) for x in (args.contexts or (SMOKE_CONTEXTS if args.smoke else DEFAULT_CONTEXTS)).split(",")]
    decode_tokens = min(args.decode_tokens, 16) if args.smoke else args.decode_tokens
    modes = [m.strip() for m in args.mtp_modes.split(",") if m.strip()]

    model_path = resolve_model_path(args.model)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, processor = load_model(args)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    resident_gb = mx.get_active_memory() / GB
    tokenizer = getattr(processor, "tokenizer", processor)

    drafter, kind, mtp_status = None, None, "not requested"
    if "on" in modes:
        drafter, info = load_mtp(model, model_path, args.draft_model)
        if drafter is None:
            mtp_status = f"unavailable ({info})"
            modes = [m for m in modes if m != "on"]
        else:
            kind, mtp_status = info, "on"
            mx.eval(drafter.parameters())

    for n in (64, min(contexts)):
        warm_prompt, warm_ids = build_prompt(tokenizer, n, args.seed + 1)
        measure(model, processor, warm_prompt, warm_ids, 4, args.prefill_step_size)
        if drafter is not None:
            measure(model, processor, warm_prompt, warm_ids, 4, args.prefill_step_size, drafter, kind)

    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        prompt, ids = build_prompt(tokenizer, ctx, args.seed)
        for mode in modes:
            for rep in range(args.repeats):
                d = drafter if mode == "on" else None
                r = measure(model, processor, prompt, ids, decode_tokens, args.prefill_step_size, d, kind)
                r.update(context=ctx, mtp=mode, repeat=rep)
                rows.append(r)
                print(
                    f"[speed] ctx={ctx} mtp={mode} prefill={r['prefill_tps']} tok/s ttft={r['ttft_s']}s "
                    f"decode={r['decode_tps']} tok/s peak={r['peak_mlx_gb']} GB cache={r['cache_gb']} GB",
                    flush=True,
                )

    def best(ctx: int, mode: str, key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r["context"] == ctx and r["mtp"] == mode and r[key] is not None]
        if not vals:
            return None
        return min(vals) if key in ("ttft_s",) or key.startswith("peak") else max(vals)

    breakdown: Dict[str, Any] = {}
    summary: Dict[str, Any] = {"weights_gb": round(weights_gb(model_path), 3), "resident_gb": round(resident_gb, 3)}
    for ctx in contexts:
        b: Dict[str, Any] = {}
        for mode in modes:
            for key in ("prefill_tps", "ttft_s", "decode_tps", "peak_mlx_gb", "peak_rss_gb", "cache_gb"):
                b[f"{key}_mtp_{mode}"] = best(ctx, mode, key)
        breakdown[str(ctx)] = b
        summary[f"prefill_tps@{ctx}"] = b.get("prefill_tps_mtp_off", b.get("prefill_tps_mtp_on"))
        summary[f"decode_tps@{ctx}"] = b.get("decode_tps_mtp_off", b.get("decode_tps_mtp_on"))
        if "on" in modes:
            summary[f"decode_tps_mtp@{ctx}"] = b.get("decode_tps_mtp_on")
        summary[f"ttft_s@{ctx}"] = b.get("ttft_s_mtp_off", b.get("ttft_s_mtp_on"))
        summary[f"peak_gb@{ctx}"] = max(v for k, v in b.items() if k.startswith("peak_mlx") and v is not None)
        summary[f"cache_gb@{ctx}"] = b.get("cache_gb_mtp_off", b.get("cache_gb_mtp_on"))
    summary["peak_gb"] = max(r["peak_mlx_gb"] for r in rows) if rows else None
    summary["headline"] = f"decode_tps@{contexts[0]}"
    settings = {
        "contexts": contexts,
        "decode_tokens": decode_tokens,
        "mtp_modes": modes,
        "mtp": mtp_status,
        "prefill_step_size": args.prefill_step_size,
        "repeats": args.repeats,
        "load_s": round(load_s, 2),
        "sampling": "greedy, EOS suppressed",
    }
    write_result(args, "speed", summary=summary, breakdown=breakdown, samples=rows, settings=settings, started=started)


if __name__ == "__main__":
    main()
