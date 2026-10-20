"""Markdown comparison of results/<model>/<suite>.json across models.

uv run python -m evals.compare [--results-dir results] [--models a,b] [--out results/compare.md]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

SUITES = ("kl", "speed", "code", "math", "longctx", "tools")
HEADLINE = [
    ("kl", "mean_kl"),
    ("kl", "top1_agree"),
    ("speed", "decode_tps@1024"),
    ("speed", "decode_tps_mtp@1024"),
    ("speed", "prefill_tps@32768"),
    ("speed", "peak_gb@131072"),
    ("code", "pass@1"),
    ("math", "accuracy"),
    ("longctx", "score"),
    ("tools", "accuracy"),
]
_LOWER = re.compile(r"(kl|ttft|peak|cache|nll|ppl_ratio|weights_gb|resident_gb|no_call|truncated|no_answer)")
_SKIP = {"headline", "n", "samples", "tokens"}


def lower_is_better(key: str) -> bool:
    return bool(_LOWER.search(key.split(":", 1)[-1]))


def load_results(results_dir: Path, models: Optional[List[str]], include_smoke: bool) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    dirs = sorted(d for d in results_dir.iterdir() if d.is_dir()) if results_dir.exists() else []
    for d in dirs:
        if models and d.name not in models:
            continue
        for suite in SUITES:
            f = d / f"{suite}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                rec = json.load(fh)
            if rec.get("smoke") and not include_smoke:
                continue
            out.setdefault(d.name, {})[suite] = rec
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        if abs(v) >= 1e-3:
            return f"{v:.4f}"
        return f"{v:.2e}"
    return str(v)


def table(rows: List[str], cols: List[str], get, header0: str = "model") -> List[str]:
    best: Dict[str, Any] = {}
    for c in cols:
        vals = [get(r, c) for r in rows]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) > 1:
            best[c] = min(nums) if lower_is_better(c) else max(nums)
    head = [header0] + [f"{c} {'↓' if lower_is_better(c) else '↑'}" for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r]
        for c in cols:
            v = get(r, c)
            s = fmt(v)
            if c in best and v == best[c]:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(results: Dict[str, Dict[str, Any]]) -> str:
    models = sorted(results)
    if not models:
        return "# Model comparison\n\nNo results found.\n"

    def summary(m: str, suite: str) -> Dict[str, Any]:
        return results[m].get(suite, {}).get("summary", {})

    out = ["# Model comparison", ""]
    head_cols = [(s, k) for s, k in HEADLINE if any(k in summary(m, s) for m in models)]
    if head_cols:
        out += ["## Headline", ""]
        out += table(models, [f"{s}:{k}" for s, k in head_cols], lambda m, c: summary(m, c.split(":", 1)[0]).get(c.split(":", 1)[1]))
        out.append("")
    for suite in SUITES:
        present = [m for m in models if suite in results[m]]
        if not present:
            continue
        keys: List[str] = []
        for m in present:
            for k, v in summary(m, suite).items():
                if k not in _SKIP and k not in keys and (v is None or isinstance(v, (int, float))):
                    keys.append(k)
        out += [f"## {suite}", ""]
        out += table(present, keys, lambda m, c, s=suite: summary(m, s).get(c))
        notes = []
        for m in present:
            st = results[m][suite].get("settings", {})
            bits = [f"{k}={st[k]}" for k in ("thinking_budget", "mtp") if k in st]
            samp = st.get("sampling")
            if isinstance(samp, dict):
                bits.append("T={temperature} top_p={top_p} top_k={top_k}".format(**samp))
            elif samp:
                bits.append(str(samp))
            if results[m][suite].get("smoke"):
                bits.append("SMOKE")
            if bits:
                notes.append(f"- {m}: " + ", ".join(str(b) for b in bits))
        if notes:
            out += [""] + notes
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="evals.compare", description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models", help="comma-separated result names (default: all)")
    p.add_argument("--out", help="write markdown here (default: stdout)")
    p.add_argument("--include-smoke", action="store_true")
    args = p.parse_args(argv)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    md = render(load_results(Path(args.results_dir), models, args.include_smoke))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
