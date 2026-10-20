"""Calibration + held-out eval sets (see docs/contracts/calib_data.md).

uv run python -m calib_data.build --model {glm,qwen} --out artifacts/<model>/data/ [--smoke]
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from . import sources
from .recipes import RECIPES, Ctx, stable_hash
from .render import Renderer, ensure_tokenizer

DOMAINS = ("code", "agentic", "reasoning", "longctx")
FORMAT_VERSION = 1
EVAL_PERMILLE = 300
CALIB_MIX = {"code": 192, "agentic": 160, "reasoning": 144, "longctx": 16}
EVAL_MIX = {"code": 64, "agentic": 62, "reasoning": 62, "longctx": 12}
MAX_LEN = {"code": 6144, "agentic": 8192, "reasoning": 8192}
LONGCTX = (16384, 65536)
SMOKE_MAX_LEN = {"code": 1024, "agentic": 2048, "reasoning": 2048}
SMOKE_LONGCTX = (2048, 4096)


def log(msg):
    print(f"[calib_data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_kv(s: str, cast=float) -> dict:
    out = {}
    for part in filter(None, s.split(",")):
        k, _, v = part.partition("=")
        out[k.strip()] = cast(v)
    return out


def allocate(total: int, weights: dict) -> dict:
    """Largest-remainder split of `total` by `weights`; every positive weight gets >= 1 when possible."""
    keys = [k for k, w in weights.items() if w > 0]
    if not keys or total <= 0:
        return {k: 0 for k in weights}
    s = sum(weights[k] for k in keys)
    raw = {k: total * weights[k] / s for k in keys}
    out = {k: int(raw[k]) for k in keys}
    if total >= len(keys):
        for k in keys:
            out[k] = max(out[k], 1)
    while sum(out.values()) > total:
        k = max((k for k in keys if out[k] > 1), key=lambda k: out[k] - raw[k])
        out[k] -= 1
    for k in sorted(keys, key=lambda k: raw[k] - int(raw[k]), reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return {k: out.get(k, 0) for k in weights}


def split_of(recipe, row) -> str:
    return "eval" if stable_hash(f"{recipe.source}:{recipe.key(row)}") % 1000 < EVAL_PERMILLE else "calib"


def pool_for(recipe, rows, split, seed):
    i, n = recipe.shard
    pool = [r for r in rows if split_of(recipe, r) == split and stable_hash(r["_src"]) % n == i]
    pool.sort(key=lambda r: r["_src"])
    random.Random(f"{seed}:{split}:{recipe.name}").shuffle(pool)
    return pool


class Builder:
    def __init__(self, renderer, load_rows, seed, max_len, longctx):
        self.r = renderer
        self.load_rows = load_rows
        self.seed = seed
        self.max_len = max_len
        self.longctx = longctx
        self._rows = {}

    def rows(self, source):
        if source not in self._rows:
            self._rows[source] = self.load_rows(source)
        return self._rows[source]

    def generator(self, recipe, split):
        ctx = Ctx(self.r, random.Random(f"{self.seed}:{split}:{recipe.name}:ctx"), self.max_len, self.longctx)
        min_len = int(self.longctx[0] * 0.95) if recipe.domain == "longctx" else 1
        for src, toks in recipe.make(pool_for(recipe, self.rows(recipe.source), split, self.seed), ctx):
            if toks and len(toks) >= min_len:
                yield src, toks

    def domain(self, domain, split, n):
        recipes = [r for r in RECIPES if r.domain == domain]
        quota = allocate(n, {r.name: r.weight for r in recipes})
        gens = {r.name: self.generator(r, split) for r in recipes}
        out = []
        for r in recipes:
            out += self._take(r.name, gens[r.name], quota[r.name])
        while len(out) < n and gens:
            for name in list(gens):
                got = self._take(name, gens[name], 1)
                if not got:
                    del gens[name]
                out += got
                if len(out) >= n:
                    break
        if len(out) < n:
            log(f"WARNING {split}/{domain}: only {len(out)}/{n} samples available")
        return out

    def _take(self, name, gen, k):
        got = []
        if k <= 0:
            return got
        for src, toks in gen:
            got.append((name, src, toks))
            if len(got) >= k:
                break
        return got

    def split(self, split, mix):
        samples = []
        for domain, n in mix.items():
            if n <= 0:
                continue
            t = time.time()
            got = self.domain(domain, split, n)
            for i, (recipe, src, toks) in enumerate(got):
                samples.append({"id": f"{split}-{domain}-{i:04d}-{hashlib.sha1(src.encode()).hexdigest()[:8]}",
                                "domain": domain, "source": recipe, "src": src, "tokens": toks})
            log(f"{split}/{domain}: {len(got)} samples, {sum(len(x[2]) for x in got)} tokens ({time.time() - t:.0f}s)")
        return samples


def stats(samples):
    out = {}
    for key in ("domain", "source"):
        groups = {}
        for s in samples:
            groups.setdefault(s[key], []).append(len(s["tokens"]))
        out[f"by_{key}"] = {k: {"n": len(v), "tokens": sum(v), "min": min(v), "mean": round(sum(v) / len(v)),
                               "max": max(v)} for k, v in sorted(groups.items())}
    out["n"] = len(samples)
    out["tokens"] = sum(len(s["tokens"]) for s in samples)
    return out


def validate(calib, evals, vocab_size):
    for s in calib + evals:
        assert s["tokens"], s["id"]
        assert all(0 <= t < vocab_size for t in s["tokens"]), f"{s['id']}: token id out of range"
        assert s["domain"] in DOMAINS
    overlap = {s["src"] for s in calib} & {s["src"] for s in evals}
    assert not overlap, f"calib/eval overlap: {sorted(overlap)[:3]}"
    assert len({s["id"] for s in calib + evals}) == len(calib) + len(evals), "duplicate ids"


def write_jsonl(path: Path, samples):
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for s in samples:
            f.write(json.dumps({"id": s["id"], "domain": s["domain"], "source": s["source"], "src": s["src"],
                                "tokens": s["tokens"]}, separators=(",", ":")) + "\n")
    tmp.replace(path)


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["glm", "qwen"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokenizer", type=Path, help="dir with tokenizer.json + chat template "
                    "(default artifacts/tokenizers/<model>, fetched from the model repo if missing)")
    ap.add_argument("--samples", type=int, help="total calib samples (scales --mix; default 512)")
    ap.add_argument("--eval-samples", type=int, help="total eval samples (scales --eval-mix; default 200)")
    ap.add_argument("--mix", help="calib domain weights, e.g. code=192,agentic=160,reasoning=144,longctx=16")
    ap.add_argument("--eval-mix", help="eval domain weights, e.g. code=64,agentic=62,reasoning=62,longctx=12")
    ap.add_argument("--max-len", help="per-domain token caps, e.g. code=6144,agentic=8192,reasoning=8192")
    ap.add_argument("--longctx", help="longctx length range in tokens, e.g. 16384-65536")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/calib_data"))
    ap.add_argument("--offline", action="store_true", help="only use cached source slices / tokenizer")
    ap.add_argument("--source", choices=["hf", "fixtures"], help="row source (default hf; fixtures under --smoke)")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 calib + 1 eval per domain, short lengths")
    ap.add_argument("--preview", action="store_true", help="also write decoded text of the first sample per domain")
    a = ap.parse_args(argv)

    source = a.source or ("fixtures" if a.smoke else "hf")
    even = {d: 1 for d in DOMAINS}
    mix = parse_kv(a.mix) if a.mix else dict(even if a.smoke else CALIB_MIX)
    emix = parse_kv(a.eval_mix) if a.eval_mix else dict(even if a.smoke else EVAL_MIX)
    max_len = {**(SMOKE_MAX_LEN if a.smoke else MAX_LEN), **parse_kv(a.max_len or "", int)}
    longctx = tuple(map(int, a.longctx.split("-"))) if a.longctx else (SMOKE_LONGCTX if a.smoke else LONGCTX)
    if a.smoke:
        n_calib, n_eval = a.samples or 2 * len(DOMAINS), a.eval_samples or len(DOMAINS)
    else:
        n_calib, n_eval = a.samples or round(sum(mix.values())), a.eval_samples or round(sum(emix.values()))
    unknown = (set(mix) | set(emix)) - set(DOMAINS)
    if unknown:
        ap.error(f"unknown domains {sorted(unknown)}")
    calib_mix, eval_mix = allocate(n_calib, mix), allocate(n_eval, emix)

    tok_dir = ensure_tokenizer(a.model, a.tokenizer, a.offline)
    renderer = Renderer(a.model, tok_dir)
    log(f"model={a.model} tokenizer={tok_dir} vocab={renderer.vocab_size} source={source} "
        f"calib={calib_mix} eval={eval_mix} max_len={max_len} longctx={longctx}")

    if source == "fixtures":
        from .fixtures import fixture_rows

        def load_rows(name):
            return fixture_rows(name)
    else:
        def load_rows(name):
            return sources.load(name, a.cache_dir, a.offline)

    b = Builder(renderer, load_rows, a.seed, max_len, longctx)
    evals = b.split("eval", eval_mix)
    calib = b.split("calib", calib_mix)
    validate(calib, evals, renderer.vocab_size)

    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out / "calib.jsonl", calib)
    write_jsonl(a.out / "eval.jsonl", evals)
    manifest = {
        "format_version": FORMAT_VERSION,
        "model": a.model,
        "seed": a.seed,
        "source": source,
        "smoke": a.smoke,
        "tokenizer": {"dir": str(tok_dir), "tokenizer_json_sha256": file_sha(tok_dir / "tokenizer.json"),
                      "chat_template_sha256": file_sha(tok_dir / "chat_template.jinja"),
                      "vocab_size": renderer.vocab_size},
        "mix": {"calib": calib_mix, "eval": eval_mix},
        "max_len": max_len,
        "longctx": list(longctx),
        "eval_permille": EVAL_PERMILLE,
        "recipes": [{"name": r.name, "domain": r.domain, "source": r.source, "weight": r.weight} for r in RECIPES],
        "sources": {k: [{"repo": p.repo, "rev": p.rev, "path": p.path, **({"row_group": p.row_group} if hasattr(p, "row_group") else {"bytes": p.nbytes})}
                        for p in v] for k, v in sources.SOURCES.items()} if source == "hf" else None,
        "stats": {"calib": stats(calib), "eval": stats(evals)},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if a.preview:
        pdir = a.out / "preview"
        pdir.mkdir(exist_ok=True)
        for split, samples in (("calib", calib), ("eval", evals)):
            seen = set()
            for s in samples:
                if s["source"] not in seen:
                    seen.add(s["source"])
                    (pdir / f"{split}_{s['source']}.txt").write_text(renderer.tok.decode(s["tokens"]))
    for split, st in manifest["stats"].items():
        log(f"{split}: {st['n']} samples / {st['tokens']} tokens  " +
            "  ".join(f"{d}={v['n']}/{v['tokens']}" for d, v in st["by_domain"].items()))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
