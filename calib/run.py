"""Layer-wise streaming calibration (see docs/contracts/calib.md).

uv run python -m calib.run --model {glm,qwen} --src <hf snapshot> --data <dir> --out <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import sens as S
from .arch import MLX_LAYER_PREFIX, make_arch
from .probe import LayerRecorder, Reservoir, balanced_mean

DOMAIN_ORDER = ("code", "agentic", "reasoning", "longctx")
FORMAT_VERSION = 1


def log(msg):
    print(f"[calib {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_samples(data_dir: Path, limit=None):
    samples = []
    for split in ("calib", "eval"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            if split == "calib":
                raise FileNotFoundError(path)
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if limit is not None:
            rows = rows[:limit]
        for r in rows:
            if not r["tokens"]:
                continue
            samples.append(
                {"id": str(r["id"]), "domain": r.get("domain", "default"), "split": split,
                 "tokens": np.asarray(r["tokens"], dtype=np.int64)}
            )
    return samples


def domain_list(samples):
    present = {s["domain"] for s in samples if s["split"] == "calib"}
    return [d for d in DOMAIN_ORDER if d in present] + sorted(present - set(DOMAIN_ORDER))


def chunk_bounds(n, size, min_tail=64):
    b = list(range(0, n, size)) + [n]
    if len(b) > 2 and b[-1] - b[-2] < min_tail:
        del b[-2]
    return list(zip(b[:-1], b[1:]))


class HiddenWriter:
    """Writes one sample's hidden states as .npy; promotes fp16 to fp32 on overflow."""

    def __init__(self, path: Path, shape, dtype):
        self.path, self.shape = path, shape
        self.mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    def write(self, a, b, x: np.ndarray):
        if self.mm.dtype == np.float16 and np.abs(x).max(initial=0) > 65000:
            old = np.array(self.mm[:a], dtype=np.float32)
            del self.mm
            self.mm = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.float32, shape=self.shape)
            self.mm[:a] = old
        self.mm[a:b] = x

    def close(self):
        self.mm.flush()
        del self.mm


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _save_npz(path: Path, arrays: dict):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _grid_arrays():
    return {"sens_bits": np.array(S.BITS), "sens_groups": np.array(S.GROUPS)}


def _sens_map(tensors, arrays):
    """Top-level JSON {name: {"b_g": v}} that build/ reads; routed experts are also
    given per expert as mlp.experts.{e}.{proj}. Cells without a value are left out."""
    out = {}
    for name, t in tensors.items():
        g = {k: v for k, v in t["sens"].items() if v is not None}
        if g:
            out[name] = g
        if t["kind"] == "switch":
            proj = name.rsplit(".", 1)[1]
            for e, ge in enumerate(arrays[f"sens.{name}"]):
                g = {k: v for k, v in S.grid_dict(ge).items() if v is not None}
                if g:
                    out[f"mlp.experts.{e}.{proj}"] = g
    return out


def _sha16(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class Calibrator:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.work = self.out / "work"
        self.dtype = getattr(mx, args.dtype)
        self.store_dtype = np.dtype(args.store_dtype)
        self.arch = make_arch(args.model, args.src, self.dtype)
        self.samples = load_samples(Path(args.data), args.limit)
        self.domains = domain_list(self.samples)
        if not self.domains:
            raise ValueError("calib.jsonl has no samples")
        self._check_data()
        self.out.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(exist_ok=True)
        self._check_meta()

    # ---- bookkeeping
    def _meta(self):
        a = self.args
        return {
            "format_version": FORMAT_VERSION,
            "model": a.model,
            "model_type": self.arch.model_type,
            "src": str(Path(a.src).resolve()),
            "num_layers": self.arch.n_layers,
            "domains": self.domains,
            "n_samples": {sp: sum(s["split"] == sp for s in self.samples) for sp in ("calib", "eval")},
            "n_tokens": {sp: int(sum(len(s["tokens"]) for s in self.samples if s["split"] == sp))
                         for sp in ("calib", "eval")},
            "grid": {"bits": list(S.BITS), "group_sizes": list(S.GROUPS), "mode": "affine"},
            "dtype": a.dtype,
            "chunk": a.chunk,
            "lin_tokens": a.lin_tokens,
            "expert_tokens": a.expert_tokens,
            "topk": a.topk,
            "seed": a.seed,
        }

    def _check_data(self):
        """The data dir must be tokenized for this model (calib_data's manifest.json)."""
        data, src = Path(self.args.data), Path(self.args.src)
        vocab = self.arch.snap.shape("model.language_model.embed_tokens.weight")[0]
        hi = max(int(s["tokens"].max()) for s in self.samples)
        if hi >= vocab:
            raise SystemExit(f"{data}: token id {hi} is outside the {vocab}-row embedding of {src}")
        path = data / "manifest.json"
        if not path.exists():
            log(f"{data} has no manifest.json; tokenizer match not checked")
            return
        man = json.loads(path.read_text())
        tok = man.get("tokenizer") or {}
        problems = []
        if man.get("model") not in (None, self.args.model):
            problems.append(f"built for --model {man['model']}")
        for fname, key in (("tokenizer.json", "tokenizer_json_sha256"),
                           ("chat_template.jinja", "chat_template_sha256")):
            if tok.get(key) and (src / fname).exists() and _sha16(src / fname) != tok[key]:
                problems.append(f"{fname} differs from {src / fname}")
        if (tok.get("vocab_size") or 0) > vocab:
            problems.append(f"tokenizer vocab {tok['vocab_size']} > embedding rows {vocab}")
        if problems and not self.args.allow_data_mismatch:
            raise SystemExit(f"{data} does not match --model {self.args.model} --src {src}: "
                             f"{'; '.join(problems)} (--allow-data-mismatch to override)")
        for msg in problems:
            log(f"warning: {msg}")

    def _check_meta(self):
        path = self.out / "meta.json"
        meta = self._meta()
        if path.exists():
            old = json.loads(path.read_text())
            keys = ("model", "model_type", "num_layers", "domains", "n_samples", "n_tokens", "grid", "dtype")
            diff = [k for k in keys if old.get(k) != meta.get(k)]
            if diff and not self.args.restart:
                raise SystemExit(f"{path} differs in {diff}; pass --restart to recalibrate")
            if diff:
                self._reset()
        _atomic_json(path, meta)

    def _reset(self):
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir()
        (self.out / "state.json").unlink(missing_ok=True)

    def state(self):
        p = self.out / "state.json"
        return json.loads(p.read_text()) if p.exists() else {"next_layer": None}

    def set_state(self, **kw):
        _atomic_json(self.out / "state.json", kw)

    def hdir(self, i):
        return self.work / f"h_{i:03d}"

    def _clean_work(self, keep):
        """Drop hidden buffers left by an interrupted run, except stage input `keep`."""
        for d in self.work.iterdir():
            if keep is None or d.name != keep.name:
                log(f"removing stale {d}")
                shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink()

    def load_hidden(self, i, k):
        return np.load(self.hdir(i) / f"{k:05d}.npy", mmap_mode="r")

    # ---- stages
    def run(self):
        if self.args.restart and (self.out / "state.json").exists():
            self._reset()
        mx.set_memory_limit(int(self.args.mem_gb * 2**30))
        mx.set_cache_limit(int(2 * 2**30))
        st = self.state()
        if st.get("done"):
            log("already complete")
            return
        self._clean_work(None if st["next_layer"] is None else self.hdir(st["next_layer"]))
        self._report_disk()
        if st["next_layer"] is None:
            self.embed_stage()
            st = {"next_layer": 0}
        stop = self.arch.n_layers if self.args.layers is None else min(self.args.layers, self.arch.n_layers)
        for i in range(st["next_layer"], stop):
            self.layer_stage(i)
        if stop < self.arch.n_layers:
            log(f"stopped after layer {stop - 1} (--layers)")
            return
        self.head_stage()

    def _report_disk(self):
        per_tok = int(np.prod(self.arch.hidden_shape(1))) * self.store_dtype.itemsize
        n = sum(len(s["tokens"]) for s in self.samples)
        need = 2 * per_tok * n
        free = shutil.disk_usage(self.out).free
        log(f"{len(self.samples)} samples / {n} tokens; hidden store ~{need / 2**30:.1f} GiB "
            f"(free {free / 2**30:.0f} GiB)")
        if need > free:
            raise SystemExit("not enough free disk for the hidden-state double buffer")

    def embed_stage(self):
        t0 = time.time()
        d = self.hdir(0)
        shutil.rmtree(d, ignore_errors=True)
        tmp = self.work / "h_000.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        rng = np.random.default_rng(self.args.seed)
        res = Reservoir(len(self.domains), max(1, self.args.lin_tokens // len(self.domains)), rng)
        for k, s in enumerate(self.samples):
            toks = s["tokens"]
            w = HiddenWriter(tmp / f"{k:05d}.npy", self.arch.hidden_shape(len(toks)), self.store_dtype)
            for a, b in chunk_bounds(len(toks), self.args.chunk):
                h = self.arch.embed(toks[a:b])
                w.write(a, b, np.array(h.astype(mx.float32)))
            w.close()
            if s["split"] == "calib":
                res.offer(np.full(len(toks), self.domains.index(s["domain"]), dtype=np.int64),
                          lambda keep: toks[keep][:, None])
        os.replace(tmp, d)
        key = "model.language_model.embed_tokens.weight"
        rows = [r[:, 0] for r in res.rows if r is not None]
        ids = np.sort(np.concatenate(rows)) if rows else np.zeros(0, np.int64)
        tensors, arrays = {}, _grid_arrays()
        if len(ids):
            grid = S.rows_sens(self.arch.snap.gather_rows(key, ids))
            name = "language_model.model.embed_tokens"
            tensors[name] = {"kind": "embedding", "shape": list(self.arch.snap.shape(key)),
                             "n_params": int(np.prod(self.arch.snap.shape(key))),
                             "n_samples": int(len(ids)), "sens": S.grid_dict(grid)}
            arrays[f"sens.{name}"] = grid
        _atomic_json(self.out / "embed.json", {"tensors": tensors, "sens": _sens_map(tensors, arrays),
                                               "grid": self._meta()["grid"]})
        _save_npz(self.out / "embed.npz", arrays)
        self.set_state(next_layer=0)
        log(f"embeddings done in {time.time() - t0:.1f}s")

    def layer_stage(self, i):
        t0 = time.time()
        mx.reset_peak_memory()
        a = self.args
        layer = self.arch.load_layer(i)
        t_load = time.time() - t0
        targets = self.arch.quant_targets(layer, i)
        mlp = self.arch.moe(layer)
        rec = LayerRecorder(self.arch, layer, targets, mlp, self.domains,
                            a.lin_tokens, a.expert_tokens, seed=a.seed * 1000 + i)
        src_dir, dst = self.hdir(i), self.hdir(i + 1)
        tmp = self.work / f"h_{i + 1:03d}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.mkdir()
        for k, s in enumerate(self.samples):
            toks = s["tokens"]
            hin = self.load_hidden(i, k)
            w = HiddenWriter(tmp / f"{k:05d}.npy", hin.shape, self.store_dtype)
            cache = self.arch.make_cache(layer)
            rec.begin(s["domain"], s["split"] == "calib")
            for lo, hi in chunk_bounds(len(toks), a.chunk):
                h = mx.array(np.asarray(hin[lo:hi], dtype=np.float32)).astype(self.dtype)[None]
                out = self.arch.forward(layer, h, mx.array(toks[lo:hi])[None], cache)
                rec.flush(out, hi - lo)
                w.write(lo, hi, np.array(out[0].astype(mx.float32)))
            w.close()
            del cache, hin
        os.replace(tmp, dst)
        t_fwd = time.time() - t0 - t_load
        meta, arrays = self.layer_stats(i, layer, rec, targets, mlp)
        rec.remove()
        _save_npz(self.out / f"layer_{i:03d}.npz", arrays)
        _atomic_json(self.out / f"layer_{i:03d}.json", meta)
        self.set_state(next_layer=i + 1)
        shutil.rmtree(src_dir, ignore_errors=True)
        del layer, rec
        mx.clear_cache()
        log(f"layer {i:3d} ({meta['layer_type']}/{meta['mlp_type']}): load {t_load:.1f}s "
            f"fwd {t_fwd:.1f}s total {time.time() - t0:.1f}s peak {mx.get_peak_memory() / 2**30:.2f} GiB")

    def layer_stats(self, i, layer, rec, targets, mlp):
        arrays = _grid_arrays()
        tensors = {}
        for name, st in rec.lin.items():
            w = st.module.weight
            entry = {"kind": "multilinear" if st.multi else "linear", "shape": list(w.shape),
                     "n_params": int(w.size)}
            groups = {}
            for key, per_dom in st.res.items():
                rows = [r.rows[0] for r in per_dom if r.rows[0] is not None]
                if rows:
                    groups[key] = np.concatenate(rows)
            n = int(sum(len(g) for g in groups.values()))
            entry["n_samples"] = n
            entry["sens"] = S.grid_dict(np.full((len(S.BITS), len(S.GROUPS)), np.nan))
            if n:
                if st.multi:
                    grid, energy = S.multilinear_sens(w, groups)
                else:
                    grid, energy = S.linear_sens(w, groups[("t", 0)])
                entry["sens"] = S.grid_dict(grid)
                entry["out_sq"] = energy
                arrays[f"sens.{name}"] = grid
            absmean = balanced_mean(st.abs_sum, st.count)
            sqmean = balanced_mean(st.sq_sum, st.count)
            if absmean is not None:
                arrays[f"act_absmean.{name}"] = absmean.astype(np.float32)
                arrays[f"act_sqmean.{name}"] = sqmean.astype(np.float32)
            tensors[name] = entry
        moe = None
        if mlp is not None:
            moe, expert_tensors = self.moe_stats(rec, mlp, arrays)
            tensors.update(expert_tensors)
        ids = rec.ple_ids() if rec.ple is not None else None
        if ids is not None:
            rows = mx.array(rec.ple.rows(ids)).view(mx.bfloat16)
            grid = S.rows_sens(rows)
            name = "ple.ple_embedding.ngram_embedding"
            tensors[name] = {"kind": "embedding", "shape": [rec.ple.row_count, rec.ple.dims],
                             "n_params": rec.ple.row_count * rec.ple.dims,
                             "n_samples": int(len(ids)), "sens": S.grid_dict(grid)}
            arrays[f"sens.{name}"] = grid
        tokens = {d: int(rec.tokens[j]) for j, d in enumerate(self.domains)}
        arrays["domain_tokens"] = rec.tokens.astype(np.int64)
        layer_type = getattr(layer, "block_type", None) or getattr(layer, "layer_type")
        meta = {
            "format_version": FORMAT_VERSION,
            "layer": i,
            "prefix": MLX_LAYER_PREFIX.format(i),
            "layer_type": layer_type,
            "mlp_type": "moe" if mlp is not None else "dense",
            "domains": self.domains,
            "n_tokens": tokens,
            "grid": self._meta()["grid"],
            "tensors": tensors,
            "sens": _sens_map(tensors, arrays),
            "moe": moe,
        }
        return meta, arrays

    def moe_stats(self, rec, mlp, arrays):
        T = np.maximum(rec.tokens, 1.0)[:, None]
        has = (rec.tokens > 0)[:, None]
        sw = mlp.switch_mlp
        E = rec.E
        K = getattr(mlp, "top_k", None) or mlp.gate.top_k
        freq_d = rec.count / (T * K)
        reap_d = np.where(rec.count > 0, rec.gnorm / np.maximum(rec.count, 1), 0.0)
        prob_d = rec.psum / T
        nd_has = max(int(has.sum()), 1)
        freq = (freq_d * has).sum(0) / nd_has
        prob = (prob_d * has).sum(0) / nd_has
        cnt_w = (rec.count / T * has).sum(0)
        reap = np.where(cnt_w > 0, (rec.gnorm / T * has).sum(0) / np.maximum(cnt_w, 1e-30), 0.0)
        out_norm = np.where(cnt_w > 0, (rec.nsum / T * has).sum(0) / np.maximum(cnt_w, 1e-30), 0.0)
        gate_mean = np.where(cnt_w > 0, (rec.wsum / T * has).sum(0) / np.maximum(cnt_w, 1e-30), 0.0)
        arrays.update({
            "expert_freq": freq, "expert_reap": reap, "router_prob": prob,
            "expert_out_norm": out_norm, "expert_gate_mean": gate_mean,
            "routed_count": rec.count.sum(0).astype(np.int64),
        })
        per_domain = {}
        for j, d in enumerate(self.domains):
            arrays[f"expert_freq.{d}"] = freq_d[j]
            arrays[f"expert_reap.{d}"] = reap_d[j]
            arrays[f"router_prob.{d}"] = prob_d[j]
            arrays[f"routed_count.{d}"] = rec.count[j].astype(np.int64)
            per_domain[d] = {"expert_freq": freq_d[j].tolist(), "expert_reap": reap_d[j].tolist(),
                             "router_prob": prob_d[j].tolist(),
                             "routed_count": rec.count[j].astype(int).tolist()}
        x_abs = sum(np.array(a, dtype=np.float64) / T[j] for j, a in enumerate(rec.x_abs))
        x_sq = sum(np.array(a, dtype=np.float64) / T[j] for j, a in enumerate(rec.x_sq))
        x_abs = x_abs / np.maximum(cnt_w, 1e-30)[:, None]
        x_sq = x_sq / np.maximum(cnt_w, 1e-30)[:, None]
        samples, weights = rec.expert_samples()
        grids, energy, h_abs, h_sq = S.expert_sens(sw, samples, sw.activation, weights)
        n_samp = np.array([0 if r is None else len(r) for r in samples])
        tensors = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            name = f"mlp.switch_mlp.{p}"
            w = getattr(sw, p).weight
            g = grids[p]
            ok = ~np.isnan(g)
            wts = freq[:, None, None] * ok
            agg = np.where(wts.sum(0) > 0, np.nansum(g * wts, 0) / np.maximum(wts.sum(0), 1e-30), np.nan)
            arrays[f"sens.{name}"] = g
            arrays[f"out_sq.{name}"] = energy[p]
            if p == "down_proj":
                arrays[f"act_absmean.{name}"] = h_abs.astype(np.float32)
                arrays[f"act_sqmean.{name}"] = h_sq.astype(np.float32)
            else:
                arrays[f"act_absmean.{name}"] = x_abs.astype(np.float32)
                arrays[f"act_sqmean.{name}"] = x_sq.astype(np.float32)
            tensors[name] = {"kind": "switch", "shape": list(w.shape), "n_params": int(w.size),
                             "n_samples": int(n_samp.sum()), "sens": S.grid_dict(agg),
                             "sens_aggregate": "expert_freq-weighted mean of per-expert NMSE"}
        moe = {"n_experts": E, "top_k": int(K), "expert_freq": freq.tolist(),
               "expert_reap": reap.tolist(), "router_prob": prob.tolist(),
               "routed_count": rec.count.sum(0).astype(int).tolist(),
               "expert_samples": n_samp.astype(int).tolist(), "per_domain": per_domain}
        return moe, tensors

    def head_stage(self):
        t0 = time.time()
        a = self.args
        n = self.arch.n_layers
        norm, lm_head = self.arch.load_head()
        ref = self.out / "ref"
        shutil.rmtree(ref, ignore_errors=True)
        tmp = self.out / "ref.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        rng = np.random.default_rng(a.seed + 7)
        res = {d: Reservoir(1, max(1, a.lin_tokens // len(self.domains)), rng) for d in self.domains}
        index = []
        for k, s in enumerate(self.samples):
            toks = s["tokens"]
            hin = self.load_hidden(n, k)
            evals = s["split"] == "eval"
            if evals:
                L = len(toks)
                top_idx = np.zeros((L, a.topk), dtype=np.int32)
                top_lp = np.zeros((L, a.topk), dtype=np.float32)
                tgt = np.full(L, np.nan, dtype=np.float32)
            for lo, hi in chunk_bounds(len(toks), a.head_chunk):
                h = mx.array(np.asarray(hin[lo:hi], dtype=np.float32)).astype(self.dtype)[None]
                x = self.arch.head_input(norm, h)
                if not evals:
                    xs = x[0]
                    mx.eval(xs)
                    res[s["domain"]].offer(np.zeros(hi - lo, dtype=np.int64),
                                           lambda keep: np.array(xs[mx.array(keep)].astype(mx.float32)))
                    continue
                logits = lm_head(x)[0].astype(mx.float32)
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                idx = mx.argpartition(-lp, kth=a.topk - 1, axis=-1)[:, : a.topk]
                vals = mx.take_along_axis(lp, idx, axis=-1)
                order = mx.argsort(-vals, axis=-1)
                idx = mx.take_along_axis(idx, order, axis=-1)
                vals = mx.take_along_axis(vals, order, axis=-1)
                nxt = toks[lo + 1 : hi + 1]
                t = mx.take_along_axis(lp[: len(nxt)], mx.array(nxt)[:, None].astype(mx.int32), axis=-1)
                mx.eval(idx, vals, t)
                top_idx[lo:hi] = np.array(idx)
                top_lp[lo:hi] = np.array(vals)
                tgt[lo : lo + len(nxt)] = np.array(t)[:, 0]
            if evals:
                fname = f"{k:05d}.npz"
                np.savez(tmp / fname, tokens=toks.astype(np.int32), top_idx=top_idx,
                         top_logprob=top_lp, target_logprob=tgt)
                index.append({"id": s["id"], "domain": s["domain"], "file": fname,
                              "n_tokens": int(len(toks)),
                              "nll": float(-np.nanmean(tgt)) if len(toks) > 1 else None})
        _atomic_json(tmp / "index.json", {"model": a.model, "topk": a.topk,
                                          "src": str(Path(a.src).resolve()), "samples": index})
        os.replace(tmp, ref)
        rows = [r.rows[0] for r in res.values() if r.rows[0] is not None]
        tensors, arrays = {}, _grid_arrays()
        if rows:
            grid, energy = S.linear_sens(lm_head.weight, np.concatenate(rows))
            name = "language_model.lm_head"
            tensors[name] = {"kind": "linear", "shape": list(lm_head.weight.shape),
                             "n_params": int(lm_head.weight.size),
                             "n_samples": int(sum(len(r) for r in rows)),
                             "sens": S.grid_dict(grid), "out_sq": energy}
            arrays[f"sens.{name}"] = grid
        _atomic_json(self.out / "head.json", {"tensors": tensors, "sens": _sens_map(tensors, arrays),
                                              "grid": self._meta()["grid"]})
        _save_npz(self.out / "head.npz", arrays)
        if not a.keep_hidden:
            shutil.rmtree(self.hdir(n), ignore_errors=True)
        self.set_state(next_layer=n, done=True)
        log(f"head/ref done in {time.time() - t0:.1f}s ({len(index)} eval samples)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Layer-wise streaming calibration")
    ap.add_argument("--model", choices=["glm", "qwen"], required=True)
    ap.add_argument("--src", required=True, help="HF snapshot dir (safetensors + index)")
    ap.add_argument("--data", required=True, help="dir with calib.jsonl / eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048, help="tokens per forward chunk")
    ap.add_argument("--head-chunk", type=int, default=512)
    ap.add_argument("--lin-tokens", type=int, default=2048,
                    help="sampled input rows per dense projection (split evenly over domains)")
    ap.add_argument("--expert-tokens", type=int, default=128, help="sampled routed rows per expert")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--store-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--mem-gb", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=None, help="use the first N samples of each split")
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-hidden", action="store_true")
    ap.add_argument("--restart", action="store_true", help="discard previous progress")
    ap.add_argument("--allow-data-mismatch", action="store_true",
                    help="run even if data/manifest.json names another model or tokenizer")
    args = ap.parse_args(argv)
    if args.chunk <= 8:
        ap.error("--chunk must exceed the decode block size (8)")
    return args


def main(argv=None):
    Calibrator(parse_args(argv)).run()


if __name__ == "__main__":
    main()
