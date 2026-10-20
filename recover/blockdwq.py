"""Blockwise local DWQ (docs/plan/05 §2.8 B1, 06 stage A).

Streams a teacher (bf16 layers from the source snapshot, via calib.arch) and the built quantized student side
by side, one decoder layer at a time. For layer i the student layer is fed the *student's* own input stream and
its quantized ``scales``/``biases`` are trained so its output matches the *teacher's* output, so each layer also
absorbs the error accumulated upstream. Codes (``weight``), bit widths and byte counts never change. A layer's
update is kept only if the held-out relative error improves with the parameters rounded to the stored dtype.

uv run python -m recover.blockdwq --model glm --src models/src-glm-5.3-flash --student models/<built> \
    --data artifacts/glm/data --out models/<built>-dwq
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from calib.arch import MLX_LAYER_PREFIX, make_arch
from calib.run import load_samples

GB = 1 << 30


def log(msg):
    print(f"[dwq {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Student:
    """Loads single layers / the embedding of a built (quantized) checkpoint."""

    def __init__(self, model: str, path):
        self.path = Path(path)
        self.arch = make_arch(model, self.path)
        self.qcfg = json.loads((self.path / "config.json").read_text()).get("quantization", {})
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def _load(self, prefix: str) -> Dict[str, mx.array]:
        by_file: Dict[str, List[str]] = {}
        for k, f in self.weight_map.items():
            if k.startswith(prefix):
                by_file.setdefault(f, []).append(k)
        out = {}
        for f, ks in by_file.items():
            arrays = mx.load(str(self.path / f))
            out.update({k[len(prefix):]: arrays[k] for k in ks})
        return out

    def _quantize(self, module: nn.Module, prefix: str):
        def pred(p, m):
            c = self.qcfg.get(prefix + p)
            return {"group_size": c["group_size"], "bits": c["bits"], "mode": c.get("mode", "affine")} if isinstance(c, dict) else False

        nn.quantize(module, class_predicate=pred)

    def layer(self, i: int) -> nn.Module:
        layer = self.arch.build_layer(i)
        prefix = MLX_LAYER_PREFIX.format(i)
        self._quantize(layer, prefix)
        layer.load_weights(list(self._load(prefix).items()), strict=True)
        layer.eval()
        return layer

    def embed(self, tokens: np.ndarray) -> mx.array:
        prefix = "language_model.model.embed_tokens."
        w = self._load(prefix)
        c = self.qcfg.get(prefix[:-1])
        if isinstance(c, dict):
            rows = mx.array(np.asarray(tokens, dtype=np.int64))
            h = mx.dequantize(w["weight"][rows], w["scales"][rows], w["biases"][rows],
                              group_size=c["group_size"], bits=c["bits"])
        else:
            h = w["weight"][mx.array(np.asarray(tokens, dtype=np.int64))]
        return self.arch.expand_streams(h)


def trainable(layer: nn.Module, max_bits: int = 8):
    """Unfreeze only scales/biases of quantized modules below ``max_bits``; returns their dotted names."""
    layer.freeze()
    names = []
    for path, m in layer.named_modules():
        bits = getattr(m, "bits", None)
        if bits is not None and hasattr(m, "scales") and bits < max_bits:
            m.unfreeze(keys=["scales", "biases"], recurse=False)
            names += [f"{path}.scales", f"{path}.biases"] if path else ["scales", "biases"]
    return names


def _no_index_grad():
    """GLM's sparse-attention indexer returns top-k *indices* computed from trainable projections; the selection
    is discrete, so block gradients there (mlx-vlm applies stop_gradient after the chunk is already kept)."""
    try:
        from mlx_vlm.models.glm5_next import language as glm
    except ImportError:
        return
    cls = glm.Glm5NextIndexer
    if getattr(cls, "_dwq_patched", False):
        return
    orig = cls.__call__
    cls.__call__ = lambda self, *a, **k: mx.stop_gradient(orig(self, *a, **k))
    cls._dwq_patched = True


class ScaledAdam:
    """Adam whose step for each element is lr x |that group's scale|: scales move by a fraction of themselves and
    biases by a fraction of one quantization step, so one lr fits every tensor regardless of magnitude."""

    def __init__(self, params: Dict[str, mx.array], lr: float, steps: int, b1=0.9, b2=0.999, eps=1e-8):
        self.ref = {}
        for k, v in params.items():
            owner = k.rsplit(".", 1)[0] + ".scales" if k.endswith(".biases") else k
            self.ref[k] = mx.abs(params[owner]).astype(mx.float32)
        self.m = {k: mx.zeros(v.shape, mx.float32) for k, v in params.items()}
        self.v = {k: mx.zeros(v.shape, mx.float32) for k, v in params.items()}
        self.lr, self.steps, self.b1, self.b2, self.eps, self.t = lr, max(steps, 1), b1, b2, eps, 0

    def step(self, params: Dict[str, mx.array], grads: Dict[str, mx.array]) -> Dict[str, mx.array]:
        self.t += 1
        lr = self.lr * 0.5 * (1 + np.cos(np.pi * min(self.t - 1, self.steps) / self.steps))
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        out = {}
        for k, p in params.items():
            g = grads[k].astype(mx.float32)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            out[k] = p - lr * self.ref[k] * (self.m[k] / c1) / (mx.sqrt(self.v[k] / c2) + self.eps)
        return out

    def state(self):
        return list(self.m.values()) + list(self.v.values())


def rel_err(y: mx.array, t: mx.array) -> mx.array:
    """Mean over tokens of ||y - t||^2 / ||t||^2 (all hyper-connection streams of a token together)."""
    y, t = y.astype(mx.float32), t.astype(mx.float32)
    axes = tuple(range(2, y.ndim))
    return (((y - t) ** 2).sum(axes) / mx.maximum((t * t).sum(axes), 1e-12)).mean()


def dir_err(y: mx.array, t: mx.array) -> mx.array:
    """1 - cosine per token over all streams, plus the same on the stream mean (what the head reads). Every consumer
    of the residual applies RMSNorm first, so the direction is what matters downstream, not the magnitude."""
    y, t = y.astype(mx.float32), t.astype(mx.float32)

    def cos(a, b, axes):
        return (a * b).sum(axes) / mx.sqrt(mx.maximum((a * a).sum(axes) * (b * b).sum(axes), 1e-24))

    axes = tuple(range(2, y.ndim))
    full = 1 - cos(y, t, axes)
    if y.ndim == 4:
        full = full + (1 - cos(y.mean(2), t.mean(2), (2,)))
    return full.mean()


LOSSES = {"mse": rel_err, "cos": dir_err}


def pick_samples(samples, budget_tokens: int, seq_len: int, rng) -> List[np.ndarray]:
    """Round-robin over domains, each sample truncated to a random window of <= seq_len tokens."""
    by_dom: Dict[str, list] = {}
    for s in samples:
        by_dom.setdefault(s["domain"], []).append(s)
    for v in by_dom.values():
        rng.shuffle(v)
    out, total, doms = [], 0, sorted(by_dom)
    while total < budget_tokens and any(by_dom.values()):
        for d in doms:
            if not by_dom[d] or total >= budget_tokens:
                continue
            toks = by_dom[d].pop()["tokens"]
            if len(toks) > seq_len:
                lo = int(rng.integers(0, len(toks) - seq_len + 1))
                toks = toks[lo : lo + seq_len]
            out.append(np.asarray(toks, dtype=np.int32))
            total += len(toks)
    return out


class BlockDWQ:
    def __init__(self, args):
        self.a = args
        self.teacher = make_arch(args.model, args.src)
        self.student = Student(args.model, args.student)
        rng = np.random.default_rng(args.seed)
        samples = [s for s in load_samples(Path(args.data)) if s["split"] == "calib"]
        rng.shuffle(samples)
        n_val = max(1, len(samples) // 10)
        self.val = pick_samples(samples[:n_val], args.val_tokens, args.seq_len, rng)
        self.train = pick_samples(samples[n_val:], args.tokens, args.seq_len, rng)
        log(f"train {len(self.train)} seqs / {sum(map(len, self.train))} tok, val {len(self.val)} / {sum(map(len, self.val))} tok")
        self.updates: Dict[str, mx.array] = {}
        self.report: List[dict] = []

    def fwd(self, arch, layer, hs):
        outs = []
        for h in hs:
            outs.append(arch.forward(layer, h, None, None))
            mx.eval(outs[-1])
        return outs

    def mean_err(self, layer, xs, ts):
        f = LOSSES[self.a.loss]
        e = [f(self.student.arch.forward(layer, x, None, None), t) for x, t in zip(xs, ts)]
        mx.eval(e)
        return float(np.mean([float(v) for v in e]))

    def train_layer(self, i, layer, s_tr, t_tr, s_val, t_val):
        names = trainable(layer)
        if not names:
            return {"layer": i, "trainable": 0, "accepted": False}
        dtype = {k: v.dtype for k, v in tree_flatten(layer.trainable_parameters())}
        before = self.mean_err(layer, s_val, t_val)
        init = {k: mx.array(v) for k, v in tree_flatten(layer.trainable_parameters())}
        # fp32 master copies: bf16 cannot hold steps of ~1e-3 relative
        params = {k: v.astype(mx.float32) for k, v in init.items()}
        layer.update(_tree(params))
        layer.train()
        opt = ScaledAdam(params, self.a.lr, self.a.epochs * len(s_tr))
        loss_and_grad = nn.value_and_grad(layer, lambda x, t: LOSSES[self.a.loss](self.student.arch.forward(layer, x, None, None), t))
        rng = np.random.default_rng(self.a.seed + i)
        tr_loss = []
        for ep in range(self.a.epochs):
            for j in rng.permutation(len(s_tr)):
                loss, grads = loss_and_grad(s_tr[j].astype(mx.float32), t_tr[j])
                params = opt.step(params, dict(tree_flatten(grads)))
                layer.update(_tree(params))
                mx.eval(list(params.values()), opt.state(), loss)
                tr_loss.append(float(loss))
        layer.eval()
        trained = {k: v.astype(dtype[k]) for k, v in tree_flatten(layer.trainable_parameters())}
        layer.update(_tree(trained))
        after = self.mean_err(layer, s_val, t_val)
        accepted = after < before * (1 - self.a.min_gain)
        if accepted:
            prefix = MLX_LAYER_PREFIX.format(i)
            self.updates.update({prefix + k: v for k, v in trained.items()})
        else:
            layer.update(_tree(init))
        return {"layer": i, "trainable": len(names), "val_before": before, "val_after": after,
                "train_first": float(np.mean(tr_loss[: max(1, len(s_tr))])) if tr_loss else None,
                "train_last": float(np.mean(tr_loss[-max(1, len(s_tr)):])) if tr_loss else None,
                "accepted": bool(accepted)}

    def run(self):
        a = self.a
        _no_index_grad()
        mx.set_memory_limit(int(a.mem_gb * GB))
        seqs = self.train + self.val
        n_tr = len(self.train)
        t_h = [self.teacher.embed(s)[None] for s in seqs]
        s_h = [self.student.embed(s)[None].astype(t_h[0].dtype) for s in seqs]
        mx.eval(t_h, s_h)
        n_layers = self.teacher.n_layers
        stop = min(n_layers, a.layers) if a.layers else n_layers
        for i in range(stop):
            t0 = time.time()
            mx.reset_peak_memory()
            tl = self.teacher.load_layer(i)
            t_h = self.fwd(self.teacher, tl, t_h)
            del tl
            mx.clear_cache()
            sl = self.student.layer(i)
            if i in a.skip:
                rec = {"layer": i, "accepted": False, "skipped": True}
            else:
                rec = self.train_layer(i, sl, s_h[:n_tr], t_h[:n_tr], s_h[n_tr:], t_h[n_tr:])
            s_h = self.fwd(self.student.arch, sl, s_h)
            del sl
            mx.clear_cache()
            rec["seconds"] = round(time.time() - t0, 1)
            rec["peak_gb"] = round(mx.get_peak_memory() / GB, 2)
            self.report.append(rec)
            vb, va = rec.get("val_before"), rec.get("val_after")
            log(f"layer {i:3d}: val rel-err {vb if vb is None else f'{vb:.5f}'} -> {va if va is None else f'{va:.5f}'} "
                f"{'ACCEPT' if rec['accepted'] else 'keep'}  {rec['seconds']}s peak {rec['peak_gb']} GiB")
        self.write()

    def write(self):
        src, out = Path(self.a.student), Path(self.a.out)
        tmp = out.with_name(out.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        shards = set(self.student.weight_map.values())
        shutil.copytree(src, tmp, ignore=lambda d, names: [n for n in names if Path(d) == src and n in shards])
        by_file: Dict[str, List[str]] = {}
        for k in self.updates:
            by_file.setdefault(self.student.weight_map[k], []).append(k)
        for f in sorted(shards):
            if f not in by_file:
                (tmp / f).hardlink_to(src / f) if _same_fs(src, tmp) else shutil.copy2(src / f, tmp / f)
                continue
            arrays = dict(mx.load(str(src / f)))
            for k in by_file[f]:
                if arrays[k].shape != self.updates[k].shape or arrays[k].dtype != self.updates[k].dtype:
                    raise RuntimeError(f"{k}: update {self.updates[k].shape}/{self.updates[k].dtype} != "
                                       f"{arrays[k].shape}/{arrays[k].dtype}")
                arrays[k] = self.updates[k]
            mx.save_safetensors(str(tmp / f), arrays, metadata={"format": "mlx"})
        report = {"student": str(src), "src": self.a.src, "args": {k: v for k, v in vars(self.a).items()},
                  "updated_tensors": len(self.updates), "layers": self.report}
        (tmp / "dwq_report.json").write_text(json.dumps(report, indent=1, default=str))
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        log(f"wrote {out} ({len(self.updates)} tensors updated, "
            f"{sum(r['accepted'] for r in self.report)}/{len(self.report)} layers accepted)")


def _tree(flat: Dict[str, mx.array]):
    from mlx.utils import tree_unflatten

    return tree_unflatten(list(flat.items()))


def _same_fs(a: Path, b: Path) -> bool:
    try:
        return a.resolve().stat().st_dev == b.resolve().stat().st_dev
    except OSError:
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Blockwise local DWQ of a built quantized checkpoint")
    p.add_argument("--model", choices=["glm", "qwen"], required=True)
    p.add_argument("--src", required=True, help="teacher: HF source snapshot")
    p.add_argument("--student", required=True, help="built quantized model dir")
    p.add_argument("--data", required=True, help="calib_data dir (calib.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=65536, help="training tokens")
    p.add_argument("--val-tokens", type=int, default=16384)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3, help="step per update as a fraction of each group's |scale|")
    p.add_argument("--loss", choices=sorted(LOSSES), default="cos")
    p.add_argument("--min-gain", type=float, default=1e-3, help="accept a layer only if val error drops by this fraction")
    p.add_argument("--layers", type=int, help="stop after this many layers (experiments)")
    p.add_argument("--skip", type=lambda s: {int(x) for x in s.split(",") if x}, default=set())
    p.add_argument("--mem-gb", type=float, default=48.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    BlockDWQ(parse_args(argv)).run()


if __name__ == "__main__":
    main()
