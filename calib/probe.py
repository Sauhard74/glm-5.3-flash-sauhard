"""Activation probes on a live mlx-vlm decoder layer.

Probes are installed by swapping a module's class for a subclass whose __call__
stashes its input; the stash is processed after each chunk's forward is evaluated
(`flush`). Nothing about the forward computation changes."""

import mlx.core as mx
import numpy as np

from mlx_vlm.models.mla import MultiLinear
from mlx_vlm.models.switch_layers import SwitchGLU

_CLASSES = {}


def _probe_class(cls, kind):
    key = (cls, kind)
    if key in _CLASSES:
        return _CLASSES[key]
    if kind == "switch":

        def __call__(self, x, indices, weights=None, shared=None, residual=None):
            rec = self.__dict__.get("_calib_rec")
            if rec is None or not rec.active or rec._in_switch:
                return cls.__call__(self, x, indices, weights, shared, residual)
            rec._in_switch = True
            try:
                routed = cls.__call__(self, x, indices)
            finally:
                rec._in_switch = False
            rec.stash.append(("moe", None, (x, indices, weights, routed)))
            return SwitchGLU._combine(routed, weights, shared, residual)

    else:

        def __call__(self, x, *args, **kwargs):
            y = cls.__call__(self, x, *args, **kwargs)
            rec = self.__dict__.get("_calib_rec")
            if rec is not None and rec.active and not rec._in_switch:
                mode = "t"
                if kind == "multi" and not kwargs.get("transpose", args[0] if args else True):
                    mode = "n"
                rec.stash.append(("lin", self.__dict__["_calib_name"], (x, mode)))
            return y

    new = type(f"Probed{cls.__name__}", (cls,), {"__call__": __call__})
    _CLASSES[key] = new
    return new


def _install(module, kind, rec, name=None):
    module.__class__ = _probe_class(type(module), kind)
    object.__setattr__(module, "_calib_rec", rec)
    object.__setattr__(module, "_calib_name", name)


def _uninstall(module):
    module.__class__ = type(module).__mro__[1]
    module.__dict__.pop("_calib_rec", None)
    module.__dict__.pop("_calib_name", None)


class Reservoir:
    """Uniform per-group sample of a stream (priority sampling): every offered item
    draws a random key and each group keeps the `cap` smallest keys."""

    def __init__(self, n_groups, cap, rng):
        self.cap = cap
        self.rng = rng
        self.keys = [np.empty(0)] * n_groups
        self.rows = [None] * n_groups

    def offer(self, groups: np.ndarray, gather):
        if self.cap <= 0 or len(groups) == 0:
            return
        u = self.rng.random(len(groups))
        thr = np.array([k.max() if len(k) >= self.cap else np.inf for k in self.keys])
        cand = np.nonzero(u < thr[groups])[0]
        if len(cand) == 0:
            return
        order = cand[np.lexsort((u[cand], groups[cand]))]
        g = groups[order]
        first = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
        run_start = np.repeat(first, np.diff(np.r_[first, len(g)]))
        keep = order[(np.arange(len(g)) - run_start) < self.cap]
        rows = gather(keep)
        gk, uk = groups[keep], u[keep]
        for grp in np.unique(gk):
            sel = gk == grp
            k = np.concatenate([self.keys[grp], uk[sel]])
            r = rows[sel] if self.rows[grp] is None else np.concatenate([self.rows[grp], rows[sel]])
            if len(k) > self.cap:
                top = np.argpartition(k, self.cap - 1)[: self.cap]
                k, r = k[top], r[top]
            self.keys[grp], self.rows[grp] = k, r

    def count(self, grp=0):
        return len(self.keys[grp])


class _LinStats:
    def __init__(self, module, n_domains):
        self.module = module
        self.multi = isinstance(module, MultiLinear)
        self.in_dim = module.weight.shape[-1]
        self.abs_sum = [None] * n_domains
        self.sq_sum = [None] * n_domains
        self.count = np.zeros(n_domains)
        self.res = {}


class LayerRecorder:
    def __init__(self, arch, layer, targets, mlp, domains, lin_tokens=2048,
                 expert_tokens=128, seed=0):
        self.arch = arch
        self.layer = layer
        self.mlp = mlp
        self.domains = list(domains)
        self.nd = len(self.domains)
        self.active = False
        self.domain = 0
        self.stash = []
        self._in_switch = False
        self.rng = np.random.default_rng(seed)
        self.lin_cap = max(1, lin_tokens // self.nd)
        self.tokens = np.zeros(self.nd)
        self.lin = {}
        self.installed = []
        for name, m in targets.items():
            self.lin[name] = _LinStats(m, self.nd)
            _install(m, "multi" if isinstance(m, MultiLinear) else "linear", self, name)
            self.installed.append(m)
        if mlp is not None:
            sw = mlp.switch_mlp
            self.E = sw.gate_proj.weight.shape[0]
            D = sw.gate_proj.weight.shape[-1]
            z = lambda: np.zeros((self.nd, self.E))
            self.count, self.gnorm, self.nsum, self.wsum, self.psum = z(), z(), z(), z(), z()
            self.x_abs = [mx.zeros((self.E, D)) for _ in range(self.nd)]
            self.x_sq = [mx.zeros((self.E, D)) for _ in range(self.nd)]
            self.expert_cap = max(1, expert_tokens // self.nd)
            self.expert_res = Reservoir(self.nd * self.E, self.expert_cap, self.rng)
            _install(sw, "switch", self)
            self.installed.append(sw)
        self.ple = None
        if "ple" in layer:
            self.ple = layer.ple.ple_embedding.ngram_embedding
            self.ple.on_rows = self._ple_rows
            self.ple_res = Reservoir(self.nd, self.lin_cap, self.rng)

    def remove(self):
        for m in self.installed:
            _uninstall(m)
        if self.ple is not None:
            self.ple.on_rows = None

    def begin(self, domain: str, active: bool):
        self.domain = self.domains.index(domain) if active else 0
        self.active = active
        self.stash = []

    def _ple_rows(self, flat):
        if self.active:
            self.ple_res.offer(np.full(len(flat), self.domain, dtype=np.int64), lambda keep: flat[keep][:, None])

    def flush(self, out, n_tokens):
        """Evaluate `out` with everything the probes stashed, then fold into stats."""
        if not self.active:
            mx.eval(out)
            self.stash = []
            return
        arrays = [a for _, _, args in self.stash for a in args if isinstance(a, mx.array)]
        mx.eval(out, *arrays)
        self.tokens[self.domain] += n_tokens
        stash, self.stash = self.stash, []
        for kind, name, args in stash:
            if kind == "lin":
                self._linear(name, *args, n_tokens)
            else:
                self._moe(*args)
        if self.mlp is not None:
            mx.eval(self.x_abs[self.domain], self.x_sq[self.domain])

    def _linear(self, name, x, mode, n_tokens):
        st = self.lin[name]
        d = self.domain
        if st.multi:
            # MLA may re-project the whole cached latent; keep only this chunk's rows
            x = x[:, :, -n_tokens:]
            rows = x.swapaxes(1, 2).reshape(-1, x.shape[1], x.shape[-1])
        else:
            rows = x.reshape(-1, x.shape[-1])
        axes = tuple(range(1, rows.ndim))
        valid = np.array(mx.any(rows != 0, axis=axes))
        idx = np.nonzero(valid)[0]
        if len(idx) == 0:
            return
        if mode == "t":
            xf = rows.astype(mx.float32)
            red = tuple(range(rows.ndim - 1))
            a, s = mx.abs(xf).sum(axis=red), mx.square(xf).sum(axis=red)
            a, s = np.array(a, dtype=np.float64), np.array(s, dtype=np.float64)
            st.abs_sum[d] = a if st.abs_sum[d] is None else st.abs_sum[d] + a
            st.sq_sum[d] = s if st.sq_sum[d] is None else st.sq_sum[d] + s
            st.count[d] += len(idx) * (rows.shape[1] if st.multi else 1)
        key = (mode, rows.shape[1] if st.multi else 0)
        if key not in st.res:
            st.res[key] = [Reservoir(1, self.lin_cap, self.rng) for _ in range(self.nd)]
        sel = mx.array(idx)
        st.res[key][d].offer(
            np.zeros(len(idx), dtype=np.int64),
            lambda keep: np.array(rows[sel[mx.array(keep)]].astype(mx.float32)),
        )

    def _moe(self, x, inds, weights, routed):
        d = self.domain
        D, K = x.shape[-1], inds.shape[-1]
        xf = x.reshape(-1, D)
        indsf = inds.reshape(-1, K)
        probs = self.arch.router_probs(self.mlp, xf)
        if weights is None:
            weights = self.arch.moe_weights(self.mlp, probs, indsf)
        wf = weights.reshape(-1, K).astype(mx.float32)
        norms = mx.sqrt(mx.square(routed.reshape(-1, K, D).astype(mx.float32)).sum(-1))
        flat_e = indsf.reshape(-1)
        xa = xf.astype(mx.float32)
        self.x_abs[d] = self.x_abs[d].at[flat_e].add(mx.repeat(mx.abs(xa), K, axis=0))
        self.x_sq[d] = self.x_sq[d].at[flat_e].add(mx.repeat(mx.square(xa), K, axis=0))
        e = np.array(flat_e).astype(np.int64)
        w = np.array(wf).reshape(-1).astype(np.float64)
        n = np.array(norms).reshape(-1).astype(np.float64)
        E = self.E
        self.count[d] += np.bincount(e, minlength=E)
        self.gnorm[d] += np.bincount(e, weights=w * n, minlength=E)
        self.nsum[d] += np.bincount(e, weights=n, minlength=E)
        self.wsum[d] += np.bincount(e, weights=w, minlength=E)
        self.psum[d] += np.array(probs.sum(axis=0), dtype=np.float64)
        self.expert_res.offer(
            d * E + e, lambda keep: np.array(xf[mx.array(keep // K)].astype(mx.float16))
        )

    def expert_samples(self):
        """Per-expert sampled rows (domains concatenated) and row weights that make the
        sample domain-balanced: domain d's rows carry (count_d[e] / T_d) / n_{d,e},
        normalized to mean 1 per expert."""
        T = np.maximum(self.tokens, 1.0)
        rows, weights = [], []
        for e in range(self.E):
            parts = [(d, self.expert_res.rows[d * self.E + e]) for d in range(self.nd)]
            parts = [(d, r) for d, r in parts if r is not None]
            if not parts:
                rows.append(None)
                weights.append(None)
                continue
            w = np.concatenate([np.full(len(r), self.count[d, e] / T[d] / len(r)) for d, r in parts])
            rows.append(np.concatenate([r for _, r in parts]))
            weights.append(w * len(w) / w.sum())
        return rows, weights

    def ple_ids(self):
        rows = [r for r in self.ple_res.rows if r is not None]
        return np.sort(np.concatenate(rows)[:, 0]) if rows else None


def balanced_mean(sums, counts):
    """Mean over domains of per-domain means (domains with no data are skipped).
    sums: list[nd] of arrays (or None); counts: [nd] (or [nd, ...] broadcastable)."""
    acc, n = 0.0, 0.0
    for s, c in zip(sums, counts):
        if s is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        ok = c > 0
        cc = np.where(ok, c, 1.0)
        if cc.ndim < np.ndim(s):
            cc = cc[..., None]
            ok = ok[..., None]
        acc = acc + np.where(ok, s / cc, 0.0)
        n = n + ok
    if isinstance(n, float):
        return None
    return np.where(n > 0, acc / np.maximum(n, 1), 0.0)
