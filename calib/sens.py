"""Quantization sensitivity: NMSE = sum ||X (Wq - W)^T||^2 / sum ||X W^T||^2 on real
captured inputs, with Wq = dequantize(quantize(W)) (MLX affine) over a bits x group grid.
The error is formed from the weight delta directly: fp32 GEMMs on Apple GPUs are not
full precision, and differencing two large outputs would put a floor under 8-bit NMSE."""

import math

import mlx.core as mx
import numpy as np

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64)


def grid_key(bits, group):
    return f"{bits}_{group}"


def qdq(w: mx.array, bits: int, group: int) -> mx.array:
    q = mx.quantize(w, group_size=group, bits=bits)
    return mx.dequantize(*q, group_size=group, bits=bits).astype(mx.float32)


def _valid(in_dim, group):
    return in_dim % group == 0


def _grid(in_dim, fn):
    out = np.full((len(BITS), len(GROUPS)), np.nan)
    for gi, g in enumerate(GROUPS):
        if not _valid(in_dim, g):
            continue
        for bi, b in enumerate(BITS):
            out[bi, gi] = fn(b, g)
    return out


def _ratio(num, den):
    num, den = float(num), float(den)
    return num / den if den > 0 else math.nan


def linear_sens(w: mx.array, x: np.ndarray, block=1024):
    """w [out, in]; x [n, in]. Returns (grid [nb, ng], mean ||y||^2 per row)."""
    X = mx.array(x).astype(mx.float32)
    wf = w.astype(mx.float32)
    den = 0.0
    for i in range(0, X.shape[0], block):
        den += float(mx.square(X[i : i + block] @ wf.T).sum())

    def err(b, g):
        dw = qdq(w, b, g) - wf
        num = 0.0
        for i in range(0, X.shape[0], block):
            num += float(mx.square(X[i : i + block] @ dw.T).sum())
        return _ratio(num, den)

    return _grid(w.shape[-1], err), den / max(X.shape[0], 1)


def multilinear_sens(w: mx.array, groups: dict):
    """w [H, out, in]; groups {(mode, h): x [n, h, dim]}; mode "t": x @ w^T, "n": x @ w."""
    wf = w.astype(mx.float32)

    def apply(wm, mode, X):
        return X @ (wm.swapaxes(-1, -2) if mode == "t" else wm)

    inputs = [(mode, mx.array(x).astype(mx.float32).swapaxes(0, 1)) for (mode, _), x in groups.items()]
    den = sum(float(mx.square(apply(wf, mode, X)).sum()) for mode, X in inputs)
    n = sum(X.shape[1] for _, X in inputs)

    def err(b, g):
        dw = qdq(w, b, g) - wf
        num = sum(mx.square(apply(dw, mode, X)).sum() for mode, X in inputs)
        return _ratio(num, den)

    return _grid(w.shape[-1], err), float(den) / max(n, 1)


def rows_sens(rows: mx.array):
    """Weight-only NMSE for embedding rows [n, dim] (each row's lookup is its output)."""
    rf = rows.astype(mx.float32)
    den = mx.square(rf).sum()
    return _grid(rows.shape[-1], lambda b, g: _ratio(mx.square(qdq(rows, b, g) - rf).sum(), den))


def expert_sens(sw, samples: list, activation, weights=None, block=16):
    """Per-expert NMSE for a stacked SwitchGLU. samples[e]: x rows [n_e, D] routed to e;
    weights[e]: optional per-row weights [n_e] (every sum and mean below is weighted).

    Returns {proj: grid [E, nb, ng]}, {proj: mean ||y||^2 per row [E]} and down_proj
    input magnitudes (|h| mean [E, I], h^2 mean [E, I]) from the same samples."""
    W = {p: getattr(sw, p).weight for p in ("gate_proj", "up_proj", "down_proj")}
    E = W["gate_proj"].shape[0]
    cap = max([len(s) for s in samples if s is not None] + [1])
    D = W["gate_proj"].shape[-1]
    I = W["down_proj"].shape[-1]
    grids = {p: np.full((E, len(BITS), len(GROUPS)), np.nan) for p in W}
    energy = {p: np.zeros(E) for p in W}
    h_abs, h_sq = np.zeros((E, I)), np.zeros((E, I))
    counts = np.array([0 if s is None else len(s) for s in samples])

    for s in range(0, E, block):
        e_ids = [e for e in range(s, min(s + block, E)) if counts[e] > 0]
        if not e_ids:
            continue
        xp = np.zeros((len(e_ids), cap, D), dtype=np.float32)
        wp = np.zeros((len(e_ids), cap), dtype=np.float32)
        for j, e in enumerate(e_ids):
            xp[j, : counts[e]] = samples[e]
            wp[j, : counts[e]] = 1.0 if weights is None or weights[e] is None else weights[e]
        X = mx.array(xp)
        Wr = mx.array(wp)[..., None]
        sw_rows = mx.sqrt(Wr)
        ids = mx.array(e_ids)
        wg = W["gate_proj"][ids]
        wu = W["up_proj"][ids]
        wd = W["down_proj"][ids]
        f = lambda w: w.astype(mx.float32).swapaxes(-1, -2)
        H = activation(X @ f(wu), X @ f(wg)).astype(mx.float32)
        n = Wr.sum(axis=1)
        h_abs[e_ids] = np.array((mx.abs(H) * Wr).sum(axis=1) / n)
        h_sq[e_ids] = np.array((mx.square(H) * Wr).sum(axis=1) / n)
        n = np.array(n)[:, 0]
        for p, w, inp in (("gate_proj", wg, X * sw_rows), ("up_proj", wu, X * sw_rows),
                          ("down_proj", wd, H * sw_rows)):
            y = inp @ f(w)
            den = mx.square(y).sum(axis=(1, 2))
            mx.eval(den)
            energy[p][e_ids] = np.array(den) / n
            for gi, g in enumerate(GROUPS):
                if not _valid(w.shape[-1], g):
                    continue
                for bi, b in enumerate(BITS):
                    dw = (qdq(w, b, g) - w.astype(mx.float32)).swapaxes(-1, -2)
                    num = mx.square(inp @ dw).sum(axis=(1, 2))
                    r = np.array(num) / np.maximum(np.array(den), 1e-30)
                    grids[p][e_ids, bi, gi] = r
    return grids, energy, h_abs, h_sq


def grid_dict(grid: np.ndarray) -> dict:
    return {
        grid_key(b, g): (None if np.isnan(grid[bi, gi]) else float(grid[bi, gi]))
        for bi, b in enumerate(BITS)
        for gi, g in enumerate(GROUPS)
    }
