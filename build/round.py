"""Rounding inside MLX's affine format (docs/plan/05 §2.6a): RTN, or a least-squares scale/bias refit.

Output is exactly what ``mx.quantize(..., mode="affine")`` returns -- packed uint32 codes, scales, biases --
so loaders and kernels are unaffected. LSQ starts from the best of MLX's RTN and a few shrunk ranges, then
alternates re-rounding the codes with a per-group (optionally imatrix-weighted) least-squares fit of
(scale, bias). Every candidate is scored with (scale, bias) rounded to the stored dtype and the best per group
is kept, so it never does worse than RTN on the weighted error.
"""

from typing import Optional, Tuple

import mlx.core as mx

ROW_CHUNK_ELEMS = 1 << 24  # fp32 working set per chunk (64 MB per temporary)
SHRINKS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)  # range-search candidates (fraction of the group's [min, max])


def pack(codes: mx.array, bits: int) -> mx.array:
    """Pack uint32 codes [..., n] into MLX's affine layout: a little-endian bitstream, value j at bit j*bits."""
    n = codes.shape[-1]
    if (n * bits) % 32:
        raise ValueError(f"{n} values x {bits} bits is not a whole number of uint32 words")
    lead = codes.shape[:-1]
    c = codes.astype(mx.uint32).reshape(*lead, n // 32, 32)
    words = [mx.zeros((*lead, n // 32), dtype=mx.uint32) for _ in range(bits)]
    for j in range(32):
        off = j * bits
        w, s = divmod(off, 32)
        v = c[..., j]
        words[w] = words[w] | (v << s)
        if s + bits > 32:
            words[w + 1] = words[w + 1] | (v >> (32 - s))
    return mx.stack(words, axis=-1).reshape(*lead, n * bits // 32)


def _fit(w, h, bits, group, iters, dtype):
    """w: [R, n] fp32, h: [R or 1, n] fp32 weights. Returns codes [R, n], scales/biases [R, G] (dtype),
    weighted error of RTN and of the result."""
    R, n = w.shape
    G = n // group
    M = float((1 << bits) - 1)
    wg = w.reshape(R, G, group)
    hg = mx.broadcast_to(h.reshape(h.shape[0], G, group), wg.shape)

    def codes_for(s, b):
        return mx.clip(mx.round((wg - b[..., None]) / s[..., None]), 0, M)

    def err(q, s, b):
        d = q * s[..., None] + b[..., None] - wg
        return (hg * d * d).sum(-1)

    _, s0, b0 = mx.quantize(w.astype(dtype), group_size=group, bits=bits, mode="affine")
    s, b = s0.astype(mx.float32), b0.astype(mx.float32)
    best_e = e_rtn = err(codes_for(s, b), s, b)
    best_s, best_b = s, b
    # Range search: clipping a group's rare extremes usually beats covering them (MSE-optimal clipping).
    lo, hi = wg.min(-1), wg.max(-1)
    mid = (lo + hi) / 2
    for f in SHRINKS:
        sc = mx.maximum(((hi - lo) * f / M).astype(dtype).astype(mx.float32), 1e-12)
        bc = (mid - (mid - lo) * f).astype(dtype).astype(mx.float32)
        e = err(codes_for(sc, bc), sc, bc)
        better = e < best_e
        best_s, best_b = mx.where(better, sc, best_s), mx.where(better, bc, best_b)
        best_e = mx.minimum(e, best_e)
    s, b = best_s, best_b
    q = codes_for(s, b)
    hs = mx.maximum(hg.sum(-1), 1e-30)
    for _ in range(iters):
        qbar = (hg * q).sum(-1) / hs
        wbar = (hg * wg).sum(-1) / hs
        dq = q - qbar[..., None]
        var = (hg * dq * dq).sum(-1)
        cov = (hg * dq * (wg - wbar[..., None])).sum(-1)
        ok = (var > 0) & (cov > 0)
        s_new = mx.where(ok, cov / mx.where(ok, var, 1.0), s)
        b_new = mx.where(ok, wbar - s_new * qbar, b)
        s = s_new.astype(dtype).astype(mx.float32)  # evaluate what will be stored
        b = b_new.astype(dtype).astype(mx.float32)
        s = mx.where(s > 0, s, best_s)
        q = codes_for(s, b)
        e = err(q, s, b)
        better = e < best_e
        best_s = mx.where(better, s, best_s)
        best_b = mx.where(better, b, best_b)
        best_e = mx.minimum(e, best_e)
        mx.eval(best_s, best_b, best_e, q)
    q = codes_for(best_s, best_b)
    return q.reshape(R, n), best_s.astype(dtype), best_b.astype(dtype), e_rtn.sum(), best_e.sum()


def quantize(
    w: mx.array,
    bits: int,
    group: int,
    rounding: str = "rtn",
    h: Optional[mx.array] = None,
    iters: int = 4,
    stats: Optional[dict] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Drop-in for ``mx.quantize(w, group, bits, mode="affine")``.

    ``h``: per-input-channel weights (calib ``act_sqmean``), shape [n] (shared) or [w.shape[0], n] (per expert of a
    3-D stacked tensor); None = unweighted. ``stats`` (optional dict) accumulates weighted error before/after.
    """
    if rounding == "rtn":
        return mx.quantize(w, group_size=group, bits=bits, mode="affine")
    if rounding != "lsq":
        raise ValueError(f"unknown rounding {rounding!r}")
    dtype = w.dtype if mx.issubdtype(w.dtype, mx.floating) and w.dtype != mx.float32 else mx.bfloat16
    n = w.shape[-1]
    if h is not None:
        h = h.astype(mx.float32)
        if h.shape[-1] != n or (h.ndim == 2 and (w.ndim != 3 or h.shape[0] != w.shape[0])) or h.ndim > 2:
            h = None
        else:
            h = h + 0.01 * mx.mean(h, axis=-1, keepdims=True)  # keep near-silent channels from going unfit
    # Split into 2-D slices: per leading index for stacked tensors with per-slice h, else row chunks.
    if w.ndim == 3 and h is not None and h.ndim == 2:
        slices = [(w[e], h[e][None]) for e in range(w.shape[0])]
        lead_split = True
    else:
        flat = w.reshape(-1, n)
        hh = h[None] if h is not None else mx.ones((1, n), dtype=mx.float32)
        step = max(1, ROW_CHUNK_ELEMS // n)
        slices = [(flat[r : r + step], hh) for r in range(0, flat.shape[0], step)]
        lead_split = False
    qs, ss, bs = [], [], []
    e0 = e1 = 0.0
    for ws, hs in slices:
        q, s, b, er, el = _fit(ws.astype(mx.float32), hs, bits, group, iters, dtype)
        p = pack(q.astype(mx.uint32), bits)
        mx.eval(p, s, b, er, el)
        qs.append(p), ss.append(s), bs.append(b)
        e0 += float(er)
        e1 += float(el)
    if stats is not None:
        stats["err_rtn"] = stats.get("err_rtn", 0.0) + e0
        stats["err_lsq"] = stats.get("err_lsq", 0.0) + e1
    if lead_split:
        return mx.stack(qs), mx.stack(ss), mx.stack(bs)
    lead = w.shape[:-1]
    return (
        mx.concatenate(qs).reshape(*lead, -1),
        mx.concatenate(ss).reshape(*lead, -1),
        mx.concatenate(bs).reshape(*lead, -1),
    )
