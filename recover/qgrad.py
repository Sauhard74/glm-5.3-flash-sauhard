"""Memory-lean gradients of quantized expert projections w.r.t. their scales and biases.

MLX's VJP of ``gather_qmm`` w.r.t. ``scales`` goes through a dense weight-shaped gradient over all experts (at GLM
shapes it tries to allocate ~17 TB). The affine format is linear in (scale, bias), so per expert e and group g:

    dL/db[o, g] = sum_t dy[t, o] * sum_{i in g} x[t, i]
    dL/ds[o, g] = sum_t dy[t, o] * sum_{i in g} q[o, i] * x[t, i]

over the tokens routed to e, one expert at a time (codes unpacked for that expert only). Inside an autodiff
transform MLX keeps every intermediate alive, so that loop must not run inside the VJP: ``ExpertGrads`` makes
the VJP only *record* (x, dy, indices) per module and return zeros for scales/biases; ``fill()`` then computes
the real scale/bias gradients eagerly after the autodiff step, freeing memory expert by expert.
"""

from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _unbroadcast(g: mx.array, shape) -> mx.array:
    """Sum ``g`` down to ``shape`` (inverse of numpy broadcasting)."""
    extra = g.ndim - len(shape)
    if extra:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, (a, b) in enumerate(zip(g.shape, shape)) if b == 1 and a != 1)
    return g.sum(axis=axes, keepdims=True) if axes else g


def scale_bias_grads(x, dy, weight, scales, indices, group_size: int, bits: int) -> Tuple[mx.array, mx.array]:
    """dL/dscales, dL/dbiases of y = gather_qmm(x, weight, scales, biases, indices, transpose=True), eagerly."""
    E, O, G = scales.shape
    I = G * group_size
    bshape = dy.shape[:-2]
    M = x.shape[-2]
    xb = mx.broadcast_to(x, (*bshape, M, I)).reshape(-1, M, I)
    dyb = dy.reshape(-1, M, O)
    ids = np.array(mx.broadcast_to(indices, bshape).reshape(-1))
    ones = mx.ones((O, G), dtype=mx.float32)
    zeros = mx.zeros((O, G), dtype=mx.float32)
    ds = mx.zeros((E, O, G), dtype=mx.float32)
    db = mx.zeros((E, O, G), dtype=mx.float32)
    for e in np.unique(ids):
        rows = mx.array(np.nonzero(ids == e)[0])
        xe = xb[rows].reshape(-1, G, group_size).astype(mx.float32)  # [T, G, gs]
        dye = dyb[rows].reshape(-1, O).astype(mx.float32)  # [T, O]
        q = mx.dequantize(weight[int(e)], ones, zeros, group_size=group_size, bits=bits).reshape(O, G, group_size)
        z = mx.matmul(xe.transpose(1, 0, 2), q.transpose(1, 2, 0))  # [G, T, O]
        ds[int(e)] = (z * dye[None]).sum(1).T
        db[int(e)] = dye.T @ xe.sum(-1)
        mx.eval(ds, db)
    return ds, db


def scale_bias_grads_batched(x, dy, weight, scales, indices, group_size: int, bits: int, chunk: int = 16):
    """Same result as ``scale_bias_grads``, batched: per-expert dW = dy_e^T x_e via one segmented_mm over
    expert-sorted rows, contracted with the unpacked codes ``chunk`` experts at a time."""
    E, O, G = scales.shape
    I = G * group_size
    bshape = dy.shape[:-2]
    M = x.shape[-2]
    xb = mx.broadcast_to(x, (*bshape, M, I)).reshape(-1, M, I)
    dyb = dy.reshape(-1, M, O)
    ids = np.array(mx.broadcast_to(indices, bshape).reshape(-1))
    order = np.argsort(ids, kind="stable")
    ids_s = ids[order]
    rows = mx.array(order)
    xs = xb[rows].reshape(-1, I).astype(mx.float32)  # [P*M, I], experts contiguous
    dys = dyb[rows].reshape(-1, O).astype(mx.float32)
    counts = np.bincount(ids_s, minlength=E) * M
    ends = np.cumsum(counts)
    starts = ends - counts
    seg_all = mx.array(np.stack([starts, ends], 1).astype(np.uint32))
    dyT = dys.T
    db = mx.segmented_mm(dyT, xs.reshape(-1, G, group_size).sum(-1), seg_all)  # [E, O, G]
    ones = mx.ones((O, G), dtype=mx.float32)
    zeros = mx.zeros((O, G), dtype=mx.float32)
    parts = []
    for lo in range(0, E, chunk):
        hi = min(E, lo + chunk)
        dw = mx.segmented_mm(dyT, xs, seg_all[lo:hi])  # [c, O, I]
        q = mx.stack([mx.dequantize(weight[e], ones, zeros, group_size=group_size, bits=bits) for e in range(lo, hi)])
        parts.append((dw * q).reshape(hi - lo, O, G, group_size).sum(-1))
        mx.eval(parts[-1])
    ds = mx.concatenate(parts)
    mx.eval(ds, db)
    return ds, db


class ExpertGrads:
    """Patches mlx-vlm's QuantizedSwitchLinear for training. Forward is the unchanged gather_qmm."""

    def __init__(self):
        self.stash: Dict[int, tuple] = {}
        self._fns: Dict[int, object] = {}
        self._install()

    def _fn(self, module):
        key = id(module)
        if key in self._fns:
            return self._fns[key]
        gs, bits, mode = module.group_size, module.bits, module.mode
        stash = self.stash

        @mx.custom_function
        def f(x, scales, biases, weight, indices):
            return mx.gather_qmm(x, weight, scales, biases, rhs_indices=indices, transpose=True,
                                 group_size=gs, bits=bits, mode=mode)

        @f.vjp
        def f_vjp(primals, cotangent, output):
            x, scales, biases, weight, indices = primals
            dx = mx.gather_qmm(cotangent.astype(x.dtype), weight, scales, biases, rhs_indices=indices,
                               transpose=False, group_size=gs, bits=bits, mode=mode)
            stash[key] = (module, x, cotangent, indices)
            return (_unbroadcast(dx, x.shape).astype(x.dtype), mx.zeros_like(scales), mx.zeros_like(biases),
                    mx.zeros_like(weight), mx.zeros_like(indices))

        self._fns[key] = f
        return f

    def _install(self):
        from mlx_vlm.models import switch_layers as sl

        cls = sl.QuantizedSwitchLinear
        if not hasattr(cls, "_orig_call"):
            cls._orig_call = cls.__call__
        owner = self

        def call(m, x, indices, sorted_indices=False):
            if m.mode != "affine" or m.get("biases") is None or not getattr(m, "_dwq_lean", False):
                return cls._orig_call(m, x, indices, sorted_indices)
            y = owner._fn(m)(x, m["scales"], m["biases"], m["weight"], indices)
            if "bias" in m:
                y = y + mx.expand_dims(m["bias"][indices], -2)
            return y

        cls.__call__ = call

    @staticmethod
    def enable(layer: nn.Module) -> Dict[int, str]:
        """Mark a layer's switch linears for the lean path; returns {id(module): dotted path}."""
        from mlx_vlm.models import switch_layers as sl

        paths = {}
        for path, m in layer.named_modules():
            if isinstance(m, sl.QuantizedSwitchLinear):
                object.__setattr__(m, "_dwq_lean", True)
                paths[id(m)] = path
        return paths

    def fill(self, grads: Dict[str, mx.array], paths: Dict[int, str]) -> Dict[str, mx.array]:
        """Replace the zero scale/bias grads of stashed modules with the real ones (call after mx.eval(grads))."""
        for key, (m, x, dy, idx) in list(self.stash.items()):
            p = paths.get(key)
            if p is None:
                continue
            ds, db = scale_bias_grads_batched(x, dy, m["weight"], m["scales"], idx, m.group_size, m.bits)
            if f"{p}.scales" in grads:
                grads[f"{p}.scales"] = ds.astype(grads[f"{p}.scales"].dtype)
            if f"{p}.biases" in grads:
                grads[f"{p}.biases"] = db.astype(grads[f"{p}.biases"].dtype)
        self.stash.clear()
        return grads
