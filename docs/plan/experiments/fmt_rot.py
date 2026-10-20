"""Tiny experiment for docs/plan/05-quantization.md (< 1.5 GB peak).

Real Qwen3.8-Flash-Next weights from the bf16 MTP drafter (sh0wie/...-MTP-Drafter-MLX-bf16):
relative weight MSE ||W - Q(W)||^2 / ||W||^2 per format, with/without a signed Hadamard rotation of
the input axis, and with an MSE-optimal (least-squares) scale/bias refit for affine.
Usage: python -I fmt_rot.py <path to model.safetensors>
"""
import json, struct, sys, time, math
import numpy as np
import mlx.core as mx

path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base = 8 + n
mm = np.memmap(path, dtype=np.uint8, mode="r")


def read(name, rows=None):
    m = hdr[name]
    a, b = m["data_offsets"]
    raw = mm[base + a: base + b].view(np.uint16).reshape(m["shape"])
    if rows is not None:
        raw = raw[rows]
    raw = np.ascontiguousarray(raw)
    return mx.array(raw).view(mx.bfloat16).astype(mx.float32)


rng = np.random.default_rng(0)
EXP = sorted(rng.choice(512, 24, replace=False).tolist())
P = "layers.0."
tensors = {
    "expert.gate_proj [24x640x2560]": read(P + "mlp.switch_mlp.gate_proj.weight", EXP).reshape(-1, 2560),
    "expert.down_proj [24x2560x640]": read(P + "mlp.switch_mlp.down_proj.weight", EXP).reshape(-1, 640),
    "shared_expert.gate_proj": read(P + "mlp.shared_expert.gate_proj.weight"),
    "attn.q_proj": read(P + "self_attn.q_proj.weight"),
    "attn.o_proj": read(P + "self_attn.o_proj.weight"),
    "hc.input_mix_weight_down": read(P + "attn_hyper_connection.input_mix_weight_down.weight"),
    "mlp.gate (router)": read(P + "mlp.gate.weight"),
    "fc_hidden": read("fc_hidden.weight"),
}


def rel(w, d):
    return float(mx.square(w - d).sum() / mx.square(w).sum())


def qdq(w, mode, bits, g):
    if mode == "affine":
        q = mx.quantize(w.astype(mx.bfloat16), group_size=g, bits=bits)
        return mx.dequantize(*q, group_size=g, bits=bits).astype(mx.float32)
    q = mx.quantize(w.astype(mx.bfloat16), mode=mode)
    return mx.dequantize(q[0], q[1], mode=mode).astype(mx.float32)


def lsq_refit(w, bits, g, iters=4):
    """Affine RTN, then alternate: codes <- round((w-b)/s); (s,b) <- least squares per group.
    Scales/biases are rounded to bf16 before computing codes/errors, as MLX stores them."""
    rows, cols = w.shape
    W = w.reshape(rows, cols // g, g)
    lo, hi = W.min(-1, keepdims=True), W.max(-1, keepdims=True)
    L = 2 ** bits - 1
    s = mx.maximum((hi - lo) / L, 1e-12)
    b = lo
    best = None
    for _ in range(iters + 1):
        s16, b16 = s.astype(mx.bfloat16).astype(mx.float32), b.astype(mx.bfloat16).astype(mx.float32)
        q = mx.clip(mx.round((W - b16) / s16), 0, L)
        D = q * s16 + b16
        err = float(mx.square(W - D).sum())
        if best is None or err < best[0]:
            best = (err, D)
        # LSQ: minimise sum (W - s q - b)^2 over s, b per group
        qm, wm = q.mean(-1, keepdims=True), W.mean(-1, keepdims=True)
        cov = ((q - qm) * (W - wm)).sum(-1, keepdims=True)
        var = mx.square(q - qm).sum(-1, keepdims=True)
        s = mx.where(var > 0, cov / mx.maximum(var, 1e-12), s)
        s = mx.maximum(s, 1e-12)
        b = wm - s * qm
    return best[1].reshape(rows, cols)


def rotate(w, block, signs):
    rows, cols = w.shape
    x = (w * signs).reshape(-1, block)
    return mx.hadamard_transform(x, scale=block ** -0.5).reshape(rows, cols)


def kurt(w):
    x = np.array(w).ravel()
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


FMTS = [("affine", 2, 64), ("affine", 2, 128), ("affine", 3, 64), ("affine", 3, 128), ("affine", 3, 32),
        ("affine", 4, 64), ("affine", 4, 128), ("affine", 4, 32), ("mxfp4", 4, 32), ("nvfp4", 4, 16),
        ("mxfp8", 8, 32), ("affine", 8, 64)]
res = {}
for name, w in tensors.items():
    cols = w.shape[1]
    signs = mx.array(np.where(rng.random(cols) < 0.5, -1.0, 1.0).astype(np.float32))
    wr = rotate(w, cols, signs)            # full-width signed Hadamard
    wb = rotate(w, 128, signs)             # block-128 (aligned to groups)
    row = {"kurtosis": kurt(w), "kurtosis_rot": kurt(wr)}
    for mode, bits, g in FMTS:
        k = f"{mode}{bits}g{g}"
        row[k] = rel(w, qdq(w, mode, bits, g))
        row[k + "+H"] = rel(wr, qdq(wr, mode, bits, g))
        row[k + "+H128"] = rel(wb, qdq(wb, mode, bits, g))
        if mode == "affine" and bits <= 4:
            row[k + "+lsq"] = rel(w, lsq_refit(w, bits, g))
            row[k + "+H+lsq"] = rel(wr, lsq_refit(wr, bits, g))
    res[name] = row
    print(name, json.dumps({k: (round(v, 5) if v < 1 else round(v, 2)) for k, v in row.items()}), flush=True)

# Online Hadamard cost at decode shapes (what prism_hadamard does on every call)
def bench(shape, reps=200):
    x = mx.random.normal(shape).astype(mx.bfloat16)
    s = mx.ones((shape[-1],))
    def f(x):
        y = (x.astype(mx.float32) * s).reshape(-1, shape[-1])
        return mx.hadamard_transform(y, scale=shape[-1] ** -0.5).astype(mx.bfloat16)
    for _ in range(10):
        mx.eval(f(x))
    t = time.perf_counter()
    ys = [f(x) for _ in range(reps)]
    mx.eval(ys)
    return (time.perf_counter() - t) / reps * 1e6

for shp in [(1, 2560), (10, 640), (1, 4096), (8, 2048), (2048, 2560)]:
    print("hadamard", shp, f"{bench(shp):.1f} us/call (batched eval, includes launch)")
print("peak GB", mx.get_peak_memory() / 1e9)
json.dump(res, open(sys.argv[0].replace(".py", ".json"), "w"), indent=1)
