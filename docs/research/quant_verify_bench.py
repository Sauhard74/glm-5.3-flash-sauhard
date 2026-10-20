import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.switch_layers import SwitchGLU

mx.set_memory_limit(6 * 1024**3)

SHAPES = {
    "qwen": dict(H=2560, I=640, E=64, k=10, L=4),
    "glm": dict(H=4096, I=2048, E=24, k=8, L=3),
}
CONFIGS = [
    ("affine", 2, 64),
    ("affine", 3, 64),
    ("affine", 3, 128),
    ("affine", 4, 64),
    ("affine", 4, 128),
    ("affine", 5, 64),
    ("mxfp4", 4, 32),
    ("nvfp4", 4, 16),
]


def build(H, I, E, L, mode, bits, gs):
    layers = []
    for _ in range(L):
        m = SwitchGLU(H, I, E)
        m.set_dtype(mx.bfloat16)
        nn.quantize(m, group_size=gs, bits=bits, mode=mode)
        mx.eval(m.parameters())
        layers.append(m)
    return layers


def timeit(fn, iters, warm=3, trials=7):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        ts.append((time.perf_counter() - t) / iters)
    return min(ts)


def run(name, P):
    H, I, E, k, L = P["H"], P["I"], P["E"], P["k"], P["L"]
    rows = []
    for mode, bits, gs in CONFIGS:
        layers = build(H, I, E, L, mode, bits, gs)
        out = dict(mode=mode, bits=bits, gs=gs)
        for T in (1, 4):
            x = mx.random.normal((1, T, H)).astype(mx.bfloat16)
            idx = [mx.random.randint(0, E, (1, T, k)).astype(mx.uint32) for _ in range(8)]
            w = mx.softmax(mx.random.normal((1, T, k)), -1).astype(mx.float32)
            sh = mx.zeros((1, T, H), dtype=mx.bfloat16)
            st = {"i": 0}

            def f():
                y = x
                for r in range(4):
                    for m in layers:
                        y = y + m(y, idx[(r + st["i"]) % 8], w, sh)
                st["i"] += 1
                return y

            out[f"T{T}_us_per_layer"] = round(timeit(f, 10) / (4 * L) * 1e6, 1)
        rows.append(out)
        print(name, out, flush=True)
        del layers
        mx.clear_cache()
    return rows


if __name__ == "__main__":
    res = {n: run(n, SHAPES[n]) for n in (sys.argv[1:] or SHAPES)}
    print("peak GB", round(mx.get_peak_memory() / 1e9, 2))
    print(json.dumps(res))
