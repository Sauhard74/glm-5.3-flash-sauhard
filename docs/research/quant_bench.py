import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

mx.set_memory_limit(6 * 1024**3)
REP = 8

SHAPES = {
    "qwen": dict(H=2560, I=640, E=64, k=10, L=4),
    "glm": dict(H=4096, I=2048, E=24, k=8, L=3),
}
CONFIGS = [("bf16", None, None)]
for b in (2, 3, 4, 5, 6, 8):
    for g in (32, 64, 128):
        CONFIGS.append(("affine", b, g))
CONFIGS += [("mxfp4", 4, 32), ("nvfp4", 4, 16), ("mxfp8", 8, 32)]


def build(H, I, E, L, mode, bits, gs):
    layers = []
    for _ in range(L):
        m = SwitchGLU(H, I, E)
        m.set_dtype(mx.bfloat16)
        if mode != "bf16":
            nn.quantize(m, group_size=gs, bits=bits, mode=mode)
        mx.eval(m.parameters())
        layers.append(m)
    return layers


def nbytes(m):
    from mlx.utils import tree_flatten

    return sum(v.nbytes for _, v in tree_flatten(m.parameters()))


def timeit(fn, iters, warm=3, trials=5):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        ts.append((time.perf_counter() - t) / iters)
    return sorted(ts)[len(ts) // 2]


def run(name, P, prefill_m):
    H, I, E, k, L = P["H"], P["I"], P["E"], P["k"], P["L"]
    rows = []
    for mode, bits, gs in CONFIGS:
        try:
            layers = build(H, I, E, L, mode, bits, gs)
        except Exception as e:
            rows.append(dict(mode=mode, bits=bits, gs=gs, err=str(e)[:80]))
            continue
        per_expert = nbytes(layers[0]) / E
        xd = mx.random.normal((1, 1, H)).astype(mx.bfloat16)
        idx_d = [mx.random.randint(0, E, (1, 1, k)).astype(mx.uint32) for _ in range(8)]

        state = {"i": 0}

        def dec():
            y = xd
            for r in range(REP):
                for m in layers:
                    y = y + m(y, idx_d[(r + state["i"]) % 8]).sum(-2)
            state["i"] += 1
            return y

        t_dec = timeit(dec, 10) / (L * REP)
        gbps = k * per_expert / t_dec / 1e9

        xp = mx.random.normal((1, prefill_m, H)).astype(mx.bfloat16)
        idx_p = mx.random.randint(0, E, (1, prefill_m, k)).astype(mx.uint32)

        def pre():
            y = xp
            for m in layers:
                y = y + m(y, idx_p).sum(-2)
            return y

        t_pre = timeit(pre, 3, warm=2) / L
        rows.append(
            dict(
                mode=mode,
                bits=bits,
                gs=gs,
                bpw=round(per_expert * 8 / (3 * H * I), 3),
                decode_us_per_layer=round(t_dec * 1e6, 1),
                decode_eff_GBps=round(gbps, 1),
                prefill_tok_s_per_layer=round(prefill_m / t_pre),
            )
        )
        print(name, rows[-1], flush=True)
        del layers
        mx.clear_cache()
    return rows


if __name__ == "__main__":
    out = {}
    for name in sys.argv[1:] or SHAPES:
        out[name] = run(name, SHAPES[name], 2048)
    print("peak GB", mx.get_peak_memory() / 1e9)
    print(json.dumps(out))
