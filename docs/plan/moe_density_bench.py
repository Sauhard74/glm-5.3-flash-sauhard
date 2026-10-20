# Tiny bench (<1 GB): bf16 SwitchGLU prefill throughput vs tokens-per-expert density.
import time, mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU
mx.set_memory_limit(3 * 1024**3)
def bench(H, I, E, k, T, reps=10):
    m = SwitchGLU(H, I, E); m.set_dtype(mx.bfloat16); mx.eval(m.parameters())
    x = mx.random.normal((1, T, H)).astype(mx.bfloat16)
    idx = mx.argsort(mx.random.uniform(shape=(1, T, E)), axis=-1)[..., :k].astype(mx.uint32)
    f = lambda: m(x, idx)
    for _ in range(3): mx.eval(f())
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(reps): mx.eval(f())
        ts.append((time.perf_counter() - t) / reps)
    dt = min(ts); flop = 2 * 3 * H * I * k * T
    return T / dt, flop / dt / 1e12
for name, H, I, E, k, real_E in (("glm", 4096, 2048, 16, 8, 288), ("qwen", 2560, 640, 32, 10, 512)):
    for tpe in (40, 57, 160, 683):
        T = max(64, tpe * E // k)
        tps, tf = bench(H, I, E, k, T)
        print(f"{name} E={E} top{k} tokens/expert={tpe:4d} T={T:5d}: {tps:9.0f} tok/s/layer  {tf:5.1f} TFLOP/s")
