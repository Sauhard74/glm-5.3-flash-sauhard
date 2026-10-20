"""REAP / frequency stats from the probes vs an independent per-expert computation."""

import mlx.core as mx
import numpy as np
import pytest

from calib.arch import make_arch
from calib.probe import LayerRecorder


def _expert_out(sw, e, x):
    f = lambda p: getattr(sw, p).weight[e].astype(mx.float32)
    up, gate = x @ f("up_proj").T, x @ f("gate_proj").T
    return sw.activation(up, gate) @ f("down_proj").T


def _routing(arch, mlp, x):
    if arch.name == "glm":
        inds, w = mlp.gate(x)
        return np.array(inds[0]), np.array(w[0].astype(mx.float32))
    probs = mx.softmax(mlp.gate(x), axis=-1, precise=True)[0]
    k = mlp.top_k
    inds = mx.argsort(-probs, axis=-1)[:, :k]
    w = mx.take_along_axis(probs, inds, axis=-1)
    w = w / w.sum(-1, keepdims=True)
    return np.array(inds), np.array(w.astype(mx.float32))


@pytest.mark.parametrize("arch_name", ["glm", "qwen"])
def test_reap_and_freq_match_manual(tiny, arch_name):
    arch = make_arch(arch_name, tiny[arch_name], mx.float32)
    layer = arch.load_layer(arch.n_layers - 1)
    mlp = arch.moe(layer)
    assert mlp is not None
    D = arch.hidden_size
    x = mx.random.normal((1, 40, D), key=mx.random.key(3))
    ref_out = mlp(x)
    mx.eval(ref_out)

    rec = LayerRecorder(arch, layer, {}, mlp, ["code"], expert_tokens=1000)
    rec.begin("code", True)
    out = mlp(x)
    rec.flush(out, 40)
    assert mx.allclose(out, ref_out, atol=1e-5).item(), "probe must not change the output"

    inds, w = _routing(arch, mlp, x)
    E = rec.E
    count, gnorm = np.zeros(E), np.zeros(E)
    sw = mlp.switch_mlp
    for t in range(inds.shape[0]):
        for k in range(inds.shape[1]):
            e = int(inds[t, k])
            n = float(mx.sqrt(mx.square(_expert_out(sw, e, x[0, t])).sum()))
            count[e] += 1
            gnorm[e] += w[t, k] * n
    assert np.array_equal(rec.count[0], count)
    # With >= 64 routed pairs SwitchGLU uses sorted gather_mm, which differs from this naive
    # fp32 reference by a uniform ~0.2%; the probe records what the model actually computes.
    assert np.allclose(rec.gnorm[0], gnorm, rtol=5e-3)
    assert np.array_equal(np.argsort(-rec.gnorm[0]), np.argsort(-gnorm)), "REAP ranking must match"
    # every routed (token, expert) pair fits under the cap, so the samples are exact
    for e in range(E):
        got = rec.expert_res.rows[e]
        assert (0 if got is None else len(got)) == count[e]
    rec.remove()
    assert type(mlp.switch_mlp).__name__ == "SwitchGLU"
