import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from recover.qgrad import ExpertGrads


class _Net(nn.Module):
    def __init__(self, E, O, I, group, bits):
        super().__init__()
        from mlx_vlm.models.switch_layers import QuantizedSwitchLinear

        self.proj = QuantizedSwitchLinear(I, O, E, bias=False, group_size=group, bits=bits)

    def __call__(self, x, idx):
        return self.proj(x, idx)


@pytest.mark.parametrize("bits,group", [(3, 32), (4, 64), (3, 128)])
@pytest.mark.parametrize("layout", ["unsorted", "sorted"])
def test_lean_scale_bias_grads_match_mlx(bits, group, layout):
    E, O, I, N, K = 6, 48, 256, 10, 3
    k1, k2, k3, k4 = mx.random.split(mx.random.key(bits * 7 + group), 4)
    if layout == "unsorted":  # x [N,1,1,I] vs indices [N,K]
        x = mx.random.normal((N, 1, 1, I), key=k2)
        idx = mx.random.randint(0, E, (N, K), key=k3).astype(mx.uint32)
        dy = mx.random.normal((N, K, 1, O), key=k4)
    else:  # _gather_sort path: x [N*K,1,I] vs sorted indices [N*K]
        idx = mx.sort(mx.random.randint(0, E, (N * K,), key=k3)).astype(mx.uint32)
        x = mx.random.normal((N * K, 1, I), key=k2)
        dy = mx.random.normal((N * K, 1, O), key=k4)
    net = _Net(E, O, I, group, bits)
    net.proj.scales = net.proj.scales.astype(mx.float32)
    net.proj.biases = net.proj.biases.astype(mx.float32)
    net.freeze()
    net.proj.unfreeze(keys=["scales", "biases"], recurse=False)

    def loss(net, x):
        return (net(x, idx) * dy).sum()

    ref_l, ref_g = nn.value_and_grad(net, loss)(net, x)  # stock MLX VJP
    ref_g = dict(tree_flatten(ref_g))
    eg = ExpertGrads()
    paths = eg.enable(net)
    l, g = nn.value_and_grad(net, loss)(net, x)
    g = dict(tree_flatten(g))
    mx.eval(l, g)
    g = eg.fill(g, paths)
    assert mx.allclose(l, ref_l).item()
    for k in ref_g:
        np.testing.assert_allclose(np.array(g[k]), np.array(ref_g[k]), rtol=2e-3, atol=2e-3 * float(mx.abs(ref_g[k]).max()))
    # input gradient through the lean path matches too
    gx_ref = mx.grad(lambda x: (type(net.proj)._orig_call(net.proj, x, idx) * dy).sum())(x)
    gx = mx.grad(lambda x: loss(net, x))(x)
    np.testing.assert_allclose(np.array(gx), np.array(gx_ref), rtol=2e-3, atol=2e-3 * float(mx.abs(gx_ref).max()))


def test_batched_matches_loop():
    from recover.qgrad import scale_bias_grads, scale_bias_grads_batched

    E, O, I, group, bits = 7, 40, 256, 64, 3
    k1, k2, k3, k4 = mx.random.split(mx.random.key(3), 4)
    q, s, _ = mx.quantize(mx.random.normal((E, O, I), key=k1) * 0.05, group_size=group, bits=bits)
    idx = mx.random.randint(0, E - 1, (50,), key=k3).astype(mx.uint32)  # expert E-1 never routed
    x = mx.random.normal((50, 1, I), key=k2)
    dy = mx.random.normal((50, 1, O), key=k4)
    a = scale_bias_grads(x, dy, q, s, idx, group, bits)
    b = scale_bias_grads_batched(x, dy, q, s, idx, group, bits, chunk=3)
    for u, v in zip(a, b):
        np.testing.assert_allclose(np.array(v), np.array(u), rtol=2e-3, atol=2e-3 * float(mx.abs(u).max()))
