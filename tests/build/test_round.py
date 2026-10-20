import mlx.core as mx
import numpy as np
import pytest

from build.round import pack, quantize


def _unpack(packed, bits, n):
    """Reference unpack of MLX's affine layout via mx.dequantize with unit scales and zero biases."""
    lead = packed.shape[:-1]
    ones = mx.ones((*lead, n // 32), dtype=mx.float32)
    return mx.dequantize(packed, ones, mx.zeros_like(ones), group_size=32, bits=bits).astype(mx.uint32)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_pack_matches_mlx_layout(bits):
    w = mx.random.normal((16, 256), key=mx.random.key(bits))
    packed, s, b = mx.quantize(w, group_size=64, bits=bits, mode="affine")
    codes = _unpack(packed, bits, 256)
    assert mx.array_equal(pack(codes, bits), packed).item()


@pytest.mark.parametrize("bits,group", [(3, 64), (3, 128), (4, 64), (2, 32), (5, 128), (6, 64), (8, 128)])
def test_lsq_never_worse_than_rtn_and_loads(bits, group):
    w = (mx.random.normal((64, 512), key=mx.random.key(0)) * 0.02).astype(mx.bfloat16)
    h = mx.random.uniform(shape=(512,), key=mx.random.key(1)) ** 4  # skewed channel importances
    stats = {}
    q, s, b = quantize(w, bits, group, rounding="lsq", h=h, stats=stats)
    assert q.shape == mx.quantize(w, group_size=group, bits=bits)[0].shape and s.dtype == mx.bfloat16
    assert stats["err_lsq"] <= stats["err_rtn"] * (1 + 1e-6)
    # what mx.dequantize reconstructs is what LSQ scored (at 6-8 bits the error is below bf16 output rounding)
    if bits > 5:
        return
    deq = mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.float32)
    hh = (h + 0.01 * h.mean())[None]
    e = float((hh * (deq - w.astype(mx.float32)) ** 2).sum())
    np.testing.assert_allclose(e, stats["err_lsq"], rtol=2e-3)
    if bits <= 4:
        assert stats["err_lsq"] < 0.97 * stats["err_rtn"]


def test_lsq_per_expert_imatrix_and_fallbacks():
    w = (mx.random.normal((3, 32, 256), key=mx.random.key(2)) * 0.02).astype(mx.bfloat16)
    h = mx.random.uniform(shape=(3, 256), key=mx.random.key(3))
    q, s, b = quantize(w, 3, 64, rounding="lsq", h=h)
    ref = mx.quantize(w, group_size=64, bits=3)
    assert q.shape == ref[0].shape and s.shape == ref[1].shape
    bad = mx.ones((5, 256))  # wrong expert count -> unweighted, still valid
    q2, s2, _ = quantize(w, 3, 64, rounding="lsq", h=bad)
    assert q2.shape == ref[0].shape
    assert mx.array_equal(quantize(w, 3, 64)[0], ref[0]).item()  # rtn is mx.quantize
