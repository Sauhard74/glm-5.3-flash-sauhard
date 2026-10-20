import mlx.core as mx
import numpy as np

from calib import sens as S
from calib.probe import Reservoir, balanced_mean
from calib.run import HiddenWriter, chunk_bounds
from calib.st import Snapshot
from tests.fixtures.make_tiny import fp8_block_quantize, write_safetensors


def test_fp8_block_dequant_roundtrip(tmp_path):
    w = mx.random.normal((300, 200), key=mx.random.key(1)) * 0.05
    q, s = fp8_block_quantize(w)
    assert s.shape == (3, 2)
    write_safetensors(tmp_path / "m.safetensors", {
        "a.weight": ("F8_E4M3", q), "a.weight_scale_inv": ("F32", s), "b": ("BF16", w),
    })
    snap = Snapshot(tmp_path)
    got = snap.load_dequantized(["a.weight", "a.weight_scale_inv", "b"], dtype=mx.float32)
    assert set(got) == {"a.weight", "b"}
    rel = (mx.abs(got["a.weight"] - w).max() / mx.abs(w).max()).item()
    assert rel < 0.07  # e4m3 has a 3-bit mantissa
    assert mx.allclose(got["b"].astype(mx.float32), w, atol=1e-2).item()
    rows = snap.gather_rows("b", np.array([5, 0, 299]))
    assert mx.array_equal(rows, w.astype(mx.bfloat16)[mx.array([5, 0, 299])]).item()


def test_chunk_bounds():
    assert chunk_bounds(10, 64) == [(0, 10)]
    assert chunk_bounds(128, 64) == [(0, 64), (64, 128)]
    assert chunk_bounds(150, 64) == [(0, 64), (64, 150)]  # short tail merged
    b = chunk_bounds(1000, 256)
    assert b[0][0] == 0 and b[-1][1] == 1000 and all(x[1] == y[0] for x, y in zip(b, b[1:]))


def test_reservoir_cap_and_uniformity():
    rng = np.random.default_rng(0)
    hits = np.zeros(1000)
    for _ in range(200):
        r = Reservoir(2, 50, rng)
        for start in range(0, 1000, 100):
            ids = np.arange(start, start + 100)
            r.offer(ids % 2, lambda keep, ids=ids: ids[keep][:, None])
        assert r.count(0) == 50 and r.count(1) == 50
        assert np.all(r.rows[0][:, 0] % 2 == 0)
        hits[r.rows[0][:, 0]] += 1
        hits[r.rows[1][:, 0]] += 1
    early, late = hits[:500].mean(), hits[500:].mean()
    assert abs(early - late) / late < 0.1


def test_balanced_mean_skips_empty_domains():
    sums = [np.array([2.0, 0.0]), None, np.array([9.0, 3.0])]
    counts = [np.array([2.0, 0.0]), 0, np.array([3.0, 1.0])]
    got = balanced_mean(sums, counts)
    assert np.allclose(got, [(1.0 + 3.0) / 2, 3.0])


def test_hidden_writer_promotes_on_overflow(tmp_path):
    p = tmp_path / "h.npy"
    w = HiddenWriter(p, (4, 3), np.float16)
    w.write(0, 2, np.ones((2, 3)))
    w.write(2, 4, np.full((2, 3), 1e6))
    w.close()
    h = np.load(p)
    assert h.dtype == np.float32 and h[0, 0] == 1 and h[3, 0] == 1e6


def test_linear_sens_monotone_and_exact_for_identity():
    rng = np.random.default_rng(0)
    w = mx.array(rng.normal(size=(64, 128)).astype(np.float32))
    x = rng.normal(size=(256, 128)).astype(np.float32)
    grid, energy = S.linear_sens(w, x)
    col = grid[:, S.GROUPS.index(64)]
    assert np.all(np.diff(col) < 0)
    wq = S.qdq(w, 4, 64)
    y, yq = x @ np.array(w).T, x @ np.array(wq).T
    want = ((yq - y) ** 2).sum() / (y**2).sum()
    assert np.isclose(grid[S.BITS.index(4), S.GROUPS.index(64)], want, rtol=1e-3)
    assert np.isclose(energy, (y**2).sum() / 256, rtol=1e-2)
    grid48, _ = S.linear_sens(w[:, :96], x[:, :96])
    assert np.isnan(grid48[:, S.GROUPS.index(64)]).all()


def test_check_validates_tiny(tiny):
    from calib.check import check

    for arch in ("glm", "qwen"):
        _, report = check(arch, tiny[arch])
        assert not any(r["problems"] for r in report.values()), report
        assert any(r.get("moe") for r in report.values())
