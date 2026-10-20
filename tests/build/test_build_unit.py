import itertools
import json

import mlx.core as mx
import numpy as np
import pytest

from build import alloc, prune
from build.calib_io import CalibStats, combine_domains, parse_grid


def test_quantized_bytes_matches_mlx():
    for shape in [(64, 256), (3, 32, 128), (5, 160)]:
        w = mx.zeros(shape, dtype=mx.bfloat16)
        for bits in alloc.DEFAULT_BITS:
            for group in (32, 64, 128):
                if shape[-1] % group:
                    continue
                q = mx.quantize(w, group_size=group, bits=bits)
                assert sum(x.nbytes for x in q) == alloc.quantized_bytes(w.size, shape[-1], bits, group)


@pytest.mark.parametrize(
    "obj",
    [
        {"4,64": 0.1, "3,32": 0.2},
        {"(4, 64)": 0.1, "(3, 32)": 0.2},
        {"b4g64": 0.1, "b3g32": 0.2},
        {"4": {"64": 0.1}, "3": {"32": 0.2}},
        [{"bits": 4, "group_size": 64, "value": 0.1}, {"bits": 3, "group": 32, "sens": 0.2}],
        [[4, 64, 0.1], [3, 32, 0.2]],
    ],
)
def test_parse_grid_formats(obj):
    assert parse_grid(obj) == {(4, 64): 0.1, (3, 32): 0.2}


def test_calib_loader_json_npz_and_domains(tmp_path):
    E = 6
    (tmp_path / "layer_003.json").write_text(
        json.dumps(
            {
                "expert_freq": {"all": [1, 2, 3, 4, 5, 6], "code": [6, 5, 4, 3, 2, 1]},
                "expert_reap_code": [1, 1, 1, 1, 1, 9],
                "expert_reap": [1, 2, 3, 4, 5, 6],
                "sens": {"4,64": {"self_attn.q_proj": 0.5}, "8,64": {"self_attn.q_proj": 0.01}},
            }
        )
    )
    np.savez(
        tmp_path / "layer_004.npz",
        expert_freq=np.arange(E, dtype=float),
        **{"expert_reap/reasoning": np.ones(E), "sens/mlp.experts.2.down_proj/4_64": np.array(0.3)},
    )
    (tmp_path / "mtp.json").write_text(json.dumps({"domains": {"code": {"expert_reap": [1] * E}}}))
    (tmp_path / "notes.txt").write_text("ignored")
    stats = CalibStats.load(tmp_path)
    l3 = stats.layers["3"]
    assert l3.freq.tolist() == [1, 2, 3, 4, 5, 6]
    assert l3.domain_freq["code"].tolist() == [6, 5, 4, 3, 2, 1]
    assert l3.domain_reap["code"][-1] == 9
    assert l3.sens["self_attn.q_proj"] == {(4, 64): 0.5, (8, 64): 0.01}
    l4 = stats.layers["4"]
    assert l4.domain_reap["reasoning"].tolist() == [1.0] * E
    assert l4.sens["mlp.experts.2.down_proj"] == {(4, 64): 0.3}
    assert "code" in stats.layers["mtp"].domain_reap
    mixed = combine_domains(l3, "reap", {"code": 1.0})
    assert mixed.tolist() == [1, 1, 1, 1, 1, 9]
    assert combine_domains(l3, "reap").tolist() == [1, 2, 3, 4, 5, 6]


def _scores(score, freq=None):
    s = np.asarray(score, dtype=float)
    return prune.ExpertScores(s, s / s.sum(), None if freq is None else np.asarray(freq, float), "test")


def test_uniform_and_global_keep():
    a = _scores([5, 1, 4, 2, 3, 0.5])
    b = _scores([1, 1, 1, 1, 10, 1])
    assert prune.keep_top(a, 3) == [0, 2, 4]
    plan = prune.global_keep({"0": a, "1": b}, total=6, min_keep={"0": 2, "1": 2})
    assert sum(len(v) for v in plan.values()) == 6
    assert len(plan["0"]) >= 2 and len(plan["1"]) >= 2
    assert 4 in plan["1"] and 0 in plan["0"]
    assert prune.prune_loss(a, list(range(6))) == 0
    assert prune.parse_prune("0.25", 512) == ("keep", 384)
    assert prune.parse_prune("320", 512) == ("keep", 320)
    assert prune.parse_prune("none", 512) == ("keep", 512)
    assert prune.parse_prune("auto", 512) == ("auto", None)


def _brute(options, budget):
    best = None
    for combo in itertools.product(*options):
        b = sum(o.bytes for o in combo)
        s = sum(o.sens for o in combo)
        if b <= budget and (best is None or s < best):
            best = s
    return best


@pytest.mark.parametrize("seed", range(6))
def test_solvers_vs_bruteforce(seed):
    rng = np.random.default_rng(seed)
    options = []
    for _ in range(6):
        numel = int(rng.integers(1, 64)) * 4096
        sens = {(b, g): float(rng.uniform(0.5, 2) * 2.0**-b * (1 + 8 / g)) for b in (2, 3, 4, 8) for g in (32, 64)}
        u = alloc.Unit("main", "x", (numel // 256, 256), 2, "attn", None)
        options.append(alloc.unit_options(u, u.shape, sens, bits=(2, 3, 4, 8)))
    lo = sum(min(o.bytes for o in opts) for opts in options)
    hi = sum(max(o.bytes for o in opts) for opts in options)
    budget = int(lo + (hi - lo) * rng.uniform(0.1, 0.6))
    ref = _brute(options, budget)
    dp = alloc.solve_dp(options, budget)
    greedy = alloc.solve_greedy(options, budget)
    assert dp.bytes <= budget and greedy.bytes <= budget
    assert dp.objective <= ref * 1.02 + 1e-12
    assert greedy.objective <= ref * 1.25 + 1e-12
    assert alloc.solve(options, budget).objective <= min(dp.objective, greedy.objective) + 1e-12
    assert alloc.solve(options, lo - 1) is None


def test_fill_sensitivities_kind_median_and_proxy():
    a = alloc.Unit("main", "l.0.mlp.switch_mlp.down_proj", (4, 64, 128), 2, "expert", "0")
    a.sens = {(4, 64): 0.2, (8, 64): 0.01}
    b = alloc.Unit("main", "l.1.mlp.switch_mlp.down_proj", (4, 64, 128), 2, "expert", "1")
    c = alloc.Unit("main", "l.1.self_attn.q_proj", (64, 128), 2, "attn", "1")
    alloc.fill_sensitivities([a, b, c])
    assert b.sens_source == "median:expert" and b.sens[(4, 64)] == pytest.approx(0.2)
    assert c.sens_source == "proxy"
    assert all(c.sens[(bb, 64)] > c.sens[(bb + 1, 64)] for bb in (2, 3, 4, 5))
    assert len(a.sens) == 12
