import json
import math

import mlx.core as mx
import pytest

from calib.st import Snapshot


def test_glm_fp8_layout_matches_real_repo(tiny_glm):
    snap = Snapshot(tiny_glm)
    cfg = json.loads((tiny_glm / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "fp8"
    assert cfg["quantization_config"]["weight_block_size"] == [128, 128]
    fp8 = [k for k in snap.keys() if snap.dtype(k) == "F8_E4M3"]
    assert fp8, "expected FP8 tensors"
    for k in fp8:
        s = k + "_scale_inv"
        assert s in snap and snap.dtype(s) == "F32"
        rows, cols = snap.shape(k)
        assert snap.shape(s) == (math.ceil(rows / 128), math.ceil(cols / 128))
    p = "model.language_model.layers"
    assert snap.dtype(f"{p}.1.mlp.experts.0.gate_proj.weight") == "F8_E4M3"
    assert snap.dtype(f"{p}.2.self_attn.q_a_proj.weight") == "F8_E4M3"
    assert snap.dtype(f"{p}.2.self_attn.kv_b_proj.weight") == "BF16"
    assert snap.dtype(f"{p}.0.self_attn.q_proj.weight") == "BF16"
    assert snap.dtype(f"{p}.0.hc_attn_base") == "F32"
    n = cfg["text_config"]["num_hidden_layers"]
    assert f"{p}.{n}.eh_proj.weight" in snap  # MTP layer kept like the real repo


def test_qwen_layout_matches_real_repo(tiny_qwen):
    snap = Snapshot(tiny_qwen)
    assert all(snap.dtype(k) in ("BF16", "I64") for k in snap.keys())
    p = "model.language_model.layers.1"
    E, two_i, D = snap.shape(f"{p}.mlp.experts.gate_up_proj")
    assert (E, two_i, D) == (8, 128, 128)
    assert snap.keys(f"{p}.ple.ple_embedding.ngram_embedding.shard_")
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in snap


@pytest.mark.parametrize("arch", ["glm", "qwen"])
def test_mlx_vlm_loads_tiny(tiny, arch):
    from mlx_vlm.utils import load_model

    model = load_model(tiny[arch])
    lm = model.language_model
    out = lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]), cache=lm.make_cache())
    assert out.logits.shape == (1, 12, 1024)
    assert not mx.isnan(out.logits).any().item()
