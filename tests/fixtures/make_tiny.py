"""Write tiny random HF-format source checkpoints for glm5_next and qwen4_exp.

The checkpoints mirror the real repos (zai-org/GLM-5.3-Flash, Qwen/Qwen3.8-Flash-Next):
same config.json schema, same tensor names, dtypes and layouts, sharded with a
model.safetensors.index.json. GLM is FP8 E4M3 with 128x128 block `weight_scale_inv`
exactly where the real repo has it; Qwen is BF16 with fused stacked experts and a
sharded PLE n-gram embedding. Vision weights are omitted (mlx_vlm drops the tower).

    uv run python tests/fixtures/make_tiny.py --arch glm --out /tmp/tiny-glm
    from tests.fixtures.make_tiny import make_glm, make_qwen
"""

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

FP8_BLOCK = 128
FP8_MAX = 448.0

_DTYPES = {
    mx.bfloat16: "BF16",
    mx.float32: "F32",
    mx.float16: "F16",
    mx.int64: "I64",
    mx.int32: "I32",
}


def _np_bytes(arr: mx.array, dtype_name: str) -> bytes:
    if dtype_name == "BF16":
        return np.array(arr.astype(mx.bfloat16).view(mx.uint16)).tobytes()
    if dtype_name == "F8_E4M3":
        return np.array(arr).astype(np.uint8).tobytes()
    return np.array(arr).tobytes()


def write_safetensors(path: Path, tensors: dict):
    """tensors: name -> (dtype_name, mx.array). Writes the raw safetensors layout."""
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        dtype_name, arr = tensors[name]
        data = _np_bytes(arr, dtype_name)
        header[name] = {
            "dtype": dtype_name,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in blobs:
            f.write(blob)


def fp8_block_quantize(w: mx.array):
    """DeepSeek-style block FP8: returns (e4m3 bytes as uint8, F32 scale_inv grid)."""
    rows, cols = w.shape
    pr, pc = (-rows) % FP8_BLOCK, (-cols) % FP8_BLOCK
    wp = mx.pad(w.astype(mx.float32), ((0, pr), (0, pc)))
    blocks = wp.reshape(
        (rows + pr) // FP8_BLOCK, FP8_BLOCK, (cols + pc) // FP8_BLOCK, FP8_BLOCK
    )
    scale = mx.maximum(mx.abs(blocks).max(axis=(1, 3)), 1e-12) / FP8_MAX
    q = (blocks / scale[:, None, :, None]).reshape(rows + pr, cols + pc)[:rows, :cols]
    return mx.to_fp8(q), scale.astype(mx.float32)


class _Builder:
    def __init__(self, seed: int):
        self.key = mx.random.key(seed)
        self.tensors = {}

    def _next(self):
        self.key, sub = mx.random.split(self.key)
        return sub

    def normal(self, shape, std):
        return mx.random.normal(shape, key=self._next()) * std

    def linear(self, name, out_dim, in_dim, fp8=False, std=None):
        w = self.normal((out_dim, in_dim), std or in_dim**-0.5)
        if fp8:
            q, s = fp8_block_quantize(w)
            self.tensors[f"{name}.weight"] = ("F8_E4M3", q)
            self.tensors[f"{name}.weight_scale_inv"] = ("F32", s)
        else:
            self.tensors[f"{name}.weight"] = ("BF16", w)

    def put(self, name, arr, dtype="BF16"):
        self.tensors[name] = (dtype, arr)

    def ones(self, name, n, dtype="BF16"):
        self.put(name, mx.ones((n,)), dtype)

    def small(self, name, shape, std=0.02, dtype="BF16"):
        self.put(name, self.normal(shape, std), dtype)


def _write_sharded(out: Path, groups: list, config: dict, extra_json: dict):
    out.mkdir(parents=True, exist_ok=True)
    n = len(groups)
    weight_map, total = {}, 0
    for k, tensors in enumerate(groups):
        fname = f"model-{k + 1:05d}-of-{n:05d}.safetensors"
        write_safetensors(out / fname, tensors)
        for name, (dt, arr) in tensors.items():
            weight_map[name] = fname
            total += arr.nbytes if dt != "F8_E4M3" else arr.size
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    for name, payload in extra_json.items():
        with open(out / name, "w") as f:
            json.dump(payload, f, indent=2)
    return out


def _copy_tokenizer(src, out: Path):
    if src is None:
        return
    for p in Path(src).iterdir():
        if p.name.startswith("tokenizer") or p.suffix == ".jinja" or p.name in (
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ):
            (out / p.name).write_bytes(p.read_bytes())


# --------------------------------------------------------------------------- GLM


def glm_config(vocab_size: int = 1024):
    layer_types = ["linear_attention", "linear_attention", "deepseek_sparse_attention"]
    n = len(layer_types)
    eos = [vocab_size - 4, vocab_size - 3, vocab_size - 2]
    text = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "eos_token_id": eos,
        "first_k_dense_replace": 1,
        "hc_eps": 1e-06,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 0,
        "hidden_act": "silu",
        "hidden_size": 256,
        "index_head_dim": 32,
        "index_kpool": 4,
        "index_kpool_always_select_tail": True,
        "index_kpool_compress": True,
        "index_n_heads": 4,
        "index_topk": 32,
        "index_share_for_mtp_iteration": True,
        "indexer_rope_interleave": True,
        "indexer_types": ["full"] * n,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "kv_lora_rank": 128,
        "layer_types": layer_types,
        "linear_attn_config": {
            "num_heads": 4,
            "gate_lower_bound": -5.0,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "kda_layers": [i for i, t in enumerate(layer_types) if t == "linear_attention"],
            "full_attn_layers": [i for i, t in enumerate(layer_types) if t != "linear_attention"],
        },
        "max_position_embeddings": 1048576,
        "mhc": True,
        "mla_use_nope": True,
        "mlp_layer_types": ["dense"] + ["sparse"] * (n - 1),
        "model_type": "glm5_next_text",
        "moe_intermediate_size": 192,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": n,
        "num_key_value_heads": 4,
        "num_nextn_predict_layers": 1,
        "output_router_logits": False,
        "pad_token_id": eos[0],
        "q_lora_rank": 96,
        "qk_head_dim": 64,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 0,
        "rms_norm_eps": 1e-05,
        "routed_scaling_factor": 2.5,
        "router_aux_loss_coef": 0.001,
        "scoring_func": "sigmoid",
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "use_cache": True,
        "v_head_dim": 64,
        "vocab_size": vocab_size,
    }
    not_convert = [
        "attn_mha", "attn_mqa", "dt_bias", "hyper_connection", "lm_head", "mapping_proj",
        "model.embed_tokens", "model.norm", "router", "weights_proj",
    ]
    for i in range(n + 1):
        p = f"model.layers.{i}"
        names = ["input_layernorm", "post_attention_layernorm"]
        if i < n:
            names += [f"hc_{s}_{x}" for s in ("attn", "ffn") for x in ("base", "fn", "scale")]
        else:
            names += ["eh_proj", "enorm", "hnorm", "shared_head.norm"]
        if i < n and layer_types[i] == "linear_attention":
            names += [f"self_attn.{x}" for x in (
                "A_log", "b_proj", "dt_bias", "f_a_proj", "f_b_proj", "fused_qkvbfg_a_proj",
                "g_a_proj", "g_b_proj", "k_conv1d", "k_proj", "o_norm", "o_proj", "q_conv1d",
                "q_proj", "qkv_proj", "v_conv1d", "v_proj")]
        else:
            names += [f"self_attn.{x}" for x in (
                "indexer.index_kpool_compress_ape", "indexer.index_kpool_compress_gate",
                "indexer.k_norm", "indexer.k_norm.bias", "indexer.weights_proj", "indexer.wk",
                "indexer.wq_b", "kv_a_layernorm", "kv_b_proj", "q_a_layernorm")]
        if i == n or text["mlp_layer_types"][i] == "sparse":
            names += ["mlp.gate", "mlp.gate.e_score_correction_bias"]
        not_convert += [f"{p}.{x}" for x in names]
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "attention_bias": True, "attention_dropout": 0.0, "depth": 1, "hidden_act": "silu",
            "hidden_size": 64, "image_size": 448, "in_channels": 3, "initializer_range": 0.02,
            "intermediate_size": 128, "model_type": "glm5_next_vision", "num_heads": 2,
            "out_hidden_size": 256, "patch_size": 14, "projection_intermediate_size": 256,
            "rms_norm_eps": 1e-05, "spatial_merge_size": 2, "swiglu_limit": 10.0,
            "temporal_patch_size": 2,
        },
        "image_token_id": vocab_size - 10,
        "video_token_id": vocab_size - 9,
        "image_start_token_id": vocab_size - 8,
        "image_end_token_id": vocab_size - 7,
        "video_start_token_id": vocab_size - 6,
        "video_end_token_id": vocab_size - 5,
        "tie_word_embeddings": False,
        "model_type": "glm5_next",
        "transformers_version": "5.16.0",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "modules_to_not_convert": not_convert,
            "quant_method": "fp8",
            "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
        },
    }


def _glm_dsa(b: _Builder, p: str, t: dict):
    D, H = t["hidden_size"], t["num_attention_heads"]
    qlr, kvlr = t["q_lora_rank"], t["kv_lora_rank"]
    nope, vd = t["qk_nope_head_dim"], t["v_head_dim"]
    ih, ihd = t["index_n_heads"], t["index_head_dim"]
    b.linear(f"{p}.self_attn.q_a_proj", qlr, D, fp8=True)
    b.linear(f"{p}.self_attn.kv_a_proj_with_mqa", kvlr + t["qk_rope_head_dim"], D, fp8=True)
    b.ones(f"{p}.self_attn.q_a_layernorm.weight", qlr)
    b.ones(f"{p}.self_attn.kv_a_layernorm.weight", kvlr)
    b.linear(f"{p}.self_attn.q_b_proj", H * nope, qlr, fp8=True)
    b.linear(f"{p}.self_attn.kv_b_proj", H * (nope + vd), kvlr)
    b.linear(f"{p}.self_attn.o_proj", D, H * vd, fp8=True)
    b.linear(f"{p}.self_attn.indexer.wq_b", ih * ihd, qlr)
    b.linear(f"{p}.self_attn.indexer.wk", ihd, D)
    b.ones(f"{p}.self_attn.indexer.k_norm.weight", ihd)
    b.put(f"{p}.self_attn.indexer.k_norm.bias", mx.zeros((ihd,)))
    b.linear(f"{p}.self_attn.indexer.weights_proj", ih, D)
    b.small(f"{p}.self_attn.indexer.index_kpool_compress_ape", (t["index_kpool"], ihd))
    b.small(f"{p}.self_attn.indexer.index_kpool_compress_gate", (ihd, D))


def _glm_moe(b: _Builder, p: str, t: dict):
    D, E, Im = t["hidden_size"], t["n_routed_experts"], t["moe_intermediate_size"]
    b.put(f"{p}.mlp.gate.weight", b.normal((E, D), D**-0.5))
    b.put(f"{p}.mlp.gate.e_score_correction_bias", b.normal((E,), 0.01), "F32")
    for e in range(E):
        b.linear(f"{p}.mlp.experts.{e}.gate_proj", Im, D, fp8=True)
        b.linear(f"{p}.mlp.experts.{e}.up_proj", Im, D, fp8=True)
        b.linear(f"{p}.mlp.experts.{e}.down_proj", D, Im, fp8=True)
    Is = Im * t["n_shared_experts"]
    b.linear(f"{p}.mlp.shared_experts.gate_proj", Is, D, fp8=True)
    b.linear(f"{p}.mlp.shared_experts.up_proj", Is, D, fp8=True)
    b.linear(f"{p}.mlp.shared_experts.down_proj", D, Is, fp8=True)


def _override(config, text_overrides):
    t = config["text_config"]
    t.update(text_overrides or {})
    if "linear_attn_config" in t and text_overrides and "linear_attn_config" in text_overrides:
        t["linear_attn_config"] = {**glm_config()["text_config"]["linear_attn_config"],
                                   **text_overrides["linear_attn_config"]}
    return config


def make_glm(out, vocab_size: int = 1024, seed: int = 0, tokenizer_dir=None,
             text_overrides=None) -> Path:
    """text_overrides: dims to change (keep the 3-layer layout), e.g. real widths for perf."""
    out = Path(out)
    config = _override(glm_config(vocab_size), text_overrides)
    t = config["text_config"]
    D, hc, n = t["hidden_size"], t["hc_mult"], t["num_hidden_layers"]
    lin = t["linear_attn_config"]
    lh, lhd, lk = lin["num_heads"], lin["head_dim"], lin["short_conv_kernel_size"]
    P = lh * lhd
    mix = (2 + hc) * hc
    groups = []

    top = _Builder(seed)
    top.put("model.language_model.embed_tokens.weight", top.normal((vocab_size, D), 1.0))
    top.put("lm_head.weight", top.normal((vocab_size, D), D**-0.5))
    top.ones("model.language_model.norm.weight", D)
    groups.append(top.tensors)

    for i in range(n + 1):
        b = _Builder(seed + 1 + i)
        p = f"model.language_model.layers.{i}"
        b.ones(f"{p}.input_layernorm.weight", D)
        b.ones(f"{p}.post_attention_layernorm.weight", D)
        if i == n:
            b.linear(f"{p}.eh_proj", D, 2 * D)
            for name in ("enorm", "hnorm", "shared_head.norm"):
                b.ones(f"{p}.{name}.weight", D)
            _glm_dsa(b, p, t)
            _glm_moe(b, p, t)
            groups.append(b.tensors)
            continue
        for site in ("attn", "ffn"):
            b.small(f"{p}.hc_{site}_fn", (mix, hc * D))
            b.put(f"{p}.hc_{site}_base", b.normal((mix,), 0.1), "F32")
            b.put(f"{p}.hc_{site}_scale", mx.ones((3,)), "F32")
        if t["layer_types"][i] == "linear_attention":
            for x in ("q", "k", "v"):
                b.linear(f"{p}.self_attn.{x}_proj", P, D)
                b.small(f"{p}.self_attn.{x}_conv1d.weight", (P, 1, lk), std=0.5)
            b.linear(f"{p}.self_attn.f_a_proj", lhd, D)
            b.linear(f"{p}.self_attn.b_proj", lh, D)
            b.linear(f"{p}.self_attn.g_a_proj", lhd, D)
            b.linear(f"{p}.self_attn.f_b_proj", P, lhd)
            b.linear(f"{p}.self_attn.g_b_proj", P, lhd)
            a = mx.random.uniform(1, 16, (lh,), key=b._next())
            b.put(f"{p}.self_attn.A_log", mx.log(a), "F32")
            b.put(f"{p}.self_attn.dt_bias", b.normal((P,), 0.1), "F32")
            b.ones(f"{p}.self_attn.o_norm.weight", lhd)
            b.linear(f"{p}.self_attn.o_proj", D, P)
        else:
            _glm_dsa(b, p, t)
        if t["mlp_layer_types"][i] == "dense":
            I = t["intermediate_size"]
            b.linear(f"{p}.mlp.gate_proj", I, D, fp8=True)
            b.linear(f"{p}.mlp.up_proj", I, D, fp8=True)
            b.linear(f"{p}.mlp.down_proj", D, I, fp8=True)
        else:
            _glm_moe(b, p, t)
        groups.append(b.tensors)

    eos = t["eos_token_id"]
    gen = {"_from_model_config": True, "eos_token_id": eos, "pad_token_id": eos[0]}
    _write_sharded(out, groups, config, {"generation_config.json": gen})
    _copy_tokenizer(tokenizer_dir, out)
    return out


# -------------------------------------------------------------------------- Qwen


def qwen_config(vocab_size: int = 1024):
    layer_types = ["linear_attention", "linear_attention", "full_attention"]
    eos = vocab_size - 4
    text = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": eos,
        "dtype": "bfloat16",
        "eos_token_id": eos,
        "full_attention_interval": 3,
        "hc_count": 4,
        "hc_lowrank": 32,
        "head_dim": 32,
        "heads_per_ngram": 8,
        "hidden_act": "silu",
        "hidden_size": 128,
        "indexer_budget": 32,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 32,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 2,
        "initializer_range": 0.02,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 32,
        "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 64,
        "mtp": {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1,
            "rope_theta": 10000000,
        },
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "ngram_size": 3,
        "ngram_vocab_size_base": 1000,
        "num_attention_heads": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": len(layer_types),
        "num_key_value_heads": 2,
        "output_gate_type": "sigmoid",
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 128,
        "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 64,
        "split_ngram_parts": 4,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": vocab_size,
    }
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": vocab_size - 8,
        "language_model_only": False,
        "model_type": "qwen4_exp",
        "text_config": text,
        "tie_word_embeddings": False,
        "transformers_version": "5.8.0.dev0",
        "video_token_id": vocab_size - 7,
        "vision_config": {
            "deepstack_visual_indexes": [], "depth": 1, "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 64, "in_channels": 3, "initializer_range": 0.02,
            "intermediate_size": 128, "model_type": "qwen4_exp", "num_heads": 2,
            "num_position_embeddings": 2304, "out_hidden_size": 128, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
        "vision_end_token_id": vocab_size - 5,
        "vision_start_token_id": vocab_size - 6,
    }


def _qwen_hc(b: _Builder, p: str, t: dict, combine=True):
    D, hc, lr = t["hidden_size"], t["hc_count"], t["hc_lowrank"]
    b.small(f"{p}.hc_norm.weight", (hc * D,))
    b.linear(f"{p}.input_mix_weight_down", lr, hc * D)
    b.linear(f"{p}.input_mix_weight_up", hc * D, lr)
    if combine:
        b.linear(f"{p}.block_inject_weight", hc, hc * D)


def _qwen_full_attn(b: _Builder, p: str, t: dict):
    D, H, KV, hd = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    b.linear(f"{p}.self_attn.q_proj", H * hd * 2, D)
    b.linear(f"{p}.self_attn.k_proj", KV * hd, D)
    b.linear(f"{p}.self_attn.v_proj", KV * hd, D)
    b.linear(f"{p}.self_attn.o_proj", D, H * hd)
    b.small(f"{p}.self_attn.q_norm.weight", (hd,))
    b.small(f"{p}.self_attn.k_norm.weight", (hd,))
    ih, ikv, ihd = t["indexer_n_heads"], t["indexer_kv_heads"], t["indexer_head_dim"]
    b.linear(f"{p}.self_attn.indexer.index_qk_proj", (ih + ikv) * ihd, D)
    b.small(f"{p}.self_attn.indexer.q_layernorm.weight", (ihd,))
    b.small(f"{p}.self_attn.indexer.k_layernorm.weight", (ihd,))


def _qwen_moe(b: _Builder, p: str, t: dict):
    D, E, Im, Is = t["hidden_size"], t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    b.put(f"{p}.mlp.experts.gate_up_proj", b.normal((E, 2 * Im, D), D**-0.5))
    b.put(f"{p}.mlp.experts.down_proj", b.normal((E, D, Im), Im**-0.5))
    b.linear(f"{p}.mlp.gate", E, D)
    b.linear(f"{p}.mlp.shared_expert.gate_proj", Is, D)
    b.linear(f"{p}.mlp.shared_expert.up_proj", Is, D)
    b.linear(f"{p}.mlp.shared_expert.down_proj", D, Is)
    b.linear(f"{p}.mlp.shared_expert_gate", 1, D)


def _qwen_ngram_layout(config: dict, layer_idx: int, ple_index: int):
    from mlx_vlm.models.qwen4_exp.config import TextConfig
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpNGramEmbedding

    tc = TextConfig.from_dict(config["text_config"])
    emb = Qwen4ExpNGramEmbedding(tc, tc.ple_embed_dim, layer_idx, ple_index)
    return emb


def make_qwen(out, vocab_size: int = 1024, seed: int = 0, tokenizer_dir=None,
              text_overrides=None) -> Path:
    out = Path(out)
    config = _override(qwen_config(vocab_size), text_overrides)
    t = config["text_config"]
    D, hc, n = t["hidden_size"], t["hc_count"], t["num_hidden_layers"]
    K = t["linear_conv_kernel_dim"]
    groups = []

    top = _Builder(seed)
    top.put("model.language_model.embed_tokens.weight", top.normal((vocab_size, D), 1.0))
    top.put("lm_head.weight", top.normal((vocab_size, D), D**-0.5))
    _qwen_hc(top, "model.language_model.hyper_connection_mixer", t, combine=False)
    groups.append(top.tensors)

    for i in range(n):
        b = _Builder(seed + 1 + i)
        p = f"model.language_model.layers.{i}"
        _qwen_hc(b, f"{p}.attn_hyper_connection", t)
        _qwen_hc(b, f"{p}.mlp_hyper_connection", t)
        if t["layer_types"][i] == "linear_attention":
            hk, hv = t["linear_num_key_heads"], t["linear_num_value_heads"]
            dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
            conv_dim = 2 * hk * dk + hv * dv
            a = mx.random.uniform(1, 16, (hv,), key=b._next())
            b.put(f"{p}.linear_attn.A_log", mx.log(a))
            b.small(f"{p}.linear_attn.conv1d.weight", (conv_dim, 1, K), std=0.5)
            b.put(f"{p}.linear_attn.dt_bias", mx.ones((hv,)))
            b.linear(f"{p}.linear_attn.in_proj_a", hv, D)
            b.linear(f"{p}.linear_attn.in_proj_b", hv, D)
            b.linear(f"{p}.linear_attn.in_proj_qkv", conv_dim, D)
            b.linear(f"{p}.linear_attn.in_proj_z", hv * dv, D)
            b.ones(f"{p}.linear_attn.norm.weight", dv)
            b.linear(f"{p}.linear_attn.out_proj", D, hv * dv)
        else:
            _qwen_full_attn(b, p, t)
        _qwen_moe(b, p, t)
        if i + 1 in t["ple_layer_ids"]:
            pe = t["ple_embed_dim"]
            q = f"{p}.ple"
            b.small(f"{q}.conv1d.weight", (hc * D, 1, t["ple_conv_kernel_size"]), std=0.5)
            b.linear(f"{q}.key_proj", hc * D, pe)
            b.linear(f"{q}.value_proj", D, pe)
            for name in ("norm_conv", "norm_key", "norm_query"):
                b.small(f"{q}.{name}.weight", (hc * D,))
            emb = _qwen_ngram_layout(config, i, t["ple_layer_ids"].index(i + 1))
            b.put(f"{q}.ple_embedding.layer_multipliers", emb.layer_multipliers, "I64")
            b.put(f"{q}.ple_embedding.ngram_heads_offsets", emb.ngram_heads_offsets, "I64")
            b.put(f"{q}.ple_embedding.ngram_heads_vocab_sizes", emb.ngram_heads_vocab_sizes, "I64")
            for k, rows in enumerate(emb.ngram_embedding.shard_sizes):
                b.put(f"{q}.ple_embedding.ngram_embedding.shard_{k}.weight",
                      b.normal((rows, emb.ngram_embedding.dims), 1.0))
        groups.append(b.tensors)

    m = _Builder(seed + 100)
    m.linear("mtp.fc_embedding", D, D)
    m.linear("mtp.fc_hidden", D, D)
    m.small("mtp.pre_fc_norm_embedding.weight", (D,))
    m.small("mtp.pre_fc_norm_hidden.weight", (hc * D,))
    _qwen_hc(m, "mtp.hyper_connection_mixer", t, combine=False)
    _qwen_hc(m, "mtp.layers.0.attn_hyper_connection", t)
    _qwen_hc(m, "mtp.layers.0.mlp_hyper_connection", t)
    _qwen_full_attn(m, "mtp.layers.0", t)
    _qwen_moe(m, "mtp.layers.0", t)
    groups.append(m.tensors)

    gen = {"bos_token_id": t["eos_token_id"], "eos_token_id": [t["eos_token_id"]],
           "pad_token_id": t["eos_token_id"]}
    _write_sharded(out, groups, config, {"generation_config.json": gen})
    _copy_tokenizer(tokenizer_dir, out)
    return out


def make(arch: str, out, **kwargs) -> Path:
    return {"glm": make_glm, "qwen": make_qwen}[arch](out, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=["glm", "qwen"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer-dir", default=None,
                    help="copy tokenizer/chat-template files from here (use with a matching --vocab-size)")
    args = ap.parse_args()
    path = make(args.arch, args.out, vocab_size=args.vocab_size, seed=args.seed,
                tokenizer_dir=args.tokenizer_dir)
    print(path)


if __name__ == "__main__":
    main()
