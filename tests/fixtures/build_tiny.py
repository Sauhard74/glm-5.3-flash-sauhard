"""Tiny random HF-format checkpoints of the real qwen4_exp / glm5_next architectures.

The weights are produced by instantiating the mlx-vlm classes with a tiny config and
inverting mlx-vlm's ``sanitize`` back to the upstream HF layout (fused Qwen expert
tensors, per-expert GLM tensors, block-FP8 E4M3 + weight_scale_inv for GLM, MTP layer).
``expected_*`` are the oracle MLX weights from running mlx-vlm's own sanitize on the
(dequantized) HF tensors.

    uv run python tests/fixtures/build_tiny.py --model qwen --out /tmp/tiny-qwen [--calib /tmp/tiny-qwen-calib]
"""

import argparse
import copy
import json
import re
import shutil
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_vlm.fp8 import _dequantize_fp8_weight
from mlx_vlm.utils import get_model_and_args, update_module_configs

HERE = Path(__file__).resolve().parent
TOKENIZERS = HERE / "tokenizers"
TOKENIZER_REPOS = {"qwen": "Qwen/Qwen3.8-Flash-Next", "glm": "zai-org/GLM-5.3-Flash"}

QWEN = {
    "model_type": "qwen4_exp",
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "tie_word_embeddings": False,
    "eos_token_id": [248046, 248044],
    "language_model_only": False,
    "transformers_version": "5.8.0.dev0",
    "text_config": {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "shared_expert_intermediate_size": 32,
        "moe_intermediate_size": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 4096,
        "hc_count": 4,
        "hc_lowrank": 16,
        "ple_layer_ids": [2],
        "ple_embed_dim": 512,
        "ngram_vocab_size_base": 1000,
        "split_ngram_parts": 4,
        "indexer_n_heads": 2,
        "indexer_head_dim": 16,
        "indexer_budget": 64,
        "indexer_compress_ratio": 4,
        "indexer_kv_heads": 1,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "heads_per_ngram": 8,
        "ngram_size": 3,
        "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32",
        "mtp": {"hybrid": True, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1, "rope_theta": 10000000},
        "mtp_use_dedicated_embeddings": False,
        "output_gate_type": "sigmoid",
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "ple_conv_kernel_size": 4,
        "router_aux_loss_coef": 0.001,
        "use_cache": True,
        "eos_token_id": 248044,
        "bos_token_id": 248044,
        "mtp_num_hidden_layers": 1,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
        },
    },
    "vision_config": {
        "model_type": "qwen4_exp",
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 2304,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
    },
}

GLM = {
    "model_type": "glm5_next",
    "architectures": ["Glm5NextForConditionalGeneration"],
    "image_token_id": 154854,
    "video_token_id": 154855,
    "image_start_token_id": 154830,
    "image_end_token_id": 154831,
    "video_start_token_id": 154832,
    "video_end_token_id": 154833,
    "tie_word_embeddings": False,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["lm_head"],
    },
    "text_config": {
        "model_type": "glm5_next_text",
        "dtype": "bfloat16",
        "vocab_size": 154880,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_shared_experts": 1,
        "n_routed_experts": 16,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 32,
        "q_lora_rank": 64,
        "qk_rope_head_dim": 0,
        "v_head_dim": 16,
        "qk_nope_head_dim": 16,
        "num_experts_per_tok": 4,
        "first_k_dense_replace": 1,
        "index_topk": 16,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "index_kpool": 4,
        "linear_attn_config": {"num_heads": 4, "head_dim": 32, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "hc_mult": 4,
        "eos_token_id": [154820, 154827, 154829],
        "pad_token_id": 154820,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "moe_router_dtype": "float32",
    },
    "vision_config": {
        "model_type": "glm5_next_vision",
        "depth": 1,
        "hidden_size": 32,
        "num_heads": 2,
        "intermediate_size": 64,
        "out_hidden_size": 64,
        "projection_intermediate_size": 64,
        "patch_size": 14,
        "image_size": 56,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

CONFIGS = {"qwen": QWEN, "glm": GLM}
_DT = {"bfloat16": "BF16", "float32": "F32", "float16": "F16", "int64": "I64", "int32": "I32", "uint8": "U8"}


# ----------------------------------------------------------------------------- models


def instantiate(cfg):
    cfg = copy.deepcopy(cfg)
    arch, _ = get_model_and_args(cfg)
    mc = arch.ModelConfig.from_dict(copy.deepcopy(cfg))
    mc = update_module_configs(mc, arch, cfg, ["text", "vision"])
    return arch, arch.Model(mc), mc


def drafter(family, cfg):
    from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter

    splitter = get_mtp_splitter(source_config=cfg)
    dcfg = {"model_type": splitter.output_model_type, "text_config": copy.deepcopy(cfg["text_config"])}
    arch, _ = get_model_and_args(dcfg)
    return splitter, arch.Model(arch.ModelConfig.from_dict(copy.deepcopy(dcfg)))


def randomize(params, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in params.items():
        if not mx.issubdtype(v.dtype, mx.floating):
            out[k] = v
            continue
        shape = v.shape
        if k.endswith(("mlp.gate.weight", ".gate.weight")) and len(shape) == 2:
            x = rng.normal(0, 0.5, shape)
        elif k.endswith("e_score_correction_bias"):
            x = rng.normal(0, 0.05, shape)
        elif len(shape) >= 2:
            x = rng.normal(0, 0.06, shape)
        elif k.endswith(("A_log", "dt_bias")):
            x = np.array(v) + rng.normal(0, 0.1, shape)
        else:
            x = np.array(v) + rng.normal(0, 0.05, shape)
        out[k] = mx.array(x.astype(np.float32))
    return out


# ----------------------------------------------------------------------------- inverse sanitize


def _hf_prefix(k):
    for new, old in (
        ("model.language_model.", "language_model.model."),
        ("model.visual.", "vision_tower."),
        ("lm_head", "language_model.lm_head"),
    ):
        if k.startswith(old):
            return new + k[len(old) :]
    return k


def unsanitize_qwen(params, mtp_params):
    hf = {}
    for k, v in params.items():
        if ".switch_mlp." in k:
            continue
        if k.endswith("conv1d.weight") and v.shape[-1] == 1:
            v = v.moveaxis(1, 2)
        if k.endswith("patch_embed.proj.weight") and v.ndim == 5:
            v = v.transpose(0, 4, 1, 2, 3)
        k = re.sub(r"\.ngram_embedding\.shards\.(\d+)\.", r".ngram_embedding.shard_\1.", k)
        hf[_hf_prefix(k)] = v
    for src, dst_prefix in ((params, None), (mtp_params, "mtp.")):
        for k in [k for k in src if k.endswith(".switch_mlp.gate_proj.weight")]:
            p = k[: -len(".switch_mlp.gate_proj.weight")]
            gate, up = src[k], src[f"{p}.switch_mlp.up_proj.weight"]
            down = src[f"{p}.switch_mlp.down_proj.weight"]
            base = _hf_prefix(p) if dst_prefix is None else dst_prefix + p
            hf[f"{base}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=1)
            hf[f"{base}.experts.down_proj"] = down
    for k, v in mtp_params.items():
        if ".switch_mlp." not in k:
            hf["mtp." + k] = v
    return hf


def _glm_layer(prefix_hf, rel, v, tc, out):
    """Invert GLM sanitize for one tensor ``rel`` (relative to a decoder layer)."""
    H = tc["num_attention_heads"]
    lin = tc["linear_attn_config"]
    hd, lh = lin["head_dim"], lin["num_heads"]
    m = re.match(r"(attn|ffn)_hc\.(fn|base|scale)$", rel)
    if m:
        out[f"{prefix_hf}hc_{m.group(1)}_{m.group(2)}"] = v
        return
    if rel == "self_attn.qkv_proj.weight":
        for name, part in zip(("q_proj", "k_proj", "v_proj"), mx.split(v, 3, axis=0)):
            out[f"{prefix_hf}self_attn.{name}.weight"] = part
        return
    if rel == "self_attn.qkv_conv.conv.weight":
        for name, part in zip(("q_conv1d", "k_conv1d", "v_conv1d"), mx.split(v, 3, axis=0)):
            out[f"{prefix_hf}self_attn.{name}.weight"] = part.moveaxis(1, 2)
        return
    if rel == "self_attn.fbg_a_proj.weight":
        f_a, b, g_a = mx.split(v, [hd, hd + lh], axis=0)
        out[f"{prefix_hf}self_attn.f_a_proj.weight"] = f_a
        out[f"{prefix_hf}self_attn.b_proj.weight"] = b
        out[f"{prefix_hf}self_attn.g_a_proj.weight"] = g_a
        return
    if rel == "self_attn.qkv_a_proj.weight":
        q_a, kv_a = mx.split(v, [tc["q_lora_rank"]], axis=0)
        out[f"{prefix_hf}self_attn.q_a_proj.weight"] = q_a
        out[f"{prefix_hf}self_attn.kv_a_proj_with_mqa.weight"] = kv_a
        return
    if rel in ("self_attn.embed_q.weight", "self_attn.unembed_out.weight"):
        return  # handled with kv_b_proj
    if rel in ("mlp.gate_up_proj.weight", "mlp.shared_experts.gate_up_proj.weight"):
        base = rel[: -len("gate_up_proj.weight")]
        g, u = mx.split(v, 2, axis=0)
        out[f"{prefix_hf}{base}gate_proj.weight"] = g
        out[f"{prefix_hf}{base}up_proj.weight"] = u
        return
    m = re.match(r"mlp\.switch_mlp\.(\w+)\.weight$", rel)
    if m:
        for e in range(v.shape[0]):
            out[f"{prefix_hf}mlp.experts.{e}.{m.group(1)}.weight"] = v[e]
        return
    out[prefix_hf + rel] = v


def _glm_kv_b(prefix_mlx, prefix_hf, src, tc, out):
    wk = src.get(f"{prefix_mlx}self_attn.embed_q.weight")
    if wk is None:
        return
    wv = src[f"{prefix_mlx}self_attn.unembed_out.weight"]
    value = mx.concatenate([wk.swapaxes(-1, -2), wv], axis=1)
    out[f"{prefix_hf}self_attn.kv_b_proj.weight"] = value.reshape(-1, tc["kv_lora_rank"])


def unsanitize_glm(params, mtp_params, tc):
    hf = {}
    for k, v in params.items():
        m = re.match(r"^language_model\.model\.layers\.(\d+)\.(.*)$", k)
        if m:
            _glm_layer(f"model.language_model.layers.{m.group(1)}.", m.group(2), v, tc, hf)
            continue
        if k.endswith("patch_embed.proj.weight") and v.ndim == 5:
            v = v.transpose(0, 4, 1, 2, 3)
        if k.endswith("downsample.weight") and v.ndim == 4:
            v = v.transpose(0, 3, 1, 2)
        hf[_hf_prefix(k)] = v
    for i in range(tc["num_hidden_layers"]):
        _glm_kv_b(f"language_model.model.layers.{i}.", f"model.language_model.layers.{i}.", params, tc, hf)
    L = tc["num_hidden_layers"]
    pre = f"model.language_model.layers.{L}."
    for k, v in mtp_params.items():
        if k.startswith("mtp_block."):
            _glm_layer(pre, k[len("mtp_block.") :], v, tc, hf)
        elif k.startswith("shared_head_norm."):
            hf[pre + "shared_head.norm." + k[len("shared_head_norm.") :]] = v
        else:
            hf[pre + k] = v
    _glm_kv_b("mtp_block.", pre, mtp_params, tc, hf)
    return hf


_GLM_FP8 = re.compile(
    r"mlp\.(experts\.\d+\.|shared_experts\.)?(gate|up|down)_proj\.weight$"
    r"|self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$"
)


def glm_fp8_keys(hf, tc):
    dsa = {i for i, t in enumerate(_layer_types(tc)) if t != "linear_attention"} | {tc["num_hidden_layers"]}
    keys = []
    for k, v in hf.items():
        if not k.startswith("model.language_model.layers.") or v.ndim != 2:
            continue
        idx = int(k.split(".")[3])
        if _GLM_FP8.search(k) or (k.endswith("self_attn.o_proj.weight") and idx in dsa):
            keys.append(k)
    return keys


def _layer_types(tc):
    return tc.get("layer_types") or [
        "linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(tc["num_hidden_layers"])
    ]


def fp8_encode(w: mx.array, block=128):
    w = w.astype(mx.float32)
    rows, cols = w.shape
    pr, pc = (-rows) % block, (-cols) % block
    padded = mx.pad(w, ((0, pr), (0, pc)))
    blocks = padded.reshape((rows + pr) // block, block, (cols + pc) // block, block)
    scale = mx.maximum(mx.abs(blocks).max(axis=(1, 3)), 1e-8) / 448.0
    q = (blocks / scale[:, None, :, None]).reshape(rows + pr, cols + pc)[:rows, :cols]
    return mx.to_fp8(q), scale


# ----------------------------------------------------------------------------- io


def write_safetensors(path, tensors, dtypes=None):
    dtypes = dtypes or {}
    header, blobs, offset = {}, [], 0
    for k in sorted(tensors):
        v = tensors[k]
        mx.eval(v)
        if v.dtype == mx.bfloat16:
            data = np.array(v.view(mx.uint16)).tobytes()
        else:
            data = np.array(v).tobytes()
        dt = dtypes.get(k) or _DT[str(v.dtype).split(".")[-1]]
        header[k] = {"dtype": dt, "shape": list(v.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


def tokenizer_dir(family):
    d = TOKENIZERS / family
    if d.exists() and (d / "tokenizer.json").exists():
        return d
    from huggingface_hub import snapshot_download

    d.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        TOKENIZER_REPOS[family],
        local_dir=str(d),
        allow_patterns=["tokenizer*", "*.jinja", "vocab.json", "merges.txt", "generation_config.json",
                        "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
                        "special_tokens_map.json", "chat_template.json"],
    )
    return d


def sanitize_oracle(arch, model, mc, weights):
    weights = model.sanitize(dict(weights))
    if hasattr(getattr(arch, "VisionModel", None), "sanitize"):
        weights = arch.VisionModel(mc.vision_config).sanitize(weights)
    if hasattr(getattr(arch, "LanguageModel", None), "sanitize"):
        weights = arch.LanguageModel(mc.text_config).sanitize(weights)
    return weights


def make_checkpoint(family: str, out, seed: int = 0, n_shards: int = 2):
    """Write a tiny HF snapshot; returns (expected_main, expected_mtp) oracle MLX weights."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(CONFIGS[family])
    tc = cfg["text_config"]
    mx.random.seed(seed)
    arch, model, mc = instantiate(cfg)
    params = randomize(dict(tree_flatten(model.parameters())), seed)
    splitter, dmodel = drafter(family, cfg)
    mtp_params = randomize(dict(tree_flatten(dmodel.parameters())), seed + 1)
    if family == "qwen":
        hf = unsanitize_qwen(params, mtp_params)
    else:
        hf = unsanitize_glm(params, mtp_params, tc)
    hf = {k: (v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in hf.items()}
    for k in list(hf):
        if k.endswith(("e_score_correction_bias", "mlp.gate.weight")) and family == "glm":
            hf[k] = hf[k].astype(mx.float32)
        if k.endswith(("A_log", "dt_bias")) or re.search(r"\.hc_(attn|ffn)_", k):
            hf[k] = hf[k].astype(mx.float32)
    dtypes = {}
    dequant = dict(hf)
    if family == "glm":
        for k in glm_fp8_keys(hf, tc):
            q, s = fp8_encode(hf[k])
            hf[k] = q
            hf[k + "_scale_inv"] = s
            dtypes[k] = "F8_E4M3"
            dequant[k] = _dequantize_fp8_weight(q, s)
    keys = sorted(hf)
    shards = [keys[i::n_shards] for i in range(n_shards)]
    weight_map = {}
    for i, ks in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        write_safetensors(out / name, {k: hf[k] for k in ks}, dtypes)
        weight_map.update({k: name for k in ks})
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    for p in tokenizer_dir(family).iterdir():
        if p.is_file() and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, out / p.name)

    main_hf = {k: v for k, v in dequant.items() if not _is_mtp(k, family, tc)}
    expected = sanitize_oracle(arch, model, mc, main_hf)
    mtp_hf = {k: v for k, v in dequant.items() if _is_mtp(k, family, tc)}
    expected_mtp = splitter.transform(mtp_hf, tc, source_is_mlx=False)
    return expected, expected_mtp


def _is_mtp(key, family, tc):
    if family == "qwen":
        return key.startswith("mtp.")
    m = re.match(r"^model\.language_model\.layers\.(\d+)\.", key)
    return bool(m) and int(m.group(1)) >= tc["num_hidden_layers"]


# ----------------------------------------------------------------------------- calib stats


def make_calib(family: str, out, seed: int = 0, style: str = "json", with_mtp: bool = True, sens: bool = True):
    """Synthetic calib/ output in the PLAN.md format (several tolerated encodings)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIGS[family]
    tc = cfg["text_config"]
    E = tc["num_experts" if family == "qwen" else "n_routed_experts"]
    L = tc["num_hidden_layers"]
    moe_layers = list(range(L)) if family == "qwen" else list(range(tc["first_k_dense_replace"], L))
    rng = np.random.default_rng(seed)
    bits, groups = (2, 3, 4, 5, 6, 8), (32, 64)

    def grid(scale):
        return {f"{b},{g}": float(scale * 2.0 ** (-b) * (1 + 16 / g)) for b in bits for g in groups}

    def expert_stats():
        freq = rng.dirichlet(np.full(E, 0.7)) * 1000
        reap = rng.gamma(2.0, 1.0, E)
        dom = {d: (rng.dirichlet(np.full(E, 0.7)) * 250, rng.gamma(2.0, 1.0, E)) for d in ("code", "agentic", "reasoning", "longctx")}
        return freq, reap, dom

    for i in list(range(L)) + ([L] if with_mtp else []):
        is_mtp = i == L
        if not is_mtp and i not in moe_layers and not sens:
            continue
        obj = {}
        if is_mtp or i in moe_layers:
            freq, reap, dom = expert_stats()
            obj["expert_freq"] = freq.tolist()
            obj["expert_reap"] = reap.tolist()
            obj["domains"] = {d: {"expert_freq": f.tolist(), "expert_reap": r.tolist()} for d, (f, r) in dom.items()}
        if sens and not is_mtp:
            s = {}
            if family == "qwen":
                attn = ["linear_attn.in_proj_qkv", "linear_attn.out_proj"] if i < 3 else ["self_attn.q_proj", "self_attn.o_proj"]
                for n in attn:
                    s[n] = grid(rng.uniform(0.5, 2))
                s[f"model.language_model.layers.{i}.mlp.experts.gate_up_proj"] = grid(rng.uniform(0.5, 2))
                for e in range(E):
                    s[f"mlp.experts.{e}.down_proj.weight"] = grid(rng.uniform(0.5, 3))
                s["mlp.shared_expert.down_proj"] = grid(1.0)
            else:
                if i in moe_layers:
                    for e in range(E):
                        for p in ("gate_proj", "up_proj", "down_proj"):
                            s[f"mlp.experts.{e}.{p}"] = grid(rng.uniform(0.5, 3))
                    s["mlp.shared_experts.gate_proj"] = grid(1.0)
                else:
                    s["mlp.down_proj"] = grid(1.0)
                if _layer_types(tc)[i] == "linear_attention":
                    s["self_attn.q_proj"] = grid(1.2)
                    s["self_attn.o_proj"] = grid(1.1)
                else:
                    s["self_attn.kv_b_proj"] = grid(2.0)
                    s["language_model.model.layers.%d.self_attn.q_b_proj" % i] = grid(1.5)
            obj["sens"] = s
        name = ("mtp" if family == "qwen" else f"layer_{i:03d}") if is_mtp else f"layer_{i:03d}"
        if style == "npz" and not is_mtp:
            arrays = {}
            if "expert_freq" in obj:
                arrays["expert_freq"] = np.array(obj["expert_freq"])
                arrays["expert_reap"] = np.array(obj["expert_reap"])
                for d, v in obj["domains"].items():
                    arrays[f"expert_reap/{d}"] = np.array(v["expert_reap"])
                    arrays[f"expert_freq/{d}"] = np.array(v["expert_freq"])
            arrays["sens_bits"] = np.array(bits)
            arrays["sens_groups"] = np.array(groups)
            for n, g in obj.get("sens", {}).items():
                arrays[f"sens/{n}"] = np.array([[g[f"{b},{gg}"] for gg in groups] for b in bits])
            np.savez(out / f"{name}.npz", **arrays)
        else:
            with open(out / f"{name}.json", "w") as f:
                json.dump(obj, f)
    if sens:
        with open(out / "head.json", "w") as f:
            json.dump({"sens": {"lm_head": grid(3.0)}}, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "glm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-style", choices=["json", "npz"], default="json")
    args = ap.parse_args()
    make_checkpoint(args.model, args.out, args.seed)
    if args.calib:
        make_calib(args.model, args.calib, args.seed, args.calib_style)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
