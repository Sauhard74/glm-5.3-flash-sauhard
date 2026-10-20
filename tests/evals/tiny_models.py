"""Tiny random MLX-format checkpoints of the real qwen4_exp / glm5_next classes,
in the layout build/ produces (mlx_vlm.load-able, native MTP weights in the same
safetensors), with the real tokenizers and chat templates.

    uv run python tests/evals/tiny_models.py --family qwen --out /tmp/tiny-qwen
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

REPOS = {"qwen": "Qwen/Qwen3.8-Flash-Next", "glm": "zai-org/GLM-5.3-Flash"}
META_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
SHARED_TOKENIZERS = Path(__file__).resolve().parents[1] / "fixtures" / "tokenizers"
CACHE = Path.home() / ".cache" / "local-ai" / "hf-meta"

QWEN_TEXT = dict(
    model_type="qwen4_exp_text",
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    rms_norm_eps=1e-6,
    max_position_embeddings=262144,
    full_attention_interval=4,
    hc_count=4,
    hc_lowrank=8,
    ple_embed_dim=64,
    ple_layer_ids=[],
    indexer_n_heads=2,
    indexer_head_dim=32,
    indexer_kv_heads=1,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    mtp_num_hidden_layers=1,
    mtp={"hybrid": True, "layer_types": ["full_attention"], "num_hidden_layers": 1, "rope_theta": 10000000},
    eos_token_id=248044,
    bos_token_id=248044,
    tie_word_embeddings=False,
    rope_parameters={
        "mrope_interleaved": True,
        "mrope_section": [3, 3, 2],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
)
QWEN = dict(
    model_type="qwen4_exp",
    architectures=["Qwen4ExpForConditionalGeneration"],
    image_token_id=248056,
    video_token_id=248057,
    vision_start_token_id=248053,
    vision_end_token_id=248054,
    tie_word_embeddings=False,
    eos_token_id=[248046, 248044],
    vision_config=dict(
        model_type="qwen4_exp",
        deepstack_visual_indexes=[],
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=2304,
    ),
)

GLM_TEXT = dict(
    model_type="glm5_next_text",
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=128,
    moe_intermediate_size=32,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    kv_lora_rank=32,
    q_lora_rank=32,
    qk_nope_head_dim=64,
    qk_rope_head_dim=0,
    qk_head_dim=64,
    v_head_dim=64,
    index_n_heads=2,
    index_head_dim=64,
    index_topk=2048,
    index_kpool=4,
    first_k_dense_replace=1,
    num_nextn_predict_layers=1,
    layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    indexer_types=["full"] * 4,
    linear_attn_config={
        "num_heads": 2,
        "head_dim": 64,
        "gate_lower_bound": -5.0,
        "short_conv_kernel_size": 4,
        "kda_layers": [0, 1, 2],
        "full_attn_layers": [3],
    },
    hc_mult=4,
    eos_token_id=[154820, 154827, 154829],
    pad_token_id=154820,
    max_position_embeddings=1048576,
)
GLM = dict(
    model_type="glm5_next",
    architectures=["Glm5NextForConditionalGeneration"],
    image_token_id=154854,
    video_token_id=154855,
    image_start_token_id=154830,
    image_end_token_id=154831,
    video_start_token_id=154832,
    video_end_token_id=154833,
    tie_word_embeddings=False,
    vision_config=dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        projection_intermediate_size=64,
    ),
)


def _meta_file(family: str, name: str) -> Path:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(REPOS[family], name)
    if isinstance(cached, str):
        return Path(cached)
    shared = SHARED_TOKENIZERS / family / name
    if shared.exists():
        return shared
    return Path(hf_hub_download(REPOS[family], name, local_dir=CACHE / family))


def tokenizer_len(family: str) -> int:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_meta_file(family, "tokenizer.json"))).get_vocab_size(with_added_tokens=True)


def _qwen_mtp(text_config: dict):
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import Qwen4ExpMTPDraftModel

    d = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig.from_dict({"text_config": copy.deepcopy(text_config)}))
    d.set_dtype(mx.bfloat16)
    return {f"mtp.{k}": v for k, v in tree_flatten(d.parameters())}


def _glm_mtp(text_config: dict):
    from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
    from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel

    d = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict({"text_config": copy.deepcopy(text_config)}))
    d.set_dtype(mx.bfloat16)
    layer = text_config["num_hidden_layers"]
    out = {}
    for k, v in tree_flatten(d.parameters()):
        if k.startswith("mtp_block."):
            k = k[len("mtp_block.") :]
        elif k.startswith("shared_head_norm."):
            k = "shared_head.norm." + k[len("shared_head_norm.") :]
        out[f"language_model.model.layers.{layer}.{k}"] = v
    return out


def build_tiny(family: str, out: Path, mtp: bool = True, seed: int = 0) -> Path:
    from mlx_vlm.utils import get_model_and_args

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text = copy.deepcopy(QWEN_TEXT if family == "qwen" else GLM_TEXT)
    text["vocab_size"] = tokenizer_len(family)
    cfg = copy.deepcopy(QWEN if family == "qwen" else GLM)
    cfg["text_config"] = text
    mx.random.seed(seed)
    model_class, _ = get_model_and_args(cfg)
    model = model_class.Model(model_class.ModelConfig.from_dict(copy.deepcopy(cfg)))
    model.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    if mtp:
        weights.update(_qwen_mtp(text) if family == "qwen" else _glm_mtp(text))
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    for name in META_FILES:
        shutil.copy(_meta_file(family, name), out / name)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen", "glm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-mtp", action="store_true")
    a = ap.parse_args()
    print(build_tiny(a.family, Path(a.out), mtp=not a.no_mtp))
