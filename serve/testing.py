"""Tiny random-initialized checkpoints of the real candidate architectures.

Used by tests/serve and smoke runs so the server's code paths (hybrid caches,
native tool templates, MTP heads) are exercised without loading real weights.

    uv run python -m serve.testing --model qwen --out /tmp/tiny-qwen
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

REPOS = {
    "qwen": "Qwen/Qwen3.8-Flash-Next",
    "glm": "zai-org/GLM-5.3-Flash",
}
META_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

QWEN_TEXT = {
    "hidden_size": 64,
    "num_hidden_layers": 8,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "hc_lowrank": 16,
    "indexer_n_heads": 2,
    "indexer_head_dim": 64,
    "indexer_budget": 64,
    "ple_embed_dim": 64,
    "ple_layer_ids": [2],
    "ngram_vocab_size_base": 4096,
    "split_ngram_parts": 4,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [3, 3, 2],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
}
QWEN_VISION = {
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "out_hidden_size": 64,
    "num_position_embeddings": 64,
}

GLM_TEXT = {
    "hidden_size": 64,
    "num_hidden_layers": 8,
    "layer_types": (["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * 2,
    "mlp_layer_types": ["dense"] + ["sparse"] * 7,
    "indexer_types": ["full"] * 8,
    "first_k_dense_replace": 1,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 32,
    "kv_lora_rank": 64,
    "qk_head_dim": 32,
    "qk_nope_head_dim": 32,
    "v_head_dim": 32,
    "index_n_heads": 2,
    "index_head_dim": 64,
    "index_topk": 64,
    "linear_attn_config": {
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "head_dim": 64,
        "short_conv_kernel_size": 4,
        "kda_layers": [0, 1, 2, 4, 5, 6],
        "full_attn_layers": [3, 7],
    },
}
GLM_VISION = {
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "out_hidden_size": 64,
    "projection_intermediate_size": 64,
}


def fetch_meta(model: str) -> Path:
    """Snapshot dir holding the real model's config and tokenizer files only."""
    from huggingface_hub import hf_hub_download

    path = None
    for name in META_FILES:
        try:
            path = hf_hub_download(REPOS[model], name, local_files_only=True)
        except Exception:
            path = hf_hub_download(REPOS[model], name)
    return Path(path).parent


def tiny_config(model: str, meta_dir: Path) -> dict:
    config = json.loads((meta_dir / "config.json").read_text())
    config.pop("quantization_config", None)
    text, vision = (QWEN_TEXT, QWEN_VISION) if model == "qwen" else (GLM_TEXT, GLM_VISION)
    config["text_config"].update(copy.deepcopy(text))
    config["vision_config"].update(copy.deepcopy(vision))
    return config


def _mtp_tensors(model: str, config: dict, splitter) -> dict:
    from mlx_vlm.utils import get_model_and_args

    draft_config = {
        "model_type": splitter.output_model_type,
        "text_config": copy.deepcopy(config["text_config"]),
    }
    arch, _ = get_model_and_args(draft_config)
    drafter = arch.Model(arch.ModelConfig.from_dict(draft_config))
    params = dict(tree_flatten(drafter.parameters()))
    if model == "qwen":
        return {f"mtp.{k}": v for k, v in params.items()}
    prefix = f"language_model.model.layers.{config['text_config']['num_hidden_layers']}."
    out = {}
    for key, value in params.items():
        if key.startswith("mtp_block."):
            key = key[len("mtp_block.") :]
        elif key.startswith("shared_head_norm."):
            key = "shared_head.norm." + key[len("shared_head_norm.") :]
        out[prefix + key] = value
    return out


def build_tiny(model: str, out: str, *, mtp: bool = True, seed: int = 0) -> Path:
    from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter
    from mlx_vlm.utils import get_model_and_args, update_module_configs

    mx.random.seed(seed)
    meta_dir = fetch_meta(model)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = tiny_config(model, meta_dir)

    arch, _ = get_model_and_args(copy.deepcopy(config))
    model_config = arch.ModelConfig.from_dict(copy.deepcopy(config))
    model_config = update_module_configs(
        model_config, arch, copy.deepcopy(config), ["text", "vision"]
    )
    net = arch.Model(model_config)
    # Small weights keep logits finite through the random hyper-connections.
    net.update(tree_map(lambda p: p * 0.5 if p.ndim > 1 else p, net.parameters()))
    weights = dict(tree_flatten(net.parameters()))
    if mtp:
        weights.update(_mtp_tensors(model, config, get_mtp_splitter(source_config=config)))
    weights = {k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v for k, v in weights.items()}
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights, metadata={"format": "mlx"})
    if not mtp:
        config["text_config"].pop("num_nextn_predict_layers", None)
        config["text_config"]["mtp_num_hidden_layers"] = 0
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for name in META_FILES[1:]:
        shutil.copy(meta_dir / name, out_dir / name)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(REPOS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-mtp", action="store_true")
    args = parser.parse_args()
    print(build_tiny(args.model, args.out, mtp=not args.no_mtp))


if __name__ == "__main__":
    main()
