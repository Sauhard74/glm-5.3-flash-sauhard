"""Per-architecture knowledge: where experts/routers/MTP live in the HF checkpoint and
which mlx-vlm classes load the result."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class Family:
    name: str
    model_type: str
    experts_attr: str
    mtp_model_type: str
    vision_prefix: str = "model.visual."
    # (HF module suffix, ...) fused by mlx-vlm sanitize into one MLX module suffix.
    fusions: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = ()
    default_pins: Tuple[Tuple[str, int, Optional[int]], ...] = ()
    ple_marker: Optional[str] = None

    def layer_prefix(self, i: int) -> str:
        return f"model.language_model.layers.{i}."

    def mtp_layer_prefix(self, text_config: dict) -> Optional[str]:
        if self.name == "qwen":
            return "mtp.layers.0."
        return self.layer_prefix(int(text_config["num_hidden_layers"]))

    def is_mtp_key(self, key: str, text_config: dict) -> bool:
        if self.name == "qwen":
            return key.startswith(("mtp.", "model.mtp.", "language_model.mtp."))
        m = LAYER_RE.match(key)
        return bool(m) and int(m.group(1)) >= int(text_config["num_hidden_layers"])

    def layer_of(self, key: str) -> Optional[int]:
        m = LAYER_RE.match(key)
        return int(m.group(1)) if m else None

    def drafter(self):
        if self.name == "qwen":
            from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import (
                Qwen4ExpMTPSplitter,
            )

            return Qwen4ExpMTPSplitter()
        from mlx_vlm.speculative.drafters.glm5_next_mtp.split import (
            Glm5NextMTPSplitter,
        )

        return Glm5NextMTPSplitter()

    def mlx_prefix(self, hf_name: str) -> str:
        for old, new in (
            ("model.language_model.", "language_model.model."),
            ("model.visual.", "vision_tower."),
            ("lm_head", "language_model.lm_head"),
        ):
            if hf_name.startswith(old):
                return new + hf_name[len(old) :]
        return hf_name


ROUTER_PIN = (r"\.mlp\.gate$", 16, None)

QWEN = Family(
    name="qwen",
    model_type="qwen4_exp",
    experts_attr="num_experts",
    mtp_model_type="qwen4_exp_mtp",
    fusions=(
        (("mlp.experts.gate_up_proj",), ("mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj")),
        (("mlp.experts.down_proj",), ("mlp.switch_mlp.down_proj",)),
    ),
    default_pins=(
        ROUTER_PIN,
        (r"\.mlp\.shared_expert_gate$", 16, None),
    ),
    ple_marker=".ple.ple_embedding.ngram_embedding.shards.",
)

GLM = Family(
    name="glm",
    model_type="glm5_next",
    experts_attr="n_routed_experts",
    mtp_model_type="glm5_next_mtp",
    fusions=(
        (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.qkv_proj",)),
        (("self_attn.f_a_proj", "self_attn.b_proj", "self_attn.g_a_proj"), ("self_attn.fbg_a_proj",)),
        (("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"), ("self_attn.qkv_a_proj",)),
        (("self_attn.kv_b_proj",), ("self_attn.embed_q", "self_attn.unembed_out")),
        (("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"), ("mlp.shared_experts.gate_up_proj",)),
        (("mlp.gate_proj", "mlp.up_proj"), ("mlp.gate_up_proj",)),
    ),
    default_pins=(ROUTER_PIN,),
)

FAMILIES: Dict[str, Family] = {"qwen": QWEN, "glm": GLM}
BY_MODEL_TYPE = {f.model_type: f for f in FAMILIES.values()}


def family_for(name_or_type: str) -> Family:
    if name_or_type in FAMILIES:
        return FAMILIES[name_or_type]
    if name_or_type in BY_MODEL_TYPE:
        return BY_MODEL_TYPE[name_or_type]
    raise ValueError(f"unknown model family {name_or_type!r}; known: {sorted(FAMILIES)}")


EXPERT_PER_KEY_RE = re.compile(r"^(?P<p>.*\.)mlp\.experts\.(?P<e>\d+)\.(?P<proj>\w+)\.weight$")
EXPERT_FUSED_RE = re.compile(r"^(?P<p>.*\.)mlp\.experts\.(?P<proj>gate_up_proj|gate_proj|up_proj|down_proj)(?:\.weight)?$")


@dataclass
class MoELayer:
    """One routed-expert block in the source checkpoint."""

    name: str  # "3" for decoder layer 3, "mtp" for the MTP block
    prefix: str  # HF prefix ending with '.'
    num_experts: int
    router_keys: List[str] = field(default_factory=list)
    fused_keys: List[str] = field(default_factory=list)
    per_expert: Dict[str, Dict[int, str]] = field(default_factory=dict)  # proj -> e -> key

    @property
    def keys(self) -> List[str]:
        out = list(self.router_keys) + list(self.fused_keys)
        for experts in self.per_expert.values():
            out.extend(experts.values())
        return out


def find_moe_layers(keys, family: Family, text_config: dict) -> Dict[str, MoELayer]:
    """Locate router + expert tensors for every MoE block (decoder layers and MTP)."""
    num_experts = int(text_config[family.experts_attr])
    layers: Dict[str, MoELayer] = {}
    mtp_prefix = family.mtp_layer_prefix(text_config)

    def block(prefix):
        if prefix == mtp_prefix:
            name = "mtp"
        else:
            idx = family.layer_of(prefix)
            if idx is None:
                return None
            name = str(idx)
        if name not in layers:
            layers[name] = MoELayer(name, prefix, num_experts)
        return layers[name]

    for key in keys:
        m = EXPERT_PER_KEY_RE.match(key)
        if m:
            layer = block(m.group("p"))
            if layer is not None:
                layer.per_expert.setdefault(m.group("proj"), {})[int(m.group("e"))] = key
            continue
        m = EXPERT_FUSED_RE.match(key)
        if m:
            layer = block(m.group("p"))
            if layer is not None:
                layer.fused_keys.append(key)
    for layer in layers.values():
        gate_prefix = layer.prefix + "mlp.gate."
        layer.router_keys = sorted(k for k in keys if k.startswith(gate_prefix))
    return layers
