"""Per-architecture glue: build ONE real mlx-vlm decoder layer, fill it from the source
snapshot through the model's own `sanitize`, and run it chunk-by-chunk with its own
cache. No attention/MoE math lives here — only wiring around the mlx-vlm classes."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.mla import MultiLinear
from mlx_vlm.models.switch_layers import SwitchGLU
from mlx_vlm.utils import get_model_and_args, load_config, update_module_configs

from .st import Snapshot

MODEL_TYPES = {"glm": "glm5_next", "qwen": "qwen4_exp"}
LAYER_PREFIX = "model.language_model.layers.{}."
MLX_LAYER_PREFIX = "language_model.model.layers.{}."


def load_model_config(src):
    config = load_config(Path(src))
    module, model_type = get_model_and_args(config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})
    model_config = module.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, module, config, ["text", "vision", "perceiver", "projector", "audio"]
    )
    return module, model_type, model_config


class Arch:
    name = ""

    def __init__(self, src, dtype=mx.bfloat16):
        self.src = Path(src)
        self.snap = Snapshot(src)
        self.module, self.model_type, self.model_config = load_model_config(src)
        if self.model_type != MODEL_TYPES[self.name]:
            raise ValueError(f"{src} is {self.model_type}, expected {MODEL_TYPES[self.name]}")
        self.cfg = self.model_config.text_config
        self.dtype = dtype
        self.load_dtype = mx.float32 if dtype == mx.float32 else mx.bfloat16
        self.n_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size

    # ---- weights
    def _cast(self, weights):
        if self.dtype == mx.bfloat16:
            return weights
        return {
            k: v.astype(self.dtype) if mx.issubdtype(v.dtype, mx.floating) else v
            for k, v in weights.items()
        }

    def _key_groups(self, keys):
        return [keys]

    def layer_source_keys(self, i):
        return self.snap.keys(LAYER_PREFIX.format(i))

    def load_layer(self, i) -> nn.Module:
        layer = self.build_layer(i)
        prefix = MLX_LAYER_PREFIX.format(i)
        weights = {}
        for group in self._key_groups(self.layer_source_keys(i)):
            src = self.snap.load_dequantized(group, dtype=self.load_dtype)
            sanitized = self._cast(self.sanitize(src, layer))
            del src
            for k, v in sanitized.items():
                if not k.startswith(prefix):
                    raise ValueError(f"unexpected key {k} while loading layer {i}")
                mx.eval(v)
                weights[k[len(prefix):]] = v
        layer.load_weights(list(weights.items()), strict=True)
        layer.eval()
        self.after_load(layer, i)
        return layer

    def after_load(self, layer, i):
        pass

    def quant_targets(self, layer, i) -> dict:
        """Quantizable dense projections of a layer by mlx module path (routed experts
        are handled as the MoE's stacked SwitchLinear tensors)."""
        pred = self.quant_predicate
        out = {}
        for name, m in layer.named_modules():
            if not name or not isinstance(m, (nn.Linear, MultiLinear)):
                continue
            if pred(MLX_LAYER_PREFIX.format(i) + name, m) is False:
                continue
            out[name] = m
        return out

    def moe(self, layer):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and isinstance(mlp.get("switch_mlp"), SwitchGLU):
            return mlp
        return None

    # ---- embedding / head
    def embed(self, tokens: np.ndarray) -> mx.array:
        key = "model.language_model.embed_tokens.weight"
        rows = self.snap.gather_rows(key, np.asarray(tokens, dtype=np.int64))
        return self.expand_streams(rows.astype(self.dtype) if self.dtype != mx.bfloat16 else rows)

    def load_head(self):
        tie = getattr(self.cfg, "tie_word_embeddings", False)
        key = "model.language_model.embed_tokens.weight" if tie else "lm_head.weight"
        w = self._cast(self.snap.load_dequantized([key], dtype=self.load_dtype))[key]
        lm_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
        lm_head.load_weights([("weight", w)])
        norm = self.build_final_norm()
        nkeys = self.final_norm_keys()
        loaded = self._cast(self.snap.load_dequantized(nkeys, dtype=self.load_dtype))
        norm.load_weights([(k[len(self.final_norm_prefix):], v) for k, v in loaded.items()])
        mx.eval(lm_head.parameters(), norm.parameters())
        norm.eval()
        return norm, lm_head

    def final_norm_keys(self):
        return self.snap.keys(self.final_norm_prefix)


class GLM(Arch):
    name = "glm"
    final_norm_prefix = "model.language_model.norm."

    def __init__(self, src, dtype=mx.bfloat16):
        super().__init__(src, dtype)
        from mlx_vlm.models.glm5_next import glm5_next, language

        self.lang = language
        self.top = glm5_next
        shared = [i for i, t in enumerate(self.cfg.indexer_types) if t == "shared"]
        if shared:
            raise NotImplementedError(f"shared-indexer layers {shared} are not supported")
        self.quant_predicate = language.LanguageModel.quant_predicate.fget(None)
        self.hc = self.cfg.hc_mult

    def hidden_shape(self, n):
        return (n, self.hc, self.hidden_size)

    def expand_streams(self, h):
        return mx.repeat(h[:, None], self.hc, axis=1)

    def build_layer(self, i):
        return self.lang.DecoderLayer(self.cfg, i)

    def _key_groups(self, keys):
        experts = {"gate_proj": [], "up_proj": [], "down_proj": []}
        rest = []
        for k in keys:
            if ".mlp.experts." in k:
                experts[k.split(".mlp.experts.", 1)[1].split(".")[1]].append(k)
            else:
                rest.append(k)
        return [rest] + [g for g in experts.values() if g]

    def sanitize(self, weights, layer):
        lm = SimpleNamespace(args=self.cfg, layers=[layer] * self.n_layers)
        fake = SimpleNamespace(
            language_model=SimpleNamespace(
                sanitize=lambda w: self.lang.LanguageModel.sanitize(lm, w)
            ),
            vision_tower=SimpleNamespace(sanitize=lambda w: w),
        )
        return self.top.Model.sanitize(fake, weights)

    def make_cache(self, layer):
        return self.lang.LanguageModel.make_cache(SimpleNamespace(layers=[layer]))[0]

    def forward(self, layer, h, tokens, cache):
        from mlx_vlm.models.base import create_ssm_mask

        mask = None
        if layer.block_type == "linear_attention":
            mask = create_ssm_mask(h[:, :, 0], cache)
        out, _ = layer(h, mask, cache, None)
        return out

    def router_probs(self, mlp, x):
        return mx.sigmoid(x.astype(mx.float32) @ mlp.gate.weight.T.astype(mx.float32))

    def moe_weights(self, mlp, probs, inds):
        raise RuntimeError("glm5_next passes routing weights to its SwitchGLU")

    def build_final_norm(self):
        return nn.RMSNorm(self.hidden_size, eps=self.cfg.rms_norm_eps)

    def head_input(self, norm, h):
        return norm(h.mean(axis=2))


class _PLERows(nn.Module):
    """Stands in for Qwen's 100 GB sharded n-gram table: same lookup, rows gathered
    from the checkpoint on demand through np.memmap."""

    def __init__(self, snap: Snapshot, keys: list[str]):
        super().__init__()
        object.__setattr__(self, "_snap", snap)
        object.__setattr__(self, "_keys", list(keys))
        sizes = np.array([snap.shape(k)[0] for k in keys], dtype=np.int64)
        object.__setattr__(self, "_offsets", np.concatenate([[0], np.cumsum(sizes)]))
        object.__setattr__(self, "dims", snap.shape(keys[0])[1])
        object.__setattr__(self, "on_rows", None)

    @property
    def row_count(self):
        return int(self._offsets[-1])

    def rows(self, flat: np.ndarray) -> np.ndarray:
        out = np.empty((flat.size, self.dims), dtype=np.uint16)
        shard = np.searchsorted(self._offsets, flat, side="right") - 1
        for s in np.unique(shard):
            sel = shard == s
            mm = self._snap.memmap(self._keys[s])
            out[sel] = mm[flat[sel] - self._offsets[s]]
        return out

    def __call__(self, indices):
        flat = np.array(indices.reshape(-1)).astype(np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self.row_count):
            raise IndexError("embedding index is outside the sharded vocabulary")
        if self.on_rows is not None:
            self.on_rows(flat)
        data = mx.array(self.rows(flat)).view(mx.bfloat16)
        return data.reshape(*indices.shape, self.dims)


class Qwen(Arch):
    name = "qwen"
    final_norm_prefix = "model.language_model.hyper_connection_mixer."
    PLE_TABLE = ".ple.ple_embedding.ngram_embedding."

    def __init__(self, src, dtype=mx.bfloat16):
        super().__init__(src, dtype)
        from mlx_vlm.models.qwen4_exp import language, qwen4_exp

        self.lang = language
        self.top = qwen4_exp
        self.quant_predicate = language.LanguageModel.quant_predicate.fget(None)
        self.hc = self.cfg.hc_count
        if self.cfg.ple_storage:
            raise NotImplementedError("ple_storage checkpoints are not HF sources")

    def hidden_shape(self, n):
        return (n, self.hc * self.hidden_size)

    def expand_streams(self, h):
        return mx.tile(h, (1, self.hc))

    def build_layer(self, i):
        layer = self.lang.Qwen4ExpDecoderLayer(self.cfg, i)
        if "ple" in layer:
            layer.ple.ple_embedding.ngram_embedding = _PLERows(self.snap, self.ple_keys(i))
        return layer

    def ple_keys(self, i):
        prefix = LAYER_PREFIX.format(i) + self.PLE_TABLE[1:]
        keys = self.snap.keys(prefix)
        return sorted(keys, key=lambda k: int(k[len(prefix):].split(".")[0].split("_")[1]))

    def layer_source_keys(self, i):
        return [k for k in super().layer_source_keys(i) if self.PLE_TABLE not in k]

    def sanitize(self, weights, layer):
        return self.top.Model.sanitize(SimpleNamespace(config=self.model_config), weights)

    def make_cache(self, layer):
        return self.lang.LanguageModel.make_cache(SimpleNamespace(layers=[layer]))[0]

    def forward(self, layer, h, tokens, cache):
        from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_ssm_mask

        if layer.is_linear:
            mask = _create_qwen3_5_ssm_mask(h, cache)
        else:
            mask = self.lang._create_qwen4_exp_attention_mask(h, cache)
        return layer(h, tokens, mask=mask, cache=cache, position_ids=None)

    def router_probs(self, mlp, x):
        return mx.softmax(x @ mlp.gate.weight.T, axis=-1, precise=True).astype(mx.float32)

    def moe_weights(self, mlp, probs, inds):
        scores = mx.take_along_axis(probs, inds, axis=-1)
        if mlp.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return scores

    def build_final_norm(self):
        return self.lang.Qwen4ExpGatedResidual(self.cfg, use_combine=False)

    def head_input(self, norm, h):
        return norm(h)


def make_arch(model: str, src, dtype=mx.bfloat16) -> Arch:
    return {"glm": GLM, "qwen": Qwen}[model](src, dtype)
