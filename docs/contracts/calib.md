# calib/ contract

Layer-wise streaming calibration for `glm5_next` (GLM-5.3-Flash) and `qwen4_exp`
(Qwen3.8-Flash-Next). Consumed by `build/` (pruning and bit allocation) and `evals/kl` (`ref/`).

```
uv run python -m calib.run   --model {glm,qwen} --src <hf snapshot> --data artifacts/<m>/data --out artifacts/<m>/calib
uv run python -m calib.check --model {glm,qwen} --src <hf snapshot>   # header-only mapping check, no weights read
```

Useful flags: `--chunk 2048` (tokens per forward chunk), `--lin-tokens 2048` (sampled rows per dense
projection, split evenly over domains), `--expert-tokens 128` (sampled routed rows per expert),
`--mem-gb 40` (`mx.set_memory_limit`), `--layers N` (stop after N layers), `--restart`,
`--keep-hidden`, `--limit N`, `--dtype bfloat16`, `--store-dtype float16`.

## How it runs

1. The **embed** stage gathers embedding rows (memmap) and expands them to the hyper-connection
   streams: GLM `[L, 4, 4096]`, Qwen `[L, 4*2560]`. The result is written to `work/h_000/`.
2. For **each layer i**, the stage builds ONE real mlx-vlm decoder layer (`glm5_next.language.DecoderLayer`
   or `qwen4_exp.language.Qwen4ExpDecoderLayer`) and fills it from the source safetensors through the
   model's own `sanitize`. GLM FP8 e4m3 tensors are dequantized with their 128x128 `weight_scale_inv`
   to bf16. Routed experts are loaded one projection at a time to bound peak memory, and the Qwen PLE
   n-gram table (~100 GB) is row-gathered via memmap and never loaded. Every sample (calib and eval)
   is then pushed through the layer chunk by chunk with the layer's own cache, the same as the
   model's chunked prefill. The layer is freed afterwards.
   Probes record stats on **calib samples only**. Outputs go to `work/h_{i+1}/`, then the layer files
   are written, then `state.json` is updated, then `work/h_i/` is deleted.
3. The **head** stage applies the final norm (GLM `norm(mean over streams)`, Qwen `hyper_connection_mixer`)
   and then `lm_head`. It writes `ref/` for eval samples and `lm_head` sensitivity from calib samples.

The run resumes at **layer granularity**: rerunning the same command continues from `state.json`.
If `meta.json` mismatches (a different model, data, grid or dtype), the run stops; pass
`--restart` to start over. The GLM MTP layer (`layers.{num_hidden_layers}`) and the Qwen `mtp.*`
weights are not calibrated (build keeps them at a fixed precision).

Verified on the tiny fixtures: the streamed reference logprobs are **bit-identical** to a full
`mlx_vlm` model forward with the same weights, both single-pass and with the model's own chunked prefill.
`calib.check` passes for all 45 GLM-5.3-Flash and all 48 Qwen3.8-Flash-Next layers against the real
checkpoint headers.

## Output layout (`--out`)

```
meta.json            run settings, domains, sample/token counts, grid
state.json           {"next_layer": i} while running; {"next_layer": N, "done": true} when complete
embed.json/.npz      sensitivity of language_model.model.embed_tokens
layer_{i:03d}.json   per-layer summary (everything except per-expert / per-channel arrays)
layer_{i:03d}.npz    per-layer arrays
head.json/.npz       sensitivity of language_model.lm_head
ref/index.json, ref/{k:05d}.npz   reference logprobs for eval.jsonl
work/                hidden states between layers (transient; needs ~2 x tokens x hidden x 2 bytes of disk)
```

### Tensor names

Names in `layer_{i}.json/.npz` are **mlx module paths relative to the layer**, after mlx-vlm's sanitize.
The full path is `prefix + name`, where `prefix = "language_model.model.layers.{i}."` (given in the
JSON). This full path is what `nn.quantize(class_predicate=...)` sees on an `mlx_vlm`-loaded model.
Examples:

- GLM linear-attention layers: `self_attn.qkv_proj`, `self_attn.fbg_a_proj`, `self_attn.f_b_proj`,
  `self_attn.g_b_proj`, `self_attn.o_proj`.
- GLM DSA/MLA layers: `self_attn.qkv_a_proj`, `self_attn.q_b_proj`, `self_attn.o_proj`,
  `self_attn.embed_q`, `self_attn.unembed_out` (the last two are `MultiLinear`, `[H, out, in]`),
  `self_attn.indexer.{wq_b,wk,weights_proj}`.
- GLM MLPs: `mlp.gate_up_proj`, `mlp.down_proj` (dense layers 0-2), `mlp.shared_experts.{gate_up_proj,down_proj}`.
- Qwen: `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`, `self_attn.{q,k,v,o}_proj`,
  `self_attn.indexer.index_qk_proj`, `{attn,mlp}_hyper_connection.{input_mix_weight_down,input_mix_weight_up,block_inject_weight}`,
  `mlp.gate`, `mlp.shared_expert.{gate,up,down}_proj`, `mlp.shared_expert_gate`, `ple.{key_proj,value_proj}`,
  and `ple.ple_embedding.ngram_embedding` (an embedding table, layer 1 only).
- Routed experts (both models): `mlp.switch_mlp.{gate_proj,up_proj,down_proj}`, stacked `[E, out, in]`.

`embed.json` uses the full name `language_model.model.embed_tokens`, and `head.json` uses `language_model.lm_head`.

### Sensitivity

Grid: bits `{2,3,4,5,6,8}` x group size `{32,64}`, MLX **affine** `mx.quantize`/`mx.dequantize`.
The JSON key is `f"{bits}_{group}"`, e.g. `"4_64"`.

`sens = NMSE = Σ‖X·(Wq−W)ᵀ‖² / Σ‖X·Wᵀ‖²` (relative squared output error), computed on real
captured layer inputs `X`:
- dense linears use up to `--lin-tokens` input rows, sampled uniformly per domain (domain-balanced);
- `MultiLinear` uses its prefill usage. `embed_q` sees `latent @ W` and `unembed_out` sees `latent @ Wᵀ`;
- routed experts use up to `--expert-tokens` rows of tokens **routed to that expert**. `gate/up`
  inputs are the MoE input, and `down_proj` inputs are `activation(up, gate)` recomputed in full precision;
- embeddings (`embed_tokens`, Qwen PLE table) give weight-only NMSE over the rows actually looked up,
  weighted by lookup frequency.
`null` (NaN in npz) means the group size does not divide the input dimension, or there were no samples.

`layer_{i}.json["tensors"][name]`:
```json
{"kind": "linear|multilinear|switch|embedding", "shape": [...], "n_params": 123,
 "n_samples": 2048, "sens": {"2_32": 0.13, "2_64": 0.16, ..., "8_64": 6e-05},
 "out_sq": 1.7}
```
`out_sq` is the mean `‖X·Wᵀ‖²` per sampled row (dense only), which converts NMSE into absolute
error if wanted. For `switch`, `sens` is the `expert_freq`-weighted mean of the per-expert NMSE.
Per-expert values are in the npz.

### Arrays in `layer_{i}.npz`

| key | shape | meaning |
|---|---|---|
| `grid_bits`, `grid_groups` | `[6]`, `[2]` | grid axes |
| `sens.{name}` | `[6, 2]`, or `[E, 6, 2]` for switch | NMSE grid (NaN = n/a) |
| `out_sq.mlp.switch_mlp.{p}` | `[E]` | mean ‖y‖² per routed row |
| `act_absmean.{name}` | `[in]`, or `[E, in]` for switch | mean \|x\| per input channel (AWQ scaling), domain-balanced |
| `act_sqmean.{name}` | same | mean x² per input channel |
| `domain_tokens` | `[n_domains]` | calib tokens seen per domain (order = `domains` in the JSON) |
| `expert_count`, `expert_count.{d}` | `[E]` | routed (token, slot) assignments |
| `expert_freq`, `expert_freq.{d}` | `[E]` | `count / (tokens*top_k)`, sums to 1; overall = mean over domains |
| `expert_reap`, `expert_reap.{d}` | `[E]` | REAP: mean over tokens routed to e of `gate_weight * ‖expert_out‖₂` |
| `router_prob`, `router_prob.{d}` | `[E]` | mean router probability over all tokens |
| `expert_out_norm`, `expert_gate_mean` | `[E]` | mean ‖expert_out‖₂ and mean gate weight over routed tokens |

All entries use **domain-balanced** weighting: each domain counts equally regardless of its token
count, so long-context data cannot dominate. Overall REAP = `Σ_d (Σ g‖f‖)_d/T_d ÷ Σ_d count_d/T_d`.
`gate_weight` is the weight the model actually applies. For GLM this is the normalized sigmoid
top-k score × `routed_scaling_factor` (2.5). For Qwen it is the softmax top-k renormalized.
`router_prob` is sigmoid scores for GLM and softmax for Qwen. Experts that never route have REAP 0
and NaN sensitivity. `act_absmean.mlp.switch_mlp.down_proj` comes from the sampled rows. For
`gate/up` it is exact over all routed tokens.

`layer_{i}.json["moe"]` (null for dense layers) repeats `n_experts, top_k, expert_freq, expert_reap,
router_prob, expert_count, expert_samples` and adds `per_domain[d] = {expert_freq, expert_reap,
router_prob, expert_count}` as lists.

### `ref/` (for `evals/kl`)

`ref/index.json`:
`{"model", "topk": 64, "src", "samples": [{"id", "domain", "file", "n_tokens", "nll"}]}`. Samples
are in `eval.jsonl` order.

`ref/{k:05d}.npz` holds:
- `tokens` int32 `[L]`;
- `top_idx` int32 `[L, 64]` and `top_logprob` float32 `[L, 64]`, sorted descending. These are the
  full-precision log-softmax at position t, i.e. the distribution of token t+1;
- `target_logprob` float32 `[L]`, the logprob of the actual next token (NaN at the last position).

KL(ref‖model) on the top-64 support: `Σ_j p_j (top_logprob_j − model_logprob[top_idx_j])` with
`p = exp(top_logprob)`. The leftover mass `1 − Σp` can be treated as a single tail bucket.

## Fixtures for other modules

`tests/fixtures/make_tiny.py` writes tiny HF-format source checkpoints with the real config schema,
tensor names, dtypes and sharded index. GLM is FP8 e4m3 with F32 `weight_scale_inv` in the same places
as the real repo, includes MTP layer `layers.3`, and has 3 layers: `[linear+dense, linear+MoE, DSA+MoE]`
with 8 experts, top-2. Qwen is BF16 with fused `experts.gate_up_proj`, a sharded PLE table on layer 1,
`mtp.*`, and 3 layers: `[linear, linear+PLE, full]` with 8 experts, top-2. The vocabulary is 1024 and
there are no vision weights (`mlx_vlm.load_model` drops the tower). Both load strictly with
`mlx_vlm.utils.load_model`.

```python
from tests.fixtures.make_tiny import make_glm, make_qwen   # repo root on sys.path
make_glm(tmp / "glm"); make_qwen(tmp / "qwen", vocab_size=1024, text_overrides={...})
```
```
uv run python tests/fixtures/make_tiny.py --arch glm --out /tmp/tiny-glm [--vocab-size N --tokenizer-dir DIR]
```
