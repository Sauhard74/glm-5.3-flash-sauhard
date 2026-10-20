# build/ contract

```
uv run python -m build.make --model {glm,qwen} --src <hf snapshot> --calib artifacts/<model>/calib \
    --prune <none|fraction|keep-count|auto> --budget-gb 68 --out models/<name> [--dry-run]
```

Streams the source checkpoint one tensor (or one expert block) at a time, so peak memory stays
around 4–7 GB (measured on real-size GLM/Qwen shapes). It never instantiates real weights.
A `--dry-run` plans, validates and prints the allocation in seconds without writing anything.

## Inputs

### Source snapshot (`--src`)
A HF directory: `config.json`, `*.safetensors` (+ index) and tokenizer/processor files. Tensors are
read straight from file byte ranges. Block-FP8 (`F8_E4M3` + `<name>_scale_inv`, 128×128 blocks,
the GLM-5.3 layout) is dequantized to bf16 on read. For pruned expert tensors, only the kept
experts' rows are read.

### Calibration stats (`--calib`, written by calib/)
Every `*.json` / `*.npz` directly in the dir is read; `ref/` and other files are ignored.
The file name gives the block:

| file name                        | block                                   |
|----------------------------------|-----------------------------------------|
| `layer_{i:03d}.{json,npz}`       | decoder layer `i`                       |
| `layer_{num_hidden_layers:03d}.*`| MTP block (GLM numbering; also accepted for Qwen) |
| `mtp*.{json,npz}`                | MTP block                               |
| anything else (e.g. `head.json`) | global (names must then be absolute)    |

Accepted keys. Every encoding below is tolerated, and missing pieces fall back as described further down.

* Expert stats, length `E` (the **unpruned** expert count):
  `expert_freq` (counts or fractions, normalized internally) and `expert_reap` (REAP saliency =
  mean over routed tokens of `gate * ||expert_out||`). Aliases: `freq`, `expert_counts`, `reap`,
  `expert_saliency`.
  Per-domain values can be given as `expert_reap_code`, `expert_reap/code`, `expert_reap.code`,
  `{"expert_reap": {"all": [...], "code": [...]}}` or
  `{"domains": {"code": {"expert_reap": [...], "expert_freq": [...]}}}`. In npz use keys like
  `expert_reap/code`.
* Sensitivities: relative output error on real activations for each `(bits, group)`.
  * JSON: `"sens": {name: grid}`. A grid can be `{"4,64": v}`, `{"(4, 64)": v}`, `{"b4g64": v}`,
    `{"4": {"64": v}}`, `[{"bits":4,"group_size":64,"value":v}]` or `[[4,64,v]]`. The inverted
    form `"sens": {"4,64": {name: v}}` also works.
  * npz: either scalars `sens/<name>/<bits>_<group>`, or 2-D `sens/<name>` arrays together with
    the axis arrays `sens_bits` and `sens_groups`.
  * `name` can be an HF tensor name (`model.language_model.layers.3.mlp.experts.gate_up_proj`,
    `...experts.17.down_proj.weight`), a name relative to the layer in that file
    (`self_attn.q_proj`, `mlp.experts.17.down_proj`), or an mlx-vlm module path
    (`language_model.model.layers.3.mlp.switch_mlp.down_proj`). `.weight` is optional.
    HF tensors that mlx-vlm fuses (GLM `q/k/v_proj`→`qkv_proj`, `f_a/b/g_a`→`fbg_a_proj`,
    `q_a/kv_a_proj_with_mqa`→`qkv_a_proj`, `kv_b_proj`→`embed_q`+`unembed_out`,
    `gate/up_proj`→`gate_up_proj`; Qwen `experts.gate_up_proj`→`switch_mlp.gate_proj`+`up_proj`)
    are mapped to the fused module. Per-expert grids are averaged over the **kept** experts,
    weighted by `expert_freq`. Names that match nothing are reported in one warning.
* Activation channel magnitudes are accepted in the files but currently ignored (AWQ scaling is
  not implemented).

Fallbacks:
* No expert stats for a block: data-free saliency `||router_row_e|| * ||down_proj_e||_F`.
  This is the normal case for the MTP block if calib skips it.
* No sensitivity for a tensor: the median curve of measured tensors of the same kind (expert,
  attn, shared_expert, …). If that kind has none, a Gaussian quantization-error proxy scaled to the
  measured tensors is used. A measured tensor with missing cells is filled the same way. Each
  tensor's source is recorded as `sens_source`.

## What it does

1. **Expert pruning** (`--prune`): `none`; a fraction to remove (`0.375` → 512 keeps 320); an integer
   number of experts to keep per layer (`320`); or `auto`.
   * `--prune-mode uniform` (default): top-K experts by REAP score in every MoE layer. `--score
     freq|mass` changes the ranking criterion, and `--domain-weights code=2,agentic=2,...` mixes
     per-domain stats, weighting each domain by its token frequency.
   * `--prune-mode global`: keeps `K × n_layers` experts in total. Experts are ranked globally by
     their share of their own layer's score, and every layer keeps at least `--min-keep`·E (default
     0.25) and at least top_k experts.
   * Router rows (`mlp.gate.weight`), GLM `e_score_correction_bias` and any other router tensor
     whose leading dimension is E are sliced with the same sorted kept indices as the experts.
     Softmax (Qwen, `norm_topk_prob`) and sigmoid+bias (GLM) routing over the kept set is then
     exactly the original routing restricted to that set.
   * `auto`: tries every expert count on a grid (`--auto-step`). For each count it solves the bit
     allocation and keeps the count minimizing
     `Σ sens^p + λ Σ_layers prune_loss^p` (`--prune-lambda`, `--sens-power`).
     `prune_loss` is the fraction of the layer's expected routed output mass (`freq × reap`) that is
     removed. The full trace is in the summary. λ is **not calibrated**: sweep it with
     `--dry-run` and evals/kl before trusting `auto`.
   * MTP (`--mtp prune|intact|drop`): mlx-vlm runs the MTP head as a standalone drafter with its
     own `text_config`, so its MoE is independent of the main model. `prune` (default) keeps the
     same number of experts as the main layers (the mean count for global mode), ranked by MTP
     stats or the weight-saliency fallback.
2. **Bit allocation**: every module mlx-vlm can quantize (it has `to_quantized`, passes the
   model's `quant_predicate`, is not vision, and has a last dim divisible by a group size) gets one
   option among `--bits 2,3,4,5,6,8` × `--groups 32,64`, or bf16 (`--no-bf16` disables bf16).
   The solver minimizes `Σ weight·sens^p` (p=2 by default: independent errors add in energy;
   `--sens-power 1` gives the literal Σ sens) subject to exact bytes ≤ `--budget-gb`·1e9.
   * Size accounting is exact: packed uint32 + bf16 scales + bf16 biases for quantized tensors,
     `numel × itemsize` for everything else. That includes embeddings, lm_head, norms, the router,
     the vision tower (bf16, unless `--drop-vision`) and the MTP drafter. The written bytes are
     asserted to equal the prediction.
   * Solver: exact DP over 2^17 byte buckets, compared with a greedy convex-hull/Lagrangian pass,
     then leftover bytes are spent greedily. Both are checked against brute force in tests.
     Real-size plans take about 3 s.
   * Always high precision: routers (`mlp.gate`, Qwen `shared_expert_gate`) are bf16, and GLM's
     fp32 router weight and correction bias stay fp32. Norms, convs, A_log, dt_bias and
     hyper-connection params are never quantized.
     Floors: `lm_head` ≥ 6 bits, `embed_tokens` ≥ 4, every other tensor ≥ `--min-bits` (3).
     Override with `--floor REGEX=B`, `--no-default-floors` or `--pin REGEX=B[:G]` (16 = bf16).
   * The Qwen PLE n-gram table (≈51 B params) is pinned to 4-bit/g32 (`--ple-bits`). That keeps it
     compatible with mlx-vlm's mmap PLE storage (`ple_storage.prepare_external_ple_model`), which
     serves the table from page cache instead of wired memory. It counts in the budget (~32 GB).
   * MTP sensitivities are scaled by `--mtp-sens-weight` (0.25), because draft quality only affects
     acceptance rate.
3. **Write**: weights are already sanitized into mlx-vlm format, so loading them re-runs sanitize
   as a no-op. They go into 2 GB shards plus `model.safetensors.index.json`.

## Output directory (`--out`)

```
models/<name>/
  config.json                  source config with: expert count updated (num_experts | n_routed_experts),
                               FP8 quantization_config removed, and
                               "quantization" == "quantization_config" ==
                               {"group_size", "bits", "mode": "affine",          # most common choice
                                "<module path>": {"group_size", "bits", "mode"}  # every quantized module
                                "<module path>": false}                          # quantizable but kept bf16
  model*.safetensors, model.safetensors.index.json
  tokenizer*, vocab.json, chat_template.jinja, generation_config.json, *processor_config.json   (copied)
  pruned_moe.py                only when per-layer expert counts differ (see below)
  build_summary.json           the allocation report (below)
  mtp/                         MTP drafter (unless --mtp drop): config.json with model_type
                               qwen4_exp_mtp | glm5_next_mtp, its own text_config expert count and
                               quantization, model.safetensors, tokenizer files
```

* Load it with `mlx_vlm.load("models/<name>")`. Load the drafter with
  `mlx_vlm.speculative.drafters.load_drafter("models/<name>/mtp")`, which returns
  `(model, "mtp")`, or pass `--draft-model models/<name>/mtp --draft-kind mtp` to mlx-vlm generate.
* **Non-uniform expert counts** (`--prune-mode global`, or `auto` with global): mlx-vlm sizes every
  MoE block from a single `num_experts`. In that case config.json also has
  `"model_file": "pruned_moe.py"` and `"expert_counts": [per decoder layer, null for dense]`.
  `pruned_moe.py` subclasses the stock mlx-vlm architecture and resizes each layer's MoE block;
  mlx-vlm imports it through its checkpoint-local `model_file` mechanism.
  `config.model_type` is unchanged.
* With `--drop-vision`, no `vision_tower.*` tensors are written and mlx-vlm sets
  `model.vision_tower = None`. Text generation is unaffected.

### build_summary.json
```
model, model_type, source, calib, budget_bytes,
predicted_bytes {main, mtp}, predicted_total_bytes, written_bytes {main, mtp},
fixed_bytes, quantized_bytes, quantized_params, avg_bits_per_weight, objective, solver,
prune {arg, mode, experts_per_layer {layer|mtp: n}, total_prune_loss, auto_trace[],
       layers {layer|mtp: {kept, of, prune_loss, score_source, kept_ids[]}}},
mtp {mode, path: "mtp", experts} | null, drop_vision,
bits_histogram_units {"4b/g64": n, "bf16": n}, bits_histogram_params {...}, bytes_by_kind {"main:expert": B, ...},
allocation [{ns: main|mtp, path, kind, shape, bits (16 = bf16), group_size, bytes, cost, sens_source, pinned}],
timings_s
```
`kept_ids` are original expert indices in ascending order. Expert `j` of the built layer is
original expert `kept_ids[j]`. evals/serve can use this to map stats back to the source.

## Notes for the orchestrator
* Before a real run, do `--dry-run --summary-json plan.json`. It validates every planned tensor
  (name and shape) against the real mlx-vlm model class, and it fails fast if `calib/` names or the
  budget don't fit.
* Peak memory is dominated by one expert block: about 2× its bf16 size for Qwen (K=320: 6.1 GB)
  and about 2.7× for GLM FP8 (K=144: 6.3 GB).
