# Decode research: speculative decoding, KV cache, long context and servers on MLX

Scope: `qwen4_exp` (Qwen3.8-Flash-Next) and `glm5_next` (GLM-5.3-Flash) in **mlx-vlm 0.7.9**, MLX 0.32.4, mlx-lm 0.32.2,
on an M5 Max with 128 GB. Date: 2026-10-20.

How this was checked. Every claim about mlx-vlm comes from the installed source in
`.venv/lib/python3.12/site-packages/mlx_vlm`. The memory numbers and behaviour come from tiny random-init models built
from the real `config.json`. The real cache dimensions were kept (heads, head dims, latent rank, indexer dims), and
layers, experts and vocab were shrunk. Those runs used less than 0.4B params and peaked under 8 GB.
Numbers measured that way are marked **[measured]**. Numbers a vendor or author published are marked **[self-reported]**.

---

## 1. MTP speculative decoding

### Status in mlx-vlm 0.7.9

| | qwen4_exp | glm5_next |
|---|---|---|
| Drafter class | `speculative/drafters/qwen4_exp_mtp` (`Qwen4ExpMTPDraftModel`) | `speculative/drafters/glm5_next_mtp` (`Glm5NextMTPDraftModel`) |
| Weights it reads | `mtp.*` / `model.mtp.*` keys | decoder layer `num_hidden_layers` (layer 45). `glm5_next` `sanitize()` drops it from the main model |
| Drafter layer / cache | one `qwen_sparse_attention` layer with a `QSAKVCache` | one DSA layer with a `CacheList(KVCache, KVCache, PoolingCache, KVCache)` |
| Default runtime block | Ceiling 4 (up to 3 drafts). Both drafters start at the native block 2 (1 draft) and grow to the ceiling only when ≥65% of the last 32 rounds hit the native depth (`_effective_mtp_block_size`, after 8 rounds) | Ceiling 3 (up to 2 drafts), same adaptive policy (`prefer_requested_block_size = False`); 2 for batched sampling |
| Rollback | `rollback_speculative_cache` / `speculative_verify_hidden` (batch-invariant verifier) | generic cache transactions (`start_speculative_cache`). It has no `rollback_speculative_cache` |
| Chunked prefill with MTP | yes (`chunked_prefill_policy` returns True when `return_hidden` and `return_shared_kv` are set) | yes (same policy) |

**How to turn it on.** Both families are in the `MTP_SPLITTERS` registry (`speculative/drafters/mtp_split.py`). That
means a native checkpoint can be passed straight to `--draft-model`. Only the drafter tensors are loaded, through the
safetensors index. This came in with PR [#2457](https://github.com/Blaizzy/mlx-vlm/pull/2457), merged 2026-10-12.

```bash
uv run python -m mlx_vlm.server  --model models/<name> --draft-model models/<name> --draft-kind mtp [--draft-block-size N]
uv run python -m mlx_vlm.generate --model models/<name> --draft-model models/<name> --draft-kind mtp --prompt ...
# or extract a standalone drafter (optionally quantized):
uv run python -m mlx_vlm.split_mtp --model <src> --output <dir> [--q-bits 8 --q-mode affine]
```

You can also set it with env vars (`MLX_VLM_DRAFT_MODEL`, `MLX_VLM_DRAFT_KIND`, `MLX_VLM_DRAFT_BLOCK_SIZE`) or at runtime
with `PATCH /v1/settings`.

**Contract for `build/`.** Keep the MTP tensors in the output directory under the source names. For Qwen that is
`mtp.*`. For GLM it is `model.layers.45.*` (or `model.language_model.layers.45.*`). The splitter selects keys by name.
Open issue [#2526](https://github.com/Blaizzy/mlx-vlm/issues/2526) reports that `convert --mtp` drops `q_mode`, which
silently pairs a non-affine main model with an affine drafter. If `build/` uses mxfp4/nvfp4, quantize the drafter in
the same mode.

**Mixed precision breaks bundled-drafter loading [measured].** The splitter renames the MTP layer to drafter paths
(`mtp_block.*`, `eh_proj`, `shared_head_norm`), but it copies the source `quantization` dict unchanged. No drafter
defines `quantization_path_aliases`. A per-tensor override keyed by the source path
(`language_model.model.layers.45.mlp...`) therefore never matches, and the tensor is loaded at the default bits. On a
tiny GLM checkpoint with one 8-bit MTP tensor, `load_drafter` failed with
`ValueError: Expected shape (512, 16) but received shape (512, 32)`. The same override keyed by the drafter path
(`mtp_block.mlp.shared_experts.down_proj`) loaded correctly (`verify/mtp_mixed_quant_probe.py`). Qwen goes through the
same `quantization_from_source` code. `build/` must quantize the MTP layer at the config's default bits, group size
and mode, or also write drafter-path keys into `quantization`. Main-model modules never have those paths, so the extra
keys don't affect the main model.

**Exactness [measured].** We ran greedy decode on tiny real-architecture models, 48 tokens, plain against MTP
(`probe/mtp_probe.py`). In float32 the outputs were identical for both models. *Verification note:* with random
weights every draft was rejected (`accept_lens` all 0), so that run only exercised the reject path. We reran with
weights rigged so the drafter agrees with the target (branch output projections ×0.05, `eh_proj = [I, 0]`;
`verify/mtp_accept_probe.py`). GLM accepted drafts in 31% of rounds and still matched plain decode over 48 distinct
tokens. The same rig did not produce any accepts on Qwen, so **Qwen's accept/commit path is still untested on tiny
models**. In bfloat16 they diverged at token 4
(Qwen) and token 17 (GLM). With random weights, near-ties plus a verify kernel whose numerics depend on batch shape
explain this. Treat MTP output as distribution-exact but not bit-exact in bf16. The mlx-vlm authors report that on the
real GLM FP8, 12 out of 12 comparisons of 64 tokens matched AR exactly at temperatures 0 and 0.8
([PR #2206](https://github.com/Blaizzy/mlx-vlm/pull/2206), still open) **[self-reported]**.

**Expected speedup.** Nobody has published an mlx-vlm number for these two models. MTPLX reports the following for
Qwen3.8-Flash-Next on an M5 Max with 128 GB, using native MTP at depth 3
([model card](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed)) **[self-reported]**:

| Workload | MTP | Plain | Speedup |
|---|---|---|---|
| Coding task | 73.5 tok/s | 43.8 tok/s | 1.7× |
| One OpenCode request | 125.8 tok/s | | |
| 65K-token prompt | 63.5 tok/s | | |
| 109K-token turn | 61.8 tok/s | | |
| 200K-token turn | 50.3 tok/s | | |

For Qwen3.8-27B on an H200, native MTP accepts 3.7–5.0 tokens per step
([z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) **[self-reported]**.

**Speculative-round memory [computed].** Linear-attention layers keep their intermediate recurrent states for rollback
(`ArraysCache.update_recurrent`, controlled by `history_capacity`). That costs about (block−1) × layers × state:

- Qwen: 3 × 36 × 3.06 MiB ≈ 0.33 GiB
- GLM: 2 × 34 × 4.14 MiB ≈ 0.28 GiB

Both are transient.

### DFlash / DFlash2 / DSpark / MTPLX

| Name | What it is | Usable for our models on MLX? |
|---|---|---|
| **DFlash** | Block-diffusion drafter from z-lab. One forward pass drafts a block of up to 8 tokens, conditioned on hidden states from several target layers. Paper: "DFlash: Block Diffusion for Flash Speculative Decoding" (ICML 2026); code: [z-lab/dflash](https://github.com/z-lab/dflash). | Its loader in mlx-vlm is `qwen3_dflash` (`--draft-kind dflash`). |
| **DFlash2** | DFlash plus a top-16 path selector and two-tap dynamic convolutions ("suffix decay" fix). About 1.3% more latency, 16–25% more tokens per verify ([inco.ai blog](https://inco.ai/blog/dflash2/)). For Qwen3.8-27B on an H200 at concurrency 1: τ 4.1–5.5 and **2.7–3.4×** over AR **[self-reported]**. Engines: SGLang, vLLM PR 52816, llama.cpp PR 27342, Ollama PR, oMLX 0.6.2. | mlx-vlm has `drafters/dflash2`, and `utils.get_model_type` routes `architectures: DFlash2DraftModel` to it. **Qwen:** `incoai/Qwen3.8-Flash-Next-DFlash2` exists (hidden 2560, 48 target layers, targets `[5,15,25,35,45]`). `qwen4_exp` supports `capture_layer_ids`, `rollback_speculative_cache` and the dflash chunked-prefill policy, so it is **structurally compatible but untested**. Its config matches the target: hidden 2560, 48 target layers,
vocab 248320. It has 0 downloads, but the repo was only created on 2026-10-19, so that count says little. It has no
published acceptance numbers, and its training target was the unpruned model. **GLM:** `incoai/GLM-5.3-Flash-DFlash2` and `modal-labs/GLM-5.3-Flash-DFlash` exist, but **this mlx-vlm cannot run them**. `validate_dflash_target` requires `rollback_speculative_cache`, which `glm5_next.LanguageModel` lacks [measured], and `Glm5NextTextModel` only emits the final hidden state. It cannot capture per-layer hidden states. |
| **DSpark** | DFlash-style drafter with a confidence head and a low-rank Markov head, trained with SpecForge and served with SGLang `--speculative-algorithm DSPARK` ([RadixArk](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)). ~~On an H200: 2.7–3.2× at c=1~~ On four GB300 GPUs (DP4 × TP1), DSpark v2 gives 2.25–3.16× at c=1, according to its
own card **[self-reported]**. Its acceptance was below both MTP and DFlash2 in z-lab's table. | mlx-vlm has `drafters/dspark` and `qwen3_dspark`, routed through `dflash_config.projector_type == "dspark"`. Qwen candidate: `shadowpa0327/Qwen3.8-Flash-Next-DSpark` (0 downloads). The GLM one (`RedHatAI/...speculator.dspark-preview`) is in speculators format. GLM is blocked for the same reasons as DFlash2. |
| **MTPLX** | Not a drafter. It is a separate MLX-native runtime (`pip install mtplx`, Apache-2.0, [github.com/youssofal/MTPLX](https://github.com/youssofal/MTPLX)) that runs the model's own MTP heads. It includes OpenAI and Anthropic `/v1/messages` endpoints, a warm-prefix session bank and an SSD session cache. HF repos named `*-MTPLX-*` are model packs in its format. It supports Qwen 3.5/3.6/3.8 (including Flash-Next) and Gemma 4. **GLM is not listed.** | It is an alternative server for Qwen only. Its Flash-Next pack needs about 83 GB resident, peaks at **87 GiB** serving, and streams a 32 GB n-gram table from SSD [self-reported]. That is above our 80 GB process cap, so it would need our own pruned weights repacked. |

---

## 2. KV-cache quantization

The flags are `--kv-bits`, `--kv-group-size`, `--kv-quant-scheme {uniform,turboquant}`, split K/V bits, and
`--quantized-kv-start` (default 5000). Results **[measured]** on tiny models with real cache dims
(`probe/kvq_probe.py`, `probe/batch_kvq_probe.py`):

| Path | qwen4_exp | glm5_next |
|---|---|---|
| Single-stream (`maybe_quantize_kv_cache`, used by `mlx_vlm.generate`), uniform 8/4-bit | **Works.** `QSAKVCache.to_quantized` gives `QSAQuantizedKVCache`. The indexer keys stay bf16. The QSA layer drops from 2372 to about 900 B/token at 4-bit. KL on the tiny model was 5e-7. | **Silent no-op.** `CacheList` has no `to_quantized`. |
| Single-stream, TurboQuant (fractional bits) | **Silent no-op.** `QSAKVCache.preserve_auxiliary_kv_state = True`, so it is skipped. | **Crash** (`[reshape] Cannot infer the shape of an empty array`). The MLA latent `KVCache` stores zero-width values. |
| Server (`BatchGenerator` → `_make_cache`), any `--kv-bits` | **Crash:** `NotImplementedError: QSAKVCache does not support quantized continuous batching`. | **Crash:** `Only real floating types can be quantized but w has type bool` (the indexer validity-flag cache). |

The last attention layer is never quantized (`cache.should_quantize_kv_layer`). A probe whose only attention layer is
the last one will therefore report "OK" or a no-op. The crashes above reproduce only when an attention layer sits
earlier in the stack, which is the case for every QSA/DSA layer except the last in the real models.

Why it matters less on these architectures: per-token KV lives only in the 12 out of 48 (Qwen) or 11 out of 45 (GLM)
full/sparse-attention layers. The recurrent state is fixed-size and fp32 and never quantized. The GLM MLA latent is
already 1 KiB per token per layer. Quantization noise on QSA/DSA keys would also feed the sparse top-k selection.
Conclusion: **do not use `--kv-bits` with the stock server.** It is not needed to meet the 8 GB budget at 128K once
the GLM issue below is fixed.

---

## 3. Exact KV + state memory (batch 1)

The `config.json` files come from `zai-org/GLM-5.3-Flash` and `Qwen/Qwen3.8-Flash-Next`. Byte counts per layer are
**[measured]** with `cache_nbytes` on real-dim tiny models at 2K, 8K, 32K and 40K tokens (`probe/cache_probe.py`) and
extrapolated linearly. They match the analytic sizes exactly.

**Qwen3.8-Flash-Next** has 48 layers: 36 GatedDeltaNet and 12 QSA. QSA uses 2 KV heads × 256 dims plus an indexer of
1 × 128 with compression ratio 4.

- Each linear layer holds 3.06 MiB: an fp32 state (48×128×128) plus a bf16 conv window (3 × 10240). PLE layer 2 adds
  0.18 MiB. Fixed total: **110 MiB**.
- Each QSA layer costs **2372 B/token**: K+V 2048, indexer key 256, block key 64, position 4.
- The MTP layer adds one more QSA layer.

**GLM-5.3-Flash** has 45 layers: 34 KDA and 11 DSA/MLA (`kv_lora_rank` 512, NoPE, `index_kpool` 4).

- Each KDA layer holds 4.14 MiB: an fp32 state (64×128×128) plus a conv window (3 × 24576). Fixed total: **141 MiB**.
- Each DSA layer costs **1089 B/token** for the essential state: latent 1024, valid flag 1, pooled indexer key 64.
- Each DSA layer also has a **projected K/V side cache** (`cache[3]`) of 64 heads × (256+256) × bf16 = **64 KiB/token**.
  It fills during prefill until the context reaches `_MAX_PROJECTED_PREFILL_TOKENS = 32768`. After that it stops
  growing, but **it is never freed**.

| Context | Qwen | Qwen + MTP | Qwen, 4-bit KV (generate only) | GLM essential | GLM essential + MTP | **GLM as shipped (with projected cache)** + MTP |
|---|---|---|---|---|---|---|
| 32K | 0.98 GiB | 1.05 GiB | 0.44 GiB | 0.50 GiB | 0.53 GiB | **24.5 GiB** |
| 128K | 3.58 GiB | **3.87 GiB** | 1.43 GiB | 1.60 GiB | **1.73 GiB** | **25.7 GiB** |
| 256K (Qwen max is 262,144) | 7.06 GiB | 7.64 GiB | 2.74 GiB | 3.06 GiB | 3.32 GiB | **27.3 GiB** |

**The GLM problem [measured].**

- At 32K the DSA cache was 2082 MiB per layer, and 2048 MiB of that was the projected cache. At 40K it was still
  2048 MiB, so the projected part stops growing at 32K but is not released.
- **The fix is bit-exact.** Replacing `cache[i].caches[3]` with a fresh `KVCache()` right after prefill gave
  max|Δlogit| = 0 over 4 decode steps plus a follow-up 512-token prefill turn (`probe/glm_proj_probe.py`). Setting
  `_MAX_PROJECTED_PREFILL_TOKENS = 0` or `4096` was also bit-exact. Decode (≤ 8 tokens) reads only the latent. A
  later prefill falls back to recomputing from the latent when `projected_cache.size() != previous_length`.
  Decode never appends to the projected cache, so after the first decoded token it can never match again. It is
  dead weight for the rest of the request.
- **`_MAX_PROJECTED_PREFILL_TOKENS = 0` is enough on its own [measured].** The projected cache fills only while
  `latent.shape[2] <= bound`. With 0 it never fills: the DSA cache was 46 MiB at 32K with no projected part, so no
  cache surgery is needed. The module global is read at call time, so this also covers the MTP drafter, which reuses
  `Glm5NextAttention`. Prefill speed on the tiny model at 32K was 8.5K tok/s with bound 0 against 8.1–8.9K tok/s at
  the default, which is within noise. On the real model the extra `embed_q`/`unembed_out` recompute per chunk is about
  2·64·512·256·2 FLOP per latent token per DSA layer. That works out to roughly 8% more prefill FLOPs at 32K;
  `evals.speed` should confirm. With bound 4096, a prompt of 4096 tokens or fewer still leaves 256 MiB per layer
  behind, so that setting needs the `caches[3]` drop as well.
- **The transient is a separate issue.** ~~Once the context passes the bound, every prefill chunk recomputes
  `embed_q` / `unembed_out` over the *entire* latent. That costs 64 KiB × context per DSA layer while the layer runs:
  **2 GiB at 32K, 8 GiB at 128K, 16 GiB at 256K**. This is a prefill-time peak independent of the chunk size.~~
  *Corrected [measured].* Past the bound, each chunk does recompute K/V over the whole latent, which is 64 KiB × context
  of bf16. But the measured last-chunk peak is larger than that and grows with the prefill step. It is about the same
  when the projected cache is used instead of recomputed (bound = ∞), so most of it is not the recompute.
  `verify/glm_peak_scale.py`, one DSA layer with real dims, peak above weights+cache:

  | step | 16K | 32K | 48K | slope 16K→48K |
  |---|---|---|---|---|
  | 512, bound 0 | 1.55 GiB | 2.88 GiB | 4.14 GiB | ~85 KiB/token |
  | 2048, bound 0 | 2.38 GiB | 4.59 GiB | 7.10 GiB | ~150 KiB/token |
  | 2048, bound ∞ | 2.53 GiB | 5.07 GiB | 5.42 GiB | |

  Linear extrapolation to 128K gives about **10–11 GiB at step 512 and 18–19 GiB at step 2048** for one layer. The
  transient does not stack across DSA layers: two DSA layers at 16K gave 2.44 GiB against 2.38 GiB for one. With 68 GB
  of weights under an 80 GB cap, a 128K GLM prefill at step 2048 would probably exceed the cap. Real-model
  `evals.speed` must measure it.

---

## 4. Chunked prefill

The server flag is `--prefill-step-size` (default 2048). Generate has the same flag. **[measured]** peak memory above
baseline for tiny models at 8K tokens:

| Model | 512 | 2048 | 4096 |
|---|---|---|---|
| Qwen | ~~0.71 GB~~ 1023 MiB | 1412 MiB | 1983 MiB |
| GLM | 1705 MiB | 2321 MiB | ~~2.9 GB~~ 2707 MiB |

(Re-measured in verification, which was repeatable; the original "GB" values were MiB/1000.) These include the
cache, and for GLM that includes the 512 MiB projected cache at the default bound. At long context the GLM DSA
transient grows with step × context (§3), so GLM long prompts favour 512–1024. The real model adds MoE activation memory proportional to the step. Tiny-model throughput is
not representative. Real-model `evals.speed` must decide between 2048 and 4096.

- **Hidden-state drafters disable chunking on GLM.** `glm5_next.chunked_prefill_policy` returns False for
  `draft_kind="dflash"`/`"eagle3"`. That would mean a single-shot prefill, but it is moot because DFlash cannot run on
  GLM.
- **Always call `model.eval()`.** In training mode GLM's KDA turns off its kernel (`use_kernel=not self.training`) and
  DSA leaves the decode path. ~~We measured 2× the memory and 5–10× slower.~~ Re-measured at 4K tokens, step 2048
  (`verify/train_mode_probe.py`): GLM peak went from 1594 to 9460 MiB (5.9×) and time from 0.37 to 1.71 s (4.6×). Qwen
  went from 1382 to 1405 MiB (about the same) and 0.36 to 0.55 s (1.5×). `mlx_vlm.load` handles this, but hand-built
  test models must call it.

---

## 5. Prompt/prefix caching for agentic loops

**mlx-vlm APC** (`apc.py`). It is **off by default**; enable it with `APC_ENABLED=1`.

- For both models the plan is `strategy=checkpoint` (`model_apc_plan`, [measured]). Recurrent state cannot be paged,
  so APC snapshots the **whole cache** at a common prefix boundary instead of hashing 16-token blocks.
- Knobs:

  | Knob | Default |
  |---|---|
  | `APC_CHECKPOINT_ENTRIES` | 2 |
  | `APC_CHECKPOINT_INTERVAL_TOKENS` | 2048 |
  | `APC_MEMORY_MAX_GB` | min(8 GB, working set/10), which is 8 GB here |
  | `APC_MEMORY_RESERVE_GB` | max(1 GiB, working set/10), which is 10.75 GiB here |
  | Disk tier | on, `APC_DISK_MAX_GB=20` at `~/.cache/mlx-vlm/apc` (`APC_DISK_ENABLED`, `APC_DISK_PATH`) |
  | `APC_DISK_MIN_FREE_RAM_GB` | 2 |

- `/v1/cache/stats` and `/v1/cache/reset` expose it.
- **Fit for Claude Code / OpenCode.** Each turn re-sends the system prompt, tools and full history, so the new prompt
  extends the previous one. The exact-prefix checkpoint is a hit as long as the client keeps the prefix byte-stable.
  Claude Code made its attribution block stable per conversation for custom base URLs in v2.1.181. Set
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` if you have doubts.
- **Each checkpoint is a full cache copy.** With Qwen and MTP at 128K that is 3.9 GiB. On GLM as shipped it is about
  25 GiB, because the projected cache gets snapshotted too. That is another reason to drop it. A checkpoint larger
  than `APC_MEMORY_MAX_GB` is not kept in memory. It is written to the disk tier instead, so on GLM as shipped that
  means 25 GiB of SSD writes per checkpoint (`apc.py`, `size <= self.memory_max_bytes`).
- **APC memory adds to the KV budget.** Up to `APC_MEMORY_MAX_GB` (8 GiB) of checkpoints sit on top of the live
  KV+state, so Qwen+MTP at 128K can use about 3.9 + 7.8 GiB. Count this in the 80 GB process budget.
- **MTP and APC.** In 0.7.9 they share `PromptProcessingBatch`. Open PR [#2206](https://github.com/Blaizzy/mlx-vlm/pull/2206)
  rebuilds speculative decoding so that APC stores target caches only. That implies the interaction is still in flux.
  Validate warm-prefix plus MTP in `evals` before relying on it.

**mlx-lm server** has an LRU prompt cache (`--prompt-cache-size`, `--prompt-cache-bytes`). It cannot load `qwen4_exp`
or `glm5_next` (no model files in mlx-lm 0.32.2) and has no Anthropic endpoint, so it is not an option.

---

## 6. Servers and coding-agent clients (2026)

| Server | Runs our architectures? | API | Notes |
|---|---|---|---|
| **mlx-vlm server** (installed) | yes | `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models` | Has continuous batching, MTP, APC and thinking budgets. `AnthropicRequest` has `extra="allow"` and accepts `thinking` and `output_config`. **Has no memory-cap flag.** `BatchGenerator` enters `wired_limit()`, which sets the wired limit to `max_recommended_working_set_size` = **107.5 GiB** on this machine. It also **never emits SSE `ping` events**: the
`/v1/messages` stream emits only message_start, content_block_*, message_delta, message_stop and error. |
| mlx_lm.server | no | OpenAI only | |
| **MTPLX** | Qwen3.8-Flash-Next only | OpenAI + Anthropic | Native MTP and an SSD session cache. Its stock pack peaks at 87 GiB, above budget. |
| **oMLX** ([jundot/omlx](https://github.com/jundot/omlx)) | unverified | OpenAI + Anthropic | Hot KV cache plus SSD tier, DFlash, TurboQuant. Version 0.6.x [third-party reports]. |
| **LM Studio** (mlx-engine) | probably | `/v1/messages` since 0.4.1 ([docs](https://lmstudio.ai/docs/developer/anthropic-compat)) | mlx-engine pins mlx-vlm commit `573562d` (2026-09-24, 166 commits behind v0.7.9). That commit includes `glm5_next`/`qwen4_exp` but predates bundled-drafter loading. MTP in LM Studio is unknown. |
| **Ollama** (MLX engine preview, v0.19+) | not for these models | Anthropic Messages since 2026-01-16; `ollama launch claude` ([blog](https://ollama.com/blog/mlx)) | |

**Claude Code against a local server.** It works through the Anthropic-format gateway path. Anthropic notes it
"doesn't support routing Claude Code to non-Claude models"
([gateway docs](https://code.claude.com/docs/en/llm-gateway)). What Claude Code needs
([protocol](https://code.claude.com/docs/en/llm-gateway-protocol)):

- **Endpoints.** It needs `POST /v1/messages` with SSE through `message_delta`/`message_stop`. Emit `ping` events
  during long thinking or prefill, or the idle watchdog aborts the request. Through a custom `ANTHROPIC_BASE_URL`, the
  byte-level and event-level watchdogs fire after 300 s with no bytes. `CLAUDE_STREAM_IDLE_TIMEOUT_MS` can raise this
  to at most 30 min. The stock mlx-vlm server sends no pings, so a long GLM/Qwen prefill can hit this limit.
  `serve/` must add pings. `/v1/messages/count_tokens` is optional;
  without it, Claude Code estimates by characters.
- **Fields it sends.** Expect `cache_control` markers, which a local server can ignore. It also sends
  `thinking: {type: "adaptive"}` and `output_config.effort` for unknown model IDs.
- Env:
  ```bash
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080  ANTHROPIC_AUTH_TOKEN=<server --api-key>
  ANTHROPIC_MODEL=<served id>  ANTHROPIC_DEFAULT_HAIKU_MODEL=<served id>  CLAUDE_CODE_SUBAGENT_MODEL=<served id>
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072    # unknown IDs are assumed 200K; auto-compaction keys off this
  CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 # drop context_management / strict / defer_loading fields
  ```
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` applies on its own only when the served ID does **not** start with `claude-` and has
  no `[1m]` suffix (model-config docs). Serve the model under a plain name such as `glm-5.3-flash-local`.

**OpenCode against a local server.** Use a custom provider with `"npm": "@ai-sdk/openai-compatible"`, set
`options.baseURL` to `http://127.0.0.1:8080/v1`, and set `limit.context`/`limit.output` per model
([docs](https://opencode.ai/docs/providers/)).

---

## Recommendations for this project

1. **serve/ must stop the GLM projected MLA cache from persisting.** The simplest fix is
   `glm5_next.language._MAX_PROJECTED_PREFILL_TOKENS = 0`. That alone keeps `caches[3]` empty on every DSA layer and
   on the MTP drafter, it is bit-exact, and no cache surgery is needed. With a bound above 0 (such as 4096), also swap
   `caches[3]` for an empty `KVCache` after each prefill, including after an APC restore. Without a fix, GLM holds an
   extra 22–24 GiB at ≥ 32K context, ~~about 10×~~ **about 3×** the 8 GB budget.
2. **Budget the GLM long-prompt prefill peak** ~~at 64 KiB × context for one layer (8 GiB at 128K)~~ from the measured
   transient. On the tiny model it was about 85 KiB/token at step 512 and about 150 KiB/token at step 2048, for one
   DSA layer at a time. Extrapolated to 128K that is roughly 10–11 GiB and 18–19 GiB (§3). Use a prefill step of
   512–1024 for long GLM prompts, and have `evals.speed` measure the real 128K peak before trusting the 80 GB cap.
   Absorbed-query (latent-space) prefill attention would remove only the 64 KiB/token K/V part. With the fix,
   steady-state KV+state at 128K is Qwen 3.9 GiB and GLM 1.7 GiB (MTP included). Both fit the 8 GB line.
3. **Override the wired limit.** The server wires up to 107.5 GiB. In `serve/`, patch `mlx_vlm.generate.ar.wired_limit`
   (or call `mx.set_wired_limit` after generator start) to about weights + 4 GiB, and call
   `mx.set_memory_limit(80 GiB)` and `mx.set_cache_limit(...)` yourself. **`mx.set_memory_limit` is not a hard cap.**
   MLX documents it as a guideline that raises only once RAM and swap are exhausted. With a 1 GiB limit we allocated
   3 GiB of live arrays without an error. Enforce the 80 GB budget yourself in `serve/`: cap context length and
   `--max-num-seqs`, size APC, and reject requests whose projected KV + prefill transient would exceed it.
4. **MTP: enable with `--draft-model <same dir> --draft-kind mtp`.** Keep the default adaptive block sizes. Both models
   start at native depth (block 2) and grow to a ceiling of 4 (Qwen) or 3 (GLM). `build/` must keep the MTP tensors
   under their source key names, quantized in the same mode as the main model (issue #2526). **It must also give them
   either uniform default-bits quantization or drafter-path quantization keys**, because per-path overrides keyed by
   the source names break `load_drafter` (§1). Measure acceptance and tok/s in `evals.speed` against a no-draft
   baseline, and confirm greedy MTP matches plain decode on the real Qwen model. Its accept path is untested.
5. **Do not enable `--kv-bits`** on the server for either architecture: it crashes under continuous batching, is a
   no-op or crash for GLM, and TurboQuant is a no-op for Qwen. It is not needed for the budget.
6. **Skip DFlash2/DSpark for now.** GLM is incompatible with this mlx-vlm. The Qwen drafter is untested, unproven
   (0 downloads, no acceptance data) and was trained on the unpruned model. Revisit if a Flash-Next DFlash2 shows
   τ > MTP on an independent benchmark.
7. **Turn on APC for agentic use:** `APC_ENABLED=1`, `APC_MEMORY_MAX_GB=8` (explicit), `APC_CHECKPOINT_ENTRIES=2`,
   `APC_DISK_MAX_GB` sized to free SSD space. Keep the disk tier off the volume receiving the 700 GB downloads, or
   disable it during downloads. Run with `--max-num-seqs 1–2` so memory stays bounded.
8. **Prefill step: 2048 by default** (the server default, confirmed). Try 4096 only if `evals.speed` shows a real gain
   and the peak stays within the cap. For GLM prompts longer than about 32K, prefer 512–1024 (rec. 2).
9. **Keep the stock mlx-vlm server as the base** and wrap it with the memory caps and GLM fixes above, since it already
   serves `/v1/messages`. The wrapper must also emit SSE `ping` events while a request is prefilling or thinking, since
   the stock server sends none and Claude Code aborts after 300 s with no bytes. Use the Claude Code env block above, and give OpenCode an `openai-compatible` provider. MTPLX
   is the only notable alternative, and only for Qwen; worth a speed comparison but not adoption, given its 87 GiB pack.

### Deps needed
None beyond the current `pyproject.toml`.

### Probe scripts (reproducible, tiny models, < 8 GB)
The probes live outside the repo in the session scratchpad (`.../scratchpad/probe/`):

- `cache_probe.py`: per-layer cache bytes and prefill peak. Env `STEP`, `GLM_TYPES`, `QWEN_TYPES`.
- `kvq_probe.py` and `batch_kvq_probe.py`: KV quantization, single-stream and server batch path.
- `glm_proj_probe.py`: projected-cache drop equivalence.
- `apc_probe.py`: APC plan.
- `mtp_probe.py`: greedy MTP compared with plain decode.

Run each with `uv run python <script> <config.json> {qwen|glm}`.

---

## Verification (adversarial re-check, 2026-10-20)

Every recommendation and number above was re-checked against the installed source (mlx-vlm 0.7.9, MLX 0.32.4). We
reran the probes on tiny real-dim models (peak transient < 10 GB, weights < 0.4B params) and re-fetched every cited
source: GitHub API, HF model cards and configs, Claude Code and OpenCode docs. New probes are in
`.../scratchpad/verify/`: `glm_bound_speed.py`, `glm_peak_scale.py`, `mtp_accept_probe.py`,
`mtp_mixed_quant_probe.py` and `train_mode_probe.py`.

| # | Claim | Verdict | Notes |
|---|---|---|---|
| 1 | Free GLM projected cache; bound ≤ 4096; 22–24 GiB extra, ~10× budget | **Mostly verified; one number refuted** | Reproduced 2048 MiB at 32K and still 2048 at 40K, 64 KiB/token, and max\|Δlogit\| = 0 for drop, bound 0 and bound 4096. The real config's DSA layers all have `indexer_types = full`, so `caches[3]` is the right slot (shared-indexer layers would use `caches[1]`). The MTP drafter fills the same cache. "~10× the 8 GB budget" is wrong: 24 GiB is about 3×. Bound = 0 alone is sufficient (rec. 1). |
| 2 | KV+state totals (Qwen 0.98/3.58/7.06, +MTP 1.05/3.87/7.64; GLM 0.50/1.60/3.06, +MTP 0.53/1.73/3.32 GiB) | **Verified** | Re-measured 2372 B/token per QSA layer (18.53 MiB per 8192 tokens), 3.06 MiB per GDN layer, 0.18 MiB PLE, 4.14 MiB per KDA layer, and 1024+1+64 B/token per DSA layer. Every total recomputes to the stated value. Note that Qwen+MTP at 256K (7.64 GiB = 8.2 GB) exceeds 8 GB. The claim is only made for 128K. |
| 3 | GLM prefill peak = 64 KiB × context, step-independent; 6.0 / 7.4 GiB measured | **Refuted** | The peak grows with step: ~85 KiB/token at step 512 and ~150 KiB/token at step 2048. It is about the same when no recompute happens, so it is not mainly the K/V recompute. The earlier 6.0/7.4 GiB did not reproduce; we measured 7.0/7.0 GiB at 32K/40K. It does stay one layer at a time. |
| 4 | MTP on both via `--draft-model/--draft-kind mtp`; splitter registry; PR #2457; issue #2526; key names; blocks Qwen 4 adaptive / GLM 3; fp32 exact | **Verified with corrections** | Registry, key prefixes, CLI flags, env vars, `/v1/settings`, PR #2457 (merged 2026-10-12) and issue #2526 (open) all check out. GLM's block is adaptive too. The fp32 exactness run accepted no drafts. A rigged rerun shows GLM's accept path is exact; Qwen's is untested. Newly found: mixed-precision per-path overrides break bundled-drafter loading. |
| 5 | Don't use `--kv-bits` | **Verified** | Qwen batch: `NotImplementedError`. GLM batch: bool-quantize error (uniform) and reshape error (TurboQuant). GLM single-stream uniform is a no-op and TurboQuant crashes. Qwen single-stream 4-bit gives ~930 B/token and Qwen TurboQuant is a no-op. Caveat: this only reproduces when an attention layer is not the last layer (§2). |
| 6 | Override wired/memory limits; 107.5 GiB; no memory-cap flag | **Verified, but the remedy is insufficient** | `ar.py:2594` and `common.py:266` confirmed, as is `max_recommended_working_set_size` = 115,448,725,504. No mlx-vlm code calls `set_memory_limit`/`set_cache_limit`. `--expert-cache-gb` applies only to offload checkpoints. But `mx.set_memory_limit` is soft: 3 GiB allocated under a 1 GiB limit with no error (rec. 3). |
| 7 | Skip DFlash/DFlash2/DSpark | **Verified** | `validate_dflash_target` requires `rollback_speculative_cache`. GLM lacks it. Qwen inherits it (and its dflash prefill policy) from `qwen3_5`. HF configs match. z-lab numbers confirmed: DFlash2 τ 4.10–5.46 and 2.67–3.43× at c=1; MTP τ 3.74–5.02. The DSpark speedup was misattributed to an H200; it was measured on 4× GB300 (corrected). The DFlash2 repo's 0 downloads is because it is one day old. |
| 8 | MTPLX: separate runtime, Qwen only, 87 GiB peak, 1.7× (73.5 vs 43.8) | **Verified** | The HF card and README confirm 87 GiB peak, ~83 GB resident, a 32 GB n-gram table on SSD, 125.8 / 61.8 / 50.3 tok/s, OpenAI + Anthropic `/v1/messages`, and the session bank and SSD cache. GLM is not listed. The README also lists Ternary Bonsai 2 27B. |
| 9 | APC knobs and checkpoint strategy | **Verified** | Off by default (`apc.py:4933`). `strategy=checkpoint` for both models. Defaults 2 entries / 2048 tokens / min(8 GiB, ws/10) / 20 GB disk at `~/.cache/mlx-vlm/apc`. PR #2206 is open and says "APC stores target caches only". Added: the APC budget sits on top of live KV, and oversize checkpoints spill to disk. |
| 10 | Keep stock server; Claude Code env; OpenCode provider | **Verified with a gap** | Routes, `extra="allow"`, the `thinking`/`output_config` fields, every env var and the OpenCode pattern check out. The quoted "doesn't support routing Claude Code to non-Claude models" is in the gateway docs. Gap: the server emits no `ping`, and the watchdogs abort after 300 s of silence. Model-ID naming matters for `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. |
| 11 | Prefill step 2048; tiny-model peaks; `model.eval()` | **Partly refuted** | Default 2048 confirmed. Qwen at step 512 was 1023 MiB, not 0.71 GB; GLM at 4096 was 2707 MiB, not 2.9 GB. "About 2× memory in training mode" is wrong: GLM is 5.9× memory and 4.6× slower, Qwen ~1.0× memory and 1.5× slower. The `model.eval()` advice stands. |

**Not re-verified:** the bf16 MTP divergence positions (tokens 4 and 17), LM Studio's pinned commit, and Ollama and
oMLX details.
