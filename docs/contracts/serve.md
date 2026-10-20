# serve/ contract

Memory-capped, single-model server for agentic coding on this Mac. It wraps
`mlx_vlm.server` (mlx-vlm 0.7.9), which already implements the OpenAI and
Anthropic protocol layers, native tool-call parsing, the automatic prefix cache
(APC) and MTP speculative decoding. `serve/` pins one model, enforces the
co-tenant memory budget from PLAN.md, and fixes the places where mlx-vlm's
defaults are wrong for this machine.

## Launch

```
scripts/serve.sh <model> [options]          # <model>: dir, models/<name>, or HF repo id
uv run python -m serve --model <dir> [options]
uv run python -m serve --model <dir> --dry-run   # JSON fit report; exit 2 if refused
scripts/launchd/install.sh <model> [--port N] [--memory-limit-gb N] [--idle-unload-s N] [--load]
```

Every option also reads `SERVE_<OPTION>` from the environment (e.g.
`SERVE_MEMORY_LIMIT_GB=80`). The LaunchAgent (`com.local-ai.serve`) starts with
`--no-preload`, logs to `logs/serve.log`, and is restarted by launchd on crash.

| Option | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8642` | Loopback only by default. |
| `--api-key` | none | Require `Authorization: Bearer <key>` (or `x-api-key: <key>`) everywhere; unauthenticated requests never load the model. |
| `--memory-limit-gb` | 80 | `mx.set_memory_limit`; also the cap APC sizes itself against and the runtime guard threshold. |
| `--kv-reserve-gb` | 8 | Headroom above the weights for KV/state/activations. |
| `--wired-limit-gb` | weights + reserve | `mx.set_wired_limit`, never above the cap. Released (0) on unload. |
| `--cache-limit-gb` | 4 | `mx.set_cache_limit` (freed-buffer cache). |
| `--idle-unload-s` | 900 | Unload after this many seconds with no request in flight; 0 disables. Reload is automatic on the next request. |
| `--no-preload` | off | Load on first request. |
| `--max-context` | 131072 | Prompt + generation limit (mlx-vlm `MAX_KV_SIZE`). |
| `--max-output-tokens` | 32768 | Upper bound on every request's `max_tokens`. |
| `--no-thinking` | off | Both candidate templates think by default; reasoning is returned as Anthropic `thinking` blocks / OpenAI `reasoning_content`. With this flag, requests that do not choose get thinking off. |
| `--mtp` | `auto` | `auto`: use the checkpoint's own MTP head if its tensors are present; `on`: require it; `off`. |
| `--draft-model` | none | Separate drafter checkpoint instead of the bundled head. |
| `--kv-bits`, `--kv-quant-scheme`, `--kv-group-size`, `--quantized-kv-start` | off | KV cache quantization (see below). |
| `--no-prefix-cache` | off | Disable APC. |
| `--prefix-cache-memory-gb` | cap − weights − reserve, clamped to 0.5–16 | Resident APC budget. |
| `--prefix-cache-disk-gb` | 32 | SSD tier; 0 disables. |
| `--prefix-cache-entries` | 4 | Whole-prefix checkpoints kept resident (hybrid models). |
| `--prefix-cache-dir` | `~/.cache/local-ai/prefix-cache` | SSD tier root. |

## What a model directory must contain (for build/)

Whatever `mlx_vlm.load(<dir>)` accepts: `config.json`, `*.safetensors`
(+ `model.safetensors.index.json` when sharded), tokenizer files and the chat
template (`chat_template.jinja` or inside `tokenizer_config.json`). The server
infers the tool-call parser from the template (`qwen3_coder` for Qwen3.8,
`glm47` for GLM-5.3), so the original templates must be kept.

MTP is detected from the checkpoint itself
(`mlx_vlm.speculative.drafters.mtp_split.detect_mtp_splitter`): the config must
declare the head (`text_config.mtp_num_hidden_layers` for `qwen4_exp`,
`text_config.num_nextn_predict_layers` for `glm5_next`) and the MTP tensors must
be in the same safetensors under the source names mlx-vlm's splitters select:

- `qwen4_exp`: keys starting `mtp.` (also accepted: `model.mtp.`, `language_model.mtp.`)
- `glm5_next`: layer `num_hidden_layers` of the decoder, e.g. `model.layers.45.*`
  or `language_model.model.layers.45.*` (`enorm`, `hnorm`, `eh_proj`,
  `shared_head.norm`, and the block's attention/MLP tensors)

Quantized MTP tensors are fine (`.scales`/`.biases` with the `quantization`
config). `serve.testing.build_tiny()` writes a tiny checkpoint in exactly this
layout and is the executable reference.

## Memory model

- Weights are sized as the sum of the safetensors files (local) or the HF file
  listing (repo id). The server refuses to start (exit 2, or HTTP 507 on lazy
  load) when `weights + kv_reserve > memory_limit`.
- On load: `set_memory_limit(cap)`, `set_cache_limit`, `set_wired_limit`.
  mlx-vlm's `BatchGenerator` normally raises the wired limit to the entire Metal
  working set (~115 GB here) while it lives; serve replaces that context manager
  so the wired limit stays at the configured value.
- Every decode/prefill step checks active memory against the cap. Over the cap it
  clears the MLX cache, then evicts idle APC state, then fails the in-flight
  request with `Memory cap exceeded` (the server and model stay up).
- APC's resident budget and admission headroom are computed against the cap,
  not the Metal working set.
- GLM-5-Next keeps a per-head projected K/V copy (~704 KB/token for GLM-5.3)
  during prefill as a speed optimization (mlx-vlm allows 32K tokens = 22 GB).
  serve caps it at `kv_reserve/4` (2 GB → ~2.9K tokens); longer prompts use
  mlx-vlm's recompute path. Reported as `glm_projected_prefill_tokens`.

Measured from the real configs (bf16 caches):

| | full-attn layers | KV bytes/token | KV @128K | other state |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | 12 of 48 | 28.4 KB | 3.5 GB | linear-attn state 108 MB |
| GLM-5.3-Flash | 11 of 45 (MLA latent + indexer) | 14.1 KB | 1.7 GB | prefill transiently expands one layer's K/V: 8 GB @128K |

GLM's transient prefill expansion means a ~68 GB GLM build has little slack at
128K under an 80 GB cap; if the guard trips on very long prompts, lower
`--max-context` (e.g. 65536) rather than raising the cap.

## Endpoints

Inference (model name in the request is ignored; the pinned model always serves):

- `POST /v1/chat/completions` — OpenAI; `stream`, `stream_options.include_usage`,
  `tools` / `tool_choice`, `tool_calls` + `role: tool` history,
  `usage.prompt_tokens_details.cached_tokens`, `reasoning_content`.
- `POST /v1/messages` (`?beta=true` fine) — Anthropic Messages; SSE events
  `message_start`, `content_block_start/delta/stop` (`text_delta`,
  `thinking_delta`, `input_json_delta` for `tool_use`), `message_delta`
  (`stop_reason` `end_turn` / `tool_use` / `max_tokens`), `message_stop`;
  `usage.cache_read_input_tokens` reports prefix-cache hits.
- `POST /v1/messages/count_tokens`, `POST /v1/responses` (mlx-vlm).

Management:

- `GET /v1/models` — one entry: `{id: served_name, loaded, max_context}`.
- `GET /serve/status` — `loaded`, `weights_gb`, `speculative{drafter,active,kind}`,
  `kv_quant{requested_bits,active_bits,error}`, `glm_projected_prefill_tokens`,
  `prefix_cache_dir`, `inflight`, `queue_depth`, `requests_served`, `idle_s`,
  `load_count`, `unload_count`, `last_load_s`, `last_unload_reason`,
  `memory{active_gb,cache_gb,peak_gb,limit_gb,cap_gb,wired_limit_gb,guard_trips}`,
  `prefix_cache` (mlx-vlm APC stats or null), `config`.
- `POST /serve/load`; `POST /serve/unload[?force=1]` (409 while a request is in flight; `/unload` is an alias).
- `GET /health`, `/metrics`, `/v1/cache/stats`, `POST /v1/cache/reset`, `GET /v1/settings` (mlx-vlm).
- `PATCH /v1/settings` is removed: it could point the server at another model or drafter and bypass the cap.

## Behavior

- One sequence decodes at a time (`MLX_VLM_MAX_NUM_SEQS=1`); concurrent requests
  queue in arrival order.
- `max_tokens` is clamped to `--max-output-tokens` and to `max_context − prompt`
  (instead of rejecting Claude Code's 32000-token requests on long prompts). A
  prompt alone over `--max-context` is a 400.
- Prefix cache: both candidates are hybrid (linear attention + attention), whose
  recurrent state cannot be trimmed to an arbitrary prefix. mlx-vlm's APC runs in
  `exact` mode for them: it stores whole-cache checkpoints at the end of each
  prompt (minus a guard token) and every 2048 tokens, and restores the longest
  checkpoint that is a prefix of the next prompt. In an agent loop each turn's
  prompt extends the previous one, so only the new turn is prefilled (verified
  with the real Claude Code CLI: turn 2 prefilled 66 of 17,865 tokens). The SSD
  tier is keyed by a fingerprint of the checkpoint files, so a rebuilt model at
  the same path never restores stale state; older fingerprints of the same served
  name are deleted at startup. The SSD tier survives idle unloads and restarts.
- MTP: when active, request `thinking.budget_tokens` is dropped (mlx-vlm cannot
  enforce budgets while drafting); `response_format` JSON-schema requests are
  rejected by mlx-vlm with MTP on — run with `--mtp off` if a client needs them.
- KV quantization: mlx-vlm 0.7.9's batch path cannot quantize Qwen4-Exp's
  `QSAKVCache` or GLM-5-Next's DSA `CacheList`. serve probes at load and falls
  back to full-precision KV, recording the reason in `kv_quant.error`; it is
  honored for architectures mlx-vlm supports. Neither candidate needs it to meet
  the 8 GB KV budget (table above).

## Clients

Claude Code:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:8642 ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=<served-name> ANTHROPIC_DEFAULT_HAIKU_MODEL=<served-name> \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
```

Keep `CLAUDE_CODE_MAX_OUTPUT_TOKENS` ≤ `--max-output-tokens`. With
`--api-key K`, use `ANTHROPIC_AUTH_TOKEN=K` (or `ANTHROPIC_API_KEY=K`).

OpenCode (`opencode.json`):

```json
{
  "provider": {
    "local-ai": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8642/v1" },
      "models": { "<served-name>": { "limit": { "context": 131072, "output": 32768 } } }
    }
  }
}
```

## Tests

`uv run python -m pytest tests/serve` — builds tiny random-initialized
checkpoints of the real `qwen4_exp` and `glm5_next` classes (8 layers, 8
experts, native MTP heads, real tokenizers/templates from the HF repos' metadata
files) and drives every endpoint in-process.
