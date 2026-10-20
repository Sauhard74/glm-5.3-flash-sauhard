# local-ai — best local model for M5 Max 128 GB, tuned for this machine

## Goal
Run the highest-quality model that fits a **co-tenant memory budget**: the machine is also used
for coding, IDEs, builds and Claude Code, so the model must never take over unified memory.

| Budget item                           | Limit                      |
|---------------------------------------|----------------------------|
| Model weights on disk / resident      | <= 70 GB                   |
| KV + linear-attn state @ 128K context | <= 8 GB                    |
| Whole server process (enforced cap)   | <= 80 GB (`mx.set_memory_limit`, `mx.set_wired_limit`) |
| Left for macOS + dev tools            | >= 45 GB                   |

Workload (calibration + eval target): **coding / agentic tool use, reasoning / math, long-document RAG**.

## Hardware
Apple M5 Max, 18-core CPU, 40-core GPU (with neural accelerators), 128 GB, 614 GB/s, macOS 27.0.1.
MLX 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.9 (the candidate architectures live in **mlx-vlm**:
`mlx_vlm/models/qwen4_exp`, `mlx_vlm/models/glm5_next`).

## Candidates
| Model                    | Total / active | Arch                                                  | Source ckpt (HF)                   |
|--------------------------|----------------|-------------------------------------------------------|------------------------------------|
| GLM-5.3-Flash            | 321B / ~18B    | `glm5_next`: linear attn + DeepSeek sparse attn (MLA), 288 experts top-8, 1 shared, MTP | `zai-org/GLM-5.3-Flash` (FP8, 328 GB) |
| Qwen3.8-Flash-Next       | 180B / ~6B     | `qwen4_exp`: 3:1 linear/full attn, 512 experts top-10, MTP | `Qwen/Qwen3.8-Flash-Next` (BF16, 360 GB) |
| Baseline (community)     |                |                                                       | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` (68 GB, stock mlx-vlm). NOT Litwein/*-oQ3e-* — needs oMLX runtime, gibberish on mlx-vlm |

Neither unpruned model fits the budget at >=3 bits, so the pipeline is:
**workload-calibrated REAP expert pruning → sensitivity-driven mixed-precision quantization → keep MTP head → memory-capped server.**

## Pipeline & module contract
All code is Python run via `uv run`. Do NOT edit `pyproject.toml`/`uv.lock`; list new deps in your report.

1. `calib_data/` — `uv run python -m calib_data.build --model {glm,qwen} --out artifacts/<model>/data/`
   Writes `calib.jsonl` and `eval.jsonl` (held out). One JSON per line: `{"id","domain","tokens":[int...]}`,
   tokens already chat-templated with the target model's own tokenizer/template.
   Domains: `code`, `agentic` (multi-turn tool calls in the model's native tool format), `reasoning`, `longctx` (16K–64K).
2. `calib/` — `uv run python -m calib.run --model {glm,qwen} --src <hf snapshot dir> --data artifacts/<model>/data --out artifacts/<model>/calib`
   **Layer-wise streaming**: load ONE decoder layer at a time from the source safetensors (dequantize FP8 block-scaled
   for GLM), push cached hidden states through it, free it. Peak memory must stay < 40 GB.
   Writes per layer `layer_{i:03d}.npz` / `.json` containing:
   - `expert_freq[E]`, `expert_reap[E]` (REAP saliency: mean over routed tokens of gate_weight * ||expert_out||₂), per domain too;
   - per quantizable tensor: `sens[name][(bits,group)]` = relative output error on real activations for bits ∈ {2,3,4,5,6,8}, group ∈ {32,64};
   - input activation channel magnitudes (for AWQ-style scaling).
   Also writes `ref/` : full-precision top-64 logprobs for `eval.jsonl` (for KL eval).
3. `build/` — `uv run python -m build.make --model {glm,qwen} --src <snap> --calib artifacts/<model>/calib --prune <ratio|auto> --budget-gb 68 --out models/<name>`
   Prunes experts per layer by REAP score (router rows + expert weights), solves the bit allocation
   (min Σ sensitivity s.t. size <= budget), quantizes with MLX, keeps the MTP layer, optionally drops vision,
   writes a directory loadable by `mlx_vlm.load` (and the server).
4. `evals/` — `uv run python -m evals.<suite> --model models/<name>` suites: `kl` (vs ref), `speed` (prefill/decode tok/s at 1K/32K/128K + peak memory),
   `code`, `math`, `longctx` (RULER-style 32K/64K/128K), `tools` (function calling). Results → `results/<name>/<suite>.json`.
5. `serve/` — memory-capped OpenAI- and Anthropic-compatible server with prompt caching and MTP speculative decoding;
   `scripts/serve.sh <model>`.

## Rules for anyone working here
- The GPU and memory are shared with the user's real work and with long downloads in progress.
  **Never load more than ~8 GB of model weights during development.** Test code paths with tiny
  random-initialized configs of the real architecture classes (e.g. 2 layers, 8 experts).
- Do not touch `~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash` or `models--Qwen--Qwen3.8-Flash-Next`
  while they download, except reading `config.json` / tokenizer files.
- Large runs (calibration, quantization, evals on real models) are run one at a time by the orchestrator.

## Status — 2026-10-20 (end of first session)
**Running now (persistent):** `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` + MTP drafter `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16`,
served on `http://127.0.0.1:8080` (OpenAI + Anthropic APIs) by the launchd control daemon.
- Dashboard: http://127.0.0.1:8090 (start/stop/restart, memory, settings). Daemon: `control/daemon.py`, installed by `scripts/install-control.sh` (`--uninstall` to remove).
- Measured (`results/runtime_tune.json`): plain ~30 tok/s → **MTP block 2: 54–58 tok/s**; peak ~48 GB (the 51B n-gram/PLE table streams from SSD in mlx-vlm 0.7.9). Prompt cache: 1.8K-token repeat 4.2 s → 1.25 s.
- Claude Code on the local model: `scripts/claude-local.sh`.
- `Litwein/*-oQ3e-*` is oMLX-only (gibberish on mlx-vlm) — don't use with this stack.

**Stopped mid-way (user request):** both multi-agent workflows.
- Research: `docs/research/{quant,reap,decode,mac}.md` — complete and adversarially verified.
- Modules `calib_data/ calib/ build/ evals/ serve/` — built by agents, not yet independently reviewed. All tests pass, and `scripts/pipeline_tiny.sh` runs the full chain (make_tiny → calib_data → calib → build → evals kl/speed → serve dry-run) for **both** architectures on tiny real-arch checkpoints. Fixed while integrating: REAP test reference, serve MTP detection of build's `mtp/` drafter dir. Builds need `--drop-vision` for text-only use.
- Master plan: chapter drafts in `docs/plan/` (unreviewed; some may be incomplete). `docs/MASTER_PLAN.md` was not yet synthesized.

**Background downloads (survive session close):** `zai-org/GLM-5.3-Flash` then `Qwen/Qwen3.8-Flash-Next` into the HF cache (log: `logs/downloads.log`). Stop with `pkill -f "hf download"`; rerun the same `hf download` to resume.

**Next steps:** finish/review master plan chapters → review + test modules → integration run on tiny fixtures → calibrate on real sources → build ≤68 GB variants → eval vs the running baseline; swap in only if it wins.

## Status — 2026-10-20 evening (GLM custom build)
**GLM calibration done** (`artifacts/glm/calib`, 45 layers + ref, 04:52–10:19). Verified correct: calib's layer stream matches stock mlx-vlm, and the source MoE matches the transformers reference.

**How to read GLM KL (important):** `evals.kl` now splits by chat role. Use `assistant_kl` (positions the model generates) as the quality headline. `user`/`system`/`tool` positions are prompt *reading*: KL there is large (2–7) for every build, because a chat model doesn't predict user/tool text verbatim and GLM's bf16 forward is hypersensitive there (a lossless fp16 round-trip alone moves it by KL ~0.3). It still tracks comprehension loss from pruning. fp32 activations were measured: no better than bf16.

**A/B so far** (`results/<name>-l12/kl.json`, 12 samples/domain):

| build | assistant KL | top-1 | tool KL | reasoning KL |
|---|---|---|---|---|
| A `glm-candidate-auto76` (REAP 153/288, RTN, 76 GB) | 0.121 | 92.4% | 6.98 | 0.069 |
| B `glm-mass153-76` (mass score) | 0.123 | 92.3% | 6.73 | 0.086 |
| C `glm-massauto-80` (mass, 162, 80 GB) | 0.118 | 92.6% | 6.69 | 0.081 |
| **A + `--rounding lsq`** `glm-reap153-76-lsq` | **0.108** | 92.4% | 6.92 | **0.066** |

- Rounding (LSQ + imatrix + range search, `build/round.py`) is the big lever: −25% weighted quant error, −10.5% assistant KL, same bytes.
- Pruning drives the tool/user gap: pure REAP is flat on GLM and pruned the most-routed experts (kept experts carry only ~52% of routing frequency).
- Running: C + LSQ (`glm-massauto-80-lsq`). Next: `recover/blockdwq.py` (blockwise DWQ, written and tested on tiny) on the winner, then task evals (code/tools), then promote to `models/glm-5.3-flash-custom` (the dashboard's glm profile).

**Ops notes:** eval footprint ≈ model + ~10 GB (the projected-cache fix is now in `evals/common.load_model`). Run only one 70+ GB job at a time: swap is ~40 GB and jetsam killed evals that overlapped other work. The Qwen server is stopped during GLM work; start it from the dashboard afterwards.
