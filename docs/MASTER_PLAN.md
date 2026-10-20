# MASTER PLAN — the best local LLM system on this M5 Max, fully calibrated

> Top-level plan for `local-ai`. MacBook Pro M5 Max, 18-core CPU, 40-core GPU, 128 GB unified memory, 614 GB/s,
> 4 TB SSD (≈ 2.0 TiB free on 2026-10-20), macOS 27.0.1, mlx 0.32.4 / mlx-lm 0.32.2 / mlx-vlm 0.7.9.
>
> **This document is the map. The chapters are the territory.** Every stage below links to the chapter section that
> explains the mechanics, the math and the evidence. Read this file to know *what to do next and why*. Read the
> chapter to learn *how it works*. Chapter index: [`plan/README.md`](plan/README.md).
>
> | Chapter | Topic |
> |---|---|
> | [01 Memory](plan/01-memory.md) | unified-memory accounting, profiles, SSD streaming, hot/cold tiers, watchdog |
> | [02 Models](plan/02-models.md) | which model, which role, "bigger than RAM" options |
> | [03 Calibration data](plan/03-calibration-data.md) | domains, personal data, self-generation, contamination |
> | [04 Pruning](plan/04-pruning.md) | REAP / REAP-sum / HOPE, the streaming sweep, the experts-vs-bits trade |
> | [05 Quantization](plan/05-quantization.md) | formats, g128, knapsack units, floors, LSQ/GPTQ, recipes |
> | [06 Recovery](plan/06-recovery.md) | DWQ on scales/biases, router KD, on-policy distillation |
> | [07 Speculative](plan/07-speculative.md) | MTP, n-gram/suffix drafting, prefix caching, kernels |
> | [08 System](plan/08-system.md) | server, clients, router, RAG, tools, observability, eval gate, ops, security |
>
> Verified research the chapters build on: [`research/mac.md`](research/mac.md), [`research/decode.md`](research/decode.md),
> [`research/quant.md`](research/quant.md), [`research/reap.md`](research/reap.md). Module contracts:
> [`contracts/`](contracts/). Project rules and status: [`../PLAN.md`](../PLAN.md).
>
> **Tags** (same meaning as in the chapters): **[measured]** run on this machine · **[computed]** arithmetic from real
> configs/headers · **[src]** read in installed source · **[self-reported]** someone else's number, not reproduced ·
> **[est]** an estimate with stated assumptions that a named measurement must replace.
>
> **Units.** GB = 10⁹ bytes (file sizes, `build.make --budget-gb`). GiB = 2³⁰ (RAM caps: `CAP_GB`, `serve
> --memory-limit-gb`, `mx.get_*_memory`). "80 GB cap" in this repo's code means **80 GiB = 85.9 GB**.

---

## Contents

1. [Executive summary and target end state](#1-executive-summary-and-target-end-state)
2. [Decisions that override chapter text (conflict resolutions)](#2-decisions-that-override-chapter-text)
3. [Concept map and learning path](#3-concept-map-and-learning-path)
4. [Phased roadmap (Phase 0 → Phase 9)](#4-phased-roadmap)
5. [Experiment matrix and decision rules](#5-experiment-matrix-and-decision-rules)
6. [Code work backlog, by module](#6-code-work-backlog-by-module)
7. [Risks and fallbacks](#7-risks-and-fallbacks)
8. [Glossary](#8-glossary)

---

## 1. Executive summary and target end state

### 1.1 The one-paragraph version

The best model this machine can run *while you are coding* is **not** the community REAP-288 build that is serving
today, and it is **not** GLM-5.3-Flash squeezed to 68 GB. It is **Qwen3.8-Flash-Next with all (or nearly all) 512
experts**, quantized by measured sensitivity to ≈ 3.6 bits per routed weight, with its 51 B-parameter n-gram/PLE table
left memory-mapped on the SSD (so it does not count against RAM), its MTP head kept for speculative decoding, and
its calibration done on your own workload. That build is ≈ 56–60 GB resident, fits the 80 GiB Work cap at 128K context
with ≈ 10 GiB to spare, and should decode at today's speed while removing the pruning damage the current build
carries ([01 §0 item 1, §4.3 row W1](plan/01-memory.md); [02 §0](plan/02-models.md); [05 §4.6 R-Q1](plan/05-quantization.md)).
GLM-5.3-Flash becomes the **secondary** "deep / second opinion" model for when you are *not* coding (All-in profile),
lightly pruned (keep ≈ 176–216 of 288 experts) and fully resident, or, as the most ambitious step, unpruned with its
coldest experts streamed from SSD ([01 §4.3 rows A1/A2](plan/01-memory.md); [02 §4.2](plan/02-models.md)).
Everything is gated by measurement: a build replaces the running one only if it beats it on per-domain KL **and**
on task, tool, rare-entity and termination guards ([08 §4.8](plan/08-system.md)).

### 1.2 Where we are (measured, 2026-10-20)

| | Value | Source |
|---|---|---|
| Serving now | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` (288 of 512 experts kept, 4b g64) + MTP drafter `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` | `control/state.json` |
| Stack | stock mlx-vlm server wrapped by `scripts/run_server.py` (memory caps, wiring 0, API key on *all* routes incl. `x-api-key`), launched by `scripts/quickserve.sh` on **:8080**, supervised by the launchd control daemon (`control/daemon.py`), dashboard **http://127.0.0.1:8090** | repo |
| Decode | ~30 tok/s plain → **54–58 tok/s with MTP block 2**; blocks 3/5/6 are slower | `results/runtime_tune.json` [measured] |
| Peak memory | ~48 GB, because the 51 B PLE/n-gram table streams from SSD (68 GB on disk) | [measured] |
| Prompt cache | 1.8K-token repeat: 4.2 s → 1.25 s | [measured] |
| Pipeline | `scripts/pipeline_tiny.sh` runs make_tiny → calib_data → calib → build → evals kl/speed → serve dry-run for **both** archs; **192 tests pass** | [measured] |
| GLM source | `models/src-glm-5.3-flash` (FP8, 328 GB) downloaded; `calib.check` passes all 45 layers | [measured] |
| Qwen source | `Qwen/Qwen3.8-Flash-Next` BF16 (360 GB) still downloading (`logs/downloads.log`) | — |

### 1.3 Target end state

| Profile (process cap) | When | Model build | Resident weights | Context | Expected decode | Expected quality | Confidence |
|---|---|---|---|---|---|---|---|
| **Work** (80 GiB; ≥ 48 GiB left for you) | coding, Claude Code running | **Qwen3.8-Flash-Next, K = 512 (or ≥ 448), ≈ 3.6 bpw routed experts** (importance tiers once supported), trunk floors 6/8-bit, MTP kept, PLE mmapped, own calibration (+ DWQ if it passes its gate) | ≈ 56–60 GB (≈ 52–56 GiB) + 29–32 GB PLE on disk | 128K, 1–2 APC checkpoints | ≈ 30 plain / ≈ 55 MTP (latency-bound, same as today) [est from measured sibling]; + n-gram drafting on tool arguments / file rewrites ≈ 2–5× on those segments [model] | no pruning loss; target per-domain KL ≤ 0.11 vs bf16 before DWQ, ≤ 0.08 after [est, 05 §4.6]; must beat REAP-288 on rare-name and tool probes | **medium-high**: the memory arithmetic is measured; the quality gain is inferred from GGUF and community evidence (MLX affine is cruder) |
| **Focus** (92 GiB; ≥ 36 GiB left) | editor + browser only, long reading or reasoning | Work build + 256K context + more APC (F1′), **or** Qwen R-Q1+ at ≈ 4.0 bpw (≈ 68 GB), **or** GLM K ≈ 160–176 if GLM wins its gate | 56–74 GB | 128–256K | as Work (Qwen) / ≈ 25–40 plain GLM [est] | Qwen saturates near 4 bpw ([01 §5](plan/01-memory.md)); spend extra bytes on context first | medium |
| **All-in** (104 GiB; ≥ 24 GiB left) | not coding: evenings, overnight agents, "think hard" | **GLM-5.3-Flash K ≈ 176–192 at ≈ 3.0–3.3 bpw** (≈ 82–90 GB), GLM serving fixes mandatory; fallback: Qwen K = 512 @ 4.5 bpw (≈ 74 GB) | 82–95 GB | 64K (128K after absorbed-query prefill) | GLM ≈ 25–40 plain, × 1.3–1.5 with MTP [model, 07 App. A]; GLM `t0` is unmeasured | GLM base is +2 AA points over Qwen and stronger on hallucination / τ² / GraphWalks; pruning eats exactly that edge, so GLM must *earn* the slot on the escalation set ([02 §5.2 D2](plan/02-models.md)) | **low-medium** |
| **Stream** (mode, any cap) | overnight "one hard question" | unpruned GLM, hot experts resident + cold tail on SSD (stock `moe_offload` first, slab path later) | trunk + cache budget | ≤ 32K | 3–17 tok/s (stock) → ≈ 28–33 (tiered A2, est) | the full 321 B model | **low**; decided by a trace simulation before any 143 GB build ([02 §4.4 step 6](plan/02-models.md)) |
| **Calibrate** (mode) | overnight; **server stopped from the dashboard** | none; `calib.run`, `build.make`, evals, recovery training | ≤ 40 GiB calib; ≤ 100 GiB training | — | — | — | — |

Honest uncertainty, stated once:
- **Speed.** Our measured plain decode (~30 tok/s) is well below the 44–48 tok/s two other runtimes report for this
  model class on an M5 Max ([07 §2.2](plan/07-speculative.md)). Part of that is the shared GPU during measurement;
  part may be mlx-vlm dispatch overhead (`t0`). Do not plan on more than today's speed from the new build; plan on
  speed gains only from speculation, caching and kernels (Phase 5, Phase 9).
- **Quality.** No one has published task scores for an unpruned MLX-affine Qwen at 3.6 bpw. The best evidence is
  GGUF (ISTA GSQ-RCO: 3.0 bpw keeps 99.4 % of bf16 task average [self-reported]) and "pruning dominates
  quantization error" ([04 §2.11](plan/04-pruning.md), [02 §2.4](plan/02-models.md)). Our own E-Q1 experiment
  (Phase 4) is what settles it.
- **GLM.** Everything about GLM speed is modelled, and the only data on GLM at ≈ 50 % pruning says it loops below
  144 experts. GLM is a bet with an explicit kill rule (§5.3 D2).

---

## 2. Decisions that override chapter text

The chapters were drafted in parallel and disagree in places. These resolutions are binding for the roadmap.
Each says what the conflict is, what we do, and why.

| # | Conflict | Resolution | Why |
|---|---|---|---|
| R1 | **GLM K = 144 at 68.0 GB.** [05 §4.6 R-G1](plan/05-quantization.md) claims a 68.0 GB recipe at ≈ 3.10 bpw experts. The build planner (`build.make --dry-run`, byte-exact) says the **minimum** for K = 144 (text-only, MTP kept) is **72.58 GB** (72.26 without default floors, 70.91 with MTP dropped). | **The planner is ground truth.** R-G1 needs two things the code does not have yet: **g128** in `calib/sens.py` and `build/alloc.py` (only g32/g64 today), and 2-bit expert tiers below the default `--min-bits 3`. Adding g128 and reconciling R-G1 against the planner is an explicit **Phase 1 task (B-1)**. Until then GLM recipes use the feasible-today table below. | Arithmetic: today's minimum is ≈ 4.35 GB fixed + 0.474 GB per kept expert-slot (3b g64 = 3.5 bpw, MTP experts included). At 3b g128 (3.25 bpw) the slope becomes ≈ 0.440, so K = 144 → ≈ 67.7 GB [computed]. That is R-G1's 68.0 GB, i.e. **R-G1 is the post-g128 floor, with no headroom** for a 4-bit hot tier unless a 2-bit cold tier pays for it. |
| R2 | **Does the PLE table count against the budget?** [05](plan/05-quantization.md) / [01](plan/01-memory.md) / [02](plan/02-models.md) / [08](plan/08-system.md) count **resident** bytes; [research/quant.md Rec 7–8](research/quant.md) counts the 68 GB as **disk** and concludes unpruned Qwen needs ≈ 2.0–2.2 bpw. | **Budget resident bytes.** The user's constraint is RAM, not disk (2 TiB free). Measured proof: REAP-288 is 68 GB on disk and peaks at ~48 GB. Disk is tracked as a separate budget. | The research note is correct *under its scope*; the scope was wrong for this constraint. The fallback if the mmapped PLE misbehaves under memory pressure is R-Q2 (disk-scope, K ≈ 352 @ 3b g128) — see §7. |
| R3 | **Personal-data weighting.** [01 §4.7](plan/01-memory.md): `personal=2`. [04 §4.2](plan/04-pruning.md): personal ≈ 40 % *of* agentic. [03 §4.6](plan/03-calibration-data.md): `personal` as its own domain at weight 1 (= 50 % of agentic+personal mass). | **Start at `personal=0.67`** (reproduces 04's 40 %) for the first personal-calibrated build, with the `general` floor at 0.3, and **use the same weights for REAP and for the bit allocator**. Then run experiment D2 (personal ∈ {0, 0.67, 1, 2}) on the screening sweep and adopt the largest weight that keeps *public* agentic KL within its CI of personal = 0. | No measurement exists; the transcripts' assistant turns are Claude's, and one person's repos are a narrow distribution, so the most conservative weight is the safe default. Token count sets *precision*, `--domain-weights` sets *importance* ([03 §2.1](plan/03-calibration-data.md)). |
| R4 | [07 §4.5](plan/07-speculative.md) says mlx-vlm "ships no training loop". | **Wrong.** mlx-vlm 0.7.9 has `mlx_vlm/trainer/` (SFT, DPO, ORPO, GRPO, LoRA incl. `LoRASwitchLinear`, a pluggable `TrainingTask`). It has **no KL-distillation objective** and nothing MTP-specific ([06 §3.4](plan/06-recovery.md)). We still write our own objective (in `recover/` and `drafter/`), reusing the trainer's gradient-accumulation and checkpoint plumbing where it fits. | 07's conclusion (write our own) stands; its premise does not. |
| R5 | **MTP block size.** [07 §2.3, §4.2](plan/07-speculative.md) predicts block 4 optimal for Qwen; [decode Rec 4](research/decode.md) says keep adaptive. | **Measured wins: block 2** on today's stack (54–58 tok/s; 3/5/6 slower). Re-sweep {2, 3, 4, 5} on every new build, because our builds ship their own quantized MTP head with a different acceptance α. | The cost model's `t0` and correlation factor are fitted from other runtimes; our measurement replaces them ([07 §5](plan/07-speculative.md)). |
| R6 | **Plain-decode anchor.** 07/02 plan with 42–48 tok/s plain, 70–80 MTP. | Plan with **measured 30 / 55**. Treat the gap as a Phase 5 investigation item (profile dispatches; idle GPU). | Measured beats self-reported. |
| R7 | **Port and serving stack.** Chapters use `serve/` on **:8642**; production runs `scripts/run_server.py` on **:8080** under `control/`. | **Keep :8080 as the canonical port.** Keep `run_server.py` + `control/` as production until `serve/` passes its P0 items and T0 gate; then point the control daemon at `python -m serve --port 8080`. Read every `8642` in the chapters as `8080`. | Clients (`scripts/claude-local.sh`, dashboard) already use 8080; changing clients buys nothing. |
| R8 | **Auth header.** [08 §4.3](plan/08-system.md) says `x-api-key` gets a 401. | Not true here: `run_server.py` accepts `Authorization: Bearer` **and** `x-api-key` on every route, and the `serve/` contract does too. Still prefer `ANTHROPIC_AUTH_TOKEN` (Bearer). | — |
| R9 | **"The deployed baseline has no MTP head"** ([08 §2.1](plan/08-system.md)). | True of the *checkpoint*; the deployment uses the **separate** sh0wie bf16 drafter, and MTP is measured active (block 2). Our own builds write the head as `<model>/mtp/` (loads via `load_drafter`; `serve` detects it). | — |
| R10 | **Profile names.** 01: Work/Focus/All-in. 02: P1/P1-lean/P2/P3. 08: work/all-in/stream/calibrate. | **Work 80 GiB, Focus 92 GiB, All-in 104 GiB** (01's caps) + two **modes**: *Stream* (SSD experts, any profile) and *Calibrate* (server stopped, overnight jobs). 02's P1 = Work, P1-lean = Work with a REAP-320 primary, P2 ≈ All-in, P3 = Stream. | 01 has the derived caps; 08's modes are operational, not memory profiles. |
| R11 | **GLM in the Work profile.** [01 §4.3 W2](plan/01-memory.md) lists GLM K = 144 in Work; [02 §4.3](plan/02-models.md) says GLM never belongs in Work (≈ 3× the divergence of Qwen at equal bytes). | **GLM in Work is an experiment (D3), not a plan.** Default Work model = Qwen. | 02's evidence; and R1 shows K = 144 does not even fit 68 GB until g128 lands. |
| R12 | **Where DWQ code lives.** [05 §4.2](plan/05-quantization.md): `build/dwq.py`, `build/dwq_band.py`. [06 §4.1](plan/06-recovery.md): a `recover/` package. | **`recover/` owns all training** (05's B1 = `recover.local`, B2 = `recover.dwq`). `build/round.py` (RTN/LSQ/GPTQ/AWQ-down) stays in `build/`, because rounding is a build-time decision. | One place for optimizers, memory self-bounding and gates. |
| R13 | **GLM projected-cache fix.** [08 §4.2c](plan/08-system.md): bound 4096 + replace `caches[3]` after prefill. [decode Rec 1](research/decode.md): `_MAX_PROJECTED_PREFILL_TOKENS = 0` is the simplest bit-exact fix. | Use **`= 0`** (verified research). Use 08's cache swap only if a bound > 0 is ever needed for speed. | Simpler, verified bit-exact. |
| R14 | **Calibration throughput.** [02 §4.4](plan/02-models.md) estimates 25–35 min (Qwen) for 4 M tokens; [04 §4.3](plan/04-pruning.md) derives ≈ 2.2 h (Qwen) / 4.5 h (GLM). | Plan with **04's numbers** (±50 %). The first real screening pass replaces both. | 04 counts probes, I/O and measured MoE density. |
| R15 | **Transcript extractor naming and counts.** 04: `claude_sessions` source; 08: `calib_data/transcripts.py`; 03: `calib_data/personal.py` + `scrub.py`. Counts range from 1,566 files / 1.5 GB (07/08) to 3,222 files / 5.8 GB incl. subagents (03). | **One module: `calib_data/personal.py` + `calib_data/scrub.py`** (03's design), shared by calibration, `evals.replay`, `drafter.collect` and `recover`. Use 03's count (newest, includes subagent files). | One privacy pipeline, audited once. |
| R16 | **Reranker.** 02: primary LLM listwise, then Nemotron-Rerank-8B; 08: Qwen3-Reranker-0.6B via mlx-vlm `/v1/rerank`. | v1 = **no rerank vs Qwen3-Reranker-0.6B vs LLM-listwise**, decided by `evals/rag.py` recall@20; Nemotron only if all three lose by ≥ 5 points. | Measure first; the 0.6B reranker is *below* its embedder on MTEB-Code [self-reported]. |
| R17 | **Tiered master vs separate builds for GLM.** 01: one tiered checkpoint, residency per profile. 02/04: separate REAP-K builds. | **Separate builds first** (the code exists), tiered master in Phase 9 (needs `build --tiers` + `serve/tiered.py`). | Ship value with existing tools; tiers are an optimization of the same idea (masking = pruning, [01 §2.5d](plan/01-memory.md)). |
| R18 | **Contract vs code.** `docs/contracts/build.md` says `--sens-power` defaults to p = 2; the code default is 1.0. | **p = 1 is correct** (NMSE is already squared error; [05 §2.3](plan/05-quantization.md)). Fix the contract text (backlog B-6). | — |

**GLM sizes you can build today vs after g128** (text-only, MTP kept, default floors; "today" rows from the planner,
the rest extrapolated with the measured slope; GB = 10⁹):

| keep K of 288 | pruned | today minimum (3b g64 floor) | after g128 (3b g128 floor) [est] | fits |
|---|---|---|---|---|
| 72 | 75 % | 38.46 | ≈ 36.0 | — (far below the 144 validated floor) |
| 86 | 70 % | 45.10 | ≈ 42.2 | — |
| 101 | 65 % | 52.20 | ≈ 48.8 | — |
| 115 | 60 % | 58.84 | ≈ 54.9 | Work (risky: below 144 GLM loops [self-reported]) |
| 128–130 | 55 % | ≈ 65.0–66.0 | ≈ 60.6–61.5 | Work at 64K, near-minimum bits; D3 experiment only |
| **144** | 50 % | **72.58** | **≈ 67.7** | Work only after g128 (or with a 2-bit cold tier via `--min-bits 2` + trunk `--floor`s) |
| 160 | 44 % | ≈ 80.2 | ≈ 74.7 | Focus |
| **176** | 39 % | ≈ 87.7 | ≈ 81.7 | **All-in** (64K) |
| 192 | 33 % | ≈ 95.3 | ≈ 88.8 | All-in, tight |
| 216 | 25 % | ≈ 106.7 | ≈ 99.3 | All-in only with 2-bit cold experts or SSD tail |

---

## 3. Concept map and learning path

### 3.1 The whole system on one page

```
                        ┌───────────────────────── THE BUDGET (01) ─────────────────────────┐
                        │ resident bytes = weights + KV/state + APC + prefill transient +    │
                        │ MLX cache + overhead ≤ profile cap;  disk and page cache are free  │
                        └──────────────────────────────┬─────────────────────────────────────┘
                                                       │ constrains every box below
   your traffic ──► (1) DATA (03) ──► (2) CALIBRATION (calib/) ──► stats per layer:
   (Claude Code       domains, personal,     stream 1 layer at a time      REAP, freq, co-activation,
    transcripts,      self-generated,        through the FULL-precision    NMSE grid per (bits,group),
    public agentic,   decontaminated,        model, ≤ 40 GiB               imatrix, ref/ top-64 logprobs
    code, math, RAG)  3 splits                                                     │
                                                                                   ▼
                       (3) PRUNING (04) ─────────► (4) QUANTIZATION (05) ──► model dir (build/)
                       which experts survive        how many bits each        + mtp/ drafter
                       (REAP / HOPE / sweep)        tensor keeps (knapsack,   + build_summary.json
                         ▲     │ re-calibrate on    floors, LSQ/GPTQ,
                         │     └ the pruned model   tiers)
                         │                                  │
                         │                                  ▼
                         │                    (5) RECOVERY (06): train only scales/biases
                         │                        and router rows toward the teacher (DWQ, KD)
                         │                                  │
                         │                                  ▼
                         │                    (6) SPEED (07): MTP / n-gram speculation,
                         │                        prefix cache, prefill chunking, kernels
                         │                                  │
                         │                                  ▼
                         │                    (7) SERVING (08): one memory-capped process,
                         │                        Anthropic+OpenAI APIs, router, tools, RAG, watchdog
                         │                                  │
                         │                                  ▼
                         └──── decisions ◄──── (8) EVAL LOOP (evals/, 08 §4.8): KL vs ref/,
                                                   guards (rare names, loops, tools, RULER),
                                                   tasks (paired), speed, replay, A/B → gate
```

The loop is the point: every arrow that goes *down* produces an artifact; every decision that goes *back up* (prune
less, more bits, different data weights, keep or drop a recovery stage) is made from an eval number, never from a
hunch.

### 3.2 The intuition behind each stage

**0. The budget — why "resident" is the word that matters.** Decode reads every *active* weight once per token, so
speed is set by active bytes; whether a model fits is set by *resident* bytes. Unified memory has no wall between the
model and your apps, so the model must stay far enough below the edge that macOS never compresses its weights.
Clean, file-backed pages (the Qwen PLE table) cost your apps nothing because the kernel can drop them; dirty pages
(`mx.load` copies) cost a lot. The plan's biggest single win comes from counting the right bytes.
→ [01 §1](plan/01-memory.md), [01 §2.1 exact accounting](plan/01-memory.md), [01 §2.2 page states](plan/01-memory.md),
[01 §4.2 profiles](plan/01-memory.md), [08 §2.1](plan/08-system.md), [research/mac.md §2–3](research/mac.md).

**1. Data — the calibration set is the specification.** REAP decides which experts survive, the NMSE grid decides
which tensors get bits, and DWQ decides where the remaining fidelity goes; all three are *averages over the
calibration distribution*. Calibrate on prose and the compressor keeps the prose experts and fires the code experts
(C4 calibration drove REAP code accuracy to 0 % in the REAP paper). Your own Claude Code sessions are the closest
thing to future traffic, because most tokens in an agent loop are *prefill*: your files, your tool outputs.
→ [03 §1](plan/03-calibration-data.md), [03 §2.1 every decision is an expectation](plan/03-calibration-data.md),
[03 §2.4 self-generation](plan/03-calibration-data.md), [03 §2.7 contamination](plan/03-calibration-data.md),
[04 §1](plan/04-pruning.md).

**2. Calibration — measuring a 321 B model on a 128 GB machine.** `calib/` never holds the model. It loads *one*
decoder layer from the source safetensors (dequantizing GLM's FP8), pushes every sample's cached hidden states
through it chunk by chunk with that layer's own cache, records statistics, writes the outputs to disk, frees the
layer, and moves on. The result is bit-identical to a real chunked prefill, at ≤ 40 GiB peak. The same trick powers
the pruning sweep ([04 §4.4](plan/04-pruning.md)), simulated quantization ([05 §4.3](plan/05-quantization.md)) and
teacher scoring for distillation ([06 §2.6](plan/06-recovery.md)).
→ [`contracts/calib.md`](contracts/calib.md), [03 §2.6 long sequences](plan/03-calibration-data.md),
[04 §4.3 cost](plan/04-pruning.md).

**3. Pruning — amputation, not blur.** An MoE layer is a library where each token borrows 8–10 books out of 288–512.
Load balancing guarantees there are no unused books, so every pruned expert is real capacity. REAP keeps the experts
that matter *when used* (gate × output norm over routed tokens), which protects rare specialists that frequency
pruning would kill. Damage concentrates on rare names, other languages, termination and agent efficiency — exactly
what averages and greedy pass@1 hide.
→ [04 §1](plan/04-pruning.md), [04 §2.2 substitution error](plan/04-pruning.md), [04 §2.3 REAP vs frequency vs
REAP-sum](plan/04-pruning.md), [04 §2.4 HOPE](plan/04-pruning.md), [02 §2.3 blur vs amputation](plan/02-models.md).

**4. Quantization — buying capacity with controlled noise.** Affine `b`-bit codes with a bf16 scale and bias per group
of `g` weights cost `b + 32/g` bits/weight. Each extra bit cuts error ≈ 4×; halving the group size buys far less per
byte, which is why g128 (not g32) is the efficient knob for experts. Bit allocation is a multiple-choice knapsack:
at the optimum every tensor's last upgrade buys the same error reduction per byte. The skeleton (routers, gates,
attention, embeddings — ≈ 3 % of parameters) is protected by floors; the routed experts (≈ 97 % of bytes) are where
bits are traded. Rounding smarter (LSQ refit, GPTQ) lowers error at identical bytes and speed.
→ [05 §1](plan/05-quantization.md), [05 §2.2 group size](plan/05-quantization.md), [05 §2.3 the knapsack and its
units](plan/05-quantization.md), [05 §2.4 what to protect](plan/05-quantization.md), [05 §2.6 rounding](plan/05-quantization.md),
[01 §2.5c half a bit per doubling of importance](plan/01-memory.md).

**Pruning × quantization — one budget, two knobs.** A gigabyte can buy more experts or more bits. At the optimum it
buys the same KL reduction either way. For Qwen the PLE insight moves this optimum all the way to "no pruning"; for
GLM it cannot.
→ [04 §4.5 the joint rule](plan/04-pruning.md), [02 §2.4 bigger-at-lower-bits](plan/02-models.md), [02 §4.3 GLM vs Qwen at
68 GB](plan/02-models.md).

**5. Recovery — the last 10–30 % of the gap, never a rescue.** Layer-local fixes cannot make layer ℓ+1 compensate
for layer ℓ's error; end-to-end distillation can. We train only the tiny, well-conditioned knobs — affine scales
and biases (DWQ) and router rows — against the uncompressed teacher's distribution, band by band, keeping a band
only if held-out KL improves. Same file, same size, same speed. Pruned knowledge does not come back.
→ [06 §1](plan/06-recovery.md), [06 §2.1–2.2](plan/06-recovery.md), [06 §2.5 memory math](plan/06-recovery.md),
[06 §4.8 schedule](plan/06-recovery.md).

**6. Speculative decoding and caching — more useful tokens per weight read.** Qwen decode is overhead-bound (~60 % of a
step is fixed cost), so verifying several drafted tokens in one pass pays, while shaving bits barely does. On a
fine-grained MoE each verified token routes to *different* experts, so drafts must be short — except when copying
from context (Edit `old_string`, file rewrites), where an n-gram drafter is right almost every time. In agent loops
the biggest latency is re-prefilling history; an exact-prefix cache turns a 40 s turn into 2 s.
→ [07 §1](plan/07-speculative.md), [07 §2.2 the overhead term](plan/07-speculative.md), [07 §2.3 the MoE verify
cost](plan/07-speculative.md), [07 §2.6 n-gram drafting](plan/07-speculative.md), [07 §2.8 prefix caching](plan/07-speculative.md).

**7. Serving — a model is not a system.** One process, a memory contract enforced by admission control (MLX's memory
limit is soft), native tool formats parsed every time, a watchdog that gives memory back before macOS takes it,
security that does not load 60 GB for an unauthenticated request, and logs that turn every regression into a number.
→ [08 §1](plan/08-system.md), [08 §4.2](plan/08-system.md), [08 §4.6 tools](plan/08-system.md), [01 §4.6 watchdog](plan/01-memory.md),
[08 §4.10 security](plan/08-system.md).

**8. The eval loop — statistics, not vibes.** KL against the full-precision reference is the most sensitive signal
per compute and ranks candidates; task suites are *guards* (164 HumanEval items can only resolve ≈ 6–10-point gaps);
paired tests (McNemar, bootstrap over samples) are mandatory; rare-entity, termination and turns-to-solve probes catch
pruning damage that averages hide.
→ [08 §2.7](plan/08-system.md), [04 §5 sample sizes](plan/04-pruning.md), [02 §5.1 which metric compares what](plan/02-models.md),
[08 §4.8 gate tiers and pass rules](plan/08-system.md), [`contracts/evals.md`](contracts/evals.md).

### 3.3 Suggested reading order (≈ 1 day of reading)

1. `PLAN.md`, then this §1–§3.
2. [01 §0–§2.1](plan/01-memory.md) (budget) → [02 §1–§2.4](plan/02-models.md) (why Qwen-unpruned wins).
3. [04 §1–§2.3](plan/04-pruning.md) → [05 §1–§2.4](plan/05-quantization.md) → [04 §4.5](plan/04-pruning.md) (the joint trade).
4. [03 §1–§2.5](plan/03-calibration-data.md) (data), then the `calib` contract.
5. [07 §1–§2.3, §2.6, §2.8](plan/07-speculative.md) (speed), [08 §1–§2.3](plan/08-system.md).
6. [06 §1–§2.2](plan/06-recovery.md) (recovery) — only needed by Phase 6.
7. Every chapter's §5 (decision criteria) and §6 (failure modes) before running that chapter's phase.

---

## 4. Phased roadmap

Rules that apply to every phase ([PLAN.md](../PLAN.md) "Rules"): never load > 8 GB of weights during development;
test code on tiny real-architecture fixtures (`tests/fixtures/make_tiny.py`); **one GPU-heavy job at a time**; heavy
calibration, builds, evals on real models and training run **overnight with the server stopped from the dashboard**
(http://127.0.0.1:8090 → Stop; restart it in the morning). Do not touch the HF cache directories of downloads in
progress except `config.json` / tokenizer files. Do not edit `pyproject.toml` / `uv.lock`; list new dependencies.

Conventions used in the commands:

```bash
SNAP_G=models/src-glm-5.3-flash                         # symlink to the FP8 snapshot (exists)
SNAP_Q=models/src-qwen3.8-flash-next                     # create once the download finishes:
#   ln -s ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/<rev> models/src-qwen3.8-flash-next
```

★ marks a flag or module that does not exist yet (it is in the backlog, §6). Everything unmarked exists today.
Time / disk / memory figures are **[est]** unless tagged; the phase's own measurements replace them.

### Phase overview

| Phase | Goal | Dev time | Machine time | Gated on |
|---|---|---|---|---|
| 0 | Done this session: baseline serving, pipeline, research, chapter drafts | — | — | — |
| 1 | Foundations: review, P0 safety fixes, planner reconcile (g128, resident scope, floors, weighting), calib v2 stats, eval gate v1 | 7–10 days | minutes (tiny fixtures) | — |
| 2 | Calibration data v2: splits, decontam, personal pipeline, new sources, self-generation | 4–6 days | 1 evening (+ optional cloud run) | Phase 1 (C1/C2 code) |
| 3 | Real calibration: GLM shakedown now; Qwen screening + final when the download lands; baselines' KL | 1 day | 2–3 nights | Qwen download; Phase 1–2 |
| 4 | **Qwen primary build** (E-Q1: experts vs bits at equal resident bytes) → promote | 1–2 days | 3–4 nights | Phase 3 |
| 5 | Speed on the promoted build: MTP sweep, APC hits, prefill step, n-gram drafter | 4–7 days | 2–3 sessions | Phase 4 |
| 6 | Recovery (DWQ, router KD, on-policy) on the promoted build | 4 days | ~7 nights | Phase 4; `recover/` |
| 7 | GLM secondary for Focus/All-in (and the D3 equal-bytes experiment) | 2–3 days | 4–6 nights | GLM fixes; Phase 3 |
| 8 | The complete system: router, observability, watchdog, launchd, tools guard, RAG, T2 gate, replay, A/B | 10–15 days | nights for T2 | Phase 4 |
| 9 | Most ambitious: tiers + live profile switching, SSD-streamed unpruned GLM, adaptive residency, EAGLE-3, custom kernels | weeks | many nights | everything above |

Phases 5, 6, 7 and 8 can interleave; 6 and 7 compete for nights, so do Phase 6 first only if Phase 4 promoted a
build, and Phase 7 first if GLM's escalation value (D2) is what you want to learn next.

---

### Phase 0 — done this session (2026-10-20)

- **Research** (adversarially verified): [mac](research/mac.md), [decode](research/decode.md),
  [quant](research/quant.md), [reap](research/reap.md).
- **Baseline deployed:** sh0wie REAP-288 4-bit + bf16 MTP drafter on stock mlx-vlm via `scripts/run_server.py`
  (memory caps, wiring 0, API key on all routes incl. `x-api-key`); launchd control daemon + dashboard
  (`control/`, `scripts/install-control.sh`); `scripts/claude-local.sh` for Claude Code.
- **Runtime tuned and measured** (`results/runtime_tune.json`): ~30 tok/s plain, 54–58 tok/s at MTP block 2, peak
  ~48 GB (PLE streams from SSD), APC 4.2 s → 1.25 s on a 1.8K repeat.
- **Modules built** (`calib_data/`, `calib/`, `build/`, `evals/`, `serve/`) with contracts in `docs/contracts/`;
  **192 tests pass**; `scripts/pipeline_tiny.sh` runs the whole chain for both architectures.
  Integration fixes: REAP test reference; serve's detection of build's `mtp/` drafter dir. Builds need
  `--drop-vision` for text-only use. `build/` writes the MTP head as `<model>/mtp/` (`load_drafter`).
- **GLM source** downloaded (`models/src-glm-5.3-flash`); `calib.check` passes all 45 layers.
- **Planner minima for GLM** computed (R1 table). The R-G1 / planner gap identified.
- **Chapter drafts** 01–08 (this plan reconciles them).
- **Litwein `*-oQ3e-*` builds are oMLX-only** (gibberish on mlx-vlm): `models/baseline-qwen-reap320` is usable only as
  a *reference for size/config*, not as a served baseline.

---

### Phase 1 — Foundations: make the tools correct before trusting their numbers

**Goal.** Every number later phases rely on comes from code reviewed and tested here: memory accounting, the planner,
the calibration statistics, and the gate.

**Steps.**

1. **Review the agent-built modules** (they are unreviewed, [PLAN.md](../PLAN.md) Status). One module at a time,
   contract first, then code, then tests. Fix what the review finds.
   ```bash
   uv run pytest -q                       # 192 pass today
   scripts/pipeline_tiny.sh both          # must print PIPELINE OK: glm qwen
   uv run python -m calib.check --model glm --src $SNAP_G
   ```
2. **Serving safety (P0)** — [08 §4.2, §4.10, §4.11](plan/08-system.md), backlog S-1…S-6:
   `serve/memory.resident_weights_bytes` ★ (exclude `ple-store.json` tensors), `serve/budget.py` ★ (the KV/transient
   formulas of [01 §4.1](plan/01-memory.md)), wiring default 0, auth before model load, Host check, CORS removal,
   SSE `ping` test, `serve/glm_fix.py` ★ (R13). Same resident-aware rule in `evals/common.weights_gb` (E-1), otherwise
   evals refuse every Qwen build with a mmapped PLE (≈ 90 GB on disk > `--max-weights-gb 80`).
3. **Planner reconcile (the explicit early task, R1)** — backlog B-1…B-6:
   - g128: `calib/sens.py GROUPS = (32, 64, 128)`, `build/alloc.DEFAULT_GROUPS += 128`, experts restricted to {64, 128}
     ([05 §4.1–4.2](plan/05-quantization.md)).
   - `build.make --budget-scope {disk,resident}` ★ ([01 §4.1](plan/01-memory.md)); PLE counts against disk, not budget.
   - Per-family floor tables in `build/families.py` ([05 §2.4](plan/05-quantization.md)) replacing
     `DEFAULT_FLOORS = (lm_head ≥ 6, embed ≥ 4)`.
   - `--weighting energy` ★ and importance-weighted expert aggregation ([05 §2.3](plan/05-quantization.md)).
   - **Reconcile R-G1 and R-Q1 against `--dry-run`.** Re-derive both recipes as planner plans and write the
     discrepancy table (expected: R-G1 ≈ the post-g128 floor; R-Q1 reproducible within ≈ 1 %).
   ```bash
   # floor check (fails fast and prints "the smallest possible build is X GB")
   uv run python -m build.make --model glm --src $SNAP_G --prune 144 --drop-vision --budget-gb 1 --dry-run
   # plan (writes nothing); --groups 64,128 is accepted only after B-1 (★)
   uv run python -m build.make --model glm --src $SNAP_G --prune 144 --drop-vision --budget-gb 68 \
       --groups 64,128 --dry-run --summary-json plans/glm-k144-68.json
   ```
4. **Calibration statistics v2** (`FORMAT_VERSION = 2`) — backlog K-1…K-12, all on tiny fixtures:
   `expert_reap_sq`, `coact_sum/cnt` (HOPE), `expert_out_norm_max` / `expert_act_max` (super-expert guard),
   `route_trace` (cache simulation), `prune_curve`, `block_influence`, `resid_sq`, MTP stage, LSQ-matched sens
   (`--sens-rounding`), position-stratified reservoirs, `target_logprob` for calib samples, `--ref-splits` ★
   ([04 §4.1](plan/04-pruning.md), [05 §4.1](plan/05-quantization.md), [03 §4.0 C7](plan/03-calibration-data.md)).
5. **Gate v1** — `evals/gate.py` ★ (T0/T1), KL paired-bootstrap CI vs a baseline, `evals/flips` (McNemar),
   rare-entity/identifier-copy probe (≥ 200 items) and termination/loop rate ([08 §4.8](plan/08-system.md),
   [04 §5](plan/04-pruning.md)).
6. **Accounting validation on the running baseline** (one evening window, server already loaded): compare
   `footprint -p <pid>` vs predicted F at 1K/32K/128K; then `memory_pressure -S -l warn` and check decode holds
   ([01 §5 T1](plan/01-memory.md), [08 §4.2a](plan/08-system.md)). This is the measurement that licenses R2.

**Estimates.** 7–10 dev days. Memory < 2 GB for tests. Disk negligible.

**Exit criteria.**
- `uv run pytest -q` green with new tests; `scripts/pipeline_tiny.sh both` → `PIPELINE OK`.
- Tiny fixture: `coact` equals brute force; `prune_curve(k=E) = 0` and non-increasing; LSQ sens ≤ RTN everywhere;
  g128 cells NaN where `in % 128 ≠ 0` ([04 §4.1](plan/04-pruning.md), [05 §4.1](plan/05-quantization.md)).
- Planner: written bytes = predicted bytes (already asserted); R-G1/R-Q1 discrepancy table committed to
  `docs/plan/05-quantization.md` (§4.6 note).
- Accounting: |F_pred − footprint| < 2 GiB at 1K/32K/128K on REAP-288 ([01 §5 T1](plan/01-memory.md)); decode under
  `memory_pressure -l warn` within 10 % of idle ([08 §5](plan/08-system.md)).
- `serve --dry-run` on REAP-288 reports ≈ 38.6 GiB resident, not 68.4 ([08 §2.1](plan/08-system.md)).

---

### Phase 2 — Calibration data v2: make the data *yours*, and leak-free

**Goal.** A deterministic, decontaminated corpus with three splits (`calib` / `dwq` / `eval`), six domains
(`code`, `agentic`, `personal`, `reasoning`, `longctx` 32K–128K, `general` floor), and an on-policy assistant side.

**Steps** ([03 §4.2–4.7](plan/03-calibration-data.md); backlog D-1…D-8):

1. **C1 + C2:** content-hash split key, the `dwq` split, decontamination against every benchmark `evals/` uses.
   Gate: a planted HumanEval prompt and an OpenR1/OpenThoughts duplicate are both caught.
2. **C5:** new public sources (SWE-smith `tool` split, Nemotron-SWE byte ranges, SWE-rebench-OpenHands,
   Nemotron-Agentic, Nemotron-Terminal-Corpus, LongAlign, fineweb-edu/fineweb-2), `longctx` 32K–128K,
   `MAX_LEN["agentic"] = 32768`. Gate: `--smoke --source hf` renders every recipe in both templates.
3. **Personal pipeline (C3).** **You** run the key-only schema probe P0 first (prints key paths and types, never
   values). Then `calib_data/personal.py` ★ + `scrub.py` ★ on synthetic fixtures, then the real run (CPU, ~5–10 min
   for 5.8 GB). Time split (`--cutoff`), per-project cap 15 %, near-dup removal across the split. Review 50 scrubbed
   snippets yourself (`--review 50`). Everything under `artifacts/personal/` (0700), never sent anywhere.
   ```bash
   uv run python -m calib_data.personal ★ --projects ~/.claude/projects --out artifacts/personal/store \
       --cutoff 2026-10-01 --max-project-share 0.15
   ```
4. **Self-generation (C4).** Option B first (local proxy = the running server, one evening, personal one-step
   regeneration ≈ 0.6 M generated tokens): `calib_data.selfgen ★ --endpoint http://127.0.0.1:8080/v1 --one-step`.
   Option A (rented 8×H100 or 4×H200, public prompts only, ≈ 1.5 h, ≈ $25–150) is optional and decided by
   experiment D3 ([03 §4.5](plan/03-calibration-data.md)).
5. **Final corpus per model** (≈ 4.4 M calib / 1.9 M dwq / 1.1 M eval tokens, [03 §2.5](plan/03-calibration-data.md)):
   ```bash
   uv run python -m calib_data.build --model qwen --out artifacts/qwen/data-final \
       --mix code=0.8,agentic=1.0,personal=0.8,reasoning=0.8,longctx=0.8,general=0.2 \
       --longctx 32768-131072 --dwq-samples 400 ★ --personal artifacts/personal/store ★ --decontam evals ★
   ```
   (`--mix` is relative sample weight → token *precision*; importance comes later from `build --domain-weights`, R3.)

**Estimates.** 4–6 dev days + 1 evening of local generation. Download ≈ 1 GB, cache ≈ 2 GB; output ≈ 50 MB per model.
CPU-only except self-gen (uses the running server; no extra RAM beyond KV).

**Exit criteria.** Leak test passes; scrub unit tests catch 100 % of planted fake secrets; manifest records
`format_version: 2`, sources, decontam drops and scrub counts; the expert-coverage diagnostic is computable after
Phase 3 (every expert ≥ 512 pooled rows); proxy self-gen rejection rate < 20 % ([03 §5–6](plan/03-calibration-data.md)).

**Shortcut.** Phase 3's *screening* pass can run on today's v1 data (`artifacts/{glm,qwen}/data`, ≈ 2.3 M calib
tokens) while Phase 2 is in progress. Only the *final* pass needs v2 data.

---

### Phase 3 — Real calibration (overnight, server stopped)

**Goal.** Full-precision statistics and `ref/` logprobs for both models, and the first real KL numbers for the
deployed baseline.

**Steps.**

1. **GLM shakedown — can start now** (the source is here; Qwen is still downloading). A 1 M-token screening pass is the
   first real-scale test of `calib/` streaming:
   ```bash
   uv run python -m calib.run --model glm --src $SNAP_G --data artifacts/glm/data \
       --out artifacts/glm/calib-screen --mem-gb 40          # resumable at layer granularity
   ```
   ≈ 70 min compute+I/O, ≤ 40 GiB, ≈ 1.5 TB SSD writes (hidden states) [est, 04 §4.3]. Check ≥ 1.2× the scratch
   size is free first (≈ 66–262 GB depending on tokens, [08 §4.9](plan/08-system.md)).
2. **Qwen** (after the download and `calib.check --model qwen --src $SNAP_Q`): screening pass (≈ 35 min) on v1 data,
   then the **final pass** on v2 data with the v2 flags:
   ```bash
   uv run python -m calib.run --model qwen --src $SNAP_Q --data artifacts/qwen/data-final \
       --out artifacts/qwen/calib --mem-gb 40 --expert-tokens 512 \
       --sens-rounding rtn,lsq ★ --ref-splits eval,dwq ★ --pos-strata 4096,32768 ★
   ```
   ≈ 2.2 h + ≈ 0.7 h for the dwq ref; ≈ 300 GB scratch peak; several TB written [est, 03 §4.6]. Use
   `--layers-per-stage 2` ★ / in-RAM shards ★ when they land to cut SSD wear ([04 §4.3](plan/04-pruning.md),
   [06 §2.6](plan/06-recovery.md)).
3. **Baseline KL** (server stopped; the baseline is loaded in-process by evals):
   ```bash
   uv run python -m evals.kl --model models/qwen-flash-next-reap288 --ref artifacts/qwen/calib/ref
   uv run python -m evals.speed --model models/qwen-flash-next-reap288 --contexts 1024,32768,131072 \
       --mtp-modes off,on --draft-model models/qwen-flash-next-mtp-drafter
   uv run python -m evals.all --model models/qwen-flash-next-reap288 --suites tools,code,longctx -- --limit 200
   ```

**Exit criteria.** `state.json` complete for every layer; `ref/index.json` lists all eval samples; split-half mask
stability ≥ 85–90 % at K = 320 (Qwen) / 144 (GLM) ([04 §4.2](plan/04-pruning.md)); expert-coverage diagnostic passes
([03 §5](plan/03-calibration-data.md)); a per-domain KL table for REAP-288 exists in `results/qwen-flash-next-reap288/kl.json`.
This KL is **the number every Qwen candidate must beat.**

---

### Phase 4 — The Qwen primary build (the first real win)

**Goal.** Settle "experts vs bits" for Qwen at equal resident bytes, build the winner, gate it against REAP-288,
promote it to the Work profile.

**Background.** [04 §4.0 "The Qwen insight", §4.5 E-Q1](plan/04-pruning.md); [02 §4.4 step 3](plan/02-models.md);
[05 §4.6 R-Q1](plan/05-quantization.md); [01 §4.3 W1](plan/01-memory.md).

**Steps.**

1. **Plan, never write first.** Resident target ≈ 60 GB. Today `--budget-gb` counts the PLE at 4b g32 = 32.0 GB, so
   use `--budget-gb 92` (≈ 60 resident); after B-2 use `--budget-scope resident --budget-gb 60.5`:
   ```bash
   B="uv run python -m build.make --model qwen --src $SNAP_Q --calib artifacts/qwen/calib --drop-vision"
   $B --prune none --mtp intact --budget-gb 92 --dry-run --summary-json plans/qwen-k512-r60.json
   $B --prune 448  --mtp intact --budget-gb 92 --dry-run --summary-json plans/qwen-k448-r60.json
   $B --prune 384  --mtp intact --budget-gb 92 --dry-run --summary-json plans/qwen-k384-r60.json
   $B --prune 320  --mtp intact --budget-gb 70 --dry-run --summary-json plans/qwen-k320-r38.json   # lean control
   ```
2. **Rank before building** (once `calib/sweep.py` ★ exists): pure-pruning KL for K ∈ {512, 448, 384, 352, 320, 288}
   and simulated-qdq KL for each plan, one streaming pass, ≈ 1 h per 8-variant batch ([04 §4.4](plan/04-pruning.md),
   [05 §4.3](plan/05-quantization.md)). Without the sweep, build the top 2–3 directly.
3. **Build finalists** (15–30 min each; ≈ 90 GB disk each, ≈ 60 GB with APFS-cloned PLE shards ★):
   ```bash
   $B --prune none --mtp intact --budget-gb 92 --out models/qwen-k512-r60 --summary-json results/qwen-k512-r60/build.json
   ```
4. **Gate** (server stopped; evals load the build in-process, `--max-weights-gb 100` until E-1 lands):
   ```bash
   for s in kl speed tools code longctx; do uv run python -m evals.$s --model models/qwen-k512-r60 \
       --max-weights-gb 100; done
   uv run python -m evals.gate ★ --candidate models/qwen-k512-r60 --baseline models/qwen-flash-next-reap288 --tier T1
   ```
5. **Promote** through the dashboard: set model `models/qwen-k512-r60`, MTP on (the build's own `mtp/` dir is the
   drafter, R9), cap 80, block 2 → re-sweep (R5). Keep REAP-288 as `models/previous` for rollback
   ([08 §4.9](plan/08-system.md)).

**Estimates.** Builds 2–4 h of machine time total; T1 gate ≈ 6 h per candidate [08 §4.8]; ≈ 3–4 nights for 3
finalists. Disk ≈ 200–300 GB. Build peak ≈ 6–7 GB; eval peak = the model's own footprint (≈ 70 GiB at 128K).

**Exit criteria (all, else keep REAP-288).**
- Peak at 128K ≤ 76 GB, plain decode ≥ 90 % of the best candidate ([02 §5.2 D1](plan/02-models.md)).
- Per-domain KL better than REAP-288 with bootstrap 95 % CI excluding 0 in ≥ 3 of 4 domains, no domain worse beyond its
  CI, every domain ≤ 1.5× the best ([05 §5](plan/05-quantization.md)).
- Guards: rare-entity copy ≥ 95 % of the K = 512 reference; loops ≤ ref + 1 pp; tool validity within 2 pp; no
  McNemar-significant regression on pooled code + tools ([04 §5](plan/04-pruning.md)).
- MTP α ≥ 0.65 on agent traces, else go to Phase 5 S5 before promoting ([07 §5](plan/07-speculative.md)).

**Then, within Phase 4, better quantization at the same bytes** (each kept only if it passes its gain row in
[05 §5](plan/05-quantization.md)): `--rounding lsq` ★ (expect −12…15 % KL), GPTQ on ≤ 3-bit units, energy-weighted
objective vs uniform, PLE nvfp4 with pre-scaling ★ (−3.2 GB disk), importance tiers ★ (Phase 9 runtime shim).

---

### Phase 5 — Speed on the promoted build

**Goal.** Turn the promoted build into the fastest *agentic* system, measured on agent traffic, not prose.

**Steps** ([07 §4](plan/07-speculative.md)):

1. **S0 — measure the right thing:** `evals/specbench.py` ★ (trace replay, per-segment acceptance) and the verify-cost
   curve T(k) at kv 1K/32K/128K. This fits `t0` and replaces the cost model's assumptions (R5, R6).
2. **S2 — MTP block sweep** on the new head: `scripts/tune_runtime.py models/<build> results/runtime_tune_<build>.json`
   (existing) and specbench `--blocks 2,3,4,5 --temps 0,0.6`.
3. **S1 — prefix cache that actually hits:** log the first-divergence token index between consecutive Claude Code
   requests (indices only, never text); `CLAUDE_CODE_ATTRIBUTION_HEADER=0` if needed; checkpoint at request end.
4. **S4 — prefill step:** `evals.speed --prefill-step-size 2048|4096|8192` within the profile's peak.
5. **S3 — n-gram / suffix drafter** `serve/ngram.py` ★ with de-numbered tool-result indexing, then the hybrid
   n-gram→MTP policy ([07 §2.6, §4.3](plan/07-speculative.md)). Requires the speculative loop to accept variable-length
   drafts; verify on the tiny model first.
6. **Investigate the 30 vs 44–48 tok/s plain-decode gap** on an idle GPU (Instruments Metal System Trace; dispatch
   count per layer), which is also K1's profile step ([07 §4.8](plan/07-speculative.md)).

**Estimates.** 4–7 dev days; 2–3 measurement sessions (server stopped or using the running server).

**Exit criteria.** Warm-turn TTFT ≤ 3 s at 60K history for +3K new tokens; ≥ 99 % prefix reuse per turn except at
compactions; MTP block = argmax tok/s at serving temperature; n-gram drafter kept only if blended agent-turn decode
≥ +15 % and no segment type regresses > 3 % ([07 §5](plan/07-speculative.md)).

---

### Phase 6 — Recovery (distillation) on the promoted build

**Goal.** Recover 10–30 % of the remaining KL gap by training only scales/biases and router rows, with a held-out gate
at every stage. **Never prune deeper because recovery exists.**

**Steps** ([06 §4.1–4.8](plan/06-recovery.md)): write `recover/` ★ (≈ 600–900 lines + tiny-fixture tests, days 1–4),
then one stage per night, server stopped:

| Night | Stage | Peak | Exit (else drop the stage) |
|---|---|---|---|
| 1 | Teacher top-64 on 6–10 M public + 2 M personal tokens (`recover.teacher` ★, in-RAM shards) | ≤ 40 GiB | token count; spot check vs `ref/` KL ≤ 1e-4 |
| 2 | A: layer-local (s, β) + router rows toward the teacher trajectory (`recover.local` ★) | ≤ 40 GiB | KL −3 % or better |
| 3–4 | B: band-wise end-to-end DWQ, 4 MoE layers per band, gated per band (`recover.dwq` ★) | ≤ 100 GiB (Qwen ≈ 79–85 GB) | KL −8 % or better; **stop the whole track if < 3 %** |
| 5 | C: router KD (R0 then R2), GLM correction bias frozen (`recover.router` ★) | All-in | KL −2 %; selection change ≤ 5 %/layer |
| 6 | D: on-policy personal round (`recover.onpolicy` ★) | All-in | personal KL −3 %, no public regression, canary probe clean |
| 7 | MTP re-alignment (`drafter.train_mtp` ★, [07 §4.5](plan/07-speculative.md)) + full gate | All-in | α ≥ start − 1 pt; `evals.gate` PASS |

**Estimates.** ≈ 4 dev days + 7 nights per model. Do it for **one** model (the one being promoted). Disk ≈ 5–20 GB of
teacher targets; one model dir per stage (≈ 60–90 GB each; delete losers).

**Exit criteria.** Promote the last stage that passed only if it passes the T1 gate against the un-recovered build
([06 §5](plan/06-recovery.md)). Cloud QAT ([06 §7.2](plan/06-recovery.md)) only if stage B shows a clear
"more tokens → more gain" slope.

---

### Phase 7 — GLM as the secondary model (Focus / All-in)

**Goal.** Find out whether GLM earns a slot: does its best feasible build solve ≥ 15 % of the primary's failures?

**Prerequisites.** `serve/glm_fix.py` (R13) — mandatory, otherwise +22–24 GiB at ≥ 32K and All-in does not fit
([08 §4.2c](plan/08-system.md)); GLM long-prompt prefill step 512–1024 ([decode Rec 2](research/decode.md)); GLM
calibration final pass (≈ 4.5 h + 1.4 h dwq ref, [03 §4.6](plan/03-calibration-data.md)).

**Steps** ([02 §4.4 step 4](plan/02-models.md), [04 §4.5 E-G1, §4.6–4.7](plan/04-pruning.md)):

1. Sweep (if `calib.sweep` exists) K ∈ {224, 192, 176, 160, 144, 128} with REAP-mean vs REAP-sum vs HOPE at the budget K.
2. Sequential re-calibration on the chosen mask (`calib.run --keep-mask` ★), ≈ 4.5 h.
3. Build the All-in candidate(s) — feasible today:
   ```bash
   G="uv run python -m build.make --model glm --src $SNAP_G --calib artifacts/glm/calib-k176 --drop-vision"
   $G --prune 176 --mtp prune --budget-gb 90 --dry-run --summary-json plans/glm-k176-r90.json
   $G --prune 176 --mtp prune --budget-gb 90 --out models/glm-k176-r90
   ```
   and the D3 Work experiment: today K ≈ 128–130 at `--budget-gb 68` (near-minimum bits, below the validated floor)
   or, after B-1, K = 144 at 68 (R-G1).
4. Gate on the **escalation set** (items the primary failed + τ²-style tool loops + knowledge QA) and the full T1
   tier with `--max-weights-gb 100 --memory-limit-gb 104`.

**Estimates.** 2–3 dev days; 4–6 nights. Disk ≈ 90 GB per GLM build.

**Exit criteria (D2).** Keep GLM as secondary only if it solves ≥ 15 % of the primary's failures and passes
termination/loop guards (GLM loops below 144 [self-reported]); peak at the chosen context ≤ 100 GiB. Otherwise drop
GLM and spend All-in on Qwen K = 512 @ 4.5 bpw + best-of-n with test selection ([02 §7 S3](plan/02-models.md)).

---

### Phase 8 — The complete system

**Goal.** Daily-driver reliability: the right thinking effort per request, tool calls that parse, memory that yields,
regressions that show up as numbers, documents beyond the context window.

**Steps** ([08 §4](plan/08-system.md), [01 §4.2, §4.6](plan/01-memory.md)), in priority order:

1. **Migrate production to `serve/`** (R7) once S-1…S-6 are done and T0 passes: control daemon launches
   `uv run python -m serve --model … --port 8080 --api-key …`.
2. `serve/router.py` ★: aliases `local-fast` / `local-main` / `local-deep`, effort → thinking, two-phase thinking
   budget that keeps MTP, context-too-long errors; update `scripts/claude-local.sh` env (`ANTHROPIC_DEFAULT_*_MODEL`).
3. `serve/obs.py` ★ JSONL request log (no prompt text) + `serve.report`; alerts on α drop, APC hit-rate drop, swap.
4. `serve/watchdog.py` ★: 01's L1 shed → L2 downshift → L3 yield ladder with hysteresis (01 §4.6 supersedes 08's
   shorter table); log to `logs/watchdog.jsonl`.
5. Profiles: `scripts/profile.sh work|focus|allin|calibrate` ★ and `POST /serve/profile` ★ behind the key.
6. Tools: `serve/tools_guard.py` ★ (validate + coerce + one grammar-forced resample); lazy grammar `serve/grammar.py`
   ★ only if malformed calls > 0.5 % ([08 §4.6](plan/08-system.md)).
7. Eval gate T2 + `evals/replay.py` ★ (personal, time-split) + `evals/ab.py` ★ + `evals/lcb.py`, `evals/swe.py` ★.
8. `rag/` ★ + `evals/rag.py` ★ (EmbeddingGemma-2 in-process, BM25 via SQLite FTS5, RRF, R16 reranker decision);
   needs `pypdf` (new dependency to list).
9. launchd socket activation + exit-on-idle; `models/current` / `models/previous` symlinks; `scripts/disk_report.py` ★.

**Estimates.** 10–15 dev days; T2 ≈ 24 h of machine time per promotion.

**Exit criteria.** T0 smoke on every server change; watchdog T9 (L1 within 2 s, L3 before swap-out > 1 GiB, ≤ 1
transition / 60 s, [01 §5](plan/01-memory.md)); co-tenant T8 (dev build time +≤ 10 %, zero sustained swap-out in
Work); BFCL parse failures < 1 %; RAG recall@20 measured on your own questions.

---

### Phase 9 — The most ambitious: push the machine's limits

Each item has its own kill criterion in its chapter; do them in roughly this order of value per effort.

1. **Importance tiers + masking as runtime pruning** — `build --tiers` ★ + `serve/tiered.py` ★
   (`TieredSwitchGLU`): hot/warm/cold bit tiers in one checkpoint, router masks for non-resident experts, live
   profile switching in seconds. T2 test: masking = pruning to 1e-5 ([01 §4.4–4.5, §5 T2](plan/01-memory.md);
   [05 §2.5](plan/05-quantization.md)). Expected: −43 % routed-weighted error at equal bytes (idealized).
2. **GLM tiered master with an SSD-streamed tail (A2):** all 288 experts available, 216 resident, 72 at 2.5 bpw on
   SSD via a slab `pread` path (not stock `OffloadedSwitchGLU`); ≈ 28–33 tok/s est. Decide first with
   `scripts/sim_expert_cache.py` ★ on `route_trace` ([02 §4.4 step 6](plan/02-models.md), [01 §2.4c, §4.3 A2](plan/01-memory.md)).
3. **File-backed zero-copy weights** (`mx.asarray(np.memmap, copy=False)`, measured to work) for warm/cold tiers,
   after the T5 pressure soak ([01 §2.6, §5 T5](plan/01-memory.md)).
4. **Session-adaptive residency** (mask now, fetch async) ([01 §7.1](plan/01-memory.md)).
5. **One knapsack over {drop, 2b, 3b, 4b} per expert** ([04 §7.1](plan/04-pruning.md), [05 §7.3](plan/05-quantization.md)).
6. **Chunked delta-rule op with a custom VJP** — unlocks 8–16K training sequences and 3–5× faster recovery
   ([06 §7.1](plan/06-recovery.md)).
7. **Personal MTP fine-tune → EAGLE-3 on Qwen** (adopt only at ≥ +10 % tok/s) ([07 §4.5–4.6](plan/07-speculative.md)).
8. **Kernels K1–K5** (compile shape-static decode sub-blocks, fused decode MoE kernel, QSA block-key buffer)
   ([07 §4.8](plan/07-speculative.md)).
9. **Cost-aware hybrid speculation with expert-overlap pricing** ([07 §7.1](plan/07-speculative.md)); grammar-aware
   verification ([08 §7.4](plan/08-system.md)).
10. **Nightly continual self-distillation** from opted-in traffic ([06 §7.5](plan/06-recovery.md)); 1M-token GLM
    context once the DSA transient is engineered away ([01 §7.7](plan/01-memory.md)).

---

## 5. Experiment matrix and decision rules

### 5.1 The matrix

Axes: **model × experts kept (K) × resident bytes (or bits) × rounding/tiers × recovery**. Each cell is planned with
`--dry-run`, ranked by the streaming sweep where available, and only the leaders are built and gated.

**Qwen3.8-Flash-Next** (E = 512, top-10; resident GB ≈ K × 0.236 × bpw/8 + ≈ 4.6 trunk + ≈ 1.1–1.4 MTP; PLE on disk)

| ID | K | Expert bpw (avg) | Resident GB | Profile | Rounding / tiers | Recovery | Question it answers | Chapter |
|---|---|---|---|---|---|---|---|---|
| Q0 | 288 | 4.5 (sh0wie) | ≈ 41 | Work | RTN | none | **the baseline to beat** | 01 W0 |
| Q-lean | 320 | ≈ 3.3–3.5 | ≈ 38 | Work-lean | RTN → LSQ | — | is pruning damage negligible with our calibration? | 02 §4.4 |
| **Q1** | **512** | **≈ 3.6** | **≈ 56–60** | **Work** | RTN → LSQ; tiers later | B → D | **E-Q1 main arm: no pruning** | 04 §4.5, 05 R-Q1 |
| Q2 | 448 | ≈ 4.1 | ≈ 60 | Work | same | — | E-Q1: light pruning, more bits | 04 §4.5 |
| Q3 | 384 | ≈ 4.25–4.9 | ≈ 54–60 | Work | same | — | E-Q1: moderate pruning, ≈ 4-bit | 04 §4.5 |
| Q-auto | auto | solver | ≈ 60 | Work | — | — | does `--prune auto` agree with the sweep? (λ must be calibrated first) | 04 §4.5 step 4 |
| Q1+ | 512 | ≈ 4.0 | ≈ 68 | Focus | LSQ, tiers | — | only if Q1 misses its KL target by > 20 % | 05 R-Q1+ |
| Q-allin | 512 | ≈ 4.5 | ≈ 74 | All-in | LSQ | — | does Qwen saturate near 4 bpw? (GSQ: 3.5 ≈ bf16) | 02 §4.2, 01 F1 |
| Q-disk | 352 / 320 | 3b g128 / 3b g64 | ≈ 34 (+29 PLE) | fallback | LSQ | — | R-Q2: if mmapped PLE fails under pressure | 05 R-Q2 |
| Q-merge | 384 / 352 | — | — | Work | REAM merge | — | stretch: does merging beat pruning at ≤ 30 %? | 04 §7.3 |

**GLM-5.3-Flash** (E = 288, top-8; fully resident; today's minima in §2)

| ID | K | Expert bpw | Resident GB | Profile | Feasible | Question | Chapter |
|---|---|---|---|---|---|---|---|
| G-D3 | 128–130 / 144 | ≈ 3.5 / ≈ 3.1 | ≈ 66–68 | Work | today / after B-1 | D3: equal bytes vs Q1 — expected to lose | 02 §5.2 D3, 05 R-G1 |
| G-160 | 160 | ≈ 3.25–3.5 | ≈ 75–80 | Focus | today (near min) | middle option | 01 F2, 04 E-G1 |
| **G-176** | **176** | **≈ 3.25–3.5** | **≈ 82–90** | **All-in** | **today** | **main secondary candidate** | 02 §4.2 P2 |
| G-192 | 192 | ≈ 3.15–3.25 | ≈ 89–95 | All-in (tight) | today (at min) | fewer experts pruned vs more bits | 02 §4.4 step 4 |
| G-216t | 216 (+72 SSD) | 3.0 resident / 2.5 tail | ≈ 88 + 24 file | All-in + Stream | Phase 9 | unpruned GLM at interactive speed? | 01 A2 |
| G-full | 288 | 3.5 | 143 on disk | Stream | today (stock offload) | overnight one-hard-question mode | 02 §4.2 P3 |

**Pruning-criterion sub-matrix** (sweep batch 2, at the budget K): REAP-mean, REAP-sum, REAP-shrunk, HOPE-cond,
HOPE-unc, max-rank + per-domain cores, frequency (control), global-norm; plus structure variants (non-uniform ±32
knapsack, top-k+1 among survivors, per-layer output scalar, block skip) ([04 §4.4](plan/04-pruning.md)).

**Data sub-matrix** D1–D6 ([03 §5](plan/03-calibration-data.md)): workload vs generic; personal weight {0, 0.67, 1, 2}
(R3); self-generated vs original assistant text; longctx 16–64K vs 32–128K; general floor 0 vs 0.3; volume 250K / 1M /
2M per domain.

**Quantization sub-matrix** ([05 §4.4 Q2–Q4](plan/05-quantization.md)): weighting {uniform, energy, fitted} × rounding
{rtn, lsq, gptq} × groups {64 only, 64+128} × tiers {none, auto3}.

**Recovery sub-matrix** ([06 §4.8](plan/06-recovery.md)): stages A → B → C → D → MTP, each kept or dropped by its gate;
band order top-down vs bottom-up; lr {2e-6, 5e-6, 1e-5}.

**Speed sub-matrix** ([07 §5](plan/07-speculative.md)): MTP block {2, 3, 4, 5} × temperature {0, 0.6, 1.0} × drafter
{mtp, ngram, hybrid} × prefill step {2048, 4096, 8192}.

### 5.2 How to read the results

- **KL is per-model.** Compare KL only between builds of the *same* model against the *same* `ref/` (the unpruned
  full-precision reference; GLM's is the FP8 model dequantized). Across families use bits-per-byte on held-out personal
  text and task suites ([02 §5.1](plan/02-models.md)).
- **Rough KL scale** (top-64 + tail, nats/token): ≈ 0.05 small; ≈ 0.10 is where good 3-bit MoE builds land
  (Litwein REAP-320 + DWQ 0.10 [self-reported]); past ≈ 0.15–0.2 task retention falls steeply ([02 §4.3](plan/02-models.md)).
  Look at **per-domain** KL; `agentic` is usually worst, `code` best.
- **Rank with KL, guard with tasks.** KL over thousands of positions ranks; task suites detect only large gaps (n ≈ 700
  paired items for a 3-point gap at 8 % discordance) and act as regression guards ([04 §5](plan/04-pruning.md)).
- **Pruning damage hides in the tail:** rare-entity copy, termination/loops, turns-to-solve, multilingual
  (`general`) KL. A build that wins on mean KL and loses a guard is rejected.
- **Speed numbers** need power mode, thermals, swap and "contended" tags; median of ≥ 3 runs; decisions only on an
  idle GPU ([07 §5](plan/07-speculative.md), [08 §5](plan/08-system.md)).
- **Simulated vs built.** Sweep KL (qdq) must match built `evals.kl` within 10 % before the sweep is trusted as the
  planning tool ([05 §5](plan/05-quantization.md)).

### 5.3 Decision rules (the ones that change the roadmap)

| ID | Decision | Rule | Source |
|---|---|---|---|
| D-promote | replace the Work build | T1 gate PASS vs current: KL CI excludes 0 in ≥ 3/4 domains, no domain worse; no hard fail (peak ≤ cap − 2 GiB, BFCL parse < 1 %, RULER@128K ≥ 0.9× baseline, termination failures ≤ 0.5 %); decode ≥ 0.95× | [08 §4.8](plan/08-system.md) |
| D1 | Qwen K | among K ∈ {512, 448, 384}: lowest agentic+code-weighted KL; must not lose to K = 320 on pooled code+tools; rare-name ≥ 9/10 | [02 §5.2](plan/02-models.md) |
| D-joint | experts vs bits | at the optimum one more GB buys equal KL either way; if the expert side buys more, prune less | [04 §4.5](plan/04-pruning.md) |
| D-knee | fallback prune limit | largest ratio before the per-domain KL slope exceeds 2× its slope at 25 % pruning; no domain > 1.5× best | [04 §4.5](plan/04-pruning.md) |
| D2 | keep GLM at all | best GLM build solves ≥ 15 % of the primary's failures; else drop GLM | [02 §5.2](plan/02-models.md) |
| D3 | GLM in Work | only if GLM K ≈ 144 comes within 2 points of Q1 on pooled code+tools **and** wins τ²/hallucination | [02 §5.2](plan/02-models.md) |
| D-quant | keep LSQ / GPTQ / DWQ | each must give its own gain (LSQ ≥ 10 % KL vs RTN; DWQ ≥ 5 %, no domain worse) | [05 §5](plan/05-quantization.md) |
| D-recover | continue recovery | stage B < 3 % → stop the track; spend nights on K/bits | [06 §5](plan/06-recovery.md) |
| D-stream | stream vs mask a tail | stream only if session hit rate ≥ 0.97 at the profile's R (< 25 % slowdown) | [01 §5](plan/01-memory.md) |
| D-spec | n-gram drafter, EAGLE-3 | n-gram: blended +15 %, no segment −3 %; MTP fine-tune: ≥ +2 α points else skip EAGLE-3; EAGLE-3 ≥ +10 % tok/s | [07 §5](plan/07-speculative.md) |
| D-ctx | spend RAM on bits or context | beyond ≈ 4 bpw Qwen saturates; spend Focus/All-in bytes on 256K context and APC first | [01 §5](plan/01-memory.md) |

---

## 6. Code work backlog, by module

Priority: **P0** = blocks correct measurement or safety; **P1** = needed for the Phase 4 promotion or daily use;
**P2** = improves quality/speed measurably; **P3** = stretch (Phase 9). Effort is [est]. Every item is tested on tiny
fixtures first. Reference convention: `05 §4.2.7` means numbered item 7 of chapter 05's §4.2; `04 §6.2` means item 2
of chapter 04's §6 (failure-mode list).

### build/

| ID | Change | Pri | Effort | Chapter |
|---|---|---|---|---|
| B-1 | g128 in `alloc.DEFAULT_GROUPS`; experts restricted to {64, 128} unless measured sens puts g32 on the hull; `fill_sensitivities` proxy covers g128 | **P0** | 0.5 d | 05 §4.2.1 |
| B-2 | `--budget-scope {disk,resident}` (PLE and non-resident tiers excluded from `--budget-gb`); replaces 02's proposed `--budget-resident-gb` | **P0** | 0.5 d | 01 §4.1, 02 §4.4, 05 §4.2.6 |
| B-3 | per-family floor/pin tables in `families.py` (routers & Qwen `in_proj_a/b` bf16; ≥ 8 for hc read gates, indexers, embed, lm_head, MTP connectors, PLE projections, GLM f/g/b; ≥ 6 for attention, linear attn, shared experts, dense MLP; ≥ 2 for `switch_mlp`) | **P0** | 0.5 d | 05 §2.4, quant Rec 2 |
| B-4 | `--weighting {uniform,energy,fitted,fisher}`; `aggregate_expert_sens` weights by `freq·expert_imp` | P1 | 1 d | 05 §2.3, §4.2.2 |
| B-5 | reconcile R-G1 / R-Q1 against `--dry-run`; write the table into 05 §4.6 | **P0** | 0.5 d | R1 |
| B-6 | fix `docs/contracts/build.md`: `--sens-power` default is 1 | **P0** | min | R18 |
| B-7 | `build/round.py`: `--rounding {rtn,lsq,gptq}`, `--awq-down`; `achieved_nmse` in summary; assert sens-rounding = build rounding | P1 (lsq) / P2 (gptq, awq) | 2–3 d | 05 §2.6, §4.2.4 |
| B-8 | `--emit-mask PATH` on `--dry-run` (masks for the sweep) | P1 | 0.5 d | 04 §4.4 |
| B-9 | new prune scores: REAP-sum, shrinkage, HOPE (cond/unc, swap solver), max-rank + per-domain cores; super-expert force-keep guard | P1 (guard, sum) / P2 (HOPE, cores) | 2 d | 04 §2.3–2.4, reap Rec 3–4 |
| B-10 | calibrate `--prune-lambda` from sweep KL (linear fit vs `total_prune_loss`) | P2 | 0.5 d | 04 §4.5.4 |
| B-11 | `--ple-mode {affine32,nvfp4} --ple-prescale 4096` (fold 2⁻¹² into `ple.value_proj`) | P2 | 1 d | 05 §4.2.7, quant Rec 6 |
| B-12 | `--tiers`, `--tier-score weighted`, `--tier-order`, monotone cut-point units; tiered shim (extends `shim.py`) | P3 | 3–4 d | 01 §4.4–4.5, 05 §2.5 |
| B-13 | post-build asserts: routers bf16/fp32, `in_proj_a/b` bf16, router rows + bias sliced together, 1K-token KL smoke vs `ref/` | P1 | 0.5 d | 04 §6.2, 05 §6.7 |
| B-14 | APFS clone (`cp -c`) of byte-identical PLE shards across Qwen builds; free-space preflight | P2 | 0.5 d | 08 §4.9 |
| B-15 | MTP: calibrated MTP REAP (from K-8) instead of the data-free fallback; connectors ≥ 8-bit | P1 | 0.5 d | 04 §2.9, quant Rec 8 |

### calib/

| ID | Change | Pri | Effort | Chapter |
|---|---|---|---|---|
| K-1 | `GROUPS = (32, 64, 128)` in `sens.py` | **P0** | min | 05 §4.1 |
| K-2 | `FORMAT_VERSION = 2` + contract update | **P0** | 0.5 d | 04 §4.1, 05 §4.1 |
| K-3 | `expert_reap_sq` (= `expert_imp`), per domain | P1 | min | 04 §4.1, 05 §4.1 |
| K-4 | `coact_sum/cnt.{d}` (HOPE F) | P2 | 0.5 d | 04 §4.1 |
| K-5 | `expert_out_norm_max`, `expert_act_max` | P1 | min | 04 §2.3 |
| K-6 | `route_trace.{d}` (first ~200K tokens/domain, uint16) | P2 | 0.5 d | 04 §4.1, 02 §4.4 step 6 |
| K-7 | `prune_curve`, `block_influence`, `resid_sq` | P1 | 0.5 d | 04 §4.1, 05 §2.3 |
| K-8 | MTP stage (REAP/freq/sens for MTP experts) | P1 | 1 d | 04 §4.1 |
| K-9 | `--sens-rounding rtn,lsq`, `--sens-rot` | P1 / P2 | 1 d | 05 §4.1 |
| K-10 | position-stratified reservoirs `--pos-strata`; `--expert-tokens 512` for the final pass | P1 | 0.5 d | 03 §2.5–2.6 |
| K-11 | `target_logprob` for calib samples (NLL filter for proxy self-gen); `--ref-splits eval,dwq` / `--ref-split calib` (06) | P1 | 0.5 d | 03 §4.0 C7, 06 §2.6 |
| K-12 | `--save-samples`, `--teacher-split` (store ĥ_N for exact full-vocab DWQ targets) | P2 | 1 d | 05 §2.8, §4.1 |
| K-13 | `calib/sweep.py`: multi-mask streaming evaluator (masks, top-k override, block skip, qdq plans) | **P1** | 3–4 d | 04 §4.4, 05 §4.3 |
| K-14 | `--keep-mask` sequential re-calibration on the pruned model | P1 | 1 d | 04 §4.6 |
| K-15 | `--layers-per-stage 2`, batch packing of short samples, in-RAM shards (SSD wear) | P2 | 1–2 d | 04 §4.3, 06 §2.6 |
| K-16 | free-space preflight; delete `work/` of private runs (`scripts/clean_private.sh`) | P1 | 0.5 d | 08 §4.9, 03 §4.4 |
| K-17 | `calib/fisher.py` streaming backward Fisher (`use_kernel=False` for gated delta) | P3 | 2 d | 05 §2.3 level 3 |

### calib_data/

| ID | Change | Pri | Effort | Chapter |
|---|---|---|---|---|
| D-1 | C1: content-hash `group` split key; three splits `calib`/`dwq`/`eval` | **P0** | 0.5 d | 03 §2.7, §4.2 |
| D-2 | C2: `decontam.py` (13-gram + exact match vs all `evals/` benchmarks) | **P0** | 0.5 d | 03 §2.7 |
| D-3 | C3: `personal.py` + `scrub.py` + `LocalJsonl`; key-only probe P0 (user-run); one module for calib, replay, drafter, recover (R15) | P1 | 1.5 d | 03 §4.4, 08 §4.8 |
| D-4 | C4: `selfgen.py` (full / one-step; filters; loopback-only for personal) | P1 | 0.5 d | 03 §4.5 |
| D-5 | C5: `JsonlRange`; recipes `swe_smith`, `nemotron_swe`, `swe_rebench_oh`, `nemotron_agentic`, `terminal`, `longalign`, `fineweb_edu`, `fineweb2_multi`, `mot_prompts`; `general` domain; longctx 32–128K; agentic `MAX_LEN` 32768 | P1 | 1 d | 03 §4.3 |
| D-6 | C6: schema v2 fields (`split`, `group`, `origin`, `teacher`, `gen_spans`, `n`) + contract | P1 | 0.5 d | 03 §4.1 |
| D-7 | `calib_data.report` diagnostics (coverage, on-policy-ness, routing JSD vs live, leak and scrub reports) | P2 | 0.5 d | 03 §5 |
| D-8 | `claude_tools.json` fallback + `serve --capture-schema` (tools array + system-prompt template only) | P2 | 0.5 d | 03 §4.4 P3 |

### evals/

| ID | Change | Pri | Effort | Chapter |
|---|---|---|---|---|
| E-1 | resident-aware `weights_gb` (share `serve/memory.resident_weights_bytes`); until then pass `--max-weights-gb 100` | **P0** | min | this plan §4 |
| E-2 | `gate.py` (T0/T1/T2, PASS/FAIL/REVIEW), KL paired-bootstrap CI vs baseline, `flips` (McNemar) | **P0** | 1.5 d | 08 §4.8 |
| E-3 | rare-entity / identifier-copy probe (≥ 200 items, 10 seeds), termination/loop rate | **P0** | 1 d | 04 §5, reap Rec 8 |
| E-4 | `speed`: footprint logging, profile tag, power/thermal, `--load-only`, verify-cost curve T(k) | P1 | 1 d | 01 §5 T1, 07 §4.0, 08 §4.9 |
| E-5 | `specbench.py` (agent-trace replay, per-segment acceptance) | P1 | 1 d | 07 §4.0 |
| E-6 | `replay.py` (personal, time-split; tool-name/arg validity, Edit old_string-exists, KL vs reference local model) | P1 | 1 d | 08 §4.8 |
| E-7 | agentic turns-to-solve mode in `code`; personal bits-per-byte (cross-family) | P2 | 1 d | 04 §5, 02 §5.1 |
| E-8 | `ab.py`, `lcb.py`, `swe.py` (mini-swe-agent, arm64 images, Docker — Calibrate mode only), BFCL multi-turn | P2 | 2–3 d | 08 §4.8 |
| E-9 | `rag.py` (recall@k, MRR per configuration) | P2 | 0.5 d | 08 §4.5 |

### serve/ (and the production wrapper)

| ID | Change | Pri | Effort | Chapter |
|---|---|---|---|---|
| S-1 | `memory.resident_weights_bytes` (exclude `ple-store.json` tensors); use in `check_fits` and APC sizing | **P0** | 0.5 d | 01 §4.1, 08 §4.2a |
| S-2 | `budget.py` (KV/state/transient formulas) + per-request admission on `active + cache + est_bytes` | **P0** | 1 d | 01 §4.1, 08 §4.2b |
| S-3 | wiring default 0 (opt-in with residency refresh) | **P0** | min | 08 §4.2d, mac Rec 2 |
| S-4 | security: auth before model load, Host-header check, remove CORS `*`, key from Keychain in launchd mode | **P0** | 0.5 d | 08 §4.10 |
| S-5 | `glm_fix.py` (`_MAX_PROJECTED_PREFILL_TOKENS = 0`, R13) + tiny-GLM bit-exact test | **P0 for GLM** | 0.5 d | 08 §4.2c, decode Rec 1 |
| S-6 | SSE `ping` during long prefill (test: ≥ 1 ping / 10 s); same for `run_server.py` while it is production | **P0** | 0.5 d | 08 §4.3, decode Rec 9 |
| S-7 | allow `embedding` / `reranker` model kinds in-process; `--embedding-model`, `--reranker-model` | P1 | 0.5 d | 08 §4.2e |
| S-8 | `router.py` (aliases, effort, two-phase thinking budget, context errors); `/v1/models` shows profile | P1 | 1 d | 08 §4.4 |
| S-9 | `obs.py` + `serve.report` (JSONL, no prompt text; α, APC hit, queue wait, pressure) | P1 | 1 d | 08 §4.7 |
| S-10 | `watchdog.py` (01's L1/L2/L3 ladder, hysteresis, `logs/watchdog.jsonl`) | P1 | 1 d | 01 §4.6, 08 §4.9 |
| S-11 | profiles: `ServeConfig.profile`, `expert_cache_gb`, `POST /serve/profile`, `scripts/profile.sh` | P1 | 1 d | 01 §4.2, 08 §4.1 |
| S-12 | `tools_guard.py` (validate, coerce, one forced resample) | P1 | 1 d | 08 §4.6 |
| S-13 | `ngram.py` + `hybrid_draft.py` (+ `spec_loop.py` fallback) | P2 | 2–4 d | 07 §4.3 |
| S-14 | `grammar.py` lazy Lark tool grammar (MTP off inside the segment) | P2 | 1–2 d | 08 §4.6 |
| S-15 | launchd socket activation + exit-on-idle; control daemon launches `serve` on :8080 (R7) | P1 | 1 d | 08 §4.9 |
| S-16 | `tiered.py` (`TieredSwitchGLU`, masks, slabs, streaming tail, adaptive) | P3 | 1–2 wk | 01 §4.5 |
| S-17 | APC: checkpoint at request end; prefix-divergence probe; session pre-restore | P2 | 1 d | 07 §4.1, 08 §7.5 |
| S-18 | `--capture-dir` on-policy capture (opt-in, privacy pipeline) | P3 | 1 d | 03 §7.1 |

### recover/ (new, Phase 6) — [06 §4.1](plan/06-recovery.md)

`teacher.py` (P1), `loss.py` (P1), `local.py` (P1), `dwq.py` (P1), `router.py` (P2), `onpolicy.py` (P2),
`train_mode.py` (P1: dense QSA plan at T ≤ 2048, checkpoint every layer class), model diff tool (P1), tests on tiny
fixtures (P1). ≈ 600–900 lines, 3–4 days. Memory self-bounding (refuse > 100 GiB; stop at plan + 10 %).

### drafter/ (new, Phase 5/6) — [07 §4.5–4.6](plan/07-speculative.md)

`collect.py` (trunk prefill over personal contexts → (h_t, ids, top-k) shards), `train_mtp.py` (KL + TTT unroll),
`eval.py` (α per segment type). P2. EAGLE-3 training P3.

### rag/ (new, Phase 8) — [08 §4.5](plan/08-system.md)

`ingest`, `chunk`, `context`, `index` (SQLite FTS5 + f16 vectors), `embed`, `search` (dense ∪ BM25 → RRF → rerank),
`answer` (Self-Route), `__main__`. P2. New dependency: `pypdf` (list it; do not edit `pyproject.toml`).

### scripts/ and ops

`scripts/sim_expert_cache.py` (P2, [02 §4.4](plan/02-models.md)), `scripts/disk_report.py` (P2, [08 §4.9](plan/08-system.md)),
`scripts/clean_private.sh` (P1, [03 §4.4](plan/03-calibration-data.md)), `scripts/profile.sh` (P1), `models/current` /
`models/previous` symlinks (P1), control daemon: refuse `--kv-bits` (P1, [05 §6.13](plan/05-quantization.md)).

### kernels (Phase 9) — [07 §4.8](plan/07-speculative.md), [06 §7.1](plan/06-recovery.md)

K1 `mx.compile` decode sub-blocks (P2), K2 fused decode MoE kernel (P3), K3 preallocated QSA block-key buffer (P2),
K4/K5 prefill fusions (P3), chunked delta-rule op with custom VJP (P3, high value for recovery).

**New dependencies to list** (not added here): `pypdf` (RAG); optional `mcp`, `tree-sitter`, `datasketch`,
`detect-secrets`, `orjson`; `mini-swe-agent` + Docker Desktop in a separate venv. `llguidance` is already installed.

---

## 7. Risks and fallbacks

| Risk | Likelihood | Signal | Fallback | Chapter |
|---|---|---|---|---|
| Mmapped PLE pages evicted under your dev workload → token stalls | medium | p99 token latency under `memory_pressure -l warn` > 10 % worse | disk-scope plan R-Q2 (K ≈ 320–352, PLE counted) or keep REAP-288 | 02 §5.3, 05 R-Q2 |
| Unpruned Q1 does not beat REAP-288 on tasks | medium-low | T1 gate REVIEW/FAIL | try Q2/Q3 (K = 448/384); if all lose, keep K = 320 and spend 24 GB on helpers/context | 02 §5.2 D1 |
| Calibration code wrong at real scale (unreviewed modules) | medium | GLM shakedown stats nonsensical; KL smoke fails | the GLM screening pass exists to catch this before Qwen; tiny-fixture bit-exactness tests | Phase 1, 3 |
| SSD wear from hidden-state I/O (multi-TB per pass) | high (certain) | `smartctl`/disk writes | 1 M-token screening, one final pass per model, in-RAM shards, `--layers-per-stage 2` | 04 §4.3, 06 §2.6 |
| Disk fills (sources 690 GB + scratch + builds) | medium | `scripts/disk_report.py` | delete losing builds; APFS clones; prune HF sources after calibration | 08 §4.9 |
| Memory pressure hurts your work | medium | swap growth, compressor growth | watchdog ladder; never wire; heavy jobs only in Calibrate mode | 01 §4.6, §6 |
| IOGPU kernel panic | low | panic log | no wiring, one sequence, one MLX process | mac §3 |
| GLM loops / termination failures at ≤ 144 experts | high | loop-rate guard | prefer K ≥ 176 in All-in; drop GLM (D2) | 04 §3.2 |
| GLM projected cache / prefill transient OOM | certain without fix | +22–24 GiB at ≥ 32K | S-5 (R13), prefill step 512–1024 | 08 §4.2c |
| MTP head stale after compression/recovery | medium | α < 0.65 | MTP re-alignment (07 S5, 06 night 7) | 07 §4.2, 06 §2.8 |
| Recovery overfits / forgets | medium | held-out KL up, `general` KL up | per-band gate, ≥ 30 % general data, stop rules | 06 §6 |
| Personal-data leak (secrets in artifacts or weights) | low but severe | canary probe, scrub report jumps | scrubber, 0700/0600, loopback-only self-gen, never publish personal-calibrated or personal-DWQ builds | 03 §4.4, 08 §4.10 |
| Leaderboard / benchmark contamination misleads | medium | A/B disagrees with scores | relative-only use of contaminated sets; replay + A/B | 08 §6 |
| Upstream mlx/mlx-vlm changes break shims | medium | tests fail after upgrade | pin versions; never patch site-packages, only our code | PLAN rules |
| Stock `moe_offload` default budget (~92 GB) | certain if forgotten | process grows past cap | always pass `--expert-cache-gb` / `EXPERT_CACHE_GB` | 01 §2.4, 02 §6.7 |
| `--kv-bits` enabled by accident | low | server crash on first batch | daemon refuses the flag | 05 §2.10 |
| Speed estimates for GLM wrong (unmeasured `t0`) | high | `evals.speed` | GLM only for latency-tolerant work | 07 §2.2 |

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Active parameters / bytes** | weights a single decode token actually reads (trunk + top-k experts + lm_head). Sets decode speed. Qwen ≈ 6 B, GLM ≈ 17 B. |
| **Affine quantization** | `ŵ = s·q + β` per group of `g` weights, `b`-bit codes `q`, bf16 scale `s` and bias `β`; costs `b + 32/g` bits/weight. |
| **APC** | mlx-vlm's automatic prefix cache. For hybrid models it snapshots the whole cache (KV + recurrent state) at prefix boundaries (`strategy=checkpoint`); RAM tier + SSD tier. |
| **α (acceptance)** | fraction of drafted tokens the target accepts; `draft_n_accepted / draft_n`. |
| **AWQ** | activation-aware scaling: scale important input channels before quantizing and fold the inverse elsewhere. |
| **bpw** | bits per weight, including group metadata. |
| **Calibration** | running the full-precision model over representative data to measure the statistics that drive pruning and quantization. |
| **Chunked prefill** | processing a long prompt in fixed-size chunks (`prefill_step_size`) to bound activation memory. |
| **Decontamination** | removing benchmark items (and near-copies) from calibration/training data. |
| **DWQ** | distilled weight quantization: train only affine scales/biases of a quantized model against teacher logits. Same file size and speed. |
| **E, K, top-k** | experts per layer (Qwen 512, GLM 288); experts kept after pruning; experts each token routes to (Qwen 10, GLM 8). |
| **Fisher / amplification `a_t`** | how much a unit of output error in tensor `t` moves the final KL; the correct weight in the bit-allocation objective. |
| **Floors / pins** | minimum bits (or bf16) forced for sensitive tensors (routers, gates, attention, embeddings). |
| **g32 / g64 / g128** | quantization group size; smaller groups track outliers better but cost more metadata. |
| **GatedDeltaNet / KDA** | the linear-attention (recurrent) layers of Qwen / GLM; fixed-size state, cheap long context, no VJP in the fast Metal kernel. |
| **GiB vs GB** | 2³⁰ vs 10⁹ bytes. RAM caps are GiB; file sizes and `--budget-gb` are GB. |
| **GPTQ** | quantize column by column and push each rounding error onto not-yet-quantized columns using the input Hessian. |
| **HOPE** | pruning criterion that adds pairwise co-activation (experts that cooperate on the same tokens) to REAP. |
| **Hyper-connections (hc / mHC)** | 4 parallel residual streams mixed by small learned gates; their read gates are protected at ≥ 8-bit. |
| **imatrix** | per-input-channel mean squared activation `E[x_c²]`, used to weight rounding. |
| **KL (top-64 + tail)** | divergence of the compressed model's next-token distribution from the reference's, on the reference's top-64 tokens plus one bucket of residual mass; a lower bound on full KL. |
| **Knapsack (MCKP)** | choose one (bits, group) option per tensor to minimize total weighted error under a byte budget; solved exactly by DP. |
| **LSQ refit** | least-squares re-fit of each group's scale and bias after rounding; lowers error at identical bytes. |
| **Masking** | setting a router logit (GLM: selection bias) to −∞ so an expert is never chosen; mathematically identical to pruning it. |
| **McNemar test** | paired test for two builds' pass/fail on the same items; only discordant items count. |
| **MTP** | multi-token-prediction head trained with the model; drafts tokens for speculative decoding. Here written as `<model>/mtp/`. |
| **NAX** | the M5 GPU's neural accelerators; used automatically by MLX for multi-row (prefill) quantized matmuls, not for decode. |
| **NMSE / sens** | relative output error `‖XΔWᵀ‖²/‖XWᵀ‖²` of a tensor at a given (bits, group), measured by `calib/` on real activations. |
| **n-gram / suffix drafting** | proposing draft tokens by copying what followed the current suffix earlier in the context; free and very accurate for edits. |
| **On-policy distillation (GKD)** | the student generates, the teacher scores; fixes exposure bias. |
| **PLE / n-gram table** | Qwen's 51.2 B-parameter per-layer embedding table, served row-by-row from a memory-mapped file (`ple_storage`); clean page cache, not resident. |
| **Profile / mode** | Work 80 / Focus 92 / All-in 104 GiB process caps; Stream and Calibrate are modes (R10). |
| **Prune curve / knee** | KL as a function of K; the knee is where the slope steepens. |
| **REAP** | router-weighted expert activation pruning: keep experts with the highest mean of `gate × ‖expert output‖` over the tokens routed to them. REAP-sum multiplies by frequency. |
| **Resident bytes** | bytes in the process's physical footprint (anonymous Metal buffers); what the profile cap limits. |
| **`ref/`** | the full-precision model's top-64 logprobs on `eval.jsonl`, written by `calib/`; every KL is measured against it. |
| **Rounding (RTN)** | round-to-nearest with min/max scales; the baseline that LSQ/GPTQ improve. |
| **Router KD** | distilling only the router rows end to end after pruning. |
| **RULER** | synthetic long-context suite (needles, multi-key, variable tracking, QA) used by `evals.longctx`. |
| **Self-generation** | calibration text whose assistant side is written by the model being compressed (or a close proxy). |
| **Speculative decoding** | draft γ tokens cheaply, verify them in one target pass; lossless in distribution. |
| **Super experts** | rare experts with huge output norms; must be force-kept. |
| **`t0`** | the fixed per-step decode overhead (launches, recurrences, syncs) that does not scale with bytes; ≈ 60 % of a Qwen step. |
| **Teacher / student** | uncompressed reference model / compressed model being recovered. |
| **Tiers (hot/warm/cold)** | groups of experts by routed importance, stored at different bits and residency. |
| **Verify cost T(k)** | time to verify k positions; on fine-grained MoE grows with the union of experts the k tokens route to. |
| **Watchdog ladder** | shed APC RAM → shrink caches → drop an expert tier → unload, with hysteresis, before macOS swaps. |
