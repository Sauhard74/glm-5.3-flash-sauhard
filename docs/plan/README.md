# Master plan chapters — index

Start at **[`../MASTER_PLAN.md`](../MASTER_PLAN.md)**. It is the top level: the target end state, the binding
resolutions of conflicts between chapters (its §2), the learning path, the phased roadmap with commands and exit
criteria, the experiment matrix, the code backlog, risks and a glossary. It links into the chapters below instead of
repeating them. The chapters hold the mechanics, the math, the evidence and the detailed per-topic plans.

Status (2026-10-20): all eight chapters are first drafts written in parallel; they were cross-read while writing
`MASTER_PLAN.md`. **Where a chapter disagrees with `MASTER_PLAN.md` §2, the master plan wins.** The most important
overrides: GLM K = 144 does not fit 68 GB until g128 lands (05 §4.6 R-G1 vs the planner, resolution R1); budgets count
*resident* bytes (R2); the measured MTP block is 2, not 4 (R5); plan with the measured 30 / 55 tok/s (R6); the
canonical port is 8080, not 8642 (R7).

## Reading order

| # | Chapter | Read it for | Phase(s) it drives |
|---|---|---|---|
| 1 | [01 — Unified-memory engineering](01-memory.md) | how 128 GB is split; resident vs disk bytes; profiles Work/Focus/All-in; SSD streaming math; hot/cold tiers; zero-copy file-backed weights; the watchdog | 1, 8, 9 |
| 2 | [02 — Model selection](02-models.md) | why Qwen-unpruned is the primary and GLM the secondary; base quality vs compression damage; bigger-than-RAM options; helpers; multi-model routing | 4, 7 |
| 3 | [04 — Expert pruning](04-pruning.md) | REAP / REAP-sum / HOPE; merging; router handling; MTP after pruning; the multi-mask streaming sweep; experts-vs-bits joint rule | 1, 3, 4, 7 |
| 4 | [05 — Quantization](05-quantization.md) | affine formats and group size; the knapsack and its correct units; what to protect; LSQ/GPTQ/AWQ; rotations; DWQ mechanics; target recipes | 1, 4 |
| 5 | [03 — Calibration data](03-calibration-data.md) | domains and budgets; personal transcripts (privacy pipeline); self-generation; long sequences; contamination | 2, 3 |
| 6 | [07 — Speed](07-speculative.md) | decode roofline and the overhead term; MTP; n-gram/suffix drafting; prefix caching; prefill; kernels | 5, 9 |
| 7 | [08 — The complete system](08-system.md) | server, clients, router, RAG, tool reliability, observability, the eval gate, operations, security | 1, 8 |
| 8 | [06 — Recovery](06-recovery.md) | DWQ on scales/biases, router KD, on-policy distillation, training memory math, the 7-night schedule | 6 |

Every chapter has the same skeleton: **§0 TL;DR · §1 intuition · §2 mechanics and math · §3 evidence · §4 plan for
this machine · §5 decision criteria · §6 failure modes · §7 stretch ideas · Sources**. Read §1–§2 to learn, §4–§6
before you run that chapter's phase.

## What each chapter decides (one line each)

| Chapter | Key decision | Key sections |
|---|---|---|
| 01 | Budget resident bytes; an unpruned 512-expert Qwen at ≈ 3.6 bpw fits the 80 GiB Work profile; three profiles with live switching; masking = pruning | §2.1, §2.4c, §2.5, §4.2, §4.3, §4.6, §5 |
| 02 | Primary = Qwen3.8-Flash-Next lightly or not pruned; GLM only in All-in / Stream; at 68 GB a 50 %-pruned GLM has ≈ 3× the divergence of Qwen | §0, §2.3–2.5, §4.2–4.4, §5.2 |
| 03 | The calibration set is the spec; add `personal` and `general` domains, agentic sources, longctx 32–128K, a `dwq` split, content-hash splits and decontamination; self-generate the assistant side | §2.1, §2.4–2.7, §4.2–4.6, §5 |
| 04 | REAP (+ REAP-sum, HOPE) chosen by held-out KL; no top-k shrinking, no attention pruning; prune → re-calibrate → quantize → DWQ; `calib/sweep.py` | §2.3–2.4, §2.11, §4.0, §4.4–4.6, §5 |
| 05 | Affine everywhere; g128 not g32 as the expert knob; knapsack weights `a_t·E_t`; floors table; LSQ → GPTQ → DWQ; KV quant off; recipes R-G1 / R-Q1 / R-Q2 | §2.2–2.6, §4.1–4.2, §4.6, §5 |
| 06 | Train only scales/biases and router rows; band-wise with per-band gates; local teacher scores, cloud only for teacher-written text; one model, ~7 nights | §2.1–2.2, §2.5, §4.1–4.8, §5 |
| 07 | Prefix caching first; MTP; n-gram drafter for agent copies; short drafts on fine-grained MoE; kernels last | §2.1–2.3, §2.6, §2.8, §4.0–4.3, §5 |
| 08 | One memory-capped process; resident-aware admission; GLM cache fix; router by alias; native tools + guard; statistical gate T0/T1/T2; launchd + watchdog; security | §2.1, §2.6–2.7, §4.1–4.11, §5 |

## Supporting material in this directory

| Path | What it is |
|---|---|
| [`moe_density_bench.py`](moe_density_bench.py) | tiny bf16 `SwitchGLU` prefill bench: throughput vs tokens-per-expert (04 §4.0) |
| [`experiments/fmt_rot.py`](experiments/fmt_rot.py), [`experiments/fmt_rot.json`](experiments/fmt_rot.json) | format / Hadamard / LSQ-refit NMSE on real Qwen experts read from the MTP drafter (05 §3.1, §8) |
| [`experiments/recovery/`](experiments/recovery/) | `gd_mem.py`, `gqmm_grad.py`, `gqmm_scale.py`: training-memory measurements on random tensors (06 §3.3) |

## Related documents

- [`../../PLAN.md`](../../PLAN.md) — goal, budget table, module pipeline, rules for working here, current status.
- [`../research/`](../research/) — verified research the chapters build on: `mac.md`, `decode.md`, `quant.md`, `reap.md`
  (plus `quant_bench.py`, `quant_verify_bench.py`).
- [`../contracts/`](../contracts/) — module contracts (CLIs, file formats): `calib_data.md`, `calib.md`, `build.md`,
  `evals.md`, `serve.md`. **The contracts and the code are the source of truth for flags**; chapter commands that use
  flags not in the code are marked ★ (proposed) and are listed in `MASTER_PLAN.md` §6.

## Tags and units used by all chapters

**[measured]** run on this machine · **[computed]** arithmetic from real configs or safetensors headers · **[src]**
read in installed package source (mlx 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.9) or this repo · **[self-reported]** a
model-card / README / paper number not reproduced here · **[paper]** peer-reviewed or arXiv · **[est] / [model] /
[hypothesis] / [assumption]** an estimate or proposal that a named measurement must replace.

GB = 10⁹ bytes (files, `--budget-gb`); GiB = 2³⁰ bytes (RAM caps: `CAP_GB`, `--memory-limit-gb`, MLX counters).
Section references like `04 §4.5` mean chapter 04, section 4.5; `05 §4.2.7` means numbered item 7 inside §4.2.
