# 07 — Speed: speculative decoding, caching and kernels

Earlier chapters choose *which* model fits (REAP + mixed precision, ≤ 80 GiB process). This chapter makes the chosen
model *fast* on this MacBook Pro M5 Max (40-core GPU, 128 GB, 614 GB/s), for the three workloads in `PLAN.md`:
agentic coding (Claude Code / OpenCode), reasoning/math, and long-document RAG.

It builds on `docs/research/decode.md` (MTP, APC, chunked prefill), `mac.md` (memory knobs, NAX, ceilings),
`quant.md` (decode µs/layer microbenchmarks, NAX kernels) and `reap.md` (MTP after pruning). They are cited as
*decode §1*, *quant §4* and so on, and their findings are not re-argued here. Chapter 08 (`08-system.md`) owns the
server, profiles and observability; this chapter owns everything that changes tokens per second.

Tags:

| Tag | Meaning |
|---|---|
| **[measured]** | Run on this machine in this session: safetensors *headers* only (no weights loaded), tokenizer-only simulations, CPU only. The GPU was busy benchmarking, so no GPU timings were taken. |
| **[src]** | Read in the installed packages (`.venv/lib/python3.12/site-packages/{mlx,mlx_vlm,mlx_lm}`). |
| **[research]** | A verified finding in `docs/research/*.md`. |
| **[self-reported]** | A number from a paper, model card, README or issue, not reproduced here. |
| **[model]** | Output of the cost model in §2.3 / Appendix A. It is a prediction with stated assumptions. A named measurement in §5 replaces each one. |
| **[computed]** | Arithmetic shown inline. |

Units: bandwidth and bytes-per-token use **GB = 10⁹** (614 GB/s is a decimal figure). Memory budgets use **GiB**,
as `serve/` does.

---

## Contents

0. [TL;DR](#0-tldr)
1. [Why it matters (intuition)](#1-why-it-matters-intuition)
2. [How it works (mechanics + key math)](#2-how-it-works-mechanics--key-math)
   - 2.1 The decode roofline: bytes touched per token, derived from real tensors
   - 2.2 Where decode time actually goes: the fixed-overhead term
   - 2.3 Speculative decoding: acceptance, verification cost, and the MoE twist
   - 2.4 Native MTP in mlx-vlm
   - 2.5 Learned drafters: EAGLE-3, DFlash, and trees (DDTree)
   - 2.6 Model-free drafting: prompt lookup / suffix decoding for agentic coding
   - 2.7 Prefill: neural accelerators, chunking, rows-per-expert
   - 2.8 Prefix caching for agent loops, SSD persistence, compaction
   - 2.9 Kernels: what is already custom, what is left
3. [Evidence](#3-evidence)
4. [Plan for this machine](#4-plan-for-this-machine)
5. [Decision criteria & what to measure](#5-decision-criteria--what-to-measure)
6. [Failure modes](#6-failure-modes)
7. [Stretch ideas (push limits)](#7-stretch-ideas-push-limits)
- [Appendix A — cost model](#appendix-a--cost-model)
- [Appendix B — n-gram acceptance simulation](#appendix-b--n-gram-acceptance-simulation)
- [Sources](#sources)

---

## 0. TL;DR

| # | Lever | Where the time goes that it removes | Expected gain on this machine | Cost to build |
|---|---|---|---|---|
| 1 | **APC prefix checkpoints (RAM + SSD)** | Re-prefilling the system prompt, tool schemas and history every agent turn | TTFT of turn *n* drops from *history ÷ prefill rate* to *new tokens ÷ prefill rate*: e.g. a 60K-token Qwen session goes from ≈ 35–40 s to ≈ 2 s per turn; GLM from ≈ 3 min to ≈ 10 s [computed, §2.8] | Already in mlx-vlm and wired by `serve/`. Needs prefix-stability checks + measurement |
| 2 | **Native MTP** (Qwen block 4, GLM block 3) | The fixed per-step overhead (≈ 60 % of a Qwen decode step), amortised over ~3 tokens | Qwen **1.6–1.9×** [model], 1.7× [self-reported, MTPLX]. GLM **1.3–1.5×** [model], +0–35 % [self-reported, oMLX on M3 Ultra] | Already in mlx-vlm. Keep the MTP tensors in `build/` |
| 3 | **Suffix / prompt-lookup drafter** (new, `serve/ngram.py`) | Re-emitting text that is already in context: `old_string`, unchanged lines of a rewritten file, paths, identifiers | Tool-call arguments **≈ 2–3×**, whole-file rewrites **≈ 4–6×** [model, from a measured tokenizer-level simulation, Appendix B]. ≈ 1× on fresh prose and reasoning | ≈ 300–500 lines; reuses mlx-vlm's verify and rollback |
| 4 | **Hybrid drafter**: n-gram when a long match exists, MTP otherwise | Both of the above, per round | Takes the max of 2 and 3 per segment | Small once 2 and 3 exist |
| 5 | **Personal-data MTP fine-tune** (then EAGLE-3 if it pays) | Draft misses on *your* code style and tools | +2–8 points of acceptance → +5–15 % decode [self-reported +2–3 % for a self-distilled sidecar; the rest is an upper-range assumption] | Medium; MLX training loop of our own (mlx-vlm ships none) |
| 6 | Prefill chunk size + expert count | GEMM tile utilisation in the MoE prefill (rows per expert) | 5–25 % prefill [self-reported, mlx-serve chunk sweeps] | A flag + `evals.speed` sweep |
| 7 | Kernel work (fused gate/up, HC fusion, fewer dispatches) | Dispatch and launch latency in a 48-layer decode step | 5–15 % decode, 10–20 % prefill (component wins of 1.4–3.4× [self-reported]) | High; only after 1–5 |

What this chapter changes in earlier documents:

- **The decode ceiling for the deployed baseline is 153 tok/s, not ≈ 235.** Real safetensors headers of
  `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` give **3.79 GB of weights per token**, plus 0.23 GB of recurrent-state
  traffic [measured, §2.1]. mac §7 assumed 3.5 bpw and only the experts.
- **Qwen decode is overhead-bound, not bandwidth-bound.** Measured plain decode on this model class is 44–48 tok/s
  (21–23 ms) on an M5 Max [self-reported, two runtimes]. Bandwidth explains ≈ 9 ms of that, so ≈ 12–13 ms per step is
  fixed overhead (§2.2). That is why speculation beats bit-shaving for Qwen. GLM is closer to bandwidth-bound.
- **On fine-grained MoE, verifying k tokens is not free.** Each drafted token routes to different experts. Verifying
  16 tokens on Qwen-288 reads ≈ 9.6× the routed bytes of one token (≈ 4.3× the total bytes), so optimal draft lengths are short (3–5) except
  when acceptance is near-certain, as with n-gram copies of context (§2.3).

---

## 1. Why it matters (intuition)

An LLM request has two phases with opposite bottlenecks.

**Prefill** pushes the whole prompt through the network at once. Every weight read from memory is reused across
thousands of tokens, so the GPU is limited by arithmetic. The M5's neural accelerators exist for this (quant §4).

**Decode** produces one token at a time. To make one token you must read every weight that token touches: the trunk
plus 8–10 experts per layer. You then do very little arithmetic with them: 2 FLOPs per weight, for one row. The GPU
spends its time waiting for memory, or waiting for the next of ~1,000 small kernels to launch.

In an agent loop both phases matter, but in different places:

```
turn n:  [ prefill: system + tools + history + new tool result ]  [ decode: thinking + tool call ]
          ~20–35K tokens on turn 1, then +1–10K per turn            ~100–2,000 tokens
```

- Without prefix caching, prefill grows with the whole history every turn. With Qwen at ~1,700 tok/s, a 60K-token
  session costs ≈ 35 s *before the first token of every turn* [computed]. With GLM at ~330 tok/s it costs ≈ 3 min.
  This is the single largest latency in local agentic coding. It is fixed by caching, not by kernels.
- Decode dominates reasoning turns (thousands of thinking tokens) and long file writes. Speed here comes from
  doing more useful tokens per weight read: speculative decoding.
- Agents re-emit context constantly. An `Edit` call copies `old_string` verbatim from a `Read` result. A file
  rewrite repeats 95 % of a file. A model-free drafter that copies from context is almost free and is right most of
  the time there.

The rest of the chapter makes these intuitions quantitative, so that each optimisation can be ranked by expected
gain per hour of work, and so that you know which measurement proves or kills it.

---

## 2. How it works (mechanics + key math)

### 2.1 The decode roofline: bytes touched per token, derived from real tensors

For batch-1 decode, a step reads every *active* weight once:

```
t_bw(1 token) = bytes_touched / BW_eff        tok/s_ceiling = 614 GB/s / bytes_touched
bytes_touched = Σ_trunk params × bpw/8  +  K_active × bytes_per_expert × L_moe  +  lm_head  +  state/KV traffic
```

Embeddings are a lookup (one row), so they do not count. The `lm_head` is a full matrix-vector product over the
vocabulary every token, so it counts in full. **REAP pruning does not change bytes per token**: top-k is unchanged,
so the same number of experts is read. Pruning changes resident memory and, as §2.3 shows, the verification cost.

**Qwen3.8-Flash-Next, deployed baseline `qwen-flash-next-reap288` (4-bit g64) [measured from safetensors headers].**
Per-layer bytes, read from the actual tensors (weights + scales + biases):

| Component (per layer) | Bytes | × layers | Per token |
|---|---|---|---|
| GatedDeltaNet `in_proj_qkv` 14.75 MB + `in_proj_z` 8.85 + `out_proj` 8.85 + a/b/conv ≈ 0.2 | 32.6 MB | 36 linear-attn layers | 1,175 MB |
| QSA `q_proj` 17.69 + `o_proj` 8.85 + k/v 1.47 + indexer 0.92 | 28.9 MB | 12 full-attn layers | 347 MB |
| Shared expert 2.76 + router (bf16) 1.47 + hyper-connection mixers 7.36 | 11.6 MB | 48 | 557 MB |
| Routed experts: 796.26 MB per layer ÷ 288 experts × top-10 | 27.65 MB | 48 | **1,327 MB** |
| PLE key/value projections (layer 1; the n-gram table itself is a lookup, mmapped) | 18.4 MB | 1 | 18 MB |
| `lm_head` 248,320 × 2,560 at 4-bit g64 | 357.6 MB | 1 | 358 MB |
| **Weights per token** | | | **3.79 GB** |
| GatedDeltaNet fp32 state 48×128×128 = 3.15 MB, read + written | 6.3 MB | 36 | 227 MB |
| QSA at 128K: block keys 64 B/token × 131,072 = 8.4 MB + 2,048 selected tokens × 2,048 B = 4.2 MB [src: `Qwen4ExpIndexer`, decode §3] | 12.6 MB | 12 | 151 MB |

Ceiling at short context: 614 / (3.79 + 0.23) = **153 tok/s**. At 128K: 614 / 4.17 = **147 tok/s** [computed].

Two things stand out:

1. **The routed experts are only 35 % of the bytes.** The trunk (DeltaNet projections, hyper-connections, shared
   experts, `lm_head`) is 2.46 GB. quant §4 found the same from the other direction: Qwen decode µs/layer barely moves
   with expert bit width.
2. **Quality floors cost decode bandwidth.** quant §Recommendations puts attention/DeltaNet ≥ 6-bit and
   hyper-connections/`lm_head` at 8-bit. On the planned K=320 build with 3.3 bpw experts, that gives
   1.69 (DeltaNet) + 0.50 (QSA) + 0.67 (HC) + 0.19 (shared) + 0.07 (router) + 0.68 (`lm_head` 8-bit) + 0.97
   (experts) + 0.03 (PLE proj) = **4.80 GB of weights** + 0.23 state = 5.03 GB → ceiling **122 tok/s** [computed].
   That is 20 % below the baseline ceiling, but because Qwen is overhead-bound (§2.2) the real cost is ≈ 10 %
   (≈ 2.3 ms per step) [model].

**GLM-5.3-Flash [measured: tensor shapes from the original checkpoint's safetensors headers, fetched by HTTP range
request; no local weights touched].** Active parameters per token:

| Component | Params per layer | Layers | Active params |
|---|---|---|---|
| KDA (q/k/v/o 4 × 33.55M + gates ≈ 3.7M) | 137.9 M | 34 | 4.69 B |
| DSA/MLA (o 67.1 + q_b 25.2 + kv_b 16.8 + indexer 7.5 + q_a 6.3 + kv_a 2.1) | 124.9 M | 11 | 1.37 B |
| Dense MLP (layers 0–2) | 151.0 M | 3 | 0.45 B |
| Shared expert | 25.2 M | 42 | 1.06 B |
| Routed experts, 8 × 25.17 M | 201.3 M | 42 | **8.46 B** |
| `lm_head` 154,880 × 4,096 | | 1 | 0.63 B |
| Router, mHC (`hc_*_fn`) | ≈ 2 M | 45 | 0.09 B |
| **Total active** | | | **16.75 B** (consistent with the "~18B" marketing figure once the MTP layer is counted) |

Bytes per token depend on the trunk bits more than the expert bits, because the trunk is half of the active
parameters:

| GLM build | Trunk | Experts | Weights/token | + KDA state (34 × 4.19 MB × 2) | Ceiling |
|---|---|---|---|---|---|
| Planned K=144 (reap §Rec. 2), floors per quant §Rec. 2 | 6.5 bpw, `lm_head` 8.5, router bf16 → 6.95 GB | 3.0 bpw → 3.17 GB | 10.12 GB | +0.28 | **59 tok/s** |
| Same, trunk at 4.5 bpw | 5.05 GB | 3.17 GB | 8.22 GB | +0.28 | 72 tok/s |
| `pipenetwork` REAP50 mixed 4/8 (assumed 8-bit trunk, 4-bit experts) | 8.08 + 0.67 + 0.1 GB | 4.76 GB | 13.6 GB | +0.28 | 44 tok/s |

**Other candidates (ceilings only, 4.5 bpw unless noted) [computed]:**

| Model | Active | Bytes/token | Ceiling | Notes |
|---|---|---|---|---|
| Qwen3.8-27B dense | 27 B | 15.2 GB | 40 tok/s | Measured community decode on M5 Max for the 27B class: 30–33 tok/s [self-reported] → **~80 % of the ceiling**. Dense models are bandwidth-bound |
| Mistral-Medium-3.5-128B dense | 128 B | 72 GB | 8.5 tok/s | Too slow for agent loops without heavy speculation |
| Nemotron-3.5-Super-120B-A12B | 12 B | 6.8 GB | 90 tok/s | |
| DeepSeek-V4-Flash 284B-A13B | 13 B | 7.3 GB | 84 tok/s | Does not fit 80 GiB without ≥ 50 % pruning or SSD streaming |

### 2.2 Where decode time actually goes: the fixed-overhead term

The roofline is an upper bound. The real step time is better described by:

```
T(1) = t0 + bytes_touched / BW_eff
```

- `BW_eff` is the bandwidth the dequantising GEMV kernels actually achieve: 310–485 GB/s for GLM-size experts and
  lower for Qwen-size experts (quant §4, measured on this machine).
- `t0` is everything that does not scale with bytes: kernel launches, the GatedDeltaNet recurrence, small
  element-wise ops between matmuls, argpartition for routing and sparse attention, sampling, and the CPU↔GPU sync.

**Fitting `t0` for Qwen.** Two independent runtimes report plain decode for Qwen3.8-Flash-Next (4-bit, top-10, same
bytes per token as our baseline) on an M5 Max: 43.8 tok/s (MTPLX) and 47–48 tok/s (mlx-serve issue #366)
[self-reported]. Take T(1) ≈ 21.5 ms. Bytes at a realistic 450 GB/s take 4.02 / 450 = 8.9 ms. So
**t0 ≈ 12.6 ms**: ≈ 60 % of the step is overhead [computed]. Per layer that is ≈ 260 µs, consistent with the 80–130 µs
quant §4 measured for the MoE block alone (latency-bound at Qwen's tiny 640 × 2560 experts).

Contrast: a dense 27B on the same machine reaches ~80 % of its bandwidth ceiling (§2.1), and a 35B-A3B MoE reaches
141 tok/s against a ≈ 360 tok/s ceiling, i.e. ~39 % [self-reported, dflash-mlx table]. **The finer the MoE, the more
of the step is overhead.** Three consequences drive the plan:

1. **Speculation pays most where `t0` is large.** Verifying 4 tokens repeats `t0` once, not 4 times.
2. **Shaving bits pays least where `t0` is large.** Going from 4.5 to 3.3 bpw experts saves 0.36 GB = 0.8 ms of a
   21.5 ms step (4 %).
3. **Kernel work should target dispatch count and fusion** (fewer, larger kernels), not bandwidth.

For GLM, `t0` is unmeasured. Assuming the same per-layer overhead (≈ 11.8 ms for 45 layers) and BW_eff 420 GB/s, the
planned K=144 build gives T(1) ≈ 11.8 + 10.4/0.42 = 36.6 ms ≈ **27 tok/s** [model]. Reference points: a 4-bit unpruned
GLM-5.3-Flash on an M3 Ultra (819 GB/s) decodes 29.3–29.7 tok/s plain, and a 2-bit build on an M5 Max decodes
17.1 tok/s at 32K with MTP on [both self-reported, oMLX]. GLM is closer to bandwidth-bound (≈ 70 % of its step is
bytes), so for GLM trunk bits *do* matter for speed.

### 2.3 Speculative decoding: acceptance, verification cost, and the MoE twist

**Mechanics.** A cheap drafter proposes γ tokens d₁…d_γ. The target runs one forward pass over
`[last_token, d₁, …, d_γ]` (γ+1 positions) and produces its own choice at each position. Accept the longest prefix
where the draft agrees, then take one "bonus" token from the target at the first disagreement. Every round yields at
least one token, so the output can never be worse than plain decoding in *tokens per round*. It can be worse in
*time*.

**What "agree" means in mlx-vlm [src: `speculative/common.py:_speculative_walk`].** The walk compares each draft token
with the token the *target sampler* picks at that position, and stops at the first mismatch. At temperature 0 that
is exact greedy decoding. At temperature > 0 each emitted token is still a sample from the target's conditional
distribution, so the output distribution is unchanged, but the acceptance probability of a draft token d is
p_target(d), not the Σ min(p, q) of Leviathan et al.'s rejection rule. **Measure acceptance at the temperature you
actually serve** (§5). Agentic coding with thinking at T ≈ 0.6–1.0 accepts less than the greedy numbers in model
cards.

**Expected tokens per round** (each draft accepted with probability α, independently):

```
E[tokens/round] = 1 + α + α² + … + α^γ = (1 − α^(γ+1)) / (1 − α)
```

α = 0.78, γ = 3 → 2.86 tokens per round [computed]. Diminishing returns: γ = 6 gives only 3.50.

**Speedup**, where `T(k)` is the time to verify k = γ+1 positions and `T_d` the time per draft token:

```
speedup = E[tokens/round] · T(1) / ( T(γ+1) + γ · T_d )
```

On a dense model `T(γ+1) ≈ T(1)` for small γ, because the weights are read once and reused for every row. On an MoE
**each verified position routes to its own top-k experts**, and the union is what you read:

```
u(k) = E · (1 − (1 − K/E)^k)              unique experts per layer, independent routing
bytes(k) = trunk + routed_per_token · u(k)/K + state · (1 + k)
```

The `state · (1 + k)` term is real: GatedDeltaNet/KDA layers write one intermediate recurrent state per verified
position so that a rejection can roll back (decode §1: 0.33 GiB transient for Qwen at block 4).

Real routing is correlated across neighbouring tokens. Cohere measured 2.5× top-k unique experts for 4-token verify
against 3.2× under independence (20–31 % fewer) [self-reported]. The model in Appendix A uses
`u_corr = K + 0.75 · (u(k) − K)`.

Verify cost relative to one plain step [model, Appendix A]:

| k (positions) | Qwen-288 4-bit: unique experts/layer | T(k)/T(1) | GLM K=144 plan: unique/layer | T(k)/T(1) |
|---|---|---|---|---|
| 1 | 10 | 1.00 | 8 | 1.00 |
| 4 | 31 | 1.36 | 24 | 1.47 |
| 8 | 56 | 1.81 | 42 | 1.99 |
| 16 | 96 | 2.56 | 67 | 2.78 |
| 32 | 149 | 3.70 | 93 | 3.72 |

Three lessons:

1. **There is an optimal draft length, and it is short.** For Qwen with α = 0.8, the model gives 1.52× at block 2,
   1.91× at block 4, 1.95× at block 5–6. For GLM the optimum is block 3–4 (≈ 1.5×), and at α = 0.6 GLM gains only
   1.2×. mlx-vlm's defaults (Qwen 4, GLM 3; decode §1) sit at these optima.
2. **Pruning helps verification.** With fewer experts (E = 144 instead of 288), more verified tokens share experts.
   GLM K=144 verifying 16 positions reads 67 unique experts per layer rather than the ≈ 80 it would read at E = 288
   [computed, Appendix A formula].
3. **Long drafts are only worth it when acceptance is near-certain.** Breaking even at k = 16 on Qwen needs > 2.56
   accepted tokens per round. That is out of reach for a learned drafter at α ≈ 0.8 (E ≤ 5 for any γ), but easy for
   an n-gram drafter copying a file (11.7 tokens per round in Appendix B).

Independent evidence of the MoE effect: on Qwen3-Coder-30B-A3B with EAGLE-3, unique experts per verify grow from
12.1 (k=1) to 30.2 (k=10) and verify time from 9.0 to 13.7 ms; an oracle draft-length policy plateaus at ≈ 2.3×
with average draft length ≈ 2.8 [self-reported, arXiv 2609.22156]. On M5 Max with MLX, DFlash gives 2.1–2.8× on
dense 27B models but only 1.8× on 35B-A3B MoEs at the same ~90 % acceptance [self-reported, dflash-mlx].

### 2.4 Native MTP in mlx-vlm

**What MTP is.** Multi-token prediction heads are trained during pre-training. The MTP layer takes the trunk's final
hidden state h_t and the embedding of the next token x_{t+1}, mixes them (`eh_proj`/`fc`), runs one extra decoder
layer (for these models a full MoE layer), and predicts x_{t+2} through the shared `lm_head`. Chaining it γ times
gives γ drafts. Because it reads the trunk's own hidden state, it knows what the trunk "was thinking", which a
separate small model does not.

**What it costs per draft token.** One decoder layer plus the full `lm_head`. For Qwen-288 at 4-bit that is ≈ 44 MB
(trunk part of one layer) + 28 MB (10 experts) + 358 MB (`lm_head`) ≈ 0.43 GB, plus one layer of overhead
→ **≈ 1.3 ms** [model]. The `lm_head` is the largest part. EAGLE-3 avoids it with a 32K "hot" draft vocabulary
(`d2t` map) [src: `drafters/eagle3/eagle3.py`]. For GLM the MTP layer has 288 experts of 25 M each, and the drafter
costs ≈ 1 GB per draft token.

**Adaptive depth [src: `speculative/mtp.py:_effective_mtp_block_size`].** If you request a block larger than the
drafter's configured depth, mlx-vlm uses the larger block only when ≥ 65 % of the last 32 rounds accepted the full
configured depth. Otherwise each round would pay for drafts that cannot be accepted. This is a crude version of
the cost-aware policy in §7.

**State after pruning (reap §7).** The unpruned Qwen MTP head accepted 76–81 % on a K=320 trunk (+32 % decode);
sh0wie's separate bf16 drafter accepted 44–68 %; Litwein's pruned head did worse until self-distilled (+2–3 %)
[self-reported]. GLM's MTP accepted ≈ 70 % at any bit width from Q2_K to Q6_K, so quantise it hard [self-reported].

**Thinking budgets.** mlx-vlm cannot enforce a thinking budget while drafting, so `serve/engine.py` drops
`thinking_budget` when MTP is active [src: `serve/engine.py:build_gen_args`]. Plan for that in the router (ch. 08).

### 2.5 Learned drafters: EAGLE-3, DFlash, and trees (DDTree)

**EAGLE-3** (Li et al., NeurIPS 2025). A one-layer decoder that reads a fusion of the target's hidden states from
three depths (low, middle, high: `target_layer_ids`, default `[2, n/2, n−3]` [src: `drafters/eagle3/config.py`]),
concatenated and projected by `fc: 3H → H`. Two ideas made it scale:

- *Direct token prediction*: EAGLE-1/2 regressed the next hidden *feature*, which capped gains from more data.
- *Training-time test (TTT)*: during training the drafter is unrolled on its *own* previous outputs for several
  steps, exactly as at inference. That removes the train/test mismatch of teacher forcing, so acceptance holds up deep
  into the draft.

Up to 6.5× on dense targets [self-reported]. It needs (a) the target to expose per-layer hidden states and (b)
`rollback_speculative_cache`. Qwen's `qwen4_exp` has both. **GLM's `glm5_next` has neither** [src: `eagle3.py:367`,
decode §1].

**DFlash / DFlash2** (z-lab; arXiv 2602.06036). A block-diffusion drafter: in one forward pass it fills a whole block
of 8–16 masked positions, conditioned on hidden states captured from ~5 target layers. Drafting costs one pass
instead of γ, which is why it allows long blocks. It has the same target requirements as EAGLE-3. Status for us
(decode §1): the Qwen DFlash2 checkpoint is structurally compatible but untested; GLM is blocked.

**Why long DFlash blocks are a weaker deal on our MoEs** [model]: τ ≈ 4.5 tokens per round on a 16-token block
costs T(16) = 2.56 T(1) on Qwen-288, giving ≈ 4.5 / 2.7 ≈ **1.7×**, about the same as MTP's 1.9×. On a dense 27B,
T(16) ≈ 1.1–1.3 T(1), so the same τ gives ≈ 3.5×, matching what DFlash reports on dense targets.

**Trees and DDTree.** A chain drafts one continuation. A tree drafts several branches and verifies them all in one
pass with an ancestor-only attention mask, so a miss at depth 2 can be rescued by the second-best token.
`mlx_vlm/speculative/ddtree.py` implements DDTree's Algorithm 1 (Ringel & Romano, arXiv 2604.12989) [src]. Given
per-position drafter log-probs ℓᵢ, the expected accepted length of a prefix-closed tree T under the drafter's
factorised distribution is

```
E[α_T] = Σ_{u ∈ T} Π_{i ≤ |u|} q_i(u_i)        (sum of path probabilities of every node)
```

So the best tree with B nodes is simply the B most probable prefixes. Lemma 1 restricts each depth to its top-K tokens.
A max-heap pops the best prefix and pushes its first child and next sibling, which is O(B log B).

**In 0.7.9 nothing calls `build_ddtree`.** It is exported from `speculative/__init__.py` and used nowhere else
[src, grep]. Trees are also hard on our targets for two structural reasons:

1. **Recurrent layers cannot use a tree mask.** 36/48 (Qwen) and 34/45 (GLM) layers are GatedDeltaNet/KDA. Their state
   is a running sum over the sequence, so each branch needs its own copy of the state. Siblings cannot share one pass
   through a masked attention.
2. **MoE union cost.** A tree of B nodes costs at least T(B) in expert reads. CaDDTree (arXiv 2606.01813) makes the
   tree budget cost-aware: maximise (1 + Φ(T)) / (C_d + C_v(|T|)), with C_v profiled per model and hardware. EcoSpec
   (arXiv 2607.12696) scores candidates by path probability divided by the number of *new* experts they add: +11–15 %
   over EAGLE-3 on Qwen3-235B and GPT-OSS-120B (1.22→1.36×, 1.14→1.31×) [self-reported].

Trees are therefore a stretch idea here (§7.4), not a plan item.

### 2.6 Model-free drafting: prompt lookup / suffix decoding for agentic coding

**Prompt-lookup decoding (PLD).** Take the last n generated tokens. Find their most recent earlier occurrence in
prompt + output. Propose the tokens that followed it. Draft cost is a CPU string search (microseconds), with no
model at all.

**SuffixDecoding** (Oliaro et al., NeurIPS 2025, arXiv 2411.04975) generalises PLD with suffix trees over the prompt
*and over previous outputs*. It scores candidate continuations by frequency and adapts the speculation length to the
match: long when the match is long and confident, short otherwise. Reported: **5.3×** mean on AgenticSQL, **2.5×** on
SWE-Bench (where EAGLE-2/3 failed on sequence length), 1.7× over PLD; behind EAGLE-3 on non-repetitive Spec-Bench,
where a hybrid reached 2.5× [self-reported]. It ships in Snowflake's ArcticInference for vLLM.

**Why it fits agent traffic.** Claude Code's `Edit` tool copies `old_string` verbatim from a `Read` result.
`new_string` is mostly `old_string`. `Write` re-emits whole files with small changes. Paths, identifiers, error
strings and test names recur constantly across turns.

**A measured detail that matters: the tool-call format.** Qwen's chat template emits tool arguments as raw XML
parameters (`<parameter=old_string>\n…raw text…\n</parameter>`) [src: `chat_template.jinja`], not JSON strings. JSON
escaping (`\n`, `\"`) produces different tokens from the raw text in context, and breaks matches. Claude Code's `Read`
output also prefixes every line with a line number (`cat -n` style), which breaks matches at every line start.
Tokenizer-level simulation with Qwen's real tokenizer over 40 real Python files [measured, Appendix B]:

| Scenario (oracle greedy target = the true output; max draft 8 / 16) | Tokens per verify round |
|---|---|
| `Edit` call, JSON-escaped arguments, numbered `Read` context | 2.86 / 3.18 |
| `Edit` call, Qwen XML arguments, numbered `Read` context | 3.26 / 3.73 |
| `Edit` call, Qwen XML arguments, **context indexed without line numbers** | **4.23 / 5.35** |
| Whole-file rewrite with 3 small changes | 6.75 / 11.68 (20.1 at max draft 32) |

The third row is a free trick. The drafter's index does not have to equal the prompt. Index a copy of each tool
result with the `^\s*\d+\t` prefixes stripped. The prompt the model sees is unchanged, so outputs are unchanged.
Only the guesses improve.

Converted to speedups with the cost model (n-gram draft cost ≈ 0) [model]:

| Segment | Qwen-288 | Qwen-320 plan | GLM-144 plan |
|---|---|---|---|
| `Edit` args, numbered context, k=8 | 1.9× | 2.2× | 1.7× |
| `Edit` args, de-numbered index, k=8–16 | **2.5×** | **2.8–3.0×** | **2.3×** |
| Whole-file rewrite, k=16 / 32 | **4.4× / 5.5×** | 5.3× / 6.7× | 4.1× / 5.5× |
| Fresh reasoning prose | ≈ 1.0× (no matches → plain steps) | | |

Expect less in real traffic. The oracle assumes the target would emit exactly the true output, and real edits mix in
new code. The direction and the ranking are robust, though. SuffixDecoding's 2.5× on SWE-Bench is an independent
anchor [self-reported].

**Hybrid policy.** Per round: if the suffix index has a match of length ≥ m_min (e.g. 4 tokens) with a continuation
of ≥ 4 tokens, draft up to k_ngram = 16 from it. Otherwise draft γ = 3 with MTP. Inside a `<parameter=old_string>`
block, prefer n-gram unconditionally.

### 2.7 Prefill: neural accelerators, chunking, rows-per-expert

Prefill FLOPs ≈ 2 · N_active · T (+ attention). Qwen: 2 × 5.9 B = 11.8 GFLOP/token. At the 20.8 TFLOP/s measured for
Qwen3.8-Flash-Next at 65K on an M5 Max (42 % of the ≈ 50–57 TFLOP/s 4-bit GEMM ceiling) that is **≈ 1,760 tok/s**,
matching the 1,300–2,500 tok/s measured there [self-reported, mlx-serve #366] [computed]. GLM: 2 × 16.75 B =
33.5 GFLOP/token → ≈ 620 tok/s at the same efficiency. oMLX measured 328 tok/s at 32K for a 2-bit GLM on an M5 Max
[self-reported]. Plan with **Qwen ≈ 1,500 tok/s and GLM ≈ 330–600 tok/s** until `evals.speed` replaces these numbers.

**Neural accelerators** are used automatically for multi-row quantized matmuls, including the sorted MoE path
`gather_qmm_rhs` (quant §4, mac §6). There is no flag. Decode kernels have no NAX variant.

**Rows per expert is the hidden knob of MoE prefill.** A prefill chunk of S tokens with top-K routing over E experts
gives each expert ≈ S·K/E rows. GEMM tiles need tens to hundreds of rows to be efficient. mlx-serve measured sorted
`gather_qmm` at 14.2 TFLOP/s for 1,024-token chunks, 28.3 at 4,096 and 41.3 at 16,384 [self-reported]. Two
consequences:

- **Larger chunks run faster**, up to the memory peak decode §4 measured. Qwen at 4,096: 4,096 × 10 / 288 =
  **142 rows per expert**, against 80 for the unpruned 512 experts [computed].
- **REAP pruning speeds up prefill** at a fixed chunk size by concentrating rows: 288 experts gives 1.78× more rows
  per expert than 512.

mlx-vlm sorts tokens by expert once `indices.size >= 64` [src: `switch_layers.py:298`]. A 7-token verify with
top-10 (70 ≥ 64) takes the sorted path. 4 tokens (40) do not.

**Attention grows with context.** At a 32K kv length, SDPA was 36 % of a prefill chunk's time for Qwen
[self-reported, mlx-serve]. Decode slows with context too: MTPLX reports 73.5 tok/s on a short coding task and
61.8 tok/s at a 109K-token turn [self-reported].

### 2.8 Prefix caching for agent loops, SSD persistence, compaction

**APC on hybrid models** (decode §5). Recurrent state cannot be split into 16-token pages, so mlx-vlm uses
`strategy=checkpoint`: it snapshots the whole cache (KV for the attention layers plus the fixed recurrent state) at
prefix boundaries every 2,048 tokens. A new request that extends a cached prefix restores the nearest checkpoint and
prefills only the remainder.

```
TTFT(turn n) ≈ t_restore + (T_n − P) / prefill_rate        instead of        T_n / prefill_rate
```

| Session state | Qwen, no APC (1,500 tok/s) | Qwen, APC hit (+3K new) | GLM, no APC (330 tok/s) | GLM, APC hit |
|---|---|---|---|---|
| Turn 1 (system + tools ≈ 25–33K [self-reported, third-party counts]) | 17–22 s | — (cold) | 76–100 s | — (cold) |
| Turn n at 60K history | **40 s** | **2 s** + restore | **182 s** | **9 s** + restore |
| Turn n at 120K history | 80 s | 2 s + restore | 364 s | 9 s + restore |

[computed]. That is a 20–40× TTFT reduction, larger than every other lever in this chapter combined, for agentic use.

**Checkpoint sizes** (decode §3, with MTP): Qwen 1.05 GiB at 32K and 3.87 GiB at 128K; GLM 0.53 / 1.73 GiB, but only
once the projected-cache fix is in (decode Rec. 1). Without it each GLM checkpoint is ≈ 25 GiB.

**SSD persistence.** `serve/` already sets the APC disk tier (`--prefix-cache-disk-gb`, default 32) under
`~/.cache/local-ai/prefix-cache/<name>-<fingerprint>`, keyed by a model fingerprint so a rebuilt model never restores
stale state [src: `serve/config.py`]. Restoring a 4 GiB checkpoint is a few GB of sequential SSD read, i.e. well under
2 s. MTPLX reports restoring a 96,760-token session in 8 ms from its SSD cache [self-reported; likely mmap]. Either way
it beats re-prefilling 60K tokens by more than an order of magnitude.

**Prefix stability is the whole game.** A single changed byte early in the prompt (a timestamp, a re-ordered tool
list, a dynamic attribution header) invalidates everything after it. One report saw 30–50 s of re-prefill every turn
on a 17K context until volatile blocks moved to the end [self-reported]. decode §5: set
`CLAUDE_CODE_ATTRIBUTION_HEADER=0` if in doubt.

**Compaction.** Claude Code compacts at `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (decode §6). A compaction rewrites the prefix,
costing one cold prefill of the summary, but it also bounds the per-checkpoint memory and the context-dependent decode
slowdown. With APC working, the reason to compact is decode speed and quality, not prefill. mlx-vlm also ships
server-side compaction (`server/compaction.py`, a fixed handoff prompt) [src].

### 2.9 Kernels: what is already custom, what is left

Already in mlx-vlm 0.7.9 [src]:

- `models/gated_delta.py`: a `mx.fast.metal_kernel` for the GatedDeltaNet/KDA recurrence (`gated_delta_kernel`).
- `models/quantized_verifier.py` and `exact_speculative_verify.py`: custom Metal kernels for batch-invariant
  speculative verification, including fused quantized MoE + hyper-connection ops. `qwen4_exp` imports them.
- `@mx.compile` on Qwen's RMSNorm, hyper-connection pointwise ops, injection and q/k normalisation
  (`qwen4_exp/language.py:743, 1743–1768`), and on two GLM helpers.
- `SwitchGLU` issues three `gather_qmm` calls (gate, up, down) plus SwiGLU and a weighted combine per layer, and sorts
  tokens by expert at ≥ 64 routed rows (`switch_layers.py:249–330`).

What the community measured on exactly this model and machine (mlx-serve #366, a different runtime linking libmlx)
[self-reported]:

| Idea | Component result | Whole-model result |
|---|---|---|
| Fused gather + reduce (combine experts in-kernel) | 2.3–2.5 ms → 0.67 ms per call at S=4,096, exact | unconfirmed |
| Hyper-connection up-projection + mix fusion | 1.44–1.60× | est. 0.4–0.5 s per 65K prefill |
| GDN pre-work widened to S=8,192 | 2.41× | — |
| MoE grouping + inverse reduction | 1.14× | — |
| Gate/up bank fusion | **block regressed 12 %** | rejected |
| Dequantise experts to dense and gather | 2.2× slower | rejected |
| "Unsorted expert path is 19× slow" | was a path production never takes | **no gap** |

BaseRT (arXiv 2607.19438, on M5 *Pro*) got 1.02–1.33× decode over MLX from "dispatch and kernel-fusion work", and up
to 3.9× prefill on MoE with NAX tensor kernels, including a fused MoE gate/up + SiLU [self-reported].

**What is realistically worth doing here** (ranked in §4.8): reduce dispatches in the decode step (`mx.compile` of
whole sub-blocks that are shape-static at decode: the hyper-connection mix → norm → projections chain, router
softmax + top-k + renormalise), and a fused decode MoE kernel (gate/up GEMV + SiLU·mul + down GEMV + weighted combine
for K experts in one dispatch). Prefill GEMMs are already NAX-tiled in MLX; beating them is a research project.

---

## 3. Evidence

| Claim | Source | Status |
|---|---|---|
| Qwen baseline: 3.79 GB weights/token; routed 1.33 GB; `lm_head` 0.36 GB | safetensors headers of `models/qwen-flash-next-reap288` (131 shards, 3,570 of 3,743 tensors; the missing 7 shards hold `lm_head`/embeddings/last layers, sized from the index + config) | **[measured]** |
| GLM: 16.75 B active; 8.46 B routed; KDA 4.69 B | `zai-org/GLM-5.3-Flash` shard headers via HTTP range request + local `config.json` | **[measured]** |
| mlx-vlm's acceptance is draft == target-sample, stop at first mismatch | `speculative/common.py:173–196` | **[src]** |
| Adaptive MTP depth: 65 % of last 32 rounds | `speculative/mtp.py:511–541` | **[src]** |
| DDTree implemented but unused | `speculative/ddtree.py`; only reference is `speculative/__init__.py` | **[src]** |
| EAGLE-3 / DFlash need `rollback_speculative_cache` + capture; GLM lacks both | `speculative/eagle3.py:367`, `drafters/compatibility.py`, decode §1 | **[src]**, [research] |
| No n-gram/prompt-lookup drafter in mlx-vlm 0.7.9 or mlx-lm 0.32.2 | grep over both packages | **[src]** |
| Qwen chat template uses raw XML tool parameters | `models/qwen-flash-next-reap288/chat_template.jinja:68,137` | **[src]** |
| PLD acceptance on edit-style outputs: 3.3–5.4 tokens/round; rewrites 6.8–20 | Appendix B simulation, Qwen tokenizer, 40 files | **[measured]** (oracle target) |
| Qwen3.8-Flash-Next plain decode 43.8 tok/s, MTP 73.5 (1.7×) on M5 Max 128 GB | [MTPLX model card](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed) via decode §1 | [self-reported] |
| Same model: decode 47–48 tok/s; prefill 1,329–2,509 tok/s at 16–131K; 20.8 TFLOP/s at 65K; gather_qmm 14→41 TFLOP/s with chunk 1K→16K | [mlx-serve #366](https://github.com/ddalcu/mlx-serve/issues/366) | [self-reported] |
| GLM-5.3-Flash 2-bit on M5 Max: PP 327.8, TG 17.1 tok/s at 32K, MTP on | [oMLX benchmark](https://omlx.ai/benchmarks/performance/88m4imqs) | [self-reported] |
| GLM-5.3-Flash 4-bit on M3 Ultra: AR 29.3–29.7; MTP +0–35 % short, ±10 % at 20K agentic | [grant-ai model card](https://huggingface.co/grant-ai/GLM-5.3-Flash-Abliterated-MLX-4bit) | [self-reported] |
| DFlash on M5 Max/MLX: dense 27B 2.1–2.8×, 35B-A3B MoE 1.8×, acceptance 85–90 % | dflash-mlx README via [explainx summary](https://explainx.ai/blog/dflash-2-mlx-m5-max-local-inference-august-2026) | [self-reported], secondary |
| MoE verify: unique experts 12.1→30.2 for k=1→10; oracle speedup plateaus at 2.34× | [arXiv 2609.22156](https://arxiv.org/html/2609.22156) | [self-reported] |
| Correlated routing: 4-token verify = 2.5× top-k unique experts vs 3.2× independent | [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) | [self-reported] |
| Expert-aware draft selection: EcoSpec 1.22→1.36× (Qwen3-235B), 1.14→1.31× (GPT-OSS-120B) | [arXiv 2607.12696](https://arxiv.org/html/2607.12696v1) | [self-reported] |
| SuffixDecoding: 5.3× AgenticSQL, 2.5× SWE-Bench, 1.7× over PLD | [arXiv 2411.04975](https://arxiv.org/abs/2411.04975v2) | [self-reported] |
| EAGLE-3: up to 6.5×; TTT; more data → more speedup | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html) | [self-reported] |
| DFlash drafters trained on ~100K target-regenerated samples; EAGLE-3 SpecForge 1.4M; DFlash stores multi-layer hidden states (5× EAGLE-3) | [DFlash arXiv 2602.06036](https://arxiv.org/pdf/2602.06036), [z-lab Qwen3-Coder card](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash), [NVIDIA Model-Optimizer dflash.md](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/doc/dflash.md) | [self-reported] |
| CaDDTree: throughput-optimal tree budget from a profiled convex C_v | [arXiv 2606.01813](https://arxiv.org/html/2606.01813) | [self-reported] |
| BaseRT on M5 Pro: decode 1.02–1.33× over MLX; prefill up to 3.9×; fused MoE gate/up + SiLU | [arXiv 2607.19438](https://arxiv.org/html/2607.19438v1) | [self-reported] |
| Claude Code fixed prompt ≈ 20–33K tokens (system + tools) | [morphllm](https://www.morphllm.com/claude-code-token-limit), [DEV](https://dev.to/ashraf_chowdury09/claude-code-burns-33k-tokens-before-it-even-reads-your-prompt-45p2) | [self-reported], third-party; measure with `/context` |

---

## 4. Plan for this machine

Order matters: each step either removes the largest remaining cost or produces the measurement the next step needs.
GPU-heavy steps run one at a time through the orchestrator (PLAN.md rules).

### 4.0 Step S0 — make speed measurable on *agentic* traffic (½ day)

`evals/speed.py` measures prefill/decode on filler prose ("continue this chronicle", thinking off). Speculative
acceptance on that text says little about coding. `scripts/tune_runtime.py` already sweeps MTP block sizes on four
short prompts. Add a trace-replay suite:

```
evals/specbench.py   uv run python -m evals.specbench --model models/<name> --traces evals/fixtures/agent_traces/ \
                         --modes plain,mtp,ngram,hybrid --blocks 2,3,4,5 --ngram-max 8,16,32 --temps 0,0.6,1.0
```

- **Traces**: 30–50 synthetic multi-turn agent sessions (Read → Edit → Bash → test output → Edit …), written once in
  Qwen's and GLM's native tool formats from `calib_data`'s `agentic` domain generator, plus 10 reasoning prompts and
  5 long-document QA prompts. Optionally, real sessions (§4.6, privacy rules apply).
- **Per request, record**: TTFT, prefill tok/s, decode tok/s, `draft_n`, `draft_n_accepted`, `draft_rounds` (mlx-vlm
  returns these in `GenerationMetrics` [src: `server/generation.py`]), mean accepted length per segment type (prose /
  thinking / tool args / code block), APC hit tokens, restore time, peak memory.
- **Verify-cost curve T(k)**: time a forward of k ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32} positions at kv lengths
  {1K, 32K, 128K}, 20 repeats after 5 warm-ups (the CaDDTree protocol). Fit `t0`, `BW_eff` and the correlation factor
  of Appendix A. **This curve is the input to every draft-length decision.**

Memory: one model load, no extra (the curve reuses the cache). Time: ≈ 1–2 h per model.

### 4.1 Step S1 — prefix caching that actually hits (1 day)

`serve/` already enables APC with an SSD tier. Work items:

1. **Prefix-stability probe.** Record the first divergence position between consecutive requests in a Claude Code
   session (log only the token index, never content). Target: divergence ≥ 99 % of the previous prompt length on
   every turn except compactions. If it fails, find the volatile block (attribution header, date, tool order) and fix
   it client-side (`CLAUDE_CODE_ATTRIBUTION_HEADER=0`, decode §6).
2. **Checkpoint placement.** The default interval of 2,048 tokens costs up to 2,047 re-prefilled tokens per hit
   (≈ 1.4 s Qwen, ≈ 6 s GLM). mlx-vlm also checkpoints at request boundaries; verify that the "end of previous
   prompt" boundary is among the stored entries (`/v1/cache/stats`). If not, add a checkpoint at the end of each
   finished request (the next turn always extends exactly there).
3. **Budget** (work profile, Qwen-320 plan, 128K max) [computed]:

   | Item | GiB |
   |---|---|
   | Resident weights (PLE table mmapped, quant §Rec. 6) | ≈ 38–40 |
   | MTP head (512 experts 4-bit, reap §Rec. 7) | ≈ 1.4 |
   | KV + state at 128K with MTP (decode §3) | 3.9 |
   | APC RAM: 2 entries × 3.9 (128K) or ≈ 6 × 1.05 (32K) | ≤ 8 (`--prefix-cache-memory-gb 8`) |
   | Speculative rollback transient (decode §1) | 0.33 |
   | Prefill transient, step 4096 (decode §4 scaled; measure) | ≈ 3 |
   | MLX buffer cache (`--cache-limit-gb 2`) | 2 |
   | **Total** | **≈ 57–59**, inside 80 |

   GLM-144 plan: 62–66 GB weights (≈ 58–61 GiB) + 1.7 KV + MTP 1.3–2.4 + APC 2 × 1.7 + prefill peak at 128K
   8 GiB (decode Rec. 2) ≈ **74–77 GiB**. That fits only with the projected-cache fix, `--prefix-cache-entries 2`,
   `--prefill-step-size 2048`, and APC RAM ≤ 4.
4. **Session pre-restore** (ch. 08 §7.5): during idle gaps, restore the latest session's checkpoint from SSD into the
   RAM tier.

Accept when warm-turn TTFT ≤ 3 s (Qwen) / ≤ 12 s (GLM) for +3K new tokens at a 60K history.

### 4.2 Step S2 — native MTP tuned on real traffic (½ day)

```bash
# server (MTP auto-detected from the model dir; build/ must keep mtp.* / layer-45 tensors, decode §1, reap §Rec. 7)
scripts/serve.sh qwen-flash-next-reap320 --mtp on --draft-block-size 4
# sweep
uv run python -m evals.specbench --model models/qwen-flash-next-reap320 --modes plain,mtp --blocks 2,3,4,5,6 --temps 0,0.6
uv run python scripts/tune_runtime.py models/qwen-flash-next-reap320 results/runtime_tune.json   # existing quick sweep
```

- Expected [model]: Qwen block 4 → **1.6–1.9×** at α = 0.7–0.8; GLM block 3 → **1.3–1.5×**.
- Check that the measured α (`draft_n_accepted / draft_n`) is close to reap §7's 76–81 % for Qwen. If it is below
  0.65, the MTP head is mismatched with the pruned/requantised trunk. Go to S5 (fine-tune) before anything else.
- GLM drafter: quantise its experts to 2–3 bit (acceptance is insensitive to bits, reap §7). Compare an MTP pruned to
  144 experts with the full 288 (1.3–1.7 vs ≈ 2.4 GB).
- Pin `--draft-block-size` per model from the sweep. The adaptive rule then only raises it when ≥ 65 % of rounds
  fill.

### 4.3 Step S3 — suffix / prompt-lookup drafter (2–4 days)

New module `serve/ngram.py` (no new dependencies; pure Python + `array`).

**Data structure.** A suffix automaton (or a hash map from the last n ∈ {1..8} token ids to their most recent end
positions) over the token stream `prompt ⊕ generated`, rebuilt incrementally per request. Per session, keep a second
index over *previous turns' outputs* (SuffixDecoding's "output tree") so that a re-emitted `new_string` from two turns
ago is found. Memory: ≈ 16–40 bytes per indexed token → 128K tokens ≈ 2–5 MB [computed]. Negligible.

**Index normalisation (the measured trick, §2.6).** For each tool-result span, also index a copy with
`^\s*\d+\t` line prefixes stripped. Optionally index JSON-unescaped copies when serving a JSON-argument model. The
prompt is never modified.

**Draft policy** (per round):

```python
def propose(self, hist, kmax=16, m_min=3):
    for n in range(8, m_min - 1, -1):             # longest suffix first
        pos = self.index.last_end(hist[-n:])        # most recent earlier occurrence
        if pos is not None:
            cont = self.tokens[pos : pos + kmax]
            if len(cont) >= 2:
                return cont, n                      # n = match length → confidence
    return [], 0
```

Choose the draft length from the match length n and the measured T(k) curve. Maximise
`E_n[accepted | k] / T(k)` with E_n estimated online per n (a 32-bucket running average). This is CaDDTree's
objective applied to a chain, and it needs no grid search.

**Integration, preferred route.** Implement `NGramDrafter` with the duck-typed interface mlx-vlm's MTP round loop
calls [src: `speculative/mtp.py:_mtp_rounds`, `_mtp_draft_block_active`]: `reset(model)`, `set_shared_kv(...)`
(ignored), `draft_block(...)` returning token ids, `config.block_size`, `accept_lens`, and
`prefer_requested_block_size = True` (so mlx-vlm's adaptive rule does not shrink n-gram blocks). Register it as kind
`mtp` through `serve/engine.py`. This reuses mlx-vlm's verify (`_mtp_verify_target`, batch-invariant kernels) and
rollback (`rollback_speculative_cache` for Qwen, cache transactions for GLM) unchanged. **Verify before coding:**
`_mtp_rounds` passes the trunk hidden state to the drafter and may require a fixed block per round. If variable-length
drafts are not supported, take the fallback.

**Fallback route.** A single-stream round loop in `serve/spec_loop.py` (`MLX_VLM_MAX_NUM_SEQS=1` is already the
default, `serve/config.py`):

```python
from mlx_vlm.speculative.common import verify_forward          # runs ≤ 8-token chunks in one cache transaction
out, txn = verify_forward(lm, mx.array([[last] + draft]), cache)
tgt = sampler(out.logits)                                       # target choice at each position
acc, new = _speculative_walk(mx.array([draft]), tgt, budget)    # common.py
txn.commit([acc + 1])                                           # keep accepted + bonus; rolls back the rest
```

`verify_forward` chunks at `DECODE_BLOCK_SIZE = 8` [src: `models/linear.py`]. Drafts > 8 tokens run as several
forwards inside one transaction, which costs more than a single pass. That is another reason to cap n-gram drafts
near 16 unless T(k) says otherwise.

**Hybrid drafter** (`serve/hybrid_draft.py`): n-gram first, MTP otherwise. The MTP head needs the trunk hidden state of
the last accepted position. After an n-gram round, take it from the verify output (`out.hidden_states`). mlx-vlm's
`_mtp_draft_hidden` does exactly this for MTP rounds.

**Memory**: < 10 MB. **Expected**: tool-argument segments 2–3×, file rewrites 4–6×, prose ≈ 1× [model]. Blended over an
agent turn, the gain depends on the mix of thinking tokens and tool tokens (measure in S0).

### 4.4 Step S4 — prefill chunk size and power (½ day)

```bash
for s in 2048 4096 8192; do
  uv run python -m evals.speed --model models/<name> --contexts 8192,32768,131072 --prefill-step-size $s --mtp-modes off
done
```

- Pick the largest step whose peak stays within the profile budget (§4.1 table). Expect 5–25 % from 2048 → 4096
  [self-reported trend]. GLM's 128K prefill peak is set by the projected recompute (8 GiB), not the step (decode §3).
- Benchmark in High Power mode (mac §5, Rec. 8). Prefill is compute-bound and gains from it; decode barely does.
- Record `prefill_tps` at 8K/32K/128K. These replace the planning numbers in §2.7–2.8.

### 4.5 Step S5 — fine-tune the MTP head on target-labelled personal data (3–5 days + overnight runs)

**Why the MTP head first, not EAGLE-3.** The vendor trained the MTP head at pre-training scale, and it already sits
near α ≈ 0.78. What it lacks is (a) adaptation to the pruned/requantised trunk and (b) *your* distribution (your
repos, your tools, your prompts). Both are small corrections, so train only a few parameters and keep the vendor's
weights as the starting point.

**What to train**:

- Qwen MTP: unfreeze `eh_proj`/`fc`, norms and the attention projections of the MTP layer (≈ 45–60 M params), plus
  LoRA rank 16 on the MTP experts' gate/up/down (16 × (2,560 + 640) × 3 ≈ 0.15 M per expert × 320 experts ≈ 49 M). Freeze the shared
  embeddings and `lm_head`.
- GLM MTP: the same recipe, but the expert LoRA is 4× larger per expert (H=4096, I=2048); train the non-expert part
  first.

**Objective (self-distillation).** For position t, the input is the trunk's final hidden h_t and emb(x_{t+1}). The
target is the *trunk's own* next-token distribution at t+2, p_trunk(· | x_≤t+1), not the text's next token. Use KL
at T=1 on the top-64 logits plus residual mass (the format `calib/` already writes for `ref/`). Add a 2–3 step
TTT-style unroll: feed the head's own draft back as x_{t+2} for the next step, so it learns depth 2–3 behaviour.

**Data: personal transcripts without decode-speed regeneration.** The user's Claude Code transcripts
(`~/.claude/projects/*/*.jsonl`) contain 1,566 files totalling 1.5 GB [measured: file count and size only, contents
not read]. They are Claude's outputs, which are the wrong labels for drafting: the drafter must imitate *the local
trunk*. The trick: use them only as **contexts**, and get the labels from the trunk by **teacher forcing at prefill
speed**:

```
for each turn context c (system + history, chat-templated in the TARGET's format by calib_data.render):
    run the trunk over c ⊕ (Claude's reply r) in prefill mode          → h_t, top-64 p_trunk at every position
    train MTP head: (h_t, emb(x_{t+1})) → p_trunk(· | x_≤t+1)          # positions inside r only
```

Prefill throughput makes this affordable: 20 M tokens / 1,500 tok/s ≈ **3.7 h** of trunk time [computed]. Regenerating
the same 20 M tokens at decode speed (≈ 45 tok/s) would take ≈ 123 h. Then add a small **on-policy** set to fix
exposure bias: ≈ 2 M tokens actually generated by the trunk on those contexts, overnight with MTP on (≈ 6–7 h at
≈ 80 tok/s). DFlash drafters were trained on ≈ 100 K target-regenerated samples [self-reported], so this scale is
reasonable.

**Privacy rules** (they also apply to §4.6 and §7): the transcripts never leave the machine. Run a secret scrubber
(API-key, token and private-key regexes; `.env` contents) before tokenisation. Store only token ids + top-k logits
under `artifacts/personal/` (git-ignored). Hold out the 10 % most recent sessions for evaluation.

**Module layout** (new package `drafter/`):

```
drafter/collect.py   # transcripts → contexts (calib_data.render) → trunk prefill → shards of (h_t, ids, top-k)
drafter/train_mtp.py # MLX training loop: freeze trunk, train MTP head params/LoRA, KL + TTT unroll
drafter/eval.py      # acceptance α per segment type on held-out traces, at T ∈ {0, 0.6}
```

**Memory** (all-in or calibrate profile, ch. 08 §4.1) [computed]: trunk resident ≈ 40 GiB (Qwen) for collection.
Training: MTP head ≈ 1.4 GiB frozen weights + ≈ 100–110 M trainable params × 16 B (fp32 master + grad + Adam m, v) ≈
1.9 GiB + activations ≈ 2–4 GiB. Storing h_t offline costs 2,560 × 2 B = 5 KB/token → 100 GB for 20 M tokens
(2.6 TB free). Or run online in one process: ≈ 47 GiB total.

**Expected**: α +2–8 points on personal traffic. +2–3 % decode [self-reported, Litwein sidecar] to ≈ +10–15 % if α
rises from 0.70 to 0.78 on a mismatched head [model: Appendix A, B4, 1.64× → 1.86×]. **Stop rule**: if held-out α
gains < 2 points, do not proceed to EAGLE-3.

### 4.6 Step S6 — EAGLE-3 on Qwen, only if S5 shows headroom (1–2 weeks, optional)

Rationale: EAGLE-3 fuses three trunk depths and uses a 32K draft vocabulary, which makes drafting much cheaper than
MTP's full `lm_head` (§2.4). It is supported for Qwen by mlx-vlm at inference (`--draft-kind eagle3`, speculators
format [src: `drafters/eagle3/config.py`]). Training is ours to write. mlx-vlm's trainer has no drafter recipes
[src: `trainer/` contents].

- **Drafter size** for H=2,560 [computed]: `fc` 3H→H 19.7 M + one decoder layer with 2H input ≈ 60–110 M + 32K-row
  draft head 84 M ≈ **0.2 B params**, 0.4 GiB bf16. Adam state ≈ 3 GiB.
- **Data**: the same `drafter/collect.py` shards, plus hidden states at three layers (e.g. `[2, 24, 45]`)
  = 15.4 KB/token → 20 M tokens = 308 GB offline, or online with the trunk resident.
- **Compute** [computed, assumption-laden]: 6 × 0.2 B × 20 M tokens × 3 epochs × 5 TTT steps ≈ 3.6 × 10¹⁷ FLOP. At an
  effective 15 TFLOP/s (≈ 25–30 % of the measured bf16 GEMM ceiling) ≈ **6.7 h**, plus collection.
- **Draft vocabulary**: the 32K most frequent target tokens in your traffic (`d2t`/`t2d` maps). Code-heavy traffic
  concentrates mass, so measure coverage (target ≥ 97 % of emitted tokens).
- **Decision**: adopt only if held-out `evals.specbench` shows ≥ 10 % higher decode tok/s than tuned MTP at the
  serving temperature. The MoE verify cost (§2.3) caps the upside near 2–2.3× on Qwen regardless of drafter quality.

GLM: blocked until `glm5_next` exposes per-layer hidden states and `rollback_speculative_cache` (decode §1). That is
the stretch item in §7.3.

### 4.7 Step S7 — long-context decode hygiene (½ day)

- Keep APC checkpoints bounded (≤ 2 at 128K) and free the GLM projected cache (decode Rec. 1).
- Measure decode at 1K / 32K / 128K. Expect −15 % at ~110K for Qwen (MTPLX 73.5 → 61.8 tok/s [self-reported]). If
  the drop is larger, look at QSA indexer work: the `index_block_keys` concatenate copies the whole block-key array
  each time a 4-token block completes [src: `qwen4_exp/language.py` indexer], which is O(N) per 4 tokens. A
  preallocated buffer would remove it (§4.8).
- Choose `CLAUDE_CODE_MAX_CONTEXT_TOKENS` from the measured decode-vs-context curve: compact where decode falls below
  ~80 % of the short-context speed, not at the model's maximum.

### 4.8 Step S8 — kernels, ranked by expected value (only after S0–S5)

Profile first: run a decode step under Instruments' Metal System Trace (or `MTL_CAPTURE_ENABLED=1` with
`mx.metal.start_capture`) on the real model and count dispatches per layer. Then:

| # | Idea | Why it should help | Expected | Effort | Kill criterion |
|---|---|---|---|---|---|
| K1 | `mx.compile` larger shape-static decode sub-blocks: HC mix → norm → q/k/v/z projections; router softmax → top-k → renormalise | Merges element-wise kernels, fewer dispatches; `t0` is 60 % of a Qwen step | 3–8 % decode | Low | < 2 % in `evals.speed` |
| K2 | Fused decode MoE kernel via `mx.fast.metal_kernel`: for K experts, gate+up GEMV → SiLU·mul → down GEMV → weighted sum in one dispatch, affine 3/4-bit | Replaces 3 `gather_qmm` + 2 element-wise + combine (≈ 6 dispatches × 48 layers); keeps the intermediate in threadgroup memory | 5–10 % decode [model: 6 × 48 × ≈ 5 µs ≈ 1.4 ms of 21.5 ms] | High | Must match `SwitchGLU` within bf16 tolerance; drop if < 4 % |
| K3 | Preallocated QSA block-key buffer (no per-block concatenate) | O(N) copy every 4 tokens at long context | 1–5 % at 128K | Low | Gain < 1 % at 128K |
| K4 | Fused gather + reduce for prefill (mlx-serve's 3.4× component win) | Removes the scatter/unsort round-trip | 3–10 % prefill | Medium | Whole-prefill gain < 3 % |
| K5 | HC up-projection + mix fusion (prefill) | 1.4–1.6× component [self-reported] | 2–5 % prefill | Medium | Same |
| K6 | NAX tensor-op verify GEMM for 4–16 rows | mlx#4682 (split-K NAX for M ≥ 8) does not cover gather or fp modes (quant §4) | unknown | Research | Watch upstream first |

Do **not** do: dequantised dense experts (2.2× slower, measured by mlx-serve), gate/up bank fusion for prefill
(−12 %), an "unsorted path" fix (no gap exists). All [self-reported].

How `mx.fast.metal_kernel` works, for K2: you write the Metal body, and MLX generates the signature from
`input_names`/`output_names`, template parameters (e.g. `BITS`, `GROUP`, `K`) and the grid/threadgroup sizes you pass
at call time. The kernel is JIT-compiled once per template instantiation. `mlx_vlm/models/exact_speculative_verify.py`
is a working local example of a quantized GEMV written this way (simdgroup reduction over 32 lanes, 4×4 tiles per
thread) [src]. Start by copying it.

### 4.9 Summary of expected throughput (Qwen-320 plan, work profile)

| Configuration | Decode tok/s, short ctx | Basis |
|---|---|---|
| Plain | ≈ 42 | [model], anchored on 44–48 measured for the 4-bit unpruned model |
| + MTP block 4 | ≈ 70–80 | [model] 1.7–1.9×; 73.5 [self-reported, MTPLX] |
| + n-gram on tool args / rewrites (segment-local) | ≈ 120 / 220 effective on those segments | [model] 2.8× / 5.3× of plain |
| + MTP fine-tune | +5–15 % on MTP segments | [model]/[self-reported] |
| + K1–K3 | +5–15 % | [model] |

TTFT (warm agent turn, +3K tokens): ≈ 2–3 s. Cold first turn: ≈ 20 s. A 128K document cold: ≈ 90 s [computed at
1,500 tok/s].

---

## 5. Decision criteria & what to measure

| Decision | Measure (suite) | Rule |
|---|---|---|
| MTP block size per model | `evals.specbench` sweep 2–6 at the serving temperature | Pick the argmax of decode tok/s; ties → smaller block (less rollback state) |
| MTP head needs repair | α on agent traces | α < 0.65 (Qwen) / < 0.55 (GLM) → S5 before anything else |
| n-gram drafter on/off | tok/s per segment type, plain vs hybrid | Keep if blended agent-turn decode ≥ +15 % and no segment type regresses > 3 % |
| n-gram max length | measured T(k) + online E_n | Maximise E_n[acc]/T(k); hard cap 32 |
| Personal fine-tune worth it | held-out α | ≥ +2 points → keep; < 2 → stop, skip EAGLE-3 |
| EAGLE-3 adoption | held-out tok/s vs tuned MTP | ≥ +10 % at serving temperature |
| Prefill step | `evals.speed` prefill tok/s + peak | Largest step within the profile budget with ≥ 5 % gain |
| APC healthy | divergence position, hit tokens, restore s | ≥ 99 % prefix reuse per turn; restore ≤ 2 s |
| Kernel item | `evals.speed` decode/prefill, numerics vs reference | Keep only if ≥ the kill criterion in §4.8 and max-abs logit diff within bf16 noise |
| Qwen vs GLM, for speed | All of the above | GLM is ≈ 3× slower to decode and ≈ 3–5× slower to prefill. Choose GLM only for work where its quality gap matters and latency does not (ch. 08 router) |

Always record: energy mode and `pmset -g therm` (mac Rec. 8); MLX/mlx-vlm versions; temperature; whether APC was
warm; context length. Repeat each speed point ≥ 3 times and report the median. The machine is shared, so contention
only ever slows a run down (quant §4).

---

## 6. Failure modes

| Failure | Symptom | Cause | Mitigation |
|---|---|---|---|
| Speculation slower than plain | tok/s(MTP) < tok/s(plain) | Low α at T > 0 (acceptance = p_target(d)); blocks too long for MoE verify cost; GLM α ≈ 0.6 | Sweep at the real temperature; shrink the block; turn off per request class |
| Silent drafter/trunk mismatch | α falls after a rebuild | Pruned/requantised trunk, stale MTP head; `convert --mtp` drops `q_mode` (issue #2526, decode §1) | Gate every build on α; quantise the drafter in the trunk's mode |
| bf16 non-bit-exactness | MTP and plain outputs diverge after N tokens | Verify kernels' numerics depend on batch shape (decode §1) | Expected; judge by quality evals, not diffs |
| APC never hits | TTFT grows with history | A volatile prefix (timestamp, attribution header, tool reordering) | Divergence probe (§4.1); fix the client |
| APC memory blow-up | Process exceeds the cap at long context | Each checkpoint is a full cache copy; GLM projected cache ≈ 25 GiB per checkpoint | Fix GLM (decode Rec. 1); cap entries; admission control (ch. 08) |
| Stale SSD checkpoint | Garbage after a rebuild at the same path | Fingerprint collision | `model_fingerprint` hashes config + file sizes + mtimes [src]; never edit weights in place |
| n-gram drafts poison quality | — | Impossible by construction: the target verifies every token | — |
| n-gram drafts waste time | No gain on prose | Matches on short common n-grams (`\n    `) | m_min ≥ 3, length-aware E_n, cap drafts at 8 when E_n is low |
| Thinking budget ignored | Runaway thinking with MTP on | mlx-vlm cannot enforce budgets while drafting (`serve/engine.py`) | Router-level `max_tokens`; disable drafting for budgeted requests |
| Drafter training leaks secrets | — | Transcripts contain tokens, keys and `.env` contents | Scrubber, local-only artifacts, git-ignored dirs |
| Drafter learns Claude, not the trunk | High offline accuracy, low live α | Trained on Claude's tokens instead of the trunk's distribution | Labels = trunk top-k (teacher forcing) + on-policy subset |
| Custom kernel wrong only for some shapes | Rare NaNs/garbage | Template edge cases (odd group sizes, 3/5/6-bit packing) | Exhaustive tests against `SwitchGLU` over every (bits, group) in the build |
| Benchmark noise | 2× run-to-run spread | Shared GPU, downloads, thermals | Best-of / median of ≥ 3; idle machine for decisions; record thermals |
| IOGPU panic under churn | Kernel panic | Wiring + concurrency + cache churn (mac §3) | Wiring off, one sequence (mac Rec. 2–3); APC churn is a trigger, so cap evictions |

---

## 7. Stretch ideas (push limits)

### 7.1 Cost-aware hybrid speculation with an expert-overlap term

Combine CaDDTree's objective with EcoSpec's expert pricing in the chain setting. Each round, for candidate draft
length k (from MTP, n-gram, or both), estimate

```
gain(k) = E[accepted | source, match length, recent α] + 1
cost(k) = t0 + (trunk + routed · û(k)/K + state·(1+k)) / BW_eff        û(k) from the router's actual top-k of the
                                                                         MTP draft path (the MTP layer routes too)
```

and choose the k that maximises gain/cost. The MTP layer is itself an MoE layer, so its routing for each draft token
is a free predictor of which experts verification will touch. Prefer drafts that reuse experts. Expected: +5–15 % over
fixed-block MTP [model; EcoSpec's +11–15 % is the closest evidence].

### 7.2 Expert-locality-aware REAP

Add a *co-routing* term to the pruning objective (reap §Rec. 9 already proposes accumulating F_ij). Experts that
co-fire on consecutive tokens make verification cheaper (§2.3). Measure û(4) and û(8) on agent traces before and
after; a 20 % lower û(8) is worth ≈ 0.1× of MTP speedup [model].

### 7.3 Unblock EAGLE-3/DFlash for GLM

Implement `capture_layer_ids` and `rollback_speculative_cache` in `glm5_next` (a shim under `serve/` or an upstream
PR). The rollback has to cover KDA recurrent states (keep k intermediate states, as Qwen does), the DSA latent
`KVCache`, the indexer cache and the (now freed) projected cache. Then evaluate `modal-labs/GLM-5.3-Flash-DFlash` or
`incoai/GLM-5.3-Flash-DFlash2`. Expected upside is small (≤ MTP on this MoE, §2.5) unless combined with 7.1. This
is mainly a learning project in hybrid-cache rollback.

### 7.4 Branch-per-batch tree verification for hybrid models

A real ancestor-masked tree does not work through recurrent layers (§2.5). A tree of b root branches can instead run
as a batch of b sequences that share the attention KV prefix. The recurrent state is small enough to copy:
110 MiB × b for Qwen. The shared-prefix KV needs a batch cache whose rows reference one prefix plus per-row tails.
`BatchKVCache` cannot do this today. With b = 2–3 top-1/top-2 MTP branches, DDTree's analysis predicts +10–20 %
expected acceptance [self-reported trend], but the verify union grows with b. Only worth it if 7.1's cost model says
so.

### 7.5 Speculative prefill and "thinking while you type"

Prefill is compute-bound and the GPU idles between agent turns. When a tool result arrives in the client but before
the request is sent, nothing can be prefilled, since the server does not see it. However, the server can (a)
pre-restore the session checkpoint (ch. 08 §7.5) and (b) for RAG, prefill retrieved documents into APC checkpoints
keyed by document hash, so that a later question restores them. Document prefixes must come first in the prompt for
this to hit.

### 7.6 Self-speculation with a low-bit copy of the trunk's experts

The PLE dividend (≈ 25 GiB spare, ch. 08 §4.1) could hold a 2-bit copy of the trunk's experts (Qwen: 120.8 B params
× 2.5/8 ≈ 38 GB unpruned, too big; K=320 at 2.5 bpw ≈ 23.6 GB, borderline). Drafting with the same model at lower
precision gives very high α. It costs ≈ 50–60 % of a trunk step per draft token, though, because `t0` dominates
Qwen. The cost model says this loses to MTP on Qwen [model]. It might win on a dense 27B, where bytes dominate. This is
here as a worked example of why `t0` changes the answer.

### 7.7 Personal n-gram memory across sessions

Persist the SuffixDecoding output index per repository (paths, identifiers, idioms, test names) under
`~/.cache/local-ai/ngram/<repo-hash>`. Seed each new session's drafter with it. Cost: a few MB per repo. This
extends SuffixDecoding's "previous outputs" tree from one session to your whole history, and is the cheapest form of
personalisation in this chapter.

### 7.8 Grammar-aware verification

Ch. 08 §7.4 already proposes masking target logits with the tool-call grammar during verification. With n-gram
drafts inside `<parameter=…>` blocks, a grammar also tells you where an argument ends. Stop drafts at `</parameter>`
so that one round never straddles two parameters.

---

## Appendix A — cost model

The script is in the session scratchpad (`spec/costmodel.py`); here is its core so it can be re-run anywhere.

```python
def uniq(E, K, k, corr=0.75):                  # unique experts per layer for k verified positions
    ind = E * (1 - (1 - K / E) ** k)           # independent routing
    return K + corr * (ind - K)                # correlated routing (Cohere: 20–31 % fewer than independent)

def T(m, k):                                   # ms to verify k positions
    u = uniq(m["E"], m["K"], k) if k > 1 else m["K"]
    gb = m["N"] + m["R"] * u / m["K"] + m["S"] * (1 + k)
    return m["t0"] + 0.3 * (k - 1) + gb / m["bw"] * 1000

def mtp_speedup(m, alpha, block):              # chain MTP, block = drafts + 1
    g = block - 1
    E = (1 - alpha ** (g + 1)) / (1 - alpha)
    return E * T(m, 1) / (T(m, block) + g * m["d"])

QWEN288 = dict(N=2.462, R=1.327, E=288, K=10, S=0.113, t0=12.6, bw=450, d=1.3)   # GB, ms, GB/s
QWEN320 = dict(N=3.829, R=0.973, E=320, K=10, S=0.113, t0=12.6, bw=450, d=1.6)
GLM144  = dict(N=6.95,  R=3.17,  E=144, K=8,  S=0.142, t0=11.8, bw=420, d=6.0)
```

Parameters: N = trunk bytes per token, R = routed bytes per token, S = recurrent state per copy (36 × 3.15 MB;
34 × 4.19 MB), t0 fitted to 21.5 ms measured for Qwen (§2.2) and scaled by layer count for GLM, BW_eff from quant §4,
d = draft cost per MTP token (§2.4; GLM's 6 ms is conservative). Output [model]:

| | T(1) | ceiling @614 | MTP α=0.7, B3/B4/B5 | MTP α=0.8, B3/B4/B5 |
|---|---|---|---|---|
| Qwen-288 4-bit | 21.5 ms (46 tok/s) | 153 | 1.60 / 1.64 / 1.61 | 1.78 / 1.91 / 1.95 |
| Qwen-320 plan | 23.8 ms (42 tok/s) | 122 | 1.67 / 1.73 / 1.72 | 1.86 / 2.02 / 2.08 |
| GLM-144 plan | 36.6 ms (27 tok/s) | 59 | 1.33 / 1.29 / 1.22 | 1.48 / 1.51 / 1.48 |

The model's two free parameters (t0, the correlation factor) are exactly what S0's T(k) curve measures. Replace them,
then re-run.

## Appendix B — n-gram acceptance simulation

`spec/pld_sim.py` (session scratchpad), run with `.venv/bin/python -I pld_sim.py <qwen tokenizer.json> <dir of .py
files>`. It uses the real Qwen3.8 tokenizer, with no model. For 40 real Python files (mlx_vlm sources > 3 KB, first
20 KB each):

- **Edit scenario**: the context is a `<tool_response>` containing the file (optionally `cat -n` numbered). The output
  is an `Edit` call whose `old_string` is a 5–25-line slice and whose `new_string` inserts a guard line and appends a
  comment. Rendered either as JSON (`json.dumps`) or in Qwen's XML parameter format. 14.7K output tokens.
- **Rewrite scenario**: the output is the whole file with 3 small changes. 104.6K output tokens.
- **Drafter**: the most recent earlier occurrence of the longest trailing n-gram (n = 4…1) over context + emitted
  tokens. Propose up to k tokens. Accept the matching prefix plus one bonus token (greedy oracle target).

| Scenario | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|
| Edit, JSON, numbered context | 2.44 | 2.86 | 3.18 | 3.52 |
| Edit, XML, numbered context | 2.68 | 3.26 | 3.73 | — |
| Edit, XML, plain (de-numbered index) | 3.16 | 4.23 | 5.35 | — |
| Whole-file rewrite | 4.09 | 6.75 | 11.68 | 20.12 |

(tokens emitted per verify round). Limitations: the oracle target emits exactly the reference output; a real model
writes some genuinely new code (lower acceptance) but also re-emits paths and identifiers outside these spans (higher).
The matcher is the simplest possible; SuffixDecoding's frequency-scored suffix trees do better [self-reported 1.7×
over PLD on SWE-Bench].

---

## Sources

- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Li et al., *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test*, NeurIPS 2025 — [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html)
- *DFlash: Block Diffusion for Flash Speculative Decoding* — [arXiv:2602.06036](https://arxiv.org/pdf/2602.06036), [z-lab/dflash](https://github.com/z-lab/dflash)
- Ringel & Romano, *Accelerating Speculative Decoding with Block Diffusion Draft Trees* (DDTree) — [arXiv:2604.12989](https://arxiv.org/abs/2604.12989), [code](https://github.com/liranringel/ddtree)
- *Cost-Aware Diffusion Draft Trees for Speculative Decoding* (CaDDTree) — [arXiv:2606.01813](https://arxiv.org/html/2606.01813)
- *The Limits of Speculation: Bounding Speculative Decoding in Mixture-of-Experts* — [arXiv:2609.22156](https://arxiv.org/html/2609.22156)
- *Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for MoE* (EcoSpec) — [arXiv:2607.12696](https://arxiv.org/html/2607.12696v1)
- *Utility-Driven Speculative Decoding for Mixture-of-Experts* — [arXiv:2506.20675](https://arxiv.org/pdf/2506.20675)
- Cohere, *Why MoE models get more from speculative decoding* — [blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- Oliaro et al., *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications*, NeurIPS 2025 — [arXiv:2411.04975](https://arxiv.org/abs/2411.04975v2), [ArcticInference](https://github.com/snowflakedb/ArcticInference)
- *Speculative Decoding: Performance or Illusion?* (n-gram on code editing) — [arXiv:2601.11580](https://arxiv.org/pdf/2601.11580)
- *BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators* — [arXiv:2607.19438](https://arxiv.org/html/2607.19438v1)
- mlx-serve issue #366, *Prefill … where the time actually goes on Qwen3.8-Flash-Next (M5 Max, measured)* — [GitHub](https://github.com/ddalcu/mlx-serve/issues/366)
- MTPLX Qwen3.8-Flash-Next pack — [model card](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed)
- oMLX GLM-5.3-Flash oQ2e on M5 Max — [benchmark](https://omlx.ai/benchmarks/performance/88m4imqs); grant-ai GLM-5.3-Flash 4-bit — [card](https://huggingface.co/grant-ai/GLM-5.3-Flash-Abliterated-MLX-4bit)
- dflash-mlx M5 Max table (secondary) — [explainx](https://explainx.ai/blog/dflash-2-mlx-m5-max-local-inference-august-2026); repo [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx)
- DFlash training data scale — [z-lab Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash), [NVIDIA Model-Optimizer dflash.md](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/doc/dflash.md)
- Claude Code fixed-prompt size (third-party) — [morphllm](https://www.morphllm.com/claude-code-token-limit), [DEV](https://dev.to/ashraf_chowdury09/claude-code-burns-33k-tokens-before-it-even-reads-your-prompt-45p2)
- MLX custom Metal kernels — [docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- Installed sources: `mlx_vlm/speculative/{common,mtp,eagle3,ddtree,cache_state}.py`, `mlx_vlm/speculative/drafters/{__init__,compatibility}.py`,
  `mlx_vlm/models/{switch_layers,gated_delta,quantized_verifier,exact_speculative_verify}.py`, `mlx_vlm/models/qwen4_exp/language.py`,
  `mlx_vlm/server/compaction.py`; repo `serve/{config,engine}.py`, `evals/speed.py`, `scripts/tune_runtime.py`.
