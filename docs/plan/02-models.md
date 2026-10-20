# 02 — Model selection strategy and the "bigger than RAM" options

Chapter of the local-ai master plan. Written 2026-10-20 for the MacBook Pro M5 Max (40-core GPU, 128 GB, 614 GB/s,
4 TB SSD with ~2.5 TB free) running mlx 0.32.4 / mlx-lm 0.32.2 / mlx-vlm 0.7.9.

It builds on, and does not contradict, the verified findings in `PLAN.md`, `docs/research/quant.md`,
`docs/research/reap.md`, `docs/research/decode.md` and `docs/research/mac.md`. Where this chapter changes a
recommendation from those files, it says so and names the new evidence.

Legend:
- **[primary]**: vendor model card or official leaderboard page, read directly.
- **[3rd-party]**: an evaluator other than the model's vendor.
- **[self-reported]**: a number from a model card, PR or blog that nobody here reproduced.
- **[measured]**: run on this machine for this chapter.
- **[computed]**: arithmetic from verified parameter counts. The working is shown so it can be re-checked.
- **[estimate]**: a reasoned guess, with the assumptions stated.

---

## 0. The decision in one page

| Role | Model and build | Profile | Resident memory | Why |
|---|---|---|---|---|
| **Primary (always on)** | **Qwen3.8-Flash-Next**, own build. Unpruned or lightly pruned (K ≥ 448 of 512 experts), ~3.7 bpw routed experts with a sensitivity-driven mix, the 51.2B n-gram table **memory-mapped** (not resident), MTP head kept | P1 co-tenant | **≈ 62 GB** | Same top-tier base quality as GLM (AA index 40 vs 42), and it fits with *no or light pruning*. Pruning, not quantization, is the main source of damage. |
| Primary, lean variant | Qwen3.8-Flash-Next REAP-320, ~3.3 bpw (what PLAN.md targets today) | P1 when helpers need the RAM | ≈ 37–39 GB | Frees ~25 GB for helpers or long-context KV, at the cost of 37.5 % pruning |
| **Secondary (escalation / second opinion)** | **GLM-5.3-Flash**, either REAP-176…192 at ~3.2–3.4 bpw held resident (≈ 85–91 GB) or **unpruned and SSD-streamed** with an expert cache | P2 all-in, P3 stream. **Never in P1** | 85–91 GB, or 55–90 GB of expert cache plus ~9 GB trunk | Highest independent index score in the class, and its strengths differ from Qwen's (lower hallucination, τ²-bench, GraphWalks). The ≤ 68 GB build would have to lose 50 % of its experts **and** drop to ~3 bpw, which removes the knowledge edge that justifies running it. |
| RAG retrieval | `google/embeddinggemma-2`, text tower only (270M) | all | ≈ 0.6 GB | Already cached. MTEB code 78.7, 8K context, Matryoshka 768→256 |
| RAG rerank | v1: the primary model as a listwise reranker over prefix-cached candidates. v2: `nvidia/Nemotron-3.5-Rerank-8B` ported to MLX | all | 0 GB / ≈ 4.7 GB (4-bit) | The Nemotron reranker has no MLX runtime and is aimed at multimodal (ViDoRe) retrieval, so prove it beats the free option first |
| "Fast lane" | **The same primary weights with thinking off / `reasoning_effort=low`**, plus batching | all | +KV only | On this machine a separate small model is not meaningfully faster (see §2.2). Qwen3.8-27B reads more bytes per token than the 180B MoE. |
| Not recommended | DeepSeek-V4-Flash, Nemotron-3.5-Super-VL, Mistral-Medium-3.5-128B, Qwen3.8-27B as primary | — | — | Each is dominated on quality-per-GB or on speed (§4.2) |

**The headline change from PLAN.md:** the ≤ 70 GB *disk* budget is the wrong constraint for Qwen. The n-gram table
(29–32 GB) is streamed by mmap (`ple_storage`), so it lives in reclaimable page cache rather than the process
footprint. What we must bound is **resident** bytes. Under the same 80 GB process cap, that buys roughly
**+23 GB of expert weights**: the difference between 37.5 % pruning and none. Concretely, `build.make
--budget-gb 94` (which counts the 32 GB affine-g32 table) produces about 62 GB resident. §4.4 gives the commands and
§5 the A/B that confirms or refutes this.

---

## 1. Why it matters (intuition)

### 1.1 Memory is the currency, and there are three ways to spend it

On a 128 GB unified-memory Mac, every model decision comes down to how you spend bytes. There are three knobs:

1. **How many parameters you keep.** Pruning removes whole experts (REAP). Removed experts are functions the model no
   longer has.
2. **How many bits each kept parameter gets.** Quantization keeps every function but blurs each one.
3. **Where the bytes live.** Resident in GPU-visible RAM, in the OS page cache (mmap, reclaimable), or on SSD,
   fetched on demand (expert streaming).

A good mental model is a library that has to fit in a small apartment. You can throw away books (prune), print them
in smaller type (quantize), or keep some in a storage unit across town and fetch them when needed (stream). Small type
costs you a little on every page. Throwing a book away costs everything when you need that book. The storage unit
costs a trip, and that is only cheap if you rarely need what is stored there.

### 1.2 Why a 180B MoE can be *faster* than a 27B dense model here

Generating one token (decode, batch 1) means reading every weight the token touches from memory once. At 614 GB/s, the
upper bound on decode speed is:

$$\text{tok/s} \le \frac{\text{bandwidth}}{\text{bytes read per token}}$$

A dense model reads **all** its weights for every token. A Mixture-of-Experts (MoE) model reads only the router-selected
experts plus the shared "trunk". MoE therefore decouples *capacity* (total bytes, which set whether it fits) from
*speed* (active bytes, which set tokens per second). In numbers **[computed]**:

| Model | Bytes read per token (4–4.5 bpw) | Ceiling at 614 GB/s |
|---|---|---|
| Qwen3.8-27B dense, 4-bit g64 | 27e9 × 4.5 / 8 = **15.2 GB** | ≤ 40 tok/s |
| Qwen3.8-Flash-Next, 180B total / 6B active | ~3.8–4.1 GB (routed experts ~1.0 GB + non-expert trunk) | ≤ ~150 tok/s, but in practice latency-bound at ~44 tok/s plain and ~74 tok/s with MTP **[self-reported, MTPLX on M5 Max]** |
| GLM-5.3-Flash, 321B / 18B active | routed 8 × 25.2M × 42 layers × 3.5/8 = 3.7 GB, plus ~6–6.5 GB trunk ≈ **8–10 GB** | ≤ 60–75 tok/s |
| Mistral-Medium-3.5, 128B dense, 4-bit | **72 GB** | ≤ 8.5 tok/s |

This is why "a small fast model plus a big slow model", the usual cloud pattern, mostly does not apply here. The big
MoE *is* the fast model. A small model only pays off if it has ≤ ~3B active parameters (35B-A3B class), and then its
quality is far lower (AA index 18 vs 40).

### 1.3 Base quality is close; compression damage is not

Among the models that can fit, GLM-5.3-Flash and Qwen3.8-Flash-Next are within ~2 points on the independent
Artificial Analysis index (42 vs 40). Under NVIDIA's single-harness cross-evaluation they split the coding and agentic
benchmarks between them (§3.1). Fitting them, however, costs very different amounts:

- Qwen at ≈ 62 GB resident: **0–12 % of experts pruned**, ~3.7 bpw.
- GLM at ≤ 68 GB: **50 % of experts pruned** *and* ~3.0 bpw, past the point where each step alone has published,
  measurable damage (§3.3).

The ranking under a memory budget is decided by *base quality minus compression penalty*, and the penalty gap is
several times larger than the base gap. That is the main argument of this chapter.

---

## 2. How it works (mechanics and key math)

### 2.1 Memory accounting

**Affine quantization storage.** MLX affine stores `b`-bit codes plus a bf16 scale and bias per group of `g` weights:

$$\text{bpw} = b + \frac{2\cdot 16}{g} \quad\Rightarrow\quad 3\text{b g64} = 3.5,\;\; 4\text{b g64} = 4.5,\;\; 4\text{b g32} = 5.0$$

**Weight bytes** of a tensor = `params × bpw / 8`. For a pruned MoE with `K` kept experts per layer:

$$W(K,\overline{\text{bpw}}) = K\cdot P_\text{slot}\cdot\frac{\overline{\text{bpw}}_\text{exp}}{8} + W_\text{trunk} + W_\text{MTP} \;(+\,W_\text{PLE}\text{ on disk only})$$

These are the verified parameter counts from the safetensors headers (`reap.md` §Verification):

| | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|
| Params per expert | 3·2560·640 = 4.92 M | 3·4096·2048 = 25.17 M |
| `P_slot` (one expert in every layer) | 0.236 B (48 layers) | 1.057 B (42 MoE layers) |
| Routed experts total | 120.8 B (512 per layer) | 304.4 B (288 per layer) |
| Non-expert trunk | 4.95 B (≈ 4.6 GB with floors) | 8.92 B (≈ 7.0 GB at ~6.25 bpw) |
| MTP | 2.52 B experts (≈ 1.4 GB at 4-bit) | 7.25 B experts + 0.19 B (≈ 1.5–2.4 GB at 2–3 bpw) |
| N-gram / PLE table | **51.2 B**, mmap-streamable (28.8 GB nvfp4 g16 / 32.0 GB affine g32) | none |

**Resident vs disk.** `mx.load` copies safetensors into Metal buffers, so they become anonymous dirty memory
(`mac.md` §2). The Qwen PLE table is the exception: `mlx_vlm/models/qwen4_exp/ple_storage.py` numpy-memmaps it and
materializes only the rows the current token looks up. Its pages are *clean, file-backed* page cache, which macOS
drops first under pressure. sh0wie's REAP-288 goes from 68 GB on disk to 39 GB resident with identical logits
**[self-reported, re-checked in reap.md]**.

**Process budget** (P1 co-tenant, 80 GB cap) **[computed]**:

| Item | GB | Source |
|---|---|---|
| KV + linear state, Qwen @128K with MTP | 4.2 (3.87 GiB) | `decode.md` §3 [measured on real-dim tiny models] |
| Prefill activations at `prefill_step_size` 2048 | ~3 | `decode.md` §4 (tiny models) plus MoE activations [estimate] |
| APC prefix-cache RAM tier | 4 | `mac.md` rec. 3 (≤ 4 GB RAM, rest on disk) |
| MLX buffer cache (`set_cache_limit`) | 2 | `mac.md` rec. 1 |
| **Left for resident weights** | 80 − 13.2 ≈ **66.8** | target **62–64** for margin |

### 2.2 Decode speed model

A decode step is bound by whichever is larger:

$$t_\text{token} \approx \max\Big(\frac{B_\text{read}}{BW_\text{eff}},\;\; L\cdot t_\text{launch}\Big),\qquad BW_\text{eff}\approx 0.5\text{–}0.8\times 614\ \text{GB/s}$$

- **GLM's experts are large** (2048×4096). Decode is bandwidth-bound. The repo microbenchmark measured 4-bit g64
  experts at ~485 GB/s effective, while 3-bit and 2-bit unpack less efficiently (~310–390 GB/s) **[measured,
  quant.md §4]**. Going below 4 bits therefore saves memory but not time.
- **Qwen's experts are tiny** (640×2560). Decode is **latency-bound** at ~80–130 µs per layer whatever the bit width
  **[measured, quant.md §4]**. Two consequences:
  1. Expert bits barely change Qwen's speed, so you can afford more bits for free.
  2. Batching 2–4 sequences adds little time per step, because the GPU is waiting on launches, not bytes. That is why
     subagents should be *batched onto the primary* (§4.5), not sent to a second model.

**Speculative decoding (MTP)** turns a latency-bound step into several tokens. With per-token draft acceptance `a`
and `k` drafted tokens, the expected tokens per verify step (assuming independent acceptance) is:

$$\mathbb E[\text{tokens/step}] = \frac{1-a^{k+1}}{1-a}$$

For GLM's default `k = 2` and `a ≈ 0.7` **[self-reported, lapadatbogdan]**: (1 − 0.343)/0.3 = 2.19 tokens per step.
The real speedup is lower, because a verify step costs more than a plain step. For Qwen, MTPLX reports 73.5 vs
43.8 tok/s (1.7×) at depth 3 **[self-reported]**.

**Time-to-answer, not tok/s, is the user-visible metric.** Artificial Analysis measured the tokens each model emits per
index task **[3rd-party]**: Qwen 108k output tokens (72k reasoning), GLM 69k (47k reasoning). Locally **[estimate]**:

- Qwen with MTP: 108k / ~74 tok/s ≈ **24 min**.
- GLM with MTP: 69k / (~30 × ~1.4) tok/s ≈ **27 min**.

Qwen's verbosity cancels most of its speed advantage. Measure both with `evals.speed` *and* the token counts from
`evals.code` / `evals.math`.

### 2.3 Two kinds of compression error: blur vs amputation

**Quantization is blur.** A uniform `b`-bit quantizer over a group with range `R` has step Δ = R/(2ᵇ − 1). With error
uniform on [−Δ/2, Δ/2]:

$$\mathbb E[\varepsilon^2] = \frac{\Delta^2}{12}\quad\Rightarrow\quad \text{each extra bit cuts error energy by }4\times\ (\approx 6\ \text{dB/bit})$$

A projection's output error is a sum of many small independent errors, so it behaves like noise. The network was
trained with noise-like perturbations (dropout, stochastic optimization), and it tolerates them until the noise
becomes comparable to the signal. That point is the well-known **cliff around 3 bpw**:

- GLM MLX builds at base 3g32 / 2g32: KLD **0.042 → 0.165** vs FP8 **[self-reported, orcarouter; quant.md]**.
- Qwen GSQ-RCO: task average 99.4 % of bf16 at 3.0 bpw, 95.7 % at 2.5 bpw **[3rd-party, IST-DASLab]**.

Sensitivity-driven allocation (the `calib/` + `build/` pipeline) moves bits from robust tensors to fragile ones.
The non-expert trunk is about 20× more sensitive per parameter than routed experts (`quant.md` §3).

**Pruning is amputation.** REAP removes the experts with the lowest

$$S_j=\frac{1}{|\mathcal X_j|}\sum_{x\in\mathcal X_j} g_j(x)\,\lVert f_j(x)\rVert_2 ,$$

the average gated contribution of expert `j` *on the tokens routed to it*. The experts that remain are untouched, but
any input that needed a removed expert gets a substitute that was never trained for it. Three things follow:

- The error is **not** noise-like. It concentrates on the inputs that used the removed experts: rare names, other
  languages, niche APIs.
- Averages hide it. sh0wie's rare-name probe collapsed at K=256 (1/10) while greedy HumanEval barely moved
  **[self-reported, reap.md §3]**.
- Knowledge-heavy evaluations fall first: Cerebras GLM-4.6 at 40 % pruning lost GPQA 78.8 → 69.7; Kimi-Linear lost
  FRAMES 55.7 → 52.3 while reasoning held **[self-reported, primary vendor]**.

**When both are applied** the errors roughly add. To first order, KL is a quadratic form in the logit perturbation,
so independent perturbations add their KL plus a cross term that is usually positive, i.e. worse:

$$\mathrm{KL}_\text{total} \approx \mathrm{KL}_\text{prune} + \mathrm{KL}_\text{quant} + 2\,\langle \delta_\text{prune},\delta_\text{quant}\rangle_F$$

Here F is the Fisher metric. The last term is why quantizing an already-pruned model hurts more than quantizing the
original: the surviving experts are now doing extra work and are more sensitive. The pruned-then-quantized build
should be calibrated *after* pruning, which is what `build/` does: it reads `sens` with the post-prune routing.

### 2.4 "Bigger at lower bits" vs "smaller at higher bits"

What the literature says, and how it applies to us:

1. **Dense LLMs at a fixed total bit budget: 4-bit is almost universally optimal.** Over 35,000 experiments, 19M–176B
   parameters, 3–8 bits: "4-bit precision is almost universally optimal for total model bits and zero-shot accuracy"
   ([Dettmers & Zettlemoyer, 2212.09720](https://arxiv.org/abs/2212.09720)) **[paper]**. Read the other way round:
   going from 4 to 3 bits to fit a model twice as large usually does *not* pay off for dense models with RTN-class
   quantizers. Small block sizes were the only thing that shifted the curve. (MLX g32/g64 and GGUF k-quants are small
   blocks.)
2. **Heavily trained models are more fragile under quantization.** "The degradation introduced by post-training
   quantization increases as models are trained on more data"
   ([Kumar et al., Scaling Laws for Precision, 2411.04330](https://arxiv.org/abs/2411.04330)). See also
   ([Ouyang et al., 2411.17691](https://arxiv.org/abs/2411.17691)): smaller models trained on more tokens suffer more
   quantization-induced degradation **[paper]**. GLM-5.3-Flash was trained on a 30T-token corpus and DeepSeek-V4 on
   32T+ **[primary]**. These are deep in the "over-trained" regime, so expect the 3-bit cliff to be sharper than
   older papers suggest.
3. **MoE experts are unusually robust to low bits.** "Expert layers in MoE models are much more robust to the
   quantization than conventional feedforward networks", and "MoE models with 2-bit expert weights can deliver better
   model performance than the dense model trained on the same dataset"
   ([MoQE, 2310.02410](https://arxiv.org/abs/2310.02410)) **[paper]**. QMoE compressed SwitchTransformer-c2048 to
   below 1 bit per parameter ([2310.16795](https://arxiv.org/abs/2310.16795)), but that model was under-trained by
   modern standards. The likely intuition is that each expert sees only a small fraction of tokens, so its weights
   are less "saturated" and carry more redundancy. That is why a well-allocated ~3–3.5 bpw build of an MoE can sit at
   ≈ bf16 quality (Qwen IQ3_S 93.26 vs bf16 93.12 **[3rd-party]**).
4. **A newer small-scale result favours "bigger-but-smaller"** with *learned per-group precision*. A 131M model at
   1.82 mean bits beat a 55M fp16 model, but the author says scaling the search past 1.5B "remains open"
   ([VBQ, 2607.02893](https://arxiv.org/abs/2607.02893)) **[paper, small scale, not MoE]**.

**Applying it to GLM vs Qwen at 68 GB.** Points 1 and 3 together say a larger MoE at ~3–3.5 bpw *experts* can beat a
smaller model at higher bits, as long as the extra parameters survive. GLM cannot keep them: at 68 GB it is not
"321B at 3 bits" but "~165B of surviving experts at 3 bits". The comparison stops being bits vs parameters and
becomes **pruning plus bits vs bits alone**. The composable-compression benchmark found expert pruning to be "the
dominant source of degradation" across 10 MoE models ([2608.21693](https://arxiv.org/abs/2608.21693)) **[paper]**, and
sh0wie found "pruning error dominates quantization error at this width" **[self-reported]**. §4.3 quantifies this.

### 2.5 Bigger than RAM: SSD expert streaming

**Mechanism in mlx-vlm 0.7.9** (`mlx_vlm/moe_offload.py`, `models/switch_layers.py::OffloadedSwitchGLU`) **[src]**:

- `python -m mlx_vlm.moe_offload --build <mlx dir> --out <dir>` rewrites each MoE layer's stacked experts into
  `experts/layer_XXX.safetensors` with per-expert keys, plus `offload_index.json`. Everything else stays resident.
- `mlx_vlm.load()` sees `offload_index.json` and swaps every `SwitchGLU` for `OffloadedSwitchGLU`. For each step it
  takes the router's indices to the CPU (`np.asarray(indices)`, one GPU→CPU sync per MoE layer), loops over the unique
  experts, and gets each one's `(w, scales, biases)` from an `ExpertStore`.
- `ExpertStore` keeps a **byte-budgeted LRU** of materialized experts. The default budget is
  `0.8 × max_recommended_working_set − resident − KV reserve`, which would be **~92 GB − resident here, far too big
  for co-tenancy**. The server flag is `--expert-cache-gb`, or the kwarg `expert_cache_gb` passed to `load()`.
- When one call touches more than half a layer's experts (a prefill chunk), `get_all()` bulk-loads the whole layer
  file and bypasses the LRU. **Prefill must be chunked**, otherwise one lazy graph pins every expert of every layer.

**Hit-rate math.** Suppose the cache holds a fraction `c` of a model's expert bytes. If routing were independent from
token to token with frequencies `p_j`, a well-tuned cache that keeps the most-used experts would hit with probability
about the routing mass of the experts it holds, `Σ_{j∈cache} p_j`. For near-uniform routing that is just `c`.

- GLM's routing is close to uniform: aux-loss-free balancing gives normalized entropy 0.92–0.99, and only 1.4 % of
  experts get under 10 % of average traffic **[self-reported, OpenMOSE; reap.md §4]**.
- mlx-vlm's own docstring says DeepSeek-V4-Flash routing "samples close to the full expert set within a couple
  hundred tokens" **[src]**.
- The only thing that lifts the hit rate above `c` is **temporal locality**: consecutive tokens reusing experts.
  MoE-Infinity reports that at batch 1, "a small number of experts are frequently reused" during decode
  ([2401.14361](https://arxiv.org/abs/2401.14361)) **[paper]**. How large this effect is for GLM is unknown and has
  to be measured (§4.4 step 6).

**Bytes from SSD per decoded token:**

$$B_\text{SSD} = L_\text{MoE}\cdot k\cdot(1-h)\cdot s_\text{expert}$$

For GLM: 42 layers × top-8 × (1−h) × 11.0 MB (25.17M params at 3.5 bpw).

**Measured SSD on this machine** (uncached `pread` with `F_NOCACHE`, 16 GiB file of unique random data, 1.5 GiB of
random reads per cell) **[measured]**:

| Read size | 1 thread | 4 threads | 16 threads |
|---|---|---|---|
| 2.1 MB (one Qwen expert @3.5 bpw) | 2.7 GB/s | 4.3 GB/s | 7.1 GB/s |
| 11 MB (one GLM expert @3.5 bpw) | 7.9 GB/s | 11.2 GB/s | 13.6 GB/s |
| 64 MB | 9.4 GB/s | 9.2 GB/s | 9.3 GB/s |

Caveats: the machine was shared with downloads; page-cache file-backed pages rose ~5.6 GB during the run, so a small
part of the reads may have been cached; a first run on a 6 GiB file read up to 24 GB/s and was discarded as cached.
mlx's lazy `mx.load` path reads one tensor at a time, so plan with the **1-thread column** unless the loader is
parallelized (stretch idea S4).

**Prefill is the hard part.** A 2048-token chunk routes to nearly all experts in every layer, so each chunk reads the
whole expert set. For unpruned GLM at 3.5 bpw that is 133 GB. At ~9 GB/s that takes ~15 s, so a 2048-token chunk runs
at **~130 tok/s** and an 8192-token chunk at ~440 tok/s **[computed]**. A 50K-token Claude Code prompt would need
2–6 minutes before the first token. Streaming is therefore a profile for *hard, short-prompt problems*, not for an
agent loop.

### 2.6 Multi-model systems: cascades and routing

A cascade sends a request to a cheap model first and escalates when needed:

$$\mathbb E[q] = q_\text{cheap} + p_\text{esc}\,(q_\text{big}-q_\text{cheap}\mid\text{esc}),\qquad \mathbb E[t]=t_\text{cheap}+p_\text{esc}\,t_\text{big}$$

- FrugalGPT ([2305.05176](https://arxiv.org/abs/2305.05176)) and RouteLLM ([2406.18665](https://arxiv.org/abs/2406.18665))
  learn the escalation decision from data **[paper]**.
- For coding agents there is a free, strong signal: **the tests**. Escalate when the primary fails the same
  test/lint twice, or loops (turns-to-solve explodes; this was the main pruning failure mode in reap.md §3).
- It only pays when q_big − q_cheap on escalated items is large *and* their errors are not correlated. Two models from
  different labs with different strengths (Qwen vs GLM, §3.2) are a better pair than two Qwen sizes.

**Claude Code already routes by model name.** It sends `ANTHROPIC_MODEL` for the main loop and
`ANTHROPIC_DEFAULT_HAIKU_MODEL` / `CLAUDE_CODE_SUBAGENT_MODEL` for subagents (`decode.md` §6). If `serve/` exposes
several *served names* that map to (model, reasoning effort) pairs, routing needs no classifier at all.

---

## 3. Evidence

### 3.1 Base quality of every candidate

**Artificial Analysis Intelligence Index v4.3.2** (independent; one harness, 10 evaluations including Terminal-Bench
4.0, SciCode, HLE, AA-LCR, AA-Omniscience, GDPval-AA) **[3rd-party]**:

| Model | Total / active | AA index | Notes | Source |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 (Max) | 552B backbone + 196B Engram / 16B decode | **45** | Released 2026-10-17. Does not fit (§3.6) | [AA open-weights](https://artificialanalysis.ai/models/open-source) |
| **GLM-5.3-Flash** | 320B / 18B | **42** | TB 4.0 33 %, HLE 40 %, AA-LCR 80 %, Omniscience +7, 77 tok/s API | [AA compare](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-qwen3-8-flash-next) |
| **Qwen3.8-Flash-Next** | 180B / 6B | **40** | TB 4.0 25 %, HLE 38 %, AA-LCR 80 %, Omniscience −10; 108k output tokens per task vs GLM 69k | same |
| Qwen3.8-27B (Xhigh) | 27B dense | 34 | 26 at Low effort | [AA open-weights](https://artificialanalysis.ai/models/open-source), [AA compare](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-qwen3-8-27b-low) |
| DeepSeek-V4-Flash-0731 (Max) | 284B / 13B | 34 | TB 4.0 12 % | [AA compare](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-deepseek-v4-flash) |
| Nemotron-3.5-Super-VL | 120B / 12B | 26 | "leading US open-weight model" | [AA article](https://artificialanalysis.ai/articles/nemotron-3-5-super-and-bolt) |
| Nemotron-3.5-Bolt | 74B / 9B | 21 | | same |
| Qwen3.6-35B-A3B | 35B / 3B | 18 | candidate "fast lane" small MoE | [AA model page](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) |
| Mistral-Medium-3.5 | 128B dense | 14 | | [AA model page](https://artificialanalysis.ai/models/mistral-medium-3-5) |

Some secondary blogs quote GLM 57.5 / Qwen 55.8. Those come from an older index version and should not be mixed with
the v4.3.2 numbers above.

**NVIDIA's NeMo Gym cross-evaluation** (from the Nemotron-3.5-Super-VL card). NVIDIA re-ran competitors under its own
published recipes, so it is a single-harness *third-party* evaluation for Qwen, GLM and DeepSeek **[3rd-party for
Qwen/GLM/DSV4]**
([card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)):

| Benchmark | Qwen3.8-FN | GLM-5.3-Flash | DSV4-Flash-0731 | Nemotron-3.5-Super | Winner |
|---|---|---|---|---|---|
| SWE-bench Verified | 78.7 | **79.1** | 75.4 | 71.1 | tie |
| SWE-bench Pro | **60.4** | 59.7 | 56.9 | 51.8 | tie |
| SWE-bench Multilingual | 77.1 | **78.3** | 72.0 | 67.7 | tie |
| Terminal-Bench 2.1 | **85.8** | 82.7 | 76.8 | 60.7 | Qwen |
| Terminal Bench 4 (offline) | **20.7** | 15.2 | 3.5 | 1 | Qwen (AA's TB 4.0 says GLM 33 vs 25: **harness disagreement**) |
| DeepSWE (offline) | 55.6 | **57.8** (63.4 vendor) | 42.3 | 24.7 | tie / GLM |
| LiveCodeBench v6 | **94.1** | 80.6 | 94.6 | 88.2 | Qwen |
| IFBench | **75.6** | 57.7 | 76.2 | 70.9 | Qwen |
| τ²-bench retail / airline / telecom | 71.6 / 80.8 / 90.4 | **78.2 / 81.8 / 98.5** | 73.8 / 83.5 / 98.2 | 69.2 / 76.5 / 73.2 | GLM |
| τ³-bench banking | **44.5** | 34.8 | 38.6 | 18.9 | Qwen |
| BFCL | — | — | — | — | not reported anywhere primary (see below) |
| AA-LCR v1.1 (long context) | 78.4 | **79.5** | 79.7 | 68.6 | tie |
| GraphWalks (long context) | 76.1 | **88.4** | 65.4 | 89.7 | GLM |
| GPQA-Diamond | **92.7** | 91.2 | 89.6 | 89.5 | tie |
| HLE text-only | **39.6** | 36.5 | 35.8 | 31.8 | Qwen |
| AA-Omniscience hallucination rate (lower is better) | 41.3 % | **30.3 %** | 95.9 % | 30.0 % | GLM |
| AA-Omniscience index | −1.9 | **14.3** | −7.8 | 4.0 | GLM |
| OSWorld-Verified | **55.8** | 14.8 | — | 51.1 | Qwen |

**Vendor-reported** (each lab's own harness, so not comparable across labs):

- *GLM-5.3-Flash* (zai-org card chart): Terminal-Bench 2.1 **84.3** (Claude Code 2.1.207 harness), DeepSWE v1.1 63.4,
  Agents' Last Exam 26.3, AutomationBench 48.8, HLE w/ tools 55.3, GDPval-AA v2 1773
  ([card](https://huggingface.co/zai-org/GLM-5.3-Flash)) **[self-reported]**. The official DeepSWE leaderboard
  reportedly lists ~63 % **[3rd-party, secondary source; not verified here]**.
- *Qwen3.8-Flash-Next* (Qwen card): SWE-bench Pro 62.5, SWE-bench Multilingual 81.0, DeepSWE 1.1 58.7 (best of two
  harnesses), LiveCodeBench v6 91.9, GPQA-D 91.7, HLE 35.9, Toolathlon 73.5, IFBench 81.3
  ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) **[self-reported]**.
- *Qwen3.8-27B*: Terminal-Bench 2.1 (Terminus) 73.0, SWE-bench Pro **61.7** (only 0.8 below Flash-Next in the same
  table), DeepSWE 42.2, LiveCodeBench v6 90.3 ([card](https://huggingface.co/Qwen/Qwen3.8-27B)) **[self-reported]**.
- *DeepSeek-V4-Flash Max*: SWE-bench Verified 79.0, Terminal-Bench 2.0 56.9, LiveCodeBench 91.6, HMMT Feb 2026 94.8,
  MRCR 1M 78.7 ([card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)) **[self-reported]**.
- *Mistral-Medium-3.5-128B*: SWE-bench Verified 77.6, τ³-Telecom 91.4
  ([card](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B)) **[self-reported]**.

**Math (AIME/HMMT).** Neither GLM-5.3-Flash nor Qwen3.8-Flash-Next appears on MathArena's AIME 2026 / HMMT Feb 2026
boards as of today (searched). The best independent number for Qwen is IST-DASLab's bf16 baseline: **AIME25 100.0,
GPQA-D 91.9, LCB v6 87.4** ([GSQ-RCO card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF))
**[3rd-party]**. There is no comparable GLM number, so `evals.math` must produce one.

**Function calling (BFCL).** Neither vendor nor any primary leaderboard publishes BFCL v3/v4 for these two models (a
search turned up only aggregator snippets with unattributed numbers). Use our own `evals.tools` (BFCL v3 subset,
already cached) plus τ²-style agent evals.

**What the base-quality evidence says:**
- GLM and Qwen are statistically tied on SWE-bench Verified/Pro/Multilingual under one harness (≤ 1.2-point gaps,
  roughly the ±2 points sampling noise of a 500-task benchmark).
- Qwen leads clearly on Terminal-Bench 2.1, LiveCodeBench and IFBench.
- GLM leads clearly on τ²-bench, GraphWalks and hallucination/knowledge (AA-Omniscience).
- Terminal-Bench 4 gives *opposite* winners under AA and NVIDIA harnesses. **Harness choice moves agentic scores by
  more than the model gap.** This is why our own Claude Code-harness evals (§5) decide, not leaderboards.

### 3.2 Complementarity: where a second model would add value

| Dimension | Better model | Evidence |
|---|---|---|
| Hallucination / factual calibration | GLM (30 % vs 41 % hallucination rate; Omniscience +14 vs −2) | NVIDIA table, AA |
| Customer-service style tool loops (τ²) | GLM (+6.6 retail, +8.1 telecom) | NVIDIA table |
| Graph/long-structure reasoning (GraphWalks) | GLM (+12.3) | NVIDIA table |
| Terminal / shell agentic, competitive coding, instruction following | Qwen | NVIDIA table, both vendors |
| Computer use (OSWorld) | Qwen (55.8 vs 14.8) | NVIDIA table |

GLM's edge lies in **knowledge and calibration**, which is exactly what expert pruning destroys first (§2.3).
A heavily pruned GLM keeps its weaknesses and loses its strengths, so as a second opinion it is only worth running
lightly pruned or unpruned.

### 3.3 Compression evidence (summarized from quant.md and reap.md, all re-checked there)

| Build | Size | Measured damage |
|---|---|---|
| Qwen-FN GSQ-RCO IQ3_S, unpruned, 3.5 bpw (GGUF) | 83.6 GB | task avg 93.26 vs bf16 93.12 (≈ lossless) **[3rd-party]** |
| Qwen-FN GSQ-RCO IQ3_XXS, unpruned, 3.0 bpw | 75.8 GB | 99.4 % of bf16 **[3rd-party]** |
| Qwen-FN GSQ-RCO IQ2_XS, unpruned, 2.5 bpw | 68.0 GB | 95.7 % (LCB −4.0) **[3rd-party]** |
| Qwen-FN REAP-320, oQ3e + DWQ (MLX) | 70.8 GB disk / ~36 GiB resident | KL20 0.104, CE-gap 0.090; agent domain worst (0.185) **[self-reported]** |
| Qwen-FN REAP 512/384/320/288/256, 4-bit | 98/80/72/68/— GB | HumanEval 93.9/92.1/90.9/91.5/88.4; rare-name probe collapses at 256 **[self-reported]** |
| GLM REAP50 (K=144), FP8, 7-domain calibration | ~165 GB | top-1 agreement 0.842, ΔNLL **+0.176**; general-knowledge ("ballast") top-1 0.58 **[self-reported]** |
| GLM REAP50, wikitext calibration | 96 GB (mixed 4/8) | PPL 3.57 → 6.08 (+0.56 nats) **[self-reported]** |
| GLM REAP144-Coder Q3_K (own sessions) | 71 GB | 48/48 tasks but ×1.80 turns; 120 experts loops, 96 unusable **[self-reported]** |
| GLM unpruned GSQ-RCO 3.0 bpw (GGUF) | 117 GB | approx. KL **0.142**, MMLU-Pro −1.95 pp vs Q8_0 **[self-reported]** |
| GLM unpruned MLX base 3g32 / 2g32 | 184 / 145 GB | KLD 0.042 / 0.165 **[self-reported]** |

### 3.4 Streaming evidence

- **mlx-vlm moe_offload exists and was built for exactly this case** (DeepSeek-V4-Flash, 256 experts × 43 layers,
  bigger than RAM) **[src]**.
- OptiQ-2bit Qwen3.8-Flash-Next SSD-streamed: ~3.8 tok/s on an M3 Max **[self-reported, quant.md §2]**.
- MTPLX streams only the 32 GB n-gram table from SSD and decodes Qwen at 44–74 tok/s on an M5 Max **[self-reported]**.
  Streaming a *lookup table* is cheap, because each token reads a few KB of rows. Streaming *experts* is expensive,
  because each token reads ~2–11 MB per expert miss.
- Mixtral offloading combined an LRU expert cache with speculative prefetch
  ([Eliseev & Mazur, 2312.17238](https://arxiv.org/abs/2312.17238)). Pre-gated MoE predicts the *next* layer's
  experts from the current layer to hide fetch latency ([2308.12066](https://arxiv.org/abs/2308.12066)). MoE-Infinity
  reports 3.1–16.7× per-token latency gains from activation-aware caching
  ([2401.14361](https://arxiv.org/abs/2401.14361)) **[paper]**.

### 3.5 Helper models

- **`google/embeddinggemma-2`** (already in `~/.cache/huggingface`): 740M total, of which the **text tower is 270M**.
  768-d with Matryoshka 512/256/128, 8,192-token context, Apache-2.0. MTEB v2 English 68.46, multilingual 61.36,
  **code 78.68**. Prompts: `task: search result | query: …` and `title: … | text: …`
  ([card](https://huggingface.co/google/embeddinggemma-2)) **[self-reported]**. MLX ports exist
  (`mlx-community/embeddinggemma-2-bf16`/`-8bit`), and mlx-vlm 0.7.9 has an `embedding_gemma2` model dir **[src]**.
- **`nvidia/Nemotron-3.5-Rerank-8B-BF16`**: 8.38B, built from Ministral-3-8B with a *bidirectional* language tower and a
  Pixtral vision encoder, 32K context, OpenMDW-1.1. The card names only ViDoRe V3 / MIRACL-VISION (no scores, no
  BEIR/code numbers) and lists Transformers/vLLM on NVIDIA GPUs only
  ([card](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)) **[primary]**. Porting it to MLX means reusing
  mlx-vlm's `mistral3`/`ministral3` modules with a non-causal mask, at ≈ 4.7 GB in 4-bit / 8.9 GB in 8-bit
  **[computed]**. It is a *visual-document* reranker first, and its value on code/text RAG is unproven.
- **Qwen3.6-35B-A3B** (3B active, AA 18): ~19.7 GB at 4.5 bpw; "112–134 tok/s for a 35B-A3B on M5 Max" via Ollama-MLX
  **[self-reported, mac.md §7]**. The only small model that is actually faster than the primary here.
- **`incoai/Qwen3.8-Flash-Next-DFlash2`** was uploaded 2026-10-19 with 0 downloads and no acceptance data. Untested,
  as decode.md says.

### 3.6 Models out of reach (and why)

| Model | Why it does not fit | Watch for |
|---|---|---|
| DeepSeek-V4.1-Flash(-1017), AA 45 | 384 experts × 40 layers × 35.4M params = 543B of experts. At 3.5 bpw that is **238 GB** (15.5 MB per expert). The 196B Engram table could be mmapped like Qwen's PLE, but the experts cannot **[computed from config.json]**. mlx-vlm has `deepseek_v41` | community REAP-50 builds; at ~120 GB + streaming it becomes a P3 candidate, though at ~3 tok/s **[estimate]** |
| GLM-5.3 (753B/40B), AA 45; Kimi K3 (2.8T); MiMo-V2.6-Pro (1T) | ≥ 5× too large | — |

---

## 4. Plan for this machine

### 4.1 Memory profiles

| Profile | When | Process cap | Resident weights | Left for macOS + dev | Notes |
|---|---|---|---|---|---|
| **P1 co-tenant** (default) | coding, IDEs, Claude Code running | 80 GB | ≤ 64 GB | ≥ 48 GB | One MLX process. Wiring off by default (`mac.md` rec. 2), `max-num-seqs` 1–2 |
| **P1-lean** | P1 plus helpers (reranker, small fast model) in the same process | 80 GB | primary ≤ 40 GB + helpers ≤ 24 GB | ≥ 48 GB | Uses the REAP-320 primary |
| **P2 all-in** | user away, overnight jobs, "think hard" sessions | 100 GB | ≤ 90 GB | ≥ 28 GB | Above the 65 % rule of thumb in `mac.md` rec. 7. Single sequence, no wiring, no concurrent MLX process. Soak-test first |
| **P3 stream** | bigger-than-RAM model (unpruned GLM) | 80 or 100 GB | trunk ~9 GB + expert cache 55–90 GB | as P1/P2 | Expert cache via `--expert-cache-gb`; short prompts only (§2.5) |

`max_recommended_working_set_size` is 107.5 GiB (115.4 GB) on this machine (`mac.md` §1), so P2 is the practical
ceiling.

### 4.2 Ranking of every candidate under each profile

Quality column: base quality (AA v4.3.2), adjusted for the compression the profile forces. ⭐ = recommended.
Speeds are **[estimate]** unless marked.

**P1 co-tenant (≤ 64 GB resident):**

| Rank | Model / build | Resident GB [computed] | Compression forced | Decode | Verdict |
|---|---|---|---|---|---|
| 1 ⭐ | Qwen3.8-FN, K=512 (or 448), ~3.7 bpw mixed, PLE mmap, MTP | 512×0.236×3.7/8 + 4.6 + 1.4 = **61.9** | none or ~12 % pruning; bits at ≈ lossless-for-MoE level | ~44 plain / ~74 MTP [self-reported, MTPLX] | Best quality per resident GB |
| 2 | Qwen3.8-FN REAP-320, ~3.3–3.5 bpw | 36.7–39.0 | 37.5 % pruning | same | Use when RAM is needed for helpers (P1-lean) |
| 3 | Qwen3.8-27B, 8-bit g64 | 28.7 (+ 8 GiB KV @128K) | ≈ none | ≤ 21 tok/s ceiling (28.7 GB/token) | Robust, Apache-2.0, but lower (AA 34) and slower |
| 4 | GLM-5.3-Flash REAP-144, ~3.0 bpw | 65.6 | 50 % pruning **and** ~3 bpw | ~25–35 | Dominated by #1 (§4.3) |
| 5 | DeepSeek-V4-Flash, REAP-50, ~3 bpw | ~55–65 [estimate] | 50 % pruning + ~3 bpw | — | Lower base (AA 34) *and* the GLM problem |
| 6 | Nemotron-3.5-Super-VL, 4-bit | 120×4.5/8 = 67.5 | 4-bit | ~12B active | AA 26; dominated |
| 7 | Mistral-Medium-3.5-128B, 4-bit | 72.0 | 4-bit | ≤ 8.5 tok/s | AA 14; dense, too slow for agents |

**P2 all-in (≤ 90 GB resident):**

| Rank | Model / build | Resident GB | Notes |
|---|---|---|---|
| 1 ⭐ | Qwen3.8-FN unpruned, 4.5 bpw (4-bit g64 class with sensitivity floors) | 512×0.236×4.5/8 + 6 = **74.0** | Near-lossless. Leaves room for 256K context (KV 7.6 GiB) |
| 2 ⭐ (secondary) | GLM-5.3-Flash REAP-176 @ ~3.3 bpw, or REAP-192 @ ~3.15 | 176: 84.4–90.2 · 192: ~88 | 33–39 % pruning; second opinion / escalation |
| 3 | GLM REAP-160 @ 3.5 | 82.7 | More bits, more pruning; sweep against #2 |

**P3 stream (bigger than RAM):**

| Rank | Model / build | Disk / resident | Decode [computed, §4.6] | Use |
|---|---|---|---|---|
| 1 | GLM-5.3-Flash **unpruned**, 3.5 bpw, expert cache 90 GB (P2) | 143 GB / ~99 GB | 5.8–13.6 tok/s | Overnight hard problems; knowledge-heavy questions |
| 2 | Same, expert cache 55 GB (P1-compatible, *no Qwen loaded*) | 143 / ~64 | 3.4–6.5 tok/s | Swapped in when the user is not coding |
| 3 | GLM REAP-216 (25 % pruned), 3.5 bpw, cache 55 GB | 109 / ~64 | 4.3–8.9 tok/s | Middle ground if unpruned thrashes |
| watch | DeepSeek-V4.1-Flash (AA 45) | ~250 GB / — | ~2–3 tok/s [estimate] | Only if a REAP build appears and the AA lead persists |

### 4.3 Quality per GB: heavily compressed GLM vs lightly compressed Qwen at the same 68 GB

This compares like with like as far as published evidence allows. KL and ΔNLL are both in nats per token against each
model's *own* uncompressed reference.

| | Qwen3.8-FN @ ~68 GB | GLM-5.3-Flash @ ~66 GB |
|---|---|---|
| Base quality (AA v4.3.2) | 40 | 42 (+5 % relative) |
| Experts kept | 320 of 512 (62.5 %) *or*, using PLE streaming, **all** | 144 of 288 (50 %) |
| Expert bpw | ~3.3 | ~3.0 (about half of the experts at 2-bit) |
| Pruning damage alone | REAP-320: part of the KL20 0.104 below | ΔNLL **+0.176** (7-domain, FP8, no quantization) |
| Quantization damage alone | ~3.0 bpw unpruned: task retention 99.4 % (GGUF) | ~3.0 bpw unpruned: KL **≈ 0.142** (best-in-class GGUF); MLX affine is cruder (`reap.md` rec. 2) |
| Combined, published | KL20 **0.104** (pruning *and* 3-bit *and* DWQ, MLX) | no published pruned+3-bit KL. Additive lower bound ≈ 0.176 + 0.142 ≈ **0.32 nats** |
| Ratio | 1× | **≈ 3×** the divergence (lower bound) |

KL20 (top-20 KL) slightly under-counts full KL. ΔNLL and KL are not identical, but both measure the extra nats per
token the compressed model spends. The ~3× ratio is the right order of magnitude, not a precise number.

**Why ~3× the divergence outweighs +5 % base quality.** Studies that map divergence to task loss
(oQ, GSQ-RCO, the Litwein ablations) show task retention falling off steeply once divergence passes ~0.15–0.2 nats:
GLM 2-bit KLD 0.165 already costs +57 % perplexity (orcarouter). The GLM build lands past that knee; the Qwen build
sits before it. Pruning also strips GLM's specific advantage (knowledge, low hallucination, §3.2).

**Verdict:** at 68 GB, and at any P1 budget, the lightly compressed Qwen wins with high confidence. GLM only competes
once ≥ ~85 GB is available (P2) or by streaming (P3). There it keeps ≥ 61 % of experts or all of them.

**What would overturn this:** (a) a pruned GLM K=144 at ~3 bpw scoring within 2 points of the Qwen build on *our* code
and agentic evals (§5); (b) a non-REAP compression for GLM (expert merging, low-rank experts) with published
end-to-end KL < 0.12 at ≤ 68 GB. Neither exists today.

### 4.4 Concrete steps against this repo

Conventions: `SNAP_Q=~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d…`,
`SNAP_G=~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/46a050be…`. Run large jobs one at a
time (PLAN.md rules) and only after the downloads finish.

**Step 0 — baselines (already on disk).** Score the two existing builds so every later build has a reference:

```bash
for m in qwen-flash-next-reap288 baseline-qwen-reap320; do
  uv run python -m evals.speed --model models/$m --contexts 1024,32768,131072 --mtp-modes off,on
  uv run python -m evals.kl    --model models/$m --ref artifacts/qwen/calib/ref   # once calib ref exists
  uv run python -m evals.code  --model models/$m --datasets humaneval+,mbpp+
  uv run python -m evals.tools --model models/$m --limit 200
done
```

**Step 1 — calibration data, including personal data** (details in the calibration chapter; the model-selection
angle):
- `~/.claude/projects/*/*.jsonl` holds **3,116 transcripts, 5.4 GB on disk** (sizes only; contents not read). At a
  rough ~25–40 % of the bytes being real dialogue/tool text and ~4 bytes/token, that is on the order of
  **300–500M tokens** of the user's own agentic traffic **[estimate]**. That is 100× more than calibration needs.
- Use it two ways:
  1. As the `agentic` domain source for REAP saliency. Re-render every turn into **each target model's own chat
     template and tool format** (`calib_data/render.py`), because routing depends on the format tokens. Self-generated
     data routes like real traffic (`reap.md` §4).
  2. As a **held-out personal eval** (§5): bits-per-byte of the transcripts' assistant turns under each model.
- Privacy: transcripts contain file contents, paths and possibly secrets. Filter with a secret scanner, keep
  everything under `artifacts/` (local only), and never upload derived builds' calibration manifests.

**Step 2 — calibrate both models** (per `docs/contracts/calib.md`):

```bash
uv run python -m calib.run --model qwen --src $SNAP_Q --data artifacts/qwen/data --out artifacts/qwen/calib --mem-gb 40
uv run python -m calib.run --model glm  --src $SNAP_G --data artifacts/glm/data  --out artifacts/glm/calib  --mem-gb 40
```

Time **[estimate from the repo microbenchmark]**: MoE-layer prefill measured 327K tok/s/layer (Qwen shapes) and
65.8K tok/s/layer (GLM shapes) in bf16 (`quant.md` §4). Summed over layers that gives ~6.8K and ~1.5K tok/s for the MoE
parts alone. With attention, sensitivity probes and hidden-state I/O, assume ~2–3K (Qwen) and ~0.7–1K (GLM) tok/s, so
**4M calibration tokens take ~25–35 min (Qwen) and ~70–95 min (GLM)** per pass. Disk churn matters more. `work/` holds
~2 × tokens × streams × hidden × 2 B per layer: for Qwen at 4M tokens, 2 × 4e6 × 10,240 × 2 = **164 GB written and
read per layer, ~7.9 TB per run**; GLM (4 × 4096 streams: 2 × 4e6 × 16,384 × 2 = 262 GB per layer × 45 layers) **~12 TB per run**. Keep calibration to 2–4M tokens and
reuse runs.

**Step 3 — Qwen builds: settle "prune vs bits" at a fixed resident budget.** `build.make --budget-gb` counts the PLE
table, pinned at `--ple-bits 4` = affine g32 = **32.0 GB** (`build/pipeline.py`, `kind == "ple_table"`). So
resident ≈ budget − 32.0. Four builds at ≈ 62 GB resident (`--budget-gb 94`) plus the lean one:

```bash
B="uv run python -m build.make --model qwen --src $SNAP_Q --calib artifacts/qwen/calib --drop-vision --mtp prune"
$B --prune none --budget-gb 94   --out models/qwen-k512-r62     # all experts, ~3.7 bpw
$B --prune 448  --budget-gb 94   --out models/qwen-k448-r62     # ~4.2 bpw
$B --prune 384  --budget-gb 94   --out models/qwen-k384-r62     # ~4.9 bpw
$B --prune auto --budget-gb 94   --out models/qwen-auto-r62     # let alloc trade prune loss vs sensitivity
$B --prune 320  --budget-gb 70   --out models/qwen-k320-r38     # P1-lean (~38 GB resident)
$B --prune none --budget-gb 106  --out models/qwen-k512-r74     # P2 all-in, ~4.5 bpw
```

Arithmetic for the first build: experts get 94 − 32.0 (PLE) − 4.6 (trunk) − 1.4 (MTP) = 56.0 GB → 56.0e9 × 8 /
120.8e9 = **3.71 bpw** average. For `--mtp prune` at K=512 nothing is pruned. Use `--mtp intact` for the K<512 builds
to keep the drafter's 76–81 % acceptance (`reap.md` §7).

Then run `--dry-run --summary-json` first on each to confirm the allocator's chosen bytes before writing ~90 GB.
Disk per build: ~90 GB. Six builds ≈ 540 GB of the 2.5 TB free.

Code change worth making (small): a `--budget-resident-gb` flag in `build/make.py` that excludes `ple_table` units
from the budget and writes `ple-store.json`, so budgets are stated in the unit that matters.

**Step 4 — GLM builds** (only after step 3 shows where Qwen lands):

```bash
G="uv run python -m build.make --model glm --src $SNAP_G --calib artifacts/glm/calib --drop-vision"
$G --prune 144 --budget-gb 66  --out models/glm-k144-r66 --mtp prune   # equal-bytes experiment vs qwen-k320/k512
$G --prune 176 --budget-gb 88  --out models/glm-k176-r88 --mtp prune   # P2 secondary candidate
$G --prune 192 --budget-gb 88  --out models/glm-k192-r88 --mtp prune
$G --prune none --budget-gb 145 --out models/glm-full-a35 --mtp intact --min-bits 3   # for P3 streaming
uv run python -m mlx_vlm.moe_offload --build models/glm-full-a35 --out models/glm-full-a35-offload
```

The GLM serving fixes in `decode.md` rec. 1 (drop the projected MLA cache after prefill,
`_MAX_PROJECTED_PREFILL_TOKENS ≤ 4096`) are **mandatory** for every GLM build. Without them GLM holds an extra
22–24 GiB at ≥ 32K context.

**Step 5 — evaluate.** See §5 for metrics and sample sizes. Evals on reasoning models are slow, so budget them
**[estimate]**:

| Suite | Items | Tokens per item (thinking) | At ~60 tok/s | Per build |
|---|---|---|---|---|
| `evals.kl` | full `eval.jsonl` | prefill only | ~2–4K tok/s prefill | ~10–20 min |
| `evals.code` HumanEval+/MBPP+ | 164 + 378 | ~2–4K | 35–65 s | ~6–10 h → use `--limit 200` (~2–3 h) |
| `evals.math` AIME24/25 × 4 samples | 240 | ~10–15K | 3–4 min | **~13 h** → use AIME25 × 2 (~3.5 h) |
| `evals.tools` BFCL subset | 200 | ~0.5–1K | ~10 s | ~0.6 h |
| `evals.longctx` 32K/64K/128K × 8 | 24 | prefill-heavy | — | ~1 h |
| Personal bits-per-byte | 2M bytes held out | prefill only | — | ~20 min |

Full sweep per build ≈ 8–10 h. Run KL and bits-per-byte (cheap) on every build, and task evals only on the top 2–3.

**Step 6 — measure streaming feasibility before building P3.** Add a routing-trace option to `calib/probe.py`: for a
sample of eval sequences, dump the `(layer, token) → top-k expert ids`. Then simulate caches offline:

```bash
uv run python scripts/sim_expert_cache.py --traces artifacts/glm/calib/traces --expert-mb 11.0 \
    --cache-gb 55,90 --policy lru,lfu,reap-pinned --ssd-gbps 2.7,7.9
```

The simulator (to be written; ~100 lines of numpy) replays the traces and reports the hit rate `h` and predicted tok/s
from the formula in §2.5. It takes minutes and decides P3 before any 143 GB build exists.

**Step 7 — serve the system** (§4.5).

### 4.5 The multi-model system (one process)

`mac.md` rec. 7: run **one MLX process** (multiple processes and repeated load/unload are linked to IOGPU panics).
So every model lives inside the one `serve/` process and is selected by served name.

```
Claude Code / OpenCode  ──►  serve/ (one process, Anthropic + OpenAI APIs)
   ANTHROPIC_MODEL=local-big              ──► Qwen primary, thinking on, MTP, APC
   ANTHROPIC_DEFAULT_HAIKU_MODEL=local-fast ─► same Qwen weights, enable_thinking=false / effort low
   CLAUDE_CODE_SUBAGENT_MODEL=local-fast   ─► (batched with the main loop, max-num-seqs 2)
   /model local-deep (manual) or auto-escalation ─► P2/P3 only: GLM (resident or streamed);
                                               in P1: Qwen, effort max + best-of-n with test selection
RAG tool  ──► /v1/embeddings (embeddinggemma-2 text tower) ──► vector index ──► rerank (primary LLM or Nemotron-Rerank)
```

Proposed code layout:
- `serve/router.py`: served-name → (backend, generation overrides). Escalation rule: the client marks a retry, or the
  same tool/test error repeats twice in the conversation, or the request asks for `local-deep`.
- `serve/embed.py`: loads embeddinggemma-2 (text-only, bf16, ~0.6 GB) and serves `/v1/embeddings` with the
  card's `task: … | query:` prompts. Use Matryoshka 256-d for the index (1M chunks × 256 × 2 B = 0.5 GB).
- `serve/rerank.py`: v1 is listwise reranking by the primary over the top-20 chunks (prefix-cached system prompt). v2
  is the Nemotron-3.5-Rerank port behind the same interface.
- `scripts/serve.sh --profile {p1,p1-lean,p2,p3}` sets `--memory-limit-gb`, the APC limits, `--expert-cache-gb` and
  which backends to load.

**Memory per profile [computed]:**

| Component | P1 (default) | P1-lean | P2 | P3 (P1-sized) |
|---|---|---|---|---|
| Primary weights | 61.9 (qwen-k512-r62) | 38.0 (qwen-k320-r38) | 74.0 (qwen-k512-r74) | 0 (Qwen unloaded) |
| Primary KV @128K + MTP | 4.2 | 4.2 | 4.2 (8.2 @256K) | — |
| Second sequence (subagent) KV @32K | 1.1 | 1.1 | 1.1 | — |
| GLM | — | — | swap-in: 88 *instead of* Qwen | trunk ~9 + cache 55 |
| Embedder | 0.6 | 0.6 | 0.6 | 0.6 |
| Reranker | 0 (LLM) | 4.7 (Nemotron 4-bit) | 4.7 | 0 |
| Small fast MoE (optional, Qwen3.6-35B-A3B 4-bit) | — | 19.7 | — | — |
| Activations + APC RAM + MLX cache | 9 | 9 | 9 | 6 |
| **Total** | **76.8** | **77.3** (with 35B-A3B) / 57.6 (without) | **93.6** | **70.6** |

P2 never holds Qwen and GLM at once (74 + 88 > 115). "Escalate to GLM" in P2 means unloading Qwen, which costs a
load of ~88 GB from SSD at ~9 GB/s ≈ **10 s** plus warm-up. That is acceptable for a deliberate deep-dive, not for
per-request routing.

### 4.6 Streaming numbers for GLM [computed]

Model: decode time = misses × 11.0 MB / SSD rate + ~20 ms compute/sync. Hit rate shown as `h = c` (no locality) and
`h = c + 0.15` (some locality):

| Build | Expert bytes | Cache | c | h | SSD GB/token | tok/s @7.9 GB/s | tok/s @12 GB/s |
|---|---|---|---|---|---|---|---|
| unpruned 3.5 bpw | 133 GB | 55 GB | 0.41 | 0.41 → 0.56 | 2.17 → 1.62 | 3.4 → 4.5 | 5.0 → 6.5 |
| unpruned 3.5 bpw | 133 GB | 90 GB | 0.68 | 0.68 → 0.83 | 1.20 → 0.64 | 5.8 → 9.8 | 8.3 → 13.6 |
| REAP-216 3.5 bpw | 100 GB | 55 GB | 0.55 | 0.55 → 0.70 | 1.66 → 1.11 | 4.3 → 6.2 | 6.3 → 8.9 |
| unpruned 3.0 bpw | 114 GB | 55 GB | 0.48 | 0.48 → 0.63 | 1.64 → 1.17 | 4.4 → 6.0 | 6.4 → 8.5 |

Prefill: each 2048-token chunk reads the full expert set, giving **~130 tok/s** (unpruned) and ~170 tok/s (REAP-216).
8192-token chunks give ~440–540 tok/s but raise the activation peak, so check it with `evals.speed`.

MTP does **not** help an I/O-bound decode much: verifying k+1 tokens touches the union of their experts. At low
locality that is nearly (k+1)× the misses.

---

## 5. Decision criteria and what to measure

### 5.1 Metrics, and which comparisons each one is valid for

| Metric | Valid for comparing | Why |
|---|---|---|
| `evals.kl` (top-64 KL vs the model's own bf16 reference), per domain | builds **of the same model** | KL against different references is not comparable across models |
| **Bits-per-byte on held-out personal transcripts** (assistant turns, teacher-forced) | **across model families** | Normalizing by UTF-8 bytes removes tokenizer differences. Caveat: it measures how well a model predicts what *Claude* wrote, which tracks style and knowledge, not task success |
| `evals.code`, `evals.tools`, `evals.math`, `evals.longctx` | everything | task success. Use **paired** tests on the same items |
| Rare-name / fabrication probes (`reap.md` rec. 8) | builds with pruning | catches amputation damage that averages hide |
| Turns-to-solve on agentic tasks; loop/termination rate | everything | the main failure mode of over-pruned models |
| `evals.speed`: TTFT, decode tok/s with and without MTP, peak memory @128K | everything | must respect the profile caps |
| **Time-to-answer** = output tokens per task ÷ decode tok/s | everything | the user-visible cost (§2.2) |

**Sample sizes (paired, McNemar).** To detect an accuracy difference Δ between two builds on the same items, with a
discordant-pair fraction p_d, at α = 0.05 and 80 % power:

$$n \approx \frac{(z_{0.975}+z_{0.8})^2\,p_d}{\Delta^2} = \frac{7.84\,p_d}{\Delta^2}$$

With p_d ≈ 0.2: Δ = 5 points needs **n ≈ 630**; Δ = 10 points needs n ≈ 160. **HumanEval-164 alone can only detect
~10-point gaps.** Pool HumanEval+ and MBPP+ (542) plus BFCL (200) for the decision. For KL, the paired bootstrap over
windows (as in `quant.md`) is enough.

### 5.2 Decision rules

**D1 — Primary build (P1).** Among `qwen-k{512,448,384,auto}-r62`:
1. Reject any build whose `evals.speed` peak at 128K exceeds 76 GB, or whose plain decode is < 90 % of the best.
2. Pick the lowest mean per-domain KL, **weighted toward `agentic` and `code`** (Litwein found `agent` the most
   damaged domain).
3. Confirm on pooled code + tools: it must not be worse than `qwen-k320-r38` by a significant McNemar result. If K=512
   does not beat K=320 on tasks, keep K=320 and give the 24 GB to P1-lean helpers instead.
4. Rare-name probe ≥ 9/10 and no increase in loop rate.

**D2 — Is GLM worth having at all?** On the escalation set (items where the primary failed in step 5, plus τ²-style
tool loops and knowledge QA):
- Keep GLM as secondary if its best P2/P3 build solves **≥ 15 % of the primary's failures** (a cascade gain worth an
  unload/reload). Otherwise drop it and spend P2 on `qwen-k512-r74` with best-of-n.
- Pick resident (`glm-k176/192-r88`) vs streamed (`glm-full-a35-offload`) by: streamed must keep **≥ 5 tok/s decode**
  in the step-6 simulation *and* in reality, and beat the resident build on knowledge/hallucination probes by a
  significant margin. Otherwise use the resident build.

**D3 — The equal-bytes experiment** (`glm-k144-r66` vs `qwen-k320-r38`/`qwen-k512-r62`). This exists to settle the
question with our own data. Expected: Qwen wins. If GLM K=144 comes within 2 points on pooled code + tools *and* wins
on τ² and hallucination, revisit §4.3.

**D4 — Helpers.**
- Reranker: adopt Nemotron-Rerank v2 only if recall@5 on our RAG eval improves ≥ 5 points over LLM-listwise v1 at
  ≤ 3 s per query.
- Small fast MoE (35B-A3B): adopt only if the `local-fast` lane (primary, thinking off) misses a latency target, for
  example FIM autocomplete under 300 ms TTFT, which a 6B-active model with ~60 GB of weights may not meet.

### 5.3 What evidence would change the recommendation

| New evidence | Change |
|---|---|
| Our A/B shows K=320 ≈ K=512 on tasks (pruning damage negligible with personal calibration) | Primary becomes `qwen-k320-r38`; RAM goes to helpers and longer context |
| Mmapped PLE causes stalls under memory pressure (page-cache eviction → SSD reads per token) | Fall back to resident PLE, i.e. the REAP-320 plan from PLAN.md |
| A Flash-Next DFlash2/DSpark drafter shows τ > MTP independently | Swap the drafter; recompute time-to-answer |
| A GLM compression with end-to-end KL < 0.12 at ≤ 68 GB (merging, low-rank experts, QAT checkpoint from Z.ai) | Re-run D3; GLM could become the P1 primary |
| AA/NVIDIA re-evaluations move GLM ≥ 5 index points ahead (e.g. a GLM-5.3-Flash update) | Re-weigh base vs compression gap |
| A REAP-50 DeepSeek-V4.1-Flash with an MLX-loadable config and published KL | Evaluate it as a P3 candidate (AA 45) |
| Wiring/IOGPU panic observed above ~80 GB | Drop P2 to ≤ 85 GB process; GLM resident builds become K ≤ 160 |

---

## 6. Failure modes

1. **Optimizing the wrong budget.** Treating disk size as the limit leaves ~23 GB unused (PLE case); treating
   resident size as the limit while forgetting KV, APC and prefill transients overshoots the 80 GB cap.
   `mx.set_memory_limit` is soft and **never refuses an allocation** (`mac.md` §2), so `serve/` must do admission control.
2. **PLE page-cache eviction.** Under heavy dev-tool memory pressure macOS drops clean PLE pages first. Each token then
   reads its n-gram rows (a few KB) from SSD. That costs little at ~10 GB/s, but measure p99 token latency under
   pressure (`memory_pressure -l warn` while running `evals.speed`).
3. **Leaderboard over-trust.** TB 4 gives opposite winners under the AA and NVIDIA harnesses (§3.1). Vendor tables use
   best-of-harness (Qwen DeepSWE) or a specific Claude Code version (GLM TB 2.1). Decide on our harness.
4. **Comparing KL across models.** Each model's KL is against its own reference. A smaller KL for Qwen than for GLM
   does not by itself mean Qwen is better. That is what bits-per-byte and task evals are for.
5. **Small-n conclusions.** 164-item suites cannot separate 3–5-point gaps (§5.1). Every published HumanEval ladder in
   `reap.md` is within noise.
6. **Over-pruning loops.** Pruned models loop for thousands of tokens (GLM at 120 experts). Set `max_output_tokens`
   and a repetition watchdog in `serve/`, and log turns-to-solve.
7. **Streaming thrash.** If the hit rate is below ~0.4, GLM P3 decodes at ~3 tok/s and prefill makes agent use
   impossible. Run the step-6 simulation first. Also, the stock `ExpertStore` default budget (~80 % of max working
   set) is set for a dedicated machine. **Always pass `--expert-cache-gb` explicitly.**
8. **GLM memory traps.** The projected MLA cache (+22–24 GiB at ≥ 32K), the 8 GiB prefill transient at 128K, and
   `--kv-bits` crashing under continuous batching (`decode.md`). Never serve GLM without the fixes.
9. **Two MLX processes.** Running the embedder or a second model in a separate MLX process risks the IOGPU panic class
   (`mac.md`). Keep everything in one server.
10. **Thinking-mode mismatch.** GLM defaults to `reasoning_effort=max` and `clear_thinking=false`. Qwen preserves
    thinking from all past turns by default. These change token counts (time-to-answer) and KV growth across a long
    agent session. Set them explicitly per served name.
11. **License drift.** Qwen3.8-Flash-Next is under `qwen-community-1.0` (not Apache). GLM-5.3-Flash and
    DeepSeek are MIT, Qwen3.8-27B is Apache-2.0, Nemotron is OpenMDW-1.1, and Mistral-Medium-3.5 is "modified MIT".
    This is fine for personal use, but note it before sharing builds.
12. **Personal-data leakage.** Calibration manifests and eval artifacts derived from `~/.claude/projects` can contain
    secrets. Never publish them or the `README.md` of a build that embeds them.

---

## 7. Stretch ideas (push the limits)

- **S1. Unpruned Qwen at 4.5 bpw in P1 by streaming *cold experts* too.** Keep the experts with the most routing mass
  resident at 4–5 bits and stream the cold ~30 % from SSD at 3 bits. Qwen's 512-way routing is far more skewed per
  domain than GLM's, and calibration measures `expert_freq` per domain. Each Qwen expert is only 2.1–2.8 MB, so misses
  are cheap in bytes but latency-bound (0.77 ms single-thread). That means it needs an **async prefetcher**: compute
  layer *l*+1's router on layer *l*'s output (pre-gating, [2308.12066](https://arxiv.org/abs/2308.12066)) and fetch
  ahead. Implement as a subclass of `OffloadedSwitchGLU` with a pinned "hot set" and a thread pool reading via
  `pread`.
- **S2. Per-expert precision by frequency.** Split each layer's `SwitchGLU` into a hot module (8-bit or 6-bit) and a
  cold module (3-bit) and remap router indices. Stock MLX has one (bits, group) per stacked tensor (`quant.md` §1), so
  this needs a loader shim, the same one non-uniform REAP needs (`reap.md` §5).
- **S3. Best-of-n with test selection on the latency-bound primary.** Qwen decode is launch-bound, so 4 parallel
  samples cost far less than 4× (measure the batch-4 slowdown with `evals.speed`). For coding tasks, sample n=4, run
  the tests in `evals/sandbox.py` and return the first passing patch. This is the cheapest quality multiplier
  available on this machine and works in P1.
- **S4. Make SSD streaming bandwidth-efficient.** Replace `ExpertStore._ensure_loaded`'s per-expert `mx.load` with a
  thread pool of `pread` calls into preallocated Metal-shared buffers. Measured: 16 threads read 11 MB chunks at
  13.6 GB/s vs 7.9 GB/s with one, which is +70 % decode for GLM P3.
- **S5. Cross-model "consult" mode.** For a hard question in P2, run Qwen (fast) and then have GLM (streamed, short
  prompt) critique only the final answer and plan. This exploits GLM's lower hallucination rate where it matters, at
  ~2–5K GLM tokens, i.e. a few minutes at 5–10 tok/s.
- **S6. Distill a domain-specialized MTP / DFlash drafter** on the user's own transcripts (rendered in Qwen's format).
  Acceptance rises when the drafter has seen the user's code style. Lossless for quality; pure speed.
- **S7. HOPE instead of REAP for any pruned build.** HOPE's gain is concentrated in agentic tasks at 40–50 % pruning
  (`reap.md` rec. 9). Log the co-routing matrix `F_ij` during calibration now, so it is a build-time option for the
  GLM P2 builds.
- **S8. Watch DeepSeek-V4.1-Flash.** It has the highest index (45), a 196B Engram table that is mmap-friendly like
  Qwen's PLE, and CED decoding (8B active in prefill). If a REAP-50 build appears, P3 streaming at ~120 GB of experts
  and 15.5 MB per expert becomes testable with the same simulator.

---

## Sources

Primary model cards (fetched 2026-10-20): [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(benchmark chart `bench_53.png`), [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash),
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (+ `-1017` config.json),
[nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
(NeMo Gym cross-evaluation table),
[mistralai/Mistral-Medium-3.5-128B](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B),
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2),
[nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16),
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
[pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF).

Independent evaluation: Artificial Analysis Intelligence Index v4.3.2:
[open-weights list](https://artificialanalysis.ai/models/open-source),
[GLM vs Qwen-FN](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-qwen3-8-flash-next),
[GLM vs DSV4-Flash](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-deepseek-v4-flash),
[GLM vs Qwen3.8-27B](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-qwen3-8-27b-low),
[Nemotron 3.5 article](https://artificialanalysis.ai/articles/nemotron-3-5-super-and-bolt),
[Mistral Medium 3.5](https://artificialanalysis.ai/models/mistral-medium-3-5),
[Qwen3.6-35B-A3B](https://artificialanalysis.ai/models/qwen3-6-35b-a3b).
MathArena ([AIME 2026](https://matharena.ai/?comp=aime--aime_2026)): neither model listed.
[SWE-bench](https://www.swebench.com/) bash-only: neither model found.

Papers: [Dettmers & Zettlemoyer 2212.09720](https://arxiv.org/abs/2212.09720) ·
[Kumar et al. 2411.04330](https://arxiv.org/abs/2411.04330) · [Ouyang et al. 2411.17691](https://arxiv.org/abs/2411.17691) ·
[MoQE 2310.02410](https://arxiv.org/abs/2310.02410) · [QMoE 2310.16795](https://arxiv.org/abs/2310.16795) ·
[VBQ 2607.02893](https://arxiv.org/abs/2607.02893) · [Composable compression 2608.21693](https://arxiv.org/abs/2608.21693) ·
[Eliseev & Mazur 2312.17238](https://arxiv.org/abs/2312.17238) · [Pre-gated MoE 2308.12066](https://arxiv.org/abs/2308.12066) ·
[MoE-Infinity 2401.14361](https://arxiv.org/abs/2401.14361) · [FrugalGPT 2305.05176](https://arxiv.org/abs/2305.05176) ·
[RouteLLM 2406.18665](https://arxiv.org/abs/2406.18665) · REAP/HOPE/REAM per `docs/research/reap.md`.

Installed source: `mlx_vlm/moe_offload.py` (`ExpertStore`, `_default_expert_cache_bytes`, repack CLI),
`mlx_vlm/models/switch_layers.py` (`OffloadedSwitchGLU`), `mlx_vlm/server/cli.py` (`--expert-cache-gb`),
`mlx_vlm/models/qwen4_exp/ple_storage.py`, repo `build/pipeline.py` (PLE counted in `--budget-gb`).

Measured here: SSD random-read table (§2.5), with a throwaway script in the session scratchpad (`ssdbench/ssd.py`,
16 GiB file of unique random data, `F_NOCACHE`, deleted afterwards). Transcript corpus size: `find`/`du` metadata only.
