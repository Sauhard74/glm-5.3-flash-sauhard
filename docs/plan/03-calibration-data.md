# 03 — Calibration data: making the model YOURS

> Part of the master plan for the best co-tenant local model on this M5 Max (128 GB, 614 GB/s, ≤ 80 GB process).
> This chapter **builds on** `PLAN.md`, `docs/research/reap.md` §4 (calibration-domain evidence, verified),
> `docs/research/quant.md` (imatrix / DWQ), `docs/contracts/calib_data.md` and `docs/contracts/calib.md`, and it
> must agree with its siblings `01-memory.md` (§4.7 personal hot sets) and `04-pruning.md` (§4.2 data, §4.3 cost,
> §7.8 self-generation). Where it changes something they say, it says so explicitly.
>
> Tags: **[paper]** arXiv / peer-reviewed · **[src]** read in this repo or in installed package source ·
> **[hf]** checked against the Hugging Face API on 2026-10-20 · **[self-reported]** a model card / README number,
> not reproduced · **[computed]** arithmetic from real configs · **[estimate]** an engineering guess with its
> arithmetic shown · **[hypothesis]** a proposal without published evidence; §5 says how to measure it.
>
> Privacy note for this chapter: the user's Claude Code transcripts were **not read**. Only file counts and sizes
> were taken (`find`/`stat`). Everything said about their format is generic and has to be confirmed by the
> key-only probe in §4.4 step P0, run by the user.

---

## 0. TL;DR (what this chapter decides)

1. **The calibration set is the spec for the compressed model.** REAP picks which experts survive, the NMSE grid
   and imatrix pick which tensors and channels get bits, and DWQ picks where the remaining fidelity goes. All three
   are *expectations over the calibration distribution*. If that distribution differs from your real traffic, the
   compressed model is optimised for someone else's workload. The evidence is strong and consistent: C4
   calibration drove REAP code accuracy to 0 % on some models **[paper]**, chat-only masks lost coding
   **[self-reported]**, and GPTQ on WikiText vs on math traces differs by 9.8 points on reasoning benchmarks **[paper]** (§3).
2. **Keep the existing `calib_data/` builder** (11 recipes, deterministic, hash-split, ~2.3 M calib tokens). Extend it in
   five ways (§4.3):
   - add two domains, `personal` (your Claude Code sessions) and `general` (a 5 % floor);
   - add four agentic sources verified on HF: SWE-smith, Nemotron-SWE, SWE-rebench-OpenHands and Nemotron-Agentic, plus Terminal-Corpus;
   - extend `longctx` to **32K–128K**;
   - add a third split, `dwq`;
   - split by **content hash**, so the same problem from two sources cannot land on both sides, and add a
     **decontamination** pass against every benchmark that `evals/` uses.
3. **Self-generate the assistant side.** Calibration statistics are always computed by teacher-forcing the *full*
   model, but the *text* it is forced through should be text that model would itself write: its `<think>` style,
   its tool-call habits, its answer lengths. Recipe:
   - prompts come from the existing sources;
   - responses come from (a) **rented GPUs running the full-precision model** (~1.5 h of 8×H100 per model, **≈ $25–50 at
     neo-cloud prices, ≈ $80–150 at hyperscaler prices**) for public prompts, or (b) **the local 4-bit proxy** for
     private prompts (~2 nights for 6 M tokens) with a full-model NLL filter;
   - agentic data uses **one-step regeneration**: real history, then the model's own next action.
4. **Your sessions are the most valuable data, and the most dangerous.** The set is large: 1,614 top-level session
   files + 1,608 subagent files, 5.8 GB of JSONL **[measured: metadata only]**. A local-only pipeline handles it:
   - parse → scrub (regex + entropy + path/identity rewriting) → dedup → per-project cap → render with each model's
     template → **time-based** held-out split;
   - every artifact derived from it, including `calib.jsonl`, `ref/` and calib `work/`, is treated as secret;
   - none of it is ever sent to a cloud GPU.
5. **Token budgets differ by consumer** (§2.5):
   - **REAP**: ≥ 1 M tokens per domain for the final mask, 250K per domain for screening (expert tail coverage drives this);
   - **sensitivity/imatrix**: ≥ 512 routed rows per expert, which the REAP budget already gives;
   - **DWQ**: a separate ~2 M-token split, disjoint from KL eval;
   - **KL eval**: ≥ 50 samples per domain (~1 M tokens, as now).
6. **Long sequences are handled correctly already.** `calib/` pushes each sample chunk by chunk through each layer with
   that layer's own cache, and the result is bit-identical to chunked prefill **[src]**. What has to change is *sampling*:
   - position-stratified reservoirs, so the rare deep-context rows are actually sampled for sensitivity;
   - domain weights in `build`, so 128K samples neither dominate nor vanish.

---

## 1. Why it matters (intuition)

Compression asks one question: *what can this model lose that you will not miss?* A calibration set is how you tell
the compressor what "you" means.

Think of a 288-expert GLM layer as a newsroom of 288 specialists, with an editor (the router) who hands each word
to 8 of them. Load balancing makes every specialist busy *on average over the pre-training mix*, and OpenMOSE found
no idle experts at all **[self-reported, reap.md §4]**. Now you have to fire half of them. If you watch the
newsroom for a day of *sports* coverage, you will keep the sports desk and fire the people who write about code.
Nothing in the method is wrong; you asked the wrong question. That is the C4 → 0 % code result in the REAP paper,
and the chat-only mask that "lost most of its coding ability".

Quantization has the same structure at a finer grain:
- **Which tensors get bits.** `calib/sens.py` measures `NMSE = Σ‖X(Ŵ−W)ᵀ‖² / Σ‖XWᵀ‖²` on *captured inputs X*. A
  tensor that is fragile only on inputs your workload never produces looks robust, and loses bits.
- **Which channels are protected.** AWQ and imatrix-weighted clipping weight each input channel by `E[x_c²]`.
  Outlier channels are real and systematic ([LLM.int8()](https://arxiv.org/abs/2208.07339) **[paper]**), but *which*
  channels are hot depends on the text. Code (indentation, brackets, identifiers) lights up different channels than
  prose.
- **Where an expert gets calibrated at all.** In an MoE, an expert's imatrix comes only from tokens routed to it.
  Calibrate on prose and the code experts get a handful of rows. Their clip search is then fitted to noise, or
  falls back to the pooled layer statistic (quant.md rec. 3). [MoEQuant](https://arxiv.org/abs/2505.03804) calls
  this *inter-expert imbalance* and shows it is a main cause of MoE PTQ damage **[paper]**.
- **What DWQ optimises.** DWQ minimises KL to the teacher *on the calibration text*. It cannot fix what it does not see.

Two consequences shape the rest of the chapter:

1. **Domain is necessary, but not sufficient.** Even inside "agentic coding", text written by *another* model (Claude
   in your transcripts, Claude/GPT-4o in SWE-smith, Qwen3-Coder in Nemotron-SWE, DeepSeek-R1 in OpenR1) is not what
   *this* model will produce. During decoding, every token's hidden state is computed from a prefix that is partly
   the model's own output. Calibrating on someone else's output measures the experts and channels that would be
   active *if this model wrote like Claude*. Hence self-generation (§2.4).
2. **Your own sessions are the closest thing to the future.** In a Claude Code loop most processed tokens are
   *prefill*: system prompt, tool schemas, your files, your build logs. Those inputs are exactly what the model will
   see, even when the assistant turns were written by Claude (01-memory §4.7). In-domain session calibration gave
   ×1.21 turns-to-solve vs ×2.29 for a generic 7-domain mask on GLM-5.3-Flash at K=144 **[self-reported, lapadatbogdan]**.

---

## 2. How it works (mechanics + key math)

### 2.1 Every compression decision is an expectation over the calibration distribution

Write the calibration distribution as a domain mixture `D_cal = Σ_d w_d D_d`, and the deployment distribution (your
real traffic) as `D_dep`.

**Pruning (REAP).** For expert j in a layer, with gate `g_j(x)` and output `f_j(x)` (04-pruning §2.3):

$$S_j(D)=\mathbb{E}_{x\sim D}\big[\,g_j(x)\,\lVert f_j(x)\rVert_2 \;\big|\; j\in T(x)\big]$$

The mask keeps the top-K by `S_j(D_cal)`. The quantity that should drive it is the substitution error under
`D_dep`. An expert is wrongly pruned when its rank under `D_cal` falls below the cut but its rank under `D_dep` would
not. The size of that error grows with the divergence between the two distributions *restricted to tokens near the
decision boundary*. Changing `D_cal` does not add noise to the answer; it changes the answer.

`calib/` computes the domain-balanced version **[src, calib contract]**:

$$S_j=\frac{\sum_d (\sum g\lVert f\rVert)_{j,d}/T_d}{\sum_d \mathrm{count}_{j,d}/T_d}$$

so each domain present in `calib.jsonl` gets equal weight, whatever its token count. The weights `w_d` are applied
later, by `build.make` through `calib_io.combine_domains(..., domain_weights)` **[src]**. **Token count per domain
therefore sets precision, and `--domain-weights` sets importance.** Keep the two separate (§4.6).

**Quantization (GPTQ / AWQ / imatrix / NMSE grid).** The layer-local objective is

$$\mathbb{E}_{x\sim D}\lVert(W-\hat W)x\rVert^2=\operatorname{tr}\!\big((W-\hat W)\,H_D\,(W-\hat W)^\top\big),\qquad H_D=\mathbb{E}_{x\sim D}[xx^\top]$$

- The imatrix is `diag(H_D)` (`act_sqmean` in calib).
- AWQ's salience is `E|x_c|` (`act_absmean`).
- The NMSE grid is this objective normalised by `E‖Wx‖²`.

Every method sees the data only through `H_D`, its second-moment matrix. For a routed expert, `D` is the routed
sub-distribution `D | j∈T(x)`. Your mix decides both *what* `H` looks like and *how many rows* estimate it.

**Methods differ in how much they lean on `H_D`.**
- RTN uses no data.
- AWQ uses one statistic per channel, plus a small grid search over scales. It is robust to domain shift: Liu et al.
  found AWQ insensitive to the calibration domain on reasoning models **[paper, 2504.04823 §4.2]**.
- GPTQ inverts `H_D` and *compensates* each quantisation error with the other columns, so it overfits the
  domain: WikiText → Numina-Math calibration gave +9.8 points averaged over reasoning benchmarks **[paper, same]**.
- This project uses imatrix-weighted clipping + LSQ refit (quant.md rec. 3). That sits between AWQ and GPTQ in how
  much it depends on the data: a weighted least-squares fit of scales and biases per group. Expect moderate domain
  sensitivity, and measure it (§5, experiment D1).

**DWQ.** `min_θ E_{x∼D} KL(p_T(·|x) ‖ p_θ(·|x))` over affine scales and biases. This is *training*, with billions of
trainable parameters (GLM K=144 at g64: 152 B expert params × 2/64 ≈ 4.8 B scale/bias values **[computed]**), so it
can overfit the calibration text, including benchmark text if any leaks in (§2.7).

### 2.2 What "domain" means at the token level

A sample is a chat-templated sequence. Its tokens come from very different processes:

| Region | Generated by | Share of tokens (current builder) | What it calibrates |
|---|---|---|---|
| System prompt, tool schemas | operator / framework | 2–15 % | template- and tool-format experts (seen every request) |
| User turns, files, tool results | humans, the environment | 40–90 % (agentic, longctx) | **prefill** routing, which dominates agentic compute |
| Assistant `<think>` + content + tool calls | *some model* | 10–60 % (reasoning ~80 %) | **decode** routing, where tokens per second are spent |

**[estimate]** These shares come from the recipe structure and the manifest means (e.g. openhands mean 7.1K tokens
with 12K-character tool outputs). Once self-generated spans exist, measure them with the `gen_spans` field (§4.2).

The third row is the problem. In the current builder, assistant text comes from DeepSeek-R1 (OpenR1, OpenThoughts),
Kimi-K2 (Toucan), Claude/GPT-4o (SWE-Gym OpenHands) and Wikipedia-style abstracts (longctx). None of it is
GLM-5.3 or Qwen3.8. In reasoning samples this region is ~80 % of the tokens.

### 2.3 The chat template is part of the distribution

The model will only ever see text in its own template, rendered with the kwargs the server uses. The builder
already renders with each model's own `chat_template.jinja` **[src, render.py]**. Three template details matter, and
calibration must match the **server's** choices for them:

- **Reasoning effort.** GLM's template injects `<|system|>Reasoning Effort: Max` unless `reasoning_effort ∈ {low, high}`.
  Qwen's defaults to `xhigh` **[src, artifacts/tokenizers/*/chat_template.jinja]**. These change the system prompt,
  and they change the length and style of the thinking that follows.
- **Thinking in history.** GLM keeps `reasoning_content` of past turns unless `clear_thinking=true`. Qwen keeps it
  unless `preserve_thinking=false` **[src]**. With Claude Code on the local server, whether past thinking comes back
  depends on how `serve/app.py` translates Anthropic thinking blocks. **Verify this, then render calibration the
  same way.** If the server drops history thinking but calibration keeps it, multi-turn calibration samples
  contain token patterns that never occur in deployment.
- **Tool syntax.** Qwen `<tool_call><function=…><parameter=…>`, GLM `<tool_call>name<arg_key>…<arg_value>…`, tool
  results in `<tool_response>` (Qwen in a user turn, GLM after `<|observation|>`). The builder already renders these
  natively from OpenAI-style `tool_calls` with dict arguments (contract).

### 2.4 Self-generated calibration: why, and what it can and cannot fix

**Mechanism.** Let `p` be the full-precision model, and `q` whatever wrote the assistant region (R1, Claude, a 4-bit
proxy). Calibration statistics at position t are functions of the hidden state `h_t(x_{<t})`. At deployment, the
assistant-region prefixes are drawn from `p` itself. Calibrating on `q`-text estimates expectations under the wrong
prefix distribution, and the error grows with `KL(q‖p)` over the assistant region. This is the exposure-bias argument
that motivates on-policy distillation ([GKD](https://arxiv.org/abs/2306.13649), [MiniLLM](https://arxiv.org/abs/2306.08543)
**[paper]**), applied to statistics instead of gradients.

**Evidence that it matters for compression:**
- [Self-calibration](https://arxiv.org/abs/2410.17170): model-generated calibration data is "consistently competitive …
  frequently outperforming even using real data" for quantization and pruning **[paper]**.
- [Beware of Calibration Data](https://arxiv.org/abs/2410.17711): calibration data closer to the training distribution
  prunes better, and a self-generation strategy gains up to 2.68 % **[paper]**.
- [MoEQuant EBSS](https://arxiv.org/abs/2505.03804): self-sampling, steered toward expert balance, gave > 10 points
  HumanEval on DeepSeek-MoE-16B at 4-bit **[paper]**.
- Liu et al. self-generate the reasoning traces they calibrate on, specifically "to mitigate the effects of domain
  shift" **[paper, 2504.04823 App. A]**.
- Community MoE pruning of *these* models: lapadatbogdan used sessions the full GLM wrote itself; Litwein distilled
  on bf16 teacher completions **[self-reported, reap.md §4]**.
- There is **no controlled A/B on these two models** (reap.md verification row 4). Experiment D3 (§5) is the A/B.

**What it cannot fix.** Self-generation changes the *text*. The statistics are still computed by the *full* model in
`calib/`, teacher-forced. So:
- the teacher does not have to fit on the Mac. Only the text does, and text is cheap to move.
- a **proxy** (the deployed 4-bit REAP-288 Qwen, or a first-generation GLM build) is acceptable only to the extent
  that its text is close to `p`'s. Its specific failure is **circularity**. A pruned proxy writes less of what its
  pruned experts were good at (other languages, rare entities). The full model's stats on that text then rate those
  experts lower, and the next mask prunes them again. Guard: score every proxy-generated span with the *full model's*
  per-token NLL during calibration (calib already computes `target_logprob` for eval samples; extend it to calib
  samples, which is cheap). Then drop the worst decile, and keep a public-source floor (§4.5).
- **Agentic trajectories need an environment.** A self-generated tool call needs a real tool result before the next
  turn. Running SWE-Gym/R2E docker sandboxes is out of scope for v1. Use **one-step regeneration** instead: take a
  real trajectory, cut it at a random assistant turn, keep the real history, and let the model generate *its own*
  next thought + tool call. That generated step is the sample's final turn. The prefill side stays real, and the decode
  side becomes on-policy. **[hypothesis]** that this captures most of the gain.

### 2.5 How many tokens each consumer needs (with the reasoning)

**REAP saliency, per expert per domain.** The routed tokens per expert are `n_j = T_d · k / E · (f_j / f̄)`:
- GLM: `k/E = 8/288 = 1/36`; Qwen: `10/512 = 1/51.2` **[computed]**.
- Tail experts get ~10 % of average traffic (1.4 % of GLM experts are below that, reap.md §4).
- The standard error of a mean of `g‖f‖` with coefficient of variation `c` (assume `c ≈ 1` **[hypothesis]**) is
  `c/√n_eff`, where `n_eff = n/deff`. The design effect `deff` comes from correlation between tokens of the same
  sequence; assume ~10 **[hypothesis]**. Measure both with the sequence bootstrap.

| tokens / domain | GLM avg expert n | GLM tail n → SE | Qwen avg n | Qwen tail n → SE |
|---|---|---|---|---|
| 250K (screening) | 6.9K | 694 → **12 %** | 4.9K | 488 → **14 %** |
| 1 M (final, minimum) | 27.8K | 2.8K → **6 %** | 19.5K | 1.95K → **7 %** |
| 2 M (reap.md "ideal") | 55.6K | 5.6K → 4 % | 39K | 3.9K → 5 % |

Why tail precision matters: the experts near the cut at K=144/288 or K=320/512 are, almost by definition, the
low-traffic ones. Their saliency has the widest error bars, so they are where the bootstrap "uncertain band" lives
(04-pruning §4.2). OpenMOSE's "8× more data changed 8 % of the selection" **[self-reported]** fits this picture: more
data mostly reshuffles the boundary. **Content matters more than volume beyond ~1 M per domain.** This matches 04's
1 M-token screening corpus (250K/domain) and 4 M-token final corpus.

**Sensitivity grid and imatrix.**
- *NMSE grid.* NMSE is a ratio of two sums over the same rows, so it converges fast. A few hundred rows rank the bit
  levels reliably, because each bit cuts NMSE by roughly 4× while row noise is ~10 %. The knapsack compares tensors
  with each other, and ~10 % noise in each tensor's estimate costs little at 1–2k items. `calib` samples
  `--lin-tokens 2048` rows per dense projection and `--expert-tokens 128` per expert **[src]**.
- *Per-channel imatrix.* This needs more rows than the grid, because each of the 2,048–4,096 channels needs its own
  stable `E[x_c²]`. `gate/up` statistics are exact over *all* routed tokens. `down_proj` statistics come only from the
  128 sampled rows.
- *Change:* raise `--expert-tokens` to **512 for the final pass**. Memory: GLM `[288, 512, 4096]` bf16 = 1.2 GB,
  Qwen `[512, 512, 2560]` = 1.3 GB per projection buffer **[computed]**, well under the 40 GB cap. The REAP budget
  (≥ 1 M/domain → ≥ 2K routed tokens even for tail experts) already guarantees 512 rows exist.

**DWQ.** This is a training set, so the questions are overfitting and coverage, not estimator variance.
- mlx-lm defaults to 2,048 samples × 1,025 tokens ≈ **2.1 M tokens** from `allenai/tulu-3-sft-mixture` **[src, mlx_lm/quant/dwq.py]**.
- Use **~2 M tokens of workload text in its own `dwq` split**, disjoint from `calib` (whose stats chose the
  quantisation; reusing them double-counts) and from `eval` (the acceptance gate, which must stay unbiased).
- Lengths: mostly 2–8K, plus ~10 % at 32K. Band-wise or layer-local DWQ on cached hidden states (quant.md rec. 5) can
  afford long sequences, and long-position fidelity matters for RAG.
- Teacher targets: top-64 logprobs + tail bucket, the same format as `ref/`, at 64 × 8 B = **512 B/token → ~1 GB for
  2 M tokens**. mlx-lm's top-1024 targets would be 16 GB **[computed]**.

**KL eval.**
- The current eval split is ~1.0 M tokens over 200 samples.
- The KL CI is bootstrapped over *samples* (04 §5). ≥ 50 samples per domain gives a usable CI. `longctx` has 12
  samples, so it gets a wide CI. Accept that, and back it with the `evals.longctx` RULER suite.
- Add a `personal` eval domain: time-split, ~60 samples.

**Summary budget** (calib tokens; ×¼ for screening):

| domain | calib (final) | eval | dwq | why this size |
|---|---|---|---|---|
| code | 0.8 M | 0.15 M | 0.4 M | ≥ 1 M/domain is the REAP rule; code is the most tolerant domain (04 §3.3), so 0.8 M |
| agentic (public) | 1.0 M | 0.2 M | 0.5 M | the most fragile domain (Litwein agent KL worst); several sources |
| personal | 0.8 M | 0.15 M | 0.3 M | the deployment proxy; capped so one person's repos don't own the mask |
| reasoning | 0.8 M | 0.15 M | 0.4 M | thinking-heavy decode; self-generated |
| longctx (32K–128K) | 0.8 M (~12 samples) | 0.4 M (~6) | 0.2 M | positions > 32K are a different regime (§2.6) |
| general (floor) | 0.2 M | 0.05 M | 0.1 M | arithmetic / prose / multilingual floor (reap.md rec. 3); low build weight |
| **total** | **4.4 M** | **1.1 M** | **1.9 M** | |

### 2.6 Long sequences in layer-wise calibration

**What `calib/` already does right** **[src, calib contract + run.py]**:
- every sample goes through the real decoder layer chunk by chunk (`--chunk 2048`, tail ≥ 64), with that layer's cache:
  KV / MLA latent, linear-attention state, DSA/QSA indexer state;
- the streamed reference logprobs are bit-identical to a full mlx-vlm forward with chunked prefill;
- a 128K sample is therefore calibrated exactly as the server would prefill it, with no truncation to 2K, which is
  the usual shortcut.

**Cost of long samples** **[computed]**:
- Hidden-state disk per boundary: GLM 4 streams × 4096 × 2 B = 32.8 KB/token, so a 128K sample = **4.3 GB**. Qwen
  4 × 2560 × 2 B = 20.5 KB/token → 2.7 GB. Resident while that sample is in flight: input + output ≈ 8.6 GB (GLM).
  That is inside the 40 GB calib cap next to a ≤ 15 GB GLM layer, but **only one 128K sample may be in flight per
  layer**. The batching speedup that 04 §4.3 proposes must not pack long samples.
- DSA indexer: O(L²). 04 estimates ~5–10 s per 64K sample, so ~20–40 s at 128K (4×). With 12 samples per pass this is
  negligible.

**What has to change, and why:**
1. **Position-stratified reservoirs.** Sensitivity rows are drawn uniformly per domain (`Reservoir`, `--lin-tokens`).
   Inside `longctx` the rows past 32K are a minority. They are also the rows whose activation statistics differ most,
   because full-attention layers' inputs shift with position (RoPE-rotated outlier channels: transforms learned on
   2K-token calibration "are unable to generalize to longer sequences" **[paper, 2504.04823 App. C]**). Change: bucket
   reservoir offers by position, `[0,4K) / [4K,32K) / [32K,∞)`, with equal quotas inside `longctx` (`calib/probe.py`).
2. **Raise the range to 32K–128K** (from 16K–64K). The workload is Claude Code at up to 128K context
   (`CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072` in `scripts/claude-local.sh` **[src]**) and long-document RAG. Both models
   handle 128K natively (Qwen 262K, GLM 1M `max_position_embeddings` **[src, config.json]**). 04 notes nobody has
   published long-context-calibrated masks for these models, so this is new ground. Keep 16K–32K covered through
   `personal` and agentic samples, which naturally run 8–40K.
3. **Domain weighting, not token weighting.** 12 samples × 64K = 0.8 M tokens would dominate a token-weighted mix.
   calib's domain balancing already prevents that. Give `longctx` a build weight of about 0.7 (§4.6).
4. **Whole-sample units.** Never chunk a long sample into independent 2K samples. That would reset the cache and
   destroy exactly the long-range state being measured.

### 2.7 Contamination: three leaks, three different harms

1. **Calib ↔ `eval.jsonl` leak.** It inflates the KL gate, and it biases the comparison between variants. The current
   builder splits on `sha1(source:row_key) % 1000 < 300` **[src]**. That is safe *within* a source but not *across*
   sources: OpenR1-Math and OpenThoughts both derive from NuminaMath, so the same problem can sit in calib under one
   source and in eval under another. GitHub files also recur between `github-code-clean` and `commitpackft`. **Fix:**
   compute the split key from **normalised content**, e.g. `sha1(norm(first user message)[:2000])` for prompt
   datasets and `sha1(repo_name)` for code. Then a given problem lands on one side of the split whichever source it
   came from.
2. **Calib/DWQ ↔ task-benchmark leak** (HumanEval+, MBPP+, AIME24/25, MATH-500, BFCL, which are what `evals/` uses
   **[src, evals/data.py]**). Pruning and quantization have no explicit memorisation, but they *preserve whatever the
   calibration text exercises*. Calibrating on AIME problems preserves AIME-specific capacity, so the eval says the
   mask is good for math while it was really tuned for those 30 problems. DWQ is training, so for DWQ this is genuine
   memorisation risk. Known exposure:
   - NuminaMath (via OpenR1, Mixture-of-Thoughts) includes competition-math sources (`amc_aime`, olympiads);
   - GitHub crawls contain copies of HumanEval;
   - tool-calling sets (xlam, Toucan, Nemotron "tool sets from publicly available datasets") can share function
     schemas with BFCL.
   **Fix:** a 13-gram overlap filter (the convention since GPT-3), plus exact match on normalised problem statements
   and on function names + descriptions, run against all benchmark items. Log every drop in `manifest.json`.
3. **Personal eval leak.** Sessions are highly self-similar: the same repo files are re-read and the same commands
   re-run. A random split by session still puts near-copies on both sides. **Fix:** a time split (all sessions after a
   cutoff go to eval), plus near-duplicate removal *across* the split boundary (§4.4).

---

## 3. Evidence

### 3.1 Calibration domain changes pruning outcomes

| Source | Setting | Finding |
|---|---|---|
| [REAP, arXiv:2510.13999](https://arxiv.org/abs/2510.13999) **[paper]** | Qwen3-30B, ERNIE, others; C4 vs domain calibration | Fine-grained MoEs "fail to produce any coherent output when calibrated on C4" on coding (Fig. A8, 0 %). Domain-specific calibration wins, and REAP is the most robust method under a mixed set (Fig. A9). The paper's agentic recipe: evol-codealpaca, xlam-function-calling, SWE-smith-trajectories, + Mixture-of-Thoughts for reasoning models, ≤ 16,384 tokens |
| OpenMOSE, GLM-5.3-Flash **[self-reported, reap.md §4]** | chat vs code saliency | Spearman 0.714; chat-only mask lost coding; code-only lost Japanese; 8× data changed 8 % of selection |
| AnonimousA, Qwen3.8-Flash-Next K=256 **[self-reported]** | code-only → multi-domain corpus, same bits | HumanEval 89.6 → 95.1 % |
| lapadatbogdan, GLM K=144 **[self-reported]** | own-session vs public 7-domain mask | 48/48 at ×1.21 turns vs 44/46 at ×2.29. Python-only calibration broke other languages |
| patrickbdevaney, GLM REAP50 **[self-reported]** | 7 domains, 2,048-token sequences | top-1 agreement math 0.920, code 0.919, agentic 0.873, "ballast" 0.580; long context "unassured" |
| [Williams & Aletras, arXiv:2311.09755](https://arxiv.org/abs/2311.09755) **[paper]** | dense LLMs, many PTQ/pruning methods | "substantial variations in downstream task performance" across calibration sets, against prior claims of robustness |
| [Bandari et al., arXiv:2410.07461](https://arxiv.org/abs/2410.07461) **[paper]** | pruning calibration study | C4 not optimal; arithmetic data on par or better; downstream-task data does *not* necessarily help its own task |
| [HOPE, arXiv:2609.18916](https://arxiv.org/abs/2609.18916) **[paper]** | pairwise co-activation | Gains in the agentic regime at 40–50 % pruning; uses the same calibration pass, so it inherits that pass's domain bias |

### 3.2 Calibration domain changes quantization outcomes

| Source | Finding |
|---|---|
| [Quantization Hurts Reasoning?, arXiv:2504.04823](https://arxiv.org/abs/2504.04823) **[paper]** | GPTQ calibrated on WikiText2 → Numina-Math-1.5 (self-generated traces): **+9.8 % average accuracy**. AWQ, KVQuant and FlatQuant are robust to the domain. Smoothing learned on short (2K) sequences does not generalise to long ones |
| [AWQ, arXiv:2306.00978](https://arxiv.org/abs/2306.00978) **[paper]** | Channel salience must come from activations, not weights: protecting 1 % of channels cuts error greatly. Claimed robust to calibration distribution because there is no reconstruction |
| [MoEQuant, arXiv:2505.03804](https://arxiv.org/abs/2505.03804) **[paper]** | WikiText/C4 calibration gives a long-tailed sample count per expert; under-calibrated experts get large errors. Expert-balanced *self-sampling* fixes it (> 10 pts HumanEval, DeepSeek-MoE-16B, 4-bit) |
| [QuantMoE-Bench, arXiv:2406.08155](https://arxiv.org/abs/2406.08155) **[paper]** | Shared vs routed experts need different bits because of activation frequency, which is a property of the calibration data |
| [Marchisio et al., arXiv:2407.03211](https://arxiv.org/abs/2407.03211) **[paper]** | Quantization hurts non-Latin languages most, and automatic metrics underestimate it (1.7 % automatic vs 16 % human-judged drop in Japanese). This is the reason for the `general` multilingual floor |
| Litwein REAP320 + DWQ **[self-reported]** | Per-domain KL20 after DWQ: agent worst (0.185), code best (0.060). The domain mix decides where DWQ capacity goes |

### 3.3 Self-generated calibration

| Source | Finding |
|---|---|
| [Self-calibration, arXiv:2410.17170](https://arxiv.org/abs/2410.17170) **[paper]** | Synthetic data from the model itself is consistently competitive with, and often better than, real data, for both quantization and pruning |
| [Beware of Calibration Data, arXiv:2410.17711](https://arxiv.org/abs/2410.17711) **[paper]** | Data similar to the training distribution prunes better. Self-generated synthesis gains up to +2.68 % (Wanda / DSnoT / OWL). A small amount of data is enough |
| [LLM-QAT, arXiv:2305.17888](https://arxiv.org/abs/2305.17888) **[paper]** | Data-free QAT on the model's own generations beats training on real data subsets |
| MoEQuant EBSS, Liu et al. (above) **[paper]** | Self-sampling is the default calibration source in both |
| reap.md verification row 4 | "Self-generated data has no controlled A/B" on these models. Hence experiment D3 |

### 3.4 Agentic datasets: verified on HF **[hf, 2026-10-20]**

| Repo | Revision | License | Content | Use |
|---|---|---|---|---|
| `SWE-Gym/OpenHands-SFT-Trajectories` | `4aaa5a4a…` (in builder) | MIT | 491 successful OpenHands trajectories (gpt-4o / Claude 3.5) | keep |
| `SWE-bench/SWE-smith-trajectories` | `08e109b4a59eaeebf80e4675cd125d42e7ac99a4` | MIT | splits `tool` 24,100 / `xml` 26,076 / `ticks` 25,826 trajectories, 4.2 GB parquet; SWE-agent with claude-3-7 / 3-5 / gpt-4o; field `resolved` ([arXiv:2504.21798](https://arxiv.org/abs/2504.21798)) | **add**, `tool` split (native function calling), `resolved=True` |
| `nvidia/Nemotron-SWE-v1` | `0fe17a965b297a9c943a59050a14c42d5f0083ce` | CC-BY-4.0 | 51,029 OpenHands trajectories on R2E-Gym (Qwen3-Coder-480B), one 11 GB jsonl | **add** (byte-range slice) |
| `nebius/SWE-rebench-openhands-trajectories` | `35455389ab51bf5e2306bfd436ef72d0f98bf882` | CC-BY-4.0 | 67,074 trajectories, 32,161 successful, 1,823 repos, avg 64 turns, OpenAI-style `tool_calls` + `tools.json` | **add**: long, real-repo, already in our schema |
| `nvidia/Nemotron-Agentic-v1` | `650d590978ca35c8f1ecea2faf136e5fac421b62` | CC-BY-4.0 | `interactive_agent` 19,028 multi-turn (simulated user + tools), `tool_calling` 316,094 | **add** both, small weights |
| `nvidia/Nemotron-Terminal-Corpus` | `a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9` | CC-BY-4.0 | ~366K terminal-agent trajectories (adapters + skill-based synthetic) | **add**: shell-heavy, close to Claude Code's Bash use |
| `Agent-Ark/Toucan-1.5M` | `0df3cf37…` (in builder) | Apache-2.0 | MCP tool use | keep |
| `Salesforce/xlam-function-calling-60k` | `26d14ebf…` | CC-BY-4.0, **gated (auto)** | single-turn function calling, used by REAP | skip: gated, single-turn, BFCL-adjacent |
| `open-r1/Mixture-of-Thoughts` | `e55fa28006c0d0ec60fb3547520f775dd42d02cd` | (none tagged) | code/math/science R1 traces, used by REAP | add as a **prompt** source for self-generation |
| `zai-org/LongAlign-10k` | `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc` | (none tagged) | 10K long instruction samples (8K–64K), 0.66 GB | add to `longctx` |
| `HuggingFaceFW/fineweb-edu`, `fineweb-2` | `87f09149…`, `af9c1333…` | ODC-BY | web text, English / multilingual | `general` floor |

Licences: calibration uses the data to compute statistics and nothing is redistributed. CC-BY still asks for
attribution, so keep the `sources` block in `manifest.json` and cite it in model cards of any published build.
`nvidia/Nemotron-Post-Training-Dataset-v2` exists but is gated (auto) and adds little here.

### 3.5 Cloud GPU prices (for teacher generation) **[self-reported by price trackers; volatile]**

- Uvik GPU index (7 Oct 2026), median on-demand per GPU-hour across 16 providers: **H100 SXM $4.20** (range
  $3.20–12.29), **H200 $5.70**, **B200 $8.19**.
- Neo-clouds are lower: H100 ~$2.0–2.5 (GMI, Spheron), B200 $3.70–5.50 (Lambda $5.29 on 8-GPU nodes).
- Hyperscalers: AWS p5.48xlarge ~$55/h, Azure ND H100 v5 ~$98/h, both for 8×H100.
  Sources: [uvik.net GPU price index](https://uvik.net/blog/gpu-price-index/),
  [Spheron H100 2026](https://www.spheron.network/blog/h100-price-per-hour-2026/),
  [Spheron B200 2026](https://www.spheron.network/blog/nvidia-b200-cloud-pricing-2026/),
  [GMI Cloud 2026](https://www.gmicloud.ai/en/blog/gpu-cloud-pricing-llm-inference-2026).
- Both model cards point to vLLM and SGLang recipes **[hf, model READMEs]**, so serving the full checkpoints on rented
  GPUs is a supported path.

---

## 4. Plan for this machine

### 4.0 What exists, and what this chapter changes

**Exists** **[src]**: `calib_data/build.py` (CLI, `allocate`, hash split at 300‰, `validate`, manifest),
`sources.py` (pinned `Parquet` row groups / `JsonlPrefix`, gz cache), `recipes.py` (11 recipes, `Ctx`, `Recipe`),
`render.py` (`Renderer.fit` cuts at turn boundaries), `fixtures.py`. The current output is ~2.33 M calib tokens:
code 0.41 M / agentic 0.77 M / reasoning 0.62 M / longctx 0.54 M over 16 samples of 18–59K **[src,
artifacts/qwen/data/manifest.json]**.

**Changes, in order of value:**

| # | Change | Files | Why |
|---|---|---|---|
| C1 | Content-hash split key + 3 splits (`calib` / `dwq` / `eval`) | `build.py`, `recipes.py` (`Recipe.key`) | stops cross-source leaks (§2.7); DWQ needs its own split (§2.5) |
| C2 | Decontamination vs every `evals/` benchmark | new `calib_data/decontam.py` | §2.7 leak 2 |
| C3 | `personal` source + scrubber | new `calib_data/personal.py`, `calib_data/scrub.py`, `LocalJsonl` in `sources.py` | the deployment proxy (§4.4) |
| C4 | Self-generation driver + `selfgen_*` recipes | new `calib_data/selfgen.py`, recipes | on-policy assistant text (§4.5) |
| C5 | New agentic sources + `longctx` 32K–128K + `general` domain | `sources.py` (`JsonlRange`), `recipes.py`, `build.py` (`DOMAINS`) | coverage (§3.4) |
| C6 | Output schema v2 (optional fields) + manifest additions | `build.py`, `docs/contracts/calib_data.md` | provenance, `gen_spans` |
| C7 | (in `calib/`) position-stratified reservoir, `--expert-tokens 512`, `target_logprob` for calib samples, `ref/` for the `dwq` split | `calib/probe.py`, `calib/run.py` | §2.5, §2.6, NLL filter |

`calib/` already accepts arbitrary domain names (`domain_list` appends unknown ones, sorted) **[src, calib/run.py]**,
so new domains need no calib change. `build.py: validate` asserts `domain in DOMAINS`, so extend `DOMAINS` there.

### 4.1 Output schema (`FORMAT_VERSION = 2`, backward compatible)

`calib.jsonl`, `dwq.jsonl`, `eval.jsonl`: one object per line. `calib/` reads only `id`, `domain`, `tokens` **[src]**,
so the new fields are optional for every consumer.

```json
{"id": "calib-agentic-0007-1a2b3c4d",
 "domain": "agentic",                       // code|agentic|personal|reasoning|longctx|general
 "split": "calib",                          // calib|dwq|eval  (new; redundant with the file, kept for safety)
 "source": "swe_rebench_oh",                // recipe name
 "src": "nebius/SWE-rebench-openhands-trajectories:trajectories.parquet:3:117",
 "group": "c:7f3e9a01d2",                   // new: content-hash split key (§2.7). personal: "p:<sha1(session)>"
 "origin": "public",                        // new: public|personal|selfgen
 "teacher": null,                           // new: e.g. "zai-org/GLM-5.3-Flash@fp8:vllm" or "local:qwen-reap288-4bit"
 "gen_spans": [[6120, 6984]],               // new: [start,end) token ranges written by `teacher` (empty if none)
 "n": 6984,                                 // new: len(tokens)
 "tokens": [248045, 8678, ...]}
```

Rules (in addition to the v1 contract): `gen_spans` lie inside `[0, n)`, do not overlap, and are sorted. `group` is
the only key used for splitting and dedup. `origin == "personal"` rows go only into files under a directory the
manifest marks `"private": true`.

**Intermediate conversation store** (pre-tokenization; written by `personal.py` and `selfgen.py`, read by
`LocalJsonl`). It is model-agnostic, so one store feeds both the GLM and the Qwen builds:

```json
{"key": "personal:3f2a…:cut04",            // stable row key
 "group": "p:3f2a…",                        // split/dedup unit (session for personal, content hash otherwise)
 "time": "2026-09-30T14:02:11Z",            // personal: session start (time split); public: null
 "domain": "personal",
 "messages": [
   {"role": "system", "content": "…"},
   {"role": "user", "content": "…"},
   {"role": "assistant", "content": "…", "reasoning_content": "…",
    "tool_calls": [{"id": "call_0", "type": "function",
                    "function": {"name": "Bash", "arguments": {"command": "uv run pytest -q"}}}]},
   {"role": "tool", "tool_call_id": "call_0", "content": "…"}],
 "tools": [{"type": "function", "function": {"name": "Bash", "description": "…", "parameters": {…}}}],
 "gen": {"from_msg": 5, "teacher": "local:qwen-reap288-4bit", "temperature": 1.0, "top_p": 0.95},  // or null
 "meta": {"project": "h:91ab…", "n_turns": 23, "scrub": {"secret": 2, "email": 1, "path": 40}}}
```

This is the same message dialect the recipes already produce (OpenAI `tool_calls` with dict arguments,
`reasoning_content`), so `Renderer.render/fit` work unchanged.

### 4.2 Builder: commands and the order to implement

```bash
# 0. tests first (existing), then the new unit tests with synthetic fixtures (no real data)
uv run pytest tests/calib_data -q

# 1. public sources + new recipes, screening size (CPU only, ~1–2 min once cached)
uv run python -m calib_data.build --model qwen --out artifacts/qwen/data-screen \
    --samples 512 --mix code=0.8,agentic=1.0,reasoning=0.8,longctx=0.8,general=0.2 \
    --longctx 32768-131072 --decontam evals --preview

# 2. personal store (local only; see 4.4) — model-agnostic, run once
uv run python -m calib_data.personal --projects ~/.claude/projects --out artifacts/personal/store \
    --cutoff 2026-10-01 --exclude-project-glob '*secret*' --max-project-share 0.15

# 3. self-generated responses (see 4.5) — local proxy for personal, cloud for public
uv run python -m calib_data.selfgen --prompts-from artifacts/personal/store --one-step \
    --endpoint http://127.0.0.1:8080/v1 --teacher local:qwen-reap288-4bit \
    --out artifacts/personal/selfgen-qwen --max-new 4096 --concurrency 4

# 4. final corpus per model (3 splits)
uv run python -m calib_data.build --model glm --out artifacts/glm/data-final \
    --mix code=0.8,agentic=1.0,personal=0.8,reasoning=0.8,longctx=0.8,general=0.2 \
    --dwq-samples 400 --personal artifacts/personal/store --selfgen artifacts/selfgen/glm-full,artifacts/personal/selfgen-glm \
    --longctx 32768-131072 --decontam evals
```

`--mix` values are relative weights. As now, sample counts are largest-remainder rounded. Lengths differ by domain,
so check the token totals in `manifest.json` and adjust. Proposed knobs: `--samples` (default 1024 calib),
`--dwq-samples`, `--decontam evals|off`, `--personal DIR`, `--selfgen DIR[,DIR]`,
`--selfgen-share agentic=0.5,reasoning=0.8,...` (fraction of each domain's samples drawn from self-generated rows).

**Split rule (C1).** `bucket = sha1(group) % 1000`: `< 300` eval (unchanged threshold), `300–449` dwq, otherwise
calib. Every row that was in eval before is still in eval, provided its group key is unchanged. For public data
`group` = the content hash, so a one-time reshuffle happens when C1 lands; record it with `format_version: 2`. For
personal data, the time split overrides the hash: `time ≥ cutoff` → eval. Before the cutoff, hash into calib/dwq.

**Disk/time** **[estimate]**:
- New public slices: one row group or a 64–256 MB byte range each → **~1 GB downloaded, ~2 GB gz cache**.
- Tokenisation: HF fast tokenizers at ~1 M tok/s → the 7.4 M tokens of all three splits take seconds per model;
  `fit`'s binary search re-renders and multiplies that by ~5–10 → **~1–3 min per model**.
- Output JSONL: ~6.5 B per token as JSON ints → 7.4 M × 6.5 B ≈ **48 MB per model**.
- Peak RAM: a few GB (source rows + tokenizer). CPU-only, so it is co-tenant-safe while the 44 GB server runs.

### 4.3 New public recipes (C5)

| recipe | domain | source slice | conversion | weight in domain |
|---|---|---|---|---|
| `swe_smith` | agentic | `SWE-bench/SWE-smith-trajectories` `data/tool-0000{0,3}-of-00008.parquet`, 1 row group, `resolved=True` | `messages` is a JSON string, already OpenAI-style with tools → reuse `toucan_messages`-style normaliser | 0.15 |
| `nemotron_swe` | agentic | `nvidia/Nemotron-SWE-v1` `data/r2e_gym.jsonl`, **`JsonlRange(offset, nbytes)`** at 2–3 offsets (a prefix would be one repo cluster) | OpenHands format → existing `openhands_messages` (verify on 1 row) | 0.15 |
| `swe_rebench_oh` | agentic | `nebius/SWE-rebench-openhands-trajectories` `trajectories.parquet`, 1 row group, successful only; `tools.json` | per the card: `trajectory` with `tool_calls` and JSON-string `arguments` → `json.loads` | 0.15 |
| `nemotron_agentic` | agentic | `nvidia/Nemotron-Agentic-v1` `interactive_agent.jsonl` + `tool_calling.jsonl` byte ranges | multi-turn tool use | 0.10 |
| `terminal` | agentic | `nvidia/Nemotron-Terminal-Corpus` `synthetic_tasks/skill_based/*/…parquet` (several skills × 1 row group) | terminal trajectories (column schema: probe 1 row first) | 0.15 |
| `openhands`, `toucan_*` | agentic | unchanged | unchanged | 0.15 / 0.15 |
| `longalign` | longctx | `zai-org/LongAlign-10k` `long.jsonl` byte ranges | real long instruction + answer | 0.25 (others rescaled) |
| `fineweb_edu`, `fineweb2_multi` | general | 1 row group each; `fineweb-2` in ~6 languages (ja, zh, de, es, ru, hi) | plain text as a user "continue / summarise" turn + gen prompt, or raw text | 0.6 / 0.4 |
| `mot_prompts` | reasoning (prompts only) | `open-r1/Mixture-of-Thoughts` 1 row group per subset | **prompt source for selfgen**; R1 answers dropped | — |

Pin `rev` to the full SHAs in §3.4 inside `sources.REV`. Probe each new source's schema with a one-row-group fetch in
a unit test that uses fixtures modelled on that probe. `JsonlRange` is `JsonlPrefix` plus an `offset`: drop the first
partial line, and keep the `len(buf) == nbytes` handling for the last one.

**Agentic length cap.** Raise `MAX_LEN["agentic"]` from 8,192 to **32,768**. Claude Code-style contexts are mostly
10–60K, and the current cap trims every long OpenHands run to its first 8K, which over-represents the *start* of
tasks. calib cost is linear in tokens, and the domain token budget is unchanged, so this means fewer, longer samples.
Keep a 30 % share under 8K, because short tool exchanges are real traffic too.

### 4.4 The personal set: your Claude Code transcripts (C3)

**What is there** **[measured: counts and sizes only, contents not read]**: `~/.claude/projects/` holds 146 project
directories, **1,614** top-level `*.jsonl` session files (median 262 KB, p90 555 KB, max 510 MB) plus **1,608** files
under `<session>/subagents/`, **5.82 GB** of JSONL in total. 04-pruning §4.2 counted 1,570 files / 1.5 GB earlier.
The set grows quickly, and the totals here include subagent files.

**Volume.** 04 estimated ~100 M tokens from 1.5 GB. After JSON overhead and heavy repetition (the same file read in
many turns, the same test output), the *unique* content is much smaller. The final corpus uses ~1.25 M tokens of it
across calib, dwq and eval. **It is selection, not quantity.**

**Format (generic, to be confirmed by P0).** Claude Code writes one JSON event per line. Typical events:
- `type ∈ {user, assistant, system, summary, …}`, with `uuid`/`parentUuid` (a tree, because edits and retries branch),
  `sessionId`, `timestamp`, `cwd`, `gitBranch`, `isSidechain`, `isMeta`;
- `message` = an Anthropic Messages object whose `content` is a list of blocks: `text`, `thinking` (often only a
  signature, or redacted), `tool_use {id, name, input}`, `tool_result {tool_use_id, content, is_error}`, `image`;
- one assistant API message can be split over several lines that share `message.id`.

The transcripts do **not** contain the system prompt or the tool JSON schemas.

**Pipeline** (`calib_data/personal.py`, all local; new deps: `orjson` optional, `detect-secrets` optional):

- **P0 — key-only schema probe** (the user runs it): walk every line and print only the *set of key paths* and value
  *types* with counts (e.g. `message.content[].type=tool_use: 81234`), never values. Pin the parser to what it shows.
  Unit tests use synthetic lines in the same shape (`tests/calib_data/fixtures/claude_session.jsonl`, invented content).
- **P1 — reconstruct threads.**
  - Group by `sessionId` and build the `parentUuid` tree. Take the **latest leaf's root path**, i.e. the conversation
    as it finally ran.
  - Merge assistant lines by `message.id`.
  - Treat `subagents/*.jsonl` as separate conversations with domain `personal` and `meta.sidechain=true`. They
    are Task/Agent-tool traffic, which is also real load.
  - At a compact/summary event, start a new conversation whose first user turn is the summary, since that is what the
    model saw.
- **P2 — map to the canonical dialect.**
  - `text` → content.
  - `tool_use` → `tool_calls[{id, function: {name, arguments: input}}]`.
  - `tool_result` → a `tool` message with the matching `tool_call_id`. List content is joined as text, images become
    `[image omitted]`, and `clip_output` (12K chars, head + tail) is applied as for other sources.
  - `thinking` → **dropped**. It is Claude's, often redacted, and not in the target model's style. Assistant turns get
    the template's empty think block, or a self-generated one (§4.5).
  - `isMeta`, local-command wrappers and hook noise → dropped.
- **P3 — tool schemas and system prompt.**
  - The best source is the *local server*: when you run `scripts/claude-local.sh`, Claude Code sends its real system
    prompt and tool list to `serve/`.
  - Add an opt-in `serve --capture-schema artifacts/personal/claude_tools.json`. It writes **only** the `tools` array
    and a system-prompt *template*, with environment lines (cwd, git status, dates) replaced by placeholders.
  - Fallback: a hand-written `calib_data/claude_tools.json` for Bash/Read/Write/Edit/Glob/Grep/Task/TodoWrite/WebFetch/WebSearch.
  - For unknown `mcp__*` tools, infer an object schema from the observed argument keys and JSON types.
- **P4 — scrub (`calib_data/scrub.py`), before anything is tokenised.**
  - *Known-format secrets*, replaced by typed, per-session-consistent placeholders (`<SECRET_GITHUB_1>`):
    AWS `AKIA/ASIA…`, GitHub `ghp_/gho_/github_pat_`, OpenAI `sk-…`, Anthropic `sk-ant-…`, Slack `xox[abpr]-`,
    Google `AIza…`, Stripe `sk_live_`, JWTs `eyJ….….…`, PEM blocks `-----BEGIN … PRIVATE KEY-----`, URLs with
    `user:pass@`, `Authorization: Bearer …`, and `.env`-style `^[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*=\S+`.
    Optionally also the `detect-secrets` plugin set.
  - *High-entropy strings*: ≥ 24 chars of `[A-Za-z0-9+/_=-]` with Shannon entropy > 4.0 bits/char (hex > 3.0) that
    are not part of a known-benign pattern (git SHAs next to the word `commit`, lockfile `integrity` hashes). They are
    replaced by `<HIGH_ENTROPY_n>`. This is conservative: some hashes get masked, which costs nothing.
  - *Identity*: the home path `/Users/<name>` → `/Users/user`; the account email and any address → `<EMAIL_n>`;
    phone numbers and IPv4/IPv6 → placeholders; `gitBranch`/remote URLs → hashed host and org.
  - *Whole-session drops*: any session that read files matching `*.pem`, `id_*`, `.env*`, `*keychain*`, `credentials*`,
    or that a user-supplied `--exclude-project-glob` matches. Also any session whose scrub count exceeds a threshold:
    many hits mean scrubbing is likely incomplete.
  - *Audit*: write `artifacts/personal/scrub_report.json` with **counts only** per category and project hash, and a
    `--review N` mode that shows the user N random scrubbed snippets around placeholders, locally.
- **P5 — dedup and balance.**
  - Exact dedup of tool results by `sha1(normalised content)`: the 5th identical `cat pyproject.toml` output becomes
    `[same content as earlier read]` in later sessions, keeping only the first occurrence.
  - Near-dup sessions via MinHash over 5-gram shingles (Jaccard ≥ 0.8 → keep the newest). Dependency: `datasketch`, or
    a 60-line numpy MinHash.
  - Cap any one project at ≤ 15 % of personal tokens (`--max-project-share`), so the mask is not fitted to one repo.
    04 also mixes public agentic data in for this reason.
- **P6 — sample cut points.** A session of 200 turns is longer than one sample. Cut at assistant-turn boundaries into
  windows of 8–40K tokens (log-uniform). Each window keeps the system prompt and tools, plus a context-continuation
  user turn if it starts mid-session. That mirrors how Claude Code's history grows. Windows from one session share its
  `group`.
- **P7 — split.**
  - Sessions with `time ≥ --cutoff` → eval. 08-system §7.2 also uses a time split.
  - Before the cutoff, hash `group` into calib/dwq.
  - Then remove from calib/dwq any window whose MinHash is ≥ 0.6 similar to an eval window (cross-boundary near-dups).
  - Optional `--holdout-projects N` keeps N entire projects out, to test generalisation to a repo calibration never saw.
- **P8 — store.** Write `artifacts/personal/store/{calib,dwq,eval}.jsonl.gz` (canonical dialect).
  - Directory mode `0700`. `artifacts/` is already in `.gitignore` **[src]**.
  - `build` renders it per model, just like any other `LocalJsonl` source.

**Rendering.** Personal windows render exactly like the public agentic samples: the target model's template, native
tool syntax, and the server's kwargs (§2.3). Claude-authored assistant text stays only as *history*. With §4.5's
one-step regeneration, the final assistant turn of each window is the local model's own.

**Privacy rules** (each is enforced in code, not left to convention):
- Nothing from `~/.claude` is sent to the network. `selfgen.py` refuses `origin=personal` rows when the endpoint is
  not loopback.
- `calib.jsonl` holds token ids, which are **reversible text**. So do `ref/` (it stores `tokens`) and calib `work/`
  (hidden states, which are invertible enough to leak content). All of them inherit `private: true`.
  `scripts/clean_private.sh` (new) deletes `work/` after each run and wipes personal artifacts on request.
- Model weights built with personal calibration contain only aggregate statistics (masks, bit choices, scale
  tweaks). Leakage through these is very low for REAP/bits. It is nonzero for DWQ, which trains on the text. **Do not
  publish a build whose DWQ split included personal data.** Use a public-only DWQ split for anything you share.
- These sessions contain Claude's outputs. Using them as private *calibration inputs*, regenerating the assistant
  side with the local model, is a different thing from training a model on Claude outputs. Keep the data private, and
  do not publish datasets derived from it.

### 4.5 Self-generation: three ways to get the teacher's text (C4)

`calib_data/selfgen.py` sends prompts to an **OpenAI-compatible endpoint**: the local `serve/`, vLLM or SGLang. It
uses the model's recommended sampling (Qwen T=1.0 top_p 0.95 top_k 20; GLM T=1.0 top_p 0.95, per the evals
contract) and server-side templating with the same kwargs as deployment. It writes canonical rows with `gen` filled.

Modes:
- `--full`: prompt → complete response (reasoning, code, longctx answers, general).
- `--one-step`: real trajectory cut at turn i → one generated assistant turn (thinking + optional tool call), which
  becomes the sample end (agentic, personal).
- **Filters**:
  - drop responses that hit `max_new` without closing `</think>`;
  - for math with a known answer, keep only correct answers (the `correctness_math_verify` column exists in OpenR1
    **[src, sources.py]**), which removes degenerate traces;
  - drop tool calls that fail to parse against the tool schema;
  - drop n-gram loops (more than 3 repeats of a 30-gram).

**Option A — full-precision teacher on rented GPUs (public prompts only; recommended for the final corpus).**

| | GLM-5.3-Flash (FP8, 328 GB) | Qwen3.8-Flash-Next (BF16, 360 GB) |
|---|---|---|
| Node | 8×H100 80 GB (640 GB) TP8, or 4×H200 (564 GB) | 8×H100 TP8, or 4×H200 (204 GB left for KV) |
| Engine | vLLM / SGLang recipe from the model card | same |
| Weights in | 328 GB from HF at ~0.5–1 GB/s ≈ 6–11 min | 360 GB ≈ 6–12 min |
| Generate | 6 M tokens (3× oversampling of ~2 M kept) at ~3K tok/s aggregate (256 concurrent, ~18B active) **[hypothesis]** ≈ 33 min | ~6B active, likely faster; same budget |
| Wall time incl. startup + slack | **~1.5 h** | **~1.5 h** |
| Cost (8 GPUs × 1.5 h = 12 GPU-h) | neo-cloud $2.0–2.5 → **$24–30**; median $4.20 → **$50**; AWS ~$55/h node → **$83**; Azure ~$98/h → **$147** | same |

**[estimate]** Arithmetic: 12 GPU-h × price. Even if the throughput guess is 3× too optimistic, wall time is ~2.5 h,
which is ~$40–85 at neo-cloud / median prices. What goes up is prompts (public, a few MB). What comes down is the
canonical jsonl (~6 M tokens ≈ 25 MB). This is the only option where the text is from **exactly** the model being
calibrated, and its cost is small next to the hours of local calibration it feeds. Before the run, smoke-test the
template and kwargs on 10 prompts and diff the rendered prompts against `Renderer.text` output.

**Option B — the local 4-bit proxy, overnight (required for personal; fallback for public).**
- Qwen: the running `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` + MTP on `127.0.0.1:8080`, measured at 54–58 tok/s
  single-stream **[measured, PLAN Status]**. `serve/engine.py` uses mlx-vlm's `BatchGenerator` **[src]**, so 4
  concurrent requests should give more aggregate throughput. ~120 tok/s is a **[hypothesis]**: measure it with a
  10-minute probe, and expect MTP gains to shrink under batching.
- Budget for personal one-step: ~500 windows × ~1.2K generated tokens = **0.6 M tokens**, plus prefill of ~500 ×
  20K = 10 M prompt tokens. Prompt caching helps little across windows. At ~55–120 tok/s decode that is **1.4–3 h**,
  and prefill at an assumed ~1.5–3K tok/s **[hypothesis]** adds 1–2 h. **One evening.**
- Public prompts by proxy (if Option A is skipped): 6 M tokens at 55–120 tok/s = **14–30 h ≈ 2 nights**.
- **GLM has no local proxy today.** It needs a GLM build first: the gen-1 K=144 ~3 bpw build (~65 GB, 04 §4.0) is
  the proxy for the *second* calibration round. That is 04 §7.8's plan, and this chapter makes it concrete.
- Memory: decoding through the already-running server adds only KV (≤ 8 GB at 128K, PLAN), so the ≤ 80 GB process
  cap holds. Run generation when you are not coding (`control/` daemon schedule), because it competes for GPU time.
- **Bias control** (§2.4):
  1. Mark rows `teacher=local:*`.
  2. In the calib pass, record the full model's mean NLL over each `gen_span` (C7).
  3. Drop proxy rows above the 90th percentile of that NLL, and any row more than 1.5× the median NLL of
     Option-A/full-teacher rows when those exist **[hypothesis thresholds]**.
  4. Cap proxy-generated rows at 50 % of any domain's calib samples.

**Option C — the full model streamed on the Mac (not recommended).**
- Layer-streamed generation reads all weights per decode step: 328 GB / ~9–12 GB/s ≈ 30–40 s per step
  **[computed from 04's measured SSD rates]**.
- Batching B sequences per step amortises the read: at B=256 and 2K new tokens, that is 2,000 steps × ~35 s ≈ 19 h for
  0.5 M tokens. But the KV/state of 256 sequences × 45 layers has to be paged to disk between layers.
- Use it only if the cloud is ruled out and the proxy NLL filter rejects too much.

**Where each domain's assistant text comes from (target for the final corpus):**

| domain | prompts | assistant region |
|---|---|---|
| code | `gh_files`, `commit_edit` (real) | 50 % self-gen (A), 50 % as now (real diffs and full files are useful targets) |
| agentic | all public trajectories | history real; 50 % of samples end in a self-gen one-step turn (A) |
| personal | your sessions | history real; 70 % end in a one-step turn (B, local only) |
| reasoning | OpenR1, OpenThoughts, Mixture-of-Thoughts prompts | **80 % self-gen (A)**, 20 % R1 traces (diversity; and R1 traces are "hard" text) |
| longctx | narrativeqa, arXiv, code_repo, LongAlign | documents real; answers self-gen (A) (short, cheap) |
| general | fineweb-edu / fineweb-2 | raw text plus a few self-gen continuations |

### 4.6 Feeding `calib/` and `build/`

```bash
# calib pass on the final corpus (04 §4.3 costs): GLM ~4.5 h, Qwen ~2.2 h for 4.4 M calib tokens,
# plus the dwq split (ref only, no probes): +1.9 M tokens ≈ +1.4 h GLM / +0.7 h Qwen  [computed: 1.9M/300 and 1.9M/620 tok/s, ×0.8 w/o probes]
uv run python -m calib.run --model glm --src $GLM_SNAP --data artifacts/glm/data-final \
    --out artifacts/glm/calib-final --mem-gb 40 --expert-tokens 512 --ref-splits eval,dwq --pos-strata 4096,32768
# build with explicit domain importance (precision came from token counts; importance comes from here)
uv run python -m build.make --model glm --src $GLM_SNAP --calib artifacts/glm/calib-final \
    --domain-weights code=1,agentic=1,personal=1,reasoning=0.8,longctx=0.7,general=0.3 ...
```

- **Domain weights are [hypothesis] defaults.**
  - 04 put personal at ~40 % *of agentic*; 01 had `personal=2` against 1 for the others. With `personal` as its own
    domain, `agentic=1, personal=1` gives personal 50 % of the agentic-plus-personal mass. That sits between the two
    and keeps both options open: set `personal=0.67` to reproduce 04's 40 %.
  - Experiment D2 decides.
  - **Use the same weights for REAP and for the sensitivity allocator**, or the two halves of the joint optimum
    (04 §4.5) are optimised for different workloads.
- **Hidden-state scratch** **[computed]**: GLM (4.4 + 1.9 + 1.1) M tokens × 32.8 KB × 2 boundaries ≈ **485 GB** peak.
  Qwen ≈ 303 GB. SSD writes per GLM pass: 7.4 M × 32.8 KB × 45 layers ≈ **10.9 TB**. That is more than 04's 5.9 TB for
  4 M tokens, so **run the dwq ref as a separate head-only follow-up only if SSD wear matters**. Otherwise accept it
  for the single final pass. Screening passes stay at 1 M tokens.

### 4.7 Order of work

| step | what | needs | time [estimate] | gate |
|---|---|---|---|---|
| 1 | C1 + C2 (content split, 3 splits, decontam) + tests | CPU | 0.5 day | leak test: inject a HumanEval prompt and an OpenR1/OpenThoughts duplicate → both caught |
| 2 | C5 (new sources/recipes, longctx 32–128K, `general`) | ~1 GB download | 1 day | `--smoke --source hf` renders every recipe in both templates; preview diff looks native |
| 3 | P0 schema probe (user) → C3 personal pipeline on synthetic fixtures → real run | CPU, ~5–10 min for 5.8 GB | 1.5 days | scrub unit tests with planted fake secrets: 100 % caught; `--review 50` passes the user's eye |
| 4 | C6 schema v2 + contract doc update | | 0.5 day | old `calib` still reads v2 files (only id/domain/tokens) |
| 5 | C4 selfgen driver, Option B probe on 20 prompts | running server | 0.5 day + 1 evening | throughput measured; filters' rejection rates logged |
| 6 | Option A cloud run (both models) | ~$50–100 | 0.5 day | template diff clean on 10 prompts |
| 7 | C7 calib changes, tested on tiny fixtures | | 1 day | stratified reservoir counts per bucket; NLL per gen_span written |
| 8 | Screening experiments D1–D5 via `calib.sweep` (04 §4.4) | GPU, one at a time | ~1 day of runs per model | §5 |

---

## 5. Decision criteria & what to measure

All comparisons use 04's statistics: per-domain KL against the *unpruned full-precision* `ref/`, with a bootstrap 95 %
CI over **samples**. A variant wins if the CI excludes 0 in its target domain and no other domain is worse beyond its
CI. Pure-pruning effects are measured with `calib.sweep` (04 §4.4, ~2 h per batch for GLM): calibrate with data
variant X, derive the mask, evaluate KL on a **fixed** eval set. **The eval set must not change between data
variants**, or every comparison is confounded.

| ID | Experiment | Variants | Primary metric | Decision |
|---|---|---|---|---|
| D1 | Workload vs generic | current mix vs C4/fineweb-only (sanity) | per-domain KL at K=144 / K=320 | Expect a large gap. If small, suspect the eval set or the sweep, not the data |
| D2 | Personal weight | personal ∈ {0, 0.67, 1, 2} | KL on **personal-eval** and on public agentic-eval | Choose the largest weight that keeps public-agentic KL within its CI of the personal=0 result |
| D3 | Self-generated vs original assistant text | reasoning+agentic with R1/Claude text vs self-gen (A) vs self-gen (B, proxy) | KL on a **self-generated eval subset** (closest to decode-time reality) + task evals (`math`, `tools`) | Adopt self-gen if D3-A beats original in ≥ 2 of 3 domains. Adopt the proxy for public data only if it is within the CI of A |
| D4 | Long-context range | longctx 16–64K vs 32–128K (same tokens) | `evals.longctx` at 64K/128K + longctx KL | Adopt 32–128K unless 32K-cell RULER drops beyond noise |
| D5 | General floor | general weight 0 vs 0.3 | KL on general eval (multilingual subset) + rare-entity probe (04 §5) | Keep the floor if it costs < its CI on the workload domains |
| D6 | Volume | 250K vs 1 M vs 2 M per domain | split-half kept-set overlap; bootstrap uncertain-band size | Final volume = smallest with overlap ≥ 90 % and band ≤ 10 % of K |

**Diagnostics to record on every built corpus** (add to `manifest.json` or to a `calib_data.report` command):
- **Expert coverage**: per domain, the minimum and 5th-percentile routed tokens per expert, from calib's
  `expert_count.{d}`. Gate: every expert ≥ 512 rows pooled, and tail experts ≥ 256 per workload domain.
- **On-policy-ness**: the full model's mean NLL per token on `gen_spans` (self-gen) vs on original assistant text.
  Expected ordering: A < B < R1/Claude **[hypothesis]**. If B ≈ R1, the proxy is not helping.
- **Routing match to deployment**: Jensen–Shannon divergence between calib `expert_freq` (domain-weighted) and live
  per-expert hit counts from the server (`live_freq.npz`, 01-memory §4.7). This is the closest available measure of
  "is calibration like my real use?". Track it per layer. Use the worst layers to decide what data to add.
- **Leak report**: counts dropped by decontam per benchmark (expect a few dozen on NuminaMath-derived math, HumanEval
  copies on GitHub), cross-source group collisions, personal near-dups across the time boundary.
- **Scrub report**: counts per category. A sudden jump between runs means a new secret format. Review it.

---

## 6. Failure modes

| Failure | Symptom | Prevention / detection |
|---|---|---|
| Secret leaks into artifacts | a key or a token in `calib.jsonl` (decodable), in `ref/` or `work/`, or in a published DWQ build | P4 scrub + whole-session drops; private dir modes; never publish personal-DWQ builds; `scripts/clean_private.sh`; grep the decoded corpus for the planted canary secrets in tests |
| Template/kwargs mismatch with the server | KL on live-like prompts much worse than on eval; tool calls malformed after DWQ | render via the same template + kwargs as `serve/`; diff `Renderer.text` against the server's prompt for the same messages (C1 test); pin `chat_template_sha256` in the manifest (already there) |
| Thinking-history mismatch | multi-turn samples contain past `<think>` blocks the server never sends (or the reverse) | verify `serve/app.py`'s translation (§2.3) and set `clear_thinking` / `preserve_thinking` to match |
| Proxy circularity | each round prunes more of the same "rare" experts; multilingual/rare-entity probes drift down across rounds | full-model NLL filter, a 50 % proxy cap, Option A for public prompts, the `general` floor, D5 probes |
| Self-gen mode collapse | low diversity (same openings, same tool sequence); `gen_spans` NLL very low | T=1.0 sampling, distinct prompts, n-gram diversity metric per domain; mix 20 % original traces back in |
| Degenerate generations kept | loops, unclosed `</think>`, max-token hits in the corpus | §4.5 filters; log rejection rates; reject the run if > 20 % rejected |
| One repo dominates personal | mask tuned to one project's language; other repos degrade | `--max-project-share 0.15`; `--holdout-projects` eval |
| Cross-source eval leak | eval KL looks better than live behaviour; variant comparisons inconsistent | content-hash `group` split (C1), decontam (C2), the collision count in the manifest |
| Long samples dominate or vanish | longctx stats noisy; or 128K-position rows absent from sensitivity | domain-balanced stats (exists), position-stratified reservoirs (C7), longctx build weight |
| Truncation inside a turn | a sample ends mid-tool-call or mid-`<think>`: unnatural token states | `Renderer.fit` cuts at turn boundaries (exists); for a single overlong turn, prefer *dropping* to hard truncation outside `reasoning` |
| Tool schemas missing or wrong | the model sees calls to tools that are not declared, so tool-format experts are mis-calibrated | P3 schema capture; unit test: every `tool_calls.name` in a sample appears in its `tools` |
| Tokenizer drift | ids shift after an upstream tokenizer update; calib/eval inconsistency | manifest `tokenizer_json_sha256` check in `calib.run` meta (already mismatch-guarded) |
| DWQ overfits its split | DWQ training loss falls, eval KL rises | separate `dwq` split, held-out gate per band (quant.md rec. 5) |
| SSD wear | a multi-TB write per final pass | 1 M-token screening, one final pass, dwq ref optional/separate (§4.6) |

---

## 7. Stretch ideas

1. **On-policy capture from your own use of the local model.** With an opt-in `serve --capture-dir`, record (rendered
   prompt token ids, generated token ids, tool schema hash) for each request while you use Claude Code on the local
   model. This is real tool environments with real results, and the generated side is the *deployed* model's own. It
   is the ideal agentic calibration set once a model is good enough to use daily. It needs the same privacy pipeline
   (P4–P8) and the same proxy-circularity guard, because the deployed model is pruned. Each month, recalibrate the
   **tier ordering** (01-memory §4.7 online refresh) on it, which is cheap. Re-prune less often.
2. **Expert-balanced self-sampling (EBSS-style).** Steer generation toward prompts or continuations that route to
   uncertain-band experts (04 §4.2): score candidate prompts by how many routed tokens they give to the band, using
   router logits from a cheap one-layer pass. This spends calibration tokens where the mask decision is actually
   uncertain, instead of on experts that are clearly in or out. [MoEQuant](https://arxiv.org/abs/2505.03804) did the
   token-level version **[paper]**.
3. **Importance-weighted calibration.** Weight each calib sample by `p_dep/p_cal`, estimated from routing features (a
   small classifier separating calib routing traces from live routing traces). REAP and the NMSE sums then become
   unbiased estimates under the deployment distribution. `calib` would need one per-sample weight field, and
   `gen_spans` already shows how to pass per-sample metadata.
4. **MTP drafter calibration on self-generated text only.** The MTP head predicts *the model's own* next tokens.
   Calibrating its pruning and quantization on R1 or Claude text optimises it for text it will never draft.
   Self-gen spans are the right data for the MTP stage that 04 §4.1 proposes. Check it with acceptance rate on
   self-gen eval.
5. **Multi-step self-play in a sandbox.** Run the full model (Option A node) as an actual OpenHands/SWE-agent on
   SWE-Gym or SWE-rebench tasks with their docker environments, giving fully on-policy trajectories. The cost is the
   environment infrastructure, not GPU time. Do it only if D3 shows that one-step regeneration helps a lot, which
   would suggest that full trajectories help more.
6. **Workload-switchable calibration packs.** Keep per-domain REAP and sensitivity (calib already stores them) and
   build two masks: "coding day" (personal + agentic heavy) and "reading day" (longctx + reasoning heavy). Swap at
   runtime with 04 §7.4's expert packs. The data side is just two `--domain-weights` vectors over the same calib pass.

---

## Sources

- Calibration-data effects: Williams & Aletras [arXiv:2311.09755](https://arxiv.org/abs/2311.09755) · Bandari et al. [arXiv:2410.07461](https://arxiv.org/abs/2410.07461) · Ji et al., Beware of Calibration Data [arXiv:2410.17711](https://arxiv.org/abs/2410.17711) · Self-calibration [arXiv:2410.17170](https://arxiv.org/abs/2410.17170) · Liu et al., Quantization Hurts Reasoning? [arXiv:2504.04823](https://arxiv.org/abs/2504.04823) · Marchisio et al., multilingual quantization [arXiv:2407.03211](https://arxiv.org/abs/2407.03211)
- MoE compression: REAP [arXiv:2510.13999](https://arxiv.org/abs/2510.13999) ([code](https://github.com/CerebrasResearch/reap)) · HOPE [arXiv:2609.18916](https://arxiv.org/abs/2609.18916) · NAEE [arXiv:2402.14800](https://arxiv.org/abs/2402.14800) · MoEQuant [arXiv:2505.03804](https://arxiv.org/abs/2505.03804) · QuantMoE-Bench [arXiv:2406.08155](https://arxiv.org/abs/2406.08155)
- Quantization: GPTQ [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) · AWQ [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) · LLM.int8() [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) · LLM-QAT [arXiv:2305.17888](https://arxiv.org/abs/2305.17888) · Wanda [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) · mlx-lm `LEARNED_QUANTS.md` and `quant/dwq.py` (installed 0.32.2)
- On-policy distillation: GKD [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) · MiniLLM [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- Agentic data: SWE-smith [arXiv:2504.21798](https://arxiv.org/abs/2504.21798), [dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) · SWE-Gym [arXiv:2412.21139](https://arxiv.org/abs/2412.21139), [trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) · [nvidia/Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) · [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) · [nvidia/Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) · [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) · [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) · [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) · [zai-org/LongAlign-10k](https://huggingface.co/datasets/zai-org/LongAlign-10k) · [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) · [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- GPU prices (volatile, vendor-published): [Uvik GPU price index](https://uvik.net/blog/gpu-price-index/) · [Spheron H100 2026](https://www.spheron.network/blog/h100-price-per-hour-2026/) · [Spheron B200 2026](https://www.spheron.network/blog/nvidia-b200-cloud-pricing-2026/) · [GMI Cloud 2026](https://www.gmicloud.ai/en/blog/gpu-cloud-pricing-llm-inference-2026)
- Model-specific community evidence (all **[self-reported]**): links in `docs/research/reap.md` §3–4 and `docs/research/quant.md` §3 (OpenMOSE, AnonimousA, lapadatbogdan, patrickbdevaney, Litwein, sh0wie).
- Local: `calib_data/{build,recipes,render,sources,fixtures}.py`, `docs/contracts/{calib_data,calib,evals}.md`, `calib/run.py`, `build/calib_io.py`, `evals/data.py`, `serve/engine.py`, `scripts/claude-local.sh`, `artifacts/tokenizers/{glm,qwen}/{chat_template.jinja,config.json}`, `artifacts/qwen/data/manifest.json`, model READMEs on HF (vLLM/SGLang support, Qwen 262K native context).
