# Chapter 4: Expert pruning and merging, or cutting off what you never use

> Part of the master plan for the best co-tenant local model on this M5 Max (128 GB, 614 GB/s).
> This chapter **builds on** `PLAN.md`, `docs/research/reap.md` (the verified REAP findings), `docs/research/quant.md`,
> `docs/research/mac.md`, `docs/research/decode.md` and `docs/contracts/calib.md`. It does not repeat their
> verification work. Where it adds something new, it says why.
>
> Tags: **[paper]** arXiv or peer-reviewed · **[src]** read in this repo or in installed package source ·
> **[self-reported]** a model card or README number, not reproduced · **[measured]** run on this machine for this
> chapter (tiny, < 1 GB) · **[computed]** arithmetic from the real `config.json` and safetensors parameter counts ·
> **[hypothesis]** a proposal with no published evidence yet. The experiments described here are what turn a
> hypothesis into a measurement.

---

## 0. TL;DR (what this chapter decides)

1. **GLM-5.3-Flash has to be pruned to about 50 % to fit the co-tenant budget (K ≈ 144 of 288).** Qwen3.8-Flash-Next,
   it turns out, **does not have to be pruned at all** once the 51 B-param n-gram/PLE table is memory-mapped
   (`ple_storage`). With 512 experts at about 3.25 bpw it is ~55 GB *resident*, against a 70 GB weight cap.
   For Qwen, pruning becomes a *trade between experts and bits at equal resident bytes*, and our own sweep has to
   settle it. That is the main new framing in this chapter (§4.2).
2. Rank experts with **REAP** (router-weighted expert activation). Also compute **REAP-sum** and **HOPE**
   (pairwise co-activation) from the *same* calibration pass. Then choose between them on held-out KL, not on
   retained saliency mass, which is circular.
3. Do **not** shrink top-k for speed. On this machine it buys about 5–10 % decode at a measurable quality cost
   (§2.8). MTP speculative decoding buys 1.5–1.7×.
4. Do **not** drop attention layers. In these hybrids the few full-attention layers do the long-context retrieval
   (§2.10). Whole-block drop is a stretch experiment that the streaming sweep evaluates for free.
5. Order: **prune first in full precision → re-calibrate on the pruned model → quantize → DWQ**.
6. The new tool this chapter specifies is a **multi-mask streaming evaluator**, `calib/sweep.py`. It loads each
   layer once and pushes N pruning variants through it, so the *pure* pruning KL curve of a 321 B model can be
   measured on a machine that cannot hold the model in bf16. A batch of 8 variants on a 256K-token eval set takes
   ~2 h for GLM and ~1 h for Qwen (§4.4).

---

## 1. Why it matters (intuition)

A Mixture-of-Experts layer is a big library where each token borrows only a few books. GLM-5.3-Flash has 288
"books" (experts) per layer and lends 8 per token. Qwen3.8-Flash-Next has 512 and lends 10. **97 % of GLM's
parameters are routed experts** (304.4 B of 321 B) **[computed, reap.md §2]**, but each token reads only
8/288 = 2.8 % of them. Pruning acts on that asymmetry: if some books are almost never borrowed for *your*
workload, or are borrowed but contribute almost nothing when they are, they cost memory and buy no quality.

Three facts make this harder than it sounds, and they drive everything below:

- **There are no dead experts.** Both models were trained with aux-loss-free load balancing (the
  `e_score_correction_bias` mechanism, [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) **[paper]**). Its
  whole job is to make every expert get traffic. OpenMOSE measured normalized routing entropy of 0.92–0.99 on
  GLM-5.3-Flash and found all 12,096 experts used **[self-reported, reap.md §4]**. So every pruned expert is real
  capacity. The question is never "which experts are unused" but "*which capacity can this workload live
  without*".
- **Which capacity you need depends on the domain.** Chat-only and code-only saliency rankings correlate at only
  Spearman 0.714. A chat-only mask "lost most of its coding ability", and a code-only mask lost Japanese
  **[self-reported, reap.md §4]**. Calibrating on generic C4 drove some models' code accuracy to 0 % in the REAP
  paper **[paper]**. **The calibration data is the pruning decision.** That is why this project calibrates on its
  own workload (code / agentic / reasoning / long-doc), and why the user's own Claude Code sessions are a valuable
  calibration source (§4.1).
- **Bytes trade against bytes.** Pruning and quantization take from the same memory budget. At K=144 GLM,
  one bit per weight across all routed experts costs 19.0 GB, about as much as **41 more experts per layer** at
  3.5 bpw. At K=320 Qwen, one bit costs 9.4 GB, about **91 experts per layer** **[computed]**. "Prune harder and
  keep more bits" and "keep more experts at fewer bits" are the two ends of one continuous trade-off, and §4.5
  gives the rule for finding the optimum.

What you get by pruning well: GLM-5.3-Flash, the strongest model of the class that fits, becomes runnable at all
inside 80 GB, and Qwen gains headroom for KV cache and MTP. What you lose by pruning badly: rare-entity recall,
other languages, termination (models that never stop), and agentic efficiency (more turns per task). Greedy
pass@1 benchmarks are bad at catching any of these (§3.4).

---

## 2. How it works (mechanics and key math)

### 2.1 Notation

For one MoE layer and one token with hidden state `x`:

$$y(x)=\sum_{j\in T(x)} g_j(x)\,f_j(x)\;+\;y_\text{shared}(x)$$

- `T(x)` is the top-k set chosen by the router (k = 8 for GLM, 10 for Qwen).
- `f_j(x) = W_down,j · (silu(W_gate,j x) ⊙ W_up,j x)` is expert j's output.
- `g_j(x)` is the combine weight the model **actually applies**.
  - GLM: experts are *selected* by `sigmoid(logits) + e_score_correction_bias`. The *weights* are the un-biased
    sigmoid scores of the selected experts, renormalized over the top-8 and multiplied by `routed_scaling_factor`
    = 2.5 (`norm_topk_prob=True`, `scoring_func=sigmoid`, `topk_method=noaux_tc`, `n_group=1` in the real
    `config.json`) **[src]**.
  - Qwen: softmax over 512 logits → top-10 → renormalize (`norm_topk_prob` defaults to True) **[src]**.
- Pruning keeps a set `S` of experts and removes the rest, `P`. The router rows of `P` are deleted, and for GLM so
  are their `e_score_correction_bias` entries.

### 2.2 What pruning does to the output: the substitution error

After pruning, the router picks the top-k **among the survivors**, `T'(x)`, and renormalizes:

$$y'(x)=\sum_{i\in T'(x)} g'_i(x)\, f_i(x).$$

Tokens whose top-k contained no pruned expert are **bit-for-bit unchanged**. With `norm_topk_prob`, the weights
depend only on the scores *within* the chosen set, and the chosen set is the same. The repo verified this:
slicing a tiny real block 32 → 16 gave max |Δy| ≤ 6e-8 on those tokens **[measured, reap.md §Verification]**.

For a token that did route to some pruned `j ∈ P ∩ T(x)`, the expert is *substituted* by the next-best survivor.
The REAP paper bounds the error of one substitution, `j → i`, by

$$\lVert \Delta y\rVert \;\lesssim\; g_j(x)\big(\lVert f_j(x)\rVert + \lVert f_i(x)\rVert\big)$$

**[paper, arXiv:2510.13999]**. You can't control the `f_i` term (which survivor gets promoted depends on the token).
What you can control is the term you know, `g_j‖f_j‖`, and it also multiplies the unknown term. Two more facts
help:
- **Renormalization error is small.** The promoted survivor was ranked k+1 or lower, so its gate is small. HOPE
  proves the replacement gate mass is at most the pruned gate mass (Lemma 3) **[paper, arXiv:2609.18916]**.
- Most of the total gate mass is preserved anyway: for GLM the renormalized weights still sum to exactly 2.5
  **[measured, reap.md]**.

### 2.3 REAP, exactly, and why it beats frequency

$$S_j^\text{REAP}=\frac{1}{|\mathcal X_j|}\sum_{x\in\mathcal X_j} g_j(x)\,\lVert f_j(x)\rVert_2,\qquad \mathcal X_j=\{x: j\in T(x)\}$$

Prune the lowest `S_j` in each layer **[paper]**. Calib already computes this as `expert_reap` (and `expert_reap.{d}`
per domain), using the true combine weight from `arch.moe_weights` **[src, calib/probe.py `_moe`]**.

**Why the conditional mean (over routed tokens only) rather than an average over all tokens.** Compare two
experts:
- *A filler expert*, picked on 30 % of tokens, nearly always in 8th place with `g ≈ 0.05` and a small output.
- *A specialist*, picked on 0.5 % of tokens (say, a Rust borrow-checker pattern or a `<tool_call>` closing
  token), with `g ≈ 0.6` and a large output.

Frequency pruning keeps the filler and drops the specialist. That is exactly the catastrophic case: the specialist
is the only thing producing that behaviour, while the filler's 5 % contribution is easily covered by its
neighbours. REAP's conditional mean asks how much an expert matters *when it is used*, and keeps the specialist.

The evidence is stark at 50 % pruning **[paper]**:
- Qwen3-Coder-480B code average: frequency 0.010 vs REAP 0.619 (base 0.636).
- Kimi-K2: frequency 0.056 vs REAP 0.624.
- GLM-4.5-Air math: frequency 0.613 vs REAP 0.857.

**The opposite risk, and REAP-sum.** A conditional mean computed over 3 tokens is noise. An expert seen 3 times
with one huge activation can rank above a workhorse. The expected per-token contribution is

$$S_j^\text{sum}=\mathbb E_x\!\left[\mathbb 1\{j\in T(x)\}\,g_j\lVert f_j\rVert\right]=\text{freq}_j\cdot k\cdot S_j^\text{REAP}$$

(the repo verified `REAP-sum = REAP-mean × freq` exactly). On GLM-5.3-Flash at 144 experts, lapadatbogdan measured
held-out KL of frequency 0.94, REAP-mean 0.85 and **REAP-sum 0.73** **[self-reported, reap.md §3]**. That source
disagrees with the REAP paper's design choice, so **we test both**. A middle path, which we propose and which is
not in any paper, is **shrinkage**:

$$\tilde S_j=\frac{n_j S_j + n_0 \bar S}{n_j+n_0}$$

Here `n_j` is the routed count and `n_0 ≈ 64`. Shrinkage leaves well-observed experts at their REAP-mean and pulls
rarely-seen ones toward the layer mean. **[hypothesis]**

**Super experts.** A few experts produce outputs 40–470× the median norm (GLM layers 17–34, OpenMOSE
**[self-reported]**). Removing them breaks the model even though their *mean* saliency can look ordinary,
which is the "super experts" phenomenon ([arXiv:2507.23279](https://arxiv.org/abs/2507.23279) **[paper]**).
The REAP code force-keeps experts whose max activation element exceeds `max(p99.5, global_max/10)` in the first
75 % of layers **[src]**. Calib does **not** yet record per-expert maxima. §4.1 adds them.

### 2.4 HOPE: pruning pairs that cooperate

REAP scores experts one at a time. But a token whose top-8 contained **two** pruned experts loses both, and the
errors add *inside a norm*. HOPE squares the REAP-style bound over the whole pruned set **[paper,
arXiv:2609.18916]**:

$$Z(x)=\Big[\sum_{j\in P\cap T(x)} g_j(x)\lVert f_j(x)\rVert\Big]^2=\sum_{i,j\in P}\mathbb 1\{i,j\in T(x)\}\,g_i\lVert f_i\rVert\, g_j\lVert f_j\rVert$$

With `p ∈ {0,1}^E` (1 = prune), the expected error is a quadratic form:

$$\min_{p}\; p^\top F p\quad\text{s.t.}\;\sum_k p_k=|P|,\qquad F_{ij}=\frac{1}{|\mathcal X_{ij}|}\sum_{x\in\mathcal X_{ij}} g_i g_j\lVert f_i\rVert\lVert f_j\rVert,\;\;\mathcal X_{ij}=\{x: i,j\in T(x)\}.$$

- **The diagonal is REAP-squared plus a variance.** `F_kk = E[(g_k‖f_k‖)²] = (S_k^REAP)² + Var`. Zeroing the
  off-diagonal recovers REAP's prune set **[paper]**.
- **The off-diagonal is cooperation.** Two experts that are often co-selected *for the same token* both with large
  contributions should not both go.
- **Normalization caveat.** HOPE normalizes `F_ij` by the co-activation count, the conditional choice, like
  REAP-mean. The *expected* per-token error instead uses the unconditional
  `F^unc_ij = (|X_ij|/N)·F_ij`, which is the second-order analogue of REAP-sum. Given the REAP-sum result above,
  **we test both**. **[hypothesis]**
- **Solve.** The paper relaxes to a continuous QP and rounds to the top |P|, at ~1–2 s per layer
  **[paper]**. With E ≤ 512 a simpler exact-objective method is just as good: start from the REAP set and apply
  best-improvement swaps (a swap changes the objective by an O(E) update) until none helps.
- **Data.** `F` is E×E per layer per domain: 0.66 MB for GLM (288²×8 B) and 2.1 MB for Qwen (512²×8 B)
  **[computed]**. HOPE needs somewhat more data for a stable selection: 10K prompts for 99 % prune-set agreement
  vs 8K for REAP **[paper]**.
- **Evidence, honestly.** HOPE has the best average rank at 50 % pruning (1.58 of 5 vs REAP 2.42), up to +6.1 % on
  agentic coding (SWE-bench Pro) **[paper]**. But per-cell results are mixed. On GLM-4.5-Air with
  Evol-CodeAlpaca calibration, the Tulu3 mean favoured REAP at 40 % and 50 %. On Qwen3.5-122B, MATH was worse in 8
  of 12 cells **[paper, reap.md §4]**. It is cheap (+6–7 % calibration time) and sits exactly in our regime, so we
  log `F` and let held-out KL decide.

### 2.5 Non-uniform per-layer budgets

Uniform pruning removes the same count from every layer. Layers are not equally sensitive. OpenMOSE's GLM-5.3 probe
found that protecting early layers cuts KL by up to ~15 % at the same budget, and that layers 42–44 are by far the
easiest to prune **[self-reported]**.

**Two ways to allocate:**

1. **Global ranking of normalized scores** (`build/prune.py: global_keep`, already implemented **[src]**):
   normalize each layer's scores to sum to 1, pool them, keep the global top-N, with a per-layer floor `min_keep`
   (default `max(top_k, 0.25·E)`). It is simple, but "share of a layer's saliency" is not comparable across
   layers: a 1 % share in a sensitive layer costs more than 1 % in a robust one. llm-compressor PR #3281 normalizes
   by the layer max instead, with the same problem **[src, reap.md §5]**.
2. **Measured error curves + knapsack** (proposed). For each layer ℓ, measure the *local relative output error*
   when keeping its top-k experts:

   $$e_\ell(k)=\frac{\sum_x\lVert y_\ell(x)-y'_{\ell,k}(x)\rVert^2}{\sum_x\lVert y_\ell(x)\rVert^2}$$

   Compute this on a few thousand real MoE inputs (the reservoir calib already keeps for `mlp.gate`) by running the
   real `switch_mlp` with the pruned routing. Then solve

   $$\min_{k_\ell}\sum_\ell w_\ell\, e_\ell(k_\ell)\quad\text{s.t.}\;\sum_\ell k_\ell\cdot\text{bytes/expert}_\ell\le B,\;\;k_\ell\ge k_\min.$$

   This is a multiple-choice knapsack: exact by DP over (layer, k-grid), and trivially small (42 layers × ~20
   grid points). `w_ℓ` is the layer's *amplification*, how much a unit of local error moves the final KL. Start
   with `w_ℓ = 1` and fit it later from sweep results (§4.4).

**The evidence is modest.** EvoESAP's evolutionary per-layer allocation on top of REAP rankings gave small,
inconsistent gains (Qwen3-30B at 25 %: Code 0.629 → 0.625, WildBench +2.3 %) **[paper, arXiv:2603.06003]**. Expect
≤ 15 % KL, not a step change. The engineering cost is already paid: `build/shim.py` writes a checkpoint-local
`pruned_moe.py` with per-layer `expert_counts` **[src]**. **Rule:** keep the shift bounded (±32 experts from
uniform) on the first attempt, and accept it only if per-domain held-out KL improves *in every domain*.

### 2.6 Merging instead of deleting

Pruning throws expert j away. Merging folds j into a survivor, hoping to keep part of its function.

**The basic obstruction** (REAP paper's theorem) **[paper]**: merge experts `i, j` into one expert `f̄ = α f_i +
(1−α) f_j` that receives gate `g_i + g_j`. The original pair produced `(g_i+g_j)[r(x) f_i + (1−r(x)) f_j]` with a
token-dependent mixing ratio `r(x) = g_i/(g_i+g_j)`. The merged expert has a *constant* α, so

$$\mathbb E\lVert\Delta y\rVert^2=\mathbb E\big[(g_i+g_j)^2\,(r(x)-\alpha)^2\,\lVert f_i-f_j\rVert^2\big]\;\approx\;\mathbb E[(g_i+g_j)^2]\cdot\mathrm{Var}[r]\cdot\mathbb E\lVert f_i-f_j\rVert^2.$$

Even with the best α = E[r], the error is set by **how much the router varies the mix across tokens**. A router
that sends a token to i or j (high Var[r]) is exactly the case where merging destroys information. The paper calls
this "functional subspace collapse": merged experts' outputs contract toward the centre of the output manifold,
most strongly in late layers of fine-grained MoEs, which both of our models are. Pruning keeps the router's
independent control over every survivor.

There is a second, practical problem. Weight averaging assumes the experts' hidden neurons line up, but neuron p in
expert i has nothing to do with neuron p in expert j (permutation symmetry, cf. Git Re-Basin,
[arXiv:2209.04836](https://arxiv.org/abs/2209.04836) **[paper]**). Merging methods therefore **align neurons** first:
they permute expert j's intermediate dimension (rows of gate/up, columns of down) to match i, typically with the
Hungarian algorithm on activation and weight similarity.

| Method | Grouping | Alignment | Merge weights | Verdict for us |
|---|---|---|---|---|
| **M-SMoE / MC-SMoE** ([arXiv:2310.01334](https://arxiv.org/abs/2310.01334)) | "dominant" experts by routing frequency; others join the most similar dominant expert by router-logit similarity | yes (permutation) | frequency-weighted; MC- adds low-rank + sparse compression of the merged experts | Collapses at 50 % on modern fine-grained MoEs (GLM-4.5-Air code 0.284 vs REAP 0.515) **[paper, REAP]** |
| **HC-SMoE** ([arXiv:2410.08589](https://arxiv.org/abs/2410.08589)) | hierarchical clustering on expert *outputs* | no | average | > 5 % drop at 25 %, > 20 % at 50 % on coding **[paper, REAP]** |
| **REAM** ([arXiv:2604.04356](https://arxiv.org/abs/2604.04356)) | REAP-salient experts become centroids; each greedily absorbs up to C most similar non-centroids by `δ_g` (gate-logit similarity) + `δ̃_E` (similarity of *gate-weighted* outputs). Most centroids stay singletons ("pseudo-pruning") | yes: activation + weight cost | saliency/frequency-weighted | Beats REAP at **25 %** on GEN tasks: Qwen3-30B 69.8 vs 68.6; Qwen3-Next-80B (512 → 384) 71.5 vs 69.6; GLM-4.5-Air 73.9 vs 71.9. At **50 %**, REAP's best GEN beats REAM's (62.8 vs 60.7) **[paper]**. Uses *sequential* calibration: layer ℓ+1 statistics are taken after merging layer ℓ (1.5 h vs 1 h on Qwen3-30B) |

On our exact model, WaveCut's REAM-288 for Qwen3.8-Flash-Next beat REAP-288 on teacher-forced KL (1.256 vs 1.307).
HumanEval/GSM8K were statistically tied (McNemar p = 1.0 / 0.59). Raw saliency-proportional coefficients were
catastrophic (KL 1.397). The best merge moved only ~9 % of the kept experts' mass **[self-reported, reap.md §8]**.

**Reading it:** merging is a *small, positive* refinement at **moderate** ratios (≤ 25–40 %), which is where Qwen
lives, and is unreliable at 50 %, which is where GLM lives. The successful variants barely move the centroid. They
behave like "prune, plus a small correction." Treat REAM as a stretch variant for Qwen only (§7.3).

### 2.7 Smaller experts instead of fewer experts

**(a) Neuron (intra-expert) pruning.** Each expert's intermediate width (GLM 2048, Qwen 640) can be cut. This is
structured, Wanda-like ([arXiv:2306.11695](https://arxiv.org/abs/2306.11695)) saliency on hidden neuron n of
expert e:

$$s_{e,n}=\mathbb E_{x\in\mathcal X_e}\big[|h_{e,n}(x)|\big]\cdot\lVert W_{\text{down},e}[:,n]\rVert_2$$

`h` is the activation that feeds `down_proj`. **Calib already writes `act_absmean.mlp.switch_mlp.down_proj`
`[E, I]`** **[src, contract]**, so this needs no new calibration, only the down-proj column norms at build time.
- **Runtime constraint:** `SwitchGLU` stacks experts as `[E, I, D]`, so every expert in a layer must keep the same
  width. Per-layer widths need a shim like `build/shim.py` that overrides `moe_intermediate_size` per layer.
- **Granularity check:** 2048 → 1536 keeps group-64 alignment on `down_proj`'s input dim (1536/64 = 24), and
  640 → 512 does too.
- **Evidence** that structured expert slimming composes with expert trimming: He et al.
  ([arXiv:2406.02500](https://arxiv.org/abs/2406.02500) **[paper]**, Mixtral). There is **no** evidence on modern
  fine-grained MoEs, whose experts are already narrow (Qwen's 640 is tiny).
- **[hypothesis]** It is more promising for GLM's wide experts than for Qwen's.

**(b) Low-rank / delta compression against a shared base.** Write each expert as a shared component plus a
cheap per-expert part:
- **D²-MoE** ([arXiv:2502.17298](https://arxiv.org/abs/2502.17298), ICML 2025 **[paper]**): Fisher-weighted merge
  → base `W̄`; SVD of each delta `W_e − W̄` → rank-r factors; structured pruning of the base. Claims >13 % gains over
  other compressors on Mixtral, Phi-3.5, DeepSeek and Qwen2-MoE at 40–60 % compression.
- **MoBE** ([arXiv:2508.05257](https://arxiv.org/abs/2508.05257), ICLR 2026 **[paper]**): each up/gate matrix
  `= A_e · B_e`, where `B_e` is a mixture of basis matrices shared across the layer. −24 to −30 % params on
  Qwen3-235B, DeepSeek-V3 and Kimi-K2 with ~1–2 % absolute accuracy loss.
- **MoE-I²** ([arXiv:2411.01016](https://arxiv.org/abs/2411.01016) **[paper]**): inter-expert pruning, then
  intra-expert low-rank.

**What MoBE's numbers say.** Frontier experts are **not very low-rank relative to each other**: 30 % is the
achievable saving at ~1–2 % loss, against 50 % for REAP at a similar loss on large models. So this does not replace
pruning. It might stack on top.

**Why it would still be attractive on a Mac:** decode is bandwidth-bound. With `W_e = W̄ + U_e V_e`, a token reads
`W̄` once, plus k small factor pairs. For GLM at rank 256, a delta costs 256·(2048+4096) = 1.57 M params vs 8.4 M
for a full matrix (19 %). Expert bytes per token would fall from 8 × 25.2 M to roughly 25.2 M + 8 × 4.8 M ≈ 63 M
params, about 3× less **[computed]**. *But* `down_proj` sees a different `h_e` per expert, so `W̄_down h_e` must be
computed k times. That needs custom kernels, and MLX has no fused base+low-rank gather. **Stretch only (§7.6).**

### 2.8 Router handling after pruning

- **Slice, don't retrain.** Delete router rows (and GLM `e_score_correction_bias` entries) for pruned experts.
  `norm_topk_prob` renormalizes over the surviving top-k by itself **[src]**. For Qwen,
  `w_i = exp(l_i)/Σ_{j∈T'} exp(l_j)` does not depend on the softmax denominator over all experts, so slicing is
  exact. Forgetting to slice the bias in sync with the rows silently shifts every index: the result is garbage, but
  it loads fine. `build/pipeline.py` keeps router weights and bias at full precision (`ROUTER_KEEP_DTYPE`)
  **[src]**.
- **Leave the GLM correction bias as it is.** The bias exists to balance load during training. On a single Mac
  there is no expert parallelism, so imbalance costs nothing at inference. Re-tuning the bias to rebalance the
  survivors would change which experts are selected, a distribution shift with no upside. **[reasoned]**
- **Per-layer output scale ("healing").** An optional scalar per layer, fitted by *replaying the pruned routing*
  on calibration data. It has never been shown to beat no healing. Per-expert least-squares healing **hurts**
  end-to-end (GLM REAP50: top-1 0.8369 vs 0.8425, z = 11.8). A first-moment estimator that ignores `norm_topk_prob`
  over-corrects by ~30 % **[self-reported, reap.md §6]**. Keep the scalar only if it improves KL.
- **Top-k changes.**
  - *Narrowing for speed is a bad trade here.*
    - Qwen: routed experts are ~1.03 GB/token at 3.5 bpw, against ~3 GB/token of non-expert weights. 10 → 8 saves
      0.21 GB/token, at most ~5 %. Decode is latency-bound anyway **[computed; quant.md §4]**.
    - GLM: 8 → 6 saves 0.92 of ~10.4 GB/token, ~9 %.
    - Quality: sh0wie measured at K=288: top-10 91.5 %, top-6 84.8 %, top-4 63.4 % HumanEval
      **[self-reported]**.
    - Dynamic variants (NAEE skip-if-`g₂ < β·g₁`, [arXiv:2402.14800](https://arxiv.org/abs/2402.14800); Ada-K,
      [arXiv:2410.10456](https://arxiv.org/abs/2410.10456) **[paper]**) cut FLOPs on GPUs, but on this machine
      the saved bytes are small. MTP (1.7× on Qwen, decode.md) is the speed lever.
  - *Widening after pruning* (top-8 → top-9 among survivors) is a cheap **[hypothesis]**. It re-admits part of
    the substituted mass, for +12.5 % expert bytes per token. The router was never trained at k+1, so it may also
    hurt. The streaming sweep can test it at zero build cost (it is a routing override, §4.4).

### 2.9 Shared expert and MTP head

- **Shared expert: never prune or merge it.** It sees every token, so the REAP saliency logic does not apply. It is
  the one dense path that absorbs common features. Keep it at ≥ 6-bit (quant.md floors; GLM's vendor FP8 also
  quantizes it, so 6-bit is at or below vendor precision). Never use it as a merge target.
- **MTP head: speculative decoding is lossless, so pruning it costs speed, not quality.** Optimize *acceptance per
  GB* **[reap.md §7]**:
  - Qwen MTP (512 experts at 4-bit, ~1.4 GB): an unpruned head on a REAP-320 trunk accepted 76–81 %
    **[self-reported]**. Keep it intact unless memory is tight.
  - GLM MTP is a full MoE layer (~7.4 B params). A 288-expert 2-bit head (~2.4 GB) matched a REAP144 trunk ~70 % of
    the time at any bit width **[self-reported]**. A pruned 144-expert head has never been compared fairly.
  - mlx-vlm loads MTP as a separate drafter with its own `text_config`, so its expert count is independent of the
    trunk's **[src]**. `build.make --mtp {prune,intact,drop}` already exists **[src]**.
  - **Gap:** calib does not calibrate the MTP layer. `build` then falls back to data-free `weight_scores`
    (‖router row‖ × ‖down_proj‖) **[src, pipeline.py compute_scores]**. §4.1 adds an MTP calibration stage.

### 2.10 Attention layers and whole blocks

**Dense-model evidence that layers are redundant:**
- ShortGPT ([arXiv:2403.03853](https://arxiv.org/abs/2403.03853)) uses Block Influence `BI_ℓ = 1 − cos(x_ℓ,
  x_{ℓ+1})` and removes ~25 % of Llama-2 layers with modest MMLU loss, but generation degrades.
- Gromov et al. ([arXiv:2403.17887](https://arxiv.org/abs/2403.17887)) remove up to ~half of Llama-2-70B's deep
  layers before knowledge QA collapses, with QLoRA healing.
- "What Matters in Transformers?" ([arXiv:2406.15786](https://arxiv.org/abs/2406.15786)) finds many *attention*
  layers droppable in dense models.
- On MoE, He et al. ([arXiv:2406.02500](https://arxiv.org/abs/2406.02500)) report Layer Drop / Block Drop on
  Mixtral as part of a 6.05× combined speedup **[all paper]**.

**Why that does not transfer here:**
- **The attention layers are already sparse.** Qwen has 12 full-attention layers out of 48 (3:1 linear:full,
  `full_attention_interval=4`). GLM has 11 DSA layers out of 45 (`layer_types`) **[computed from config]**. These
  few layers carry long-range retrieval. Retrieval in transformers is concentrated in a small set of "retrieval
  heads" ([arXiv:2404.15574](https://arxiv.org/abs/2404.15574)), and in hybrids RULER recall rises with the share of
  full-attention layers ([arXiv:2507.06457](https://arxiv.org/abs/2507.06457)) **[paper]**. Dropping any of them
  attacks the long-document RAG workload directly.
- **Attention is a small share of the bytes.** GLM's whole non-expert trunk is 8.9 B params, ~6 B of it attention.
  Dropping 25 % of attention saves ~1.2 GB at 6.5 bpw, about 2.6 GLM expert slots **[computed]**. That is not
  worth the risk.
- **Whole blocks are worth more, and still dubious.** One GLM block at K=144 and 3.5 bpw holds 1.59 GB of experts
  plus ~0.1 GB of attention, about 3.7 expert slots per layer **[computed]**. With the hyper-connections (GLM mHC,
  Qwen `hc_count=4`), the block also *mixes the 4 residual streams*. "Skipping" a block has two non-equivalent
  forms: identity on every stream, or zero block output with the stream mixing kept. Both are untested.
  OpenMOSE's finding that layers 42–44 are the easiest to *prune experts from* points to non-uniform expert budgets
  as the safer use of that redundancy.

**Verdict:** no attention-layer pruning. Block drop goes in the sweep only as a free diagnostic (§4.4, §7.7).
Block Influence comes for free from calib's stored hidden states.

### 2.11 Prune first, then quantize

1. **Score in full precision.** Quantization noise perturbs `‖f_j‖` and the router logits. Ranking on the
   quantized model mixes two error sources. Our calib already scores on bf16 (FP8-dequantized for GLM) **[src]**.
2. **Calibrate quantization on the *pruned* model.** After pruning, each survivor receives the tokens its pruned
   neighbours used to get. Its input distribution, and so its imatrix, its NMSE sensitivity and the "experts with
   too few samples" fallback, all change. Today's calib computes `sens` and `act_absmean` under **unpruned**
   routing. Fix: a second, *sequential* calib pass with the keep-mask applied (§4.6). This is also what made REAM's
   sequential variant work **[paper]**.
3. **Distil last.** DWQ on scales and biases recovers part of the combined gap. Litwein recovered 33.6 % of the
   pruning gap with KL-DWQ **[self-reported, quant.md]**.
4. **One budget, two knobs.** The composable-compression benchmark finds expert pruning to be the dominant source
   of degradation when combined with quantization ([arXiv:2608.21693](https://arxiv.org/abs/2608.21693)
   **[paper]**). So the *pruning* ratio is the knob to set most carefully. §4.5 gives the joint rule.

### 2.12 Pruning vs streaming cold experts from SSD

mlx-vlm 0.7.9 ships `moe_offload`: experts are repacked into per-expert mmap files and kept resident in a
byte-budgeted LRU (`--expert-cache-gb`) **[src, `mlx_vlm/moe_offload.py`]**. Could we keep "pruned" experts on disk
instead of deleting them? The arithmetic decides it.
- **Decode budget.** Target ≥ 20 tok/s, i.e. 50 ms/token. GLM at K=144 already needs ~10 GB/token of weight
  reads, ~22 ms at ~470 GB/s, leaving ~28 ms for misses.
- **SSD.** A 2 GB `F_NOCACHE` test here gave **9.1 GB/s sequential read and 12.7 GB/s write [measured]**. That
  probably flatters sustained speed (SLC and controller cache). Plan with 6–9 GB/s.
- **Misses allowed.** 28 ms × ~7 GB/s ≈ 200 MB ≈ 18 GLM experts at 3.5 bpw (11.0 MB each), out of 42 × 8 = 336
  routed slots per token. **The hit rate must be ≥ ~95 %.**
- **What balanced routing gives.** Routing entropy is 0.92–0.99, so the "cold" half of the experts probably
  carries tens of percent of the slots on generic text. That points to ~5 tok/s. One related data point: the SSD-
  streamed Qwen OptiQ-2bit build runs at ~3.8 tok/s on an M3 Max **[self-reported, quant.md]**.
- **The open question** is whether *in-domain* (your coding sessions) routing is skewed enough. That is measurable
  offline from routing traces (§4.1 adds them; §7.2 is the stretch). Until a trace says otherwise, **prune, don't
  stream.**

---

## 3. Evidence: how much pruning each model and domain tolerates

### 3.1 Qwen3.8-Flash-Next (512 experts, top-10)

| Source | K kept | Finding |
|---|---|---|
| sh0wie ladder, MLX 4-bit, HumanEval-164 | 512/384/320/288/256 | 93.9/92.1/90.9/91.5/88.4 %. 288/320/384 are within noise. **Rare-name probe: 288 → 9/10, 256 → 1/10** (code-only calibration) **[self-reported]** |
| sh0wie Q8E | 288 | 8-bit 90.9 vs 4-bit 91.5: "pruning error dominates quantization error at this width" **[self-reported]** |
| AnonimousA | 256 | Corpus swap at the same bits, code-only → multi-domain: HumanEval **89.6 → 95.1 %**. The 256 floor is soft and depends on calibration **[self-reported]** |
| Litwein | 320, 3-bit, DWQ | KL20 0.104 overall. **Agent is the worst domain (0.185), code the best (0.060)** **[self-reported]** |
| WaveCut | 288 REAP vs REAM | KL 1.307 vs 1.256, tasks tied **[self-reported]** |
| ISTA GSQ-RCO (GGUF), **no pruning** | 512 | 2.4–2.5 bpw: 95.7 % task retention at 66–68 GB; 3.0 bpw: 99.4 % at 75.8 GB **[self-reported, quant.md]** |

### 3.2 GLM-5.3-Flash (288 experts, top-8)

| Source | K kept | Finding |
|---|---|---|
| pipenetwork, wikitext PPL | 216/181/144 | +0.20 / +0.34 / **+0.56** nats/token **[self-reported]** |
| patrickbdevaney REAP50 | 144 | top-1 agreement by domain: math 0.920, code 0.919, agentic 0.873, science 0.830, finance 0.755, **"ballast" 0.580** **[self-reported]** |
| lapadatbogdan, own-session calibration | 144 / 120 / 96 | 48/48 repo-task attempts at ×1.21 turns; **120 loops for tens of thousands of tokens; 96 unusable**; Python-only calibration broke other languages **[self-reported]** |
| OpenMOSE | 228 | KL 0.453, top-1 86.0 %, PPL +2.8 % **[self-reported]** |

### 3.3 Cross-model and per-domain pattern (primary sources)

| Domain | Tolerance | Evidence |
|---|---|---|
| **Math / formal reasoning** | highest | GLM-4.6 REAP MATH-500 95.5 → 97.0/94.8/93.3 at 25/30/40 %; Kimi-Linear REAP-30 % AIME25 30 → 40, MATH-500 81.8 → 80.8 **[self-reported, Cerebras]**; GLM-4.5-Air 50 % math 0.882 → 0.857 **[paper]**; GLM-5.3 REAP50 math top-1 0.920 |
| **Code (non-agentic)** | high, *if calibrated on code* | REAP on large models: −1.2 % at 50 % **[paper]**; Qwen3-Coder-480B 50 %: code 0.636 → 0.619 |
| **Agentic / tool use** | medium: the efficiency cost shows before the accuracy cost | Qwen3-Coder SWE-bench-V 54.0 → 52.2, τ²-telecom 50.0 → 47.1 at 50 %; GLM-5.3 at 144: same pass rate, ×1.21 turns with in-domain calibration and ×2.29 with a generic mix; Litwein agent KL worst |
| **Long-context reasoning** | medium-high | Kimi-Linear REAP-30 % at 128K: LongBench v2 36.8 → 37.2 **[self-reported]** |
| **Knowledge recall / rare entities / non-English** | lowest | GLM-4.6 GPQA 78.8 → 69.7 at 40 %; Kimi-Linear FRAMES 55.7 → 52.3; GLM REAP50 "ballast" top-1 0.58; Qwen rare-name collapse at 256 |

**Interpretation for this workload:**
- Coding and math survive aggressive pruning when calibrated for.
- Agentic work degrades first in *turns*, *loops* and *termination*, not in pass@1.
- RAG partly offsets the knowledge loss, because facts come from the documents. But exact copying of rare
  strings from context, such as identifiers, file paths and names, must be tested directly (§5).

### 3.4 Why the usual evals miss pruning damage

- Greedy pass@1 on 164 HumanEval problems cannot separate 3-point gaps.
- Retained saliency mass is circular: each mask wins on its own corpus.
- Pruning damage concentrates in the tail: rare tokens, other languages, stopping. KL on held-out
  workload text, sampled probes and turns-to-solve catch it (reap.md §8). §5 sizes these tests statistically.

---

## 4. Plan for this machine

### 4.0 Budgets and profiles (resident bytes are the constraint, not disk)

Expert bytes by kept count `K` and bpw. Per-slot params across all MoE layers: GLM 42 × 3·4096·2048 = **1.057 B**,
Qwen 48 × 3·2560·640 = **0.236 B** **[computed]**. Bytes = K × slot × bpw / 8.

| | 2.5 bpw | 3.0 | 3.25 | 3.5 | 4.25 | 4.5 |
|---|---|---|---|---|---|---|
| GLM K=144 | 47.6 | 57.1 | 61.8 | 66.6 | 80.9 | 85.6 GB |
| Qwen K=320 | 23.6 | 28.3 | 30.7 | 33.0 | 40.1 | 42.5 GB |
| Qwen K=512 (unpruned) | 37.7 | 45.3 | **49.1** | 52.8 | 64.2 | 67.9 GB |

**Fixed parts (from quant.md/reap.md):**
- GLM: trunk ~8.9 B at floored precision (~7–8 GB), MTP ~1.3–2.4 GB. GLM is fully resident.
- Qwen: ~4.6 GB of floored non-expert weights, ~1.0–1.4 GB MTP, and the **PLE table, 28.8 GB (nvfp4) or 32.0 GB
  (affine g32), memory-mapped and not resident** **[src, `ple_storage`]**.

**Profiles** (process cap 80 GB, ≥ 45 GB left for macOS and dev tools; KV ≤ 8 GB @128K per PLAN):

| Profile | GLM | Qwen |
|---|---|---|
| **Co-tenant default** (≤ 70 GB resident weights) | K=144 @ ~3.0–3.25 bpw → ~65–70 GB (reap.md table: 65.7 GB @ 3.0) | **Option A:** K=320 @ 3.0–3.5 → ~34–39 GB resident. **Option B:** K=512 @ 3.25 → 49.1 + 4.6 + 1.4 ≈ **55 GB resident** (88 GB on disk). **Option C:** K=384 @ 4.25 → 48.1 + 6 ≈ 54 GB resident |
| **All-in** (not coding, ~100 GB process) | K=192 @ 3.25 → ~92 GB (pipenetwork REAP37 ≈ K=181 was +0.34 vs +0.56 nats at REAP50) | K=512 @ 4.25–4.5 → 70–74 GB resident |
| **Lean** (headroom for big builds / 2 models) | K=128 @ 3.0 → ~59 GB (below the 144 "validated" floor: expect loops; sweep first) | K=288 @ 3.25 → ~33 GB resident |

**The Qwen insight.** PLAN.md caps *disk* at ≤ 70 GB. The co-tenant constraint is *resident* memory, and
reap.md already flagged that PLE streaming "matters more than any K choice". Drop the disk cap for Qwen (2.6 TB of
SSD is free) and the unpruned 512-expert model fits co-tenant at ~3.25 bpw. Whether that beats K=384 @ 4.25 or
K=320 @ 4.5 at the *same resident bytes* is the most important Qwen experiment in this plan (§4.5, E-Q1). One
caveat comes from the measurements in this chapter: more experts mean **fewer tokens per expert per prefill chunk**,
and that lowers MoE prefill efficiency on M5. At GLM shapes, 57 vs 160 tokens per expert ran 9.2 vs 12.3 TFLOP/s.
At Qwen shapes, 40 vs 160 ran 5.1 vs 14.8 TFLOP/s (`docs/plan/moe_density_bench.py`, bf16 SwitchGLU, GPU shared with a
running benchmark) **[measured]**. So K=512 prefills slower than K=320 at equal chunk size. Raise
`prefill_step_size` to compensate, which is a memory-versus-speed knob for the serving chapter.

### 4.1 Step 1: instrument calibration (no GPU needed; test on tiny fixtures)

New statistics, all added in `calib/probe.py: LayerRecorder._moe` and written by `calib/run.py: moe_stats`. Bump
`FORMAT_VERSION` to 2 and document them in `docs/contracts/calib.md`. Each is a few lines of numpy on arrays `_moe`
already has (`inds`, `weights`, `routed` norms):

| Key (`layer_{i}.npz`) | Shape | Purpose | Cost |
|---|---|---|---|
| `expert_reap_sq`, `.{d}` | `[E]` | `Σ(g‖f‖)²/count`, the HOPE diagonal and variance | trivial |
| `coact_sum.{d}`, `coact_cnt.{d}` | `[E,E]` f64 | HOPE `F` (conditional and unconditional). For each token, scatter-add the outer product of the k values `v_j = g_j‖f_j‖` | GLM 0.66 MB, Qwen 2.1 MB per domain |
| `expert_out_norm_max`, `expert_act_max` | `[E]` | super-expert guard (both statistics, see reap.md §1) | trivial |
| `route_trace.{d}` | `[T_s, k]` uint16 | routing trace for the first ~200K tokens per domain: LRU/offload simulation (§7.2), mask-switching studies | 200K × 8 × 2 B = 3.2 MB per layer per domain |
| `prune_curve` | `[n_k]` per domain | local `e_ℓ(k)` for k on a grid (GLM: 96…288 step 16; Qwen: 192…512 step 32), from re-running `switch_mlp` with the REAP keep-mask on the ~2,048 reservoir rows of `mlp.gate` inputs | one extra MoE forward per grid point on 2K rows: seconds per layer |
| `block_influence` | scalar per domain | `1 − cos(h_ℓ, h_{ℓ+1})` on sampled rows (averaged over streams) | trivial |

Also add an **MTP stage** after the head stage. It runs the MTP layer on (final hidden state, next-token
embedding) for the calib samples (both inputs are in `work/h_N` and the tokens) and records REAP/freq for the MTP
experts. That removes the data-free `weight_scores` fallback in `build/pipeline.py`.

Tests: in `tests/calib/`, use `tests/fixtures/make_tiny.py` (8 experts, top-2). Check `coact` against a brute-force
loop and `prune_curve(k=E)` = 0, and check that the REAP-mask curve is non-increasing in k.

### 4.2 Step 2: calibration data that decides the right experts

- **Domains:** `code`, `agentic`, `reasoning`, `longctx` (16K–64K), plus the small `general` floor that reap.md
  recommends (5–10 %) so arithmetic, prose and other languages don't collapse.
- **Your own sessions** (`~/.claude/projects/*/*.jsonl`: 1,570 files, 1.5 GB of JSONL, metadata only, contents
  not read). At a rough 3–4 bytes/token after stripping JSON overhead, that is on the order of **100 M tokens**
  **[rough estimate]**, far more than needed. Use 2–4 M.
  - **Why it is valuable:** the inputs (your repos' files, your build and test output, your prompting style)
    are exactly the traffic the model will see. In-domain session calibration gave ×1.21 turns vs ×2.29 for a
    generic mix on GLM **[self-reported]**.
  - **How:** add a `claude_sessions` source in `calib_data/sources.py` and render it in `calib_data/render.py`.
    Map `tool_use` → each model's native tool-call syntax and `tool_result` → the tool role. Teacher-force
    everything. The assistant turns are Claude's text, not the target model's, but routing statistics depend
    mostly on the *inputs* being in-domain. A later refinement regenerates the assistant turns with the target
    model.
  - **Hygiene:** scrub secrets first (API keys, `.env` contents, tokens: regex plus entropy filter). Split
    calib/eval **by session**, never by chunk. Keep everything local. Delete `work/` after a run, because hidden
    states are derived from private text.
  - **Weighting:** ~40 % of the `agentic` domain, with Toucan/OpenHands as the rest, so the mask isn't overfitted
    to one person's repos.
- **Volume:** a 1 M-token *screening* corpus (250K/domain) for the sweep, and a 4 M-token *final* corpus
  (≥ 1 M/domain; reap.md says ≥ 2M/domain is ideal for the final). Check stability with a split-half test
  (kept-set overlap ≥ 85–90 %) and a **bootstrap over sequences** (100 resamples). Experts kept in 5–95 % of
  resamples form the *uncertain band*. That band is the only place where HOPE, REAP-sum or per-domain cores can
  change anything.

### 4.3 Step 3: calibration pass and its cost

```bash
# screening pass (fp16 hidden states, 40 GB cap; co-tenant safe)
uv run python -m calib.run --model glm  --src $GLM_SNAP  --data artifacts/glm/data-screen  --out artifacts/glm/calib-screen  --mem-gb 40
uv run python -m calib.run --model qwen --src $QWEN_SNAP --data artifacts/qwen/data-screen --out artifacts/qwen/calib-screen --mem-gb 40
```

**Time per pass, with the arithmetic** **[computed from config + measured density bench; ±50 %]**:

*FLOPs per token.*
- **GLM:** per MoE layer, routed 8 × 25.17 M + shared 25.17 M + attention/hc ≈ 136 M → 362 M params. Over 42 MoE
  layers, 3 dense layers (~287 M each) and the head (0.63 B), that is ≈ 16.7 B active params, ≈ **33 GFLOP/token**.
- **Qwen:** per layer, routed 10 × 4.92 M + shared 4.92 M + ~77 M other → 131 M. Over 48 layers plus the head:
  ≈ 6.9 B active, ≈ **14 GFLOP/token**.

*Throughput.*
- The routed part runs at the measured **5–12 TFLOP/s** at chunk-2048 density (GLM ≈ 57 tokens per expert, Qwen
  ≈ 40). The dense part runs at ~20 TFLOP/s. Add ~30 % for probes, which sync to CPU numpy.
- **GLM:** (8.4 B·2/10e12 + 8.3 B·2/20e12) × 1.3 ≈ **3.3 ms/token → ~300 tok/s**.
- **Qwen:** (2.4 B·2/6e12 + 4.5 B·2/20e12) × 1.3 ≈ **1.6 ms/token → ~620 tok/s**.

*Hidden-state I/O* (fp16 per token: GLM 4 × 4096 × 2 = 32.8 KB, Qwen 4 × 2560 × 2 = 20.5 KB; each layer writes
and reads its boundary):
- GLM 1 M tokens = 32.8 GB per boundary → ~4 s write + ~5 s read at the measured 12.7/9.1 GB/s (call it ~12 s
  sustained). × 45 layers ≈ 9 min.

*Weights:* GLM 328 GB FP8 read once ≈ 40–60 s + dequantization. Qwen ~260 GB (the PLE table is row-gathered,
not read) ≈ 30–45 s.

*Sensitivity grid* (`calib/sens.py`, 12 qdq configs per tensor): memory-bound at ~25 B/element-config. GLM:
7.25 G expert elements/layer × 12 × 25 B ≈ 2.2 TB → ~5 s per layer at ~400 GB/s → ~4 min total. It is not a
bottleneck.

| Pass | GLM | Qwen |
|---|---|---|
| 1 M-token screening | ~55 min compute + ~10 min I/O ≈ **~70 min** | ~27 min + ~6 min ≈ **~35 min** |
| 4 M-token final | ≈ **4.5 h** | ≈ **2.2 h** |
| `longctx` extra: a 64K-token sample's DSA indexer is O(L²) (32 heads × 128 dim, kpool 4 → ~9 TFLOP per DSA layer, × 11 layers) | ~5–10 s per 64K sample | QSA indexer, similar order |
| SSD written per pass (hidden states) | 1 M: 1.5 TB; 4 M: **5.9 TB** | 1 M: 1.0 TB; 4 M: 3.9 TB |
| Peak memory | ≤ 40 GB (one GLM layer in bf16 ≈ 15 GB) | ≤ 40 GB (one layer ≈ 5 GB) |

**Two cheap speedups** (calib changes) **[hypothesis, from the density bench]**:
- **Pack short samples into B > 1 batches** so the experts see more tokens per call. 4× density gave 1.3–2.9× MoE
  throughput.
- **`--layers-per-stage 2`** for GLM (2 × 15 GB < 40 GB). It halves hidden-state I/O and SSD wear.

SSD wear matters: 5.9 TB per final GLM pass is a real fraction of a consumer SSD's rated endurance. Run screening
passes at 1 M tokens and only one final pass per model.

### 4.4 Step 4: the multi-mask streaming sweep (`calib/sweep.py`, new)

**Problem.** The pure-pruning KL of GLM K=144 cannot be measured by building it. A bf16 K=144 GLM is ~320 GB, and
a quantized build mixes in quantization error.

**Idea.** Pruning only changes routing and which weights exist. So, as calib does, load layer ℓ **once** and push
**N variant hidden streams** through it, each with its own keep-mask or routing override. At the end, apply
norm + `lm_head` to each stream and compare with `ref/` (the full-precision top-64 logprobs that calib already
writes) using the KL definition in `evals/kl.py`. Every error compounds through the layers, so this is true
end-to-end KL, measured under the exact model code.

```bash
uv run python -m calib.sweep --model glm --src $GLM_SNAP --calib artifacts/glm/calib-screen \
    --data artifacts/glm/data-screen --variants sweeps/glm-batch1.json --out results/sweep/glm-batch1.json
# variants file: [{"name":"reap-k144","mask":"masks/glm/reap-k144.json"},
#                 {"name":"reapsum-k144","mask":"masks/glm/reapsum-k144.json"},
#                 {"name":"hope-k144","mask":...}, {"name":"k144-top9","mask":...,"top_k":9},
#                 {"name":"k144-skip43","mask":...,"skip_layers":[43]}, ...]
```

Masks come from `build/prune.py`. Add `--emit-mask PATH` to `build.make --dry-run`: it writes
`{layer: [kept ids]}`, keeps `prune_loss` per layer, and makes no weights.

**Cost** (eval set 256K tokens = 64K per domain; N = 8 variants) **[computed]**:
- Compute: GLM 8 × 256K / 300 tok/s ≈ 1.9 h; Qwen 8 × 256K / 620 ≈ 55 min. Stream-level parallelism doesn't
  change FLOPs, but each layer's weights are read only once.
- Disk: GLM 256K × 32.8 KB = 8.4 GB per variant per boundary. 8 variants double-buffered is ~134 GB of scratch
  disk and ~0.75 TB written per batch. The head stage adds ~3 min.
- Memory: one layer (≤ 15 GB) plus chunk activations. Co-tenant safe.

Output per variant: overall and per-domain KL (top-64 + tail bucket), top-1 agreement, ΔNLL, plus a per-domain
bootstrap 95 % CI over *samples*, not tokens.

**Sweep grid** (3 batches per model; ~6 h GLM, ~3 h Qwen in total):

| Batch | GLM | Qwen |
|---|---|---|
| 1. ratio ladder (REAP-mean) | K ∈ {224, 192, 176, 160, 144, 128, 112, 96} | K ∈ {512*, 448, 384, 352, 320, 288, 256, 224} (*sanity: KL ≈ 0) |
| 2. criteria at the budget K | {REAP-mean, REAP-sum, REAP-shrunk, HOPE-cond, HOPE-unc, max-rank+cores, freq (control), global-norm} at K=144 | same set at K=320 |
| 3. structure | knapsack non-uniform (±32), top-9 among survivors, per-layer output scalar, skip block {43}, {42,43,44}, both skip semantics | knapsack non-uniform, top-11, REAM (if built), skip block {46} |

### 4.5 Step 5: find the knee, then decide jointly with bits

1. **Pruning curve.** From batch 1, fit `KL_p(K)` per domain. A monotone cubic spline in log-K is enough.
2. **Quantization curve.** From the quantization chapter's builds, or from `calib` NMSE propagated through the
   allocator, take `KL_q(b)` at the candidate K.
3. **Joint optimum.** Approximate additivity `KL(K,b) ≈ KL_p(K) + KL_q(b)` and *validate it on 3 built points*.
   Minimize `KL_p(K) + KL_q(b)` subject to `bytes = K·s·b/8 + fixed ≤ B`. With multiplier μ, the conditions are
   `−∂KL_p/∂K = μ·s·b/8` and `−∂KL_q/∂b = μ·s·K/8`:

   $$\underbrace{\frac{-\,\partial KL_p/\partial K}{s\,b/8}}_{\text{KL saved per GB spent on one more expert}}\;=\;\underbrace{\frac{-\,\partial KL_q/\partial b}{s\,K/8}}_{\text{KL saved per GB spent on more bits}}$$

   **At the optimum, a gigabyte buys the same KL reduction either way.** If an extra GB on experts saves more KL
   than on bits, prune less and quantize harder, and vice versa. sh0wie's "pruning error dominates at 288" says
   Qwen at 288 sits on the "prune less" side.
4. **Replace `--prune-lambda`.** `build.make --prune auto` currently adds `λ·Σ prune_loss^p` (REAP-mass units) to
   the quantization objective (NMSE units) **[src, pipeline.py search_auto]**. Calibrate λ so that one unit of
   `prune_loss` equals its measured KL. That is a linear fit of the batch-1 KL against `total_prune_loss` across
   masks.
5. **The knee as a fallback rule** when the bits side isn't measured yet: the largest prune ratio before the
   per-domain KL slope exceeds **2× its slope at 25 % pruning**, and no domain is worse than **1.5× the best
   domain's KL**. Both thresholds are **[hypothesis]**. Recalibrate them against the task evals from Step 7.

**Named experiments:**
- **E-Q1 (Qwen, equal resident bytes ≈ 55 GB):** K=512 @ 3.25 vs K=384 @ 4.25 vs K=320 @ ~4.5 (plus PLE nvfp4 vs
  affine g32). Decided on per-domain KL and §5 probes.
- **E-G1 (GLM, ≤ 70 GB):** K ∈ {128, 144, 160} with the allocator solving bits for each (≈ 3.4 / 3.0 / 2.7 bpw),
  as reap.md recommends.

### 4.6 Step 6: sequential re-calibration on the pruned model

```bash
# proposed flag: apply the mask while streaming, so stats/sens/imatrix reflect pruned routing and pruned upstream layers
uv run python -m calib.run --model glm --src $GLM_SNAP --data artifacts/glm/data-final \
    --out artifacts/glm/calib-k144 --keep-mask masks/glm/hope-k144.json
```

- **Why:** survivors absorb the pruned experts' tokens (§2.11), and layer ℓ+1 sees the *pruned* model's hidden
  states. The REAM paper found this sequential mode worth its +50 % time **[paper]**.
- **Optional iterative pruning:** prune layer ℓ, then rank layer ℓ+1's experts on the pruned upstream. This
  corrects the "independent layers" assumption in REAP/HOPE. **[hypothesis]**
- Cost: one final-size pass (GLM ~4.5 h, Qwen ~2.2 h). `ref/` stays from the *unpruned* run (copy it over),
  because KL must always be measured against the original model.

### 4.7 Step 7: build the finalists and run task evals

```bash
uv run python -m build.make --model glm --src $GLM_SNAP --calib artifacts/glm/calib-k144 \
    --prune 144 --score reap --budget-gb 68 --mtp prune --drop-vision --out models/glm53-k144-v1
for s in kl speed code math tools longctx; do uv run python -m evals.$s --model models/glm53-k144-v1; done
```

- **Budget:** 3 finalists per model × (build ≈ 15–30 min, dominated by reading 328/260 GB and writing ~65 GB
  **[estimate]**) + evals (KL ≈ minutes; task suites take hours).
- **Run alone:** builds and evals are GPU-heavy, so the orchestrator runs them one at a time (PLAN rule).
- **Non-uniform finalists** load through the `pruned_moe.py` shim (`build/shim.py`). Verify they load with
  `mlx_vlm.load` and with the server's loader.

---

## 5. Decision criteria and what to measure

| Metric | Tool | Accept a mask/K if | Why |
|---|---|---|---|
| Per-domain KL (top-64 + tail) and top-1 vs `ref/` | `calib.sweep` (pure pruning), `evals.kl` (built) | Better than the alternative with **bootstrap 95 % CI excluding 0 in ≥ 3 of 4 domains**, and **no domain worse** beyond its CI | Averages hide workload-specific failures [arXiv:2608.21693] |
| Rare-entity / identifier copy probe: names, paths and identifiers from the *context* reproduced exactly, 10 seeds each | `evals/longctx.py` + new probe set (≥ 200 items) | ≥ 95 % of the K=max reference | The 256 collapse showed only here (9/10 → 1/10) |
| Termination / loop rate: max-tokens hits, repeated n-grams, unclosed `<think>` | `evals/generation.py` on 300 agentic + 300 reasoning prompts | ≤ reference + 1 pp | GLM 120 looped while passing tasks |
| Tool-call validity and BFCL-style accuracy | `evals.tools` | within 2 pp of the reference, McNemar p > 0.05 | Tool-format tokens have specialist experts |
| Turns-to-solve and tokens-to-solve on agentic tasks | `evals.code` agentic mode | ≤ 1.25× the reference (lapadatbogdan K=144 own-calibration: 1.21×) | The efficiency cost comes before the accuracy cost |
| Math (AIME/MATH-500), code (HumanEval+/MBPP+) | `evals.math`, `evals.code` | within noise of the reference; use a **paired** test | Expected to be robust; a regression here signals a calibration bug |
| MTP acceptance and decode tok/s | `evals.speed` | maximize acceptance per GB | Pruning MTP is lossless; only speed is at stake |
| Prefill tok/s at 32K | `evals.speed` | record it (K affects expert density) | §4.0 density effect |
| Mask stability | bootstrap over calibration sequences | split-half overlap ≥ 85–90 % | Otherwise you're pruning noise |

**Sample sizes.** For paired pass/fail outcomes with discordance rate `d`, detecting a gap Δ at α = 0.05 with
80 % power needs `n ≈ (1.96 + 0.84)² · d / Δ²`. With d = 0.08 and Δ = 0.03, that is **n ≈ 700 items**. HumanEval's
164 can only resolve Δ ≈ 6 pp. So rank masks with KL (thousands of token positions per domain), and use task suites
as **guards** against regressions, not as the ranking. The same logic is why the fabrication probe at n = 12 proved
nothing (p = 0.22).

**Final decision rule per model:**
1. Choose K and the criterion by the §4.5 joint optimum on KL.
2. Reject it if any guard in the table fails, and step K up by one grid notch.
3. Among survivors within KL noise of each other, prefer the one with fewer turns-to-solve, then higher decode
   tok/s.

---

## 6. Failure modes (and how each is caught)

1. **Wrong gate in the saliency.** Reference REAP's softmax/topk-of-logits path is wrong for GLM (sigmoid +
   bias selection, un-biased renormalized weights × 2.5). With bias std 0.2, only 20–26 % of routing slots match
   **[measured, reap.md]**. Calib uses `arch.moe_weights`, the real path. Test: the tiny-fixture REAP must equal a
   brute-force recomputation from the mlx-vlm block's own outputs.
2. **Router rows, bias and expert tensors sliced inconsistently** (or the MTP drafter config claiming a different
   `n_routed_experts` than its tensors). The model loads and produces fluent garbage. Catch: a post-build check that
   every kept expert's router row and bias moved together, plus a 1K-token KL smoke test against `ref/` with a
   threshold.
3. **Super experts pruned.** Catastrophic, and invisible in mean saliency. Catch: force-keep by the
   `expert_out_norm_max` / `expert_act_max` rule (§4.1). Make the sweep include a "no guard" variant to measure how
   much the guard matters.
4. **Narrow calibration.** For example, Python-only calibration broke other languages "at every quant level", and
   C4 gave code 0 % **[self-reported / paper]**. Catch: the `general` floor domain, multilingual items in the eval
   set, and the per-domain KL guard.
5. **Circular metrics.** Retained saliency mass and calibration-set KL always favour the mask built on that data.
   Rank only on held-out `eval.jsonl` (split by session).
6. **Thin statistics.** An expert routed 3 times gets a noisy REAP-mean. Use shrinkage (§2.3), the bootstrap
   uncertain band, and ≥ 1 M tokens per domain for the final mask.
7. **Over-pruning past the knee.** GLM below 144 loops; Qwen below ~288 collapses on rare names with code-only
   calibration. Catch: termination and rare-entity guards. Never ship below the measured knee.
8. **Long-context blind spot.** Calibration at ≤ 4K tokens leaves long-context experts unmeasured (no public mask
   was calibrated long). Keep `longctx` 16K–64K in both calibration and eval, and check RULER 32K/64K/128K in
   `evals.longctx`.
9. **Per-expert healing / LoRA healing.** Per-expert healing fits reconstruction and loses end to end. Full-model
   LoRA healing diverged after ~16 steps **[self-reported]**. Only per-layer scalars, and only if KL improves.
10. **Non-uniform checkpoints fail in other runtimes.** Stock loaders size every `SwitchGLU` from one count. Ship
    the shim and test `mlx_vlm.load`, the server, and `moe_offload` (`_n_experts` reads the shapes).
11. **Prefill slowdown from more experts.** Lower tokens per expert per chunk makes prefill less efficient (§4.0).
    `evals.speed` at 32K catches it. Tune `prefill_step_size`.
12. **Machine-level risks.** Calibration fills SSD scratch (~66–134 GB per boundary for big sweeps) and writes TBs.
    Check free space before each run and use `--layers-per-stage 2`. The GPU is shared, so run one heavy job at a
    time, and keep calib at `--mem-gb 40` so it co-exists with coding (mac.md: `set_memory_limit` is soft, so the
    script must bound its own working set).
13. **Privacy leak through artifacts.** Hidden states in `work/` and `ref/` logprobs derive from your sessions.
    Keep them local and delete `work/`. Never push calibration artifacts to HF.

---

## 7. Stretch ideas (pushing limits, all feasible on this machine)

### 7.1 One knapsack for "drop / 2-bit / 3-bit / 4-bit" per expert

Instead of deleting the bottom experts, let each expert take one of {drop, 2b, 3b, 4b}, and solve a single
knapsack over (expert, option) for the whole model. Error units (per layer, expected squared output error per
token):
- quantize option q: `freq_e·k · NMSE_e(q) · E[(g_e‖f_e‖)²]`, using per-expert NMSE from `sens.mlp.switch_mlp.*`
  `[E, 6, 2]` (calib already writes it) and the new `expert_reap_sq`.
- drop: `freq_e·k · c_sub · E[(g_e‖f_e‖)²]`, with `c_sub ∈ [1, (1+ρ)²]` fitted from the `prune_curve` probe.

A 2-bit g64 expert costs 2.5/3.5 = 71 % of a 3-bit one but keeps most of its function, where a drop keeps none.
**Runtime:** `QuantizedSwitchLinear` has one bit width per stacked tensor **[src, quant.md]**. So split each layer
into 2–3 `SwitchGLU` tiers with an index remap: one shim class, one `gather_qmm` per tier. Costs and checks:
- **Qwen decode:** +2 kernel launches per projection per layer. Qwen decode is latency-bound at ~80–130 µs/layer,
  so measure the cost with `quant_bench.py`-style tiers before committing.
- **GLM decode:** bandwidth-bound, so tier overhead should be small.
- **Value:** the sweep evaluator tests the quality side by applying qdq in the variant stream.
- **[hypothesis]**, and in our view the most promising unexplored idea in this chapter.

### 7.2 Hot/cold tiers with SSD fallback (soft pruning)

Keep the top-K experts resident and the rest in `moe_offload`'s mmap store, so a token that "needs" a cold expert
pays an SSD read instead of a substitution error.
- **Decide offline first.** Replay `route_trace.{d}` through an LRU/LFU simulator for capacity C and get hit rate
  per domain. It costs seconds of CPU and needs no GPU.
- **Gate:** go only if the in-domain hit rate is ≥ 95 % for GLM (§2.12 arithmetic) and the measured decode stays
  ≥ 20 tok/s.
- **Prefill:** at worst it touches every expert in a chunk. Precompute the chunk's needed set and prefetch, or
  route prefill only to resident experts. That is a hybrid: prune for prefill, exact for decode.

### 7.3 REAM merging for Qwen at moderate ratios

At K=384/352, merge each pruned expert into its centroid after neuron alignment:
- **Statistics:** gate-logit similarity is recomputable at build time from the reservoir rows (logits = X·W_rᵀ).
  Output similarity comes from running both experts on the same rows. Raise `--expert-tokens` to 512 for this.
- **Alignment:** `scipy.optimize.linear_sum_assignment` (scipy 1.18 is installed). On 640×640 cost matrices it
  takes milliseconds per pair.
- **Gate:** accept only if KL improves *and* tasks don't regress. The 288 precedent was KL-only.
- **Not for GLM at 50 %:** REAP beat REAM there.

### 7.4 Workload-switchable expert packs

Domain masks differ (Spearman 0.714). Ship the *union* of the code-pack and RAG-pack experts on disk, and let the
server swap the resident subset per profile.
- **GLM at K=144:** 144 × 42 × 11 MB = 66 GB per pack. If packs differ by 20 %, a swap moves ~13 GB, about
  ~1.5–2 s from SSD at the measured 9 GB/s.
- **What it buys:** a "coding GLM" and a "long-doc GLM" in one 68 GB resident footprint. The sweep evaluates
  each pack on its own domain.

### 7.5 Layer-local router re-fit on cached hidden states

Hold everything frozen except the surviving router rows of layer ℓ. Minimize
`‖y_full(x) − y'_ℓ(x; W_r)‖²` on cached layer inputs: the selection is non-differentiable, but the weights are
differentiable. That is 288 × 4096 = 1.2 M params per GLM layer, trained layer by layer inside the calib stream.
It is a far smaller, more stable problem than the full-model LoRA that diverged. Accept it per layer only if held-out
local error falls *and* end-to-end KL in the sweep falls. **[hypothesis]**

### 7.6 Expert delta spectra, as a measurement before building anything

From the source safetensors (CPU only, one layer at a time, later when allowed), compute for a GLM layer:
- the mean expert `W̄` and its alignment-permuted version;
- the energy captured by rank-r SVD of `W_e − W̄` for r ∈ {64, 128, 256, 512}.

If rank 256 captures < ~80 % of delta energy, D²-MoE/MoBE-style compression can't beat pruning here, and the idea
is dead in an afternoon. If it captures > 95 %, it becomes a serious option for bandwidth-bound GLM decode (§2.7b),
worth a custom Metal kernel.

### 7.7 Block drop and stream-mixing semantics as a diagnostic

Use the sweep's `skip_layers` variants with both mHC semantics (§2.10) to measure how much each block matters.
Even if no block is ever dropped, this per-block KL importance is the best available prior for the non-uniform
expert knapsack weights `w_ℓ` (§2.5).

### 7.8 Self-generated calibration

After a first good build exists, regenerate the assistant turns of your session prompts with the *unpruned* model,
running it streamed (slow but feasible overnight) or with the all-in profile. Then recalibrate. Calibration tokens
in the model's own distribution, `<think>` and tool syntax included, route like real traffic (reap.md §4). That
makes the second-generation mask the one tuned to how *this* model actually works on *your* code.

---

## Sources

- REAP: [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), [code](https://github.com/CerebrasResearch/reap) · HOPE: [arXiv:2609.18916](https://arxiv.org/abs/2609.18916) · REAM: [arXiv:2604.04356](https://arxiv.org/abs/2604.04356) · EvoESAP: [arXiv:2603.06003](https://arxiv.org/abs/2603.06003) · Composable compression benchmark: [arXiv:2608.21693](https://arxiv.org/abs/2608.21693) · llm-compressor global REAP: [PR #3281](https://github.com/vllm-project/llm-compressor/pull/3281)
- Merging/compression: MC-SMoE [arXiv:2310.01334](https://arxiv.org/abs/2310.01334) · HC-SMoE [arXiv:2410.08589](https://arxiv.org/abs/2410.08589) · D²-MoE [arXiv:2502.17298](https://arxiv.org/abs/2502.17298) · MoBE [arXiv:2508.05257](https://arxiv.org/abs/2508.05257) · MoE-I² [arXiv:2411.01016](https://arxiv.org/abs/2411.01016) · Unified MoE compression / Layer & Block Drop [arXiv:2406.02500](https://arxiv.org/abs/2406.02500) · Git Re-Basin [arXiv:2209.04836](https://arxiv.org/abs/2209.04836) · Wanda [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) · Super Experts [arXiv:2507.23279](https://arxiv.org/abs/2507.23279)
- Routing: aux-loss-free balancing [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) · NAEE [arXiv:2402.14800](https://arxiv.org/abs/2402.14800) · Ada-K [arXiv:2410.10456](https://arxiv.org/abs/2410.10456)
- Layers/attention: ShortGPT [arXiv:2403.03853](https://arxiv.org/abs/2403.03853) · Gromov et al. [arXiv:2403.17887](https://arxiv.org/abs/2403.17887) · What Matters in Transformers [arXiv:2406.15786](https://arxiv.org/abs/2406.15786) · Retrieval heads [arXiv:2404.15574](https://arxiv.org/abs/2404.15574) · Hybrid linear attention analysis [arXiv:2507.06457](https://arxiv.org/abs/2507.06457)
- Model-specific community evidence (all **[self-reported]**): see the links in `docs/research/reap.md` §3 and `docs/research/quant.md` §3 (sh0wie, AnonimousA, Litwein, WaveCut, ISTA GSQ-RCO, pipenetwork, patrickbdevaney, lapadatbogdan, OpenMOSE, Cerebras).
- Local: `build/prune.py`, `build/pipeline.py`, `build/shim.py`, `build/make.py`, `calib/probe.py`, `calib/run.py`, `calib/sens.py`, `docs/contracts/calib.md`, `.venv/.../mlx_vlm/moe_offload.py`, the real `config.json` of both models; measurements from `docs/plan/moe_density_bench.py` and a 2 GB `F_NOCACHE` SSD test (this chapter).
