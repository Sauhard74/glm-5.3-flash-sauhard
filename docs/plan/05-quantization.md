# 05 — Quantization: sensitivity-driven mixed precision and beyond

> Part of the master plan for the best co-tenant local model on this M5 Max (128 GB, 614 GB/s, macOS 27).
> This chapter **builds on** `docs/research/quant.md` (verified; cited as *quant §n* / *quant Rec n*),
> `docs/research/{reap,decode,mac}.md`, and the sibling chapters `01-memory.md` (budgets, hot/cold tiers) and
> `04-pruning.md` (what to cut before quantizing). It does not repeat their verification work, and it does not
> overturn any finding in *quant*. Where it adds something, the new evidence is shown.
>
> Tags: **[paper]** arXiv or peer-reviewed · **[src]** read in this repo or the installed packages (mlx 0.32.4,
> mlx-lm 0.32.2, mlx-vlm 0.7.9) · **[self-reported]** a model-card / README / PR number, not reproduced ·
> **[measured]** run on this machine for this chapter (tiny, no model loaded; script in §8) · **[computed]**
> arithmetic from real `config.json` / safetensors parameter counts · **[est]** an estimate with stated assumptions
> · **[hypothesis]** a proposal with no published evidence yet.
>
> Units: GB = 10⁹ bytes (what `--budget-gb` means), GiB = 2³⁰ (RAM caps). Same as *01-memory*.

---

## 0. TL;DR (what this chapter decides)

1. **Affine is the format for every weight; scales/biases cost exactly `32/g` bits per weight.** mxfp4 is ~60 %
   worse than affine 4b g64 on real Qwen3.8-Flash-Next experts, nvfp4 ~24 % worse, mxfp8 ~14× worse than affine
   8b g64 **[measured, §3.1]**. None decodes faster on M5 (*quant §4*), and DWQ can only train affine scales and
   biases. The one exception is the 51 B PLE table, which may use nvfp4 with pre-scaling (*quant Rec 6*).
2. **Use g128, not g32, as the cheap/expensive knob for experts.** For RTN, going g64 → g128 saves 0.25 bpw and
   costs +23 % error. Spending those 0.25 bpw on bits instead would buy ×1.41 less error. g32 costs +0.5 bpw for
   only ×1.27 less error, where 0.5 bpw of bits buys ×2 **[measured + derived, §2.2]**. So for routed experts the
   efficient menu is **{2b g64, 3b g128, 3b g64, 4b g128, 4b g64}**. `calib/sens.py` must add g128; it measures
   only {32, 64} today.
3. **Bit allocation is a multiple-choice knapsack, and `build/alloc.py` already solves it exactly, but in the
   wrong units.** The objective adds *relative* NMSE across tensors with weight 1. The correct objective is
   expected ΔKL ≈ Σ_t a_t·NMSE_t·E_t: amplification × relative error × output energy (§2.3). Fix the weights
   first. It is cheap and changes every allocation the solver makes.
4. **Rounding beats rotation for these weights.** Routed experts are near-Gaussian (kurtosis 3.8–3.9). A signed
   Hadamard rotation cuts their 3-bit error by only ~7 %, while a least-squares scale/bias refit cuts it by **19 %**
   (2-bit: 32 %) at zero runtime cost **[measured]**. Rotation removes 36–44 % of the error on the heavy-tailed
   dense tensors (attention q/o, shared experts: kurtosis 13–55). But those sit at ≥ 6-bit floors, where the error
   is already negligible. Rotation also **hurts** the outlier-dominated MTP `fc_hidden` (kurtosis 2040). Order of
   work: **LSQ refit → imatrix-weighted / GPTQ rounding → DWQ → (stretch) online Hadamard on `down_proj`**.
5. **Protect by measured physics, not folklore.** Routers and linear-attention decay/β gates stay bf16/fp32
   (cheap, and they act through discontinuous or exponential maps). Hyper-connection read gates, embed, lm_head,
   MTP connectors get ≥ 8-bit. Attention, DeltaNet/KDA projections and shared experts get ≥ 6-bit. All of these
   together are ≈ 3 % of params and ≈ 7–11 % of bytes (§2.4).
6. **Hot/cold bit tiers are the biggest *allocation* win left** (half a bit per doubling of routed importance,
   −43 % idealized routed error at equal bytes, *01 §2.5c*). Stock MLX holds one format per stacked expert tensor,
   so this needs the tiered `SwitchGLU` shim from *01 §4.5* / *04 §7.1*. The optimal tier assignment is
   **monotone in importance** (rearrangement inequality, §2.5), so it reduces to choosing cut points.
7. **DWQ when the teacher doesn't fit:** store the teacher's **final normed hidden state** (Qwen 5.1 KB/token,
   GLM 8.2 KB/token) and recompute exact full-vocab logits with a 1.3 GB bf16 head. That is exact, and costs less
   than mlx-lm's top-1024 targets (6.1 KB/token). Run **blockwise local DWQ inside the calib stream** (≤ 25 GB,
   ~30 min for GLM) first, then optionally an end-to-end band DWQ in the All-in profile (§2.8).
8. **KV-cache quantization stays off** (*decode §2*). It crashes in the server's batch path for both
   architectures, is a no-op or crash for GLM, and would save only ~2.3 GB at 128K for Qwen (§2.10).
9. **Recipes (§4.6).** GLM-5.3-Flash K=144: experts ≈ 3.10 bpw, total **68.0 GB**. Expected quant-only KL
   ≤ 0.10 on top of a pruning KL that dominates. Qwen3.8-Flash-Next **unpruned 512**: experts ≈ 3.58 bpw, **59.7 GB
   resident** with the PLE mmapped (the *01 W1* build), target KL ≤ 0.08 after DWQ. If the 68 GB budget must also
   cover the PLE on disk: K=352 @ 3b g128, 67.8 GB. Verify every recipe with the sweep's *simulated* qdq KL before
   building, and with `evals.kl` after.

---

## 1. Why it matters (intuition)

A weight matrix stored in bf16 spends 16 bits on every number. Quantization replaces each number with a few-bit
integer code plus a little shared metadata, and accepts a small, controlled error. On this machine that error
buys two things:

- **Capacity.** Every bit you don't spend on a weight can hold more model. GLM-5.3-Flash has 304 B routed-expert
  params. Dropping their average by one bit frees 38 GB, which is more than half the whole budget.
- **Speed, sometimes.** Batch-1 decode reads every active weight once per token. Where decode is bandwidth-bound
  (GLM's 2048×4096 experts), 3-bit experts decode 15–20 % faster than 4-bit (*quant §4*, re-measured). Where it
  is latency-bound (Qwen's 640×2560 experts), bits barely matter for speed.

Quantization is not one decision but four nested ones, and this chapter takes them in order:

1. **Format.** How a group of numbers is encoded: affine integer with scale and bias, or FP4/FP8 with a shared
   exponent.
2. **Allocation.** Which tensor gets how many bits. This is where most of the quality is won or lost: 3 % of the
   params (the "skeleton": attention, gates, embeddings) cause most of the damage if crushed (*quant §3*,
   pipenetwork table: +20.6 % PPL uniform 4-bit → +1.3 % with those 3 % at 8-bit **[self-reported]**).
3. **Rounding.** Given a format and a bit width, which codes to pick. Round-to-nearest (RTN) is not optimal.
   Fitting the scale and bias, or letting later columns absorb earlier rounding errors (GPTQ), lowers error at
   *identical bytes and identical speed*.
4. **Distillation.** Fine-tune the few continuous knobs left (scales, biases) so that the whole network's *output
   distribution* matches the teacher, not just each matrix.

Three facts about our two models shape every choice below:

- **The bytes live in the routed experts** (97 % of GLM params, 68 % of Qwen's, 96 % excluding Qwen's mmapped PLE
  table **[computed, *quant* Verification]**). So the expert bit width sets the size, while the skeleton's
  precision sets most of the quality.
- **Experts are well-behaved numbers.** Our measurement on real Qwen3.8-Flash-Next weights shows routed experts
  at kurtosis ≈ 3.9 (a Gaussian is 3.0). The dense skeleton is heavy-tailed (13–55; the router 1378). That is why
  outlier-killing tricks (rotations) matter far less for the experts than the literature on dense LLMs suggests,
  and why smarter rounding matters more (§3.1).
- **Routing is skewed within a workload.** The hottest 25 % of Qwen experts carry 66 % of routing mass (*01 §2.5a*,
  measured). A bit spent on a hot expert is worth several bits spent on a cold one.

---

## 2. How it works (mechanics + key math)

### 2.1 Formats in MLX 0.32.4, exactly

`mx.quantize(w, /, group_size=None, bits=None, mode='affine', *, global_scale=None, stream=None) -> (w_q, scales,
biases)`, read from the installed docstring **[src]**:

| mode | group | bits | scale | bias | bits/weight |
|---|---|---|---|---|---|
| affine | 32, **64**\*, 128 | 1, 2, 3, **4**\*, 5, 6, 8 | same dtype as input (bf16 here) | yes, same dtype | `b + 32/g` |
| mxfp4 | 32 | 4 (E2M1) | E8M0 (power of two) | no | 4.25 |
| nvfp4 | 16 | 4 (E2M1) | E4M3 (+ optional fp32 `global_scale`) | no | 4.5 |
| mxfp8 | 32 | 8 (E4M3) | E8M0 | no | 8.25 |

**Affine mechanics.** Each group of `g` consecutive input-axis weights gets its own line. With `α = max w`,
`β = min w`, the scale is `s = (α−β)/(2^b−1)`, the codes are `q_i = round((w_i−β)/s) ∈ [0, 2^b−1]`, and the
dequantized value is `ŵ_i = s·q_i + β`. Codes are bit-packed into uint32, lowest bits first.

**Why FP4 loses to affine on weights.** E2M1 has only 8 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}, and they crowd
near zero. An E8M0 scale can only be a power of two, so up to half the code range can be wasted when the group max
falls just above a power of two. nvfp4's E4M3 scale fixes the second problem, and its 16-wide groups help with
outliers, which is why it lands within ~20 % of affine 4b g64. FP formats exist for *activation* quantization on
tensor-core hardware (`mx.qqmm`), which this plan does not use (*quant §1*).

**Exact size accounting.** For a tensor with `R` rows (all leading axes flattened, e.g. experts × out) and input
width `C`:

$$\text{bytes} = \underbrace{R\cdot \tfrac{C\,b}{32}\cdot 4}_{\text{packed codes}} \;+\; \underbrace{2\cdot R\cdot \tfrac{C}{g}\cdot 2}_{\text{bf16 scale + bf16 bias}} \quad\Rightarrow\quad \text{bpw}=b+\frac{32}{g}$$

This is `alloc.quantized_bytes` **[src]**, and the build asserts written bytes = predicted bytes. Worked example,
Qwen `switch_mlp.gate_proj` `[512, 640, 2560]` at 3b g128: codes 512·640·2560·3/8 = 314.6 MB, plus scales+biases
2·512·640·20·2 = 26.2 MB, total 340.8 MB = **3.25 bpw** over 838.9 M weights **[computed]**. Constraints: `C` must
divide by `g`, and `C·b` by 32. Both hold for every expert width here (640, 2048, 2560, 4096). The PLE rows are 160
wide, so g64 and g128 are impossible there and only g32 or nvfp4 g16 work (*reap §2*).

The metadata share is `(32/g)/(b+32/g)`: 20 % of a 2b g64 tensor is scales and biases, 7.7 % of a 3b g128 one
(*01 §2.1a*). At low bits, group size is a first-order cost.

**Speed on M5** (from *quant §4*, re-measured there; not re-run here because a 44 GB server is up):

- **Prefill.** NAX kernels exist for every affine (bits × group) and for the FP modes. MoE prefill speed is within
  ±10 % across formats.
- **Decode.** There are no NAX kernels for decode.
  - GLM shape: bandwidth-bound. 4b g64 ≈ 226–247 µs/layer, 3b g64 192–200, 2b g64 144–153.
  - Qwen shape: latency-bound at 75–130 µs/layer for every format.
  - mxfp4 and nvfp4 show no decode advantage.
- **MTP verify (T ≤ 8).** mlx-vlm fuses affine 4/5-bit only, yet 3b g64 is still not slower at T = 4.

### 2.2 Group size: a rate–distortion argument

For one group, the step is `s = range/(2^b−1)` and RTN error is ≈ uniform on ±s/2, so MSE ≈ s²/12. For
Gaussian-like weights, the expected range of `g` samples grows like `2σ√(2 ln g)`. Hence:

$$\text{MSE}(b,g)\;\propto\;\frac{\ln g}{(2^b-1)^2}$$

- **One bit** cuts MSE by ≈ 4× (6 dB) and costs 1 bpw.
- **Halving g** cuts MSE by `ln g / ln(g/2)` and costs `32/g` bpw more.

So, per bpw spent:

| move | Δbpw | MSE ratio (theory) | **measured, Qwen experts, RTN** | the same Δbpw spent on bits |
|---|---|---|---|---|
| g64 → g128 | −0.25 | ×1.17 worse | ×1.23 (3b), ×1.25 (4b) | ×1.41 worse |
| g64 → g32 | +0.50 | ×1.20 better | ×1.27 (3b), ×1.29 (4b) | ×2.0 better |
| 3b → 4b @ g64 | +1.00 | ×4 better | ×4.4 | — |
| 2b → 3b @ g64 | +1.00 | ×4 better | ×4.3 | — |

**Reading it.** Moving to g128 is the *cheapest* way to save bytes on Gaussian-like tensors: you lose less than
you would by dropping bits for the same saving. g32 is the *most expensive* way to buy quality. So the convex hull
of (bytes, error) for experts is {2b g64 2.5, 3b g128 3.25, 3b g64 3.5, 4b g128 4.25, 4b g64 4.5}, and g32
falls off it. Heavy-tailed tensors gain more from small groups (q_proj: g64 → g32 ×1.46), but still less than ×2.
**[measured, §3.1; derived]**

With LSQ refit (§2.6), 3b g128 reaches 0.0387, the same as RTN 3b g64 (0.0389), for 0.25 bpw less. Better rounding
moves the hull, so the allocator must see sensitivities measured *with the rounding the build will use*.

### 2.3 Bit allocation as a multiple-choice knapsack (derivation)

**Setup.** Units `t = 1..N` are the quantizable tensors. For GLM K=144 that is ≈ 45 layers × ~10 per layer plus
the MTP head, about 600 units; Qwen has ~1,000. Each unit has options `o ∈ O_t` (format, or bf16), each with exact
bytes `B_{t,o}` (§2.1) and distortion `D_{t,o}`. The problem:

$$\min_{o_t\in O_t}\;\sum_t w_t\,D_{t,o_t}\quad\text{s.t.}\quad \sum_t B_{t,o_t}\;\le\; B_\text{budget}-B_\text{fixed}$$

This is the multiple-choice knapsack problem (MCKP).

**Lagrangian view (why greedy-on-the-hull is almost exact).** Relax the budget with a price λ per byte. Each unit
then solves its own problem, `min_o w_t D_{t,o} + λ B_{t,o}`, and picks a vertex of the lower convex hull of its
(B, wD) points. Sweeping λ from ∞ to 0 moves each unit up its hull one vertex at a time, in order of
`−Δ(wD)/ΔB`. At the optimum, **every unit's last upgrade has the same marginal "error saved per byte"**. That is
the equal-slope condition; *04 §4.5* uses the same idea to trade experts against bits. The greedy hull walk solves
the LP relaxation exactly, and the integer gap is at most one unit's single step. `alloc.solve` runs this greedy,
an exact DP over 2¹⁷ byte buckets, and a leftover-byte fill, and keeps the better result **[src]**. With ~6–9 k
(unit, option) items the solver takes ~3 s (*build contract*).

**What D and w must be: from KL to per-tensor terms.** Let `θ` be all weights and `δ` the quantization
perturbation. The student's per-token KL against the teacher is zero, with zero gradient, at `δ = 0`. So to second
order:

$$\mathbb E\,\mathrm{KL}(p_\theta\,\|\,p_{\theta+\delta})\;\approx\;\tfrac12\,\delta^\top F\,\delta,\qquad F=\mathbb E_{x}\,\mathbb E_{y\sim p_\theta}\big[\nabla\log p\,\nabla\log p^\top\big]$$

`F` is the Fisher information: the KL Hessian at the teacher is the Fisher. Push the perturbation forward to each
tensor's output `y_t = W_t x`, so `δy_t = ΔW_t x`. Two assumptions keep the sum diagonal:
(i) rounding errors in different tensors are uncorrelated, so cross terms vanish in expectation;
(ii) within a tensor, the error direction is uncorrelated with the output-gradient direction.
Then:

$$\mathbb E\,\Delta\mathrm{KL}\;\approx\;\tfrac12\sum_t \underbrace{\frac{\mathbb E\|\nabla_{y_t}\log p\|^2}{d_t}}_{a_t:\ \text{amplification}}\cdot\underbrace{\frac{\mathbb E\|\Delta W_t x\|^2}{\mathbb E\|W_t x\|^2}}_{\text{NMSE}_t\ (\texttt{calib sens})}\cdot\underbrace{\mathbb E\|W_t x\|^2}_{E_t:\ \texttt{out\_sq}}$$

So the right weight is **`w_t = a_t·E_t`**, and `D_{t,o}` = NMSE as calib measures it. It follows that:

- **Today's objective is in the wrong units.** `build/pipeline.py options_for` uses weight 1 per unit (0.25 for MTP)
  and `sens**p`, so Σ NMSE treats a 1 % error in a 123 K-param DeltaNet β-gate the same as a 1 % error in a
  151 M-param expert stack, whatever each one contributes to the output **[src]**.
- **`p` should be 1.** NMSE is already a squared error, and independent errors add in energy. The CLI default is
  `--sens-power 1.0`, but the contract says "p=2 by default". Fix the contract **[src]**.
- **Experts aggregate by importance, not frequency.** A stacked `switch_mlp.*` unit's NMSE should average the
  per-expert NMSE with weights `freq_e·E[g_e²‖f_e‖²]`, the expert's share of MoE-output energy. Today
  `aggregate_expert_sens` weights by `freq` only **[src]**. The pruning chapter's `expert_reap_sq` statistic
  (*04 §4.1*) is exactly `E[(g‖f‖)²]`.

**Getting `a_t` without a resident model.** Three levels, cheapest first:

1. **Energy-normalized (no new pass).** `a_t ≈ a_kind / E‖h_ℓ‖²`, where `h_ℓ` is the residual stream at the
   tensor's layer. A perturbation's effect is set by its size *relative to the residual it is added to*. Calib
   must log `resid_sq[ℓ]`; this is a one-line probe.
2. **Fitted per (kind, depth band) from the sweep.** *04 §4.4* already specifies `calib/sweep.py`, which streams N
   variants through each layer at once. Add qdq variants: "quantize all `attn` in layers 0–11 to 3b g64, nothing
   else", and so on, about 12–16 variants. Then fit `ΔKL_v ≈ ½ Σ_{t∈v} a_{kind,band} NMSE_t E_t` by non-negative
   least squares. One GLM batch of 8 variants takes ~2 h (*04 §4.4*).
3. **Streaming backward Fisher (the exact `a_t`).** After the forward calib pass, every layer's input is on disk.
   Run a *reverse* stream: at the head, sample `y ~ p_teacher` and take `∂(−log p(y))/∂h_N`. Then, for
   ℓ = N−1…0, load layer ℓ, recompute its forward from the stored `h_ℓ`, and take one `mx.vjp`. That gives the
   cotangent at every module output (accumulate `E‖·‖²`, giving `a_t`) and the cotangent for `h_ℓ`, which carries
   on to the next layer down. Memory: one layer plus the cotangents. Cost ≈ 2–3× a forward pass, ~2.5 h for 1 M GLM
   tokens. Caveat: mlx-vlm's `gated_delta_update` defaults to a `metal_kernel` with no VJP. Use
   `use_kernel=False` (the ops path, `mlx_vlm/models/gated_delta.py`) for this pass **[src; must verify on the tiny
   fixture]**.

Validate the additivity assumption: on 3 built points, compare `Σ_t` predicted ΔKL with measured KL (*04 §4.5*
does the same for pruning). Expect under-prediction at ≤ 2.5 bpw, where errors stop being small.

### 2.4 Which tensors to protect, and why

The common reason: a tensor deserves protection when its error is *amplified non-linearly* (selection, exponent,
every-token use) or when it is cheap. The per-param sensitivity of the non-expert skeleton is ~20× that of the
experts (*quant §3*).

| tensor | treatment | why (mechanism) | bytes at stake |
|---|---|---|---|
| **Router** `mlp.gate` (+ GLM `e_score_correction_bias`, fp32) | **bf16/fp32, pinned** (`ROUTER_PIN`) | Top-k selection is discontinuous. A logit error comparable to the gap between the k-th and (k+1)-th expert *swaps an expert*, which is the full substitution error of *04 §2.2*, not a small perturbation. Measured kurtosis **1378**: a few huge entries make RTN poor (8b g64 NMSE 1.4e-4, ~3× a Gaussian tensor) | Qwen 63 M → 0.13 GB; GLM K=144 0.10 GB (fp32) |
| **Linear-attention gates**: Qwen `in_proj_a` (decay) / `in_proj_b` (β); GLM `fbg_a_proj`, `f_b_proj`, `g_b_proj` | **bf16** (Qwen a/b), **≥ 8-bit** (GLM low-rank f/g/b) | The state recurrence is `S_t = exp(−A·softplus(a_t))·S_{t−1} + β_t k_t v_tᵀ`. A bias δ in log-decay compounds over L tokens as `e^{Lδ}`, so a 0.1 % decay error changes retention by ~e^{0.001·L}: invisible at 1K, large at 64K+. That is the long-doc RAG workload. Vendors keep all of these BF16 (*quant §3*) | Qwen a/b: 36 × 2 × 48 × 2560 = 8.8 M params (18 MB). Negligible |
| `A_log`, `dt_bias`, conv1d, norms, hc mixing scalars | never quantized (`cast_predicate`) | Exponent parameters; 1-D | negligible |
| **Hyper-connection read gates** `input_mix_weight_{down,up}`, `block_inject_weight` | **≥ 8-bit g64** | They decide how the 4 residual streams are mixed into every block's input. Pipenetwork: 8-bit on these alone cut uniform-4-bit PPL damage from +20.6 % to +11.3 % **[self-reported]** | Qwen 0.64 B → 0.68 GB at 8.5 bpw |
| **Embed** / **lm_head** | **≥ 8-bit** (code floors today: lm_head ≥ 6, embed ≥ 4; raise) | lm_head error lands on *every logit of every token* and directly sets the KL. An embedding error is per-token, but it is the first input of the whole chain | Qwen 1.28 B → 1.36 GB; GLM 1.27 B → 1.35 GB |
| **MTP connectors** `fc_embedding`, `fc_hidden`, `eh_proj` | **≥ 8-bit**, never rotate | Measured `fc_hidden` kurtosis **2040**: a few columns carry most of the energy. Lossless speculative decoding means its error costs acceptance, not quality, but acceptance is ~1.7× speed (*decode*) | ~13 M params |
| **Attention** (Qwen QSA q/k/v/o; GLM MLA, KDA projections), **DeltaNet** `in_proj_qkv/z`, `out_proj` | **≥ 6-bit**, allocator may raise | Heavy-tailed (q kurtosis 20, o 14 measured). Long-range retrieval lives in the 12 or 11 full/sparse-attention layers (*04 §2.10*). Errors in q/k also perturb the sparse **indexer's** top-k block selection, the same discontinuity as the router | Qwen 2.7 B; GLM 6.0 B → 4.9 GB at 6.5 |
| **Indexer** (QSA `index_qk_proj`, DSA `wq_b/wk/weights_proj`) | **≥ 8-bit** | It picks which KV blocks attention may see. A wrong pick is a hard miss | ≈ 0.1 B |
| **Shared experts** | **≥ 6-bit** (Litwein/oQ: 8b g128) | Every token passes through them, so they get 1/freq ≈ 50× the traffic of an average routed expert. Measured kurtosis **55** | Qwen 0.24 B; GLM 1.06 B |
| GLM's 3 **dense MLP** layers | ≥ 6-bit | Every token; early layers | 0.45 B |
| **Routed experts** | allocator, 2–4.5 bpw, tiered by importance (§2.5) | Each sees ~1/E of tokens. Near-Gaussian | 97 % of bytes |

Two sanity checks:
- **This matches the vendors.** The vendor FP8 configs keep exactly the routers, linear-attention, indexers, hc,
  embed and head in BF16. GLM's vendor FP8 *does* quantize shared experts, dense MLPs and MLA q_a/q_b/kv_a/o,
  so ≥ 6-bit there is at or below vendor precision (*quant §3*, verified).
- **It also matches oQ's real choices.** Reading the local Litwein REAP-320 oQ3e config: experts at 3b g64 (one
  layer at 4b), shared experts and `shared_expert_gate` at 8-bit, DeltaNet 4–6 bits (gates `in_proj_a/b` at 5b
  g128 or 6b g64), hyper-connections 4–8 bits by layer, embed and lm_head 8b **[measured, config.json]**.

**Concrete floor table for `build/`** (replaces `DEFAULT_FLOORS = (lm_head ≥ 6, embed ≥ 4)` plus `--min-bits 3`;
§4.2): pin bf16 for routers, `shared_expert_gate`, Qwen `linear_attn.in_proj_{a,b}`. Floor 8 for
`input_mix_weight_*`, `block_inject_weight`, `indexer.*`, `embed_tokens`, `lm_head`, `fc_*`, `eh_proj`,
`ple.{key,value}_proj`, GLM `f_b_proj|g_b_proj|fbg_a_proj`. Floor 6 for `self_attn.*`, `linear_attn.*`,
`shared_expert(s).*`, dense `mlp.*`. Floor 2 for `switch_mlp.*`.

### 2.5 Frequency-weighted hot/cold expert bits

*01 §2.5c* derives the rule. With `w_j = freq_j·E[g_j²‖f_j‖² | routed]` and the idealized `ε(b) ∝ 4^{−b}`:

$$b_j^\* = \bar b + \tfrac12\log_2\!\big(w_j/\bar w_\text{geo}\big)$$

That is half a bit per doubling of importance. With Qwen's measured skew it is worth −43 % routed-weighted error at
equal bytes. This chapter adds three things.

**(1) Monotone assignment, so tiers become cut points.** Suppose a layer must use a fixed multiset of formats
(tier sizes `n_1..n_T` with error levels `ε_1 < … < ε_T`), and assume every expert's error curve has the same
shape (`ε_j(b) ≈ c·ε(b)`, which holds when the experts are equally Gaussian; measured kurtosis 3.8–3.9 across
experts). Then the rearrangement inequality says Σ w_j ε_{τ(j)} is minimized by giving the smallest error to the
largest `w`. **Sort experts by `w_j`; the tiers are contiguous slices of that order.** The full problem is
therefore "per layer, pick cut points and a format per slice". Each (layer, tier) becomes one MCKP unit whose
options are (cut, format) pairs, and the existing DP solves it. Use measured per-expert `sens` from calib
(`[E, bits, groups]`), not the 4^−b idealization.

**(2) Runtime cost.** `QuantizedSwitchLinear` holds one (bits, group, mode) per stacked tensor **[src, *quant §1*]**.
A tiered layer is 2–3 `SwitchGLU`s plus an index remap, one `gather_qmm` per tier per projection. That is +1–2
kernel launches per projection.
- On Qwen (latency-bound ~80–130 µs/layer) this needs measuring.
- On GLM (bandwidth-bound) it should be small.

The shim is already specified (*01 §4.5* `serve/tiered.py`, *04 §7.1*). Prefill with sorted experts is unaffected
in FLOPs.

**(3) Hot depends on what you calibrate on.** Static frequency comes from calibration. REAP-pruned survivors
inherit their pruned neighbours' traffic, so **compute `w_j` from the re-calibrated, pruned-model stats**
(*04 §4.6*). Keep a floor of 3 bits for experts with a high `expert_out_norm_max` ("super experts", *04 §2.3*):
rare but decisive experts are exactly what a frequency-only rule starves. GLM routing is flatter (entropy
0.92–0.99, *01 §2.5a* **[self-reported]**), so expect a smaller tiering gain there.

### 2.6 Rounding inside the format: LSQ refit, imatrix, GPTQ, AWQ

All of these change *which codes* are stored. None changes the format, so all run at stock MLX speed.

**(a) Least-squares scale/bias refit (what oQe added).** RTN's (s, β) come from min/max, which is optimal only for
the extremes. Given codes `q`, the MSE-optimal affine fit per group is ordinary least squares:

$$s=\frac{\sum_i h_i (q_i-\bar q)(w_i-\bar w)}{\sum_i h_i (q_i-\bar q)^2},\qquad \beta=\bar w - s\,\bar q$$

(weighted means with weights `h_i`). Alternate re-rounding `q = clip(round((w−β)/s))` with re-fitting for 3–4
rounds. Round (s, β) to bf16 before evaluating the error, since that is what gets stored, and keep the best
iterate.
- **Unweighted** (`h ≡ 1`): −19 % weight NMSE at 3b g64 and −32 % at 2b g64 on real experts **[measured]**.
- **imatrix-weighted** (`h_i` = calib's `act_sqmean` for input channel i): this minimizes the *output* error
  `E‖ΔW x‖²` under a diagonal approximation of `E[xxᵀ]`. oQe's isolated A/B: −11.6…−15.3 % text KL
  **[self-reported, *quant §3*]**.
- Calib already writes `act_sqmean` for every dense input and per expert for `down_proj`; `gate/up` per expert
  inputs are exact over all routed tokens **[src, calib contract]**.

**(b) GPTQ (full-Hessian error feedback).** For one linear with input samples `X`, the output error is
`‖ΔW X‖²_F = tr(ΔW H ΔWᵀ)` with `H = XXᵀ`. GPTQ quantizes input column `c` and then pushes the rounding error onto
the not-yet-quantized columns:

$$\Delta w_{:,c+1:}\;=\;-\,\frac{w_{:,c}-\hat w_{:,c}}{[H^{-1}]_{cc}}\;[H^{-1}]_{c,\,c+1:}$$

This is optimal brain surgeon, one column at a time, with Cholesky of `H⁻¹` for stability
([arXiv:2210.17323](https://arxiv.org/abs/2210.17323) **[paper]**). The group scales are set when the column
loop enters each group, from the already-updated weights. For MoE, three details:
- **Per-expert, gate-weighted Hessian:** `H_e = Σ_{x→e} g_e(x)²·x xᵀ`, because the expert's output enters the
  MoE scaled by `g_e`. `calib/sens.expert_sens` already accepts per-row weights **[src]**. mlx-lm's GPTQ instead
  builds one Hessian shared by all experts and cannot see mlx-vlm's `SwitchLinear` anyway (*quant §1*).
- **Rank deficiency:** with `--expert-tokens 128`, `H_e` (2560² for Qwen gate/up, 4096² for GLM) has rank ≤ 128.
  Use shrinkage `H_e + λ·tr(H_e)/tr(H̄)·H̄` toward the layer's pooled `H̄`, plus the usual 1 % diagonal damping,
  and raise `--expert-tokens` to 256–512 for GPTQ runs (sample storage grows linearly, §4.1).
- **Cost:** `mx.linalg.cholesky` is CPU-only in 0.32.4 ("not yet supported on the GPU"). 2560² takes 29 ms
  (+44 ms for `cholesky_inv`) **[measured]**. Qwen: 512 × 48 gate/up Hessians ≈ 30 min of CPU, runnable in
  parallel with GPU work. GLM 144 × 42 × 4096² ≈ 30–40 min **[est]**.

**(c) AWQ-style channel scaling, folded exactly.** Scale input channels by `s` (`W' = W·diag(s)`, `x' = x/s`) to
shrink important channels' *relative* rounding error, with `s = (mean|x|)^α` searched over α
([arXiv:2306.00978](https://arxiv.org/abs/2306.00978) **[paper]**). `1/s` must be folded into whatever produces
`x`:
- **`down_proj`, per expert, exactly:** the input is `h = silu(gate·x) ⊙ (up·x)`, which is linear in `up`. Scaling
  row i of `up_e` by `1/s_i` scales `h_i` by `1/s_i`, so `down_e' = down_e·diag(s_e)` is an exact per-expert
  re-parametrization. mlx-vlm's AWQ skips experts entirely (*quant §1*). This one is new **[hypothesis]**.
- **`gate/up`:** the input is the post-attention RMSNorm output, shared by router, shared expert and all experts.
  One `s` per layer: fold `1/s` into the norm weight and multiply the router and shared-expert columns by `s`
  (the router is bf16, so that is exact). It gives a single compromise `s` for 512 experts, which is weaker.

**Order and expectations.** LSQ refit first: cheap, data-light, robust. Then imatrix-weighted LSQ. Then GPTQ only
for tensors the allocator leaves at ≤ 3 bits. At 4-bit, RTN+LSQ is already close (4b g64: 0.0088 → 0.0077).

### 2.7 Rotations: Hadamard, QuaRot, SpinQuant, and mlx-lm's `prism_hadamard_qwen35`

**Why rotations kill outliers.** Quantization error in a group scales with its *range*, and a single outlier sets
the range for 63 innocent neighbours. Multiply the input axis by an orthogonal `Q`. The layer is unchanged,
`W x = (W Qᵀ)(Q x)`, but every rotated weight is now a ±-weighted average of the whole row. For a normalized
Hadamard `H` with random signs `D` (`Q = H D/√n`), each output coordinate is a sum of n independent terms. By the
CLT the rotated row is near-Gaussian, and the max/RMS ratio falls from "one huge entry" to ≈ √(2 ln n). This is
"incoherence processing" (QuIP [arXiv:2307.13304](https://arxiv.org/abs/2307.13304), QuIP#
[arXiv:2402.04396](https://arxiv.org/abs/2402.04396) **[paper]**). A Walsh–Hadamard transform costs
`O(n log n)`, so it can run online.

- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456) **[paper]**) rotates the *residual stream*.
  RMSNorm is rotation-invariant (`‖Qx‖ = ‖x‖`) once its weight is folded into the next linear, so `Q` folds
  offline into every writer (embedding rows, `o_proj`/`down_proj` outputs) and every reader (q/k/v, gate/up,
  lm_head inputs). It adds *online* Hadamards only where no linear sits in between, chiefly before `down_proj`
  (after SwiGLU's elementwise product), plus on the KV heads for KV quantization.
- **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406) **[paper]**) *learns* the foldable rotations
  (Cayley-parametrized, on the Stiefel manifold) to minimize the quantized network's loss. Random rotations vary a
  lot in final accuracy, and learned ones close most of the gap to fp at W4A4.

Both papers are mainly about *activation* (W4A4/W4A8) quantization, where outliers in `x` are catastrophic. For
weight-only quantization with g64 groups, small groups already isolate outliers, so the gain is smaller.

**What `mlx_lm/models/prism_hadamard_qwen35.py` does [src].**
- It is a loader for pre-packed **dense Qwen3.5** checkpoints: schema 1 = text, 2 = mlx-vlm VL.
- Every packed `nn.Linear` / `nn.Embedding` becomes a `HadamardQuantizedLinear`: **2-bit affine, g128, fp16
  scales/biases**, with a per-module signed blockwise Walsh–Hadamard (block ∈ {0, 512, …, 8192}, 0 = none) folded
  into the weight's input axis.
- At runtime it rotates **activations online** on every call: cast x to fp32, multiply by ±1 `signs`, run
  `mx.hadamard_transform(x.reshape(-1, block), scale=1/√block)`, then the stock `quantized_matmul`. Embeddings
  dequantize the looked-up rows and apply the inverse rotation.
- It touches **only `nn.Linear` and `nn.Embedding`** (no `SwitchLinear`), refuses any other quantization, and
  validates the sign vectors.
- In other words it is QuIP#-style incoherence processing with plain affine codes, used to make **2-bit** usable
  on a dense model.
- It does not apply to `qwen4_exp`/`glm5_next` as is. It shows that MLX has the needed primitive
  (`mx.hadamard_transform` supports n = 2^k·{1, 12, 20, 28}; all of 640, 2048, 2560, 4096, 6144, 10240 work
  **[measured]**) and the runtime pattern.

**What it would buy *us*** (real weights, §3.1) **[measured]**:

| tensor | kurtosis → after H | 3b g64 RTN → +H | +LSQ | +H+LSQ |
|---|---|---|---|---|
| routed expert gate/down | 3.9 → 3.1 | 0.0389 → 0.0362 (−7 %) | 0.0316 (−19 %) | 0.0297 (−24 %) |
| shared_expert.gate | 55 → 4.3 | 0.0642 → 0.0359 (−44 %) | 0.0528 | 0.0296 |
| attn.q_proj | 20 → 6.0 | 0.0569 → 0.0362 (−36 %) | 0.0452 | 0.0296 |
| MTP fc_hidden | 2040 → 1.6 | 0.0123 → **0.0186 (+51 %)** | 0.0106 | 0.0160 |

- **After rotation every tensor converges to the Gaussian RTN error** (~0.036 at 3b g64, ~0.0083 at 4b g64).
  Rotation makes error independent of the distribution. That helps when the distribution was bad and does nothing
  when it was already Gaussian, which is the case for the experts that hold 97 % of the bytes.
- **Rotation can hurt** when a few columns hold most of the energy (`fc_hidden`). Without rotation, those columns'
  groups get their own scales and the remaining groups stay fine-grained. Rotation smears the giants into every
  group. The allocator should treat "+H" as just another option with its own measured `sens`, never a global
  switch.
- nvfp4 gets slightly *worse* with rotation (its E4M3 group scales already adapt). mxfp4 improves but stays behind
  affine.

**Online cost at decode shapes:** 7–12 µs per `hadamard_transform` call for (1, 2560), (10, 640), (1, 4096),
(8, 2048), launch included **[measured]**. One rotation on Qwen's `down_proj` input per layer is 48 × ~10 µs ≈
0.5 ms per token, about 1.5 % of a 33 ms token. Prefill (2048 × 2560) takes 0.5 ms per call. So the *speed* cost
is acceptable. The *quality* case for weight-only experts is weak: ~5–7 % NMSE, against 19 % from LSQ at zero
cost. **Verdict: stretch (§7.1), behind LSQ/GPTQ/DWQ.**

### 2.8 DWQ: distilling scales and biases against the teacher

**What `mlx_lm.dwq` does [src].**
- Unfreeze only `scales` and `biases` of affine modules with bits < 8.
- Minimize `KL(softmax(t/T) ‖ softmax(s/T))` at T = 2 against teacher logits. The targets are precomputed as the
  **top-1024 logits + indices** per position, and the student logits are gathered at those indices.
- Adam, lr 1e-6, 2048 samples × ≤ 1025 tokens by default.

Why it works:
- **Few, well-conditioned knobs.** At g64 there are 2 knobs per 64 weights (0.5 bpw of parameters), and
  `ŵ = s·q + β` is linear in them, so gradients are clean. The codes stay fixed.
- **It corrects what layer-local objectives can't see.** Error *correlations* across layers, and the fact that
  some output directions matter more for the logits than others.

The mlx-lm notes: it helps 2–4-bit most, ≥ 6-bit barely; g32 doubles the knobs and "can work much better"
**[src, LEARNED_QUANTS.md]**. Evidence: Litwein 3-round band DWQ cut KL20 by 11.2 % vs the pruned initializer
and recovered 33.6 % of the pruning gap **[self-reported, *quant §3*]**.

**Why we can't run it as shipped.** `mlx_lm.dwq` cannot load `qwen4_exp`/`glm5_next`, and whole-model training
state (fp32 master, grad, Adam m/v = 16 B per trainable param) for a ~60 GB student is ~35–40 GB on top of the
weights (*quant §3*). Two ways around it:

**(A) Teacher targets without a resident teacher.**

| target per token | Qwen (V = 248,320) | GLM (V = 154,880) | exact? |
|---|---|---|---|
| full logits fp16 | 497 KB | 310 KB | yes; **497 / 310 GB per 1 M tokens**: infeasible |
| top-1024 (bf16 logit + int32 idx, mlx-lm) | 6.1 KB | 6.1 KB | approx; 6.1 GB per 1 M |
| top-64 (fp16 logprob + int32 idx + tail mass) | 388 B | 388 B | approx; 0.39 GB per 1 M; `ref/` already stores this format in fp32 |
| **final normed hidden state `ĥ_N` (fp16), logits recomputed with the bf16 lm_head** | **5.1 KB** (2560) | **8.2 KB** (4096) | **exact, full vocab**; 5.1 / 8.2 GB per 1 M |

**[computed]** The last row is the one to use. In both models the head input is a single vector:
- Qwen: `hyper_connection_mixer` → norm → 2560.
- GLM: `norm(mean over streams)` → 4096.

Calib's head stage already computes it (*calib contract*), so storing it for a training split is a few lines.
During DWQ, keep a bf16 copy of `lm_head` (Qwen 0.64 B = 1.27 GB, GLM 0.63 B = 1.27 GB) and recompute exact
teacher logits per batch. Top-k is a poor fit at T = 2: tempering flattens the teacher, moving real mass into the
tail that top-k discards.

**(B) Where the gradient comes from.**

- **(B1) Blockwise local DWQ, inside the calib stream (default).** Run two hidden streams through the layer pass:
  the teacher stream `h^T` (bf16 layer) and the student stream `h^S` (quantized layer). For layer ℓ:
  1. Load bf16, compute `h^T_{ℓ+1}`.
  2. Build the quantized layer from the chosen plan (with LSQ/GPTQ codes).
  3. Train its scales and biases to minimize `‖layer^Q_ℓ(h^S_ℓ) − h^T_{ℓ+1}‖² / ‖h^T_{ℓ+1}‖²` over all 4 hc streams.
  4. Emit `h^S_{ℓ+1}` with the trained parameters.

  Feeding the *student's own* input with the *teacher's* target makes each layer absorb the upstream error as well
  (the sequential-input idea of GPTQ, and Block-AP in EfficientQAT,
  [arXiv:2407.11062](https://arxiv.org/abs/2407.11062); BRECQ [arXiv:2102.05426](https://arxiv.org/abs/2102.05426);
  OmniQuant [arXiv:2308.13137](https://arxiv.org/abs/2308.13137) **[paper]**). Weight the loss per output
  coordinate by the Fisher `a_t` of §2.3 when available. Otherwise the loss treats all residual directions as
  equal, which is its known weakness.
  - Memory: bf16 GLM layer ~15 GB + quantized ~4 GB + scales/biases state (K=144, g64: 144·3·2048·4096/64·2 =
    113 M params × 16 B = 1.8 GB) + chunk activations, under ~25 GB. Co-tenant safe at `--mem-gb 40`.
  - Time: 3× forward FLOPs × 2 epochs × 64K tokens per layer ≈ 0.28 PFLOP per GLM layer ≈ 35 s at ~8 TFLOP/s,
    so ~30 min for 45 layers plus I/O. About half that for Qwen **[est]**.
- **(B2) End-to-end band DWQ (Litwein-style).** Load the whole quantized student (Qwen ~56 GiB). Train one band of
  ~4 layers' scales and biases at a time against `ĥ_N`-recomputed teacher logits with gradient checkpointing, and
  accept a band only if held-out KL improves.
  - Memory: Qwen band state 4 layers × 78.6 M (g64 experts) × 16 B ≈ 5.0 GB, plus 1.3 GB teacher head, plus
    checkpointed boundaries (1K tokens × 48 × 20.5 KB ≈ 1 GB) → **≈ 64–70 GB**. That needs the **All-in
    profile**, with the serving model stopped.
  - The backward pass goes through `gather_qmm` (NAX backward is in 0.32.4, mlx#4684, *quant §4*) and through
    GatedDeltaNet's ops path (no kernel VJP), which is slow.
  - Time ≈ 3 × 14 GFLOP/token at ~5 TFLOP/s ≈ 120 tok/s. 12 bands × 64K tokens ≈ 1.8 h per round. GLM would need
    ~68 GB resident plus state: run it only in the All-in profile, and only if (B1) leaves a measurable gap
    **[est]**.

### 2.9 2-bit experts and how to recover them

2b g64 RTN on real experts has NMSE **0.167**, 4.3× the 3-bit error. LSQ brings it to 0.114, and LSQ + Hadamard
to 0.108 **[measured]**. At these error levels the "small perturbation" assumption behind §2.3 starts to fail, so
measure, don't extrapolate. Recovery options, best first:

1. **Only where it is cheap in KL.** Use 2-bit for the cold tier (low `w_j`, §2.5) and for the MTP drafter's
   experts. Draft acceptance on GLM was 68–71 % across Q2_K–Q6_K draft experts **[self-reported, *reap §7*]**.
2. **LSQ + imatrix + GPTQ** (§2.6). The relative gain from better rounding is largest at 2 bits.
3. **DWQ** (§2.8). The gain is largest at 2–3 bits (mlx-lm tips).
4. **Not low-rank error correction.** An RTN residual is close to white noise. A rank-r SVD captures ≈ r/min(m,n)
   of a white matrix's energy (rank 16 of 640 ≈ 2.5 %), and rank-16 factors on a Qwen expert projection cost
   ~20 % of a 2-bit tensor's bytes. LQER ([arXiv:2402.02446](https://arxiv.org/abs/2402.02446) **[paper]**) makes
   this work by SVD-ing the *activation-scaled* residual, which concentrates the energy that matters. It needs an
   extra matmul per expert and gives poor bytes-per-error compared with going 2b → 3b g128. **Skip.**

For GLM, 2-bit cold experts also buy decode speed (~35 % faster per expert byte, *quant §4*). That is a second
reason to give GLM a 2-bit cold tier.

### 2.10 KV-cache quantization: why it stays off

Mechanics: K and V are quantized group-wise (affine, along the head dim) once a sequence passes
`--quantized-kv-start` (default 5000). Attention then runs on dequantized blocks. The usual refinement quantizes
keys per channel and values per token (KIVI, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) **[paper]**).

Why we don't enable `--kv-bits` (*decode §2*, **[measured]** there):

- **It crashes the server.** `BatchGenerator._make_cache` raises `NotImplementedError: QSAKVCache does not support
  quantized continuous batching` for Qwen. GLM raises "Only real floating types can be quantized" on the
  indexer's bool validity-flag cache.
- **For GLM it does nothing or crashes.** `CacheList` has no `to_quantized`, so single-stream uniform is a silent
  no-op, and TurboQuant crashes on the zero-width MLA value cache.
- **It saves little.** Per-token KV lives only in the 12 QSA / 11 DSA layers. The recurrent state (110 / 141 MiB)
  is fixed-size and stays fp32. Qwen at 128K: 28,464 B/token → ~10.8 KB/token at 4-bit (900 B per QSA layer) saves
  **≈ 2.3 GB** **[computed]**. GLM's MLA latent is already 1,089 B/token per layer.
- **It hits the wrong place.** Key noise feeds the *sparse top-k indexer* (QSA/DSA block selection), the same
  discontinuity that makes routers fragile.

Revisit only if upstream fixes the batch path. Then gate on RULER 64K/128K and per-domain KL at 64K.

---

## 3. Evidence

### 3.1 New measurement for this chapter: formats, rotation, refit on real weights **[measured]**

Source: the bf16 MTP drafter `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` (in the HF cache), which holds a full
Qwen3.8-Flash-Next decoder layer with **512 real experts** at the real shapes. 24 random experts (gate and down)
plus the dense tensors were read by byte range. The metric is relative weight MSE `‖W−Q(W)‖²/‖W‖²`. Peak 3.4 GB
(the H128/LSQ variants of 24 × 640 × 2560 fp32 copies), run once, a few seconds of GPU. Script:
`scratchpad/q5/fmt_rot.py` (§8). Caveat: this is *weight* error, not activation-weighted output error. imatrix-
and GPTQ-type gains need real activations, which calib provides.

| routed experts (gate_proj; down_proj within 2 %) | NMSE |
|---|---|
| affine 2b g64 / g128 | 0.167 / 0.200 |
| affine 3b g32 / g64 / g128 | 0.0306 / 0.0389 / 0.0479 |
| affine 4b g32 / g64 / g128 | 0.0068 / **0.0088** / 0.0110 |
| mxfp4 (4.25 bpw) | 0.0142 (1.6× affine 4b g64) |
| nvfp4 (4.5 bpw) | 0.0109 (1.24×) |
| mxfp8 / affine 8b g64 | 7.1e-4 / 5e-5 (14×) |
| 3b g64 + LSQ refit / + Hadamard / both | 0.0316 / 0.0362 / 0.0297 |
| 2b g64 + LSQ / both | 0.114 / 0.108 |
| 3b g128 + LSQ (3.25 bpw) | 0.0387 (≈ RTN 3b g64 at 3.5 bpw) |

These agree with *quant §3*'s EmbeddingGemma check (mxfp4 clearly worse, nvfp4 within ~15–25 %, mxfp8 ~15× worse
than affine 8b), now on the target model's own experts.

### 3.2 Published evidence, by question

| question | evidence | tag |
|---|---|---|
| Allocation beats uniform | Qwen3.8-FN pipenetwork: uniform 4b +20.6 % PPL; protecting ~3 % (hc read gates, attention/DeltaNet/shared/PLE projections, embed/head) at 8-bit → +1.3 % | [self-reported], *quant §3* |
| Allocation by sensitivity + knapsack | OptiQ (per-layer KL, greedy knapsack); RCO (exact budget, Gumbel-STE + DP); HAWQ-V2 Hessian-trace mixed precision [arXiv:1911.03852](https://arxiv.org/abs/1911.03852) | [self-reported] / [paper] |
| Rounding (same allocation) | oQe LSQ refit: −11.6…−15.3 % text KL (isolated A/B, 9B/35B) | [self-reported], [PR #4385](https://github.com/jundot/omlx/pull/4385) |
| GPTQ | OPT/BLOOM 3–4-bit with large gains over RTN | [paper] [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ | activation-aware scaling protects ~1 % salient channels | [paper] [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| Rotations | QuaRot W4A4 Llama-2-70B within ~0.5 PPL; SpinQuant learned rotations close most of the remaining gap; QuIP# 2-bit with incoherence + E8 lattice | [paper] [2404.00456](https://arxiv.org/abs/2404.00456), [2405.16406](https://arxiv.org/abs/2405.16406), [2402.04396](https://arxiv.org/abs/2402.04396) |
| FP4 vs INT4 | "NVFP4 and INT4 provide similar quality", MXFP4 worse (mostly W4A4) | [paper] [arXiv:2509.23202](https://arxiv.org/abs/2509.23202), *quant §3* |
| MoE quantization | Expert-usage-aware bit allocation among the PTQ heuristics benchmarked for MoE | [paper] [arXiv:2406.08155](https://arxiv.org/abs/2406.08155) |
| DWQ | Litwein REAP-320 oQ3e + 3-round band DWQ: KL20 0.1035, −11.2 % vs init, 33.6 % of the pruning gap recovered | [self-reported], [card](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX) |
| Block-wise then end-to-end | EfficientQAT: Block-AP then E2E training of step sizes only | [paper] [arXiv:2407.11062](https://arxiv.org/abs/2407.11062) |
| Qwen3.8-FN KL anchors | RTN 4b g64 0.288, g32 0.232 (4×1024 tokens; noise floor 0.024); oQ4e (≈4.6 bpw, unpruned) code 0.0776, novel 0.171 | [self-reported], *quant §3* (different corpora; compare within a source only) |
| GLM-5.3-Flash KL anchors | MLX 3-bit (3g32, ~4.7 bpw overall) KLD 0.042 vs FP8; GSQ-RCO 3.0 bpw ≈ 0.142; IQ3_S 3.11 bpw 0.283; pruning K=144: +0.56 nats PPL (pipenetwork) | [self-reported], *quant §3*, *04 §3.2* |
| KV quant | crashes/no-ops measured on tiny real-dim models | [measured] *decode §2* |

---

## 4. Plan for this machine

All commands use the real CLIs. **★ = new flag or module specified here.** Per PLAN.md status, `calib/` and `build/`
are unreviewed drafts: run `uv run pytest tests/calib tests/build` and the tiny-fixture end-to-end first. Real
runs are one at a time, by the orchestrator, never while another GPU-heavy job runs.

### 4.1 Changes to `calib/` (all testable on `tests/fixtures/make_tiny.py`)

| change | where | output | cost |
|---|---|---|---|
| **Add g128** to the sens grid: `GROUPS = (32, 64, 128)` | `calib/sens.py` | `sens.{name}` becomes `[6, 3]` (`[E,6,3]` for switch); JSON keys `"3_128"` | +50 % qdq work. The sens grid is ~4 min per GLM pass (*04 §4.3*), so negligible |
| **Rounding-matched sens**: `--sens-rounding rtn,lsq` ★ computes the grid also with the imatrix-weighted LSQ refit (§2.6a), keys `"3_64_lsq"` | `calib/sens.py` (`qdq_lsq(w, b, g, h)`) | lets the allocator see the hull the build will actually produce | ~4× the qdq cost of RTN (4 refit iterations). Still minutes |
| **Rotation option** `--sens-rot` ★: also measure "+H" for dense linears and for experts' `down_proj` | `calib/sens.py` | `"3_64_h"` keys | small |
| **Importance stats**: `expert_imp = E[(g‖f‖)²]` per expert (same as *04*'s `expert_reap_sq`) | `calib/probe.py _moe`, `run.py` | `[E]`, per domain | trivial |
| **Residual energy** `resid_sq` per layer (mean ‖h_ℓ‖² over sampled rows, all hc streams) | `calib/run.py` | scalar per layer | trivial |
| **Save samples** `--save-samples` ★: the already-sampled input rows (`--lin-tokens`, `--expert-tokens`) and their gate weights, fp16 | `samples/layer_{i:03d}.safetensors` | enables GPTQ/AWQ in `build/` without re-streaming | at `--expert-tokens 128`: Qwen gate/up inputs 128 × 2560 × 2 B × 512 experts = 336 MB per layer → **16 GB** for 48 layers (+ down 4 GB); GLM 128 × 4096 × 2 B × 288 = 302 MB per layer → **12.7 GB**. Grows linearly with `--expert-tokens` **[computed]** |
| **Teacher head inputs** `--teacher-split train` ★: store `ĥ_N` fp16 for a DWQ training split (exact full-vocab targets, §2.8A) | `teacher/{k:05d}.npz` | `ĥ_N [L, d]` | 1 M tokens: Qwen 5.1 GB, GLM 8.2 GB |
| **Streaming Fisher** (optional) `calib/fisher.py` ★: reverse pass, `fisher.{name}` = E‖∂log p/∂y_t‖² with sampled y | per layer | exact `a_t` (§2.3 level 3) | ≈ 2–3 × forward. GLM 1 M tokens ≈ 2.5 h |
| **MTP stage** (from *04 §4.1*) | `calib/run.py` | MTP sens + REAP | small |
| Bump `FORMAT_VERSION` to 2 (shared with *04*); update `docs/contracts/calib.md` | | | |

Tests:
- LSQ: `sens_lsq ≤ sens_rtn` everywhere; the refit is idempotent on already-optimal groups.
- g128 cells are NaN where `in % 128 ≠ 0`.
- `fisher` matches a finite-difference ΔKL on the tiny fixture within 10 %.

### 4.2 Changes to `build/`

1. **Menu.** `alloc.DEFAULT_GROUPS = (32, 64, 128)`. For `kind == "expert"`, restrict to {64, 128} unless the
   measured sens puts g32 on the hull (it never did in §3.1). Let `fill_sensitivities`' Gaussian proxy cover g128
   until calib v2 data exists.
2. **Objective units (§2.3).** Add `--weighting {uniform,energy,fitted,fisher}` ★ (default `energy` once calib v2
   lands).
   - `energy`: `w_t = out_sq_t / resid_sq_ℓ`.
   - `fitted`: `× a_{kind,band}` from a JSON written by the sweep fit.
   - `fisher`: `w_t = fisher_t·out_sq_t / d_t`.

   Apply it in `Builder.options_for` (today `weight = 0.25 if mtp else 1.0`). Change `aggregate_expert_sens` to
   weight experts by `freq·expert_imp` (fallback: `freq·gate_mean²·out_sq`). **Fix the contract text: `p` default
   is 1**, and say why.
3. **Floors (§2.4).** Replace `DEFAULT_FLOORS` with per-family floor/pin tables in `build/families.py`
   (`default_floors=`), using the regexes in §2.4. Keep `--floor`/`--pin` as overrides.
4. **Rounding.** New `build/round.py` ★ with `rtn` (today's `mx.quantize`), `lsq` (needs `act_sqmean` from calib),
   `gptq` (needs `--save-samples` data; CPU Cholesky in a thread pool), and `awq-down` (per-expert `down`/`up`
   scaling, folded exactly). It is called from `Builder.write` in place of the bare `mx.quantize(...)` call. The
   output layout stays identical (`weight`, `scales`, `biases`), so loaders are unaffected. CLI:
   `--rounding {rtn,lsq,gptq} --awq-down`. Record per-tensor achieved NMSE in `build_summary.json` (`allocation[i].achieved_nmse`).
5. **Tiers.** `--tiers` / `--tier-score weighted` from *01 §4.4*, with §2.5's monotone cut-point units and the
   `pruned_moe.py`-style shim extended to tiered `SwitchGLU`s (*01 §4.5*).
6. **Resident budget.** `--budget-scope resident` (*01 §4.1*): `ple_table` units count against disk, not the
   budget.
7. **PLE nvfp4 with pre-scaling** (*quant Rec 6*): `--ple-mode {affine32,nvfp4} --ple-prescale 4096` ★. Multiply
   the table by 2¹² before `mx.quantize(..., mode="nvfp4")` and fold 2⁻¹² into `ple.value_proj` (and into the
   value-side `weight_scale`, which `qwen4_exp/fp8.py` drops for sharded tables).
8. **DWQ.** `build/dwq.py` ★ = §2.8 B1, run as a calib-style stream over a *built* checkpoint plus the source.
   `build/dwq_band.py` ★ = B2, All-in only. Both rewrite only `*.scales` / `*.biases` tensors in place, so the
   byte count is unchanged.

Tests (tiny fixtures):
- `--rounding lsq` reduces the measured output NMSE vs `rtn` on every unit.
- GPTQ on a dense tiny linear reproduces a numpy reference to 1e-5.
- AWQ-down folding is exact before quantization (max |Δy| < 1e-5).
- PLE pre-scaling round-trips through `ple_storage` with identical lookups.
- DWQ never changes `weight` (codes) tensors.

### 4.3 Changes to `calib/sweep.py` (specified in *04 §4.4*): simulated quantization

Add a variant type `{"name":"G1","quant_plan":"plans/glm-g1.json","rounding":"lsq"}`. The plan is the
`allocation` array from `build.make --dry-run --summary-json`. When the sweep loads layer ℓ in bf16, it applies
`round.qdq(W, bits, group, rounding)` per tensor, in that variant's stream only. Because pruning is a mask
and quantization is a weight substitution, **one pass measures pure-pruning KL, pure-quant KL and their sum on the
same tokens**. That is the only way to see GLM's quant-induced KL without building, or affording, a bf16 pruned
GLM (~320 GB). Memory is unchanged (one bf16 layer + one qdq copy at a time ≈ 2 × 15 GB for GLM).

### 4.4 Order of work, time and memory

| step | what | time [est] | peak memory |
|---|---|---|---|
| Q0 | Tests + tiny-fixture e2e for all §4.1–4.3 changes | 1–2 days of dev; minutes to run | < 2 GB |
| Q1 | calib v2 pass (the *04* final pass, plus g128/LSQ sens, samples, teacher states) | GLM ≈ 4.5 h, Qwen ≈ 2.2 h (*04 §4.3*) + ~10 % for the new probes | ≤ 40 GB (`--mem-gb 40`) |
| Q2 | Dry-run allocations: 3 weightings × 2 roundings × 2 budgets | seconds each | < 8 GB |
| Q3 | Sweep fit of `a_{kind,band}`: 16 qdq variants = 2 batches | GLM ≈ 4 h, Qwen ≈ 2 h | ≤ 40 GB |
| Q4 | Sweep the 3–4 finalist plans (simulated KL, pruned + quantized) | 1 batch: GLM ≈ 2 h, Qwen ≈ 1 h | ≤ 40 GB |
| Q5 | Build finalists with `--rounding lsq` (then `gptq` for ≤ 3-bit units) | 15–30 min each; GPTQ adds ~30–40 min CPU | 6–7 GB (+ ~3 GB GPTQ) |
| Q6 | Blockwise DWQ (B1) on the best build | GLM ≈ 30–45 min, Qwen ≈ 20–30 min | ≤ 25 GB |
| Q7 | `evals.kl` (all domains), then the *04 §5* guards, `evals.speed` | KL: minutes; task suites: hours | the model's own footprint (needs the serving model stopped, or the Work-profile headroom) |
| Q8 (opt.) | Band DWQ (B2), All-in profile, Qwen only first | ~2 h per round | ~64–70 GB |

Disk:
- calib v2 adds ~13–20 GB of samples plus 5–8 GB of teacher states per model.
- Each build writes ~60–90 GB.

Check free space before each step; *04 §6* gives the SSD-wear warning.

### 4.5 Commands

```bash
# Q1: calibration v2 (★ flags are new)
uv run python -m calib.run --model qwen --src $QWEN_SNAP --data artifacts/qwen/data-final \
    --out artifacts/qwen/calib --mem-gb 40 --expert-tokens 256 \
    --sens-rounding rtn,lsq ★ --sens-rot ★ --save-samples ★ --teacher-split train ★

# Q2: plan, never write
uv run python -m build.make --model qwen --src $QWEN_SNAP --calib artifacts/qwen/calib --prune none \
    --budget-gb 60.5 --budget-scope resident ★ --groups 32,64,128 --bits 2,3,4,5,6,8 \
    --weighting energy ★ --rounding lsq ★ --tiers auto3 ★ --mtp intact --drop-vision \
    --ple-mode nvfp4 --ple-prescale 4096 ★ --dry-run --summary-json plans/qwen-q1.json

# Q4: simulated KL of the plan (pruning mask + qdq), no build
uv run python -m calib.sweep --model qwen --src $QWEN_SNAP --calib artifacts/qwen/calib \
    --data artifacts/qwen/data-final --variants sweeps/qwen-quant.json --out results/sweep/qwen-quant.json
#   sweeps/qwen-quant.json: [{"name":"q1-rtn","quant_plan":"plans/qwen-q1.json","rounding":"rtn"},
#                            {"name":"q1-lsq","quant_plan":"plans/qwen-q1.json","rounding":"lsq"}, ...]

# Q5/Q6: build, then blockwise DWQ on the scales/biases
uv run python -m build.make ... (same flags without --dry-run) --out models/qwen-fn-512-q1
uv run python -m build.dwq ★ --model qwen --src $QWEN_SNAP --built models/qwen-fn-512-q1 \
    --data artifacts/qwen/data-final --tokens-per-layer 65536 --epochs 2 --lr 1e-5 --mem-gb 40

# Q7: verify
uv run python -m evals.kl --model models/qwen-fn-512-q1 --ref artifacts/qwen/calib/ref
for s in speed code math tools longctx; do uv run python -m evals.$s --model models/qwen-fn-512-q1; done
```

### 4.6 Target recipes (byte-exact plans; the allocator refines within them)

**R-G1: GLM-5.3-Flash, ≤ 68 GB (Work profile, 64K context; *01 W2*).** K=144 from *04*. Params from *01 §2.1a* /
*quant* Verification **[computed]**:

| part | params | format | GB |
|---|---|---|---|
| routed experts, 144 × 42 × 25.17 M | 152.2 B | **avg 3.10 bpw**: tiers H 16 @ 4b g128 (4.25), M 80 @ 3b g128 (3.25), C 48 @ 2b g64 (2.5) → (68 + 260 + 120)/144 = 3.11 (the allocator trims the last ~0.2 GB). Untiered fallback: 80 % @ 3b g128 + 20 % @ 2b g64 by the allocator | **58.98** |
| attention (KDA + MLA) | 5.99 B | 6b g64 (6.5), allocator may raise | 4.87 |
| shared experts | 1.06 B | 6b g64 | 0.86 |
| dense MLP (layers 0–2) | 0.45 B | 6b g64 | 0.37 |
| embed + lm_head | 1.27 B | 8b g64 (8.5) | 1.35 |
| DSA indexer | 0.08 B | 8b g64 | 0.09 |
| routers (fp32, sliced to 144) + bias | 0.025 B | fp32 | 0.10 |
| hc, norms, A_log, conv | — | bf16/fp32 | ≈ 0.05 |
| MTP experts, 144 kept | 3.62 B | 2b g64 (acceptance ~insensitive to bits) | 1.13 |
| MTP non-expert | 0.19 B | 8b g64 | 0.20 |
| **total** | | | **68.0** |

- **Rounding:** LSQ everywhere, GPTQ on the 2-bit tier, then B1 DWQ.
- **Expected KL** (top-64 + tail vs the FP8 reference):
  - Pruning dominates. K=144 is +0.56 nats PPL (pipenetwork) and OpenMOSE K=228 already gave KL 0.45
    **[self-reported]**.
  - **Quant-only budget: ≤ 0.10**, measured in the sweep as KL(pruned-bf16 stream ‖ pruned-qdq stream) and as
    the difference between the two arms against `ref/`.
  - Total expected **≈ 0.5–0.8** **[est]**.
- **Speed:** experts at ~3.1 bpw read ≈ 8 × 42 × 25.17 M × 3.1/8 ≈ 3.3 GB/token, about 11 % below an all-3.5-bpw
  build. Expect ~40 tok/s plain on an idle machine (*01 §4.3*).

**R-Q1: Qwen3.8-Flash-Next unpruned, resident ≤ 60.5 GB (Work, 128K context, ~11 GiB spare; = *01 W1*).
Recommended.**

| part | params | format | GB |
|---|---|---|---|
| routed experts, 512 × 48 × 4.92 M | 120.8 B | **avg 3.58 bpw**: H 192 @ 4b g128 (4.25), M 288 @ 3b g128 (3.25), C 32 @ 2b g64 (2.5) → (816 + 936 + 80)/512 = 3.578 | **54.0** |
| DeltaNet (`in_proj_qkv/z`, `out_proj`) | 2.09 B | 6b g64 (`in_proj_a/b` bf16) | 1.70 |
| full attention + QSA indexer | 0.62 B | 6b g64 / indexer 8b | 0.50 |
| hyper-connections | 0.64 B | 8b g64 | 0.68 |
| embed / lm_head | 0.64 + 0.64 B | 8b g64 | 1.36 |
| shared experts | 0.24 B | 8b g128 (8.25; oQ's choice) | 0.25 |
| routers (bf16) | 0.06 B | bf16 | 0.13 |
| MTP (512 experts + connectors) | 2.52 + ~0.1 B | experts 3b g128; connectors 8b | 1.12 |
| **resident total** | | | **59.7** |
| PLE n-gram table (mmapped, page cache) | 51.2 B | nvfp4 g16, pre-scaled ×2¹² | 28.8 on disk |

- **Disk:** 88.5 GB.
- **Expected KL:** target **≤ 0.11 before DWQ, ≤ 0.08 after** (top-64 + tail vs bf16 ref, mean over 4 domains;
  `agentic` will be the worst, cf. Litwein agent 0.185 vs code 0.060) **[est]**. Anchors: oQ4e unpruned at ~4.6
  bpw 0.078 on code; Litwein REAP-320 at 3.76 bpw 0.1035 *including* pruning and DWQ **[self-reported]**. R-Q1 has
  no pruning loss. Pure quantization at 3.6 bpw with importance tiers and LSQ should sit between them.
- **Speed:** like today (latency-bound decode, *quant §4*), ~30 plain / ~55 MTP.
- **R-Q1+ (Focus, 68 GB resident):** experts 4.03 bpw (H 320 @ 4b g64, M 192 @ 3b g128). Use it only if R-Q1
  misses its KL target by > 20 %.

**R-Q2: Qwen with the 68 GB counting the PLE on disk** (the *quant §7* accounting). Fixed parts: PLE 28.8 (nvfp4)
+ trunk 4.6 + MTP (pruned, 2b g64) ≈ 0.6 = 34.0 GB, leaving **34.0 GB for experts**.
- **K=352 @ 3b g128:** 352 × 0.236 B = 83.1 B params → **33.7 GB**, total **67.8 GB**.
- **K=320 @ 3b g64 (Litwein-like):** 75.5 B params → 33.0 GB.

Whether 32 more experts per layer beat g64 over g128 at equal bytes (+23 % expert NMSE, §2.2) is exactly the
*04 §4.5* "a GB buys the same KL either way" test. Run both in one sweep batch. The rare-entity guard (*04 §5*)
decides ties.

---

## 5. Decision criteria & what to measure

| measure | tool | accept if | why |
|---|---|---|---|
| Predicted = written bytes, resident ≤ budget | `build_summary.json`, `serve/memory.resident_weights_bytes` (*01 §4.1*) | exact | the budget is the constraint |
| **Simulated vs built KL** | `calib.sweep` qdq arm vs `evals.kl` | within 10 % (relative) | validates the sweep as the planning tool |
| Per-domain KL (top-64 + tail), top-1 agreement | `evals.kl` | meets the recipe target; **every domain** ≤ 1.5× the best domain; better than the deployed baseline (sh0wie REAP-288 4-bit, evaluated against the same `ref/`) with bootstrap 95 % CI over samples excluding 0 in ≥ 3 of 4 domains | KL is the cheapest high-resolution signal (*04 §5* sample-size argument) |
| Additivity check | 3 built points | measured KL within ±25 % of Σ predicted ΔKL | justifies the knapsack's diagonal objective |
| Rounding gain | `achieved_nmse` per unit + sweep | LSQ ≤ RTN on every unit; ≥ 10 % KL drop vs RTN at equal plan, else drop the extra complexity | expected −12…15 % (*quant*) |
| DWQ gain | `evals.kl` before/after on held-out eval | ≥ 5 % KL drop and no domain worse | expected ~10 % (Litwein) |
| Long context | `evals.longctx` RULER 32K/64K/128K | ≤ 2 pp below the higher-precision reference | linear-attn gates/decay (§2.4) |
| Rare-entity copy, termination/loops, tool validity | *04 §5* guards | same thresholds | the 4-bit quantizer alone cost 30 pts on non-Latin rare names in WaveCut's harness **[self-reported, *reap* table]** |
| Decode / prefill tok/s, MTP acceptance | `evals.speed` | ≥ baseline − 5 % (Qwen); record the GLM bit-speed gain | tiers add launches (§2.5) |

**Final rule.**
1. Pick the plan with the lowest *simulated* KL that fits the budget.
2. Build it with LSQ.
3. Keep GPTQ and DWQ only if each passes its own gain row.
4. Ship only if every guard passes and it beats the deployed baseline. Otherwise keep the baseline.

---

## 6. Failure modes

1. **Sens in one rounding, build in another.** If the allocator sees RTN curves while the build writes LSQ codes,
   the hull is mis-estimated and bytes go to the wrong tensors. Catch: `--sens-rounding` must match `--rounding`,
   and the build asserts this from `meta.json`.
2. **Wrong objective units.** Uniform weights over relative NMSE over-protect tiny tensors and under-weight the
   residual-dominant ones. Catch: compare the `energy` and `uniform` plans in the sweep. Expect `energy` to win;
   if not, the `resid_sq` probe is wrong.
3. **Group size not dividing the width.** PLE rows (160) take only g32/nvfp4. The build drops such options
   (`allowed_groups`). Watch the GLM MLA `embed_q`/`unembed_out` `MultiLinear` shapes.
4. **2-bit beyond the quadratic regime.** The knapsack under-predicts damage at NMSE > ~0.1. Catch: the
   additivity check. Cap the 2-bit tier by measured KL, not by the solver.
5. **Rotation applied globally.** It hurts outlier-column tensors (`fc_hidden` +51 %). Catch: "+H" only as a
   per-unit option with measured sens.
6. **nvfp4 PLE underflow.** At the table's true scale (std ≈ 0.0074) the e4m3 scales go subnormal, giving 1.8×
   the error (*quant Rec 6*). Catch: pre-scale and fold, and a lookup-equality test.
7. **Router or decay gate quantized by accident.** `qwen4_exp.quant_predicate` returns 8-bit for `mlp.gate` if
   our pin is lost. Catch: post-build assert that routers are bf16/fp32 and `in_proj_a/b` are bf16.
8. **DWQ overfitting the calibration domain.** Catch: held-out `eval.jsonl` split by session (*04 §4.2*); accept
   per band/layer only if held-out KL improves.
9. **DWQ breaking the fused verify path.** mlx-vlm fuses affine 4/5-bit at T ≤ 8. DWQ doesn't change bits, so
   this is safe. But tiers that change a layer from 4-bit to 3-bit lose the fused path (measured: still not
   slower, *quant §4*).
10. **Backward through kernels without a VJP** (`gated_delta` `metal_kernel`, QSA kernels). Silent zero or
    missing gradients. Catch: finite-difference gradient test on the tiny fixture with `use_kernel=False`.
11. **GPTQ on rank-deficient expert Hessians.** It overfits 128 rows. Catch: shrinkage plus a held-out NMSE on
    rows *not* used for H; fall back to LSQ for that expert if held-out NMSE is worse.
12. **Measuring KL against the wrong reference.** Keep `ref/` from the *unpruned* full-precision run (*04 §4.6*).
    For GLM the reference is the FP8 model dequantized, not bf16 "truth".
13. **Turning on `--kv-bits`.** The server crashes on the first batch (§2.10). The control daemon should refuse
    the flag.
14. **Memory during builds/DWQ while serving.** Q5–Q7 need the 44 GB server stopped or the Work-profile headroom
    checked first (*01 §4.6* watchdog). B2 DWQ is All-in only.

---

## 7. Stretch ideas

1. **Online Hadamard on expert `down_proj` (QuaRot R4 for MoE).** A `RotatedSwitchGLU` shim: rotate `h`
   (`[T, k, I]`, one call) before `gather_qmm`, and fold `H D` into `down_e` at build time. Cost ~0.5 ms/token on
   Qwen (§2.7). Gain ~5–7 % NMSE on `down_proj`. Do it only after LSQ/GPTQ/DWQ, and only if `down_proj` dominates
   the remaining KL.
2. **Residual-stream rotation (QuaRot R1) through hyper-connections.** RMSNorm and `hc_norm` are
   rotation-invariant per stream; `input_mix_weight_down`/`block_inject_weight` read the flattened streams and can
   absorb a block-diagonal `Q`; embed, lm_head, PLE `value_proj` and `o_proj`/`down_proj` outputs absorb it too.
   Each reader needs a proof that no elementwise op touches the residual. That makes the shared experts and
   attention near-Gaussian and could let them drop from 6-bit to 4–5-bit (~1–2 GB freed on GLM). Learn it with
   SpinQuant-style Cayley updates on the streaming objective **[hypothesis]**.
3. **One knapsack over {drop, 2b, 3b, 4b} per expert** (*04 §7.1*), using the same importance weights. It merges
   this chapter and pruning into one solve.
4. **Vector / trellis codes** (QuIP# E8P lattice, QTIP [arXiv:2406.11235](https://arxiv.org/abs/2406.11235)
   **[paper]**) for the cold tier at ~2 bpw with ~3-bit quality. This needs a custom Metal decode kernel; `gather_qmv`
   is the template. Bandwidth-bound GLM decode would gain the most.
5. **W4A8/nvfp4 `qqmm` prefill.** The FP modes have NAX `qqmm`. A *prefill-only* nvfp4 shadow copy of the hottest
   experts could speed long-document prefill. It costs RAM, so it is All-in only.
6. **Quantization-aware MTP fine-tune.** Train the drafter against the *quantized* trunk's own outputs. That
   raises acceptance at 2-bit experts, a pure speed gain (*07-speculative §4.5*).
7. **Session-adaptive tiers.** Swap which experts sit in the 4-bit tier per workload profile (*04 §7.4*),
   re-ranking `w_j` from live routing counts.
8. **Learned per-channel output scales (OmniQuant LET-lite) in B1 DWQ.** Add a per-output-channel scale vector to
   each quantized tensor; it folds into the next norm or into the scales. More knobs, same bytes.

---

## 8. Reproduce this chapter's measurement

`/private/tmp/claude-501/-Users-sauhard-work-naijm-local-ai/5b39c431-c6f9-4fdd-8029-dd210e9c770d/scratchpad/q5/fmt_rot.py`
(session scratchpad; copy it to `docs/plan/` if it should be kept). It reads 24 experts and 6 dense tensors from
the bf16 Qwen MTP drafter by byte range, then computes RTN / LSQ / ±Hadamard NMSE for every MLX format and times
`mx.hadamard_transform`:

```bash
uv run python -I <scratchpad>/q5/fmt_rot.py \
  ~/.cache/huggingface/hub/models--sh0wie--Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16/snapshots/*/model.safetensors
```

## Sources

- Installed source: `mlx.core.quantize/dequantize` docstrings; `mlx_lm/models/prism_hadamard_qwen35.py`;
  `mlx_lm/quant/{dwq,gptq,awq}.py`; `mlx_lm/LEARNED_QUANTS.md`; `mlx_vlm/models/qwen4_exp/language.py
  (quant_predicate)`; `mlx_vlm/models/gated_delta.py`; repo `build/{alloc,pipeline,families,make}.py`,
  `calib/{sens,probe,run}.py`, `docs/contracts/{build,calib}.md`.
- Papers: GPTQ [2210.17323](https://arxiv.org/abs/2210.17323) · AWQ [2306.00978](https://arxiv.org/abs/2306.00978) ·
  QuIP [2307.13304](https://arxiv.org/abs/2307.13304) · QuIP# [2402.04396](https://arxiv.org/abs/2402.04396) ·
  QuaRot [2404.00456](https://arxiv.org/abs/2404.00456) · SpinQuant [2405.16406](https://arxiv.org/abs/2405.16406) ·
  QTIP [2406.11235](https://arxiv.org/abs/2406.11235) · BRECQ [2102.05426](https://arxiv.org/abs/2102.05426) ·
  OmniQuant [2308.13137](https://arxiv.org/abs/2308.13137) · EfficientQAT [2407.11062](https://arxiv.org/abs/2407.11062) ·
  HAWQ-V2 [1911.03852](https://arxiv.org/abs/1911.03852) · LQER [2402.02446](https://arxiv.org/abs/2402.02446) ·
  KIVI [2402.02750](https://arxiv.org/abs/2402.02750) · MoE PTQ benchmark [2406.08155](https://arxiv.org/abs/2406.08155) ·
  FP4 vs INT4 [2509.23202](https://arxiv.org/abs/2509.23202) · OCP MX spec (linked from the `mx.quantize` docstring).
- Community (all **[self-reported]**; links in *quant §2–3*): oQ/oQe docs and PR #4385, OptiQ, ISTA GSQ-RCO,
  Litwein DWQ card, pipenetwork ablation, nace-ai, orcarouter, AesSedai, pfeifferj, sh0wie, WaveCut.
