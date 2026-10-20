# 06 — Recovery: distillation and healing after compression

> Part of the master plan for the best co-tenant local model on this MacBook Pro M5 Max (40-core GPU, 128 GB,
> 614 GB/s, macOS 27). Chapter 04 decides *what to cut* (REAP pruning) and the quantization research decides *how many
> bits each tensor keeps*. This chapter covers what happens next: **training a small set of parameters of the
> compressed student so it moves back toward the uncompressed teacher.** It builds on `PLAN.md`,
> `docs/research/{quant,reap,decode,mac}.md` (cited as *quant §n*, *reap §n* …) and the sibling chapters
> `01-memory.md`, `04-pruning.md`, `07-speculative.md`, `08-system.md` (cited as *01 §n* …). It does not re-verify their
> findings. It agrees with all of them, and where it refines one (§3.4), it says so and shows the evidence.
>
> Tags: **[paper]** arXiv or peer-reviewed · **[src]** read in the installed packages (mlx 0.32.4, mlx-lm 0.32.2,
> mlx-vlm 0.7.9) or in this repo · **[self-reported]** a model card, README or blog number that was not reproduced ·
> **[measured]** run on this machine for this chapter, using random tensors only and no model weights (§3.3) ·
> **[computed]** arithmetic from the real `config.json` files, with the working shown · **[est]** an estimate with
> stated assumptions that a named measurement in §5 replaces · **[hypothesis]** a proposal with no published evidence.
>
> Units: **GB = 10⁹ bytes** for files and tensors. **GiB = 2³⁰** for process caps, to match `serve/config.py`
> (*01 §0*).

---

## 0. TL;DR (what this chapter decides)

1. **Recovery is the last 10–30 % of the gap, not a rescue.** The best published MLX data point is Litwein's KL-DWQ on
   Qwen3.8-Flash-Next REAP-320 at 3 bits. It cut KL20 by **11.2 %** and recovered **33.6 % of the pruning gap**, from
   11.75 M teacher tokens, and only 6 of 12 layer bands passed its held-out gate **[self-reported]**. Pruning damage that
   is "unrepairable by giving the experts more bits" (sh0wie's rare-name collapse below 288 experts, *reap §3*) is not
   repaired by distillation either. **Never prune deeper because recovery exists.** Choose K by chapter 04's rules
   first.
2. **Train quantizer parameters, not adapters.** DWQ trains only the affine **scales and biases** of the quantized
   weights against teacher logits. The result is the *same file format, at the same size and the same speed*. A LoRA
   adapter on a ≤ 4-bit tensor either stays unfused, which costs decode time on this overhead-bound machine (§2.3), or
   gets re-quantized when fused, which adds back fresh rounding noise. **Recommendation: DWQ first. Router-only
   distillation second. LoRA only on the ≥ 6-bit non-routed trunk, and only if DWQ plateaus.**
3. **The stock tools do not run on these models, so we write a small loop.** `mlx_lm.dwq` cannot load `qwen4_exp` or
   `glm5_next` (*quant §1*, verified). mlx-vlm 0.7.9 ships a trainer (`mlx_vlm/trainer/`: SFT, DPO, ORPO, GRPO, LoRA
   including `LoRASwitchLinear`, and a pluggable `TrainingTask`) but **no KL-distillation objective and no
   quantizer-parameter mode** **[src]**. The new `recover/` package (§4.1) is about 600–900 lines. It reuses mlx-lm's
   Metal `kl_div_loss` kernel and `grad_checkpoint`, and the mlx-vlm model classes.
4. **Three measured facts set the training budget on this machine** **[measured, §3.3]**:
   - **Backward through the linear-attention layers is the bottleneck.** In train mode, mlx-vlm switches GatedDeltaNet
     and KDA to the pure-ops sequential path, because the Metal kernels have no VJP **[src]**. That path needs
     **6.3 MiB per token per layer at Qwen shapes and 9.1 MiB at GLM shapes**, and takes 0.7–2.5 ms per token per layer.
     The sequence length that fits is 1–2K tokens, not 16K.
   - **Scale and bias gradients through `gather_qmm` are cheap only when the indices are sorted.** Unsorted, the peak
     memory grows with *routed rows × out × in*: 16 GB for a 512-token toy batch. Sorted, it is about one fp32 dense
     dW per expert tensor, independent of tokens. mlx-vlm's `SwitchGLU` sorts whenever there are ≥ 64 routes
     **[src, `switch_layers.py:298`]**, so real training batches take the safe path. Any custom loop must keep it.
   - `mx.set_memory_limit` did **not** contain either spike, consistent with *mac §2*. The training script must bound
     its own working set.
5. **Memory: one band of four MoE layers at a time fits the All-in profile, and nothing fits Work.**
   - GLM K=144 at T = 1024 tokens: **≈ 94 GB (88 GiB)**.
   - Qwen 512-expert W1 at T = 2048: **≈ 85 GB (79 GiB)**.
   - Training all scales of all layers at once needs 60–76 GB of Adam state alone, so it does not fit at all (§2.5).
   - Training therefore runs overnight in the **calibrate/all-in profile with the server stopped** (*08 §4.1*).
6. **The teacher can score locally but cannot generate locally.**
   - Layer-wise streaming (`calib/`) scores teacher-forced text at about **390 tok/s for GLM and 800 tok/s for Qwen**.
     That is 10 M tokens per night for Qwen, or 7 h for GLM, written as top-64 logprobs + a tail bucket: **388 B/token,
     3.9 GB per 10 M tokens** **[computed]**.
   - Teacher *generation* through streaming would re-read 328 GB of weights for every decode step.
   - So data the teacher writes (completions, reasoning traces) comes from the cloud: about **$35–110 per model for
     tens of millions of tokens** on 8×H100 or 4×H200 at today's prices **[est, §2.6]**. Or it is skipped in favour of
     **on-policy** data, which the *student* writes and the local teacher scores. That is the better recipe anyway
     (GKD, §2.7). Personal transcripts never leave the machine.
7. **Schedule: 7 nights after about 4 days of coding.**
   - Each stage has a held-out exit gate, and a stage that misses its gate is dropped, not tuned forever.
   - Order: teacher signal → layer-local refit → band-wise end-to-end DWQ → router KD → on-policy personal round →
     MTP re-alignment → full gate (§4.8).
   - Swap the result into `serve/` only if it beats the un-recovered build on the `evals.gate` tier.

---

## 1. Why it matters (intuition)

**Compression adds independent errors, and the network was never trained to absorb them.** Every quantized weight
is off by a small rounding error. Every pruned expert means some tokens get their k-th choice replaced by a
lower-ranked substitute (*04 §2.2*). Each error on its own is small. But they compound through 45–48 layers, and the
residual stream at layer ℓ is no longer what layer ℓ+1 was trained to read. Layer-local fixes (GPTQ, the oQe
least-squares refit, chapter 04's re-calibration on the pruned model) make each layer locally as close as possible
to its original. They cannot make layer ℓ+1 *compensate* for what layer ℓ got wrong, because they never see the final
output.

**Distillation adds the missing feedback loop.** Run the compressed student. Compare its next-token distribution with
the teacher's. Push gradients back through the whole network, and let a few chosen parameters move so that the
*errors cancel downstream* instead of piling up. That is why end-to-end KL beats layer-wise reconstruction even with
the same trainable parameters: the per-expert least-squares healing in *reap §6* lowered reconstruction error in 41 of
42 layers and still *lost* end to end (top-1 0.8369 vs 0.8425) **[self-reported]**. That is local fidelity against
global fidelity.

**Why the student should learn from the teacher's distribution, not from text.** The goal is to *restore* the
teacher, not to teach anything new. A teacher's soft distribution (for example "70 % `return`, 20 % `yield`, 5 %
`raise`") carries far more information per token than the one-hot next token. It also makes the objective
**self-limiting**: once the student matches the teacher, the gradient is zero. Fine-tuning on hard labels from
another model, such as Claude's replies in your transcripts, would instead pull the student *away* from its own
distribution, toward a different style and different tool formats. That is the "tool-format drift" failure in §6.

**Why the trainable set must be tiny.** The student holds 120–152 B routed parameters. Training even 5 B of them would
need 80 GB of optimizer state (§2.5). The parameters that matter most for correcting rounding are the per-group
**scales and biases**: 2 numbers per 64 weights, about 3 % of the parameter count. They control exactly the thing
quantization got wrong, the mapping from integer codes back to real values. Training them is how "quantization-aware"
behaviour is bought after the fact, in the PEQA, EfficientQAT E2E-QP and DWQ line of work (§3.1). The router rows are
the other tiny, high-leverage set. Once experts are removed, the router's decision boundaries were drawn for a model
that no longer exists, and "Router KD" reports larger gains on *fine-grained* MoEs like ours (288/512 experts) than on
coarse ones **[paper, arXiv:2603.02217]**.

**What recovery cannot do.**
- It cannot restore knowledge that lived in deleted experts.
- It cannot fix what the training data never exercises. Train on 1K-token code and you get no guarantee for 128K-token
  RAG.
- Every gain must hold up on *held-out* domains. That is why §5 is as long as §4.

---

## 2. How it works (mechanics + key math)

### 2.1 The affine quantizer and what DWQ trains

MLX affine quantization stores, for each group g of G consecutive weights (G ∈ {32, 64, 128}), integer codes
`q_i ∈ {0 … 2^b − 1}`, a scale `s_g` and a bias `β_g`, both in bf16. At use time each weight is
`ŵ_i = s_g · q_i + β_g` **[src, `mx.quantize` docs]**. Storage costs 32/G extra bits per weight, the "+0.5 bpw at
g64" in *quant §1*.

**DWQ freezes the codes q and trains (s, β)** against a teacher **[src, `mlx_lm/quant/dwq.py`]**:
- The trainable set is every module with `bits < 8` in `affine` mode, and only its `scales` and `biases`
  (`m.unfreeze(keys=["scales","biases"])`).
- The parameters are kept in **fp32 master copies** and cast to bf16 for each forward pass.
- The optimizer is Adam with lr 1e-6 by default.
- The loss is KL at temperature 2 on stored teacher targets (top-1024 logits by default).
- The script warns when the final validation loss is worse than the initial one, but it **still saves the model**.
  Our loop must gate instead.

The gradient with respect to a scale is a group-reduced outer product. For a linear layer `y = Ŵx`:

$$\frac{\partial L}{\partial s_g} = \sum_{i\in g} q_i\,\frac{\partial L}{\partial \hat w_i},\qquad
\frac{\partial L}{\partial \beta_g} = \sum_{i\in g}\frac{\partial L}{\partial \hat w_i},\qquad
\frac{\partial L}{\partial \hat W} = \sum_t \delta_t\,x_t^{\top}$$

So computing it *mathematically* requires the dense weight gradient `∂L/∂Ŵ`, summed over the tokens that used that
weight. For a routed expert this is a sum over the tokens routed to it. That is the root of the `gather_qmm`
memory behaviour in §3.3. If the implementation forms a dW per token-row before reducing, memory grows with
tokens × out × in. If it groups rows by expert first (sorted indices), memory is one dW per expert.

**Why DWQ is safe in a way that full fine-tuning is not.** Each scale only stretches or shifts a group of 64 weights
around its quantized grid. It cannot move a weight to an arbitrary new value, so it cannot "learn new things", only
re-centre what is there. The mlx-lm notes say the same thing from experience: DWQ helps 2–4-bit models far more than
≥ 6-bit models, because at 6-bit "the loss starts out so low that it's difficult to reduce further", and g32
"doubles the number of tunable parameters and can work much better" **[src, `LEARNED_QUANTS.md`]**.

### 2.2 The loss: top-k + tail KL, temperature, direction

The teacher signal we can afford is truncated (§2.6). Write `p_1..p_k` for the teacher's top-k probabilities at one
position, `p_⊥ = 1 − Σ p_j` for the tail mass, and `q_j`, `q_⊥ = 1 − Σ q_j` for the student's probabilities on the
same token ids. The coarse-grained KL is

$$\mathrm{KL}_{k+\perp}(p\,\|\,q) = \sum_{j=1}^{k} p_j \log\frac{p_j}{q_j} \;+\; p_\perp\log\frac{p_\perp}{q_\perp}.$$

This is the same quantity `evals/kl.py` reports, and a lower bound on the full-vocabulary KL **[src, `evals/kl.py`
docstring]**. Training on it and evaluating on it keeps one consistent definition. Three design choices follow:

- **Keep the tail bucket. Do not renormalize within the top-k.** mlx-lm's DWQ gathers the student's logits at the
  teacher's top-k ids and applies a softmax over just those k values (`take_along_axis`, then `kl_div_loss` over k
  columns) **[src]**. That compares two *renormalized* distributions. It cannot see the student leaking mass to tokens
  outside the top-k, and that leakage is exactly where compressed models go wrong (rare tokens, repetition). The
  Sparse-Logit-Sampling paper shows that top-k KD is a **biased** estimator that teaches a miscalibrated, sharpened
  student **[paper, arXiv:2503.16870, ACL 2025]**. The tail bucket removes most of that bias. Its fix, importance-sampled
  token ids, is a stretch option (§7.4).
- **Temperature: T = 1 with stored logprobs.** T = 2 (mlx-lm's default) softens both distributions and
  emphasizes the tail. With truncated targets the tail is exactly what we do not have, and rescaling stored top-k
  logprobs by 1/T cannot reconstruct the tail mass correctly. If T = 2 tests better, store k = 256 for that run.
- **Direction: forward KL(p‖q), plus on-policy data later.**
  - Forward KL is mass-covering. It punishes the student for putting near-zero probability where the teacher has
    mass, which is how rare-entity and termination failures look.
  - Off-policy forward KL alone has an *exposure-bias* gap: the student is trained only on teacher-like prefixes,
    never on its own drifting ones.
  - GKD closes that gap by mixing in student-generated sequences scored by the teacher, optionally with a generalized
    JSD **[paper, arXiv:2306.13649]**. §2.7 uses this with the local streaming teacher.

### 2.3 LoRA / QLoRA healing, and why it is second choice here

LoRA learns `W + (α/r)·BA` with rank r ≪ min(in, out) **[paper, arXiv:2106.09685]**. QLoRA does the same on a
frozen 4-bit base **[paper, arXiv:2305.14314]**. The machinery exists for our models:
- `mlx_vlm.trainer.peft.LoRASwitchLinear` adds a per-expert `A[E, r, in]`, `B[E, out, r]` to a `SwitchLinear` or
  `QuantizedSwitchLinear` **[src]**.
- Its `fuse()` method dequantizes, adds `BA`, and **re-quantizes at the same bits and group** **[src, `lora_layers.py`
  `fuse`]**.

There are four reasons it comes second on this machine:

1. **Fusing re-introduces rounding noise.** Re-quantizing `Ŵ + BA` at 3 bits adds a fresh rounding error of the same
   order as the one LoRA was compensating for. QA-LoRA was designed to avoid exactly this: its group-wise
   constraint makes the adapter merge exactly into the quantizer's zero-points **[paper, arXiv:2309.14717]**. A
   merge-exact adapter is, in effect, a trained *bias* per group, which is a subset of what DWQ already trains.
2. **Unfused adapters cost decode time.** Qwen decode is overhead-bound: ≈ 12–13 ms of a ≈ 21–23 ms step is fixed
   per-kernel overhead (*07 §2.2*).
   - An unfused LoRA on 5 projections × 48 layers adds about 480 small matmul dispatches per token.
   - At ~5–10 µs each, that is **≈ 2.5–5 ms, or 10–20 % of decode** **[est]**.
   - DWQ adds zero.
3. **LoRA on routed experts is enormous here.** At rank 8 on gate/up/down of every routed expert:
   - GLM K=144: 3·8·(4096+2048)·144·42 = **0.89 B params**.
   - Qwen K=512: 3·8·(2560+640)·512·48 = **1.89 B params**.
   - With fp32 master + grad + Adam moments (16 B/param), that is **14–30 GB of state** **[computed]**.
   - Each expert sees only ~1/30–1/50 of the tokens, so these adapters also overfit fastest.
4. **The one published attempt at this scale diverged.** Full-model LoRA healing of a REAP-pruned GLM diverged after
   ~16 steps (lapadatbogdan, *reap §6*) **[self-reported]**. This is consistent with *04 §6 item 9*.

**Where LoRA still makes sense.** The ≥ 6-bit non-routed trunk is the exception: attention, KDA/DeltaNet projections,
shared experts and GLM's dense MLPs, about 3 % of parameters (*quant Rec 2*).
- Rank 16 there is **≈ 28 M params for Qwen and ≈ 56 M for GLM**, which is 0.45 / 0.9 GB of state **[est, from layer
  shapes]**.
- Fusing into a 6- or 8-bit tensor adds little noise. 6-bit rounding error is about 1/8 of 3-bit error: two times
  smaller per bit, three bits.
- Use it only if DWQ has plateaued, gate it like everything else, and fuse before serving.
- "LoRA learns less and forgets less" **[paper, arXiv:2405.09673]** is the right prior. Low rank limits both the
  recovery and the damage.

### 2.4 Router re-tuning: train only the gates

After REAP slicing, the router is exact for tokens that never chose a pruned expert, and substitutes for those that
did (*04 §2.2*). Three ways to adapt it, cheapest first:

| Method | Trainable | Objective | Needs backprop through the network? | Evidence |
|---|---|---|---|---|
| **R0** Per-layer gate temperature τ_ℓ and offset (2 scalars/layer) | 90–96 | end-to-end KL | yes, but it converges in a few hundred steps | **[hypothesis]**. A cheap probe of whether routing is miscalibrated at all |
| **R1** Layer-local router refit on cached hidden states (*04 §7.5*) | the surviving router rows of one layer: GLM 144×4096 = 0.59 M, Qwen 512×2560 = 1.3 M | ‖y_full − y_pruned‖² on cached layer inputs | no (rides the calib stream) | **[hypothesis]** in 04 |
| **R2** Router KD: all router rows, end-to-end | GLM 42×144×4096 = **24.8 M**; Qwen 48×512×2560 = **62.9 M** (fp32) | KL to the teacher | yes | **[paper, arXiv:2603.02217]**: consistent recovery across pruning, editing and merging, larger on fine-grained MoEs. Router-only LoRA beat all-module LoRA at matched rank with half of Mixtral's experts removed **[paper, arXiv:2608.07890]** |

The mechanics need care:
- **Top-k is not differentiable.** The gradient reaches the router only through the combine weights of the selected
  experts: the GLM sigmoid scores renormalized × 2.5, or the Qwen softmax renormalized over top-10. A logit that
  rises enough *changes the selection*. This is the intended effect, but it is a discrete jump that small learning
  rates cross only rarely. Use lr ≈ 1e-5 to 1e-4 for the router (fp32, 62.9 M params, 1 GB of Adam state) and
  monitor the *selection-change rate* per layer, not only the loss.
- **Keep GLM's `e_score_correction_bias` frozen**, as *04 §2.8* says. It only balances load during training. Moving
  it shifts selection for every token with no quality upside on one device.
- **Qwen normalizes over the selected set.** Raising one selected expert's logit lowers the others' weights inside
  the top-10. That is why R2 must be trained end to end and not per expert.
- **The MTP drafter has its own router copy** (*04 §2.9*). Router KD changes the trunk's routing, not the drafter's.
  The drafter is re-aligned afterwards (§2.8).
- **Routers stay unquantized** (*quant Rec 2*), so R2 trains them directly. No quantizer is involved.

### 2.5 Memory math for training a ≈ 68 GB quantized MoE student in 128 GB

Everything trains *on top of* the resident student. Inputs are from the real configs: GLM H = 4096, 45 layers
(42 MoE, 34 KDA, 11 DSA), 4 hyper-connection streams, V = 154,880. Qwen H = 2560, 48 layers (36 GatedDeltaNet, 12
QSA), 4 streams, V = 248,320 **[computed from `config.json`]**.

**(a) Trainable-state cost.** Adam with fp32 master + fp32 grad + m + v is 16 B per trainable scalar. At g64 there
are 2 scalars per 64 weights (halve this for g128):

| Trainable set | GLM K=144 | Qwen K=512 (W1) | Qwen K=320 |
|---|---|---|---|
| Routed params | 152.2 B | 120.8 B | 75.5 B |
| Scales + biases, all routed layers | 4.76 B scalars → **76.1 GB** state | 3.77 B → **60.4 GB** | 2.36 B → **37.7 GB** |
| … per MoE layer | 113 M → **1.81 GB** | 78.6 M → **1.26 GB** | 49 M → 0.79 GB |
| Band of 4 layers | **7.2 GB** | **5.0 GB** | 3.1 GB |
| Non-routed trunk scales (≥ 6-bit, rarely worth it) | ~278 M → 4.4 GB | ~153 M → 2.5 GB | same |
| Router rows (R2) | 24.8 M → 0.4 GB | 62.9 M → 1.0 GB | 39.3 M → 0.6 GB |
| LoRA r=16, non-routed trunk | ~56 M → 0.9 GB | ~28 M → 0.45 GB | same |

*quant §3* already found that "all scales at once" (the stock `mlx_lm.dwq` behaviour) does not fit. This table puts a
number on it: 60–76 GB of state on top of a 60–68 GB model. **Band-wise training is the necessary choice, not a
nice-to-have**, which is also what Litwein did. Also keep Qwen's 51 B-param PLE table on its `ple_storage` memmap: as
an mx module, `unfreeze` would make its 6.4 GB of scales trainable, adding ~51 GB of state (*quant §3*).

**(b) Activation cost with per-layer gradient checkpointing.**
- **Checkpoints.** Only the residual streams at layer boundaries are kept:
  - GLM: 4 × 4096 × 2 B = 32.8 KB/token × 45 layers.
  - Qwen: 20.5 KB/token × 48 layers.
- **Live set during backward.** The live set is one layer being recomputed. Its largest term is the
  recurrent-attention backward on the ops path, measured in §3.3:
  - **≈ 9.1 MiB/token** at GLM KDA shapes (64 heads × 128 × 128).
  - **≈ 6.3 MiB/token** at Qwen DeltaNet shapes (48 v-heads, 16 k-heads, 128 × 128).
- **MoE layers inside the band** also need a transient fp32 dense dW per projection for the scale gradient (§3.3):
  - GLM: 144 × 4096 × 2048 × 4 B = **4.8 GB per projection**.
  - Qwen 512: 512 × 640 × 2560 × 4 B = **3.4 GB**.
  - Three projections: 14.5 GB / 10 GB, if MLX schedules them concurrently.
- **Logits** (fp32, plus their gradient): `T × V × 8 B`. Chunk the loss over 512-token slices to cap this at ≈ 1 GB.

**(c) Totals** [computed + measured terms; ±15 %]:

| Item | GLM K=144 (W2 build, 66.4 GB), T = 1024 | GLM, T = 2048 | Qwen W1 (60.5 GB), T = 2048 | Qwen W1, T = 1024 |
|---|---|---|---|---|
| Student weights resident | 66.4 | 66.4 | 60.5 | 60.5 |
| Layer-boundary checkpoints | 45 × 32.8 KB × 1024 = 1.5 | 3.0 | 48 × 20.5 KB × 2048 = 2.0 | 1.0 |
| Recompute peak: max(recurrent bwd, MoE dW) | max(9.9, 14.5) = 14.5 | max(19.4, 14.5) = 19.4 | max(13.6, 10.1) = 13.6 | max(7.2, 10.1) = 10.1 |
| Logits + grad (chunked) | 1.3 | 1.3 | 1.0 | 1.0 |
| Band state (4 MoE layers) | 7.2 | 7.2 | 5.0 | 5.0 |
| MLX free-buffer cache, misc | 3 | 3 | 3 | 3 |
| **Total** | **≈ 94 GB = 88 GiB** | ≈ 100 GB = 93 GiB | **≈ 85 GB = 79 GiB** | ≈ 81 GB = 75 GiB |

How to read it:
- **Nothing trains alongside the running server.** The 44–48 GB server plus any of these is more than 128 GB. Training
  is a calibrate-profile job: server stopped, High Power, `caffeinate` (*08 §4.1*).
- **The cap is the All-in 104 GiB** (*01 §4.2*). It sits below `max_recommended_working_set_size` = 107.5 GiB ≈ 115 GB,
  the ceiling this task allows. GLM at T = 2048 is the most that fits with ≥ 10 GiB margin.
- **Gradient checkpointing caveat [src].** `grad_checkpoint(layer)` patches `type(layer).__call__`, so it covers every
  instance of *that one class*.
  - GLM has one `DecoderLayer` class.
  - Qwen has `Qwen4ExpDecoderLayer` and a separate `Qwen4ExpPLELayer` (PLE on layer 2).
  - Also, Qwen's forward runs through `Qwen4ExpBatchInvariantForward`, which calls sub-blocks directly.
  - **Verify on the tiny fixture that every layer is checkpointed**: peak memory must scale with 1 layer, not
    48. Otherwise checkpoint by wrapping the per-layer function yourself.
- **Sequence length is capped by the recurrent backward, not by attention.** At T = 16K the GLM recompute alone would
  be ≈ 155 GB. Long-context behaviour therefore cannot be trained end to end here. It is *guarded*, not optimized
  (§5, §6).
- **Sparse-attention kernels.** GLM's DSA path sets `use_kernel = not self.training` **[src]**. Qwen's QSA dispatcher
  picks `INDEXED_SPARSE_PREFILL` (a Metal kernel) with no training check **[src, `qsa_kernel.py`]**. Below the indexer
  budget (2048 for both models) the sparse selection covers every block, so dense attention is exact. **Train at
  T ≤ 2048 and force the dense plan in train mode.** This is a one-line patch in `recover/`, never in site-packages.

**(d) What trains when, in summary.** Routed-expert scales, band by band. Router rows, all at once (small). Non-routed
LoRA, optionally. Never: routed LoRA, integer codes (Litwein tried straight-through updates of the 3-bit codes and they
failed its gate **[self-reported]**), embeddings, `lm_head`, norms, `A_log`/`dt_bias`, the hyper-connection mixers, or
GLM's correction bias.

### 2.6 The teacher signal: what to store, and where to get it

**(a) Layer-wise streaming on this Mac (the default).** `calib/` already streams the source checkpoint one layer at a
time and writes top-64 reference logprobs for `eval.jsonl` (`ref/`). Its output was verified bit-identical to a full
forward on the tiny fixtures **[src, `docs/contracts/calib.md`]**. Extend it with `--ref-split calib`, so it writes the
same records for the training set. Throughput is taken from *04 §4.3*, which measured ~300 tok/s (GLM) and ~620 tok/s
(Qwen) *with* probes and a +30 % probe overhead. Without probes that gives **≈ 390 and ≈ 800 tok/s** **[est]**.

**In-RAM shards avoid SSD wear.** A pass over N tokens that keeps hidden states on disk writes N × 32.8 KB per layer
boundary. For 10 M GLM tokens that is 328 GB × 45 ≈ **15 TB written**, a real fraction of the SSD's endurance
(*04 §6 item 12*). Instead:
- Process shards small enough that both hidden-state buffers stay in RAM.
  - GLM: 0.25 M tokens × 32.8 KB × 2 = 16.4 GB, plus one bf16 layer (~15 GB) ≈ 31 GB.
  - Qwen: 0.75 M tokens × 20.5 KB × 2 = 31 GB, plus a ~5 GB layer ≈ 36 GB.
  - Both stay under the 40 GB calib contract.
- Re-read the weights for every shard. Reading costs no endurance: 328 GB at 6–9 GB/s ≈ 40–55 s per shard, against
  ≈ 640 s of GLM compute per 0.25 M tokens, so **< 10 % overhead and ~0 bytes written**.

**Storage for N tokens × top-k** (int32 id + fp16 logprob per entry + 4 B tail mass, i.e. 6k + 4 bytes/token)
**[computed]**:

| N tokens | k = 20 (Litwein) | **k = 64 (ours, = `ref/`)** | k = 256 | k = 1024 (mlx-lm default) |
|---|---|---|---|---|
| 1 M | 0.12 GB | 0.39 GB | 1.5 GB | 6.2 GB |
| 10 M | 1.2 GB | **3.9 GB** | 15 GB | 61 GB |
| 50 M | 6.2 GB | 19 GB | 77 GB | 307 GB |
| 200 M | 25 GB | 78 GB | 308 GB | 1.23 TB |

For comparison, full-vocabulary fp16 logits would be 310 KB/token (GLM) and 497 KB/token (Qwen), i.e. **3.1–5.0 TB per
10 M tokens**. top-64 is 800–1,300× smaller and already covers almost all the mass: Litwein reports ≈ 99.6 % at top-20
**[self-reported]**.

**Local time for teacher scoring** [est]: 10 M tokens takes **≈ 7.1 h for GLM** (10⁷ / 390) and **≈ 3.5 h for Qwen**,
which is one night each. 50 M tokens is 36 h / 17 h, i.e. 4–5 nights for GLM.

**(b) Renting cloud GPUs for a few hours.** Where cloud wins:
- (i) when N ≥ ~50 M tokens.
- (ii) when we want the *teacher to write* data. Local streaming would cost one full 328 GB weight read per decode step,
  ≈ 40 s per step even at batch 64. That is useless.

Both source models are served by vLLM: there are NVIDIA Dynamo recipes for GLM-5.3-Flash on 8×H200 (TP8, FP8), a
Baseten preset on 8×H100, and vLLM/SGLang recipes for Qwen3.8-Flash-Next **[self-reported, vendor docs]**.

| Job | Hardware | Price basis (Oct 2026) | Throughput assumption | Wall time | **Cost** |
|---|---|---|---|---|---|
| GLM teacher top-64 scoring, 100 M tokens of public data | 8×H100 SXM (640 GB; FP8 328 GB fits) | $2.0–4.0 per GPU-h (Vast H100 SXM ~$1.5–2.2, RunPod Secure ~$3.3, Lambda ~$4.0) → $16–32/h | 15–40 K tok/s prefill + on-GPU top-k **[est]**: 33 GFLOP/token at ~25 % of 8 × 989 TFLOP/s bf16 ≈ 60 K tok/s, halved for logprob extraction and I/O | 0.7–1.9 h + ~1 h setup (download, load) | **≈ $35–95** |
| Qwen teacher scoring, 100 M tokens (BF16) | 4×H200 (564 GB) or 8×H100 | H200 ~$2.5–3.6/GPU-h → $10–15/h | 14 GFLOP/token → ~2× GLM's rate | ~1–2 h total | **≈ $15–45** |
| GLM teacher *completions*, 20 M generated tokens (with top-64 logprobs) | 8×H200 | ~$20–29/h | 2–5 K tok/s aggregate at high concurrency **[est]**. The only H100 figure found is 1,430 tok/s for Qwen on 4×H100 at 64 concurrent requests **[self-reported, secondary]** | 1.1–2.8 h + setup | **≈ $40–110** |
| Bringing results home | — | — | 100 M × 388 B = 39 GB | ~10–50 min | — |

Prices are taken from comparison sites dated March–September 2026 that disagree with each other. Re-check the
providers' live pages before renting.

Cloud rules:
1. **Only public prompts go to the cloud.** Use the calib_data public domains (code, agentic, reasoning, longctx).
   Personal transcripts never leave the machine (*07 §4.5*, *08 §6*).
2. **The cloud job writes the same `ref/` record format**: token ids in the *target model's* chat template (the
   calib_data rendering), plus top-64 ids/logprobs/tail. The local trainer then cannot tell where a record came
   from.
3. **Use FP8 teacher weights for GLM**, which is the vendor format and what local streaming dequantizes. For Qwen, run
   the BF16 checkpoint, not the FP8 one, so the cloud teacher matches the `ref/` used for evaluation.
4. **Spot-check 1K tokens of cloud logprobs against local streaming** (KL ≤ 1e-3) before using a cloud batch. Different
   kernels, TP reductions and FP8 GEMMs give small but real differences.

**Verdict.** We need about 5–20 M tokens: Litwein used 11.75 M; mlx-lm's DWQ default is 2,048 samples × 1,025 tokens
≈ 2 M. The end-to-end trainer only consumes **0.5–2 M tokens per night** (§4.4). **Local streaming is enough for
scoring.** Rent the cloud only for teacher-written completions, and only if the on-policy round (§2.7) underperforms
(≈ $100, one evening).

### 2.7 Self-distillation on personal workload data

"Self-distillation" here means *the uncompressed model teaching its own compressed copy, on your traffic*. It is not
SFT on Claude's outputs. The data comes from the Claude Code transcripts that chapters 07 and 08 already plan to
extract and scrub: 1,567 sessions, 1.59 GB, metadata only **[measured, 08 §4.8]**.

- **Off-policy (cheap): contexts from transcripts, labels from the teacher.** Render each context (system + tool
  schemas + history) in the *target's* template. Append Claude's reply as the continuation and stream the teacher over
  it to get top-64 at every position. Train only on positions inside the reply and tool calls. The student learns
  *what the teacher would have predicted* on your kind of prompts, never Claude's tokens as targets. That is what keeps
  tool formats native.
- **On-policy (GKD-style): the student writes, the teacher scores.**
  - Sample 1–2 K held-in prompts and have the **student** generate with MTP on, at your serving temperature. That is
    ≈ 300 K tokens at 40–55 tok/s ≈ 1.5–2 h (`serve/` batch generation would help).
  - Stream the teacher over the generated sequences. That is ≈ 13 min for 300 K tokens on GLM.
  - Train with forward KL on a 30 % on-policy / 70 % off-policy mix.
  - This is exactly the setting where GKD reports its gains: the student is corrected on the prefixes it actually
    produces, which is where compounding errors live **[paper, arXiv:2306.13649]**.
  - The SDFT paper makes the related point that training on targets rewritten by the model itself avoids
    distribution shift and forgetting **[paper, arXiv:2402.13669]**.
- **The domain mix holds a floor.** Keep ≥ 30 % public general/reasoning/multilingual data in every batch.
  Agentic-only data "destroys arithmetic and other languages" in pruning calibration (*reap §4*), and the same risk
  applies to training.
- **Privacy.** Store only token ids + top-k under `artifacts/personal/` (0700/0600, git-ignored). Run the canary
  secret probe on the recovered model (*08 §6*). Never publish weights trained on personal data.

### 2.8 MTP head re-alignment after compression

The MTP layer drafts from the trunk's final hidden state h_t and the next token's embedding (*07 §2.4*). Every trunk
change shifts h_t: pruning, quantization, *and every recovery stage in this chapter*. The head was trained against the
original h_t, so its acceptance α drops even though the trunk got *better*. The published numbers:
- 76–81 % acceptance for an unpruned Qwen head on a K=320 trunk.
- About 70 % for GLM at any bit width.
- +2–3 % decode from a self-distilled sidecar.
All **[self-reported, reap §7]**.

**Rule: re-align the MTP head last, after the trunk is frozen for good.** The recipe is *07 §4.5 (step S5)*, and this
chapter does not repeat it:
- Train only the MTP layer's connectors, norms and attention, plus optional small expert LoRA.
- Target the **compressed trunk's own** next-token distribution, because acceptance means agreement with the trunk you
  serve.
- Feed the trunk's hidden states at prefill speed, with no backprop through the trunk. It is cheap: ≈ 2–4 GiB of
  activations and ≈ 1.9 GiB of state (*07 §4.5*).

One refinement from this chapter: the off-policy and on-policy corpora from §2.7 *are* S5's training contexts, so
collect them once.

**What to expect.** The recovery stages move h_t by a small amount (KL to the teacher changes by ~10 %), so expect
the *loss* of acceptance from recovery alone to be small (≤ 2 points) **[est]**. Most of S5's gain still comes from
adapting to your distribution. Measure α on held-out personal traces at T ∈ {0, 0.6} before and after (*07 §5*).

---

## 3. Evidence

### 3.1 Primary sources (methods)

| Claim | Source |
|---|---|
| Training only quantizer scales of a frozen low-bit model recovers fine-tuned quality at small memory | PEQA, [arXiv:2305.14152](https://arxiv.org/abs/2305.14152) (NeurIPS 2023) **[paper]** |
| Block-wise all-parameter training, then end-to-end training of *only quantization parameters* (E2E-QP); 2-bit Llama-2-70B on one A100 in 41 h, 69.48 vs 72.41 avg | EfficientQAT, [arXiv:2407.11062](https://arxiv.org/abs/2407.11062) **[paper; numbers self-reported]** |
| Calibrating each layer against the *full-precision* model's outputs (asymmetric) fixes accumulated error that symmetric GPTQ misses | GPTAQ, [arXiv:2504.02692](https://arxiv.org/abs/2504.02692) (ICML 2025) **[paper]**, the basis for §4.3's "teacher-trajectory" targets |
| DWQ algorithm, defaults, tips | `mlx_lm/quant/dwq.py`, `mlx_lm/LEARNED_QUANTS.md` **[src]** |
| Top-k KD is biased and miscalibrates; importance-sampled sparse logits are unbiased | Sparse Logit Sampling, [arXiv:2503.16870](https://arxiv.org/abs/2503.16870) (ACL 2025) **[paper]** |
| On-policy distillation (student samples, teacher scores) fixes train/inference mismatch | GKD, [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) **[paper]** |
| Self-generated targets reduce forgetting during fine-tuning | SDFT, [arXiv:2402.13669](https://arxiv.org/abs/2402.13669) **[paper]** |
| LoRA / QLoRA / quantization-aware LoRA that merges exactly | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685), [arXiv:2305.14314](https://arxiv.org/abs/2305.14314), QA-LoRA [arXiv:2309.14717](https://arxiv.org/abs/2309.14717) **[paper]** |
| LoRA underperforms full fine-tuning but forgets less | [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) **[paper]** |
| Router-only KD recovers compressed MoEs; larger gains for fine-grained MoEs | Router KD, [arXiv:2603.02217](https://arxiv.org/abs/2603.02217) **[paper]** (abstract read; no numbers in it) |
| Router-only LoRA beats all-module LoRA at matched rank after removing half of Mixtral's experts | [arXiv:2608.07890](https://arxiv.org/abs/2608.07890) **[paper; self-reported]** |
| Expert pruning dominates degradation when combined with quantization | [arXiv:2608.21693](https://arxiv.org/abs/2608.21693) (*04 §2.11*) **[paper]** |

### 3.2 Results on these models (all self-reported)

| Build | Recovery | Result | Source |
|---|---|---|---|
| Qwen3.8-FN REAP-320 oQ3e (MLX) | 3 rounds of KL-DWQ (scales/biases), whole resident student, gradient checkpointing, **12 layer bands**, gradient accumulation 2. Each band accepted only if better on 24 unseen records (6/12 passed). Teacher: bf16 completions + top-20 logprobs (≈ 99.6 % mass), 5,377 text records = **11.75 M tokens** | Held-out KL20 0.1065 → 0.1035 over v7→v9. **−11.2 % KL20 / −14.9 % CE-gap vs the pruned initializer = 33.6 % of the 0.047-nat gap.** Independent 457K-token gate: KL20 0.0845 → 0.0833. *Rejected:* fp32 master for lm_head/trunk; straight-through updates of 3-bit codes. No task scores | [Litwein card](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX) |
| GLM-5.3 REAP50 | Per-expert closed-form LS healing | Lower reconstruction error in 41/42 layers but **worse end to end**: top-1 0.8369 vs 0.8425, z = 11.8 | patrickbdevaney via *reap §6* |
| GLM-5.3 REAP-144 | Full-model LoRA healing | **Diverged after ~16 steps** | lapadatbogdan via *reap §6* |
| Qwen REAP-320 MTP | Self-distilled MTP sidecar | +2–3 % decode | Litwein MTPLX via *reap §7* |

**What the evidence does and does not say.**
- Recovery on these exact architectures has only been tried for (s, β) distillation, and it helps modestly.
- Every training-based healing that touched more parameters either diverged or lost end to end.
- No source reports task scores before and after recovery. **So whether a −10 % KL change moves HumanEval, BFCL or
  turns-to-solve is an open question**, and §5 has to answer it.
- Litwein's numbers come from a different runtime (oMLX), not from mlx-vlm.

### 3.3 Measurements made for this chapter [measured]

Scripts are in the session scratchpad: `gd_mem.py`, `gqmm_grad.py`, `gqmm_scale.py`. They use random tensors only.
The GPU was shared with the running 44 GB model server, so the timings are pessimistic. *Two runs went above the 2 GB
guideline (peaks of 12.9–18.5 GB and 16 GB) despite `mx.set_memory_limit`, which is itself a finding (§0 item 4). No
model weights were loaded.*

**(a) Gated-delta / KDA backward on the training (ops) path.** `gated_delta_update(..., use_kernel=False)` with
gradients with respect to q, k, v, a, b:

| Shape | T = 256 | T = 1024 | T = 2048 |
|---|---|---|---|
| Qwen DeltaNet (Hk 16, Hv 48, 128×128): peak, MiB/token | 7.9 | 6.7 | **6.3** |
| … fwd+bwd, ms/token/layer | 0.53 | 0.66 | **1.23** |
| GLM-like (64 heads, 128×128, scalar gate): peak, MiB/token | 10.4 | 9.3 | **9.1** |
| … fwd+bwd, ms/token/layer | 0.50 | 0.84 | **2.50** |

- Memory is linear in T: the autograd graph keeps one recurrent state per step, 48 × 128 × 128 × 4 B = 3.1 MB for Qwen.
- Time per token *grows* with T, because the ops path is a Python loop of T steps that builds a graph of size O(T).
- Taking the gradient with respect to v only is misleadingly cheap (0.2–0.9 GB total, flat in T), because the
  state-dependent branches get pruned. Real training needs the input gradients.
- GLM's real KDA uses a per-dimension gate, which is slightly costlier than the scalar gate measured here.

**(b) Scale/bias gradients through `gather_qmm`** (3-bit, g64; 8 experts, 256×1024; k = 4):

| Routed rows | unsorted: peak extra | sorted (`sorted_indices=True`): peak extra |
|---|---|---|
| 32 | 24 MiB | 8.1 MiB |
| 128 | 72 MiB | 7.9 MiB |
| 512 | 263 MiB | 7.1 MiB |

- Unsorted grows like rows × out × in × 2 B: a per-row dW is materialized. A 64-expert, 640×2560, 5,120-row toy hit a
  **16 GB** peak.
- Sorted stays at ≈ E × out × in × 4 B (8 MiB here), i.e. one fp32 dW per expert.
- mlx-vlm sorts when `indices.size ≥ 64` **[src]**. Training batches always qualify, but a custom loop that calls
  `gather_qmm` directly must sort.

**(c) Throughput model for end-to-end training** [est from (a) + FLOPs].
- A training step with frozen codes costs ≈ 4× forward FLOPs: forward, recompute, input-gradient pass, and the
  scale-gradient reduction.
- GLM: 4 × 33 GFLOP at ~7 TFLOP/s ≈ 19 ms/token, plus 34 KDA layers × 0.84 ms × 1.33 (recompute) ≈ 38 ms. **Total
  ≈ 57 ms/token → ≈ 0.5 M tokens per 8-h night at T = 1024.**
- Qwen: 4 × 14 GFLOP ≈ 8 ms, plus 36 × 1.23 ms × 1.33 ≈ 59 ms at T = 2048 (or 32 ms at T = 1024). **≈ 0.4–0.7 M
  tokens per night.**
- Band-wise training is cheaper: layers below the band need no backward, which roughly halves the average cost.
- With a chunked delta-rule op that has its own VJP (§7.1; ≈ 5× on the recurrent term **[hypothesis]**), the figures
  rise to ≈ 1 M (GLM) and ≈ 2 M (Qwen) tokens per night.

### 3.4 Refinements to sibling chapters (not contradictions)

- *07 §4.5* says mlx-vlm "ships none" (no training loop). More precisely, 0.7.9 has `mlx_vlm/trainer/` with a generic
  `TrainingTask` engine, LoRA/DoRA for Linear, Embedding and SwitchLinear, SFT, DPO, ORPO and GRPO. It has no KL-to-teacher
  objective and no MTP-specific code, so 07's conclusion (write our own objective) stands. The engine's gradient
  accumulation and checkpoint plumbing can be reused through a custom `TrainingTask` **[src]**.
- *04 §6 item 9* ("only per-layer scalars") is about *healing fitted to reconstruction*. This chapter's end-to-end
  KL stages are a different object. They keep 04's spirit: tiny trainable sets, and accept only if held-out KL
  improves.

---

## 4. Plan for this machine

### 4.1 Code: the `recover/` package (new; ≈ 600–900 lines + tests on the tiny fixtures)

```
recover/teacher.py   # wraps calib streaming: --ref-split calib, in-RAM shards, writes ref-format shards (top-64 + tail)
recover/loss.py      # KL_{k+⊥} (§2.2) on gathered student logprobs, chunked over 512 positions; optional mlx-lm kl_div_loss for full-vocab tests
recover/local.py     # stage A: layer-local (s,β) and router-row refit to teacher-trajectory targets, rides the calib stream
recover/dwq.py       # stage B: band-wise end-to-end DWQ on mlx_vlm.load(student); fp32 masters; per-band accept gate
recover/router.py    # stage C: R0/R2 router KD (bias frozen), selection-change-rate logging
recover/onpolicy.py  # stage D: student generation via serve/ engine → teacher scoring via recover.teacher
recover/train_mode.py# train-mode patches inside our code only: force dense QSA plan for T<=2048, checkpoint every layer class
tests/test_recover_*.py  # tiny glm5_next/qwen4_exp fixtures: grads non-zero only on intended params, KL decreases, saved model loads in mlx_vlm + serve
```

Design rules:
- **Re-use, don't fork.** Load the model with `mlx_vlm.load`, use `model.train()` (which flips the recurrent and DSA
  paths to differentiable ops), freeze everything, and then `unfreeze(keys=["scales","biases"])` on the
  `QuantizedSwitchLinear` modules of the current band. This mirrors `dwq.py`'s `unfreeze` **[src]**.
- **Every stage writes a new model dir** (`models/<name>-rA`, `-rB` …), with the same `config.json["quantization"]`
  and only the scales/biases (and router) tensors changed. A diff tool checks that nothing else moved (failure mode 7).
- **Self-bounded memory.** The script computes the §2.5 table for the chosen band and T before it starts, refuses to
  run above 100 GiB, and checks `mx.get_peak_memory()` after every step. If the peak exceeds plan + 10 %, it saves and
  exits.
- **No edits to `pyproject.toml`.** No new dependencies.

### 4.2 Night 1: teacher signal (local streaming)

```bash
# server stopped, calibrate profile (08 §4.1); one model per night
scripts/profile.sh calibrate
uv run python -m recover.teacher --model qwen --src $QWEN_SNAP --data artifacts/qwen/data-final \
    --split calib --tokens 8M --topk 64 --shard-tokens 750K --mem-gb 40 --out artifacts/qwen/teacher
uv run python -m recover.teacher --model qwen --src $QWEN_SNAP --data artifacts/personal/qwen \
    --split calib --tokens 2M --topk 64 --out artifacts/personal/qwen/teacher        # off-policy personal (§2.7)
```

- **Time:** 10 M tokens / 800 tok/s ≈ 3.5 h for Qwen. For GLM the same 10 M takes ≈ 7.1 h, so do 6 M ≈ 4.3 h
  [est].
- **Disk:** 10 M × 388 B ≈ 3.9 GB. Writes ≈ the output only (in-RAM shards).
- **Data:** the calib_data mix (code, agentic, reasoning, longctx truncated to ≤ 2K windows for training) plus the
  general floor. The held-out `eval.jsonl` and its `ref/` stay **untouched** and are split by session (*04 §6 item 5*).
- **Exit:** a spot check of 1K tokens matches `ref/`-style streaming within KL ≤ 1e-4 (same code path). The token
  count matches the plan.

### 4.3 Night 2: stage A, layer-local refit to teacher-trajectory targets (co-tenant-safe, ≤ 40 GB)

For each MoE layer ℓ, in the same streaming pass:
- Inputs are `X_ℓ^S`, the *student's* own hidden states, produced by the already-refit student layers 0..ℓ−1.
- Targets are `Y_ℓ^T`, the *teacher's* layer output on the *teacher's* hidden states.
- Minimize `‖f_ℓ^S(X_ℓ^S; s, β) − Y_ℓ^T‖²`, weighted by the imatrix, over (s, β) of that layer's experts and,
  optionally, its router rows (R1).

This is GPTAQ's asymmetric-calibration idea **[paper]** with a gradient optimizer, and EfficientQAT's Block-AP
restricted to (s, β) **[paper]**. Unlike oQe's least-squares refit (*quant Rec 3*), it also corrects drift accumulated
upstream, because the target is the teacher's trajectory, not the teacher's layer applied to student inputs.

```bash
uv run python -m recover.local --model glm --src $GLM_SNAP --student models/glm53-k144-v1 \
    --data artifacts/glm/data-final --tokens 256K --epochs 2 --lr 1e-5 --router-rows \
    --mem-gb 40 --out models/glm53-k144-rA
```

- **Cost** [est]:
  - This is one calib-like pass over 0.25 M tokens with *two* streams (teacher and student). GLM ≈ 0.25 M / 390 × 2 ≈
    22 min of forward passes.
  - Add the per-layer optimization: one MoE layer ≈ 2 GFLOP/token × 3 for training × 0.5 M token-epochs ÷ 7 TFLOP/s
    ≈ 7 min per layer, ≈ 5 h for 42 layers.
  - Qwen: ≈ 2 h.
  - SSD: 0.25 M × 32.8 KB × 2 streams × 45 boundaries ≈ 0.74 TB written, or 0 with in-RAM shards at ≤ 0.12 M tokens.
- **Memory:** one bf16 teacher layer (15 GB) + the student layer (~1.5 GB) + 2 × 2 boundary buffers. **≤ 40 GB**, so
  this stage is the only one that can run while you code (in the Work profile, next to a 48 GB server: 88 GB ≤ 128 GB,
  leaving ≈ 40 GB). Prefer to run it at night anyway.
- **Exit:**
  - Held-out per-domain KL (`evals.kl`) falls **≥ 3 %** relative, with a paired-bootstrap CI excluding 0, in ≥ 3 of
    4 domains, and no domain worse.
  - Otherwise discard rA and start stage B from v1. This stage is a cheap initializer, not a requirement.

### 4.4 Nights 3–4: stage B, band-wise end-to-end DWQ (All-in, server stopped)

```bash
uv run python -m recover.dwq --student models/glm53-k144-rA --teacher artifacts/glm/teacher \
    --bands 4 --order top-down --seq 1024 --batch 1 --grad-accum 8 --lr 2e-6:1e-5 --warmup 50 \
    --tokens-per-band 120K --val artifacts/glm/teacher-val --accept-margin 0.5% \
    --mem-gb 100 --out models/glm53-k144-rB
uv run python -m evals.kl --model models/glm53-k144-rB
```

- **Bands:** GLM has 42 MoE layers → 11 bands of 4 (the last one has 2). Qwen has 48 → 12. **Order top-down**: later
  layers are cheaper to train, because the backward stops at the band's start, and they sit closest to the logits, so
  they give gains early. **[hypothesis]** Compare with bottom-up on one night if time allows.
- **Learning rate:** the mlx-lm guidance is that lower bits need higher rates. Sweep {2e-6, 5e-6, 1e-5} on the first
  band for 200 steps and keep the best held-out KL. Use Adam without weight decay, gradient clip 1.0, and fp32
  masters cast to bf16 per step **[src, dwq.py]**.
- **Accept per band:** after training a band, compute held-out KL on 64 unseen records, split by session, for which
  teacher targets exist. **Keep the band's new (s, β) only if KL improves by ≥ 0.5 % relative, and revert otherwise.**
  This is Litwein's gate. Expect about half the bands to fail **[self-reported]**.
- **Budget** (§3.3c, ±50 %): ≈ 0.5–1 M tokens per night. With 120 K tokens per band × 11 bands = 1.3 M tokens, this is
  **2 nights for GLM** and 1–2 for Qwen.
- **Rounds:** Litwein ran 3 rounds. Run a second round only if round 1 gained ≥ 5 %.
- **Memory:** §2.5c gives GLM T = 1024 ≈ 88 GiB and Qwen T = 2048 ≈ 79 GiB, under the 100 GiB self-limit.
- **Exit (stage B):**
  - Per-domain held-out KL **−8 % or better** vs v1, CI excluding 0 in ≥ 3/4 domains, with no domain worse beyond its
    CI.
  - top-1 agreement not lower.
  - The guards in §5 pass.
  - **If it gains < 3 %, stop the whole recovery track for this model** and spend the nights on chapter 04's K/bits
    trade instead.

### 4.5 Night 5: stage C, router KD (R0 then R2)

```bash
uv run python -m recover.router --student models/glm53-k144-rB --teacher artifacts/glm/teacher \
    --mode temp-offset --steps 300 --seq 1024 --out models/glm53-k144-rC0          # R0: 2 scalars/layer, ~1 h
uv run python -m recover.router --student models/glm53-k144-rB --teacher artifacts/glm/teacher \
    --mode rows --lr 3e-5 --tokens 400K --freeze-bias --log-selection-change --out models/glm53-k144-rC
```

- **Order:** run R0 first. If R0 gains nothing (< 1 % KL), the router is not miscalibrated in a way that a global
  shift can fix. Run R2 anyway, because it can fix *per-expert* boundaries, but cap it at 400 K tokens.
- **Memory:** the same as stage B, minus the band state, plus ≈ 0.4–1 GB for the router. Backward runs through all
  layers, because every layer has a router, so the cost per token is the full ≈ 57 ms (GLM). 400 K tokens ≈ 6.3 h.
- **Exit:**
  - KL −2 % or better vs rB, with the same CI rule.
  - The **selection-change rate** on held-out tokens is ≤ 5 % of routing slots per layer. More than that means the
    router is being re-learned, not nudged.
  - The expert-load histogram has no new "dead" experts. Load must not collapse onto a few survivors (failure mode 5).
- **Interaction with ch. 01 tiering:** if a tiered/masked build is used (*01 §4.2*), train R2 under the *Work* mask
  (the hardest one), and check KL under the other masks too.

### 4.6 Night 6: stage D, on-policy personal round

```bash
# evening: student generates on 1.5K held-in personal + public prompts, MTP on, serving temperature
uv run python -m recover.onpolicy generate --model models/glm53-k144-rC --prompts artifacts/personal/prompts-train.jsonl \
    --max-new 1024 --temperature 0.6 --out artifacts/personal/glm/onpolicy     # ≈ 300K tokens, ≈ 1.5–2 h
uv run python -m recover.teacher --model glm --src $GLM_SNAP --data artifacts/personal/glm/onpolicy \
    --split calib --topk 64 --out artifacts/personal/glm/onpolicy-teacher     # ≈ 15 min
uv run python -m recover.dwq --student models/glm53-k144-rC --teacher artifacts/glm/teacher,artifacts/personal/glm/teacher,artifacts/personal/glm/onpolicy-teacher \
    --mix 0.4,0.3,0.3 --bands accepted-only --tokens 400K --out models/glm53-k144-rD
```

- Re-train **only the bands accepted in stage B**, at half the stage-B learning rate.
- **Exit:**
  - KL on held-out *personal* traces −3 % or better vs rC.
  - **No regression on held-out public domains** (the same CI rule).
  - `evals.replay` tool-call validity not lower (*08 §4.8*).
  - The canary secret probe is clean.

### 4.7 Night 7: MTP re-alignment and the full gate

```bash
uv run python -m drafter.collect --trunk models/glm53-k144-rD --contexts artifacts/personal/glm ...   # 07 §4.5 S5
uv run python -m drafter.train_mtp ...                                                                 # 07 §4.5
for s in kl speed code math tools longctx; do uv run python -m evals.$s --model models/glm53-k144-rD; done
uv run python -m evals.gate --candidate models/glm53-k144-rD --baseline models/glm53-k144-v1             # 08 §4.8
```

- **MTP exit:** held-out α ≥ the v1 head on the v1 trunk, minus 1 point. 07's stop rule applies to further drafter
  work.
- **Swap rule:** promote rD over v1 only if the gate PASSes: KL improves, and code, tools and longctx are within noise
  or better (§5). If only some stages passed, promote the last stage that passed.

### 4.8 The schedule at a glance

| When | Stage | Profile / peak | Time | Output | Exit criterion (else) |
|---|---|---|---|---|---|
| Days 1–4 | Write `recover/` + tests on tiny fixtures; tiny-fixture smoke of each stage; measure real step time on a 2-layer slice | dev, < 2 GB | 3–4 days | package | Grads touch only intended tensors; the saved model loads in `mlx_vlm` + `serve/`; KL falls on the tiny fixture |
| Night 1 | Teacher top-64 on 6–10 M public + 2 M personal tokens | calibrate, ≤ 40 GB | 3.5–7 h | `artifacts/<m>/teacher` (≈ 4 GB) | Token count, spot-check vs `ref/` |
| Night 2 | A: layer-local (s, β) + router rows to the teacher trajectory | ≤ 40 GB | 2–5 h | `-rA` | KL −3 % or better (else skip A) |
| Nights 3–4 | B: band-wise E2E DWQ, gated per band | all-in, ≤ 100 GiB | 2 × 8 h | `-rB` | KL −8 % or better vs v1 (**stop the track if < 3 %**) |
| Night 5 | C: router KD (R0 → R2), bias frozen | all-in | ≤ 8 h | `-rC` | KL −2 % or better, selection change ≤ 5 % |
| Night 6 | D: on-policy personal round | all-in | 2 h generation + 6 h training | `-rD` | Personal KL −3 % or better, no public regression |
| Night 7 | MTP re-align (07 S5) + full eval gate | all-in | 8 h | promoted build | `evals.gate` PASS |
| Optional evening | Cloud teacher completions for D, if on-policy underperforms | cloud | 2–3 h | +20 M tokens | Only if D gained < 3 % |

The total is ≈ 7 nights per model. Do it for **one** model: whichever wins chapter 02's and 04's comparisons. Run
GLM's recovery only if it is the chosen All-in model.

---

## 5. Decision criteria & what to measure

**Primary metric: per-domain KL_{64+⊥} and top-1 agreement on the held-out `eval.jsonl`**, always against the
*unpruned* teacher's `ref/`. Never use the training targets.
- **Report KL reduction relative to the start build**, `ΔKL_rel = (KL_v1 − KL_r)/KL_v1`.
- **When the unpruned reference at the same bits exists, also report gap recovery**:
  `R = (KL_v1 − KL_r)/(KL_v1 − KL_unpruned@bits)`. This is Litwein's 33.6 % figure. It tells you how much of the
  *pruning* damage came back.
- **Statistics:** a paired bootstrap over samples (10K reps, *08 §4.8*). A stage counts as positive only if the 95 % CI
  of ΔKL excludes 0 in ≥ 3 of 4 domains and no domain is worse beyond its CI (the same rule as *04 §5*).

**Guards** (a stage fails if any guard fails, even if KL improved):

| Guard | Suite | Threshold vs the start build | Why it can break here |
|---|---|---|---|
| Code pass@1 (HumanEval+, MBPP+) + per-item flips | `evals.code` (+ McNemar) | within noise (paired), p > 0.05 for a regression | Over-fitting to the code-heavy training mix shows up as flips both ways |
| Tool-call validity + accuracy (BFCL categories) | `evals.tools` | validity not lower; accuracy within 2 pp | Tool-format drift (§6.3) |
| Personal replay: tool name + arguments, error rate | `evals.replay` (*08 §4.8*) | not lower | The personal round can over-fit to recent sessions |
| Long context: RULER 32K/64K/128K | `evals.longctx` | within 1 pp per length | Training at ≤ 2K can disturb the long-range path (§6.4) |
| Rare-entity copy probe, termination/loop rate | *04 §5* probes | ≥ 95 % of start; loops ≤ +1 pp | KL averages hide these (*reap §3*) |
| Math (MATH-500 subset, AIME) | `evals.math` | within noise | Forgetting of under-represented domains |
| Multilingual / general-floor KL | `evals.kl` domain `general` | not worse | Forgetting |
| MTP acceptance α (T = 0 and 0.6), decode tok/s | `evals.speed` / `drafter/eval.py` | α ≥ start − 1 pt; tok/s unchanged | Trunk change shifts h_t (§2.8) |
| Model files | diff tool | only scales/biases/router changed; same bytes | Silent format corruption |

**Sample sizes.** KL over thousands of positions per domain ranks stages, while the task suites are guards
(*04 §5*: detecting a 3 pp paired gap needs ≈ 700 items). The held-out KL set for band gating (64 records) is
*separate* from `eval.jsonl`, so that per-band decisions don't over-fit the final metric.

**Decision rule.**
1. Run stages in order and stop at the first stage that fails its exit criterion.
2. Promote the last passing build if it passes the full gate.
3. If stage B gains < 3 %, recovery is not worth its nights for this model. Spend the time on chapter 04's E-Q1/E-G1
   trade (more experts or more bits), which moves KL by tens of percent, not ten.

---

## 6. Failure modes

1. **Overfitting the training distribution.**
   - *Signature:* training KL falls while held-out KL rises; gains appear only in the dominant domain.
   - *Catch:* the per-band held-out gate, held-out data split by session, per-domain CIs.
   - *Prevent:* ≤ 1–2 epochs, ≥ 30 % general-floor data, small learning rate, early stop on held-out KL.
2. **Forgetting.** Domains outside the mix get worse: multilingual, arithmetic, vision (vision is dropped in our builds
   anyway).
   - *Catch:* the `general` KL domain and the math guard.
   - *Prevent:* forward KL on teacher targets, which preserves the whole distribution, plus the data floor.
     LoRA-style low capacity also forgets less (arXiv:2405.09673).
3. **Tool-format drift.** The model starts emitting tool calls in a format that is not its native template, or wraps
   JSON differently. *Cause:* any hard-label training on another model's outputs, or contexts rendered in the wrong
   template.
   - *Prevent:* targets are always the teacher's distribution, and all contexts are rendered by `calib_data.render` in
     the target's own template.
   - *Catch:* `evals.tools` validity and `evals.replay`.
4. **Long-context regression from short-context training.**
   - *Catch:* RULER at 32K–128K.
   - *Mitigate:* include training windows taken from deep inside long documents (positions 16K–64K, with the
     recurrent state carried in from a no-grad prefix using the fast kernel). This truncated-BPTT variant is a
     **[hypothesis]**, and it is cheap to try.
5. **Router collapse.** R2 concentrates load on a few survivors, KL looks fine, and rare domains die.
   - *Catch:* the selection-change rate and per-expert load entropy logged every 100 steps. Stop if normalized load
     entropy drops by more than 0.02 (0.92–0.99 is the baseline, *04 §1*).
6. **Truncated-target miscalibration.** Training on renormalized top-k makes the student over-confident and leak tail
   mass (arXiv:2503.16870).
   - *Prevent:* the tail-bucket loss of §2.2.
   - *Catch:* the student's mean `q_⊥` vs the teacher's `p_⊥` on held-out data. They should match within 10 %.
7. **Silent corruption at save time.** Training casts fp32 masters back to bf16. A wrong key path or a lost
   `quantization` config entry still loads fine and outputs garbage.
   - *Catch:* the diff tool (only intended tensors changed, same dtypes and shapes), plus a 1K-token KL smoke test
     against `ref/` before any long eval (*04 §6 item 2*).
8. **Memory blow-ups.** Two found here (§3.3): unsorted `gather_qmm` gradients, and the O(T) recurrent backward.
   `mx.set_memory_limit` did not stop either.
   - *Catch:* the self-bounding check in §4.1 and a tiny-fixture assertion that peak memory scales with one layer.
9. **Divergence** (cf. lapadatbogdan).
   - *Prevent:* gradient clipping, warmup, fp32 masters, lr sweeps on one band, gating every band. Never train more
     than one band plus the router at once.
10. **MTP mismatch.** The trunk improves but decode slows down because α fell.
    - *Catch:* α and tok/s in the gate.
    - *Fix:* stage-7 re-alignment. Never ship a recovered trunk with the old head without measuring α.
11. **Thermal throttling / shared machine.** Multi-hour all-in runs on a laptop.
    - *Prevent:* High Power mode, `caffeinate`, checkpoint every band so a reboot loses at most one band, and run only
      when the user isn't working (*08 §4.1*). The watchdog (*01 §4.6*) stops training before the system swaps.
12. **Privacy.** Personal top-k shards, on-policy generations and the trained weights all derive from your sessions.
    Keep them local (0600) and canary-probe the weights. Never upload them, including to the cloud teacher job.

---

## 7. Stretch ideas

### 7.1 A chunked delta-rule op with a custom VJP (the highest-value engineering item)

Write the chunkwise-parallel (WY / UT-transform) form of the gated delta rule, the formulation used by
flash-linear-attention, as an `mx.custom_function`:
- The forward stores states only at chunk boundaries (e.g. every 64 tokens).
- The backward recomputes inside each chunk.

What it buys:
- Memory drops from ~6–9 MiB/token/layer to ≈ 1/64 of that plus chunk temporaries.
- Time drops from a T-step Python loop to T/64 batched matmuls.

Expected effect: 3–5× faster end-to-end training, and T = 8–16K becomes trainable, which turns failure mode 4 from a
guard into an objective **[hypothesis]**. Verify against `gated_delta_ops` on the tiny fixtures (max-abs gradient
difference ≤ 1e-3 in fp32). It is also an upstream contribution candidate for mlx-vlm.

### 7.2 Cloud QAT with an exact MLX affine emulator

Port DWQ to PyTorch on 8×H100, with a fake-quant layer that reproduces MLX affine dequantization bit for bit:
`ŵ = s·q + β` in bf16, with the same group layout. Train scales/biases and routers on 50–100 M tokens with full
teacher logits on the GPUs (no truncation), then export the tensors back into the MLX checkpoint. Cost: ≈ 8–12 h ×
$16–32/h ≈ **$130–380** **[est]**. It is worth doing only if stage B shows a clear "more tokens → more gain" slope.

### 7.3 Importance-sampled sparse targets

Store 12–24 token ids sampled from the teacher's distribution, with their probabilities, instead of the top-64.
That gives unbiased KD at ≈ 1/3 the storage (arXiv:2503.16870). Compare it with top-64 + tail at an equal byte budget.

### 7.4 Ghost expert: distil pruned experts into one extra shared low-rank path

Add a small always-on low-rank MLP per layer: rank 64 is ≈ 0.8 M params per GLM layer and ≈ 35 M overall. Initialize it
by fitting the *mass the pruned experts carried* (teacher-trajectory residual of the pruned routing), then train it
end to end. It spends ≈ 0.05 GB to return some of the deleted capacity on exactly the tokens that lost their expert
**[hypothesis]**.

### 7.5 Nightly continual self-distillation

Each night, collect the day's traffic. The `serve/` request log stores token ids only when opted in. Score it with the
teacher and run a 50 K-token stage-D refresh on the accepted bands, promoted only through `evals.gate`. This is
recovery as a habit rather than a project. Watch for drift toward the most recent week, using a held-out set fixed
from the first month.

### 7.6 Joint recovery of trunk and drafter

Train the MTP head's connectors together with stage D. Add an acceptance-proxy loss, the KL between the head's
depth-1 prediction and the trunk's next distribution, weighted by λ, so the trunk is not pushed into regions the head
cannot follow. Evaluate the decode-speed × quality product rather than either one alone **[hypothesis]**.

### 7.7 Recovery as a pruning-depth probe

Run stage B on two adjacent K values (e.g. GLM 128 vs 144). If recovery closes most of the 128-vs-144 KL gap *and* the
rare-entity/termination guards pass for 128, the freed 8 GB buys context or bits (*04 §4.5*). Expect the guards to fail.
sh0wie's "unrepairable" result predicts it, and this is the experiment that would show otherwise.

---

## Sources

- Installed source **[src]**: `mlx_lm/quant/dwq.py`, `mlx_lm/tuner/{losses,trainer,lora,utils}.py`,
  `mlx_lm/LEARNED_QUANTS.md`; `mlx_vlm/trainer/{api,core,common/*,peft/lora_layers}.py`;
  `mlx_vlm/models/{gated_delta.py, switch_layers.py, glm5_next/language.py, qwen3_5/{language,speculative_verifier}.py,
  qwen4_exp/{language,qsa_kernel}.py}`; repo `evals/kl.py`, `docs/contracts/calib.md`, `calib/run.py`.
- Configs **[computed]**: `zai-org/GLM-5.3-Flash` and `Qwen/Qwen3.8-Flash-Next` `config.json` (HF cache; read only).
- Papers: PEQA [2305.14152](https://arxiv.org/abs/2305.14152) · EfficientQAT [2407.11062](https://arxiv.org/abs/2407.11062) ·
  GPTAQ [2504.02692](https://arxiv.org/abs/2504.02692) · Sparse Logit Sampling [2503.16870](https://arxiv.org/abs/2503.16870) ·
  GKD [2306.13649](https://arxiv.org/abs/2306.13649) · SDFT [2402.13669](https://arxiv.org/abs/2402.13669) ·
  LoRA [2106.09685](https://arxiv.org/abs/2106.09685) · QLoRA [2305.14314](https://arxiv.org/abs/2305.14314) ·
  QA-LoRA [2309.14717](https://arxiv.org/abs/2309.14717) · LoRA Learns Less and Forgets Less [2405.09673](https://arxiv.org/abs/2405.09673) ·
  Router KD [2603.02217](https://arxiv.org/abs/2603.02217) · Router sensitivity [2608.07890](https://arxiv.org/abs/2608.07890) ·
  Composable compression [2608.21693](https://arxiv.org/abs/2608.21693).
- Model cards / community **[self-reported]**: [Litwein REAP320-oQ3e-DWQ](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX);
  patrickbdevaney, lapadatbogdan, sh0wie via `docs/research/reap.md`.
- Serving recipes **[self-reported]**: [NVIDIA Dynamo GLM-5.3-Flash recipe](https://docs.nvidia.com/dynamo/recipes/glm-5-3-flash),
  [Baseten GLM-5.3-Flash](https://docs.baseten.co/examples/models/llm/glm-5.3-flash),
  [vLLM recipe Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next),
  [Yotta Labs Qwen3.8-Flash-Next requirements](https://www.yottalabs.ai/post/qwen-3-8-flash-next-hardware-requirements-gpu-memory-2026).
- GPU prices (Oct 2026, cross-check live): [deploybase.ai ranking](https://deploybase.ai/articles/cheapest-gpu-cloud-in-2026-provider-pricing-ranked),
  [GMI Cloud pricing blog](https://www.gmicloud.ai/en/blog/gpu-cloud-pricing-llm-inference-2026),
  [CloudZero GPU pricing](https://www.cloudzero.com/blog/cloud-gpu-pricing/),
  [Keenable H100 effective prices](https://select.keenable.ai/r/effective-nvidia-h100-cloud-prices/0DRsK7gE-5uMsmQ7IiUM1A).
- Measurements **[measured]**: scratchpad scripts `rec/gd_mem.py`, `rec/gqmm_grad.py`, `rec/gqmm_scale.py`
  (random tensors, GPU shared with the running server).
