# MLX low-bit quantization for huge MoE models (state as of 2026-10-20)

Scope: 2.5–4 bits/weight (bpw) for `qwen4_exp` (Qwen3.8-Flash-Next) and `glm5_next` (GLM-5.3-Flash) on an
M5 Max running MLX 0.32.4 / mlx-lm 0.32.2 / mlx-vlm 0.7.9. Tags: **[local]** = read from the installed package
or measured on this machine; **[self-reported]** = numbers from a model card or PR author, not independently checked.

---

## 1. What MLX 0.32.4 can store and run

Taken from `mx.quantize.__doc__` and the kernel names in `mlx/lib/mlx.metallib` **[local]**:

| mode   | bits                   | group sizes         | scale / bias                  | effective bpw (affine: bits + 32/g)          |
|--------|------------------------|---------------------|-------------------------------|----------------------------------------------|
| affine | 1, 2, 3, 4, 5, 6, 8    | 32, 64 (default), 128 | scale + bias in the weight dtype (bf16/fp16) | e.g. 2b g64 = 2.5, 3b g64 = 3.5, 3b g128 = 3.25, 4b g64 = 4.5, 4b g128 = 4.25 |
| mxfp4  | 4 (E2M1)               | 32 only             | e8m0 scale, no bias           | 4.25                                          |
| nvfp4  | 4 (E2M1)               | 16 only             | e4m3 scale (+ optional fp32 `global_scale`), no bias | 4.5                       |
| mxfp8  | 8 (E4M3)               | 32 only             | e8m0 scale                    | 8.25                                          |

- 1-bit affine was merged on 2026-10-09 ([mlx#3161](https://github.com/ml-explore/mlx/pull/3161)).
- `mx.qqmm` / `gather_qqmm` and `nn.quantize(..., quantize_input=True)` add activation quantization
  (W4A4/W8A8), but ~~only for `nvfp4` and `mxfp8`~~ only for the fp modes: `mxfp4` runs too on Metal in 0.32.4,
  and affine fails with a missing-kernel error **[local, verified]**.
- Granularity: `QuantizedSwitchLinear` holds **one (bits, group, mode) per stacked expert tensor**, so stock
  MLX can mix precision per (layer, projection), not per expert. GGUF has the same limit
  ([GSQ-RCO card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)). To give hot experts more
  bits you would have to split a layer's experts into two switch modules and remap indices, which needs custom code.
- The per-module overrides go in `config.json["quantization"][<path>]`, and `mlx_lm.utils.infer_quant_config`
  can recover the mode from the tensor shapes/dtypes **[local]**. That is how oQ, OptiQ and the Litwein baseline
  all ship ordinary mixed-precision checkpoints.

### Quantization tools that ship in mlx-lm 0.32.2 / mlx-vlm 0.7.9 [local]
| Tool | What it does | Can we use it on our two models? |
|---|---|---|
| `mlx_lm.convert --quant-predicate mixed_{2_6,3_4,3_6,4_6}` | Q4_K_M-style rule: v_proj/down_proj in the first and last eighths (and every 3rd layer between) plus lm_head get the high bits | Only as a baseline. It knows nothing about routers, hyper-connections or DeltaNet. |
| `mlx_lm.dynamic_quant` | Gradient × (q_low − q_high) sensitivity per layer, then a threshold binary search for a target bpw; only 2 levels | Loads the whole model plus a deep copy, so it does not fit the memory budget. |
| `mlx_lm.dwq` | Distills the student's affine **scales and biases** (only for bits < 8) against teacher logits. Defaults: KL at temperature 2, top-1024 logit targets, lr 1e-6, Adam | The algorithm is fine, but it trains the whole resident model. See §3. **[verified]** It also cannot load either model: `mlx_lm.utils._get_classes` raises "Model type qwen4_exp/glm5_next not supported". The same applies to every `mlx_lm.quant` tool. |
| `mlx_lm.gptq` | GPTQ, but `assert bits in {2, 4, 8}` (**no 3-bit**). For `SwitchLinear` it builds **one Hessian from all tokens, shared by every expert**, and the full model must be loaded | No. **[verified]** In addition, it matches `type(l) in {nn.Linear, mlx_lm SwitchLinear}`, and our models use `mlx_vlm.models.switch_layers.SwitchLinear`, so it would not touch the experts at all. |
| `mlx_lm.awq` | Block-streamed AWQ scale and clip search, MoE-aware | Only for llama, mistral/ministral3/mistral3, llava, qwen2/3, gemma3 and deepseek_v2. No `qwen4_exp` or `glm5_next` config. |
| `mlx_vlm.convert --quant-method awq` | AWQ that folds scales into the norms, **dense `nn.Linear` attention/MLP only** | ~~Skips SwitchGLU experts~~ **[verified]** It does nothing on either model. `_awq_layer` returns 0 unless the layer has `self_attn.{q,k,v,o}_proj` **and** a dense `mlp.{gate,up,down}_proj`. Every Qwen layer is MoE, and GLM's 3 dense MLPs use a fused `gate_up_proj`, so no layer qualifies. |
| `qwen4_exp.quant_predicate` | `mlp.gate` and `shared_expert_gate` at 8-bit g64; PLE shards use g32 (row width 160) or nvfp4 | Use as the floor of our predicate. |
| `glm5_next.quant/cast_predicate` | Never quantizes `attn_hc`, `ffn_hc`, `index_kpool_compress`; keeps `A_log`, `dt_bias`, `e_score_correction_bias` in fp32 | Use as the floor of our predicate. |
| `qwen4_exp.ple_storage` | Memory-maps the 51.2B-param n-gram table and materializes only the rows looked up | Supports only **affine 4b g32** or **nvfp4 g16**. |
| `mlx_vlm.split_mtp` (`qwen4_exp_mtp`, `glm5_next_mtp`) | Writes the MTP head as a separate drafter directory, optionally quantized | `qwen4_exp.sanitize` **drops `mtp.*`** from the main checkpoint, so the MTP head must be written through this path. |

---

## 2. What the community quant names actually are

| Name | What it is | Primary source |
|---|---|---|
| **oQ / oQe** (`oQ3e`, `oQ4e`, `oQ2.5`, …) | The quantizer in **oMLX** (jundot/omlx, a Mac menu-bar server). It streams the model tensor by tensor and measures per-layer sensitivity (128×256 tokens; Qwen4-Exp is calibrated **one layer at a time**). It then spends the budget on the most sensitive tensors and writes standard MLX affine weights. The **e** variant adds an imatrix, i.e. per-input-channel activation energy (128–1024 seqs × 512 tokens; for MoE it keeps sampling until every expert has enough tokens). It does an imatrix-weighted clipping search per group, then (since 2026-10-09) a **weighted least-squares refit of scale and bias** with the scale and bias rounded to their storage dtype. Default protection: routers at full precision, shared-expert gates 8-bit, embeddings and lm_head 8-bit when the budget allows, SSM/linear-attention state params and norms unquantized, vision unquantized, MTP connector layers full precision with the rest ≥ 4-bit. Routed experts stay at base bits; oQ3.5 gives down_proj +1 bit; oQ2.5/2.7 lift whole expert layers to 3-bit. The number is the base bit width, not the average: oQ3e is about 3.5–3.7 bpw and oQ4e about 4.6–4.7. | [docs/oQ_Quantization.md](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md), [PR #4385](https://github.com/jundot/omlx/pull/4385) |
| **OptiQ** (`mlx-optiq`) | Per-layer **KL sensitivity**, one layer simulate-quantized at a time against a bf16 reference (or a uniform 4-bit reference when bf16 does not fit). A greedy "knapsack" upgrades the layer with the best ΔKL per extra bit, usually with candidates {4, 8} or {2, 4}. Protected: lm_head, embed_tokens, the first and last attention blocks, routers, shared experts. Fused experts get one entry per (block, projection). Calibration is 40 samples × 6 domains. The docs describe no rounding improvement beyond RTN. The Qwen3.8-Flash-Next OptiQ-2bit build is SSD-streamed, ~3.8 tok/s on an M3 Max, has no MTP head and publishes no quality score. | [docs/sensitivity](https://mlx-optiq.com/docs/sensitivity), [OptiQ-2bit card](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit) |
| **GSQ-RCO** | IST-DASLab. **GSQ** ([arXiv 2604.18556](https://arxiv.org/abs/2604.18556)) starts from GPTQ and refines each weight's grid choice and the group scales with a Gumbel-Softmax relaxation, layer by layer, on a symmetric scalar grid. **RCO** ([arXiv 2605.00649](https://arxiv.org/abs/2605.00649)) assigns one of K formats per tensor under an *exact* size budget: Riemannian optimization on the budget manifold plus Gumbel-STE with a knapsack DP, trained on end-to-end KL. Its released code also does RCO **expert pruning**. Released as GGUF; the only MLX port ([uqer1244/mlx-GSQ-RCO](https://huggingface.co/uqer1244/Qwen3.8-27B-GSQ-RCO-MLX)) needs a custom IQ-kernel runtime and is slow (22 tok/s prefill on M3 Pro). GSQ's tooling is CUDA/H100-only. | [ISTA card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), [GSQ repo](https://github.com/IST-DASLab/GSQ), [RCO repo](https://github.com/IST-DASLab/RCO) |
| **DWQ** | mlx-lm's distilled weight quantization (see §1). In the Litwein baseline it was customised: a full-network KL against bf16 top-20 logprobs, trained in "layer bands" with gradient checkpointing and early stopping, and each band kept only if it passes a held-out gate. | [Litwein card](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX) |
| **M4Q** | Named on the Litwein card as "Hessian-guided M4Q initialization" for 3-bit experts. I found no paper, repo or definition for it, so treat it as an unverified private method. | n/a |
| **mixed / DQ3_K_M / mix6** | Hand-written llama.cpp-style predicates, such as down_proj +1 bit or 6-bit attention. | various cards |

The schemes that measurably beat plain RTN on MLX at the same size all combine three things:
(a) a **sensitivity-driven bit allocation** (KL or relative output error),
(b) **better rounding inside the affine format** (imatrix-weighted clip search plus a least-squares scale/bias refit, or GPTQ/GSQ codes), and
(c) for the last few percent, **logit distillation of scales and biases** (DWQ).
None of them change the kernel format, so all of them run at stock MLX speed.

---

## 3. Evidence on what helps (MLX unless noted)

**Rounding (same bits, same allocation):**
- oQe least-squares refit vs the previous oQe: ~~text KL drops 12–18% on Qwen3.5-9B, Qwen3.6-35B-A3B and Qwen3.8-27B (paired bootstrap, 16/16 windows better)~~.
  **[re-checked]** In the *isolated* A/B (same imatrix and same allocation), text KL fell by 11.6% / 15.3% on Qwen3.5-9B oQ3e / oQ4e and by 12.2% / 13.1% on Qwen3.6-35B-A3B. The Qwen3.8-27B −13.8…−19.1% figures (16/16 windows) compare two *releases*, and the PR states that "other oQ changes also landed" between them.
  Qwen3.8-Flash-Next oQ4e mean KL went from 0.0877 to **0.0776** (−11.6%) on code and from 0.1909 to **0.1713** (−10.2%) on the "Novel (English)" corpus (16×2048 tokens). That is also a comparison between releases (2026-08-27 vs 2026-10-11) and cannot be attributed to the refit alone. [self-reported] [PR #4385](https://github.com/jundot/omlx/pull/4385), [card](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp)
- oQ4e vs mlx-lm uniform 4-bit, accuracy on 1,600 questions: gemma-4-26B-A4B 81.62 vs 79.88; Qwen3.5-9B 72.38 vs 70.12. [self-reported] [oQ docs](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md)
- Qwen3.8-Flash-Next, KL vs bf16 over 4×1024 tokens of code, prose and agent chat. Uniform RTN 4b g64: **0.288**; RTN 4b g32: **0.232**; a mixed build with best-of(min-max, symmetric, LSQ refit) rounding: **0.129** (top-1 86.8%, PPL 4.109 vs 4.054). bf16 weights with bf16 activations already give 0.024, which is the noise floor. [self-reported] [nace-ai card](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit)
  **[re-checked] This is not a same-size comparison and does not isolate rounding.** The mixed build is 106 GB on disk (~73 GB on the GPU) and reads 4.13 GB of weights per decoded token, against 3.79 GB for RTN g64. Its 0.232→0.129 gain combines bit allocation with rounding. The card also lists the bf16 PPL as 4.054 in its header and 4.021 in the table.

**Which tensors need high precision (Qwen3.8-Flash-Next, wikitext-2, 145×2048 tokens, paired bootstrap)** [self-reported] [pipenetwork card](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit):

| everything 4-bit g64 except … at 8-bit | PPL (bf16 4.4708) | Δ |
|---|---|---|
| nothing (uniform 4-bit) | 5.3914 | +20.6% |
| hyper-connection read gates `input_mix_weight_down/up` (0.6B params) | 4.9744 | +11.3% |
| attention + DeltaNet + shared experts + PLE projections (~2.3B) | 4.8969 | +9.5% |
| embed_tokens + lm_head (1.3B) | 5.2843 | +18.2% |
| all three groups (~4.2B = 3% of params) | 4.5286 | **+1.3%** |
| 6-bit everything / 8-bit everything | 4.4767 / 4.4749 | ≈ bf16 |

**[re-checked]** The table matches the card: WikiText-2 test, 145×2048 windows, paired bootstrap with 20k resamples. The card gives no per-row CIs for the ablation rows, and the ablation builds are not published. By a lazy count over the real config (§Verification), the three groups hold about 4.9B params, not 4.2B. That is 2.8% of all 177B params, or 3.9% excluding the n-gram table.

Per parameter, the non-expert weights are roughly 20× more sensitive than the routed experts. The vendors' own
FP8 releases agree **[primary]**:
- [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json) converts only the routed experts and the n-gram table. Everything else stays BF16: attention, all `linear_attn.*` (in_proj_qkv/z/a/b, out_proj, conv1d), shared experts and shared-expert gate, router, every hyper-connection weight, PLE key/value/conv, QSA indexer, lm_head, embeddings, and MTP `fc_embedding/fc_hidden` plus the MTP layer's non-expert weights.
- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) (FP8 128×128 blocks) keeps BF16 for every KDA linear-attention projection (q/k/v/o, `fused_qkvbfg_a_proj`, f/g/b projections, conv1d, A_log, dt_bias), the DSA indexer (wq_b, wk, weights_proj, kpool compress), MLA `kv_b_proj`, router plus `e_score_correction_bias`, all `hc_*` (mHC), MTP `eh_proj`, embeddings, lm_head and vision.
  **[verified from both configs]** Note what the GLM vendor *does* convert to FP8: shared experts, the 3 dense MLPs, and the MLA `q_a/q_b/kv_a/o` projections of the 11 DSA layers. A ≥ 6-bit floor for those is therefore at or below vendor precision, not above it.

**Bits vs size for the two models** (different corpora and methods, so compare only within a row group):

| Model / build | bpw (weights) | size | metric | source |
|---|---|---|---|---|
| Qwen3.8-FN GSQ-RCO Q2_0 (GGUF) | 2.40 | 66.4 GB (incl. 28.8 GB n-gram) | task avg 89.07 vs bf16 93.12 (AIME25 96.7, GPQA-D 89.4, LCB 81.1) | [self-reported] [ISTA](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| Qwen3.8-FN GSQ-RCO IQ2_XS | 2.50 | 68.0 GB | 89.16 (LCB 83.4) | same |
| Qwen3.8-FN GSQ-RCO IQ3_XXS | 3.00 | 75.8 GB | **92.57** (AIME25 100, GPQA-D 91.4, LCB 86.3) | same |
| Qwen3.8-FN GSQ-RCO IQ3_S | 3.50 | 83.6 GB | 93.26 (≈ bf16) | same |
| Qwen3.8-FN REAP-320 oQ3e + 3-round DWQ (MLX, **baseline**) | 3.76 excl. n-gram | 70.8 GB | KL20 0.1035, CE-gap 0.090 on held-out calibration domains. DWQ recovered 33.6% of the pruning gap and cut KL20 by 11.2% vs the pruned initializer | [self-reported] [Litwein](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX) |
| Qwen3.8-FN REAP-288 / 320 / 384 / 512, 4-bit MLX | 4.5 | 68 / 72 / 80 / 98 GB | HumanEval 91.5 / 90.9 / 92.1 / 93.9 | [self-reported] [sh0wie](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) |
| GLM-5.3-Flash MLX 6/4/3/2-bit (mlx-lm-style, linear-attn/indexer/mHC/embed/head in BF16) | ~~~6.5/4.5/~4/~3~~ base 6g64/4g64/**3g32**/2g32; overall ≈7.5/5.2/4.7/3.7 by size | 296/204/184/145 GB (+ a 102 GB "2bit-lite" 2g128, KLD 0.346) | KLD vs FP8 0.006 / 0.013 / **0.042** / 0.165; PPL +0.2 / +3.0 / +10.0 / +57% | [self-reported] [orcarouter](https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX) |
| GLM-5.3-Flash GGUF IQ4_XS / IQ3_S / IQ2_S (~~Q8_0~~ Q8_0 / Q6_K / Q6_K attention) | 3.97 / 3.11 / 2.83 | 148 / 116 / 106 GiB | KLD 0.117 / **0.283** / 0.375 (reference not stated; PPL is vs BF16) | [self-reported] [AesSedai](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) |
| GLM-5.3-Flash GSQ-RCO 3.5 / 3.0-bit (GGUF) | 3.5 / 3.0 | 137 / 117 GB | approx. KL 0.071 / 0.142; MMLU-Pro −1.4 / −1.95 pp vs Q8_0 | [self-reported] [pfeifferj](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) |
| GLM-5.3-Flash oQ3e (MLX) | 3.64 | 146 GB | no KL published | [Jundot](https://huggingface.co/Jundot/GLM-5.3-Flash-oQ3e-mtp) |

What this means for us: GLM-5.3-Flash unpruned is already 117–146 GB at 3–3.6 bpw. To reach ≤ 68 GB it would
need about 50–60% of its experts pruned **and** roughly 3 bpw, a regime where even the unpruned model is at KLD ≈ 0.14–0.28.
~~For Qwen, ~2.4–3.0 bpw with a good allocator keeps 96–99% of task accuracy without any pruning.~~
**[corrected]** For Qwen, only the 2.40/2.50 bpw GSQ-RCO builds fit 68 GB (95.7% task retention). The 3.0 bpw build (99.4%) is 75.8 GB.
Those builds use GSQ codes in IQ/Q2_0 GGUF formats with a 28.8 GB IQ4_NL n-gram table, and MLX affine cannot reproduce them at equal bpw.
In MLX the fixed part is about 34–37 GB: n-gram table 28.8 (nvfp4) or 32.0 (affine), about 4.6 GB of floored non-expert weights, and about 1 GB of MTP.
That leaves about 31–34 GB for 120.8B unpruned expert params, i.e. **≈2.0–2.2 bpw**, below 2b g64 (2.5).
**An MLX Qwen build within budget therefore needs REAP pruning**: about 320 experts at 3b g64 (3.5 bpw) or about 288 at ≈3.9 bpw.

**FP4 vs affine:** on PTQ, MXFP4's power-of-two scales cost accuracy. NVFP4 is close to good INT4, and FP4 is
"not an automatic upgrade over INT4" ([Egiazarian et al., ICLR 2026](https://arxiv.org/abs/2509.23202)).
`mlx_lm.dwq` only trains affine scales and biases. So affine is the right base format for DWQ-refined weights.
**[re-checked]** The paper's comparisons are mostly W4A4 against symmetric INT4 g32. Its finding is "NVFP4 and INT4 provide similar quality [on average]", with NVFP4 ranked first, so it does **not** show that NVFP4 is worse than INT4.
Local RTN weight-MSE check on 120 real bf16 projection matrices (EmbeddingGemma):

| | affine 4b g128 | affine 4b g64 | affine 4b g32 | mxfp4 | nvfp4 | mxfp8 | affine 8b g64 |
|---|---|---|---|---|---|---|---|
| bpw | 4.25 | 4.5 | 5.0 | 4.25 | 4.5 | 8.25 | 8.5 |
| relative weight MSE | 9.6e-3 | 7.7e-3 | 6.0e-3 | 1.39e-2 | 8.8e-3 | 7.3e-4 | 4.9e-5 |

- mxfp4 is clearly worse than affine at equal bpw, and mxfp8 is about 15× worse than affine 8b.
- nvfp4 is only about 14% worse than affine 4b g64, and on heavy-tailed synthetic weights (Student-t, df 3–5) it *beats* affine 4b g64.

The case for affine therefore rests on DWQ/LSQ tunability and allocation granularity, not on a large RTN quality gap against nvfp4.

**DWQ memory:** for a ~68 GB student, the trainable affine scales and biases (0.5 bpw at g64 → about 4.5–5 GB in bf16) need fp32
master weights, grads and Adam m/v. That is about 35–40 GB on top of the weights, which goes past the 80 GB process cap.
**[re-checked]** The 4.5–5 GB figure holds only for the ~36 GB *non-n-gram* part at g64; it is about 75B REAP-320 expert weights × 0.5 bit.
If the 32 GB n-gram table were loaded as an affine-g32 module, `dwq` would also unfreeze its 6.4 GB of bf16 scales/biases, adding about 51 GB of state.
The table is safe from this only when memory-mapped through `ple_storage`, which uses numpy memmaps rather than mx parameters.
Even then, about 36–40 GB resident plus about 40 GB of state (fp32 master + fp32 grad + Adam m/v = 8× the bf16 size) leaves no room under 80 GB.
Two ways to keep it inside the cap: train one **band of layers** at a time, as Litwein did, or do **layer-local** distillation on
cached hidden states, which fits our `calib/` streaming. The mlx-lm tips also apply: DWQ helps 2–4-bit far more than ≥ 6-bit;
g32 doubles the tunable parameters and "can work much better"; distill from an 8-bit teacher if bf16 does not fit.

---

## 4. M5 neural accelerators (NAX) and kernel speed

**Do quantized matmuls use the GPU neural accelerators?** Yes for prefill and other multi-row matmuls. No for batch-1 decode.
- The installed metallib has NAX variants of `affine_qmm_t_nax`, `affine_qmm_n_nax`, `affine_gather_qmm_t_nax`
  and `affine_gather_qmm_rhs_nax_{nt,nn}` for **every affine bits {1,2,3,4,5,6,8} × group {32,64,128}**. It also has NAX
  `gather_qmm_rhs_nax` / `qmm_t_nax` for nvfp4 (g16), mxfp4 (g32) and mxfp8 (g32) **[local]**.
  `gather_qmm_rhs` is the sorted-expert MoE prefill path that `SwitchGLU` uses once `indices.size >= 64`.
- Decode kernels (`qmv`, `qmv_fast`, `qmv_quad`, `qmv_wide`, `gather_qmv`) have **no NAX variant**. Batch-1 decode is
  bandwidth or latency bound, which matches Apple's description: TTFT improves ~~3.5–4×~~ 3.33–4.06× vs M4, decode only 19–27%
  ([Apple ML research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). **[re-checked]** Those figures compare a 24 GB MacBook Pro M5 (153 GB/s)
  with an M4 (120 GB/s), not M5 Max. The NAX kernels' presence is verified in the metallib; that prefill actually dispatches them is inferred, not traced.
- Open upstream work (not in 0.32.4):
  - A 128×64×128 NAX qmm tile, +4–10% end-to-end prefill ([mlx#4692](https://github.com/ml-explore/mlx/issues/4692)).
  - A split-K NAX route for M = 8…hundreds of rows, up to 2× on short inputs ([mlx#4682](https://github.com/ml-explore/mlx/pull/4682)).
    **[re-checked]** It covers dense transposed affine `quantized_matmul` only. The PR says "Gather kernels and fp modes are unchanged" and that dispatch is unchanged for M < 8, so it speeds up neither the routed experts nor a ≤ 7-token MTP verify.
  - An opt-in int8 staging of affine qmm (`MLX_QMM_INT8`), prefill 1.4–1.5× at a quality cost of +0.3…+2% PPL ([mlx#4722](https://github.com/ml-explore/mlx/pull/4722)).
    It is dense-only ("gather_qmm untouched").
  - A `qmv_nax` for batch 6–16, 1.7–2.1× on fp4 modes, ~~relevant to MTP verification and batching~~ ([mlx#4712](https://github.com/ml-explore/mlx/pull/4712)).
    **[re-checked]** It covers fp modes only (nvfp4/mxfp4/mxfp8), so it does **not** apply to the affine-only build recommended here.
  - ~~Merged on 2026-10-14 (likely not in the 0.32.4 wheel)~~ **[corrected]** Merged on 2026-10-14 and **included in v0.32.4**, which is 4 commits ahead of the merge commit and was tagged 2026-10-14 23:37: non-transposed NAX `gather_qmm_rhs` ([mlx#4684](https://github.com/ml-explore/mlx/pull/4684)).
    This is a backward/training path (QLoRA steps 1.13–1.27× faster), so it helps band-wise DWQ, not inference.
- On non-NAX chips, `quantized_matmul` is 14–41% slower than dequantize + bf16 matmul above 512 rows
  ([mlx#4621](https://github.com/ml-explore/mlx/issues/4621), M2 Max). That issue is not about M5.

**Microbenchmark on this M5 Max [local]**: `docs/research/quant_bench.py`. It runs real `mlx_lm` `SwitchGLU` layers
with random weights at the expert shapes of each model. Qwen: H = 2560, I = 640, top-10 of 64 experts, 4 layers. GLM: H = 4096,
I = 2048, top-8 of 24 experts, 3 layers. Peak memory was 6 GB. The machine was shared with active downloads and the user's
work, so run-to-run spread was up to 2×. The table shows best-of-runs, since contention only slows a run down. Re-run it on an idle machine before relying on gaps under 20%.

| config (bpw) | GLM decode µs/layer (eff. GB/s) | GLM prefill tok/s/layer @2048 | Qwen decode µs/layer | Qwen prefill tok/s/layer |
|---|---|---|---|---|
| bf16 (16) | 729 (552) | 65.8K | 212 | 327K |
| affine 2b g64 (2.5) | 201 (314) | 70.7K | 84 | 307K |
| affine 3b g64 (3.5) | 228 (387) | 66.0K | 120 | 279K |
| affine 3b g128 (3.25) | 222 (368) | 68.8K | 112 | 308K |
| affine 4b g64 (4.5) | 234 (485) | 65.7K | 107 | 310K |
| affine 4b g128 (4.25) | 292 (366) | 68.1K | 102 | 277K |
| affine 8b g64 (8.5) | 504 (424) | 61.5K | 164 | 239K |
| mxfp4 (4.25) | 271 (395) | 67.9K | 126 | 331K |
| nvfp4 (4.5) | 297 (381) | 67.1K | 131 | 281K |

Reading:
- **Prefill:** every quantized mode runs at roughly bf16 speed (NAX dequant-in-tile). Bits and mode change MoE
  prefill by about ±10%, which is within the noise here. The 3-bit and odd group-32 configs were slowest in single runs.
- **Decode, GLM-size experts (2048×4096):** decode is bandwidth bound. 4b g64 reached ~485 GB/s of the 614 GB/s peak;
  ~~3-bit and 2-bit unpack less efficiently (~310–390 GB/s). So **3-bit and 4-bit decode take about the same time per layer**
  (≈ 230 µs). Below 4 bits you gain memory, not decode speed.~~
  **[refuted by re-run]** The 3b g64 row above was a contended outlier. Two re-runs of this script and a third harness on mlx-vlm's own `SwitchGLU` (min of 7 trials) agree. Per layer, 3b g64 takes 192–200 µs vs 226–247 µs for 4b g64 (**~15–20% faster**), and 2b g64 takes 144–153 µs (~35% faster).
  The effective bandwidth is 410–460 GB/s at 2–3 bits vs 460–500 GB/s at 4 bits, so decode time scales nearly with bytes. Below 4 bits you gain **both** memory and GLM-shape decode speed.
- **Decode, Qwen-size experts (640×2560):** decode is latency bound at about 80–130 µs per layer for every mode, because each
  expert matrix is tiny. Expert bit width barely moves Qwen decode speed. Bytes per token are dominated by the ~~~3.6B~~ ~4.2B
  non-expert matmul weights that every token reads: DeltaNet 2.09B, attention 0.62B, hyper-connections 0.64B, lm_head 0.64B and shared expert 0.24B, by lazy count.
  For speed, MTP drafting and batch-verify kernels matter more. **[re-run]** Qwen-shape decode measured 75–117 µs/layer for 2–5 bits and all modes, which confirms it is latency bound.
- **MTP verify (T = 2…8) [new, local]:** mlx-vlm's `SwitchGLU` routes 1 < T ≤ 8 through fused `fast_ops` kernels, but only for **affine 4/5-bit**. Other bits fall back to `gather_qmm`.
  Even so, at T = 4, 3b g64 is not slower than 4b g64: 192–200 vs 215–219 µs/layer at the Qwen shape and 543–581 vs 637–740 at the GLM shape. The fused path is no reason to prefer 4-bit experts.
- **mxfp4 / nvfp4** show no decode advantage over affine 4b g64. ~~and give lower PTQ quality~~ mxfp4 has clearly lower RTN quality; nvfp4 is roughly on par (§3).
  On re-run, mxfp4 decoded at about the same speed as affine 4b g64 (230–257 vs 226–247 µs at the GLM shape), not slower as in the table above. They are worth it only for
  the PLE table (below) or for future W4A4 `qqmm` prefill.
- **Group size:** g64 is the speed sweet spot for both shapes. Use g128 to save 0.25 bpw on robust tensors and g32 to buy
  quality on sensitive ones. Pick by the allocator, not for speed.

---

## Recommendations for this project

1. **Use affine mode for the whole backbone.** Let the allocator choose per (layer, projection) from
   {2b g64 (2.5), 3b g128 (3.25), 3b g64 (3.5), 3b g32 (4.0), 4b g128 (4.25), 4b g64 (4.5), 5b g64 (5.5), 6b g64 (6.5), 8b g64 (8.5)}.
   Add **g128** to the `calib/` `sens[(bits, group)]` grid; it is currently {32, 64}.
   Do not use mxfp4/nvfp4/mxfp8 for experts: they give no speed gain on M5 decode, ~~lower PTQ quality~~ mxfp4/mxfp8 have clearly lower RTN quality while nvfp4 is roughly on par (§3), and DWQ and LSQ refit cannot tune them.
   **[verified]** The format menu is exactly what `mx.quantize` accepts in 0.32.4: affine bits {1,2,3,4,5,6,8} (not 7) × g{32,64,128}, mxfp4 g32, nvfp4 g16, mxfp8 g32. `calib/sens.py` currently has `GROUPS = (32, 64)`.
2. **Hard floors in `build/`.** These go beyond what the allocator chooses and follow the vendors' FP8 configs and the pipenetwork ablation.
   - *Never quantized (bf16/fp32):* routers (`mlp.gate`, `router`), `e_score_correction_bias`, `A_log`, `dt_bias`,
     conv1d, norms, GLM `attn_hc`/`ffn_hc`/`hc_*`/`index_kpool_compress*`, vision tower.
   - *≥ 8-bit g64:* `shared_expert_gate`, Qwen hyper-connection `input_mix_weight_*` and `block_inject_weight`,
     the QSA and DSA indexer projections, embed_tokens, lm_head, MTP `fc_*`/`eh_proj`, PLE key/value projections.
   - *≥ 6-bit, allocator may raise:* full-attention q/k/v/o, MLA projections, DeltaNet/KDA projections, shared experts,
     GLM's 3 dense MLP layers. All of these together are about 3% of parameters; the bulk of the budget goes to routed experts.
3. **Rounding inside the affine grid: copy oQe.** Weight the per-group clip search by the `calib/` input-channel
   magnitudes (imatrix). Then refit (s, b) by weighted least squares, alternating with rounding, with scale and bias
   rounded to bf16 when the error is evaluated. Keep the refit only if the error drops. This is cheap, runs layer-local,
   and is self-reported at ~~−12…18% KL~~ −12…15% text KL in the isolated A/B on 9B/35B models (§3). Use per-expert imatrices. For experts with too few routed calibration tokens, fall back to the
   layer's pooled imatrix. Skip mlx-lm GPTQ: it has no 3-bit and uses a shared Hessian across experts, and it cannot load these models or match mlx-vlm's `SwitchLinear` anyway. Skip AWQ folding:
   ~~it does not support SwitchGLU or hyper-connections~~ mlx-vlm AWQ is a no-op on both models, and mlx-lm AWQ has no config for them (§1).
4. **Allocation objective.** Measure the per-tensor relative output error at all (bits, group) points, which `calib/`
   already does. Solve a multiple-choice knapsack under the byte budget, which is exact via DP over the ~1–2k (tensor, format) items.
   Then **verify with end-to-end KL** on `eval.jsonl`, and do 1–2 rounds of swapping the worst-ranked items, which is a cheap
   approximation of RCO. Treat the REAP prune ratio as a variable of the same search. The data points are 512 experts at ≈2.4–2.5 bpw
   (GSQ-RCO: ~~95.6% task average~~ 95.7% retention of the bf16 task average at 66–68 GB, in GGUF formats not available in MLX), REAP-320 with 3-bit experts (the baseline), and 288 with 4-bit (HumanEval −2.4 pts).
   **[caveats added]**
   - sh0wie's HumanEval numbers come from one run of 164 problems, so 1 problem = 0.6 pt; 288 (91.5) scoring above 320 (90.9) shows the noise.
   - sh0wie also reports that aggregate KL "suggests pruning deeper", yet sampled rare-token reliability collapses below 288 experts (256 experts: 1/10 vs 9/10 intact rare names), and that this is "unrepairable by giving the experts more bits".
   - End-to-end KL alone must therefore not pick the prune ratio. Add a sampled rare-token / long-tail check as a hard constraint.
   - The knapsack has about 0.7–1k tensors × ~9 formats, i.e. ~6–9k items, which a DP over a discretized byte budget still handles easily.
5. **DWQ as the final pass, band-wise.** Distill only affine scales and biases with KL at T = 2 against the bf16
   reference top-k (≥ 64 logprobs plus residual mass). Train one band of about 4 layers at a time, keeping the rest frozen, which keeps
   optimizer state under ~4 GB. Use gradient checkpointing and accept a band only if held-out KL improves.
   Expected gain is about 10–15% KL at 3 bits [self-reported, Litwein].
6. **Qwen n-gram/PLE table (51.2B params):** it must be **affine 4b g32 (5.0 bpw, 32 GB) or nvfp4 g16 (4.5 bpw, 28.8 GB)**
   to use `ple_storage` mmap, so it stays off the resident set. nvfp4 saves 3.2 GB of the 68 GB disk budget. Validate it with KL
   before adopting, because no public quality comparison of the two exists. GSQ-RCO uses IQ4_NL (4.5 bpw) for this table.
   **[new caveat, local]** `ple_storage`'s nvfp4 layout stores only `weight` + U8 `scales`, with **no `global_scale`**, and dequantizes without one.
   - At its true magnitude the table is small: std ≈ 0.0074, measured on 20k rows of the Litwein table × its stored `weight_scale` 1.99e-4. At that size the e4m3 group scales fall into the subnormal range.
   - nvfp4 relative MSE is then 1.65e-2, vs 9.1e-3 with the table pre-scaled by 1/weight_scale. Synthetic weights at std 0.001 underflow completely.
   - So if nvfp4 is used, quantize the table pre-scaled (e.g. ×2^12) and fold the inverse into `ple.value_proj`. The key path is RMS-normalized by `norm_key` and needs no change.
   - Note also that `qwen4_exp/fp8.py` *drops* `ngram_embedding.weight_scale` whenever `shards.*` keys exist, so any scale must be folded at build time.
7. **Model choice implication.** Qwen3.8-Flash-Next fits the 68 GB budget with resident weights of about 36–40 GB:
   ≈2.5–3.5 bpw experts (~~pruned or not~~ **pruned**: unpruned would need ≈2.0–2.2 bpw experts in MLX, see §3), ~4.5 GB of floored (6/8-bit) non-expert weights, mmapped PLE and ~~~1.5 GB~~ ~0.7–1.4 GB MTP (≈2.6B params; 0.7 GB with REAP-320 experts at 3.5 bpw).
   GLM-5.3-Flash needs heavy pruning plus about 3 bpw, a regime with no good published data point. Spend the first
   build/eval cycles on Qwen.
8. **MTP:** quantize the MTP experts like the backbone and keep the MTP connectors at ≥ 8-bit. Write the head with
   `mlx_vlm.split_mtp` or an equivalent, because `qwen4_exp.sanitize` strips `mtp.*`. Prune its experts to the same REAP keep set.
   **[caveat, local]** Stock `split_mtp --q-bits` quantizes *every* 2-D weight uniformly except `mlp.gate`. That includes `fc_embedding/fc_hidden`, the hyper-connection weights and `shared_expert_gate`, which breaks the ≥ 8-bit connector floor.
   `split_mtp` also does not prune. So write the drafter unquantized (or call the splitter's `rename`/`depth` helpers) and quantize and prune it with the `build/` predicate and keep set.
9. **Speed evals:** expect identical prefill across bit choices. Expect Qwen decode to be latency bound and GLM decode
   bandwidth bound, so for GLM, 3-bit experts decode about 15–20% faster than 4-bit. Re-run `quant_bench.py` and `evals.speed` on an idle machine. Watch mlx#4682, #4712 and #4722; ~~they are
   prefill/verify wins that need no re-quantization~~ for an affine MoE build only #4682 and #4722 apply, and only to dense projections. #4712 is fp-mode only (§4).

### Dependencies
No new dependencies are needed for any of the above. Everything runs on the installed mlx 0.32.4, mlx-lm 0.32.2 and mlx-vlm 0.7.9.

---

## Verification (adversarial re-check, 2026-10-20)

Method:
- Re-read every primary source: `gh api` for the MLX and oMLX PRs, HF cards, arXiv, and the vendor `config.json` files, including `Qwen/Qwen3.8-Flash-Next-FP8`.
- Read the installed mlx 0.32.4, mlx-lm 0.32.2 and mlx-vlm 0.7.9 source and the metallib symbol table.
- Ran small experiments, all with peak memory under 5 GB:
  - re-ran `quant_bench.py`;
  - ran `quant_verify_bench.py` (mlx-vlm `SwitchGLU`, decode T = 1 and verify T = 4);
  - RTN error per format on real small bf16 weights and on 20k real n-gram rows;
  - built the real `qwen4_exp`/`glm5_next` classes from the real configs **lazily** (no weights materialized) to count parameters.

Parameter counts from the lazy real-config models:
- Qwen: 177.4B total = experts 120.8 + n-gram 51.2 + DeltaNet 2.09 + hyper-connections 0.64 + embed 0.64 + lm_head 0.64 + full attention 0.62 + vision 0.45 + shared experts 0.24 + router 0.06, plus a separately split MTP of about 2.6B.
- GLM: 313.9B = experts 304.4 + attention (KDA+MLA) 5.99 + shared experts 1.06 + embed/head 1.27 + vision 0.56 + dense MLPs 0.45 + indexer 0.08.

| # | Claim | Verdict | What changed |
|---|---|---|---|
| 1 | Affine-only backbone, format menu, add g128, no mxfp4/nvfp4/mxfp8 for experts | **Holds, one sub-claim corrected** | The format/bit/group support, the NAX kernel set and DWQ being affine-only were all verified. "Bits 1–8" is really {1–6, 8}. The arXiv paper does *not* show nvfp4 worse than INT4; locally nvfp4 is within ~14% of affine 4b g64 on RTN MSE. The 271/297 µs fp-mode decode numbers did not reproduce: on re-run mxfp4 was at parity, still with no speed gain. |
| 2 | Precision floors | **Holds** | Both vendor FP8 configs were verified line by line. The pipenetwork table matches its card; the non-expert group is about 4.9B params, not 4.2B. GLM's vendor config puts shared experts, dense MLPs and MLA q_a/q_b/kv_a/o in FP8, so a ≥ 6-bit floor is consistent with it. |
| 3 | oQe-style clip search + LSQ refit; skip GPTQ/AWQ | **Holds, numbers corrected** | The isolated refit A/B is −11.6…−15.3% text KL. The 27B and Flash-Next before/after numbers compare releases, not the refit alone. nace-ai's 0.129 is a larger mixed build, so it is not rounding evidence. GPTQ/AWQ rejection strengthened: neither mlx-lm tool can load these archs, GPTQ would skip mlx-vlm `SwitchLinear`, and mlx-vlm AWQ is a no-op. |
| 4 | MCKP + E2E-KL + swaps; joint prune ratio | **Holds with caveats** | GSQ-RCO, Litwein and sh0wie numbers verified. HumanEval is single-run (164 problems). sh0wie reports that KL hides a rare-token collapse below 288 experts, so pruning depth needs a non-KL guard. RCO's arXiv abstract does not mention KL or expert pruning, but the repo supports {keep, prune} search. |
| 5 | Band-wise DWQ; full `mlx_lm.dwq` does not fit | **Holds** | The 35–40 GB state figure assumes the n-gram table is mmapped (not trainable) and g64. Litwein (6/12 bands, −11.2% KL20 / −14.9% CE vs the pruned init) and the LEARNED_QUANTS tips verified. `mlx_lm.dwq` cannot load the archs anyway. mlx#4684, already in 0.32.4, speeds up the gather-qmm backward. |
| 6 | PLE table affine 4b g32 (32 GB) or nvfp4 g16 (28.8 GB) via `ple_storage` | **Holds, caveat added** | Layouts, row width 160 and sizes verified. New: the nvfp4 layout has no global scale, and at the table's true magnitude e4m3 scales go subnormal, giving 1.8× the error. Pre-scale the table and fold the inverse into `value_proj`; `fp8.py` drops `weight_scale` for sharded tables. |
| 7 | NAX prefill only; bits barely change prefill; <4 bits = memory not speed; Qwen latency-bound | **Partly refuted** | The kernel inventory holds, as do prefill parity within ±10–15% and Qwen latency-bound decode. **Refuted:** for GLM-shape experts, 3-bit decodes 15–20% faster than 4-bit and 2-bit about 35% faster. Non-expert bytes are ~4.2B, not 3.6B. Apple's ratios are base M5 vs M4. #4712 is fp-only and #4682/#4722 skip gather; #4684 *is* in 0.32.4. |
| 8 | Prioritize Qwen over GLM | **Conclusion holds; evidence corrected** | The GLM math was re-derived (68 GB needs about 53–56% pruning at 3.25–3.5 bpw). **Refuted:** "Qwen fits at 2.4–3.5 bpw with 96–99% retention". Only the 2.4/2.5 bpw GGUF builds fit 68 GB (95.7%), and in MLX unpruned Qwen needs ≈2.0–2.2 bpw experts, so pruning is required. |
| 9 | Separate MTP via `split_mtp`; connectors ≥ 8-bit; prune MTP experts | **Holds, caveat added** | `sanitize` dropping `mtp.*`, both splitters, the oQ docs and the vendor FP8 config were verified. Stock `split_mtp --q-bits` is uniform (it would quantize `fc_*` to low bits) and does not prune, so `build/` must apply its own predicate and keep set. |

Reproduce: `uv run python docs/research/quant_bench.py glm` and `uv run python docs/research/quant_verify_bench.py`.
