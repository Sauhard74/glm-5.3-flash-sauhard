# Expert pruning (REAP and related) for GLM-5.3-Flash and Qwen3.8-Flash-Next

Scope: how deep to prune each candidate so it fits the <= 70 GB weight budget in `PLAN.md`, how to calibrate the
saliency, and which implementation details matter on these two architectures. Researched 2026-10-20.

Legend: **[paper]** peer-reviewed or arXiv; **[src]** read in source code (REAP repo at commit `1970473`, 2026-04-17,
or the installed mlx-vlm 0.7.9); **[self-reported]** a number from a model card or README that was not reproduced
here. Every community eval below is self-reported, usually from a single run on a small sample. Read the numbers as
directions, not rankings.

## 1. The REAP criterion (exact)

Paper: Lasby et al., *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*, ICLR 2026,
[arXiv:2510.13999](https://arxiv.org/abs/2510.13999) **[paper]**:

$$S_j=\frac{1}{|\mathcal X_j|}\sum_{x\in\mathcal X_j} g_j(x)\,\lVert f_j(x)\rVert_2,\qquad \mathcal X_j=\{x : j\in \mathrm{TopK}(x)\}$$

- `g_j(x)` is the router gate weight and `f_j(x)` is the routed expert's output, before gating.
- The mean runs only over the tokens routed to expert `j`. This keeps rare but strong specialists.
- In each layer, the experts with the lowest `S_j` are pruned. The same count is removed from every layer.

Reference implementation ([`src/reap/pruning_metrics.py`](https://github.com/CerebrasResearch/reap/blob/main/src/reap/pruning_metrics.py)) **[src]**:
- `reap[i] = (ean_norm * active_router_weights).mean()` over the active tokens, accumulated with a count-weighted
  online mean.
- The code also saves `weighted_ean_sum`, which equals `reap × frequency`, plus `expert_frequency`, `ean_mean` and
  `max_activations`.
- `routing_weights = softmax(router_logits)`. With `renormalize_router_weights=True` the weights are divided by
  their top-k sum. A README update on 2026-03-11 says this renormalization is now correct and averages a 1.9 % drop
  instead of 2.6 % across five models.
- `prune.py` slices the expert weights, `router.weight`, `router.bias` and `e_score_correction_bias` to the kept
  indices. It uses a single `n_experts_to_prune` for every layer and does not retrain.
- Optional `perserve_super_experts` / `perserve_outliers` force-keep experts whose `max_activations` value exceeds
  `max(p99.5, global_max/10)`.

Two reference-code shortcuts are **wrong for GLM-5.3-Flash** and must not be copied **[src]**:
1. The observer computes `topk(router_logits)` and `softmax(router_logits)`. GLM-5.3 routes with `sigmoid` plus
   `e_score_correction_bias` (`noaux_tc`) for expert *selection*. It then takes the *weights* from the un-biased
   sigmoid, renormalizes them over the top-8 (`norm_topk_prob=True`) and multiplies by 2.5. This is
   `_expert_select` in `mlx_vlm/models/glm5_next/language.py`. The patrickbdevaney card says
   ([FP8-v2](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)) that most implementations get this
   detail silently wrong **[self-reported]**.
2. Zero saliency is replaced by the smallest non-zero value, which only matters for merging. OpenMOSE reports that
   for pruning this creates ties that can drop a used expert in favour of an unused one
   ([card §2.5](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B)) **[self-reported]**.
   *Verification: no such replacement exists in `pruning_metrics.py`/`prune.py` at `1970473`. There, an unused
   expert keeps `reap = 0` and is pruned first. This item is OpenMOSE's claim only, possibly about another version.*

A third point is a mismatch, not a bug. `perserve_super_experts` thresholds the largest single activation *element*
(`selected_activations.max()`), not the output norm. By default it only looks at the first 75 % of layers. OpenMOSE's
40–470× outliers are measured on output *norm*, so record both statistics **[src]**.

**For this project:** take `g_j(x)` as the actual combine weight the mlx-vlm gate returns. That is
`(indices, weights)` from `MoEGate` for GLM, and the softmax → top-10 → renormalized `scores` for Qwen
(`Qwen3_5MoeSparseMoeBlock`). This matches the forward pass exactly, including the renormalization fix.
`routed_scaling_factor` is a per-layer constant and does not change the ranking.

## 2. Architecture facts that drive the budget

From the model `config.json` files and model cards:

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| MoE layers | 42 of 45 (3 dense first) + MTP layer 45 | 48 + 1 MTP layer |
| Routed experts / top-k | 288 / top-8, sigmoid + bias, `norm_topk_prob`, ×2.5 | 512 / top-10, softmax, `norm_topk_prob` (default True) |
| Params per expert | 3·4096·2048 = 25.2 M | 3·2560·640 = 4.9 M |
| Params per kept-expert slot (all layers) | 1.057 B | 0.236 B |
| Routed-expert params | 304.4 B (97 %) | 120.8 B |
| Non-expert trunk | ~~~9.5 B~~ 8.9 B (+0.19 B MTP non-expert) + 0.56 B vision | ~5 B (4.95 B) + 0.45 B vision |
| Not prunable | — | **51.2 B n-gram/PLE table**, ~32 GB at 4-bit g32. Rows are 160 wide, so affine g64 is impossible (`quant_predicate` falls back to g32). ~~g32 is required~~: `ple_storage.py` accepts exactly two 4-bit layouts, affine g32 (5.0 bpw, 32.0 GB) or **nvfp4 g16 (4.5 bpw, 28.8 GB)**. The mmap streamer `QuantizedMMapNGramEmbedding` lets the table stay off-resident **[src]** |
| MTP layer | ~7.4 B (288 experts) | ~2.5 B experts (512) |

Weight-file size (decimal GB) by kept experts `K` and average expert bits per weight (bpw) including scales.
Assumptions: trunk at ~6–6.5 bpw, MTP experts pruned to the same `K`, vision dropped, Qwen n-gram table at 4-bit g32:

| Qwen K | 3.5 bpw | 4.0 | 4.5 | | GLM K | 2.75 bpw | 3.0 | 3.25 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 63.2 | 67.0 | 70.9 | | 128 | 54.9 | 59.2 | 63.5 | 67.8 |
| 288 | 66.6 | 70.9 | 75.2 | | 144 | 60.8 | **65.7** | 70.5 | 75.4 |
| **320** | **69.9** | 74.8 | 79.6 | | 160 | 66.8 | 72.2 | 77.6 | 83.0 |
| 352 | 73.3 | 78.6 | 83.9 | | 176 | 72.7 | 78.7 | 84.6 | 90.6 |

*Verification:* recomputed from the real safetensors headers (parameter counts per tensor). All cells agree within
0.3 GB. In MLX affine terms, "3-bit g64" is 3.5 bpw and "2-bit g64" is 2.5 bpw. So "3.0 bpw" means roughly half of
the expert weights at 2-bit. Qwen K=320 at uniform 3-bit g64 is 69.7 GB, which is **over the PLAN `--budget-gb 68`**.
Fitting 68 GB needs about 3.3 bpw of experts, or the nvfp4 PLE table (−3.2 GB → 66.5 GB).

*Residency:* the table above is **disk** size. With `ple_storage` (mmap streaming, shipped in mlx-vlm 0.7.9), the
~29–32 GB PLE table is not resident. sh0wie's REAP-288 drops from 68 GB to 39 GB resident with identical logits.
Litwein reports ≈36 GiB resident for its 70.8 GB build **[self-reported]**. For the co-tenant budget this matters
more than any K choice.

These estimates match the published builds:
- Litwein REAP-320 at 3-bit experts: 70.8 GB with a bf16 vision tower.
- lapadatbogdan REAP144 Q3_K: 71.4 GB, without MTP.
- pipenetwork REAP50 at mixed 4/8-bit: 96.3 GB.

## 3. Measured quality vs prune ratio, for these two models

### Qwen3.8-Flash-Next (512 experts)

| Source | Build | Result |
|---|---|---|
| [sh0wie REAP-288 MLX 4-bit](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit), 11-point ladder, HumanEval-164, 1 run, calibrated on ~686K tokens of agentic-coding traffic | 512 / 384 / 320 / 288 / 256 experts (4-bit) | **93.9 / 92.1 / 90.9 / 91.5 / 88.4 %** pass@1 **[self-reported]** |
| same | rare-name sampling probe | 288: 9/10, **256: 1/10**. "Reliability collapses below 288", which greedy evals do not show **[self-reported]** |
| same | top-k narrowing at 288 | top-6: 84.8 %, top-4: 63.4 %. Do not narrow routing **[self-reported]** |
| [sh0wie REAP-288 Q8E](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX) | 288 experts at 8-bit vs 4-bit | 90.9 vs 91.5 %. "Pruning error dominates quantization error at this width" **[self-reported]** |
| [WaveCut REAM-288](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-REAM-288), independent harness on mlx-vlm | sh0wie REAP-288 4-bit vs REAM-288 4-bit | HumanEval-164 89.0 vs 89.6 %; GSM8K-200 56.0 vs 58.0 % (capped at 320 new tokens). McNemar p = 1.0 / 0.59. Teacher-forced top-256 KL vs stock (bf16): 1.307 vs 1.256 **[self-reported]**. *The same REAP-288 file scores 91.5 % in sh0wie's harness and 89.0 % here. Treat a 2–3 point gap between sources as noise. The 4-bit quantizer alone cost 30 points on non-Latin rare names (bf16 93.3 % vs 4-bit 62.7 %), so bits do matter for rare-entity recall.* |
| [AnonimousA REAP-320](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF) (llama.cpp, imatrix-based salience, 30 % agentic / 30 % code / 15 % chat / 12.5 % math / 12.5 % writing) | K=320 at Q3_K_XL lineage, 69 GB | HumanEval 96.3 % (unpruned UD-IQ1_M: 95.7 %); 6-task battery 13/13/12; **fabrication 25 % vs 0 %** for the unpruned model **[self-reported]**. *The fabrication probe has 12 questions (3/12 vs 0/12, Fisher p = 0.22). The "unpruned" baseline is a 1-bit quant.* |
| same | equal disk: K=320 Q2 vs K=256 Q3 | HumanEval 95.1 vs 89.6 %, rare-name 9/10 vs 7/10, fabrication 25 vs 42 %. ~~"64 extra experts beat 1 extra bit"~~ **[self-reported]**. *Confounded. The K=256 Q3 build is the old code-only "duo". The same card's K=256 multi-domain rebuild (same bits) also scores 95.1 %. With the corpus held fixed, extra experts and extra bits tie on HumanEval.* |
| same | corpus swap at K=256, same bits | code-only → multi-domain: HumanEval **89.6 → 95.1 %** **[self-reported]** |
| [Litwein REAP320-oQ3e-DWQ](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX), the PLAN baseline | K=320 from sh0wie manifests, 3-bit experts, KL-DWQ | Held-out CE-gap / KL20 vs bf16: overall 0.090 / 0.104. Agent is the worst domain (0.224 / 0.185), code the best (0.044 / 0.060). DWQ recovered 33.6 % of the pruning gap. No task scores **[self-reported]** |

### GLM-5.3-Flash (288 experts)

| Source | Build | Result |
|---|---|---|
| [pipenetwork REAP25/37/50 MLX](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit), calibration 65K tokens of wikitext + Wikipedia + code | wikitext-2 PPL, 141 × 2048 windows, mixed 4/8-bit | unpruned 3.571 (8-bit: 3.461) → REAP25 **4.225** (+0.20 nats/token) → REAP37 **4.875** (+0.34) → REAP50 **6.076** (+0.56). Kept saliency mass 86.9 / 78.1 / 67.2 % **[self-reported]** |
| [patrickbdevaney REAP50 FP8-v2](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), 5.5 M tokens, 7 domains, 2048-token sequences | 144 experts, teacher-forced on 241K held-out tokens | top-1 agreement 0.842, ΔNLL +0.176. Per domain: math 0.920, code 0.919, agentic 0.873, science 0.830, finance 0.755, **ballast 0.580** **[self-reported]** |
| same ([GGUF](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)) | NIAH, llama.cpp with DSA run dense | 3/3 at 2K, 8K and 32K **[self-reported]** |
| [lapadatbogdan REAP144-Coder](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF) ([repo](https://github.com/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder)), calibrated on the model's own Python agentic sessions | ~~48 real repository tasks~~ 48 attempts (about 26 Python tasks × 2, from 3 repos; tests must pass). The author says 1–3 tasks is noise | full Q8 48/48; REAP144 Q8 48/48 at **×1.21 turns**; Q4_K_M 48/48 ×1.58; Q3_K (71 GB) 48/48 ×1.80; IQ2_S 45/48; IQ2_XXS 39/48 **[self-reported]** |
| same | masks at 144 experts | held-out KL: frequency 0.94, REAP-mean 0.85, **REAP-sum 0.73**. A public 7-domain REAP50 mask: 44/46 at ×2.29 turns **[self-reported]** |
| same | depth ladder | **120 experts: 35/38, loops for tens of thousands of tokens; 96: unusable**. Code PPL at 144 is +22 % **[self-reported]** |
| same | domain failure | Python-only calibration got 17×23 wrong and looped on other languages, at every quant level **[self-reported]** |
| [OpenMOSE REAP-250B](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B) | 228 experts (−20.8 %), domain-aware selection | KL 0.453, top-1 86.0 %, PPL +2.8 %. A uniform 224 selection on a single mixed set: KL 0.476, top-1 85.7 %, PPL +5.8 % (measured in a different run) **[self-reported]**. *Not equal size: 228 vs 224 kept.* |

### Cross-model REAP evidence (primary sources)
- **Paper, 50 % pruning** **[paper]**:
  - Qwen3-Coder-480B: code 0.636 → 0.619, SWE-bench-V 0.540 → 0.522, BFCL 0.690 → 0.674, MC 0.750 → 0.692.
  - GLM-4.5-Air: code 0.580 → 0.515, math 0.882 → 0.857.
  - Frequency-based pruning collapses (Qwen3-Coder-480B code 0.010).
  - Merging (M-SMoE, HC-SMoE) is worse than REAP at 50 %.
- **Cerebras checkpoints** **[self-reported, primary vendor]**:
  - [Qwen3-Coder-REAP-246B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8) at 50 %: SWE-bench-V
    54.0 → 52.2, BFCL-v3 69.0 → 67.4, τ²-telecom 50.0 → 47.1, TerminalBench 30.5 → 30.0.
  - [GLM-4.6-REAP](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8) at 25 / 30 / 40 %: GPQA 78.8 →
    76.8 / 75.8 / **69.7**, BFCL-v3 78.4 → 77.3 / 76.8 / 74.2, MATH-500 95.5 → 97.0 / 94.8 / 93.3.
    Knowledge-heavy evals degrade first.
- **Linear-attention hybrid (closest relative of both models)**:
  [Kimi-Linear-REAP-35B](https://huggingface.co/cerebras/Kimi-Linear-REAP-35B-A3B-Instruct) at 30 %, evaluated at
  128K: LongBench v2 36.8 → 37.2, **FRAMES 55.7 → 52.3**, AIME25 30 → 40, MATH-500 81.8 → 80.8. Long-context
  reasoning holds. Knowledge-dependent long-context QA drops.
- **[Composable compression benchmark](https://arxiv.org/abs/2608.21693)** **[paper]**: 10 MoE models from 30B to
  235B, pruning at 20–50 %. It finds that expert pruning is the dominant source of degradation, and that averages hide
  failures specific to one workload.

## 4. Calibration domain decides which experts survive

- **Neither model has dead experts.** OpenMOSE found that all 12,096 GLM experts were used. Routing entropy is
  0.92–0.99 and only 1.4 % of experts get under 10 % of average traffic, because aux-loss-free balancing works.
  Every pruned expert is real capacity, and the calibration mix picks which capacity goes **[self-reported]**.
- **Selections depend heavily on domain.**
  - Chat-only vs code-only saliency has a Spearman correlation of 0.714 (88 % overlap at k = 224).
  - A chat-only mask "lost most of its coding ability". The code specialists ranked ~240 under chat calibration and
    ~186 when code was included.
  - A code-only mask lost Japanese entirely.
  - 8× more data changed only 8 % of the selection. Content matters, quantity does not (OpenMOSE) **[self-reported]**.
- **The REAP paper agrees.** C4 calibration collapses code accuracy, sometimes to 0 %. Domain-specific calibration
  wins, and REAP is the most robust method under mixed calibration **[paper]**.
- **Mixed beats narrow, even on the narrow domain** (Qwen, AnonimousA: ~~HumanEval 96.3 multi-domain vs 95.7
  code-only~~). *96.3 vs 95.7 is one problem (158 vs 157 of 164), and 95.7 is also the card's number for the
  unpruned 1-bit quant. The usable evidence is the controlled K=256 corpus swap, 89.6 → 95.1 %.* But in-domain
  *agentic* data beats a generic mix for agentic tasks (GLM, lapadatbogdan: ×1.21 vs ×2.29
  turns). Narrow agentic-only data destroys arithmetic and other languages. The safe pattern is a workload-heavy mix
  plus a small general floor.
- **Cerebras' agentic recipe** ([README](https://github.com/CerebrasResearch/reap#huggingface-checkpoints))
  **[self-reported, primary]**: evol-codealpaca 4,096 samples, Mixture-of-Thoughts 12,288 (code / math / science),
  xlam-function-calling 4,096, SWE-smith tool trajectories 4,096. Sequences up to 16,384 tokens.
- **Combine per-domain rankings, not pooled scores.**
  - OpenMOSE computes saliency per domain, ranks experts within each domain, and scores each expert by its best
    normalized rank across domains. It also force-keeps the top-64 of every domain. ~~At equal size this beats a
    pooled mix on every metric~~. *It kept 228 experts against the pooled build's 224, and the reference came from
    another run. The gap (KL 0.453 vs 0.476) is small and confounded.* OpenMOSE also notes that the union of the
    per-domain cores bounds how far you can prune: the chat and code top-224 sets together already cover ≈251. At
    K=144 with 4–5 domains, protected cores must stay small (well under 144/domains) **[self-reported]**.
  - [HOPE, arXiv:2609.18916](https://arxiv.org/abs/2609.18916) **[paper]** generalizes REAP to pairwise
    interactions: it minimizes `pᵀFp` with `F_ij = mean g_i g_j ‖f_i‖‖f_j‖` over co-routed tokens. ~~On Qwen3.5-122B
    at 50 % it reports +2.8 % on average~~ *Correction: the "+2.8 % mean, up to +6.1 %" applies only to the agentic
    benchmark (SWE-bench Pro), at 40–50 % pruning, across Qwen3.5-122B-A10B, Qwen3.5-35B-A3B and GLM-4.5-Air. HOPE
    beat REAP in every one of those runs.* Calibration takes 6–7 % longer. Results are mixed per task: on
    Qwen3.5-122B, MATH is worse in 8 of 12 cells, up to −12.2 points at 50 %.
  - Retained saliency mass is circular. Each mask wins on its own corpus (AnonimousA correction). Rank masks only by
    held-out KL or behaviour.
- **Self-generated data helps.** lapadatbogdan used sessions the full model wrote itself. Litwein distilled on bf16
  teacher completions including reasoning. Calibration tokens in the model's own distribution, including the
  `<think>` and tool-call formats, route like real traffic.
- **Long context.** patrickbdevaney calibrated only at 2,048 tokens and flags long context as unassured. OpenMOSE
  packed 4,096. Nobody has published long-context-calibrated masks for either model, so the PLAN `longctx` domain
  (16K–64K) is novel and worth keeping. Expect knowledge recall (FRAMES-like) to drop more than long-document
  reasoning, which RAG partly offsets.

## 5. Uniform vs non-uniform per-layer allocation

- **Everyone ships uniform allocation**, including the REAP paper, HOPE, every Cerebras checkpoint and all the
  community builds above. The reasons are config and engine compatibility. GLM and Qwen configs have a single
  `n_routed_experts` / `num_experts`, and mlx-vlm builds every `SwitchGLU` from it **[src]**.
- **Evidence for non-uniform is weak to moderate.**
  - [EvoESAP, arXiv:2603.06003](https://arxiv.org/abs/2603.06003) **[paper]**: with REAP ranking, an evolutionary
    per-layer allocation gave small and inconsistent gains. Qwen3-30B at 25 %: ~~coding 0.629 → 0.580, MC 0.706 →
    0.665~~ *Code Avg 0.629 → 0.625 and MC 0.706 → 0.704 (Table 2), with WildBench +2.3 % and Math +1.4 %*.
    OLMoE improved (Code 0.160 → 0.189). ~~The authors conclude that REAP's uniform allocation "may leave little
    room".~~ *That quote is not in the paper. It says "there is still room for allocation search … but the
    additional gains are naturally more modest".* The big gains go to weaker rankings such as frequency.
  - OpenMOSE probed GLM-5.3 per-layer pruning error (k = 16…272) **[self-reported]**:
    - allocations that protect early layers cut KL by up to ~15 % at the same budget;
    - layers 42–44 are by far the easiest to prune;
    - the MoE blocks of DSA layers are slightly easier than those of KDA layers;
    - layers 17–34 have outlier experts with 40–470× the median output norm that must be kept.
  - [llm-compressor PR #3281](https://github.com/vllm-project/llm-compressor/pull/3281) (~~draft~~ open, not
    draft) adds global REAP pruning. Raw scores are not comparable across layers, so it normalizes by each layer's
    maximum. It keeps at least `num_experts_per_tok` experts per layer. ~~and forbids non-uniform pruning for
    group-limited routers~~ *What it protects is hash-routed layers (DeepSeek `tid2eid`), not group-limited
    routers. Neither model has hash routing.* It records per-layer counts in `moe_sparsity_config`, which mlx-vlm
    does not read.
- **Verdict:** ship uniform first. Treat non-uniform as a later experiment: a small, bounded shift (for example
  ±16 experts per layer, more kept in early layers and fewer in the last 3) chosen on held-out KL. It needs a loader
  shim that builds each layer's `SwitchGLU` from the shapes in the weight file, because stock `mlx_vlm.load` cannot
  express per-layer counts.

## 6. Router renormalization and post-prune "healing"

- **At inference no router change is needed** beyond slicing the router rows (and, for GLM,
  `e_score_correction_bias`). Both models use `norm_topk_prob=True`, so the gate renormalizes over the top-k of the
  surviving experts on its own **[src]**. *GLM sets it in `config.json`. Qwen's `config.json` omits it, and the
  default is True in both mlx-vlm and transformers 5.20. A tiny-block experiment (below) confirmed that slicing
  reproduces stock outputs exactly.*
- **The saliency side does matter.** Use the renormalized combine weights (§1). The REAP authors measured 1.9 % vs
  2.6 % mean drop with vs without renormalization. HOPE finds the renormalization error term "typically small"
  **[paper]**.
- ~~**Per-layer output-scale healing helps.**~~ *Per-layer healing has never been compared with no healing in any
  source. It is only "less bad than per-expert".* **Per-expert healing hurts** (patrickbdevaney, GLM-5.3 REAP50)
  **[self-reported]**:
  - A per-layer scalar on the down-proj scales (median 0.91) is the shipped default.
  - A closed-form per-expert least-squares fit lowered reconstruction error in 41 of 42 layers, yet was worse end to
    end: top-1 0.8369 vs 0.8425, McNemar z = 11.8.
  - A first-moment estimator that ignores `norm_topk_prob` over-corrects by ~30 %. Measure the scalar by replaying
    the pruned routing.
- **Training-based recovery is unstable at this scale.** LoRA healing diverged after ~16 steps (lapadatbogdan).
  KL-DWQ on quantizer scales and biases recovered 33.6 % of the gap (Litwein) **[self-reported]**.

## 7. Effect on the MTP head

- **Qwen: the MTP head survives an unpruned trunk-vs-head mismatch.** Unsloth's unpruned MTP head on the REAP-320
  trunk accepted 76–81 % of drafts (mean accepted length 2.5) and gave +32 % decode (AnonimousA).
  - Litwein pruned the MTP experts to 320 by weight energy, since the drafter has no routing data, and reports lower
    acceptance than the full drafter. They then self-distilled an MTP sidecar: +2–3 %.
  - sh0wie's separate bf16 drafter: 44–68 % acceptance **[self-reported]**.
- **GLM: the MTP layer is large (~7.4 B, ~4 GB at 4-bit unpruned).**
  - With all 288 experts unpruned at Q2_K (2.6 GB), drafts matched the REAP144 trunk 70 % of the time
    (lapadatbogdan).
  - lapadatbogdan: Q2_K through Q6_K draft experts all gave 68–71 %, so the drafter's bits barely matter.
  - patrickbdevaney measured 46.5 % acceptance at depth 2 and 58.5 % at depth 3 on REAP50. *Their GGUF MTP block is
    3.81 B params, which is exactly 144 experts (144 × 25.17 M + 0.19 B), so this measures a pruned MTP. The metric
    differs from lapadatbogdan's next-token match, so the two numbers do not compare directly.*
  - pipenetwork and the patrickbdevaney FP8 build dropped MTP entirely.
  - OpenMOSE copied it unpruned. Its card says engines that size it from `n_routed_experts` ~~break~~ "may fail to
    load" it **[self-reported]**.
- **mlx-vlm splits MTP into a separate drafter** (`speculative/drafters/{glm5_next_mtp,qwen4_exp_mtp}`) with its own
  `text_config`. The MTP expert count can therefore differ from the trunk's **[src]**.
- **Speculative decoding is lossless, so MTP pruning only affects speed.** Optimize it for acceptance per GB.

## 8. Merging instead of pruning

- REAM ([arXiv:2604.04356](https://arxiv.org/abs/2604.04356)) merges each pruned expert into its most similar kept
  expert **[paper]**.
- On Qwen3.8-Flash-Next-288, a REAM merge beat REAP on KL (1.256 vs 1.307). HumanEval and GSM8K were statistically
  tied. Raw saliency-proportional coefficients were catastrophic (KL 1.397): the best merge gives away only ~9 % of
  the kept experts' mass (WaveCut) **[self-reported]**.
- The REAP paper shows merging collapses the functional subspace at 50 % **[paper]**.
- **Not worth the complexity for v1.** Revisit only if KL is the deciding metric.

## Recommendations for this project

1. **Qwen3.8-Flash-Next: target K = 320 of 512 (37.5 % pruned), ~~3–3.5~~ ≈3.0–3.3 bpw routed experts** at
   `--budget-gb 68` (uniform 3-bit g64 = 3.5 bpw = 69.7 GB does not fit).
   - Keep the n-gram table at 4-bit. ~~It is required for group 32~~ Affine g64 is impossible. The streamer accepts
     affine g32 (32.0 GB) or nvfp4 g16 (28.8 GB, 3.2 GB smaller; compare their KL before choosing). Write the
     `ple_storage` manifest so the table is streamed instead of resident (≈38 GB resident instead of ≈70 GB).
   - Drop vision. This lands at ~~~66–70~~ ~65–70 GB (64.8 at 3.0 bpw, 69.7 at 3.5).
   - Floor K = 288: below it, rare-token reliability collapsed in one source (sh0wie 256: 1/10 vs 9/10, Fisher
     p = 0.001, code-only calibration). ~~AnonimousA K=256: −5.5 points HumanEval~~ *That drop was the code-only
     corpus. AnonimousA's multi-domain K=256 rebuild scored 95.1 %, so the floor is soft and calibration-dependent.*
     ~~At equal bytes, extra experts beat extra bits (AnonimousA; sh0wie Q8E).~~ *Not shown. The AnonimousA
     comparison is confounded by corpus. sh0wie Q8E only shows that 8-bit is no better than 4-bit at K=288, and
     WaveCut shows 4-bit already costs rare non-Latin names. Nothing tests 3-bit. Our own sweep decides this.*
   - The HumanEval ladder cannot separate 288/320/384 (all within ~1σ; sh0wie: "a point or two … within noise").
     For `--prune auto`, sweep K ∈ {288, 320, 352}, solve bits for each, and choose by per-domain KL plus the
     `code`, `tools` and rare-name evals.
2. **GLM-5.3-Flash: target K = 144 of 288 (50 % pruned), ~3.0 bpw routed experts (≈ 62–66 GB).**
   - This is already at REAP's validated ceiling. Do not go below 144: 120 experts loops and 96 is unusable
     (lapadatbogdan).
   - It needs sensitivity-driven 2/3/4-bit mixing. Q3_K at K=144 is 71 GB and did 48/48 at ×1.8 turns; IQ2_S did
     45/48. *These are imatrix k/i-quants. MLX affine 2-bit is much cruder, roughly 2× the relative error of 3-bit on
     tiny tensors here. Expect worse results than the GGUF numbers at the same bpw.*
   - Expect a ~~clearly~~ larger general-knowledge hit than Qwen at K=320 (wikitext +0.56 nats/token; ballast top-1
     0.58). *Plausible, since GLM loses 50 % against Qwen's 37.5 %, but no source compares the two head to head.*
   - If GLM is picked, sweep K ∈ {128, 144, 160}.
3. **Calibration.**
   - Compute REAP per domain (`code`, `agentic`, `reasoning`, `longctx`), plus a small `general` floor of about
     5–10 % so arithmetic and prose do not collapse.
   - Select by **max-normalized-rank across domains with protected per-domain top-N cores** (OpenMOSE), not pooled
     saliency. *Evidence is one confounded comparison (228 vs 224 kept). Keep N small enough that the union of
     cores is ≤ K, and A/B against pooled REAP on held-out KL.*
   - Write `expert_reap` (mean) and `expert_freq` per domain. REAP-sum (`reap × freq`) then comes for free; try it
     as a variant, since it beat REAP-mean on GLM in lapadatbogdan's KL (0.73 vs 0.85).
   - Prefer self-generated, chat-templated traffic with native tool calls and `<think>` blocks.
   - Use ≥ 2M tokens per domain. Check stability with a split-half test: kept-set overlap ≥ 85–90 %.
4. **Saliency implementation.**
   - Take the gate weights from the mlx-vlm gate's actual output: GLM selects with sigmoid + bias, weights without
     bias, renormalized over top-k; Qwen uses softmax → top-10 → renormalized.
   - Keep zero-saliency experts at zero.
   - Record per-expert max output norm *and* max activation element, and force-keep outlier "super experts" (REAP's
     p99.5 / max÷10 rule, which REAP applies to the activation element over the first 75 % of layers). GLM layers
     17–34 have 40–470× outliers in output norm.
   - Slice the GLM `e_score_correction_bias` together with the router rows.
5. **Allocation.** Use a uniform K per layer so the output stays loadable by stock `mlx_vlm.load`. Log the per-layer
   probe (output error vs k) during calibration so a non-uniform ±16 allocation can be tried later behind a loader
   shim. Expected gain is ≤ ~15 % KL, and per the literature it is unproven for REAP rankings.
6. **Router / healing.**
   - Do not retrain the router.
   - Optionally apply a **per-layer** output scalar measured by replaying the pruned routing on calibration data.
     Keep it only if eval KL improves. Never fit per-expert coefficients.
7. **MTP.**
   - Qwen: keep all 512 MTP experts at 4-bit (~1.4 GB), or prune them to K by REAP measured on the MTP forward pass.
     The unpruned head gets 76–81 % acceptance on a K=320 trunk.
   - GLM: prune the MTP experts to 144 using REAP measured on the MTP block during calibration, at 2–3 bit
     (~~~1.2–1.6~~ ~1.3–1.7 GB including the non-expert part). *No source shows that a pruned GLM drafter matches
     the full one. The full 288 experts at 2-bit cost ~2.4 GB and gave ~70 % regardless of bits. Measure both.*
   - Write MTP as the separate mlx-vlm drafter with its own `n_routed_experts`. Never let a pruned trunk config
     claim a different count than the MTP tensors carry (the OpenMOSE failure mode).
8. **Evals that catch pruning damage** that greedy pass@1 and KL miss:
   - a rare-name / entity sampled probe (10 seeded generations);
   - a fabrication probe;
   - termination and empty-content checks;
   - turns-to-solve on agentic tasks;
   - per-domain KL on `eval.jsonl`.
   - Size each probe so it can detect the effect. 12 fabrication questions cannot separate 25 % from 0 % (p = 0.22),
     and 164 HumanEval problems cannot separate 3-point gaps.
9. **Merging / HOPE.** Skip REAM for v1 (KL-only gain, tasks tied). ~~Skip HOPE~~ *HOPE's gain is concentrated
   exactly in this project's regime (agentic, 40–50 % pruning), and it costs only +6–7 % calibration time.* For v1,
   also accumulate the per-layer co-routing matrix `F_ij` (E×E per layer, ≤ 1 MB for Qwen). HOPE then becomes a
   build-time variant that needs no recalibration.

## Verification

Adversarial check on 2026-10-20. Sources were re-read from primary text: HF model-card READMEs fetched with
`hf download <repo> README.md`, the lapadatbogdan GitHub README, the REAP repo cloned at `1970473`, the arXiv HTML for
REAP, HOPE and EvoESAP, and llm-compressor PR #3281 via `gh`. Parameter counts come from the real safetensors
headers of both base repos (`get_safetensors_metadata`, headers only). Experiments use the real mlx-vlm 0.7.9 block
classes (`Glm5NextMoE`, `Qwen4ExpSparseMoeBlock`) with tiny random configs from `tests/fixtures/make_tiny.py`
(32 experts, hidden size 128–256). No real weights were loaded.

**Experiments**
- *Parameter budget:* GLM experts 304.41 B, MTP experts 7.25 B, other trunk 8.92 B, vision 0.56 B. Qwen experts
  120.80 B, PLE 51.20 B, MTP experts 2.52 B, other trunk 4.95 B, vision 0.45 B. The §2 size table reproduces within
  0.3 GB.
- *Router slicing is exact:* a block pruned 32 → 16 by slicing `gate.weight`, `e_score_correction_bias` and
  `switch_mlp` gives max |Δy| ≤ 6e-8 against the stock block, on every token whose top-k all survived (GLM and
  Qwen). GLM's pruned gate weights still sum to 2.5.
- *Reference REAP is wrong for GLM:* we compared the true gate (sigmoid + bias selection, un-biased renormalized
  weights) against REAP's observer path (`topk(logits)` + softmax renormalization) over 4,096 tokens.
  - Bias std 0: Spearman of the saliency rankings is 0.58–0.75 even though selections are identical, so the
    softmax-vs-sigmoid weighting alone reorders experts.
  - Bias std 0.05 / 0.2: only 68 % / 20–26 % of routing slots match, and Spearman falls to 0.14–0.58.
- *REAP-sum = REAP-mean × freq* holds exactly. Saving `expert_reap` and `expert_freq` is enough.
- *PLE quantization:* MLX refuses g64 on 160-wide rows. Affine g32 and nvfp4 g16 both work. nvfp4 is 4.5 bpw
  (relative error 0.10 vs 0.08 for affine g32 on Gaussian rows).

**Verdicts on the eight recommendations**

| # | Claim | Verdict | Notes |
|---|---|---|---|
| 1 | Qwen K=320 @ 3–3.5 bpw, PLE 4-bit g32, ≥288, sweep {288,320,352} | **Partly refuted** | Sizes and the Litwein 70.8 GB figure check out. Refuted: "g32 required" (nvfp4 g16 also supported), "experts beat bits at equal disk" (confounded by corpus), and 3.5 bpw does not fit `--budget-gb 68`. The 288 floor rests on one probe with code-only calibration. The HumanEval ladder is within noise. Missed: PLE streaming roughly halves resident memory. |
| 2 | GLM K=144 @ ~3.0 bpw, ≥144 | **Holds, with caveats** | 25.17 M/expert and 1.057 B/slot verified. 65.5 GB at 3.0 bpw. lapadatbogdan and pipenetwork numbers match their cards ("48 tasks" is 48 attempts). The GGUF quant results do not transfer to MLX affine 2/3-bit. "Larger hit than Qwen" is untested. |
| 3 | Exact REAP saliency; GLM sigmoid+bias, Qwen softmax→top-10→renorm; reference code wrong for GLM | **Verified** | Matches `pruning_metrics.py`, `observer.py` and mlx-vlm `_expert_select`, plus the experiment above. The renormalization fix (1.9 vs 2.6 %) is in the README. The zero-saliency claim about reference code was not found in source. |
| 4 | Per-domain calibration + max-rank with protected cores; self-generated data; save mean+freq | **Holds; weakly evidenced** | The REAP paper's C4 collapse, OpenMOSE's Spearman 0.714, the AnonimousA K=256 corpus swap and lapadatbogdan's KL 0.73/0.85/0.94 all verified. Refuted: OpenMOSE "equal size" (228 vs 224) and AnonimousA 96.3 vs 95.7 as evidence. Self-generated data has no controlled A/B. |
| 5 | Uniform v1, log per-layer probes, ±16 later | **Holds** | Both configs use a single expert count, and mlx-vlm builds `SwitchGLU` from it. EvoESAP numbers corrected (0.629→0.625, not 0.580) and a misquote removed. PR #3281 details corrected. |
| 6 | No router retrain; per-layer scalar only if KL improves; no per-expert; keep super experts | **Holds** | patrickbdevaney numbers verified (0.83693 vs 0.84249, z = 11.8, ~30 %). Refuted: "per-layer healing helps", since it was never compared with no healing. Qwen's `norm_topk_prob` comes from the default, not the config. The REAP super-expert rule uses the largest activation element and the first 75 % of layers. |
| 7 | MTP as separate drafter; Qwen 512@4-bit ~1.4 GB; GLM prune to 144 @ 2–3 bit | **Holds, GLM part unproven** | Drafters take their own `text_config`. `sanitize` drops layers ≥ 45. 76–81 % / +32 % verified. 512 experts at 4.5 bpw is 1.42 GB. patrickbdevaney's 46.5/58.5 % was on a 144-expert MTP. Full 288-expert Q2_K–Q6_K drafters all reach about 70 %, so pruned versus full GLM MTP needs measuring. |
| 8 | Skip REAM and HOPE; add pruning-specific evals | **Partly refuted** | REAM: WaveCut numbers verified, skip stands. HOPE: the "+2.8 %" is agentic-only at 40–50 %, which is this project's regime, for +6–7 % calibration cost. So log `F_ij` now. The evals stand, but fabrication at n = 12 is not significant (p = 0.22). |

## Sources
- REAP: [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), [code](https://github.com/CerebrasResearch/reap) (commit `1970473`)
- HOPE: [arXiv:2609.18916](https://arxiv.org/abs/2609.18916) · REAM: [arXiv:2604.04356](https://arxiv.org/abs/2604.04356) · EvoESAP: [arXiv:2603.06003](https://arxiv.org/abs/2603.06003) · [Composable compression benchmark](https://arxiv.org/abs/2608.21693) · [llm-compressor #3281](https://github.com/vllm-project/llm-compressor/pull/3281)
- Cerebras: [Qwen3-Coder-REAP-246B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-246B-A35B-FP8), [GLM-4.6-REAP-218B](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8), [Kimi-Linear-REAP-35B](https://huggingface.co/cerebras/Kimi-Linear-REAP-35B-A3B-Instruct)
- Qwen3.8-Flash-Next derivatives: [sh0wie REAP-288](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit), [sh0wie REAP-384](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit), [sh0wie Q8E](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX), [Litwein REAP320](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX), [Litwein MTPLX](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX), [AnonimousA REAP-320](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF), [WaveCut REAM-288](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-REAM-288)
- GLM-5.3-Flash derivatives: [pipenetwork REAP50](https://huggingface.co/pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit) (also REAP25/REAP37), [patrickbdevaney FP8-v2](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), [patrickbdevaney GGUF](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF), [OpenMOSE REAP-250B](https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B), [lapadatbogdan REAP144-Coder](https://huggingface.co/lapadatbogdan/GLM-5.3-Flash-REAP144-Coder-GGUF)
- Installed sources: `mlx_vlm/models/glm5_next/language.py` (`_expert_select`, `sanitize` drops layer ≥ 45), `mlx_vlm/models/qwen3_5_moe/language.py` (`Qwen3_5MoeSparseMoeBlock`), `mlx_vlm/models/qwen4_exp/language.py` (`quant_predicate`), `mlx_vlm/models/qwen4_exp/ple_storage.py`, `mlx_vlm/speculative/drafters/*_mtp/`
