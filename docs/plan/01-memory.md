# 01 — Unified-memory engineering: using every GB well

> Chapter 1 of the master plan. It covers how this machine's 128 GiB gets split between the model, its context, the
> page cache and your own work, and how to push model quality as far as that split allows. Sibling chapters:
> `04-pruning.md` (what to cut), `08-system.md` (serving, profiles, operations). This chapter builds on the verified
> research in `docs/research/{mac,decode,quant,reap}.md`, cited as *mac §n*, *decode §n* and so on. It does not
> overturn any finding there. Where it adds a correction, the new evidence is shown.
>
> Tags: **[measured]** = run on this machine for this chapter (tiny experiments, ≤ 2 GB, no model weights loaded);
> **[src]** = read in installed package source (mlx 0.32.4, mlx-vlm 0.7.9) or upstream at tag `v0.32.4`;
> **[computed]** = arithmetic from real `config.json` / safetensors parameter counts; **[self-reported]** = a number
> from a model card, README or issue that was not reproduced; **[est]** = an estimate with stated assumptions that
> must be measured before it is trusted.
>
> Units: **GB = 10⁹ bytes** (file sizes, `--budget-gb`, `--expert-cache-gb`). **GiB = 2³⁰** (RAM caps, MLX counters).
> Note that `serve/config.py` defines `GB = 1 << 30`, so `--memory-limit-gb 80` actually means **80 GiB = 85.9 GB**.
> This chapter writes all profile caps in GiB to match the code.

---

## 0. TL;DR: what this chapter decides

1. **Budget resident bytes, not disk bytes.** Qwen3.8-Flash-Next's 51.2 B-parameter n-gram/PLE table (29.8 GiB) is
   served from file-backed memory and does not count against the process (*08-system §2.1*, *reap §2*). Once the
   budget counts only what is resident, an **unpruned 512-expert Qwen at ≈3.6 bpw fits the 80 GiB Work profile**
   with ≈11 GiB to spare at 128K context: 56.4 GiB of weights + 12.9 GiB of runtime = 69.2 GiB [computed, §4.3].
   This is the largest quality lever in this chapter: it costs no pruning.
2. **Three profiles, one switch.** Work (cap 80 GiB, leaves 48 for you), Focus (92 GiB, leaves 36), All-in
   (104 GiB, leaves 24, just under `max_recommended_working_set_size` = 107.5 GiB). For GLM-5.3-Flash, use **one
   tiered checkpoint with three residency levels**. The profile only changes how many experts per layer are
   resident: R = 144 / 176 / 216 of 288. Experts that are not resident are **masked in the router**, which is
   exactly runtime REAP-pruning, or **streamed from SSD**. A profile switch then releases or loads tens of GB
   without a restart.
3. **Hot/cold tiering by routing frequency.** Measured on real routing counts (686K tokens, 48 layers × 512
   experts): the 25 % hottest experts carry **65.7 %** of the routing mass, and the hottest 50 % carry **88.4 %**
   [measured, §2.5]. Rate-distortion theory says to give an expert ½ bit more for every doubling of its routed
   importance. A 4/3/2-bit tier split stored at an average of 3.375 bpw has an idealized routed-weighted error
   **43 % below** uniform bits at the same bytes [computed].
4. **SSD streaming is a latency problem, not a bandwidth problem.** This SSD reads at 5.8 GB/s single-stream and
   11.8 GB/s with 4 streams. Random expert-sized reads run at 2.6–6.3 GB/s for 2.77 MB and 4.5–11.9 GB/s for 11 MB
   [measured]. Decode streams layer by layer, and each layer has to wait for its own misses, so the per-layer miss
   latency dominates (formula in §2.4). Expect a **20–50 % decode slowdown** for configurations that keep ≥ 75 % of
   the routing mass resident, and a **2–4×** slowdown when only half of it is resident [est].
5. **Stock `mlx-vlm --expert-cache-gb` is a correctness-first prototype.** It syncs to the CPU on every layer,
   loops in Python over experts, re-parses the layer file header on every miss, and in prefill re-reads every touched
   expert from file, resident ones included [src, §2.4]. Use it to measure hit rates. Build a slab-based
   `serve/tiered.py` for speed.
6. **New capability found [measured]:** `mx.asarray(np.memmap(...), copy=False)` adopts a page-aligned file mapping
   as a Metal buffer **with zero copy**. It does not appear in `phys_footprint`, GPU kernels read it, and
   `quantized_matmul` on it is bit-exact. File-backed weights are therefore possible: the kernel can reclaim them
   for free under pressure, and several processes can share one copy (§2.6). `mx.load` and `mx.array(...)` still
   copy (*mac §2*, re-confirmed).
7. **Watchdog.** Poll `kern.memorystatus_vm_pressure_level`, `kern.memorystatus_level`, swap-out rate and compressor
   growth every second. Respond in steps: shed the APC RAM tier → shrink the MLX cache → drop a GLM expert tier
   (profile downshift) → unload. Use hysteresis to avoid flapping (§4.6). This machine has already swapped out
   **≈392 GiB** over its lifetime and currently has **7.3 of 8 GiB** of swap in use [measured], so pressure here is
   not hypothetical.

---

## Contents

1. Why it matters (intuition)
2. How it works (mechanics + key math)
   - 2.1 Exact memory accounting
   - 2.2 Where a page lives: wired, anonymous, file-backed, compressed, swapped
   - 2.3 MLX's three knobs (recap and rules)
   - 2.4 SSD expert streaming: how `--expert-cache-gb` works and what it costs
   - 2.5 Expert-usage skew and hot/cold tiering
   - 2.6 Sharing weights across processes; zero-copy file-backed weights
3. Evidence
4. Plan for this machine
   - 4.1 The new budget rule and code changes
   - 4.2 Profiles and live switching
   - 4.3 **The table: model × quant × streaming per profile, with speeds**
   - 4.4 Building the candidates (commands)
   - 4.5 `serve/tiered.py`: tiered experts, masking, and the fast streaming path
   - 4.6 The memory-pressure watchdog
   - 4.7 Personal calibration data for hot sets
   - 4.8 Order of work and time estimates
5. Decision criteria & what to measure
6. Failure modes
7. Stretch ideas (push limits)
8. Sources

---

## 1. Why it matters (intuition)

**One pool, three claimants.** On Apple silicon the CPU, GPU and SSD controller share one 128 GiB DRAM pool. Every
byte goes to one of three uses:

- **Model weights.** More bytes buy more bits per weight or more experts, which means quality.
- **Context.** KV cache, linear-attention state, prefix-cache checkpoints and prefill scratch. More bytes buy longer
  prompts and faster agent turns.
- **Everything else.** macOS, Xcode, browsers, Docker, Claude Code itself, and the **page cache**.

There is no separate VRAM, so there is no hard wall between these uses either. If the model takes too much, the
kernel squeezes your apps. If you take too much, it squeezes the model.

**Decode is a memory-bandwidth machine.** Generating one token reads every active weight once:
`t_token ≈ bytes_active / BW_eff`. MoE helps a lot here. Qwen3.8-Flash-Next activates ~5.9 B of 180 B parameters
per token, and GLM-5.3-Flash ~16–18 B of 321 B (*mac §7*). Total model size therefore sets only the **memory**
requirement; speed is set by the **active** bytes. That asymmetry is what this chapter exploits:

- Most of an MoE's bytes are idle for any given token.
- Idle bytes do not need the fastest memory. They can live at lower precision, in reclaimable page cache, or on an
  SSD that reads at 6–12 GB/s.
- Only the hot fraction needs to sit in RAM at high precision.

**Clean versus dirty memory is the central idea for a co-tenant machine.**

- **Anonymous (dirty) memory**, which is what `mx.load` produces (*mac §2*), has no backing file. Under pressure the
  kernel must compress it or write it to swap, and reading it back costs CPU time and SSD writes. A model whose
  weights get compressed decodes at a fraction of normal speed, because every token touches pages spread across all
  active experts.
- **File-backed (clean) memory**, which is what the Qwen PLE table uses, can simply be **dropped** under pressure and
  re-read from SSD later. For the user's apps this is free memory: it costs nothing to reclaim. For the model it is
  free speed whenever RAM happens to be idle.

The ideal co-tenant model is small and dirty where it is hot, and large and clean where it is cold.

**Why "push limits" is achievable here.** This machine is unusual in three ways:

- 614 GB/s of memory bandwidth.
- An internal SSD that sustains ~12–14 GB/s by review benchmarks, and 11.8 GB/s measured here with 4 streams.
- Two candidate models whose architectures are already memory-frugal: hybrid linear attention keeps per-token KV at
  12 KB (GLM) to 28 KB (Qwen), while a dense 128B model needs 352 KB per token (§2.1).

The scarce resource is not raw capacity. It is **resident capacity that does not hurt the user**.

---

## 2. How it works (mechanics + key math)

### 2.1 Exact memory accounting

The model process footprint is:

```
F = W_res + S_fix + b·T + C_apc + A_peak + M_cache + O_proc            (all in the process's phys_footprint)
P = page cache used by mmapped tables / streamed experts                (outside F, reclaimable by the kernel)
```

The same decomposition appears in *08-system §2.1*. This section derives each term from first principles and the
real configs.

#### (a) Weights: `W_res = Σ_tensors n_params × bpw / 8`

MLX affine quantization stores, for every group of `g` consecutive input weights, `b`-bit integer codes plus one
**scale and one bias in bf16** (16 + 16 = 32 bits). Dequantization is `w ≈ s·q + β`. Hence:

```
bpw = b + 32/g
```

The overhead is pure metadata, and its share of storage grows as bits fall [computed]:

| format | bpw | share of bytes that are scales/biases | use |
|---|---|---|---|
| 2b g64 | 2.5 | **20.0 %** | cold experts only |
| 3b g128 | 3.25 | 7.7 % | robust experts |
| 3b g64 | 3.5 | 14.3 % | default expert floor |
| 4b g128 | 4.25 | 5.9 % | cheap 4-bit |
| 4b g64 | 4.5 | 11.1 % | hot experts, MLX speed sweet spot (*quant §4*) |
| 4b g32 | 5.0 | 20.0 % | PLE table (rows are 160 wide, so g64 is impossible: *reap §2*) |
| nvfp4 g16 | 4.5 | 11.1 % (e4m3 scale, no bias) | PLE alternative, saves 3.2 GB (*quant Rec 6*) |
| 8b g64 | 8.5 | 5.9 % | gates, embeddings, lm_head, hyper-connections |

*Intuition:* a group's scale and bias let the quantizer fit a fresh line `s·q + β` to every 64 weights. Smaller
groups track local outliers better, which lowers error, but they cost more metadata. g128 at 3 bits (3.25 bpw) can
beat g64 at 3 bits (3.5 bpw) per byte on well-behaved tensors. That is why *quant Rec 1* adds g128 to the
allocator's grid.

**Parameter budget** (verified from the safetensors headers, *reap §Verification*):

| | GLM-5.3-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| params per routed expert | 3 × 4096 × 2048 = **25,165,824** | 3 × 2560 × 640 = **4,915,200** |
| MoE layers × experts × top-k | 42 × 288 × 8 | 48 × 512 × 10 |
| routed experts total | 304.41 B | 120.80 B |
| n-gram / PLE table | – | 51.20 B (file-backed via `ple_storage`) |
| MTP experts | 7.25 B | 2.52 B |
| other trunk (attention, linear attn, shared experts, routers, hyper-connections, embed, head) | 8.92 B (+0.19 B MTP non-expert) | 4.95 B |
| vision (dropped) | 0.56 B | 0.45 B |

**Bytes per expert** [computed]: Qwen **2.765 MB** at 4.5 bpw, 2.150 MB at 3.5, 1.536 MB at 2.5.
GLM **14.16 MB** at 4.5, 11.01 MB at 3.5, 7.86 MB at 2.5. Each expert's three projections are stored separately
(`gate/up/down`).

Trunk bytes after floors (*quant Rec 2*): Qwen ≈ **4.6 GB** and GLM ≈ **7.8 GB** [est from the floor policy:
embed + lm_head at 8.5 bpw, the rest at ≈ 6.5 bpw]. MTP: Qwen's 512 experts at 4.5 bpw = 1.42 GB; GLM's at
K = 144 and 3.0 bpw = 1.36 GB.

**Validation against something measured.** The local REAP-288 4b g64 build predicts
288 × 48 × 2.765 MB = 38.2 GB of experts plus ≈ 3 GB of trunk, i.e. ≈ 41 GB resident. `results/runtime_tune.json`
logged a **peak of 47.9 GB** in `mlx_vlm.generate` on short prompts with a drafter loaded. The file does not record
which local Qwen build ran. The ~6–7 GB difference is runtime (terms c–f below). That is the size of error to
expect before measuring: about 10 %.

#### (b) Fixed recurrent state `S_fix` and per-token KV `b`

These are derived from the real `config.json` and match the measurements in *decode §3* byte for byte.

**Qwen3.8-Flash-Next** has 48 layers. With `full_attention_interval: 4`, 36 are GatedDeltaNet and 12 are QSA.

- **GatedDeltaNet state per layer** = `num_v_heads × d_k × d_v × 4 B` (fp32) = 48 × 128 × 128 × 4 = 3.0 MiB.
  The conv window adds `(kernel−1) × conv_dim × 2 B`, where conv_dim = 16·128·2 + 48·128 = 10,240, so
  3 × 10,240 × 2 = 60 KiB. Total **3.06 MiB × 36 = 110 MiB**, fixed for any context length.
- **QSA KV per token per layer:**

  | part | config fields | bytes |
  |---|---|---|
  | K + V | 2 KV heads × 256 dims × 2 (K and V) × 2 B | 2048 |
  | indexer key | 1 head × 128 × 2 B | 256 |
  | compressed block key | 256 B ÷ `indexer_compress_ratio` 4 | 64 |
  | position | int32 | 4 |
  | **total** | | **2,372** |

  × 12 layers = **28,464 B/token**. MTP adds one more QSA layer: 30,836 B/token.

**GLM-5.3-Flash** has 45 layers: 34 KDA and 11 DSA/MLA.

- **KDA state** = 64 × 128 × 128 × 4 B = 4 MiB, plus a conv window of 3 × 24,576 × 2 B. Total **4.14 MiB × 34 =
  141 MiB**.
- **DSA latent per token per layer:**

  | part | config fields | bytes |
  |---|---|---|
  | MLA latent | `kv_lora_rank` 512 × 2 B (NoPE, `qk_rope_head_dim: 0`, so no rope key) | 1024 |
  | pooled indexer key | 128 × 2 B ÷ `index_kpool` 4 | 64 |
  | valid flag | | 1 |
  | **total** | | **1,089** |

  × 11 layers = **11,979 B/token**. MTP adds one layer. As shipped, a 64 KiB/token projected side-cache is added on
  every DSA layer up to 32K tokens and never freed. *decode Rec 1* gives a bit-exact fix, and everything below
  assumes it.

| context | Qwen KV+state (+MTP) | GLM KV+state (+MTP, fixed) |
|---|---|---|
| 32K | 1.05 GiB | 0.54 GiB |
| 128K | **3.87 GiB** | **1.73 GiB** |
| 256K | 7.64 GiB | 3.33 GiB |

**Why this is a gift.** Compare dense attention, computed from the real configs:

- **Mistral-Medium-3.5-128B:** 88 layers × 8 KV heads × 128 × 2 × 2 B = **352 KiB/token → 44 GiB at 128K**.
- **Qwen3.8-27B:** only 16 full-attention layers, 64 KiB/token → 8 GiB at 128K.

Linear-attention layers compress the whole past into a fixed d_k × d_v matrix per head. The recurrence is roughly
`S_t = decay_t·S_{t−1} + β_t·k_t v_tᵀ` (the delta rule). Context therefore costs the hybrid models 10–30× less RAM
than a dense model. In a co-tenant budget, that freed RAM can go to experts.

#### (c) Prefill transient `A_peak`

These are the largest temporaries a single prefill chunk of `T = prefill_step_size` tokens creates at context
`ctx` [src + computed]:

- **Qwen QSA indexer.** `scores = query.astype(f32) @ pooled_keys.astype(f32).T` has shape
  `[B, 4 heads, T, ctx/4]` in fp32. That is `4 × T × ctx/4 × 4 B` = **1.07 GiB at T = 2048, ctx = 128K**, and it
  doubles at 256K or at T = 4096 (`qwen4_exp/language.py:1007`).

  MoE activations per layer: sorted inputs of `T × K × H × 2 B` = 105 MB, plus gate/up outputs of 52 MB. The
  hyper-connection residual is `T × 4 × 2560 × 2 B` = 42 MB. Budget **≈ 1.3 GiB at 128K**.
- **GLM.** The DSA prefill re-projects the entire latent: **64 KiB × ctx per DSA layer while it runs** (*decode §3*),
  i.e. 8 GiB at 128K. The indexer is query-chunked at 512 (`glm5_next/language.py:260`): a temporary of
  `512 × 32 heads × ctx/4 × 4 B` = 2 GiB at 128K. Budget **≈ 10 GiB at 128K, 5 GiB at 64K, 2.5 GiB at 32K**, until
  absorbed-query prefill (*decode Rec 2*) removes the 8 GiB term.
- **Logits.** Prefill chunks can skip logits (`skip_logits`). A full-chunk lm_head would be
  2048 × 248,320 × 2 B = 1 GB. Make sure only the last position is computed.

#### (d) MLX free-buffer cache `M_cache` and the counters

MLX keeps freed Metal buffers for reuse. They stay in `phys_footprint` but leave `get_active_memory()`.
`get_peak_memory()` is the high-water mark of **active memory only**. A user on an M5 Max 128 GB saw
`get_peak_memory()` report 46 GB while `footprint` showed 110 GB of dirty IOAccelerator memory
([mlx#3896](https://github.com/ml-explore/mlx/issues/3896)) **[self-reported]**. The allocator's own limit check
uses `active + cache` [src]. The rules:

- Use `set_cache_limit(2 GiB)`.
- Run admission on `active + cache` (*mac Rec 3*).
- Trust `footprint -p <pid>`, not `get_peak_memory()`, as ground truth.

#### (e) Prefix-cache checkpoints `C_apc` and process overhead `O_proc`

Recurrent state cannot be paged, so mlx-vlm's APC snapshots the **whole cache** (*decode §5*). One checkpoint costs
the KV+state at that length: 3.9 GiB for Qwen at 128K. `O_proc` covers Python, tokenizer, FastAPI and Metal
libraries, ≈ 1.5 GiB [est]. The MTP rollback history is 0.33 GiB (*decode §1*).

**Runtime budget** (everything except weights; 1 APC checkpoint, 2 GiB MLX cache) [computed, `scratchpad/mem/calc.py`]:

| ctx | Qwen runtime | GLM runtime (projected-cache fixed) |
|---|---|---|
| 32K | 6.5 GiB | 7.4 GiB |
| 64K | 8.6 GiB | 10.7 GiB |
| 128K | **12.9 GiB** | **17.3 GiB** (of which 10 GiB is the prefill transient) |

### 2.2 Where a page lives: wired, anonymous, file-backed, compressed, swapped

| State | Example here | Under pressure | Cost to get it back |
|---|---|---|---|
| **Wired** | MLX residency sets (`set_wired_limit`), kernel | never reclaimed | – |
| **Anonymous, in use** | `mx.load` weights (IOAccelerator, dirty: *mac §2*), KV | compressed, then swapped | decompress (CPU) or swap-in (SSD read + CPU) |
| **File-backed, clean** | Qwen PLE memmap; stock-offload expert files; our zero-copy weights (§2.6) | **dropped instantly** | re-read from SSD at 6–12 GB/s, no CPU work |
| **Compressed** | idle anonymous pages | swapped if it gets worse | decompression on fault |
| **Swapped** | | | SSD read + fault |

**macOS policy in one paragraph.** When free pages run low, the kernel first steals clean file-backed pages,
because that is cheap. Next it compresses inactive anonymous pages; the compressor is normal and healthy in
moderation. If that is not enough, it swaps compressed segments out, and finally jetsam kills processes. Activity
Monitor's "memory pressure" colour summarizes these stages. The trouble signal is **swap-outs and compressor size
both rising during steady work**
([guide](https://blog.chrischodev.com/macos-memory-pressure-guide/), [mole](https://mole.fit/blog/mac-memory-pressure-explained)).

This machine right now [measured, `vm_stat`, `sysctl`]:

| metric | value |
|---|---|
| page size | 16 KiB |
| wired | 7.1 GiB |
| file-backed pages | 33.8 GiB |
| anonymous pages | 84 GiB |
| compressor occupied | 2.3 GiB (storing 4.5 GiB) |
| swap in use | **7.3 of 8 GiB** |
| lifetime swap-outs | 25.67 M pages = **392 GiB** |
| `kern.memorystatus_vm_pressure_level` | 1 (normal) |
| `kern.memorystatus_level` | 92 (percent available) |

The machine has been running downloads, calibration and benchmarks in parallel. It has clearly been through hard
pressure, so the watchdog in §4.6 is needed.

**Why weights must not be compressed.** One decode token touches every active expert in every layer. With Qwen
that is 480 expert slots, a few MB each, spread over the whole weight set. If even 10 % of weight pages are
compressed, each token takes dozens of decompression faults on the critical path. A report on an M1 Max 64 GB shows
19.7 GB compressed while an MLX model sat idle
([dev.to](https://dev.to/sleepyquant/what-19-gb-of-memory-compression-taught-me-about-mlx-on-m1-max-3eha))
**[self-reported]**.

Wiring prevents compression but is tied to IOGPU panics, and macOS drops it on idle (*mac §3*). The plan therefore
has three parts: **don't wire** (*mac Rec 2*), keep the footprint **far enough below pressure that the compressor
never reaches the weights**, and have the watchdog **shrink before the kernel does**.

### 2.3 MLX's three knobs (recap; verified in *mac §2*)

- **`mx.set_memory_limit(n)`** is soft. It throttles scheduling and triggers cache GC; it never fails an
  allocation. The cap must be enforced by admission control, bounded KV, bounded prefill chunks and an APC byte cap.
- **`mx.set_cache_limit(n)`** limits the free-buffer pool. Use 2 GiB.
- **`mx.set_wired_limit(n)`** wires allocations at allocation time, up to `n`, in residency sets of ≤ 5 % of
  `max_rec`. It is overridden by mlx-vlm and mlx-lm (they set `max_rec` = 107.5 GiB), so wrap
  `mx.set_wired_limit` itself. Default off. If enabled: weights only, after `clear_cache()`, with
  `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000`.
- **`max_recommended_working_set_size` = 107.5 GiB** on this machine is a ceiling. Wiring above it raises
  `ValueError`, and the All-in profile stays below it.

A new observation [measured]: `sysctl kern.memorystatus.wired_max_proc_pct` = **25**. Its effect on IOGPU wiring is
**not verified**. If it limited a process's wired memory to 32 GiB, weights-only wiring of a 60 GiB model would
silently be partial. Add this to the wiring soak test in §5.

### 2.4 SSD expert streaming: how `--expert-cache-gb` works and what it costs

#### (a) What mlx-vlm 0.7.9 does [src: `mlx_vlm/moe_offload.py`, `models/switch_layers.py:356-460`, `utils.py:668-700`]

1. **Repack (offline).** `python -m mlx_vlm.moe_offload --build <mlx dir> --out <dir>` splits each MoE layer's
   stacked `[E, out, in]` tensors into per-expert keys `e{j}.{gate,up,down}_proj.{weight,scales,biases}`. It writes
   one `experts/layer_XXXX.safetensors` per layer, `resident-*.safetensors` for everything else (shared experts
   stay resident), and `offload_index.json`. The streaming is shard by shard, and free disk must be 2× the source.
2. **Load.** `mlx_vlm.load` sees `offload_index.json` and calls `patch_model`. That swaps every `SwitchGLU` whose
   expert count matches for an `OffloadedSwitchGLU`, resolving per-projection (bits, group, mode) from the
   `quantization` config, so mixed precision survives. It refuses partial repacks loudly.
3. **`ExpertStore`.**
   - `mx.load(layer_file)` for each layer gives lazy arrays.
   - On `get(layer, j)`: if `(layer, j)` is in the LRU it is a **hit**. Otherwise it is a **miss**: the expert's
     arrays are added, its bytes are charged to the budget, and LRU experts are evicted until `resident ≤ budget`.
     Eviction drops the dict keys so MLX can free the buffers, then calls `mx.clear_cache()`.
   - The budget is `--expert-cache-gb × 10⁹`. **Default: 0.8 × max_rec − resident bytes − KV reserve**, which on
     this machine is ≈ 92 GB minus resident. That is far above the Work cap, so always set it explicitly.
   - Stats are at `GET /v1/moe-offload/stats` (`hits`, `misses`, `evictions`, `hit_rate`, `resident_bytes`).
4. **`OffloadedSwitchGLU.__call__`.**
   - `idx = np.asarray(indices)` forces a **GPU→CPU sync in every MoE layer**: the CPU must know the routing before
     it can fetch.
   - It then loops in Python over the unique experts. For each one it gathers the tokens routed there, runs three
     `mx.quantized_matmul` calls, and scatter-adds into the output.
   - If more than half of the layer's experts are touched (`len(uniq)·2 > E`, i.e. any real prefill chunk), it calls
     `get_all`, which loads the needed experts through a **fresh `mx.load`**, bypassing the LRU.
5. **Prefill must be chunked.** `generate.common` forces `prefill_step_size` if None. One lazy full-prompt graph
   would pin every touched expert until the final eval.

**What this costs, and why it matters for planning** [src, reasoning]:

- **Miss path = `pread` copy.** `mx.load` reads with `pread` (*mac §2*), not mmap. An evaluated miss allocates an
  anonymous Metal buffer, and the OS page cache keeps a clean copy of the same bytes. A warm expert can therefore
  cost **up to 2× its size** until the kernel reclaims the page-cache copy. The source comment says "mmap, lazy";
  the bytes reach the GPU through a copy.
- **Each miss re-parses the layer's safetensors header.** `_ensure_loaded` calls `mx.load(path)` again. For Qwen
  that is a 4,608-entry header (512 experts × 9 tensors) per miss.
- **Decode does ~5 small kernels per (layer, expert) plus one sync per layer.** Qwen: 48 syncs and ~2,400 tiny
  kernels per token, against roughly 3 fused `gather_qmv` launches per layer in the resident `SwitchGLU`. On
  Qwen-sized experts (latency-bound already, *quant §4*), this overhead alone could exceed the resident token
  time [est].
- **Prefill re-reads every touched expert from file, including those already in the LRU** (`get_all` path). Each
  2,048-token chunk touches ~95 % of a layer's experts (table below), so every chunk re-reads ~the whole expert
  corpus from page cache or SSD.

Conclusion: stock offload is the right tool to **measure hit rates and validate correctness** on a real model, and
a weak tool for **production speed**. §4.5 describes a slab design that removes the Python loop and the
double copy, and keeps one sync per layer only when a miss is possible.

#### (b) This SSD [measured, `scratchpad/mem/ssd2.py`]

Method: files verified cold with `mincore` before reading, `F_NOCACHE`, and a shared machine with a GPU benchmark
running. Treat the numbers as lower bounds, ±30 %.

| access | QD1 | QD4 | QD16 |
|---|---|---|---|
| sequential, 8 MB blocks | 5.83 GB/s (1 stream) | 11.78 GB/s (4 streams) | – |
| random 2.77 MB (one Qwen expert at 4b g64) | 2.64 GB/s, 1.05 ms/read | 5.14 GB/s | 6.29 GB/s |
| random 11 MB (one GLM expert at 3b g64) | 4.54 GB/s, 2.42 ms/read | 11.75 GB/s | 11.89 GB/s |
| random 16 KB page | 0.12 GB/s, 0.14 ms/read | 0.37 GB/s | 0.28 GB/s |

Published numbers for this SSD: 12.6–13.6 GB/s read on the 4 TB M5 Max
([PetaPixel](https://petapixel.com/2026/03/13/macbook-pro-with-m5-max-review-this-is-better-than-your-desktop/),
[MacRumors](https://macrumors.com/review/macbook-pro-m5-pro-m5-max), Apple quotes up to 14.5)
**[self-reported]**. The flash-moe author measured 17.5 GB/s on an M3 Max
([flash-moe](https://github.com/danveloper/flash-moe)) **[self-reported]**.

Lessons:

- **(i) Parallelism is mandatory.** One stream gets half the bandwidth.
- **(ii) Small reads die on latency.** Reading a 16 KB page at a time manages 0.1–0.4 GB/s. This is why flash-moe
  found mmap-faulted experts **5× slower** than explicit `pread` **[self-reported]**: page faults are QD1 at 16 KiB.
- **(iii) F_NOCACHE did not stop caching.** The post-run `mincore` showed 17–52 % of the touched files resident
  (read-ahead). Do not rely on it to keep streamed experts out of RAM.

#### (c) The decode formula (derivation)

Let each MoE layer `l` route `K` experts. A fraction `m` of the routing mass falls on non-resident experts, a
fraction `p` of misses is served from page cache, and `S` is the expert size. Layers are **sequential**: layer
`l+1`'s routing depends on layer `l`'s output, so a miss in layer `l` cannot be hidden behind compute of later
layers without a predictor. The cost per layer is:

```
expected misses       n  = K·m
P(layer stalls)       P  = 1 − (1 − m)^K                    (≥ 1 miss; independence approximation)
stall time            t_l = P·(ℓ_ssd·(1−p) + σ)  +  n·S·[ (1−p)/BW_ssd(n) + p/BW_pc ]

t_token = t_resident + Σ_l t_l           tok/s = 1 / t_token
```

- `ℓ_ssd` ≈ 0.10–0.15 ms is the first-byte latency, measured from the 16 KB QD1 reads.
- `σ` ≈ 0.1 ms is the GPU→CPU sync to learn the routing [est].
- `BW_ssd(n)` is the queue-depth-dependent bandwidth from the table above, ~3–5 GB/s for one or two concurrent Qwen
  expert reads.
- `BW_pc` ≈ 40 GB/s is the memcpy rate from page cache [est].

Two limits make the shape clear:

- `m → 0` gives `t_resident`.
- Large `m` gives `L·K·m·S / BW`, which is bandwidth-bound.

For Qwen-sized experts the **latency term `P·ℓ` and the QD1 bandwidth dominate**, because each layer sees about one
miss: 2.77 MB at 2.6 GB/s ≈ 1 ms, which is huge next to a ~0.7 ms resident layer. For GLM-sized experts the
**bandwidth term dominates**. *08-system §7.3* gives a bandwidth-only upper bound; this formula adds the latency
term, which is why its numbers are lower.

**Worked examples** [computed with `scratchpad/mem/stream.py`]. Assumptions:

- `m` comes from the measured static hit-rate curve of §2.5. For GLM, a +30 % miss "penalty" is added because
  GLM's routing is flatter (entropy 0.92–0.99, OpenMOSE) **[self-reported]**.
- `t_resident` is 33 ms for Qwen (measured ~30 tok/s plain on this machine, `results/runtime_tune.json` and
  `logs/server.log`) and 25 ms for GLM [est: 40 tok/s ≈ 50 % of the 78 tok/s ceiling, *mac §7*].

| configuration | miss m | misses/token | MB/token from SSD | tok/s, cold page cache | tok/s, 70 % page-cache hits | resident-only |
|---|---|---|---|---|---|---|
| Qwen 512e: 288 hottest resident @3.5, 224 on SSD | 0.082 | 39 | 85 | **16.4** | 22.3 | 30 |
| Qwen 512e: 384 resident | 0.021 | 10 | 22 | **24.6** | 27.5 | 30 |
| Qwen 512e: 448 resident | 0.005 | 2.4 | 5 | **28.7** | 29.6 | 30 |
| GLM 288e: 144 resident @3.5, 144 on SSD (+30 % penalty) | 0.15 | 51 | 558 | **9.8** | 16.6 | 40 |
| GLM 288e: 176 resident (+30 %) | 0.08 | 27 | 297 | **14.9** | 22.6 | 40 |
| GLM 288e: 216 resident @3.25, 72 @2.5 on SSD (+30 %) | 0.027 | 9 | 72 | **27.7** | 33.1 | 40 |

Read the table this way:

- Streaming is cheap only when ≥ ~97 % of the routing mass is resident.
- At ~90 %, decode halves.
- At ~85 %, decode drops 3–4×.
- The page cache helps when RAM is idle, so streaming builds get faster when you stop coding, with no action needed.
- With a stock `OffloadedSwitchGLU`, subtract its Python-loop overhead as well (§2.4a).

**Reality check.** flash-moe runs Qwen3.5-397B (60 layers, K = 4, 6.75 MB experts) on a 48 GB M3 Max. It reported
**2.41 ms per layer** for the parallel K = 4 `pread`, a **~71 % page-cache hit rate** with a 35 GB cache (≈ 17 % of
the 209 GB expert set), and **4.36 tok/s**. Every custom cache it tried was slower than "trust the OS page cache"
**[self-reported]**. Its per-layer stall matches the formula's order of magnitude.

The 71 % hit rate with 17 % cached is also evidence that **session locality beats global frequency**: the global
static curve in §2.5 predicts only ~53 % for 17 % of experts. Real sessions reuse a narrower expert set than the
corpus-wide counts suggest.

#### (d) Prefill and MTP under streaming

A prefill chunk of `T` tokens touches `U(T)` unique experts per layer. From the measured Qwen frequencies, using
an independence approximation that over-counts somewhat because real tokens are correlated [measured,
`scratchpad/mem/uniq.py`]:

| tokens in the step | unique experts per layer (of 512) | cold experts touched (of 224, with 288 hottest resident) |
|---|---|---|
| 1 (decode) | 9.8 (1.9 %) | 0.8 |
| 4 (MTP verify block) | 36 (7 %) | 3.2 |
| 16 | 116 (23 %) | – |
| 64 | 262 (51 %) | 43.5 |
| 256 | 394 (77 %) | – |
| 2048 | 484 (94.5 %) | 196 |

- **Prefill I/O per chunk ≈ the whole cold set** (`L·U_cold·S`), so prefill I/O **per token falls as 1/T**.
  Example: GLM with 37 GB cold (112 experts @2.5).
  - Chunk 2048: +3.7 s per chunk at 10 GB/s, capping prefill at ≈ 550 tok/s.
  - Chunk 8192: ≈ 2,200 tok/s.

  So streamed builds want **big prefill chunks**, limited by the activation transient in §2.1c. An alternative is to
  **mask cold experts during prefill only** (a "prefill-time REAP"), accepting a small quality cost on prompt
  tokens.
- **MTP verify** of B = 4 tokens touches ~4× the cold experts of one token, but the cost is amortized over the
  accepted tokens (≈ 2.5). I/O per output token is therefore ≈ 1.3× that of plain decode. The latency term `P·ℓ` is
  paid once per verify instead of once per token, so **MTP still helps under streaming**, less than when resident.
- **Agentic turns** (Claude Code) prefill a few hundred to a few thousand new tokens per turn after an APC hit.
  Each turn is one or two chunks, i.e. **one full cold-set read per turn** (2–4 s for GLM at 37 GB). This hidden
  cost favours masking over streaming in the Work profile.

### 2.5 Expert-usage skew and hot/cold tiering

#### (a) How skewed is routing? Measured on real counts

The baseline build ships its calibration counts: `models/qwen-flash-next-reap288/manifests/saliency_full_505.npz`
holds `count[48, 512]` over **685,786 tokens** of agentic-coding traffic, 10.007 assignments per token per layer.
Analysis [measured, `scratchpad/mem/skew.py`]:

| hottest N experts per layer (static set) | share of experts | **routing mass covered** (mean / min over layers) |
|---|---|---|
| 32 | 6.2 % | 0.292 / 0.157 |
| 64 | 12.5 % | 0.449 / 0.276 |
| **128** | **25 %** | **0.657** / 0.468 |
| 192 | 37.5 % | 0.792 / 0.623 |
| **256** | **50 %** | **0.884** / 0.748 |
| 288 | 56 % | 0.918 / 0.803 |
| 320 | 62.5 % | 0.944 / 0.853 |
| 384 | 75 % | 0.979 / 0.933 |
| 448 | 87.5 % | 0.995 / 0.986 |

Further statistics from the same counts:

- Normalized routing entropy H/ln E: 0.829 min, **0.911 median**, 0.965 max.
- Max/median expert frequency per layer: **15.5×** (median over layers).
- Dead experts: 31 of 24,576.
- Early layers are the most skewed: layer 0 top-128 covers 0.468, layer 42 covers 0.678.
- The REAP-288 kept set covers only **79.4 %** of the routing mass (min 62.5 %). REAP keeps by saliency, not
  frequency, so it retains rare-but-strong specialists (*reap §1*). Frequency and importance are different
  rankings, which §2.5c handles.

*Intuition:* aux-loss-free balancing keeps every expert alive, so there are almost no dead experts. Within a
workload, though, a "head" of experts does most of the work: the distribution is Zipf-like. GLM is reported flatter
(entropy 0.92–0.99, only 1.4 % of experts below 10 % of average traffic, OpenMOSE) **[self-reported]**, so assume
GLM's curve sits below Qwen's.

#### (b) Two kinds of hot

- **Static hot:** corpus-wide frequency. That is the table above, and it is what calibration gives you.
- **Session hot:** what this conversation or repository uses right now. flash-moe's 71 % page-cache hit rate from
  17 % residency is far above the static curve. On the other hand its *token-to-token* expert prediction reached
  only 25 % **[self-reported]**. Routing is therefore sticky at the scale of a session (hundreds or thousands of
  tokens), not between adjacent tokens. Design consequence: an **LFU/LRU over a window of thousands of tokens** can
  adapt the resident set, while per-token prefetch prediction is not worth building.

#### (c) Bits by importance: the reverse water-filling rule

The MoE output is `y = Σ_{j∈TopK(x)} g_j(x) f_j(x)`. Quantizing expert `j` adds `Δy = Σ g_j Δf_j(x)`. With
independent rounding errors, the expected squared error decomposes per expert:

```
D = Σ_j  w_j · ε_j(b_j),       w_j = freq_j · E[ g_j² ‖f_j‖² | routed to j ],    ε_j(b) = NMSE of expert j at b bits
```

Every term is already written by `calib/`: `expert_freq`, `expert_gate_mean`, `expert_out_norm`/`out_sq`, and
per-expert `sens[E, bits, group]` (*docs/contracts/calib.md*). Minimize `D` subject to
`Σ_j n_j · bpw(b_j)/8 ≤ B`.

Using the rule of thumb `ε(b) ≈ c·4^{−b}` (each bit halves the RMS error), the Lagrangian condition
`w_j c ln4 · 4^{−b_j} = λ n/8` gives:

```
b_j* = b̄ + ½ · log₂( w_j / w̄_geo )
```

**Every doubling of an expert's importance earns half a bit.** This is the classic rate-distortion
"reverse water-filling" result. With Qwen's 15.5× max/median frequency spread, the hottest experts deserve
½·log₂ 15.5 ≈ **2 more bits** than the median one.

**Worked example** [computed]. Qwen, three tiers using the measured mass curve:

| tier | experts | bits | routing mass |
|---|---|---|---|
| hot | top 128 | 4b (4.5 bpw) | 0.657 |
| mid | next 192 | 3b (3.5 bpw) | 0.287 |
| cold | last 192 | 2b (2.5 bpw) | 0.056 |

- Storage: (128·4.5 + 192·3.5 + 192·2.5)/512 = **3.375 bpw**, i.e. 51.0 GB for all 512 experts.
- Routed-weighted error ∝ 0.657·4⁻⁴ + 0.287·4⁻³ + 0.056·4⁻² = **0.0106**.
- Uniform bits at the same bytes would be 2.875 bits, error 4^−2.875 = 0.0186. The tiers are **43 % lower**.
- Even uniform 3-bit (0.0156), which costs *more* bytes (3.5 bpw), is 32 % worse than the tiers.

Tokens effectively "see" 0.657·4.5 + 0.287·3.5 + 0.056·2.5 ≈ **4.1 bpw** of quality at 3.375 bpw of storage.

Caveats:

- The 4^−b rule is idealized. Use the measured `sens` grid, which the knapsack DP in `build/alloc.py` already
  handles if each expert tier becomes its own allocation unit.
- `w_j` must include gate and norm, not frequency alone. REAP's lesson is that rare experts can be decisive, so keep
  a **3-bit floor** for REAP "super experts" and outliers (*reap Rec 4*).
- 2-bit affine is crude in MLX (*reap Rec 2*). Keep the cold tier small, or mask it rather than quantize it to 2 bits.

#### (d) The three-tier memory hierarchy, and pruning as its limit

```
tier H  (RAM, high bits)   — the routing head; most tokens' work happens here
tier W  (RAM, low bits)    — the shoulder; profile-dependent residency
tier C  (SSD / page cache) — the tail; three possible policies:
            stream  : fetch on miss           → exact model, latency cost (§2.4c)
            mask    : router logit = −∞       → exactly REAP-pruning of these experts, zero latency cost
            hybrid  : mask now, fetch async   → "session-adaptive pruning" (§4.5, stretch)
```

**Masking a non-resident expert is mathematically identical to pruning it.**

- GLM selects with `sigmoid(logits) + e_score_correction_bias` and takes the combine weights from the un-biased
  sigmoid, renormalized over the top-k (*reap §1*). Setting the **selection bias of non-resident experts to −∞**
  removes them from the top-k, and the renormalization runs over survivors only. That is exactly the
  slice-the-router-rows pruning that *reap §Verification* showed reproduces a pruned block to 6e−8.
- Qwen (softmax → top-10 → renormalize): setting the logits to −∞ before the softmax gives the same result.

So **a single checkpoint can be "pruned to R" at load time or live, for any R**. Pruning becomes a runtime
residency decision, which is what makes live profile switching cheap.

Which experts go in tier H depends on the objective:

- **Frequency** maximizes the streaming hit rate.
- **REAP saliency** minimizes masking damage.

For a masked tail, rank by REAP. For a streamed tail, rank by frequency, because a streamed expert is never lost,
only slow. Choose per profile.

### 2.6 Sharing weights across processes; zero-copy file-backed weights

**Today.** `mx.load` copies weights into anonymous Metal buffers (*mac §2*), so two processes that load the same
model pay twice. Multiple MLX processes are also among the IOGPU-panic correlates (*mac §3*). The rule follows: **one
model server process, many clients over HTTP**. Embedder and reranker go in the same process (*08-system §4.2e*).
Claude Code, OpenCode, scripts and evals all talk to `127.0.0.1:8642`. Evals that need the model in-process should
run in the `calibrate` window, not next to a live server.

**New: zero-copy file-backed Metal buffers work in this MLX** [measured, `scratchpad/mem/nocopy2.py`,
`nocopy3.py`]:

```python
mm = np.memmap("weights.bin", dtype=np.uint8, mode="r")      # page-aligned file region
a  = mx.asarray(mm, copy=False)                               # adopts the mapping; no copy
```

| test | result |
|---|---|
| `phys_footprint` after adopting 512 MiB | **unchanged (24 MB)**, while `mx.get_active_memory()` counts 512 MiB |
| zero-copy proof (mode `r+`): write through numpy, read through mx | change visible: shared memory |
| `mx.quantized_matmul` on a 4b g64 expert whose weight/scales/biases are adopted from a page-aligned file | **max \|diff\| = 0.0** vs in-RAM arrays |
| `mx.array(np.memmap)` (the default constructor) | **copies**: +512 MB IOAccelerator dirty |
| mode `r` (read-only mapping) | works |

The upstream source matches: `convert.cpp` at `v0.32.4` tries `cpu_nd_array_to_mlx_no_copy` →
`allocator::make_buffer` → `newBufferWithBytesNoCopy` for `copy=None/False`. The array constructor hard-codes
`copy=True`, which is why only `mx.asarray(..., copy=False)` takes this path [src]. Requirements: page-aligned
pointer (16 KiB) and an item-size-aligned length.

What this enables:

1. **Weights the kernel can reclaim for free.** Under pressure, clean pages are dropped instead of compressed or
   swapped, so the user never pays a compression storm for the model. The model pays SSD re-reads instead.
2. **Cross-process sharing.** Two processes mapping the same file share the page cache: one physical copy. This
   follows from how the OS works; it has not been tested with two MLX processes.
3. **Instant "load".** Mapping is O(1); pages fault in on first use.

Open questions:

- **Behaviour under pressure while the GPU uses the pages.** Does IOGPU wire them during command-buffer execution?
  Can an eviction mid-kernel cause a GPU fault or stall?
- **Cold fault speed.** It is 16 KiB QD1, so 0.1–0.4 GB/s per the table, unless warmed with parallel `pread`
  (`madvise(WILLNEED)` was not tested).

Use: tier W and tier C experts, and the PLE table. **Not** the hot tier until a soak test passes (§5). Packing needs
16 KiB-aligned tensor offsets: a custom file layout, or a safetensors repack with padding. See §4.5.

---

## 3. Evidence

| Claim | Source | Status |
|---|---|---|
| MLX memory/cache/wired semantics, `max_rec` 107.5 GiB, `mx.load` copies, wiring/panic evidence | *mac §1–4* | [measured]/[src] earlier, not re-run |
| KV+state bytes per token and fixed state; GLM projected-cache leak and fix | *decode §3*, re-derived here from `config.json` | [measured]+[computed] |
| Parameter counts per component | *reap §Verification* (safetensors headers) | [computed] |
| Routing skew curve (65.7 % at 25 %, 88.4 % at 50 %, entropy median 0.911, REAP-288 keeps 79.4 % of mass) | `saliency_full_505.npz` in the local baseline; `scratchpad/mem/skew.py` | **[measured]** (counts from sh0wie's calibration run) |
| SSD: 5.8 GB/s single stream, 11.8 GB/s ×4, 2.6–6.3 GB/s random 2.77 MB, 4.5–11.9 GB/s random 11 MB | `scratchpad/mem/ssd2.py` on mincore-verified cold files | **[measured]**, shared machine |
| M5 Max SSD 12.6–13.6 GB/s read (Blackmagic / The Verge) | [PetaPixel](https://petapixel.com/2026/03/13/macbook-pro-with-m5-max-review-this-is-better-than-your-desktop/), [MacRumors review](https://macrumors.com/review/macbook-pro-m5-pro-m5-max), [MacDailyNews](https://macdailynews.com/2026/03/10/apples-new-m5-max-macbook-pro-delivers-a-huge-ssd-read-write-speed-boost/) | **[self-reported]** |
| Zero-copy `mx.asarray(memmap, copy=False)`; qmm exact on file-backed weights; `mx.array` copies | `scratchpad/mem/nocopy2.py`, `nocopy3.py`; mlx `python/src/convert.cpp`, `array.cpp` @ v0.32.4 | **[measured]**+[src] |
| `get_peak_memory` excludes the buffer cache (46 GB reported vs 110 GB footprint) | [mlx#3896](https://github.com/ml-explore/mlx/issues/3896) | [self-reported] + [src] allocator lines quoted in the thread |
| mlx-vlm offload mechanics (LRU, sync per layer, Python loop, `get_all` bypass, default budget) | `mlx_vlm/moe_offload.py`, `models/switch_layers.py`, `server/app.py:1187` | [src] |
| flash-moe: 4.36 tok/s for 397B on 48 GB M3 Max; 2.41 ms/layer `pread`; 71 % page-cache hit; custom caches slower; mmap 5× slower; prefetch DMA slowed GPU −73 %; temporal prediction 25 % | [github.com/danveloper/flash-moe](https://github.com/danveloper/flash-moe) | **[self-reported]** |
| Mixtral offloading: LRU expert cache + speculative loading, 2–3 tok/s on consumer GPUs | [Eliseev & Mazur, arXiv:2312.17238](https://arxiv.org/abs/2312.17238) | [paper] |
| Mixed-precision offloading: load low-precision copies of less-critical missed experts, up to 9.93× decode vs prior offloading | [HOBBIT, arXiv:2411.01433](https://arxiv.org/abs/2411.01433) | [paper], self-reported speedups |
| Dynamic resident expert subset under a tunable memory budget ("virtual experts") | [SwapMoE, ACL 2024, arXiv:2308.15030](https://arxiv.org/abs/2308.15030) | [paper]. Whether it routes only to resident experts was **not verified** from the full text |
| Qwen GSQ-RCO unpruned: 3.0 bpw → 99.4 % task retention, 3.5 bpw ≈ bf16 | *quant §3* ([ISTA](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)) | [self-reported], GGUF codes, not MLX affine |
| "Pruning error dominates quantization error at this width" (Qwen REAP-288 8-bit ≈ 4-bit) | *reap §3* (sh0wie Q8E) | [self-reported] |
| Local decode: Qwen REAP build ~29–33 tok/s plain, ~58–62 with MTP; peak 47.9 GB | `results/runtime_tune.json`, `logs/server.log` | **[measured]** on a busy GPU; the build is not recorded in the file |
| macOS pressure: compressor → swap → jetsam; watch swap-out and compressor *growth* | [chrischodev](https://blog.chrischodev.com/macos-memory-pressure-guide/), [mole.fit](https://mole.fit/blog/mac-memory-pressure-explained) | secondary |
| `kern.memorystatus_vm_pressure_level` = 1 now; `wired_max_proc_pct` = 25 | `sysctl` | **[measured]**. The 2 = warn / 4 = critical encoding is from XNU's `NOTE_MEMORYSTATUS_PRESSURE_*`, not observed here |

---

## 4. Plan for this machine

### 4.1 The new budget rule and code changes

**Rule:** a profile's cap applies to `F` (§2.1), with `W_res` = **resident** weight bytes. Disk size is tracked
separately against the SSD budget (2.5 TiB free).

1. **`serve/memory.py`.** Add `resident_weights_bytes(model_dir)`, as specified in *08-system §4.2a*.
   - Parse safetensors headers only.
   - Exclude tensors listed in `ple-store.json`, tensors under `experts/` of an offload dir, and tiers marked
     non-resident in a tiered build (§4.5).
   - Use it in `check_fits`. Today `weights_bytes()` sums every safetensors file, which would **refuse** the
     unpruned Qwen build: 60.5 + 32 GB on disk = 86 GiB > 80, even though only 56.4 GiB is resident.
2. **`serve/budget.py`** (new, pure functions, unit-testable on tiny fixtures):

   ```python
   KV = {"qwen": dict(per_tok=28_464, mtp_tok=2_372, fixed=110 << 20),
         "glm":  dict(per_tok=11_979, mtp_tok=1_089, fixed=141 << 20)}
   def kv_bytes(fam, ctx, mtp=True): ...
   def prefill_transient(fam, ctx, step=2048, glm_absorbed=False):
       if fam == "qwen": return 4 * step * (ctx // 4) * 4 + 300 << 20          # §2.1c
       t = 512 * 32 * (ctx // 4) * 4                                            # chunked indexer
       return t + (0 if glm_absorbed else 65_536 * ctx)                         # DSA re-projection
   def runtime_bytes(fam, ctx, apc_entries=1, cache=2 << 30, overhead=1.5 * GiB): ...
   def fits(profile, resident_w, fam, ctx) -> (bool, dict)                      # used by admission + /serve/status
   ```

   Validate against `footprint` in `evals.speed` (§5).
3. **`build/make.py`.**
   - Add `--budget-scope {disk,resident}` (default `disk`, for compatibility). With `resident`, `ple_table` units and
     non-resident tier units are excluded from `--budget-gb`.
   - Add `--tiers` (§4.5).
4. **`serve/config.py`.**
   - `expert_cache_gb: Optional[float]` → env `EXPERT_CACHE_GB`, read by mlx-vlm in `server/generation.py:234`.
   - `profile: str = "work"`.
   - Wired default 0 (*08-system §4.2d*).

### 4.2 Profiles and live switching

| profile | process cap (`--memory-limit-gb`, GiB) | left for macOS + you | when | watchdog thresholds |
|---|---|---|---|---|
| **Work** (default) | **80** | ≥ 48 GiB | coding, builds, Claude Code | warn at `memorystatus_level` < 30 |
| **Focus** | **92** | ≥ 36 GiB | editor + browser, no Docker or big builds; long RAG/reasoning sessions | warn < 25 |
| **All-in** | **104** (= `max_rec` 107.5 − 3.5) | ≥ 24 GiB | not coding (evenings, overnight agents) | warn < 20, auto-downshift on any warn |

This is the same structure as *08-system §4.1*: work = Work, all-in = All-in. Focus is the middle step. "stream" in
08 is a *mode* here, available in any profile.

**What a switch changes, cheapest first:**

1. **Runtime knobs, instant, no reload:** memory limit, `max_context` clamp, APC RAM budget (`APC_MEMORY_MAX_GB`),
   prefill step, MLX cache limit. mlx-vlm already supports some runtime changes via `PATCH /v1/settings`
   (*decode §1*). The rest go through our own endpoint.
2. **Expert residency, seconds, no restart (tiered builds only):**
   - **Upshift:** read the tier-W/C slabs (12.7 GB + 14.5 GB for GLM Focus/All-in) at ~6–12 GB/s, which takes
     **≈ 2–5 s**, then flip the router mask.
   - **Downshift:** flip the mask first, so no token can route to the experts, then drop the slab references and call
     `mx.clear_cache()`. Memory returns in **< 1 s** [est].
3. **Model swap, tens of seconds:** unload, then load a different build. 60 GB at ~6–12 GB/s is **≈ 6–15 s** plus
   warmup. Required between Qwen and GLM. Avoid doing it repeatedly in one process (*mac Rec 7*), and prefer a
   process restart through launchd for model swaps.

**API.** `POST /serve/profile {"name": "focus"}` returns the new limits and the predicted `F`.
`GET /serve/status` gains `{profile, resident_gb, disk_gb, F_pred, footprint_gb, pressure}`. A
`scripts/profile.sh work|focus|allin` wrapper calls it, or rewrites the launchd env (*08-system §4.1*) when a model
swap is needed. Switch only **between requests**: the engine already tracks `begin()`/`end()`.

### 4.3 The table: model × quant × streaming per profile, with speeds

Arithmetic: `W_res` from §2.1a; runtime from §2.1e at the stated context; total `F` in GiB. Decode speeds:

- "measured" means a sibling build measured on this machine.
- "est" means from §2.4c, or from Qwen's measured resident speed scaled by bytes, or from GLM's 78 tok/s ceiling
  (*mac §7*) at ≈ 50 %.
- MTP multipliers: Qwen measured ≈ 1.9×; GLM ≈ 1.4–1.8× at ~70 % draft acceptance (*reap §7*) [est].

All speeds are on an idle GPU.

| # | profile | build | W_res (GB → GiB) | ctx | F (GiB) | ≤ cap? | decode plain / MTP (tok/s) | quality expectation |
|---|---|---|---|---|---|---|---|---|
| W0 | Work | **current**: sh0wie Qwen REAP-288 4b g64 (PLE mmap) | ≈ 41 → 38.6 | 128K | ≈ 52 | ✓ (28 GiB idle) | **~30 / ~58 measured** (sibling build) | baseline: HumanEval 91.5 [self-rep.], rare names 9/10 |
| **W1** | **Work** | **Qwen 512 unpruned, sensitivity-allocated avg 3.6 bpw experts, trunk floors, MTP 512@4.5, PLE mmap** | 60.5 → 56.4 | 128K | **69.2** | ✓ (10.8 spare) | ~30 / ~55 est (latency-bound decode barely depends on bits, *quant §4*) | **no pruning loss.** GSQ-RCO unpruned 3.0 bpw kept 99.4 % [self-rep.]. MLX affine is cruder, so expect 97–99 % |
| W1′ | Work | Qwen 512, tiered 128@4b / 192@3b / 192@2b (3.375 bpw) | 57.1 → 53.2 | 128K | 66.1 | ✓ | same | routed-weighted error −43 % vs uniform at equal bytes (§2.5c, idealized) |
| W1″ | Work | Qwen 512, tiered 320@4b / 192@3b (4.125 bpw) | 68.5 → 63.8 | 128K | 76.7 | ✓ (tight) | same | near 4-bit everywhere, no pruning |
| W2 | Work | GLM tiered, R = 144 resident (tier H @3.0), 144 masked | 66.4 → 61.9 | 64K | 72.6 | ✓ | ~40 / ~60 est | = REAP-144 quality (*reap Rec 2*: validated ceiling, ×1.2–1.8 turns [self-rep.]) |
| W2′ | Work | same, 128K | 61.9 | 128K | 79.2 | ✗ until absorbed-query prefill (then 71.5) | | |
| W3 | Work | GLM 288 streamed: 144 resident @3.5 + 144 on SSD (stock offload, `--expert-cache-gb 55`) | ≈ 63 resident + 61 GB on SSD | 32K | ≈ 66 | ✓ | **~10–17** est (cold → warm cache), worse with the stock Python loop | **unpruned GLM at 3.5 bpw**: highest quality, slow; overnight only |
| F1 | Focus | Qwen 512 unpruned @4.5 avg | 74.1 → 69.0 | 128K | 81.9 | ✓ | ~30 / ~55 est | ≈ 4-bit unpruned. Beyond this Qwen gains little from bits (GSQ 3.5 bpw ≈ bf16) |
| F1′ | Focus | W1 + 256K context + 3 APC checkpoints | 56.4 | 256K | ≈ 84 | ✓ | same | spends RAM on **context and prefix cache**, often the better use for agents |
| **F2** | **Focus** | **GLM tiered, R = 176 (H 144@3.0 + W 32@3.0)**, 112 masked | 79.4 → 74.0 | 128K | 91.3 | ✓ (tight); 84.7 at 64K | ~38 / ~58 est | between REAP-144 and REAP-176. 32 more experts per layer is about +6 points of routing mass on Qwen's curve |
| F3 | Focus | GLM R = 176 resident + 112 @2.5 streamed (slab path, §4.5) | 74.0 + 37 GB on SSD | 64K | 84.7 | ✓ | **~15–23** est | unpruned GLM, cold tier at 2.5 bpw |
| **A1** | **All-in** | **GLM tiered, R = 216 (+ tier C 40@2.75)**, 72 masked | 94.3 → 87.9 | 64K | 98.6 | ✓ | ~36 / ~55 est | REAP-216 class (25 % pruned): GLM-4.6-REAP 25 % kept GPQA −2 [self-rep.] |
| A1′ | All-in | same, 128K | 87.9 | 128K | 105.2 | ✗ (1.2 over) until absorbed prefill (then 97) | | |
| **A2** | **All-in** | **GLM all 288: R = 216 resident + 72 @2.5 streamed (24 GB file, page cache fills free RAM)** | 87.9 + 24 GB file | 64K | 98.6 (+ page cache outside F) | ✓ | **~28–33** est | **unpruned GLM-5.3-Flash** at ≈ 3.0 bpw average: the best model this machine can run at interactive speed |
| A3 | All-in | Qwen 512 @5.0 + PLE pinned resident (nvfp4) | 110.5 → 102.9 | 128K | 115.8 | ✗ | – | pointless: pinning the PLE buys speed only if lookups miss the page cache |

How to read the table:

- **Work → W1.** An unpruned Qwen replaces REAP-288 at the same speed and ~17 GiB more memory. This is the
  highest-value, lowest-risk change in the whole plan. It is the same idea as *08-system §7.1*, at fewer bits
  (3.6 instead of 4.5) so that 128K context and one APC checkpoint fit with ~11 GiB margin.
- **When you are not coding → A2.** GLM-5.3-Flash with every expert available, 75 % of them resident at ≥ 2.75 bpw
  and the tail streamed at 2.5 bpw, decoding at an estimated ~30 tok/s. Use it for long reasoning and hard agentic
  tasks, after the per-profile quality gate in §5.
- **Focus** is the middle: GLM F2 if GLM wins the evals, otherwise Qwen F1′ (longer context, more APC).
- The GLM rows depend on two fixes from *decode*: the projected-cache drop (required) and absorbed-query prefill
  (optional, unlocks 128K in W2 and A1).

### 4.4 Building the candidates (commands)

These commands use real flags in `build/make.py`. **New** flags are marked ★ and are specified in §4.1/§4.5.

```bash
# W1: Qwen unpruned, resident budget 60.5 GB (PLE excluded), PLE nvfp4 to save 3.2 GB of disk
uv run python -m build.make --model qwen --src ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/<rev> \
  --calib artifacts/qwen/calib --prune none --budget-gb 60.5 --budget-scope resident ★ \
  --drop-vision --mtp intact --groups 32,64,128 --min-bits 3 --out models/qwen-fn-512-r60 --summary-json results/qwen-fn-512-r60/build.json
uv run python -m build.make ... --dry-run          # first: prints the allocation and predicted bytes, writes nothing

# W1′: tiered by routed importance (w_j from calib: freq · gate_mean² · out_sq)
uv run python -m build.make --model qwen ... --prune none --tiers "128:4,192:3,192:2" ★ --tier-score weighted ★ \
  --budget-scope resident ★ --out models/qwen-fn-512-tier

# GLM tiered master (one checkpoint for W2/F2/A1/A2): experts ordered by REAP-mean (mask policy) within the
# resident tiers and by frequency for the streamed tail
uv run python -m build.make --model glm --src ~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/<rev> \
  --calib artifacts/glm/calib --prune none --tiers "144:3,32:3,40:2.75,72:2.5" ★ --tier-order reap,reap,reap,freq ★ \
  --mtp prune --drop-vision --out models/glm-53-tiered

# Stock offload for hit-rate measurement (any build): repack, then serve with an explicit budget
uv run python -m mlx_vlm.moe_offload --build models/glm-53-tiered --out models/glm-53-tiered-offload
EXPERT_CACHE_GB=55 PREFILL_STEP_SIZE=2048 scripts/serve.sh models/glm-53-tiered-offload --memory-limit-gb 80 --max-context 32768
curl -s 127.0.0.1:8642/v1/moe-offload/stats | jq '{hit_rate, resident_bytes, misses, evictions}'
```

The `--tiers` spec is `count:bits` per tier, in hotness order. A bits value like 2.75 means the allocator picks a
(bits, group) mix within that tier averaging 2.75 bpw, e.g. 3b g128 plus 2b g64.

**Before any real build:** run every path on tiny fixtures (`tests/fixtures/make_tiny.py`, 8 experts). Load with
`mlx_vlm.load`, check bit-exactness against the untiered model with all tiers resident, and check equality with
`build.prune` slicing when tiers are masked (§5 test T2).

### 4.5 `serve/tiered.py`: tiered experts, masking, and the fast streaming path

**Checkpoint layout** (written by `build.make --tiers`):

```
models/glm-53-tiered/
  config.json            # + "expert_tiers": {"counts":[144,32,40,72], "bits":[...], "order_file":"tiers/order.json",
                         #                    "resident_by_profile":{"work":1,"focus":2,"allin":3}}
  pruned_moe.py          # model_file shim (reuses build/shim.py mechanism) → TieredSwitchGLU
  model-*.safetensors    # trunk + tier H stacked [144, out, in] per layer (normal mlx_vlm load)
  tiers/order.json       # per layer: original expert id for each tier slot (the permutation)
  tiers/W/layer_XXX.bin  # tier W slab, raw, 16 KiB-aligned tensors (weight|scales|biases per projection)
  tiers/C/layer_XXX.bin  # tier C slab
  tiers/T/layer_XXX.bin  # streamed tail, per-expert 16 KiB-aligned records (for pread)
```

**`TieredSwitchGLU`** (decode path):

- **Router.** Compute logits as usual, then add `mask_bias[E]`: 0 for resident experts, −∞ (GLM: on the selection
  bias only) for masked ones. Map the selected original ids through `perm[E] → (tier, slot)`.
- **Compute.** Per resident tier `t`, run `gather_qmm` on that tier's stacked tensor:
  - index `slot` for the token-slots routed into tier `t`;
  - index 0 with a **zero combine weight** for the others.

  This keeps shapes static: no CPU sync, no Python loop, one fused kernel per projection per tier. The cost is extra
  reads of slot 0 for the non-matching slots, which sit in cache after the first read [est: ≤ 1.3× MoE kernel time
  with 3 tiers; measure in `docs/research/quant_bench.py` style].
- **Profile change.** Set `mask_bias` and attach or detach tier slabs (§4.2).
- **Slab source.** `mx.asarray(np.memmap(tiers/W/layer.bin), copy=False)` makes tier W/C **file-backed** (§2.6),
  warmed once with parallel `pread`. `mx.load` copies give an anonymous slab. Default: anonymous for H, and the
  memmap option behind `--file-backed-tiers` until the §5 soak test passes.

**Streaming tail (`policy=stream`)**, replacing stock `OffloadedSwitchGLU` for speed:

- Keep a per-layer **staging slab** of `s` slots (e.g. 8) in the tail tier's format, as one stacked Metal buffer.
- After routing, `mx.eval` only the tiny `[K]` index array of tail hits. This is the single sync, and it is done
  only if `mask_tail_hits.any()` could be true, i.e. always for streamed layers.
- Issue a parallel `pread` of the missing experts into the slab slots via a thread pool, at QD = number of misses.
- Run `gather_qmm` on the slab.
- Keep an LFU over the last ~4,096 tokens per layer to choose victims (session locality, §2.5b). Record per-expert
  hit counts to `artifacts/<model>/live_freq.npz` for re-tiering.

**Hybrid (`policy=adaptive`, stretch):** mask on the current token, never stalling. If a masked expert would have
been selected (top-k of unmasked logits), its id goes into a small GPU array that is read **asynchronously**
(`mx.async_eval`, read after the token). The I/O thread swaps it into tier C, evicting LFU. Decode speed stays at
resident speed, and quality converges toward unpruned for the session's working set. This is close in spirit to
SwapMoE's "virtual experts" ([arXiv:2308.15030](https://arxiv.org/abs/2308.15030)).

### 4.6 The memory-pressure watchdog (`serve/watchdog.py`)

**Signals**, sampled every 1 s; each costs microseconds through `sysctlbyname` via ctypes, no subprocess:

| signal | source | meaning |
|---|---|---|
| `lvl` | `kern.memorystatus_vm_pressure_level` | 1 normal, 2 warn, 4 critical |
| `avail` | `kern.memorystatus_level` | percent of memory available (92 now) |
| `so_rate` | Δ `vm.swapusage` used + Δ swap-outs (`host_statistics64`) over 10 s | MB/s |
| `cmp_rate` | Δ compressor-occupied pages over 60 s | GiB/min |
| `F` | `footprint -p <self>` every 30 s (heavier), plus `mx.get_active_memory()+mx.get_cache_memory()` every 1 s | ground truth vs our counters |

**Ladder with hysteresis.** Escalate at once on the trigger. De-escalate only after 60 s fully back in "normal"
(`lvl` = 1 and `avail` ≥ threshold + 15).

| level | trigger (any) | action | cost to the model |
|---|---|---|---|
| L0 | – | none | – |
| L1 shed | `avail` < profile threshold, or `so_rate` > 50 MB/s, or `cmp_rate` > 1 GiB/min | move APC RAM checkpoints to the SSD tier; `mx.clear_cache()`; `set_cache_limit(0.5 GiB)`; clamp new requests' `max_context` to 32K | next turn may miss APC RAM (disk tier restores it) |
| L2 downshift | L1 for 30 s, or `lvl` = 2 | drop one expert tier (All-in → Focus → Work) via the mask-then-free sequence (§4.2); for non-tiered builds, refuse requests > 32K and queue | slightly lower quality (more experts masked); frees 12–15 GB per GLM tier |
| L3 yield | `lvl` = 4, or L2 for 60 s, or `avail` < 8 | finish the current token, abort the stream with an Anthropic `overloaded_error` (529) or HTTP 503 with `Retry-After: 60`, then `Engine.unload("pressure")` | model gone. Reload when back to L0 for 60 s (launchd keeps the process alive) |

Design rules:

- **The model always yields first.** The point is that the user's apps never swap because of the model.
- **Never wire** while the watchdog is active (*mac Rec 2*). Unwired weights under pressure get compressed, and the
  ladder exists to act *before* that happens.
- **Log every transition** to `logs/watchdog.jsonl` with the signal values and the before/after `F`. These logs are
  the data that tunes the thresholds.
- **Hook points.** `Engine.begin()/end()` for between-request actions, the existing idle thread for sampling, and
  the `MemoryGuard` for in-flight.

### 4.7 Personal calibration data for hot sets

Your Claude Code transcripts (`~/.claude/projects/*/*.jsonl`) are the closest thing to the real future workload.
They were not read for this chapter; the following is reasoning about how to use them.

- **Why they fit hot-set estimation especially well.** In an agentic loop most processed tokens are **prefill**:
  system prompt, tool schemas, file contents and tool outputs. Decode is a minority. The routing statistics of
  *prefilling* your real transcripts through the local model therefore approximate its real expert load closely,
  even though the assistant turns were written by Claude rather than the local model. For REAP *saliency* the
  generation side matters more; *reap §4* prefers self-generated data there.
- **How much is enough.** Frequency rankings converge fast: 8× more data changed only 8 % of a kept set (OpenMOSE)
  **[self-reported]**. Run a split-half stability check: top-R overlap ≥ 90 % between halves.
- **How.**
  - Add `calib_data` source `personal`: parse the jsonl locally, re-render each episode in the target chat
    template with native tool-call format (`calib_data/render.py`), and time-split it (train on older than
    2 weeks, evaluate on newer, *08-system §7.2*).
  - Run `calib.run` with the domain weights, e.g. `--domain-weights personal=2,code=1,agentic=1,reasoning=1,longctx=1`
    on `build.make`.
  - Tier ordering for the *streamed* tail uses `expert_freq.personal`. Masked tiers use pooled REAP with a
    `personal` floor.
- **Privacy.** Everything stays local. Strip secrets with a regex pass (API keys, tokens) before rendering. Keep
  `artifacts/*/data/personal*` out of git.
- **Online refresh.** `serve/tiered.py` logs live per-expert hits (`live_freq.npz`). A monthly `build.make --tiers`
  re-order **without re-quantizing** only permutes tier membership, i.e. a rewrite of slabs from the same
  quantized experts. This is cheap if every expert is stored at its tier's bits **and** at the tail bits; see the
  stretch idea in §7.

### 4.8 Order of work and time estimates

| step | what | memory | time [est] | gate |
|---|---|---|---|---|
| 1 | `serve/budget.py` + resident accounting + tests on fixtures | tiny | 0.5 day | unit tests: the formulas match *decode §3* bytes exactly |
| 2 | Validate accounting on W0 (load in the calibrate window): `footprint` vs predicted `F` at 1K/32K/128K | 52 GiB | 1 h | \|pred − footprint\| < 2 GiB |
| 3 | Watchdog L1/L3 (shed + yield) with `memory_pressure -S -l warn` simulation | – | 0.5 day | ladder fires, no swap growth during `memory_pressure -l warn` |
| 4 | Build W1. Source read ≈ 360 GB BF16 at ~6 GB/s ≈ 1 h of I/O, plus quantization | build peak bounded by `build/` streaming | 2–4 h | KL vs `ref/` < W0's KL; T1 gate in *08-system §4.8* |
| 5 | Promote W1 if it wins | 69 GiB | – | decode within 10 % of W0 |
| 6 | Stock offload on W1 with 384 resident (`EXPERT_CACHE_GB` = 384/512 of experts) to **measure real hit rates** vs §2.5's static curve | ≤ 80 GiB | 2 h | gives the first measured `h_session` |
| 7 | `TieredSwitchGLU` (mask + gather) on fixtures, then GLM tiered build (328 GB FP8 read ≈ 1 h + dequant/quant) | build < 40 GiB | 2–3 days | T2 bit-exactness; KL per R ∈ {144, 176, 216} |
| 8 | Profiles + live tier switching | | 1 day | switch < 5 s up, < 1 s down; `F` matches |
| 9 | Slab streaming path (A2/F3) | | 3–5 days | tok/s within 25 % of the §2.4c prediction |
| 10 | File-backed tiers soak (§2.6) | | 1 day + overnight | §5 T5 |

---

## 5. Decision criteria & what to measure

| ID | test | how | pass / decide |
|---|---|---|---|
| T1 | Accounting accuracy | `evals.speed --profile <p>` logs `footprint -p` (IOAccelerator, mapped file, phys_footprint), `get_active+get_cache`, predicted `F` at 1K/32K/128K, prefill step 2048/4096 | \|F_pred − footprint\| < 2 GiB, else fix `budget.py` constants |
| T2 | Masking = pruning | tiny fixture: tiered build with tail masked vs `build.prune` slicing to the same kept set | max \|Δlogit\| < 1e−5 (fp32) |
| T3 | Hit-rate curve, real | stock offload `/v1/moe-offload/stats` over a 1-hour replay of personal transcripts at resident fractions 0.5/0.75/0.875 | compare with §2.5 static curve; if `h_session` ≥ static + 0.05, streaming profiles become more attractive |
| T4 | Streaming decode speed | `evals.speed` on F3/A2 with cold (`sudo purge`) and warm page cache; record misses/token, ms/layer stall | within 25 % of §2.4c; else re-fit `ℓ`, `σ` |
| T5 | File-backed weights under pressure | adopt tier W via `mx.asarray(memmap, copy=False)`; decode loop + `memory_pressure -S -l critical` for 30 min | no GPU errors or panics; decode recovers within 10 s of pressure ending; footprint stays flat |
| T6 | Wiring cap reality | with wiring opt-in, wire 40 GiB and read `vm_stat` wired-down | if < 40 GiB, `wired_max_proc_pct` applies; record it in *mac* |
| T7 | Quality per profile | KL vs `ref/` + the *08-system §4.8* gate (code, tools, rare names, turns-to-solve) for W0, W1, W2, F2, A1, A2 | promote a profile's build only if it beats the profile below it on the gate. **Rule:** GLM-A2 replaces Qwen for "not coding" only if its gate score beats W1/F1 by more than the gate's CI |
| T8 | Co-tenant impact | run a fixed dev workload (Xcode build + `pytest` + browser) with the model serving Claude Code; log swap-outs, compressor, build time vs model-off | build time +≤ 10 %, zero sustained swap-out in Work |
| T9 | Watchdog | inject pressure (allocate 30 GiB in another process) | L1 within 2 s, L3 before swap-out > 1 GiB; no oscillation (≤ 1 transition / 60 s) |

**Decision rules:**

- **Bits vs experts.** At equal resident bytes, prefer the configuration with lower per-domain KL *and* better
  rare-name/tool probes. The W1 vs W0 comparison is the first real data point for "experts vs bits" on this model
  (*reap Rec 1* says our own sweep decides).
- **Mask vs stream for the tail.** Stream if T3 shows `h_session` ≥ 0.97 at the profile's R, which means < 25 %
  slowdown per §2.4c. Otherwise mask, and use adaptive mode if it is implemented.
- **When to spend RAM on context instead of weights.** For Qwen, beyond ~4 bpw average (F1) quality saturates
  (GSQ 3.5 bpw ≈ bf16 [self-rep.]). Spend further Focus/All-in bytes on 256K context and APC checkpoints (F1′).

---

## 6. Failure modes

| failure | symptom | cause | mitigation |
|---|---|---|---|
| **Weights compressed under pressure** | decode drops 5–50× while the user builds; compressor grows | unwired anonymous weights, footprint too close to the edge | watchdog L1/L2 before the compressor reaches weights; profile caps; file-backed tiers (they get dropped, not compressed) |
| **Silent over-cap** | footprint ≫ `get_peak_memory` | buffer cache not counted; peak = active only ([mlx#3896](https://github.com/ml-explore/mlx/issues/3896)) | admission on active + cache; `footprint` sampling; cache limit 2 GiB |
| **GLM long-prompt OOM** | +22–24 GiB at ≥ 32K, or an 8 GiB spike in prefill | projected-cache leak; DSA re-projection transient | *decode Rec 1* fix (required); budget the 64 KiB × ctx transient; absorbed-query prefill |
| **Stock offload default budget** | process grows to ~90 GB | `ExpertStore` default = 0.8·max_rec − resident | always pass `--expert-cache-gb` / `EXPERT_CACHE_GB` |
| **Offload prefill thrash** | each turn re-reads the whole expert set; TTFT of seconds | `get_all` bypasses the LRU on > 50 % touch | slab path; large chunks; mask-during-prefill |
| **Double residency** | page cache + Metal copy of the same experts | `pread` path fills the page cache | accept it (the page-cache copy is reclaimable); use file-backed slabs to remove the copy |
| **Router/tier mismatch** | gibberish or quality cliff after a profile switch | mask flipped after the slab was freed; permutation off by one | always mask first, then free; checksum `order.json` against the config; T2 in CI |
| **MTP/drafter mismatch with masking** | acceptance drops after a downshift | drafter routes to experts the trunk masked (drafter has its own experts) | the drafter's experts are independent (*reap §7*); masking the trunk is fine, since speculative decoding stays lossless |
| **APC checkpoint after a tier change** | restored state computed with different experts | prefix cache built under another mask | include the mask hash in the APC key (`model_fingerprint` + tier state) |
| **IOGPU panic** | kernel panic | wiring + concurrency + cache churn (*mac §3*) | no wiring, one sequence, one MLX process |
| **Watchdog flapping** | repeated load/unload | thresholds too close; reload pushes pressure up again | hysteresis (60 s, +15 points); reload only after L0 for 60 s; L2 before L3 |
| **Personal-data bias** | experts for languages or tools you rarely use get masked; failures when you use them | hot set fitted to your past | general floor domain (5–10 %, *reap Rec 3*); masked experts are reloadable (they stay on disk); adaptive mode |
| **F_NOCACHE assumptions** | page cache still fills | read-ahead ignores it [measured] | treat page cache as uncontrolled; measure warm and cold |
| **File-backed buffer evicted mid-kernel** | GPU timeout or stall (unknown) | untested OS behaviour | T5 soak before use; keep hot tier anonymous |

---

## 7. Stretch ideas (push limits)

1. **Session-adaptive expert residency ("adaptive" policy, §4.5).** Never stall, mask, and fetch asynchronously.
   Speed stays at resident speed while quality converges toward unpruned for the experts the current session
   actually uses. Measure KL against unpruned over a long session as a function of R. If R = 144 adaptive reaches
   R = 216's KL after a few thousand tokens, the Work profile gets near-All-in quality.
2. **Dual-precision tail ("HOBBIT on Apple").** Store every tail expert twice: 2-bit (fast to fetch, 7.9 MB for GLM)
   and 4-bit (14.2 MB). On a miss, fetch the 2-bit version immediately and the 4-bit one in the background, then
   upgrade if the expert stays hot. HOBBIT reports up to 9.9× decode over offloading baselines by loading
   low-precision copies of less-critical misses ([arXiv:2411.01433](https://arxiv.org/abs/2411.01433))
   **[self-reported]**. Disk cost: +24 GB for GLM's 72-expert tail, which is trivial against 2.5 TiB free.
3. **Zero-copy everything for instant profile switches.** If T5 passes, store the GLM master entirely as
   16 KiB-aligned slabs and adopt every tier with `mx.asarray(memmap, copy=False)`. A profile switch then needs no
   reads at all: tiers become "resident" when the router stops masking them and the page cache warms. The
   footprint the kernel sees is only KV + activations. The model becomes the ideal co-tenant: macOS reclaims its
   pages under pressure like any file cache, and it re-warms at SSD speed.
4. **Cross-layer prefetch with a learned 1-layer-ahead router.** flash-moe found token-ahead prediction poor
   (25 %), but **layer-ahead** prediction is a different problem: predict layer `l+1`'s experts from layer `l`'s
   hidden state. Pre-gated MoE does this by design. Train a tiny linear predictor per layer on `calib/` hidden states
   (already streamed layer by layer) and measure recall@16 for 10 true experts. At > 80 % recall the per-layer
   latency term in §2.4c can overlap with compute. flash-moe also reports prefetch DMA slowing the GPU by 73 %
   **[self-reported]**, so measure GPU contention.
5. **Re-tiering without re-quantizing.** Quantize each expert once at every candidate tier width (2/2.75/3/4 bits:
   disk ≈ 3× experts ≈ 300 GB for GLM, affordable). Promotion and demotion between tiers then become pure file
   copies, enabling weekly re-tiering from `live_freq.npz`.
6. **Prefill-time masking with decode-time completeness.** Mask the tail during prefill (prompt tokens, where
   errors are partly washed out by later layers and only affect the cache), and stream it for decode tokens. This
   removes the cold-set read per turn (§2.4d) while keeping exact decode routing. Measure KL on the decode positions
   only.
7. **KV-free context, for real.** With GLM at 1.73 GiB per 128K, a 1M-token context is ≈ 13 GiB of KV
   (`max_position_embeddings` = 1,048,576). It fits the All-in profile **if** the prefill transients are fixed
   (absorbed-query DSA prefill, indexer chunking over keys). For long-document RAG this may beat retrieval outright.
   The 64 KiB × ctx DSA transient (64 GiB at 1M) is the blocker to engineer away first.
8. **Two models sharing one trunk budget.** Qwen W1 (56 GiB) + GLM in R = 0 masked mode is not useful, but
   **Qwen W1 + a GLM drafter-only "oracle"** is not either. The realistic multi-model idea is the opposite: one
   process that holds Qwen W1 resident, plus GLM's tiers file-backed (idea 3). Route hard requests to GLM: page
   cache warms GLM's hot tier in seconds while Qwen's anonymous weights stay put. This needs ≥ 70 GiB of
   page-cache headroom, so it is All-in only, and it is the boldest configuration in this plan.

---

## 8. Sources

Local research (verified earlier): `docs/research/mac.md`, `decode.md`, `quant.md`, `reap.md`;
`docs/contracts/calib.md`; sibling chapters `docs/plan/04-pruning.md`, `docs/plan/08-system.md`.

Installed source read for this chapter:
- `.venv/.../mlx_vlm/moe_offload.py`
- `.venv/.../mlx_vlm/models/switch_layers.py` (`OffloadedSwitchGLU`)
- `.venv/.../mlx_vlm/utils.py:655-700`
- `.venv/.../mlx_vlm/generate/common.py:70-95`
- `.venv/.../mlx_vlm/server/{generation.py:234, app.py:1187}`
- `.venv/.../mlx_vlm/models/qwen4_exp/{language.py:990-1060, ple_storage.py}`
- `.venv/.../mlx_vlm/models/glm5_next/language.py:240-300`
- MLX `v0.32.4`: `mlx/backend/metal/allocator.cpp`, `python/src/convert.cpp`, `python/src/array.cpp`

Experiments (session scratchpad `…/scratchpad/mem/`):
- `skew.py`, `uniq.py`: routing skew and unique-expert counts from `saliency_full_505.npz`
- `ssd.py`, `ssd2.py`, `resid.py`: SSD bandwidth on mincore-verified cold files
- `nocopy.py`, `nocopy2.py`, `nocopy3.py`: zero-copy memmap adoption and file-backed `quantized_matmul`
- `calc.py`, `stream.py`, `tiers.py`: all tables in §2.1, §2.4, §2.5, §4.3

External:
- [flash-moe](https://github.com/danveloper/flash-moe) (Qwen3.5-397B on 48 GB M3 Max, SSD streaming, "trust the OS") [self-reported]
- [Eliseev & Mazur 2023, Fast Inference of MoE LMs with Offloading, arXiv:2312.17238](https://arxiv.org/abs/2312.17238)
- [HOBBIT: mixed-precision expert offloading, arXiv:2411.01433](https://arxiv.org/abs/2411.01433)
- [SwapMoE, arXiv:2308.15030 (ACL 2024)](https://arxiv.org/abs/2308.15030)
- [mlx#3896: get_peak_memory under-reports footprint](https://github.com/ml-explore/mlx/issues/3896)
- M5 Max SSD: [PetaPixel review](https://petapixel.com/2026/03/13/macbook-pro-with-m5-max-review-this-is-better-than-your-desktop/), [MacRumors review](https://macrumors.com/review/macbook-pro-m5-pro-m5-max), [MacDailyNews](https://macdailynews.com/2026/03/10/apples-new-m5-max-macbook-pro-delivers-a-huge-ssd-read-write-speed-boost/)
- macOS memory pressure: [chrischodev guide](https://blog.chrischodev.com/macos-memory-pressure-guide/), [mole.fit](https://mole.fit/blog/mac-memory-pressure-explained), [dev.to 19 GB compression report](https://dev.to/sleepyquant/what-19-gb-of-memory-compression-taught-me-about-mlx-on-m1-max-3eha)
- Model configs: `zai-org/GLM-5.3-Flash`, `Qwen/Qwen3.8-Flash-Next` (local HF cache), `mistralai/Mistral-Medium-3.5-128B`, `Qwen/Qwen3.8-27B` (`config.json` via HF)
