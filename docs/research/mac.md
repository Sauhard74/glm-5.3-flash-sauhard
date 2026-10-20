# macOS / M5 Max memory and performance controls for a co-tenant MLX server

Scope: running a ~70 GB MLX model on this machine (MacBook Pro 16" `Mac17,6`, M5 Max 40-core GPU, 128 GB,
macOS 27.0.1 build 26A434, 140 W adapter) with the whole server capped at ~80 GB. MLX 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.9.

Legend: **[src]** read in the installed package or upstream source at tag `v0.32.4`; **[measured]** run on this machine
with small arrays (<= 4 GB); **[self-reported]** numbers from issues, blogs or forums, not reproduced here.

## 1. What this machine reports

`mx.device_info()` **[measured]**:

| key | value |
|---|---|
| `architecture` | `applegpu_g17s` (gen 17, `s` = Max) |
| `memory_size` | 137,438,953,472 (128 GiB) |
| `max_recommended_working_set_size` | 115,448,725,504 (**107.5 GiB, 84 % of RAM**) |
| `max_buffer_length` | 86,586,540,032 (80.6 GiB, the largest single `MTLBuffer`) |
| `resource_limit` | 499,000 buffers |

`sysctl iogpu` **[measured]**: `wired_limit_mb: 0`, `wired_lwm_mb: 0`, `dynamic_lwm: 1`, `disable_wired_collector: 0`.
Swap is 3 GB total with 1.6 GB used. About 69 GB is "inactive", mostly page cache from the running downloads, which the
kernel can reclaim.

With `wired_limit_mb=0` the kernel picks the limit itself. On this macOS 27 / 128 GB machine it resolves to 107.5 GiB.
Community guides quote "~75 %, i.e. ~96 GB on 128 GB"
([devnote](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap),
[ModelPiper](https://modelpiper.com/blog/iogpu-wired-limit-mb-mac), which measured 78 % on a 32 GB M2 Max)
**[self-reported]**. That figure is out of date here. Read the value from the device and never hard-code it.

## 2. MLX 0.32.4 memory knobs: what they actually do

The docstrings undersell some of these. Each behavior below was confirmed in
[`allocator.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/allocator.cpp),
[`transforms.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/transforms.cpp) and
[`resident.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/resident.cpp) **[src]**.

### `mx.set_memory_limit(bytes)`: a soft limit, not a hard cap
- Default is `min(1.5 × max_rec, 0.95 × RAM)`, which is **121.6 GiB** here **[measured]**. The docstring says only
  "1.5× recommended", which is incomplete.
- It does two things:
  1. During `eval`, if `active > limit` and tasks are in flight, the scheduler waits for tasks to finish before it
     encodes more work (`transforms.cpp`). This throttles; it does not refuse.
  2. It sets `gc_limit = min(limit, 0.95 × max_rec)`. When `active + cache + new >= gc_limit`, `malloc` first releases
     cached free buffers.
- **It never makes an allocation fail.** An exception happens only when Metal itself returns nil, i.e. the OS is out of
  RAM and swap, when one buffer exceeds `max_buffer_length`, or when the live buffer count reaches `resource_limit`
  (499,000; `allocator.cpp` throws "Resource limit exceeded"). **[measured]**: with the limit at 1 GiB, allocating
  6 GiB succeeded with no error. `taskpolicy -m 2048` (a per-process memory limit) did not stop a 4 GiB MLX
  allocation either, so macOS offers no simple hard cap here.
- Consequence: PLAN.md's "enforced cap (`mx.set_memory_limit`)" is wrong. **The 80 GB cap has to be enforced by the
  server**, by bounding KV size, prefill chunk size, concurrency and prompt-cache bytes.

### `mx.set_cache_limit(bytes)`: limits how much freed memory MLX keeps
- This is the pool of freed `MTLBuffer`s that MLX keeps for reuse. It defaults to the memory limit, so 121.6 GiB.
  Freed buffers stay in the process footprint until they are reused, until the cache exceeds the limit, or until
  `mx.clear_cache()` runs.
- With the default, a 128K prefill can leave many GB of freed activations held by the process even though nothing uses
  them. A small cap (2–4 GB) lets go of that memory with only a minor cost from re-allocation. mlx-vlm already calls
  `mx.clear_cache()` between requests in several places.

### `mx.set_wired_limit(bytes)`: keeps memory resident
- Default is `0`, so MLX puts nothing in a residency set. ~~nothing is wired~~ That does not mean nothing is wired:
  **[measured]** with the limit at 0, a 3 GiB working set still raised `vm_stat` "wired down" by ~3 GiB while the GPU
  was busy. The driver makes in-use buffers resident on its own, and drops them within seconds of idle. Values above `max_rec` raise
  `ValueError: Setting a wired limit larger than the maximum working set size is not allowed` **[measured]**.
- Mechanism (`resident.h`): every allocation is tracked. Allocations are added to `MTLResidencySet`s, each with a
  standing `requestResidency()`, until the running total reaches the wired limit. Anything past the limit stays
  tracked but not wired. Each set is capped at `MLX_RESIDENCY_SET_MAX_PCT` (default **5 %** of `max_rec`, about
  5.4 GiB here), with at most 32 sets. Lowering the limit unwires allocations until the total fits again.
- **Correction (wiring order):** an allocation is only wired *at the moment it is allocated*, and only if it fits.
  When the limit is raised later (`resize()`), MLX walks an `std::unordered_map` keyed by buffer pointer, so the order
  is arbitrary, not allocation order. Freed buffers sitting in MLX's cache are still tracked and **keep their wired
  budget** until they are actually released. **[src]** So "weights are allocated first, so they get wired first" is
  only true if nothing else is tracked when the limit is set: call `mx.clear_cache()` first. Lowering the limit
  also unwires in arbitrary order, so it can drop weights and keep KV.
- Every insert or erase of a wired buffer calls `addAllocation`/`removeAllocation` plus `commit()` on a residency
  set. That per-buffer churn is what a controlled user report ties to the IOGPU kernel panic (section 3).
- **mlx-lm and mlx-vlm override whatever you set.** `mlx_vlm.generate.common.wired_limit()` and
  `mlx_lm.generate.wired_limit()` call `mx.set_wired_limit(max_rec)`, i.e. 107.5 GiB. mlx-vlm's batch generator
  enters that context in `__init__` (`mlx_vlm/generate/ar.py:2594`), and `mlx_lm.server` calls
  `maybe_set_recommended_wired_limit()` at startup. Left as is, everything the server allocates, KV included, can be
  wired up to 107.5 GiB. **[src]**
- **Correction:** `mlx_vlm.generate.dispatch.stream_generate` also enters `wired_limit` (`dispatch.py:1055`), as do
  the audio, diffusion and sam3 paths. Each module did `from .common import wired_limit`, so replacing
  `common.wired_limit` changes none of them. Also, `mlx_vlm.generate` resolves to the `generate` *function* on
  attribute access, so `mlx_vlm.generate.ar.wired_limit = ...` raises `AttributeError`; you have to go through
  `sys.modules["mlx_vlm.generate.ar"]`. **[measured]** The simplest override that covers every call site
  (mlx-vlm, mlx-lm and the trainers) is to wrap `mlx.core.set_wired_limit` itself, at module level, before the
  generator is built.

### Weights after loading: copied, not mmap'd
- **[measured]** (`footprint`/`vmmap` after `mx.load` of a 1 GiB safetensors): +1.0 GiB in `IOAccelerator (graphics)`,
  i.e. Metal buffers counted in the process's physical footprint. `mapped file` stayed at 77 MB.
- So the weights are anonymous dirty memory, not clean file pages. The kernel cannot simply drop them under pressure;
  it has to compress them or swap them out, then decompress or swap them back in on the next token.
- The read also fills the page cache with the file once. That memory is clean and reclaimable, but for a few minutes it
  competes with the downloads' cache.
- The mlx-vlm `moe_offload` `ExpertStore` (per-expert mmap'd files plus an LRU) is the only path that uses file-backed
  weights. We do not need it at 70 GB.

## 3. Memory pressure: wired vs. unwired weights

| | Unwired (default 0) | Wired (`set_wired_limit`) |
|---|---|---|
| Under pressure | Weights get compressed or swapped. Decode collapses: one token touches every active expert's pages. | Weights are not paged. Other apps get compressed or swapped instead. |
| Idle | Compressor may squeeze idle weights. Report on M1 Max 64 GB: 19.7 GB compressed while a 35B model sat idle ([dev.to](https://dev.to/sleepyquant/what-19-gb-of-memory-compression-taught-me-about-mlx-on-m1-max-3eha)) **[self-reported]** | macOS drops residency about 2 s after the GPU goes idle anyway. The next request pays a re-residency cost ([mlx#4609](https://github.com/ml-explore/mlx/issues/4609): 342 ms vs 58 ms first compute, from a run printing `mlx 0.32.3`, which the reporter's table maps to a MacBook Air M2 24 GB on macOS 27.2, ~~M4 Pro, macOS 27.0~~; 9–12 s TTFT after 12 min idle on a 16 GB M4 mini, macOS 26.7) **[self-reported]**. **Reproduced here [measured]:** 3 GiB with `set_wired_limit(4 GiB)`; after 8 s idle the extra wired memory fell to ~0, the same as with no limit, and the first compute took 37–62 ms against 6–7 ms warm. With `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` it stayed at ~2.9 GiB and the first compute took 9–10 ms. **So without the refresh variable, wiring does nothing once the GPU is idle.** |
| Failure mode | Slowness, swap growth. | IOGPU kernel panics (`completeMemory() prepare count underflow`): [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) (96 GB M3 Ultra, 80 GB wired, unbounded KV), [mlx#3346](https://github.com/ml-explore/mlx/issues/3346) (more likely above ~80 % GPU memory use, with repeated load/unload, with multiple MLX processes; closed by the maintainer and merged into [mlx#3186](https://github.com/ml-explore/mlx/issues/3186), which is still **open**). **New evidence, which changes the picture:** in #3186 one user ran a single-variable test (M4 32 GB, macOS 26.4.1, mlx 0.31.2). Two concurrent request streams plus prompt-cache eviction churn panicked in about 100 s, 3 of 3 times. With `set_wired_limit` never called it ran 10 of 10 runs (4.2 h) with no panic. Lowering the wired cap or using RotatingKVCache did not panic but gave a clean `kIOGPUCommandBufferCallbackErrorOutOfMemory` instead. Skipping wiring cost ≤ 3 % throughput. A second user (M3 Ultra, macOS 27.0 beta, IOGPUFamily 162.11) has run without wiring for days with no panic, but has not tested with wiring on macOS 27. The original #3186 report is a panic during a ~173K-token prefill. **[self-reported]** |

The idle-unwire problem is fixed in MLX 0.32.4 by
`MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS` ([PR #4633](https://github.com/ml-explore/mlx/pull/4633), docs in
[environment_variables.rst](https://github.com/ml-explore/mlx/blob/v0.32.4/docs/src/usage/environment_variables.rst)).
It defaults to `0` (off). When set, a thread re-calls `requestResidency()` on every non-empty set at that interval. It
submits no GPU work. A maintainer calls it a temporary fix for a macOS bug. The other workaround,
`sudo sysctl iogpu.disable_wired_collector=1`, applies to the whole system. Reports on it disagree: the #4609
reporter says it kept 37.3 GiB wired on macOS 27.0, and a later commenter says it did not help on macOS 26.7. It also
works against a co-tenant setup. Do not use it.

## 4. `iogpu.wired_limit_mb`

- `0` means automatic, which is 107.5 GiB here (section 1). What was measured is Metal's
  `recommendedMaxWorkingSetSize`. That it equals the kernel's effective wired cap is inferred from community reports,
  not measured; testing it needs sudo. It resets on reboot, and `/etc/sysctl.conf` is ignored;
  making it persistent takes a LaunchDaemon.
- **Raising it:** not needed. 80 GB is below 107.5 GB. Raising it only lets something wire more memory, the opposite
  of what we want. mlx-lm's README suggests it only for models that are close to RAM size
  ([mlx-lm README, "Large models"](https://github.com/ml-explore/mlx-lm#large-models)).
- **Lowering it** (e.g. `sudo sysctl iogpu.wired_limit_mb=84000`) would add a system-wide backstop that limits every
  process's wired GPU memory, including mlx-vlm's automatic `max_rec` wiring. Community reports say it moves
  `recommendedMaxWorkingSetSize` with it **[self-reported]**. That would also lower MLX's derived limits
  (`gc_limit`, the default memory limit). It needs sudo and is global, so treat it as optional and do not make it the
  primary mechanism.

## 5. GPU sharing, QoS, power, thermals

- **No GPU priority control.** Metal exposes priority only for IO queues (`MTLIOPriority`). Compute and render queues
  run in submission order, and nothing documented ties thread QoS or `taskpolicy` to GPU scheduling
  ([MTLCommandQueue](https://developer.apple.com/documentation/metal/mtlcommandqueue),
  [MTLIOPriority](https://developer.apple.com/documentation/metal/mtliopriority)). `taskpolicy -b` / background QoS
  would only throttle the server's CPU and disk I/O (tokenizer, HTTP, loading).
- What helps instead (inference, not measured here): keep each command buffer short so WindowServer and IDE GPU work
  can run between them. That means a bounded prefill chunk (`prefill_step_size` 2048) and MLX's per-buffer limits;
  for the Max tier `device.cpp` defaults to 50 ops / 50 MB per command buffer (`MLX_MAX_OPS_PER_BUFFER`,
  `MLX_MAX_MB_PER_BUFFER`).
- **Energy mode:** this machine is on Automatic (`pmset powermode 0`, High Power: No, Low Power: No), on AC with a
  140 W adapter. On the 16" model, High Power Mode allows higher sustained clocks and fan speeds. Prefill (compute-bound,
  uses the neural accelerators) gains the most; decode is bandwidth-bound and gains little. Apple's M5 vs M4 numbers:
  3.3–4.1× TTFT but only 1.19–1.27× decode
  ([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). These compare a *base* M5
  with an M4, both 24 GB, not the Max. Apple attributes the decode gain to bandwidth (153 vs 120 GB/s), so they say
  nothing about High Power Mode. On the 14" M5 Max,
  Notebookcheck reports throttling and a ~96 W draw, and owners report ~10–15 % lower sustained performance than the
  16" ([MacRumors](https://forums.macrumors.com/threads/14-m5-max-users-local-ml-ai-workload-full-load-experiences.2481239/))
  **[self-reported]**. One Apple forums report says High Power was not applied after Migration Assistant
  ([dev forums](https://developer.apple.com/forums/thread/835131)); check with
  `system_profiler SPPowerDataType | grep "High Power"`.
- **Low Power Mode** lowers GPU and CPU clocks. Avoid it for benchmarks; it is acceptable on battery.
- Monitoring: `pmset -g therm` (warning levels; none recorded so far), `sudo powermetrics --samplers gpu_power`,
  `memory_pressure`, `vm_stat`, `sysctl vm.swapusage`, `footprint -p <pid>`.
- Keep the machine awake during long runs with `caffeinate -i -s` (idle sleep is already held off by other caffeinate
  processes, but do not rely on that).

## 6. M5-specific MLX flags

- **Neural accelerators (NAX):** turned on automatically. `is_nax_available()` requires macOS ≥ 26.2 and arch
  generation ≥ 17 (`device.cpp`). `applegpu_g17s` passes, so **no flag is needed**. MLX targets them through Metal 4
  TensorOps / Metal Performance Primitives ([Apple](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)).
  0.32.4 also enabled the non-transposed NAX `gather_qmm_rhs`, the MoE expert matmul
  ([release notes](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)). Do not set `MLX_METAL_GPU_ARCH`; a wrong
  value picks incompatible kernels.
- Other env vars in the 0.32.4 binary (from `strings libmlx.dylib`), with defaults read from `mlx/utils.h` **[src]**:
  - `MLX_METAL_FAST_SYNCH=0`. A faster CPU/GPU sync path, which helps a decode loop that calls `.item()` each token.
    A microbenchmark here was too noisy because the GPU is shared (1.08–1.45 ms/step either way), so enable it only if
    `evals.speed` shows a gain.
  - `MLX_ENABLE_TF32=1`. Affects only fp32 matmuls; irrelevant for bf16 or quantized models.
  - `MLX_RESIDENCY_SET_MAX_PCT=5`. Leave as is.
  - `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`. Tune only in `evals.speed`.
  - `MLX_SDPA_*` and `MLX_CONV_*`. Kernel tuning; not relevant here.

## 7. Throughput ceilings (napkin math, 614 GB/s)

Decode reads the active weights once per token:
- GLM-5.3-Flash: ~18 B active × ~3.5 bpw ≈ 7.9 GB/token, so ≤ ~78 tok/s.
- Qwen3.8-Flash-Next: ~6 B active ≈ 2.6 GB/token, so ≤ ~235 tok/s.

Real decode usually reaches 50–70 % of these. MTP speculation multiplies by the acceptance rate.
Cross-check from the real `config.json` files: Qwen comes to ≈ 5.9 B active (11 × 3 × 2560 × 640 MoE per layer × 48,
plus attention, linear attention and the LM head). GLM's routed and shared experts, dense layers, MLA and LM head come
to ≈ 16 B before the indexer and hc/mHC extras, which is consistent with "~18 B". These ceilings ignore KV and
linear-state reads, which matter at 128K, and assume a uniform 3.5 bpw.

For reference, Ollama's MLX backend reports 112–134 tok/s for a 35B-A3B on an M5 Max
([lilting.ch summary](https://lilting.ch/en/articles/ollama-mlx-apple-silicon-preview)) **[self-reported]**.

## Recommendations for this project

Budget: weights ≤ 70 GB, KV/state ≤ 8 GB, process ≤ 80 GB, ≥ 45 GB left for the user.

1. **Set limits once, after load, in `serve/` (and `evals/`)**, sized from the real weight bytes:
   ```python
   GB = 1 << 30
   weights = tree_reduce(lambda a, x: a + x.nbytes if isinstance(x, mx.array) else a, model, 0)
   cap = 80 * GB
   mx.set_memory_limit(cap - 4 * GB)    # 76 GB: eval throttles above this and the cache gc starts
   mx.set_cache_limit(2 * GB)           # do not keep freed prefill activations
   mx.clear_cache()                     # cached buffers would otherwise compete for the wired budget
   wired = weights if WIRE else 0       # see 2: weights only, exactly, or nothing
   mx.set_wired_limit(min(wired, info["max_recommended_working_set_size"] - 8 * GB))
   ```
   ~~`set_wired_limit(min(weights + 8 GB, ...))`. The wired limit counts weights first, because they are allocated
   first.~~ Refuted: when the limit is raised after load, the order is arbitrary, and cached buffers hold wired budget
   (section 2). A limit of exactly `weights` after `clear_cache()` wires all the weights and leaves the budget full,
   so later KV and activation buffers are never added to a residency set. That keeps residency-set churn close to
   zero (see 2).
2. **Make wiring opt-in, default off, until a soak test passes on this macOS.** The only controlled evidence (mlx#3186)
   points to `set_wired_limit`'s per-buffer residency-set churn, together with concurrency and prompt-cache eviction,
   as the trigger for the IOGPU panic. Turning wiring off cost ≤ 3 % throughput there **[self-reported]**. Wiring
   also gives nothing while idle unless the refresh variable is set (section 3, **[measured]**). Override every
   call site in one place, before the generator is built:
   `_set = mx.set_wired_limit; mx.set_wired_limit = lambda n: _set(min(n, OURS))` (or a no-op when `OURS == 0`).
   ~~Replacing `ar.wired_limit` and `common.wired_limit`~~ misses `dispatch.stream_generate` and the other modes
   (section 2). `serve/memory.py` currently wires `weights + kv_reserve` and does not stop mlx-vlm from re-wiring to
   `max_rec`. If wiring is turned on (`WIRE=1`), use weights-only plus the refresh variable, and A/B it in
   `evals.speed` under memory pressure.
3. **Enforce the 80 GB cap in the server**, because `set_memory_limit` cannot. Pre-allocate or bound KV for 128K,
   **one concurrent sequence by default** (concurrency plus cache churn is the reported panic trigger), chunked
   prefill (`prefill_step_size=2048`), an LRU byte cap on the prompt cache (APC) of ≤ 4 GB in RAM with the rest on
   disk, and per-request admission:
   `if mx.get_active_memory() + mx.get_cache_memory() + est_kv(req) > cap: 503 / queue`. Include the cache:
   `get_active_memory()` leaves out freed-but-cached buffers, which are still in the process footprint. Log
   `get_peak_memory()` per request. `evals.speed` must report the peak at 128K.
4. **Env for `scripts/serve.sh`:** `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` **only when wiring is on.** It has
   no effect while the residency sets are empty, but it keeps wired memory pinned through idle by design. Leave
   every other MLX var at its default; try `MLX_METAL_FAST_SYNCH=1` only behind an `evals.speed` A/B.
5. **Idle policy (only when wiring is on):** after N minutes with no requests (default 10), call `mx.clear_cache()`
   and then `mx.set_wired_limit(0)`. That empties the sets, so the refresh thread does nothing. To re-wire on the next
   request, call `clear_cache()` again before `set_wired_limit(weights)` so that only live arrays compete for the
   budget. Expect a one-time TTFT hit: ~0.3 s without pressure and ~2 s with pressure, for a 35 GB model on an M5
   Pro (PR #4633, **[self-reported]**); likely more at 70 GB. Each switch is a bulk `resize()` with one commit per
   set, not per-buffer churn.
6. **Leave `iogpu.wired_limit_mb=0`.** Never raise it. Do not set `iogpu.disable_wired_collector`. Optional backstop
   for long unattended runs: `sudo sysctl iogpu.wired_limit_mb=84000` (resets on reboot; its effect on
   `max_rec` is **[self-reported]**).
7. **Do not chase full-RAM wiring.** Keep total GPU memory below ~65 % of RAM (80/128). Run one MLX model process at a
   time, and avoid repeated load/unload in one process (evals should load once per suite). Caveat: the "> ~80 %"
   threshold comes from one anecdotal list. The controlled report panicked without memory pressure, so staying under
   65 % alone is not a defense; items 2 and 3 matter more.
8. **Power:** for benchmark and calibration runs, plug in and set Energy Mode to High Power (verify with
   `system_profiler SPPowerDataType`). Record `pmset -g therm` and energy mode in `results/*/speed.json`. Day to day,
   Automatic is fine; decode cost from it is small. Never benchmark in Low Power Mode or on battery.
9. **Loading:** load once (`mx.load` copies into Metal buffers and touches ~70 GB of page cache). Prefer starting the
   server when the downloads are done or paused, and expect a transient rise in memory pressure during load.
10. **No new dependencies needed.** Everything above is in `mlx` 0.32.4 and macOS tools.

## Verification

Adversarial pass on 2026-10-20 against mlx 0.32.4 (source at tag `v0.32.4`, fetched from GitHub), installed
mlx-vlm 0.7.9 and mlx-lm 0.32.2, the GitHub threads cited above, and small experiments on this machine (≤ 6 GiB,
scripts in the session scratchpad). Verdicts:

| # | Claim | Verdict | Notes |
|---|---|---|---|
| 1 | `set_memory_limit` is soft; the server must enforce the 80 GB cap | **Verified, two corrections** | `transforms.cpp` waits only while tasks are in flight; `allocator.cpp` sets `gc_limit` and frees cache only. Default limit measured at 121.6 GiB. A 1 GiB limit did not stop 6 GiB of allocations, and neither did `taskpolicy -m`. Corrections: malloc also throws at `resource_limit` (499,000 buffers), and admission has to add `get_cache_memory()`. |
| 2 | mlx-vlm and mlx-lm override our wired limit; patch `ar`/`common.wired_limit` | **Partly refuted** | The override is real (`common.py:266`, `ar.py:2594`, `mlx_lm/server.py:2080`, `utils.py:83`). But patching `common` does nothing to modules that already imported the name, `dispatch.stream_generate` is missed, and `mlx_vlm.generate.ar` cannot be reached by attribute access. Wrapping `mx.set_wired_limit` at module level was tested and covers mlx-vlm and mlx-lm. |
| 3 | Limits `76 / 2 / min(weights+8, 76, max_rec-8)`; weights get wired first | **Partly refuted** | The ValueError above `max_rec` and the cache default (= memory limit) are verified. "Weights first" is wrong when the limit is set after load: `resize()` walks an `unordered_map` in arbitrary order, and cached buffers hold wired budget. Changed to `clear_cache()`, then exactly `weights`, opt-in. |
| 4 | `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` plus an idle unwire policy | **Verified, made conditional** | PR #4633 is merged and in v0.32.4 (release notes, `resident.cpp`, dylib strings). Reproduced here: without refresh, wiring is lost within 8 s of idle and the first compute is 5–9× slower; with refresh it stays wired. The 342/58 ms figure is misattributed in the original (M2 Air on macOS 27.2, not M4 Pro on 27.0). Only useful if wiring is on (item 2). |
| 5 | Leave `iogpu.wired_limit_mb=0` (107.5 GiB, 84 %); optional 84000 backstop | **Verified, one correction** | `max_rec` and the sysctls were measured. That `max_rec` equals the kernel wired cap is inferred. The mlx-lm README text matches. The `disable_wired_collector` evidence is mixed: it worked on macOS 27.0 in #4609 and did not on 26.7. |
| 6 | Stay below ~80 % GPU memory to avoid IOGPU panics | **Weakened** | Threads cited correctly; #3346 was closed into #3186, which is open. A controlled report in #3186 ties the panic to wiring plus concurrency plus cache churn, not memory level. Recommendations changed: wiring off by default, one sequence. |
| 7 | `mx.load` copies weights into Metal buffers; ~70 GB of page cache | **Verified** | Reproduced: `IOAccelerator (graphics)` grew by 1024 MB and `mapped file` was unchanged. `mlx/io/load.cpp` uses `pread` without `F_NOCACHE`. |
| 8 | NAX automatic; env defaults | **Verified, caveat** | `is_nax_available()`: macOS ≥ 26.2 and gen ≥ 17 (≥ 18 for `p`). Defaults confirmed in `utils.h`; ops/MB per buffer (50/50 for `s`) come from `device.cpp`, not `utils.h`. The v0.32.4 notes list #4684, the NAX `gather_qmm_rhs`. Apple's numbers are base M5 vs M4 at 24 GB, not Max. |
| 9 | No GPU priority control; keep command buffers short | **Not refuted (unverifiable)** | `man setpriority` and `taskpolicy`: background state throttles CPU scheduling, disk and network only, and no GPU option exists. The idea that short buffers help interleaving remains unmeasured reasoning. |
| 10 | Power and decode ceilings | **Verified** | `Mac17,6`, 3456×2234, 140 W, High/Low Power both No, `powermode 0`, no thermal warnings. 614 GB/s matches published specs. The ceiling arithmetic checks out (7.9 GB → 78 tok/s, 2.6 GB → 235 tok/s); active counts are consistent with the configs. |
| 11 | No new dependencies | **Verified** | All APIs exist in mlx 0.32.4 (`mx.*`, not only `mx.metal.*`); the override needs nothing new. |
