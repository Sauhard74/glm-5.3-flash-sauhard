# 08 — The complete local system: serving, tools, RAG, evaluation, operations

Earlier chapters produce a model directory: calibration, REAP pruning, mixed-precision quantization, MTP head. This
chapter covers what turns that directory into a system you can rely on every day on this MacBook Pro M5 Max (128 GB,
614 GB/s, macOS 27.0.1):

- one memory-capped server that speaks the **Anthropic** and **OpenAI** APIs;
- **Claude Code** (`ANTHROPIC_BASE_URL`) and **OpenCode** pointed at it;
- a **router** that picks how hard to think and which model or profile to use;
- a **RAG stack** for documents longer than the context window, or too long to prefill cheaply;
- **tool calls** that parse every time;
- **observability**, so every regression shows up as a number;
- an **evaluation harness** that gates every new build;
- **operations**: launchd, idle exit, memory-pressure watchdog, power and thermals, 1.5 TB of checkpoints;
- **security**.

It builds on the verified findings in `docs/research/{mac,decode,quant,reap}.md` and does not repeat their evidence.
They are cited as *mac §3*, *decode §5* and so on.

Tags used throughout:

| Tag | Meaning |
|---|---|
| **[measured]** | Run on this machine during this session. Metadata only for the transcripts; safetensors *headers* only for the models. No weights loaded. |
| **[src]** | Read in the installed package source (`.venv/lib/python3.12/site-packages/...`). |
| **[research]** | Verified finding in `docs/research/*.md`. |
| **[self-reported]** | A vendor or community number not reproduced here. |
| **[computed]** | Arithmetic shown inline. |
| **[assumption]** | A placeholder that a named measurement must replace. |

Units: **GiB = 2³⁰ bytes**, the unit of `serve/` and `mx.get_*_memory()`. GB = 10⁹, used only when a source uses it.

---

## Contents

0. [System at a glance](#0-system-at-a-glance)
1. [Why it matters (intuition)](#1-why-it-matters-intuition)
2. [How it works (mechanics + key math)](#2-how-it-works-mechanics--key-math)
   - 2.1 Memory: what actually occupies RAM
   - 2.2 Prefill, decode and prefix caching (the three costs)
   - 2.3 Speculative decoding math (what the acceptance log tells you)
   - 2.4 Routing: why the requested model name is the best signal
   - 2.5 RAG: retrieval, fusion, reranking, and when long context wins
   - 2.6 Tool-call reliability: native formats and constrained decoding
   - 2.7 Evaluation as a statistical gate
   - 2.8 macOS operations primitives
3. [Evidence](#3-evidence)
4. [Plan for this machine](#4-plan-for-this-machine)
   - 4.1 Profiles (work / all-in / stream / calibrate)
   - 4.2 Server: memory accounting, admission, GLM fix, wiring
   - 4.3 Clients: Claude Code and OpenCode
   - 4.4 Router
   - 4.5 RAG stack
   - 4.6 Tool calling
   - 4.7 Observability
   - 4.8 Evaluation harness and regression gate
   - 4.9 Operations
   - 4.10 Security
   - 4.11 Module map and build backlog
5. [Decision criteria & what to measure](#5-decision-criteria--what-to-measure)
6. [Failure modes](#6-failure-modes)
7. [Stretch ideas (push limits)](#7-stretch-ideas-push-limits)

---

## 0. System at a glance

```
             Claude Code  (ANTHROPIC_BASE_URL=http://127.0.0.1:8642, ANTHROPIC_AUTH_TOKEN=<key>)
             OpenCode     (@ai-sdk/openai-compatible, baseURL http://127.0.0.1:8642/v1)
             rag CLI / MCP, evals/*, curl
                      │  HTTP (localhost only, Bearer key, Host-header check)
                      ▼
 ┌──────────────────────────── python -m serve  (one process, launchd agent) ────────────────────────────┐
 │ security mw → activity mw → ROUTER (alias → tier: fast | main | deep; effort → thinking; ctx → RAG)    │
 │   │                                                                                                    │
 │   ├─ mlx_vlm.server protocol layer: /v1/messages, /v1/chat/completions, /v1/responses, count_tokens   │
 │   │    native tool parsers (qwen3_coder / glm47) + llguidance grammar (lazy, inside <tool_call>)       │
 │   │    APC prefix cache: RAM tier ≤ N GiB + SSD tier ~/.cache/local-ai/prefix-cache/<fingerprint>     │
 │   │    MTP speculative decoding (when the checkpoint ships an MTP head)                                │
 │   ├─ /v1/embeddings  (EmbeddingGemma-2 text, ~0.5 GiB)      /v1/rerank (Qwen3-Reranker-0.6B, ≤1.2 GiB) │
 │   ├─ MEMORY: resident-aware accounting, per-request admission, MemoryGuard, cache limit 2 GiB, no wire │
 │   ├─ WATCHDOG: memory pressure, swap, thermal → shed APC RAM tier → refuse long ctx → exit             │
 │   └─ OBS: logs/requests.jsonl (tokens, cached, TTFT, tok/s, α, peak, pressure, tool errors)           │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
          │ weights: models/<promoted>  (Qwen PLE n-gram table mmapped from SSD, not resident)
          ▼
 evals/gate  ── nightly/promotion ──> results/<build>/*.json ──> evals.compare ──> promote / reject
```

---

## 1. Why it matters (intuition)

**A model is not a system.** The quantized directory from `build/` is a probability distribution over next tokens.
What you actually experience in Claude Code depends on several other things:

1. **Time to first token on turn N of an agent loop.** Claude Code resends the whole conversation every turn: system
   prompt, tool schemas and the full history. Without prefix caching, turn 30 of a 60K-token session re-prefills 60K
   tokens. At the ~1,100 tok/s prefill reported for this model class on an M5 Max, that is **~55 s per turn**
   [computed from self-reported, §3]. With an exact-prefix cache hit, it costs only the new tokens: a tool result of
   ~1–3K tokens, i.e. **1–3 s**. The cache matters more than any kernel optimization.
2. **Whether a tool call parses.** An agent fails on its weakest step. A 2% malformed-tool-call rate over a 50-step
   task gives 1 − 0.98⁵⁰ = **64% of tasks hit at least one failure** [computed]. Claude Code usually recovers by
   retrying, but every retry costs a round of decode plus context.
3. **Whether the machine stays usable.** The model shares unified memory with Xcode, browsers, Docker and Claude Code
   itself. When the model pushes macOS into swap, *both* get slow. Decode reads every active expert every token, so
   swapped weights turn 50 tok/s into a crawl (mac §3).
4. **Whether you notice a regression.** Calibration, pruning and quantization change the model in small ways that
   greedy pass@1 hides: rare-name hallucination, broken termination, worse tool arguments (reap §Rec 8). Without a gate
   you will ship a worse build and blame the prompts.

So the system has to do four things: **keep the context cache hot, make tool calls valid by construction, keep memory
inside a contract, and measure everything against a reference.**

---

## 2. How it works (mechanics + key math)

### 2.1 Memory: what actually occupies RAM

The total process footprint is

```
footprint = W_resident + S_fixed + b·T + C_apc + A_prefill + H_helpers + M_cache + overhead
```

| Term | Meaning | Qwen3.8-Flash-Next (`qwen4_exp`) | GLM-5.3-Flash (`glm5_next`) |
|---|---|---|---|
| `W_resident` | Weights in Metal buffers (copied, not mmapped: mac §2) | see below | see below |
| `S_fixed` | Linear-attention recurrent state (fp32, fixed size) | 110 MiB | 141 MiB |
| `b` | KV bytes per context token (full/sparse-attention layers only) | 12 × 2372 = **28,464 B** | 11 × 1089 = **11,979 B** (after the projected-cache fix) |
| `T` | Prompt + generated tokens | | |
| `C_apc` | Resident prefix-cache checkpoints. Each is a **full cache copy** for hybrid models (decode §5) | | |
| `A_prefill` | Transient activations of one prefill chunk (step 2048) | measure | measure, plus a 64 KiB × T DSA transient (decode §3) |
| `M_cache` | MLX free-buffer pool (`set_cache_limit`) | 2 GiB | 2 GiB |

Check against decode §3 [computed]:

- Qwen at 128K: 28,464 × 131,072 / 2³⁰ + 0.107 = **3.58 GiB**, which matches decode §3.
- GLM at 128K: 11,979 × 131,072 / 2³⁰ + 0.138 = **1.60 GiB**, which matches.

**The new fact this chapter adds: on Qwen, disk bytes ≠ resident bytes.** The n-gram/PLE table is served by
`ple_storage` (numpy memmap, rows gathered on demand, quant §1). I read the safetensors **headers** of the two local
Qwen builds [measured]:

| Build (`models/…`) | Total safetensors | n-gram table | Routed experts | MTP | Vision | Everything else | `ple_storage` in config |
|---|---|---|---|---|---|---|---|
| `qwen-flash-next-reap288` (sh0wie, 4b g64, deployed baseline) | **68.44 GiB** | **29.80 GiB** | 35.60 GiB | **0.00 GiB** | – | 3.05 GiB | `{"manifest": "ple-store.json", "cache_rows": 0}` |
| `baseline-qwen-reap320` (Litwein oQ3e) | 65.94 GiB | 29.80 GiB | 30.76 GiB | 0.89 GiB | 0.84 GiB | 3.65 GiB | none |

Three consequences:

1. **`serve/memory.py: weights_bytes()` sums every safetensors file**, so it counts 68.44 GiB for the sh0wie build.
   If the PLE table really stays file-backed (to verify, §4.2), the resident weights are **68.44 − 29.80 = 38.64 GiB**.
   The admission check `weights + kv_reserve ≤ cap` (68.44 + 8 = 76.44 ≤ 80) is then ~30 GiB too pessimistic.
   That is room for helpers, a larger APC tier, or a much bigger expert set (§7.1).
2. **The deployed baseline has no MTP head** (0 bytes of `mtp.*`). `scripts/quickserve.sh` passes
   `--draft-model "$MODEL" --draft-kind mtp` anyway. On this build speculative decoding cannot be active; `serve`'s
   `mtp=auto` detects that correctly through `detect_mtp_splitter`. Every number observed on it is plain AR decode.
3. Mmapped pages are **clean and reclaimable**. Under pressure the kernel drops them and re-reads from SSD, unlike the
   anonymous Metal weight buffers (mac §2). A row lookup that misses the cache costs one random SSD read. How many
   rows one token touches is not documented: the config has `heads_per_ngram: 8`, `ngram_size: 3`, and
   `ple_layer_ids: [2]` (one PLE layer). Count the rows with an instrumented tiny-model run instead of assuming.

**Why one sequence at a time.** With `max_num_seqs = 1`, `T` is bounded by `max_context`, so the KV term has a hard
ceiling. Concurrency plus prefix-cache eviction churn plus wiring is the reported IOGPU-panic trigger (mac §3). On a
single-user laptop, batching buys little: decode is bandwidth-bound, and a second stream mostly slows the first.

### 2.2 Prefill, decode and prefix caching (the three costs)

- **Prefill** is compute-bound. FLOPs ≈ 2·N_active·T plus attention. It is the phase that uses the M5 neural
  accelerators (quant §4). Self-reported for this model on an M5 Max: **1,094–1,548 tok/s at 65K**, 1,453 tok/s at
  4K (MTPLX 2.12, §3).
- **Decode** is bandwidth-bound. Each token reads the active weights:
  tok/s ≤ 614 GB/s ÷ bytes_active_per_token. For Qwen at 4.5 bpw:
  10 experts × 48 layers × 4,915,200 params × 4.5/8 B = **1.33 GB of routed experts per token** [computed], plus
  attention, shared experts, hyper-connections and lm_head. mac §7 puts the ceiling at ≈ 235 tok/s; real decode is
  50–70 % of the ceiling or less. Self-reported plain decode is 43.8 tok/s, MTP 73.5 tok/s (decode §1).
- **Prefix caching** turns "re-prefill T tokens" into "restore a checkpoint and prefill Δ tokens". Hybrid models carry
  recurrent state that cannot be paged in 16-token blocks, so mlx-vlm's APC uses `strategy=checkpoint`. It stores a
  whole-cache snapshot at a prefix boundary, every `APC_CHECKPOINT_INTERVAL_TOKENS` = 2048 (decode §5). A hit needs
  the new prompt to *extend* the old one byte for byte. Claude Code does that per conversation
  (`CLAUDE_CODE_ATTRIBUTION_HEADER=0` if in doubt).

Time to first token for a request with prompt length T and cached prefix P:

```
TTFT ≈ t_restore(P) + (T − P) / prefill_rate + t_first_decode
```

| Situation | T − P | TTFT at 1,100 tok/s [computed] |
|---|---|---|
| Cold first Claude Code turn: system prompt + ~20–30K of tool schemas | 30,000 | **≈ 27 s** |
| Warm turn: tool result appended | 2,000 | **≈ 1.8 s** + restore |
| Cold 128K document | 131,072 | **≈ 120 s** (85 s at 1,548 tok/s) |

The SSD tier keeps checkpoints across idle unloads and restarts, keyed by a checkpoint fingerprint
(`ServeConfig.prefix_cache_path`). That is what makes "exit when idle" (§4.9) cheap. MTPLX reports restoring a
96,760-token session from its SSD cache in 8 ms [self-reported]. mlx-vlm's restore cost must be measured
(`apc_restore_s` in §4.7).

### 2.3 Speculative decoding math (what the acceptance log tells you)

The MTP head drafts γ tokens and the trunk verifies them in one forward pass. Assume each draft token is accepted
with probability α, independently, and acceptance stops at the first rejection. The tokens emitted per verify round
are then (Leviathan et al., §3):

```
E[tokens/round] = (1 − α^(γ+1)) / (1 − α)
speedup ≈ E / (1 + c)          c = (cost of drafting γ + verifying γ+1) / (cost of one plain decode step) − 1
```

With α = 0.78 (the 76–81 % reap §7 reports for the unpruned MTP head on a K=320 trunk) [computed]:

| γ | E[tokens/round] |
|---|---|
| 2 | 2.39 |
| 3 | 2.86 |
| 4 | 3.23 |

If a γ = 3 round costs 1.25× a plain step, the speedup is ≈ 2.3×. MTPLX observes 1.7× on a coding task
[self-reported], so the effective c is larger, or α is lower on code.

Why log α *per request*: α is a direct measure of how well the drafter matches the trunk on *your* traffic. A pruned
or requantized trunk with a stale MTP head shows up as a falling α before any benchmark moves. mlx-vlm already
returns `draft_rounds`, `draft_n_accepted` and `draft_n` in its metrics envelope [src: `server/generation.py`,
`GenerationMetrics`], so α = `draft_n_accepted / draft_n`.

Exactness: rejection sampling makes the output distribution equal to the trunk's (Leviathan Thm. 1). In bf16 it is
distribution-exact, not bit-exact (decode §1).

### 2.4 Routing: why the requested model name is the best signal

A router maps a request to (model, thinking mode, effort, max tokens, drafting, grammar). Learned routers such as
RouteLLM train a classifier on preference data to predict whether a weak model's answer would be good enough (§3).
That needs labels you do not have yet. Claude Code already gives you two strong signals for free:

1. **The model alias it asks for.** Claude Code sends background chores (titles, summaries, quick classification)
   to `ANTHROPIC_DEFAULT_HAIKU_MODEL`. The main loop goes to `ANTHROPIC_MODEL` or the Sonnet alias. `/model opus`
   is *your* explicit "this is hard". If those env vars are set to the aliases `local-fast`, `local-main` and
   `local-deep`, the alias *is* the difficulty label, chosen by the client or by you.
2. **`output_config.effort` and `thinking`.** Claude Code sends `thinking: {type: "adaptive"}` and
   `output_config.effort` for unknown model IDs (decode §6). Effort maps directly onto thinking off / budgeted /
   unbounded.

The second axis is **context length**. Once the prompt plus `max_tokens` would exceed `max_context` or the
admission budget, the choice is no longer which model but how to shrink the context: compaction, or retrieval for
documents (§2.5). Do not route to a bigger model; a bigger model has *less* RAM left for KV.

The third axis is the **profile** (§4.1). The "big model" (GLM-5.3-Flash REAP50, 89.6 GiB) only exists when the
machine is not being used for coding. Swapping models means a process restart: mac §Rec 7 warns against repeated
load/unload in one process. So "route to big" is a profile switch with a ~15–20 s reload [computed in §4.9], not a
per-request decision.

### 2.5 RAG: retrieval, fusion, reranking, and when long context wins

**Why retrieval at all, when the model reads 128–256K?**

1. **Cost.** A 128K prefill is ~2 min here (§2.2).
2. **Effective context < advertised context.** RULER shows many models degrade well before their nominal window on
   anything harder than single-needle retrieval (§3).
3. **Position effects** ("lost in the middle"): accuracy is U-shaped in where the evidence sits.
4. **Corpus size.** Your docs folder is not 128K tokens.

**Dense retrieval.** Encode query q and chunk d into unit vectors, score by s(q, d) = ⟨e_q, e_d⟩. With
Matryoshka-trained embedders you can keep the first k dimensions (EmbeddingGemma-2: 768/512/256/128, "close to
lossless down to 256" [self-reported]) and trade quality for memory. Brute-force search over N chunks is one
matrix-vector product: N × d × 2 bytes. For 200K chunks at d = 768 in fp16 that is **0.29 GiB**; 1M chunks is
1.43 GiB, or 0.48 GiB at 256 dims [computed]. No vector database is needed at personal scale.

**Lexical retrieval (BM25)** scores exact term overlap:

```
BM25(q,d) = Σ_t IDF(t) · f(t,d)(k1+1) / (f(t,d) + k1(1 − b + b·|d|/avgdl))
```

It finds identifiers, error strings and rare names that dense models blur. SQLite FTS5 implements it (`bm25()`), and
it is in the venv's Python: SQLite 3.54.0, FTS5 verified [measured]. No dependency needed.

**Fusion (RRF).** Two ranked lists have incomparable scores, so fuse by rank:

```
RRF(d) = Σ_lists 1 / (k + rank_list(d)),    k = 60
```

This is robust because it only needs ranks (Cormack et al., §3).

**Reranking.** A cross-encoder reads (query, passage) *jointly* and outputs P(relevant). It is far more accurate than
bi-encoder dot products, and too slow to run on the whole corpus, so it runs only on the top 50–100 fused candidates.
Qwen3-Reranker scores relevance as the probability of a "yes" token. mlx-vlm 0.7.9 implements this as
`_binary_scores` behind `/v1/rerank` [src]. On the top-100 candidates from Qwen3-Embedding-0.6B, the 0.6B reranker
raises MTEB-R from 61.82 to 65.80, but on MTEB-Code it scores 73.42 against the embedder's own 75.41, i.e. lower. The
4B reranker reaches 69.76 / 81.20 [self-reported]. So for code, test whether reranking helps at all before paying for it.

**Contextual retrieval.** Before indexing, prepend to each chunk a 50–100-token blurb, written by the LLM, that
situates the chunk in its document. Anthropic reports a 35 % reduction in top-20 retrieval failures from contextual
embeddings, 49 % with contextual BM25 added, and 67 % with reranking [self-reported, §3]. Cost here [computed]:

- Put the whole document as the *prompt prefix* and the chunk after it, so every chunk request shares the
  document prefix. Example: a 100K-token document split into 200 chunks of 500 tokens, ~100 tokens of instruction,
  ~60 generated tokens per chunk.
- With APC: one prefill of 100,000 / 1,100 ≈ 91 s, plus 200 × (600 / 1,450 + 60 / 60) ≈ 283 s,
  **≈ 6.2 min per 100K-token document**.
- Without a prefix cache: 200 × 101K / 1,100 ≈ **5.1 h**.
- This is the same trick Anthropic's write-up uses with prompt caching. Here APC is the prompt cache.

**When long context wins** (§4.5 has the decision table):

- the answer needs *global* reading (summarize, compare across the whole document);
- the document fits and you will ask many questions (prefill once, cache, ask N times for ~1–3 s each);
- RULER at that length on *our* build is close to its short-context score.

Self-Route (Li et al., §3) formalizes the hybrid. Try RAG first and ask the model whether the passages answer the
question. Only if not, fall back to full long context. The authors report LC-level quality at much lower cost
[self-reported].

### 2.6 Tool-call reliability: native formats and constrained decoding

**Use the native format.** Each model was post-trained on one tool-call syntax inside its own chat template
[measured from the templates]:

| Model | Syntax | mlx-vlm parser [src: `mlx_vlm/tools/parsers/`] |
|---|---|---|
| Qwen | `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>…</function>\n</tool_call>` | `qwen3_coder.py` |
| GLM | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>` | `glm47.py` |

Asking either model for OpenAI-style JSON in the prompt moves it off-distribution. The server's job is to translate
the native syntax into Anthropic `tool_use` blocks and OpenAI `tool_calls`, which mlx-vlm already does.

**Constrained decoding.** At each step, compute the set A_t of tokens that keep the output a valid prefix of a
grammar G, mask the rest, and renormalize:

```
p̃(x_t | x_<t) = p(x_t | x_<t) · 1[x_t ∈ A_t] / Z_t,      Z_t = Σ_{x∈A_t} p(x | x_<t)
```

Two facts follow:

1. **The output is valid by construction** (if G is right).
2. **The distortion is measurable.** −log Z_t is the probability mass the model wanted to put on invalid tokens.
   Sum it over a tool call and log it. A near-zero value means the grammar changed nothing; a large value means the
   model was confused and the grammar *forced* something. Such calls deserve a flag, not silent trust.

**But constraints can hurt reasoning.** Forcing structured output from the first token lowers reasoning accuracy
("Let Me Speak Freely", §3). CRANE shows the fix: alternate unconstrained reasoning with constrained segments,
delimited by trigger tokens. It reports up to +10 points over both pure strategies (§3). That maps exactly onto
native tool calls: think freely, then once the model emits `<tool_call>`, switch the grammar on until
`</tool_call>`. This is a **lazy (triggered) grammar**. llguidance (installed: 1.9.1 [measured]) computes token masks
from a Lark-style grammar or a JSON schema, and its authors report sub-millisecond mask times per token, which is
small against a ~15–20 ms decode step [self-reported, §3; measure with `grammar_mask_ms` in §4.7]. It already backs mlx-vlm's `response_format` path
(`structured.build_json_schema_logits_processor`, plus `ThinkingAwareLogitsProcessor`, which turns the mask on only
after the thinking block) [src].

What mlx-vlm 0.7.9 does **not** do [src]:

- **It does not constrain tool calls.** `tools/policy.py` implements forced `tool_choice` by *appending an
  instruction* to the system and user message.
- **It does not appear to apply logits processors in the MTP verify path.** `speculative/mtp.py` calls `sampler(...)`
  with no processor hook. Treat "grammar + MTP" as unsupported until a tiny-model test proves otherwise.

Speculative sampling composes with masking mathematically: verify against p̃ instead of p, and acceptance stays exact
for the masked distribution. That is a code change, not a theory problem.

### 2.7 Evaluation as a statistical gate

A gate decides whether a candidate build B is at least as good as the current build A, given noisy measurements.
Three tools cover it.

**KL to a reference** is the most sensitive metric per compute.

```
KL_t = Σ_v p_ref(v) [log p_ref(v) − log p_B(v)]
```

- `calib/` stores the reference's top-64 log-probabilities for `eval.jsonl`, so KL is estimated over the top-64 plus
  one bucket of residual mass.
- It is evaluated with teacher forcing (prefill only, no generation), so it costs roughly `eval tokens / prefill rate`.
- It compares *distributions*, so it catches the damage that accuracy hides. Accuracy can stay flat while individual
  answers flip, and flip rate tracks KL (Dutta et al., §3).
- Report mean KL per domain, top-1 agreement, and the **flip rate** on task evals: the fraction of items whose
  correctness changed versus the reference.

**Paired tests for pass/fail evals.** Run both builds on the same items with the same seed. Only the discordant
items carry information. With b = (A right, B wrong) and c = (A wrong, B right), McNemar's statistic is

```
χ² = (|b − c| − 1)² / (b + c)        reject "equal" at 5 % if χ² > 3.84
```

Unpaired, HumanEval+ (n = 164, p ≈ 0.9) has a standard error of √(0.9 · 0.1 / 164) = **2.3 points** [computed].
That is why reap §3 could not separate K = 288/320/384. Pairing removes the between-item variance.

**Blind A/B preference.** Show two anonymized answers to the same prompt, in random order, and record which wins.
Fit Bradley–Terry (the model behind Chatbot Arena, §3), or for two systems just test the win rate against 50 %.
Non-tie judgments needed to detect a true 60 % win rate at α = 0.05 with 80 % power [computed]:

```
n ≈ ((1.96 · 0.5 + 0.84 · √(0.6 · 0.4)) / 0.1)² ≈ 194      (≈ 85 for a 65 % win rate)
```

### 2.8 macOS operations primitives

- **launchd LaunchAgent** (per user, `~/Library/LaunchAgents`).
  - `KeepAlive {SuccessfulExit: false}` restarts after a crash but not after a clean exit.
  - **Socket activation:** a `Sockets` dict makes launchd own the listening socket and start the job on the first
    connection. The process collects the fds with `launch_activate_socket("Listeners")`, which is present in
    libSystem here [measured via ctypes].
  - With socket activation, "idle" can mean "**process gone, 0 bytes**" instead of "process alive, model unloaded".
    This avoids repeated in-process load/unload (mac §Rec 7).
- **Memory pressure.**
  - `sysctl kern.memorystatus_vm_pressure_level`: 1 = normal, 2 = warn, 4 = critical. It reads 1 right now
    [measured].
  - `kern.memorystatus_level` is the percentage of memory available (86 now [measured]).
  - Swap: `sysctl vm.swapusage` shows 7.49 GiB used of 9 GiB *right now*, while the GPU benchmark runs [measured].
    A watchdog must look at both.
- **Thermals and power.**
  - `pmset -g therm`, and `sudo powermetrics --samplers gpu_power,thermal`.
  - Energy mode is set in System Settings → Battery → Energy Mode → *On power adapter: High Power*.
  - `pmset -g` shows `powermode 0` (Automatic) here [measured]. A developer-forum report on an M5 Max shows
    `powermode 2` for High Power, but there is no documented `pmset` setter [self-reported], so verify with
    `system_profiler SPPowerDataType | grep "High Power"` (mac §5).

---

## 3. Evidence

Primary sources, with self-reported numbers marked. Anything in the installed packages was read directly (**[src]**).

| Claim | Source |
|---|---|
| mlx-vlm 0.7.9 server routes: `/v1/messages`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/metrics`, `/v1/cache/stats`. Auth is `Authorization: Bearer` only (`_require_management_api_key`, `secrets.compare_digest`) | [src] `mlx_vlm/server/app.py:61-85,496,1154`, `server/embeddings.py:132`, `server/reranking.py:403` |
| Per-request metrics envelope with `prompt_tps`, `generation_tps`, `cached_tokens`, `peak_memory`, `draft_rounds`, `draft_n_accepted` | [src] `server/generation.py:342-430, 788-860` |
| JSON-schema constrained decoding via llguidance, with a thinking-aware wrapper; forced `tool_choice` is a prompt instruction, not a grammar | [src] `mlx_vlm/structured.py`, `tools/policy.py` |
| MTP verify path uses `sampler`, with no logits-processor hook | [src] `mlx_vlm/speculative/mtp.py:131-390` |
| Embedding archs `qwen3_embedding`, `embedding_gemma2`, `gemma3_embedding`, `lfm2_colbert`; rerankers: Qwen3-style yes/no and sequence-classification | [src] `mlx_vlm/models/`, `embedding_loader.py`, `reranker_loader.py` |
| SSD expert offload: `moe_offload.ExpertStore`, per-expert mmap + byte-budgeted LRU, `expert_cache_gb`; requires chunked prefill | [src] `mlx_vlm/moe_offload.py:1-14, 524-750` |
| The repo's `serve/` rejects `model_kind` other than `auto`/`text_generation` (embeddings/rerank would 400), wires `weights + kv_reserve` by default, loads the model in middleware *before* the auth check, and sets CORS `*` | [src] `serve/engine.py:get_cached_model`, `serve/memory.py:wired_limit_bytes`, `serve/app.py:ActivityMiddleware`, `create_app` |
| Qwen build byte breakdown (n-gram 29.80 GiB, MTP 0 on sh0wie REAP-288) and HF sizes (sh0wie 73.5 GB; pipenetwork GLM REAP50 96.3 GB = 89.6 GiB) | [measured] safetensors headers; HF API `?blobs=true` |
| Prefill 786→1,453 tok/s (4K) and 768→1,094 tok/s (65K), 1,548 tok/s with newer kernels; 125.8 tok/s on one OpenCode request (18,364 of 18,539 tokens from cache); 96,760-token session restored in 8 ms | [self-reported] [MTPLX card](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed), [release notes 2.12.0](https://mtplx.com/releases/2.12.0/) |
| Speculative decoding expected-length formula and exactness | [Leviathan et al. 2023, arXiv 2211.17192](https://arxiv.org/abs/2211.17192) |
| Learned routing between strong and weak models | [RouteLLM, arXiv 2406.18665](https://arxiv.org/abs/2406.18665) [self-reported cost savings] |
| RAG vs long context, Self-Route hybrid | [Li et al., arXiv 2407.16833](https://arxiv.org/abs/2407.16833) [self-reported] |
| Order-preserving RAG beats LC at equal budget for some tasks | [OP-RAG, arXiv 2409.01666](https://arxiv.org/abs/2409.01666) [self-reported] |
| Effective context < claimed context | [RULER, arXiv 2404.06654](https://arxiv.org/abs/2404.06654) |
| U-shaped position effect | [Lost in the Middle, arXiv 2307.03172](https://arxiv.org/abs/2307.03172) |
| Contextual retrieval: −35 % / −49 % / −67 % top-20 failures | [Anthropic, Sept 2024](https://www.anthropic.com/news/contextual-retrieval) [self-reported] |
| Late chunking (embed the whole document, then pool per chunk) | [arXiv 2409.04701](https://arxiv.org/abs/2409.04701) |
| RRF, k = 60 | Cormack, Clarke, Büttcher, SIGIR 2009, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods" |
| Qwen3-Embedding-0.6B MMTEB 64.33, 8B 70.58 (#1 on 2025-06-05); 32K context; instructions +1–5 % | [self-reported] [Qwen3-Embedding-0.6B card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Qwen3-Reranker-0.6B / 4B: MTEB-R 65.80 / 69.76, MTEB-Code 73.42 / 81.20 (top-100 from Qwen3-Embedding-0.6B) | [self-reported] [Qwen3-Reranker-0.6B card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| EmbeddingGemma-2: 740M total, 270M text backbone, 768-d (MRL 512/256/128), 8K context, task prompts (`task: code retrieval \| query: …`) | [self-reported] [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (in local HF cache) |
| Constrained formats can hurt reasoning | [Let Me Speak Freely?, arXiv 2408.02442](https://arxiv.org/abs/2408.02442) |
| Alternating free/constrained generation, up to +10 pts | [CRANE, arXiv 2502.09061](https://arxiv.org/abs/2502.09061) (ICML 2025) [self-reported] |
| Token-mask engines | [llguidance](https://github.com/guidance-ai/llguidance); [XGrammar, arXiv 2411.15100](https://arxiv.org/abs/2411.15100) [self-reported speeds] |
| Accuracy hides flips; KL correlates with flips | [Dutta et al., "Accuracy is Not All You Need", arXiv 2407.09141](https://arxiv.org/abs/2407.09141) |
| Bradley–Terry pairwise ranking | [Chatbot Arena, arXiv 2403.04132](https://arxiv.org/abs/2403.04132) |
| LLM-as-judge biases (position, verbosity) | [Zheng et al., arXiv 2306.05685](https://arxiv.org/abs/2306.05685) |
| BFCL V4 = non-live/live single-turn, irrelevance, multi-turn, agentic (web search, memory, format sensitivity). An Epoch audit found defects in 24/50 sampled tasks, worst in memory | [Gorilla blog](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html); [Epoch review](https://epoch.ai/benchmarks/berkeley-function-calling-leaderboard/review) [self-reported] |
| LiveCodeBench v6 = 1,055 problems from May 2023 to Apr 2025 (before both candidates' Aug 2026 release, so contaminated as an absolute score) | [LiveCodeBench, arXiv 2403.07974](https://arxiv.org/abs/2403.07974); [dataset README](https://huggingface.co/datasets/ledale/code_generation_lite/blob/main/README.md) |
| SWE-bench on Apple Silicon: `--namespace ''` builds images locally, arm64 "experimental", ~120 GB disk | [SWE-bench](https://github.com/SWE-bench/SWE-bench) docs (via [swebench 4.1.0](https://simple-repository.app.cern.ch/project/swebench)); [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) |
| Claude Code gateway protocol, env vars, idle watchdog/ping | [LLM gateway](https://code.claude.com/docs/en/llm-gateway), [protocol](https://code.claude.com/docs/en/llm-gateway-protocol), [model config](https://code.claude.com/docs/en/model-config) (decode §6) |
| OpenCode custom provider | [opencode.ai/docs/providers](https://opencode.ai/docs/providers/) |
| High Power Mode on Max chips; `powermode 2` observed | [MacRumors guide](https://macrumors.com/guide/high-power-mode); [Apple dev forums 835131](https://developer.apple.com/forums/thread/835131) [self-reported] |
| Personal transcripts: 1,567 `.jsonl` sessions in 135 projects, **1.59 GB**, median 266 KB, p90 556 KB, max 507 MB; 1,390 modified in 2026-10 | [measured] file metadata only (`stat`), contents not read |

---

## 4. Plan for this machine

### 4.1 Profiles

The working constraint (model process ≤ ~80 GiB, ≥ 45 GiB left for macOS and dev tools) is one profile, not a law of
nature. The plan defines four, each a `scripts/profile.sh <name>` that writes the launchd environment and restarts
the agent:

| Profile | When | Model | Cap (`--memory-limit-gb`) | Left for the OS | Notes |
|---|---|---|---|---|---|
| **work** (default) | You are coding | Promoted Qwen build (today `qwen-flash-next-reap288`) + embedder + reranker | **80** | ≥ 48 GiB | 1 seq, APC RAM ≤ 6 GiB + SSD 32 GiB, no wiring, idle exit 15 min |
| **all-in** | Not coding: evenings, long reading or reasoning sessions | GLM-5.3-Flash REAP50 mixed 4/8 (89.6 GiB) **only after the projected-cache fix**, or Qwen with more experts (§7.1) | **104** = `max_rec` 107.5 − 3.5 | ≥ 24 GiB | Watchdog thresholds tighter, no Docker |
| **stream** | "Big but slow" offline answers | Unpruned GLM 4-bit via `moe_offload` (171 GB experts on SSD) | 80 | ≥ 48 GiB | Decode bound by expert-cache hit rate (§7.3); batch and overnight use only |
| **calibrate** | Overnight `calib.run` / `build.make` / `evals.gate` | none (server stopped) | contract: calib peak < 40 GiB | rest | High Power, `caffeinate -dimsu`, Docker allowed |

Memory budget of the **work** profile on today's build [computed; ✱ = must be measured]:

| Item | Conservative accounting (what `serve/memory.py` does today) | Resident-aware accounting (to build, §4.2) |
|---|---|---|
| Weights | 68.44 GiB (all safetensors) | 38.64 GiB (minus mmapped n-gram 29.80) |
| KV + state at 128K, no MTP head | 3.58 | 3.58 |
| APC resident checkpoints | `cap − weights − reserve` = 3.56 → about one 128K checkpoint | 6.0 (two 64K checkpoints or ~six 30K ones; ~1.0 GiB per 30K) |
| Prefill activations, step 2048 ✱ | inside the 8 GiB reserve | 3.0 reserve ✱ |
| MLX buffer cache | 2.0 (`--cache-limit-gb 2`, not the default 4) | 2.0 |
| EmbeddingGemma-2 (text only, 270M params, bf16) | does not fit | ≈ 0.5 |
| Qwen3-Reranker-0.6B (bf16 / 8-bit) | does not fit | ≈ 1.2 / 0.6 |
| **Total** | **≈ 78–80** (at the cap) | **≈ 55** (~25 GiB spare under 80) |

The spare ~25 GiB is the most important number in this chapter. §7.1 spends it.

### 4.2 Server: memory accounting, admission, GLM fix, wiring

The base stays the stock mlx-vlm server wrapped by `serve/` (decode §Rec 9). Changes, in priority order:

**(a) Resident-aware weight accounting** (`serve/memory.py`).

- Add `resident_weights_bytes(model_dir)`. Parse safetensors headers (8-byte length + JSON, no tensor data read: the
  code in §3 "[measured]" does exactly this). Exclude tensors whose names match the `ple_storage` manifest
  (`ple-store.json` → `shards[*].weight/scales/biases.file+offset`), and count the rest.
- Use it in `check_fits` and in `prefix_cache_memory_bytes`. Keep the old number as `disk_bytes` in `/serve/status`.
- **Verification before trusting it:** load the build in the `calibrate` profile window, then compare
  `footprint -p <pid>` (IOAccelerator vs mapped-file lines), `mx.get_active_memory()` and the computed resident
  number. Then run `memory_pressure -S -l warn` and confirm decode speed holds while the mapped n-gram pages are
  reclaimed. Acceptance: |active − predicted| < 1 GiB, and decode under `warn` within 10 % of idle.

**(b) Per-request admission** in `serve/engine.py`, before the request is queued:

```python
def est_bytes(prompt_tokens, max_tokens, fam):          # 2.1
    b, s_fixed = {"qwen": (28_464, 110 << 20), "glm": (11_979, 141 << 20)}[fam]
    if mtp_active: b += {"qwen": 2_372, "glm": 1_089}[fam]          # MTP layer adds one attention layer
    T = prompt_tokens + max_tokens
    a = prefill_act_bytes(step)                                       # measured table from evals.speed
    glm_t = 65_536 * prompt_tokens if fam == "glm" and prompt_tokens > 32_768 else 0   # decode §3 transient
    return s_fixed + b * T + a + glm_t

if mx.get_active_memory() + mx.get_cache_memory() + est_bytes(...) > cap:
    evict APC RAM tier to SSD → mx.clear_cache() → re-check → else 503 (Anthropic error body) with "use RAG/compact"
```

Include `get_cache_memory()` (mac §Rec 3). The existing `MemoryGuard` stays as the in-flight backstop.

**(c) GLM projected-cache fix** (`serve/glm_fix.py`, decode §Rec 1). After every prefill, replace `caches[3]` with an
empty `KVCache()` on all 11 DSA layers and on the MTP drafter's cache, and set
`glm5_next.language._MAX_PROJECTED_PREFILL_TOKENS = 4096`. This was bit-exact in the tiny-model probe. Without it,
GLM holds +22–24 GiB at ≥ 32K, and the **all-in profile does not fit**: 89.6 + 25.7 > 104.

**(d) Wiring off by default.** `serve/memory.py: wired_limit_bytes` currently defaults to `weights + kv_reserve`. That
contradicts mac §Rec 2: the only controlled evidence ties wiring plus concurrency plus cache churn to IOGPU panics,
and wiring does nothing once idle without the refresh variable. Change the default to `0`. Keep
`--wired-limit-gb weights` as opt-in, with `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000`. `scripts/run_server.py`
already does this (`WIRE_GB=0`).

**(e) Allow helper model kinds.** `Engine.get_cached_model` raises 400 for any `model_kind` other than
`auto`/`text_generation`, so `/v1/embeddings` and `/v1/rerank` are dead behind the wrapper. Let `embedding` and
`reranker` through to the original loader. Add `--embedding-model` and `--reranker-model`, mapped to
`MLX_VLM_PRELOAD_EMBEDDING_MODEL` / `MLX_VLM_PRELOAD_RERANKER_MODEL` [src], and add their bytes to `check_fits`.
Keep them **in the same process**: mac §3 lists multiple MLX processes among the panic correlates.

**(f) Prefill step.** Keep 2048 (decode §Rec 8). Let `evals.speed` decide 4096 for the work profile only if peak at
128K stays ≤ cap − 4 GiB.

**(g) KV quantization stays off** (decode §2: it crashes or is a no-op on both architectures in the batch path).
`serve._warmup` already falls back.

### 4.3 Clients: Claude Code and OpenCode

Generate a key once and keep it in the login Keychain, not in dotfiles:

```bash
security add-generic-password -U -s local-ai -a serve -w "$(openssl rand -hex 32)"
LOCAL_AI_KEY="$(security find-generic-password -s local-ai -a serve -w)"
```

**Claude Code** (`~/.zshrc` function, so only explicitly local sessions use it):

```bash
claude-local() {
  ANTHROPIC_BASE_URL=http://127.0.0.1:8642 \
  ANTHROPIC_AUTH_TOKEN="$(security find-generic-password -s local-ai -a serve -w)" \
  ANTHROPIC_MODEL=local-main \
  ANTHROPIC_DEFAULT_SONNET_MODEL=local-main \
  ANTHROPIC_DEFAULT_OPUS_MODEL=local-deep \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=local-fast \
  CLAUDE_CODE_SUBAGENT_MODEL=local-main \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 \
  CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
  claude "$@"
}
```

- Use **`ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.** The first is sent as `Authorization: Bearer`, which is the
  only header mlx-vlm checks [src]. The second goes out as `x-api-key` and would get a 401.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` must equal `--max-context`, or auto-compaction fires too late and requests start
  failing admission.
- Long prefills need SSE `ping` events, or Claude Code's idle watchdog aborts the request (decode §6). The cold
  27–120 s TTFTs in §2.2 make this mandatory. Add a test in `tests/serve` that streams a request whose prefill is
  delayed and asserts at least one ping per 10 s.

**OpenCode** (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local-ai (M5 Max)",
      "options": { "baseURL": "http://127.0.0.1:8642/v1", "apiKey": "{env:LOCAL_AI_KEY}" },
      "models": {
        "local-main": { "name": "local main",  "limit": { "context": 131072, "output": 32768 } },
        "local-fast": { "name": "local fast",  "limit": { "context": 131072, "output": 4096 } },
        "local-deep": { "name": "local deep",  "limit": { "context": 131072, "output": 32768 } }
      }
    }
  }
}
```

Standardize on port **8642**, the `serve/config.py` default. `scripts/quickserve.sh` uses 8080; change it or retire
it.

### 4.4 Router

New file `serve/router.py`, applied in `rn._build_gen_args`, which `engine._patch_mlx_vlm` already hooks.

| Alias (from the client) | Thinking | Effort map (`output_config.effort`) | `max_tokens` clamp | MTP | Grammar |
|---|---|---|---|---|---|
| `local-fast` (Haiku slot: titles, summaries, quick classification) | off | ignored | 2,048 | on | tool grammar on |
| `local-main` (default loop) | effort `low` → off; `medium` → budget 4K; `high` → budget 16K; absent → template default (on) | as stated | 32,768 | on outside thinking budgets* | on |
| `local-deep` (`/model opus`) | on, unbounded | `max` | 32,768 | on | on |
| any other name | as `local-main` | | | | |

\* `serve/engine.build_gen_args` drops `thinking_budget` when MTP is active, because mlx-vlm cannot enforce a budget
while drafting [src]. Implement budgets with a **two-phase generation** that keeps MTP:

1. Generate with stop string `</think>` and `max_tokens = budget`.
2. If the budget ran out, append `"\n</think>\n\n"` to the assistant prefix and continue. The prompt plus thoughts is
   an exact prefix, so APC makes phase 2's prefill nearly free.

**Context routing.** If `prompt_tokens + max_tokens > max_context` or admission fails, return an Anthropic
`invalid_request_error` saying "context too long". Claude Code then compacts. Never truncate silently.

**Profile routing.** `local-deep` in the work profile is the same weights with unbounded thinking. If you want the
all-in model, switch profile explicitly: `scripts/profile.sh all-in`, then a ~15–20 s reload (§4.9). Write the
current profile into `/v1/models` so clients can see what they are talking to.

**When to add a separate fast model.** Background Haiku-slot requests queue behind the main loop (`max_num_seqs=1`)
and can evict APC checkpoints. Log `route`, `queue_wait_s` and `apc_hit` per request (§4.7). If Haiku-slot requests
cause more than 10 % of main-loop queue wait, or more than 5 % of main-loop APC misses over a week, add a resident
Qwen3.5-2B 4-bit helper (HF `mlx-community/Qwen3.5-2B-4bit`; size to verify, ≈ 1–1.5 GiB) for `local-fast` only. It
needs a second text-generation slot in the engine, because mlx-vlm keeps one model per kind (`for_kind`). That is
real work, so measure first.

### 4.5 RAG stack

New package `rag/`. No new dependencies for text, Markdown and code. PDF extraction needs one new dependency,
`pypdf` (not installed [measured]; list it, do not edit `pyproject.toml`).

```
rag/
  ingest.py   # walk paths → Document{id, path, mtime, sha256, text, kind}; pdf via pypdf (new dep)
  chunk.py    # structure-aware: markdown headings / code defs (regex for py/ts/go/rs; tree-sitter = stretch) → ~512 tok, 12% overlap
  context.py  # contextual blurbs via the local server, doc-as-prefix for APC hits (2.5)
  index.py    # SQLite: chunks(id, doc_id, ord, text, ctx, tokens), FTS5 virtual table (porter unicode61); vectors.f16.npy + ids.npy
  embed.py    # client for /v1/embeddings with task prompts ("task: search result | query: …", "title: … | text: …")
  search.py   # dense top-100 ∪ BM25 top-100 → RRF(k=60) → /v1/rerank top-50 → top-k packed in document order
  answer.py   # pack + cite chunk ids; Self-Route: RAG first, model judges sufficiency, else long context
  __main__.py # uv run python -m rag {add,update,search,ask,stats} --index ~/.local/share/local-ai/rag/<name>
```

**Model choice.** Start with **EmbeddingGemma-2**: already in the HF cache, `embedding_gemma2` supported [src],
8K context, code-retrieval prompts, ~0.5 GiB in bf16 for the text path (needs a text-only load: the card says
selective encoder loading differs per library). Compare it against **Qwen3-Embedding-0.6B** (32K context,
self-reported MMTEB 64.33; `qwen3_embedding` supported) on your own queries before switching. The reranker is
**Qwen3-Reranker-0.6B**. The 4B (self-reported +4 MTEB-R, +7.8 MTEB-Code) costs ~2.5 GiB at 4-bit, which fits in the
resident-aware work budget.

**Building a retrieval eval** (`evals/rag.py`). Take 100–200 (question, gold chunk ids) pairs:

- auto-generate questions from random chunks with the local model;
- hand-write 30 hard ones.

Report recall@k (k = 5, 20) and MRR for each configuration: dense, BM25, RRF, RRF + rerank, plus contextual. Choose
by recall@20 after rerank. Each eval run is seconds; the questions cost ~1 s each to generate.

**Long context vs retrieval: decision table** (`rag/answer.py`). The thresholds depend on our RULER numbers:

| Condition | Use | Why |
|---|---|---|
| Doc ≤ 32K tokens | Long context (whole doc) | ≤ ~30 s cold prefill (32,768 / 1,100 tok/s), then cached |
| 32K < doc ≤ `max_context` − 16K **and** RULER(our build, that length) ≥ 0.9 × RULER(4K) **and** ≥ 3 questions expected | Long context + APC (prefill once, ~1–3 s per later question) | Amortized: 120 s / N questions |
| Global question (summarize/compare everything) and doc > budget | Map-reduce over sections with the `local-fast` alias, then reduce with `local-main` | No single prompt holds it |
| Otherwise (corpus, single lookup, RULER below threshold) | Hybrid RAG: top 8–20 chunks ≈ 6–12K tokens, in document order (OP-RAG) | ~6–12 s prefill, robust to position effects |
| RAG answer judged insufficient (Self-Route probe: "Can the passages answer? yes/no", 1 token) | Fall back to long context if it fits, else widen k | Li et al. hybrid |

**Claude Code integration.** v1 is a CLI that Claude Code can call through Bash
(`uv run python -m rag search "…" --k 8 --json`), plus a `CLAUDE.md` note. Stretch: an MCP server exposing
`rag_search` / `rag_ask` (needs the `mcp` package, a new dependency).

**Memory and time.**

- Index for 200K chunks: 0.29 GiB of vectors + SQLite (~2× the text size) [computed].
- Embedding 200K chunks × ~600 tokens = 120M tokens. Embedder throughput is unmeasured: **[assumption]
  5–20K tok/s → 1.7–6.7 h** as a one-time overnight job in the calibrate profile. Measure with
  `rag stats --bench` on 1K chunks first.
- Contextual blurbs: ~6 min per 100K-token document (§2.5), so do them only for docs you query often.

### 4.6 Tool calling

Layered, cheapest first:

1. **Correct template + native parser** (exists).
   - `tests/serve/test_serve_protocols.py` covers OpenAI `tool_calls`, Anthropic `tool_use` and thinking blocks for
     both architectures.
   - Add tests for **parallel calls**, **tool call immediately after `</think>`**, **unicode / multi-line parameter
     values** (Qwen's XML format carries raw code in parameters, which is where JSON escaping would fail), and
     **truncation at `max_tokens` inside `<tool_call>`**, which must yield `stop_reason: "max_tokens"`, not a
     half-parsed call.
2. **Schema validation and coercion** (`serve/tools_guard.py`). After parsing, validate the arguments against the
   request's tool `input_schema`. Coerce safely (`"5"` → `5` for integer, `"true"` → `true`), as `qwen3_coder`-style
   parsers do by type. If the call is still invalid: re-sample once from the `<tool_call>` offset with the grammar
   forced, then return a well-formed call. Log `repair: coerced | resampled | failed`.
3. **Lazy grammar** (`serve/grammar.py`). Build a Lark grammar per request from the tool list:

   ```lark
   start: "<tool_call>\n<function=" NAME ">\n" param* "</function>\n</tool_call>"
   NAME: "Read" | "Edit" | "Bash" | ...            // from request.tools
   param: "<parameter=" PNAME ">\n" VALUE "\n</parameter>\n"
   PNAME: ...                                       // union of the chosen tool's properties (stretch: per-tool branches)
   VALUE: /(.|\n)*?/                                // or %json{...} sub-schemas for non-string types
   ```

   Activate it when `<tool_call>` is sampled, through a processor modeled on `ThinkingAwareLogitsProcessor`, and
   release it after `</tool_call>`. GLM uses the `<arg_key>/<arg_value>` variant.
   - **Disable MTP drafting inside the grammar segment** until masked verification exists (§2.6). Tool calls are
     ~50–300 tokens, so the cost is small. Example: 150 tokens at a plain 44 tok/s ≈ 3.4 s, versus ≈ 2 s with MTP at
     74 tok/s, so about +1.4 s per call [computed from self-reported].
   - Record Σ(−log Z_t) per call as `grammar_mass_removed`.
4. **Measure from the conversation itself.** Every Claude Code request carries the previous turns' `tool_result`
   blocks with `is_error`. So the server can compute a **live tool-error rate per tool** (Edit, Bash, Read, …) with
   no client changes. `Edit` failures ("old_string not found / not unique") are the classic local-model failure;
   track them separately.

Decision: ship layers 1–2 first. Turn on layer 3 only if the BFCL `irrelevance` score and the live tool-error rate
show that malformed calls, rather than wrong decisions, dominate. Grammars fix syntax, not judgment.

### 4.7 Observability

A request log, JSONL, one line per request, at `logs/requests.jsonl` (0600, rotated daily, 30-day retention):

```json
{"ts": "...", "req": "uuid", "client": "claude-cli/2.x | opencode | rag | evals", "endpoint": "/v1/messages",
 "alias": "local-main", "route": {"thinking": "budget:4096", "mtp": true, "grammar": false},
 "profile": "work", "build": "qwen-flash-next-reap288@<fingerprint>",
 "prompt_tokens": 41234, "cached_tokens": 39100, "apc_hit": true, "apc_restore_s": 0.21,
 "queue_wait_s": 0.0, "ttft_s": 2.4, "prefill_tok_s": 1180, "decode_tok_s": 61.2, "gen_tokens": 812,
 "think_tokens": 540, "draft": {"rounds": 280, "proposed": 840, "accepted": 655, "alpha": 0.78},
 "mem": {"active_gib": 47.1, "cache_gib": 1.2, "peak_gib": 52.3, "pressure": 1, "swap_used_gib": 0.4},
 "thermal": "nominal", "finish": "tool_use", "tool_calls": 1, "tool_repair": null,
 "prev_tool_errors": {"Edit": 1}, "grammar_mass_removed": null, "error": null}
```

- **Source of most fields:** wrap `ServerMetricsStore.record_success/record_failure` [src] in
  `engine._patch_mlx_vlm`. The envelope already carries tokens, tok/s, cached tokens, peak memory and draft stats.
  Add memory (`serve/memory.snapshot()`), pressure (`kern.memorystatus_vm_pressure_level`), swap and route.
  - **Never log prompt text by default.** Log hashes and lengths. `--log-prompts` writes to a separate file for
    debugging.
- **Summaries:** `uv run python -m serve.report [--since 7d]`. It prints p50/p95 TTFT split by APC hit/miss, decode
  tok/s, α by alias and domain, tool error rate per tool, admission rejects, guard trips, pressure events and queue
  wait.
- **Live:** `/serve/status` (exists) plus `/serve/metrics` (rolling aggregates).
- **Optional:** Claude Code can export its own OpenTelemetry metrics (`CLAUDE_CODE_ENABLE_TELEMETRY=1`; see the
  Claude Code monitoring docs) for client-side session and cost views. Not needed for v1.

What to alert on (log at WARNING, and `osascript -e 'display notification …'` from the watchdog):

- α drops > 0.1 below its 7-day median for an alias → drafter/trunk mismatch or a domain shift;
- APC hit rate on `local-main` turns ≥ 2 drops below 80 % → the client prefix is unstable;
- peak > cap − 2 GiB;
- swap growth > 2 GiB/h while serving.

### 4.8 Evaluation harness and regression gate

Existing suites (`evals/*.py`, results in `results/<name>/<suite>.json`, compared by `evals.compare`): `kl`, `speed`,
`code` (HumanEval+, MBPP+), `math` (AIME24/25, MATH-500 subset), `longctx` (RULER-style: niah_single, niah_multikey,
variable_tracking, qa at 32K/64K/128K), `tools` (BFCL single-turn categories).

To build:

| Suite | What | Size / cost [computed, at 1,100 tok/s prefill and ~60 tok/s decode, both to be replaced by `evals.speed`] |
|---|---|---|
| `evals.kl` additions | Per-domain KL, top-1 agreement, **paired bootstrap CI of ΔKL vs the baseline build** (resample samples, 10K reps) | `eval.jsonl` size ÷ prefill rate. 2M tokens ≈ **30 min** |
| `evals.flips` (in `code`/`math`) | Per-item correctness vs the reference build; McNemar | free once both runs exist |
| `evals.lcb` | LiveCodeBench, the newest window available. Contaminated in absolute terms, still valid for *relative* B vs A | 200 problems × ~800 tokens (thinking off) ≈ 45 min; thinking on ×5–10 |
| `evals.tools` additions | BFCL `multi_turn`. Agentic memory subsets later, given the 48 % defect finding | ~1 h |
| `evals.swe` | mini-swe-agent on a fixed 50-task SWE-bench Verified subset whose images build on arm64 (`--namespace ''`), plus **personal SWE**: commits from your own repos where a test fails before and passes after | per step ≈ 2K new tokens / 1,400 + 300 / 60 ≈ 6.4 s; 30 steps ≈ 3.2 min/task + tests → **≈ 3–4 h for 50**. Docker VM 8–12 GiB → calibrate profile only |
| `evals.replay` | Personal workload from Claude Code transcripts (below) | 200 episodes × 40K context ≈ 2.0 h cold prefill (APC helps for episodes from the same session) + 0.3 h generation |
| `evals.ab` | Blind pairwise preference UI (CLI or a small local HTML page served by `serve`), stores `results/ab/<A>_vs_<B>.jsonl` | ~194 non-tie judgments to detect 60 %; ~85 for 65 % |
| `evals.gate` | Runs a tier, compares to `results/<baseline>/`, writes `results/<candidate>/gate.json` with PASS / FAIL / REVIEW | orchestration only |

**Personal-workload replay.** `calib_data/transcripts.py` is shared with the calibration chapter.
`~/.claude/projects/*/*.jsonl` holds 1,567 sessions, 1.59 GB [measured, metadata only]. Raw bytes overstate tokens:
JSON framing plus large tool outputs. The plan never reads them during development; the extractor runs once, locally,
in the calibrate profile.

1. **Extract episodes.** For each assistant turn containing a `tool_use`: context = all prior messages plus the tool
   schemas; label = (tool name, arguments); outcome = whether the following `tool_result` had `is_error`, and whether
   the user interrupted or rejected. Skip episodes > 120K tokens.
2. **Scrub.** Apply secret regexes (AWS/GCP/GitHub/OpenAI/Anthropic key shapes, `-----BEGIN … KEY-----`, `.env`-style
   `KEY=…`, JWTs, high-entropy strings > 32 chars), and drop `tool_result` bodies over 8K tokens to a head/tail
   excerpt. Write to `artifacts/personal/episodes.jsonl` with mode 0600. Add `artifacts/personal/` to `.gitignore`.
   It never leaves the machine and is never uploaded with a model.
3. **Render in the target's chat template** with its native tool format (as `calib_data/render.py` does for public
   data).
4. **Metrics.** These *compare to Claude's choice*, which is a reference point, not ground truth:
   - tool-name agreement;
   - argument validity against the schema;
   - "same target": the same file path for Read/Edit, the same command family for Bash;
   - `Edit` old_string-exists rate, checked against the file content captured in the earlier Read result;
   - KL vs the *reference local model* on the same prompts. This is the regression signal, and it does not depend
     on Claude.
5. **Split by time.** Episodes from the last 2 weeks are the held-out eval; older ones can feed calibration. This
   prevents tuning on the test set.

**Gate tiers.**

| Tier | When | Contents | Time [computed] |
|---|---|---|---|
| T0 smoke | every build, every server change | load, one Anthropic stream with ping, one tool call per arch, 128K admission check, `pytest tests/serve` | ~5 min |
| T1 candidate | every candidate build (overnight) | `kl` (full), `speed` (1K/32K/128K, MTP on/off), `tools` (BFCL single-turn, 400 items), `code` thinking off (542 items ≈ 1.0 h), `longctx` (4 tasks × n=10 × 32/64/128K ≈ 9.2M prefill tokens ≈ **2.3 h**), `replay` (100 episodes ≈ 1.2 h) | **≈ 6 h** |
| T2 promotion | before replacing the work-profile build | T1 + `math` (AIME 60 × mean@2 × ~12K thinking tokens ≈ **6.7 h**), `code` thinking on (≈ 7.5 h), `lcb`, `swe` (≈ 3–4 h), `ab` (≥ 85 judgments over the week) | **≈ 24 h of machine time** + your A/B time |

**Pass rules** (`evals/gate.py`, encoded so that nobody, including future you, can rationalize a regression):

- **Hard fails:**
  - peak memory at 128K > cap − 2 GiB;
  - any T0 failure;
  - tool-call parse failure > 1 % on BFCL;
  - RULER@128K < 0.9 × the baseline build's score;
  - termination failures (no EOS within `max_tokens` on short prompts) > 0.5 %.
- **KL:** the upper 95 % CI of ΔKL (B − A) < 0 for promotion as "better". Any domain's mean KL worse by > 10 %
  means REVIEW.
- **Tasks:** no McNemar-significant regression (χ² > 3.84 with c < b) on code, math or tools. Flip rate reported.
- **Speed:** decode tok/s at 1K ≥ 0.95 × baseline. TTFT on an APC hit ≤ 1.2 × baseline.
- **Preference:** the A/B win rate CI must not lie entirely below 50 %.
- **Replay:** argument validity ≥ baseline − 1 point; live tool-error rate (from the next week of logs) not worse
  by > 20 % relative. That check is *post-promotion*, with automatic rollback (§4.9).

### 4.9 Operations

**launchd service.**

- The existing `scripts/launchd/install.sh` and `com.local-ai.serve.plist.template` already render a LaunchAgent with
  `--no-preload` and `--idle-unload-s`.
- Upgrade it to **socket activation plus exit-on-idle** (`--idle-exit-s 900`):
  - add `<key>Sockets</key><dict><key>Listeners</key><dict><key>SockNodeName</key><string>127.0.0.1</string><key>SockServiceName</key><string>8642</string></dict></dict>`;
  - drop `RunAtLoad`;
  - in `serve/cli.py`, if `launch_activate_socket` returns fds, hand uvicorn the socket (`uvicorn.Server(config).run(sockets=[sock])`) instead of binding;
  - after the idle timeout, finish in-flight work, flush APC to SSD, and `exit(0)`.
- Result: **0 GiB while idle**, no in-process reload churn, and a crash restarts through KeepAlive on non-zero exit.
- Cold start: load ≈ resident bytes ÷ effective read rate. **[assumption] 5 GB/s:** 38.6 GiB → **8 s**; 68.4 GiB →
  15 s; 89.6 GiB (GLM) → 19 s. Add warmup. Measure with `evals.speed --load-only`.
- The SSD APC tier means the next Claude Code turn after a cold start still finds its prefix.

**Memory-pressure watchdog** (`serve/watchdog.py`, a thread in the server). It polls every 2 s:
`kern.memorystatus_vm_pressure_level`, `vm.swapusage`, `mx.get_active_memory()` and `mx.get_cache_memory()`.

| Signal | Action | Hysteresis |
|---|---|---|
| pressure = 2 (warn), or swap +1 GiB in 5 min | Spill the APC RAM tier to SSD, `mx.set_cache_limit(0)` + `mx.clear_cache()`, lower `prefix_cache_memory` to 1 GiB, refuse new requests whose `est_bytes` > 4 GiB (503 with "memory pressure; try again / use RAG") | restore after 5 min at level 1 |
| pressure = 4 (critical) | Finish the current token stream with `stop_reason: "max_tokens"` and an error note. Exit the process: launchd restarts it on demand, and APC on SSD preserves the session | — |
| `active + cache` > cap − 1 GiB | existing `MemoryGuard` path | — |

Log every action, plus a macOS notification on critical. This turns "my IDE froze" into a log line saying the server
gave memory back.

**Thermals and power.**

- *Work profile:* Automatic is fine, because decode is bandwidth-bound (mac §5).
- *Overnight calibrate/gate:* plug in, set High Power in System Settings, and run
  `caffeinate -dimsu -w <pid>` for the job. Record `system_profiler SPPowerDataType | grep -i "high power"`,
  `pmset -g therm` and `pmset -g | grep powermode` in every `results/*/speed.json` (mac §Rec 8).
- Sample `pmset -g therm` every 60 s in the watchdog. On any CPU or GPU thermal warning, log it and tag
  concurrent speed results as invalid.

**Disk hygiene** (~2.5 TiB free now; plan for ~1.5 TB of checkpoints).

| Item | Size | Policy |
|---|---|---|
| HF sources (`~/.cache/huggingface/hub`): GLM FP8 328 GB, Qwen BF16 360 GB | ~690 GB | Keep while calibrating. Afterwards keep only `config.json`/tokenizer plus `calib/` outputs, if disk is needed (re-download is ~hours) |
| `calib/` work dir (transient hidden states) | 2 × T × width × 2 B: Qwen width 10,240, GLM 16,384. At T = 4M tokens: **164 GB / 262 GB** [computed] | Deleted layer by layer by `calib.run`. Make sure ≥ 1.2× that is free before starting |
| `artifacts/<m>/calib` layer stats + `ref/` | GBs | Keep. They are what makes rebuilding cheap |
| Candidate builds `models/<name>` | 40–90 GiB each | Keep ≤ 3 candidates + promoted + previous promoted (rollback) |
| **Shared Qwen n-gram shards** | **29.80 GiB per build** | `build/` writes the PLE table byte-identically across variants → **APFS clone** (`cp -c`) or hardlink them. Saves ~30 GiB per extra Qwen build [computed] |
| APC SSD tier | ≤ 32 GiB (`--prefix-cache-disk-gb`) | Already keyed by fingerprint and stale dirs pruned (`Engine._prune_stale_prefix_caches`) |
| Logs | small | 30-day rotation |

`scripts/disk_report.py` (to build) prints this table from the filesystem and flags orphans: models not referenced
by `results/` or the launchd plist, `work/` dirs from dead runs, and `hf cache prune` candidates. Use `hf cache ls`,
`hf cache rm`, `hf cache prune` (available in the installed `hf` CLI [measured]).

**Rollback.** `models/current` → symlink to the promoted build; `models/previous` → the one before. The plist points
at `models/current`. Rollback is `ln -sfn` plus `launchctl kickstart -k gui/$UID/com.local-ai.serve`. The
APC SSD tier is fingerprint-keyed, so it never serves stale KV to a different build.

### 4.10 Security

| Risk | Today | Fix |
|---|---|---|
| Network exposure | `--host 127.0.0.1` default ✔ | Keep. For remote use, an SSH tunnel or Tailscale ACL, never `0.0.0.0` |
| Any local process or web page can use the model | `--api-key` optional; CORS `allow_origins=["*"]` in `serve/app.py` | **Require** a key in launchd mode (read from Keychain at start). Remove the CORS middleware or restrict it to no origins. A browser page on any site could otherwise drive the API with `*` |
| DNS rebinding (an attacker's domain resolving to 127.0.0.1) | no Host check | Middleware: reject unless `Host` ∈ {`127.0.0.1:8642`, `localhost:8642`} |
| Unauthenticated request triggers a 40–90 GiB load (memory DoS) | `ActivityMiddleware` loads the model **before** the router's auth dependency runs | Check the Bearer key in the outer middleware before `ensure_loaded` |
| Prompt injection via RAG documents or tool results → harmful tool calls | — | Claude Code's permission prompts remain the boundary. Do **not** combine a local model with `--dangerously-skip-permissions` on untrusted documents. Tag RAG passages in the prompt as untrusted data |
| Secrets in logs, transcripts and calibration data | — | No prompt text in logs by default. Scrub transcripts (§4.8). `logs/` and `artifacts/personal/` at 0700/0600. Never publish a model calibrated on personal data without checking for memorized secrets: a canary probe of the top secrets regexes as generation prompts |
| Mutating endpoints | `REMOVED_ROUTES` drops `PATCH /v1/settings` ✔; `/serve/unload` requires the key ✔ | Keep. Add `/serve/profile` only behind the key |

### 4.11 Module map and build backlog

| Component | Exists in repo | Must still be built | Effort [estimate] | Priority |
|---|---|---|---|---|
| Memory-capped server | `serve/{config,memory,engine,app,cli}.py`, `scripts/serve.sh` | resident-aware accounting; per-request admission; wiring default 0; helper kinds allowed; SSE ping test | 1–2 days | **P0** |
| GLM projected-cache fix | (probe only, decode §3) | `serve/glm_fix.py` + test that it is bit-exact on tiny GLM | 0.5 day | **P0 for GLM / all-in** |
| Security | key + localhost | auth-before-load, Host check, CORS removal, Keychain in launchd | 0.5 day | **P0** |
| Router | single pinned model, `build_gen_args` hook | `serve/router.py` (aliases, effort, two-phase thinking budget, context errors) | 1 day | P1 |
| Observability | `/serve/status`, mlx-vlm `/metrics` | `serve/obs.py` JSONL log, `serve.report`, tool-error mining | 1 day | P1 |
| Watchdog | `MemoryGuard`, idle unload | `serve/watchdog.py` (pressure, swap, thermal) | 0.5–1 day | P1 |
| launchd | `scripts/launchd/*` | socket activation + exit-on-idle, `scripts/profile.sh` | 1 day | P1 |
| Tool reliability | native parsers via mlx-vlm, protocol tests | `serve/tools_guard.py` (validate, coerce, resample); `serve/grammar.py` (lazy Lark, MTP off in segment); extra tests | 2–3 days | P1 (guard) / P2 (grammar) |
| RAG | — | `rag/` package, `evals/rag.py`; `pypdf` (new dep) | 3–4 days | P2 |
| Eval gate | `evals/{kl,speed,code,math,longctx,tools,compare}.py`, sandbox | `evals/{gate,flips,lcb,swe,replay,ab,rag}.py`, BFCL multi-turn, KL bootstrap | 4–6 days | P1 (gate, flips, replay) / P2 (swe, lcb, ab) |
| Personal data | — | `calib_data/transcripts.py` (extract + scrub + render) | 1–2 days | P1 (shared with calibration) |
| Disk / ops | — | `scripts/disk_report.py`, APFS-clone shared shards in `build/`, `models/current` symlink | 0.5 day | P2 |

New dependencies to list (not added to `pyproject.toml`): `pypdf` (RAG PDFs); optional `mcp` (RAG as an MCP
server); optional `tree-sitter` + grammars (code chunking); `mini-swe-agent` + Docker Desktop (SWE eval, separate
venv recommended). `llguidance` is already installed.

---

## 5. Decision criteria & what to measure

| Decision | Measure | Rule |
|---|---|---|
| Trust resident-aware accounting | `footprint` IOAccelerator vs computed resident; decode under `memory_pressure -l warn` | \|Δ\| < 1 GiB and decode within 10 % → switch `check_fits`; else keep disk bytes |
| Spend the freed ~25 GiB | §7.1 builds: KL / tools / replay at equal speed | Higher experts K or bits if the T1 gate is better; else APC RAM + 4B reranker |
| Prefill step 2048 vs 4096 | `evals.speed` prefill tok/s and peak at 128K | 4096 only if ≥ +10 % prefill and peak ≤ cap − 4 GiB |
| MTP block size | `scripts/tune_runtime.py` + α from logs | Keep adaptive default unless a fixed γ is ≥ 5 % faster across code/agentic/reasoning |
| Separate fast model | `queue_wait_s` and APC miss caused by the Haiku slot | > 10 % main-loop wait or > 5 % APC misses → add Qwen3.5-2B helper |
| Grammar on | BFCL irrelevance + live malformed-call rate vs wrong-call rate | Turn on if malformed > 0.5 % of calls; keep if `grammar_mass_removed` p95 is small and task scores do not drop |
| Embedder | `evals/rag.py` recall@20 after rerank on your queries | Switch from EmbeddingGemma-2 to Qwen3-Embedding only if it gains ≥ 3 points |
| Reranker on code | recall@20 with and without rerank, code subset | Skip rerank for code-only indexes if it does not help (the 0.6B card is *below* its embedder on MTEB-Code) |
| Long context vs RAG thresholds | RULER per length on the promoted build; TTFT | Table in §4.5; re-derive after every promotion |
| Promote a build | `evals.gate` T2 | §4.8 pass rules, no exceptions |
| Profile all-in with GLM | Peak at 128K with the fix; T2 vs the work build | Peak ≤ 100 GiB and T2 better on your priority domains |

Always record, next to every number: build fingerprint, profile, power mode, thermal state, `vm.swapusage`, and
whether other GPU jobs were running. Results taken while the GPU is shared are tagged `contended` and excluded from
gates.

---

## 6. Failure modes

| Symptom | Likely cause | Detection | Fix |
|---|---|---|---|
| Claude Code aborts long requests mid-prefill | No SSE ping during a 30–120 s prefill | client "idle timeout"; log shows ttft > 60 s | ping events; route big docs to RAG |
| Every turn is slow (TTFT ≈ full prefill) | Prefix not byte-stable (attribution header, timestamps, tool-list order) or APC evicted by Haiku-slot calls | `apc_hit=false` on consecutive `local-main` turns | `CLAUDE_CODE_ATTRIBUTION_HEADER=0`; more `prefix_cache_entries`; separate fast helper |
| 401 from Claude Code | Used `ANTHROPIC_API_KEY` (sent as `x-api-key`) | server log | use `ANTHROPIC_AUTH_TOKEN` |
| Server refuses a model that "should fit" | Disk-byte accounting counts the mmapped n-gram table | `/serve/status` weights 68.44 GiB | resident-aware accounting (§4.2a) after verification |
| GLM all-in OOM or swap at ≥ 32K | Projected MLA cache (+22–24 GiB) or the 64 KiB × T prefill transient | peak in logs | `serve/glm_fix.py`; admission includes the transient |
| Kernel panic (IOGPU `prepare count underflow`) | Wiring + concurrency + cache churn (mac §3) | machine reboot, panic log | wiring off, 1 seq, one MLX process |
| Speculative decoding "on" but no speedup | Build has no MTP head (sh0wie REAP-288), or a stale head after pruning | `speculative.active=false` or α < 0.5 | `build/` keeps and prunes MTP consistently (reap §Rec 7) |
| Thinking budget ignored | MTP active → `thinking_budget` dropped [src] | `think_tokens` > budget | two-phase budget (§4.4) |
| Grammar + MTP emits invalid tool syntax | Verify path skips logits processors [src] | grammar segments with `tool_repair=failed` | MTP off inside segments |
| Grammar makes answers worse | Constraint applied during reasoning, or a wrong grammar | task-score drop, high `grammar_mass_removed` | lazy trigger only after `<tool_call>`; tests per tool schema |
| Eval scores look great, users unhappy | Contaminated benchmarks (LCB v6 predates the models), or a judge with position/verbosity bias | A/B vs scores disagree | relative-only use of contaminated sets; blind A/B with order randomization; personal replay |
| Gate passes a regression | Unpaired comparisons, too-small n, tuned on test | — | paired McNemar, bootstrap CIs, time-split replay data |
| IDE stutters while the model runs | Memory pressure from KV/APC growth, or long command buffers on the GPU | watchdog pressure=2 events | watchdog spill; prefill step 2048; cap |
| Disk fills overnight | `calib` work dir (164–524 GB) plus several builds | `disk_report` | pre-flight free-space check in `calib.run`/`build.make`; APFS clones |
| Secrets in a calibration set or model | Unscrubbed transcripts | canary probe | scrubber + 0600 + never publish personal-calibrated weights |
| A web page drives your local model | CORS `*` without a key | access log with an `Origin` header | key required; CORS off; Host check |

---

## 7. Stretch ideas (push limits)

### 7.1 Spend the PLE dividend: unpruned (or lightly pruned) Qwen experts inside 80 GiB

If §4.2a verifies that the n-gram table stays file-backed, the binding constraint is **resident** bytes, not disk
bytes. Routed experts at 4.5 bpw cost 2.765 MB each [computed: 3 × 2560 × 640 × 4.5 / 8]. All 512 × 48 experts take
**63.3 GiB** (sh0wie's own REAP-512 4-bit at 98 GB disk minus 29.8 GiB of n-gram gives ≈ 61.5 GiB).

Resident ≈ 63.3 experts + 3.05 other + ~1 MTP = **≈ 67.4 GiB**. Add 3.9 KV/state (with MTP) + 2 cache + ~3
activations, and the total is **≈ 76 GiB ≤ 80**, with APC mostly on SSD.

That is a *no-pruning* 4-bit Qwen in the work profile. It contradicts PLAN.md's "neither unpruned model fits at
≥ 3 bits" only because PLAN's budget counted the n-gram table as resident. Alternatives for the same bytes: K = 384
at ~5 bpw, or K = 448 at 4.5 bpw with the allocator's floors. Decide with the T1 gate, especially the rare-name and
tool probes where pruning hurts most (reap §Rec 8).

### 7.2 Personal calibration and personal routing

- Use the time-split transcript episodes (older than 2 weeks) as an extra `calib_data` domain, `personal`, for REAP
  saliency and imatrix/sensitivity. Experts that *your* repos and tools use then survive pruning.
- Re-use the replay labels later to train a tiny RouteLLM-style classifier (logistic regression on prompt features
  plus alias, effort and history length) that predicts "needs deep". The training signal comes from A/B outcomes
  and live tool-error rates.

### 7.3 SSD expert streaming for the full GLM-5.3-Flash

`moe_offload` [src] streams experts from per-expert mmapped files with an LRU cache. GLM's routed experts at 4.5 bpw
are 171 GB in total; one token touches 8 × 42 experts × 14.16 MB = **4.76 GB** [computed]. With hit rate h and SSD
read rate S, the decode ceiling from misses alone is S / ((1 − h) · 4.76 GB):

| h | S = 3 GB/s | S = 6 GB/s | S = 10 GB/s |
|---|---|---|---|
| 0.90 | 6.3 tok/s | 12.6 tok/s | 21.0 tok/s |
| 0.95 | 12.6 tok/s | 25.2 tok/s | 42.0 tok/s |
| 0.98 | 31.5 tok/s | 63.1 tok/s | 105 tok/s |

These are upper bounds; the compute and attention reads come on top. The self-reported reality check is OptiQ's
SSD-streamed Qwen at ~3.8 tok/s on an M3 Max (quant §2). **Estimate h before building:** `calib/` already writes
per-layer `expert_freq`. The mass of the top C experts per layer is a lower bound on the hit rate of a frequency-
pinned cache of C experts:

```
h_static(C) = Σ_layers Σ_{top-C} freq  /  Σ_layers Σ freq
```

Choose C so that C × 42 × 14.16 MB fits the cache budget. 45 GiB holds ≈ 3,400 experts ≈ 81 per layer, 28 % of 288.
Then measure S with a cold-cache read of the expert store. Use it for "ask the 321B model one hard question
overnight", not for Claude Code.

### 7.4 Grammar-aware speculative verification

Implement masked verification in `speculative/mtp.py`'s verify step: apply the llguidance mask to the target logits
at each drafted position, and reject a draft token that falls outside A_t. Exactness then holds for the masked
distribution (§2.6), and tool calls keep their MTP speedup (~1.4 s saved per 150-token call [computed]). Upstream
this to mlx-vlm.

### 7.5 Session-aware prefetch

Claude Code turns arrive seconds to minutes apart. During idle gaps, pre-restore the most recent session's APC
checkpoint from SSD into the RAM tier. Then the next turn's TTFT is just the prefill of Δ new tokens.

### 7.6 Late chunking for long documents

Embed whole sections up to the embedder's 8K (EmbeddingGemma-2) or 32K (Qwen3-Embedding) context, then mean-pool
token embeddings per chunk span (late chunking, §3). Each chunk vector then "knows" its surrounding context with no
LLM blurb cost. A/B it against contextual retrieval in `evals/rag.py`. It needs token-level outputs from the
embedder, a small change to `_embed`.

### 7.7 Self-distilled "fast lane"

Train a Qwen3.5-2B (or a DFlash2-style drafter) on *your* replay episodes, with outputs generated by the promoted
build, in the calibrate profile with layer-band LoRA. The helper then matches the big model on your Haiku-slot
traffic, and as a drafter it would raise α on your distribution. Gate it like any build: KL vs the promoted model on
held-out replay, plus A/B.

### 7.8 Continuous evaluation from real use

Every week, sample 20 real `local-main` requests from the log. Prompts are stored only when you opt in with
`--log-prompts`. Re-run them on the candidate build and queue them into `evals.ab`. Over a month that yields ~85
non-tie judgments, enough to detect a 65 % preference [computed in §2.7], drawn from exactly the traffic you care
about.
