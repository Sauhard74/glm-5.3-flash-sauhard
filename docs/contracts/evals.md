# evals/ contract

Owner: `evals/` (+ `tests/evals/`). Everything runs through `uv run`.

```
uv run python -m evals.<suite> --model models/<name> [common flags] [suite flags]
uv run python -m evals.all     --model models/<name> [--suites kl,speed,...] [--smoke] [-- extra flags]
uv run python -m evals.compare [--results-dir results] [--models a,b] [--out results/compare.md]
```

Suites: `kl`, `speed`, `code`, `math`, `longctx`, `tools`. Each writes
`results/<name>/<suite>.json`, where `<name>` = `--name` or the basename of `--model`.

## Inputs

**Model dir** (output of `build/`, or any `mlx_vlm.load`-able dir):
- `config.json` with `model_type` `qwen4_exp` (family `qwen`) or `glm5_next` (family `glm`); other types run as family = model_type.
- tokenizer files and the model's own `chat_template.jinja` (used to render prompts and native tool calls).
- `generation_config.json` (sampling defaults). Missing keys fall back to the upstream values
  (qwen: T=1.0 top_p=0.95 top_k=20; glm: T=1.0 top_p=0.95; other: T=0.6 top_p=0.95 top_k=20).
- Native MTP weights in the same safetensors (`mtp.*` for qwen, `layers.<num_hidden_layers>.*` for glm)
  are picked up through `mlx_vlm.speculative.drafters.load_drafter`; `--draft-model` overrides.

**KL reference** — read exactly as calib writes it (`--ref`, default `artifacts/<family>/calib/ref`):
- `ref/index.json`: `{"model", "topk", "src", "samples": [{"id", "domain", "file", "n_tokens", ...}]}`
- `ref/<file>.npz`: `tokens` int32 `[L]`, `top_idx` int32 `[L, k]`, `top_logprob` float32 `[L, k]`
  (full-vocab log-softmax, any order), optional `target_logprob` float32 `[L]` (NaN where no target).
  Row `t` is the distribution after `tokens[:t+1]`, i.e. predicting `tokens[t+1]`.
  The reference carries its own tokens, so `eval.jsonl` is not read by `kl`.

## Common flags

| flag | default | meaning |
|---|---|---|
| `--model` | required | model dir (or HF repo id) |
| `--name` | basename of model | results sub-directory |
| `--results-dir` / `--out` | `results` | output root / explicit json path |
| `--limit N` | suite-specific | items per dataset / category / (task, length) cell / KL domain |
| `--smoke` | off | offline fixtures, tiny limits, greedy, budget 16 + 24 answer tokens |
| `--seed` | 0 | `mx.random.seed` and synthetic-task seeds |
| `--resume` | off | reuse finished items from `<suite>.partial.jsonl` if settings are unchanged |
| `--memory-limit-gb` | 80 | `mx.set_memory_limit` for the process |
| `--max-weights-gb` | 80 (env `LOCAL_AI_MAX_WEIGHTS_GB`) | refuse to load bigger checkpoints |

Generation suites (`code`, `math`, `longctx`, `tools`) also take:
`--thinking/--no-thinking`, `--thinking-budget`, `--answer-tokens`, `--max-tokens`,
`--temperature/--top-p/--top-k/--min-p`, `--greedy`, `--reasoning-effort`, `--mtp {auto,on,off}` (default auto),
`--draft-model`, `--prefill-step-size` (2048).

| suite | thinking budget | answer tokens | default size |
|---|---|---|---|
| code | 16384 | 4096 | all HumanEval+ (164) + MBPP+ (378) |
| math | 24576 | 2048 | AIME24 (30) + AIME25 (30) + MATH-500 strided 100 (`--math500-n`), `--samples 1` |
| longctx | 1024 | 256 | 4 tasks x {32K, 64K, 128K} x 8 |
| tools | 2048 | 1024 | 50 per category |

**Thinking budget.** Prompts are rendered with the model template (`enable_thinking`, GLM gets a
closed `<think></think>` when thinking is off). If the first `thinking_budget` tokens do not close
`</think>`, generation stops, the Qwen3-report early-exit text
(`"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"`)
is appended and generation continues for `answer_tokens` after re-prefilling prompt + reasoning.
This is done in `evals/generation.py` because mlx-vlm cannot force tokens on the MTP path.
Answers are always parsed from the text after `</think>`.

## Suites

- **kl** — chunked prefill (`--chunk 1024`) with a fresh cache per sample; per token
  `KL(P_ref||Q)` on the top-k support plus one tail bucket (lower bound of full KL; exact when k = V),
  top-1 agreement (`argmax Q == argmax P_ref`), NLL of the actual next token. Reported overall
  (token-weighted), per domain, `macro_*` (domain-averaged), percentiles, `ppl_ratio = exp(nll - ref_nll)`.
  `--smoke` without a ref builds a self-reference from the model (KL ~ 0).
- **speed** — contexts `--contexts 1024,8192,32768,131072`, `--decode-tokens 256`, `--mtp-modes off,on`
  (on is skipped with the reason recorded when no drafter loads), greedy with EOS suppressed, synthetic
  prose prompt with thinking off. Per run: `ttft_s`, `prefill_tps = prompt_tokens/ttft`,
  `decode_tps = (n-1)/(t_last - t_first)`, `peak_mlx_gb` (`mx.get_peak_memory`), `peak_rss_gb` (sampled
  every 50 ms), `cache_gb` (sum of prompt-cache `nbytes`: KV + linear-attention state), plus
  `weights_gb` / `resident_gb`. Two warmup runs precede measurement; `--repeats` keeps the best.
- **code** — HumanEval+ (`evalplus/humanevalplus`) and MBPP+ (`evalplus/mbppplus`). Last ```python block
  of the answer; HumanEval bodies without the `def` get the prompt prepended. Each program runs in
  `evals/sandbox.py`: `python -I -B` in a temp dir under `sandbox-exec` (no network, writes only to the
  temp dir), socket monkeypatch, CPU/FSIZE rlimits, `--timeout 20` s wall clock with process-group kill,
  `--workers 4`. `pass@1_plus` (evalplus tests) for both, `pass@1_base` (`test_list`) for MBPP+.
- **math** — prompt `"{problem}\n\nPlease reason step by step, and put your final answer within \boxed{}."`;
  last `\boxed{}` of the answer (falls back to "final answer is ..." and then the reasoning tail);
  AIME graded as integers, MATH-500 with normalized LaTeX / numeric / tuple equivalence (no sympy).
- **longctx** — RULER-style, offline and deterministic per `(seed, task, length, i)`: `niah_single`,
  `niah_multikey` (4 distractor needles), `variable_tracking` (5-hop chain + 5-hop distractor chain),
  `qa` (one fact among thousands of similar synthetic documents). Prompts are filled to within 1% of the
  target length measured with the model's tokenizer + template. Score = recall of expected strings in the
  answer (whole output if thinking never closed). Lengths above `max_position_embeddings` are skipped.
- **tools** — BFCL v3 (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) categories
  `simple,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple` (any BFCL single-turn
  category works). Tools are converted to OpenAI/JSON-schema form, rendered by the chat template, and the
  answer is parsed with mlx-vlm's parser for that template (`qwen3_coder` / `glm47`). AST-style check:
  function name (`.` == `_`), required params, no unexpected params, values against BFCL's acceptable
  lists (strings normalized like BFCL; scalars also match via their string form since XML formats are
  stringly typed), any-order matching for parallel, no call for irrelevance. Items whose ground truth
  contradicts their own schema are excluded (28, `INCONSISTENT_IDS`, `--keep-inconsistent` to keep).

`--oracle` (code, math, tools) scores reference solutions without loading the model, to validate the
harness: tools = 100% on all 9 single-turn categories in both native formats; math = 100%;
code = 99.6% (HumanEval/32's evalplus test is broken in the HF copy, Mbpp/590 differs in the last float digit).

## Output: `results/<name>/<suite>.json`

```json
{
  "suite": "math", "model": "<name>", "model_path": "...", "family": "qwen|glm|...",
  "created": "ISO-8601", "smoke": false, "limit": null,
  "settings": {"thinking": true, "thinking_budget": 24576, "max_tokens": 26624,
               "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
               "mtp": "on | off | unavailable (<reason>)", "...": "suite specific"},
  "summary": {"<metric>": 0.0, "headline": "<metric key>"},
  "breakdown": {"<dataset|category|domain|context>": {"<metric>": 0.0}},
  "samples": [{"id": "...", "...": "per item record incl. text, reasoning, answer, tokens, timings"}],
  "runtime": {"seconds": 0.0, "peak_memory_gb": 0.0, "versions": {"mlx": "...", "mlx-vlm": "..."}}
}
```

Summary keys (all flat scalars; `compare.py` treats keys matching
`kl|ttft|peak|cache|nll|ppl_ratio|weights_gb|resident_gb|no_call|truncated|no_answer` as lower-is-better):

| suite | headline | other summary keys |
|---|---|---|
| kl | `mean_kl` | `top1_agree`, `median_kl`, `p90_kl`, `p99_kl`, `nll`, `ref_nll`, `ppl_ratio`, `macro_mean_kl`, `macro_top1_agree`, `<domain>_kl`, `tokens`, `samples` |
| speed | `decode_tps@<first ctx>` | `prefill_tps@C`, `decode_tps@C`, `decode_tps_mtp@C`, `ttft_s@C`, `peak_gb@C`, `cache_gb@C`, `peak_gb`, `weights_gb`, `resident_gb` |
| code | `pass@1` | `humaneval+_pass@1`, `mbpp+_pass@1`, `n` |
| math | `accuracy` (mean over datasets) | `aime24_acc`, `aime25_acc`, `math500_acc`, `n` |
| longctx | `score` | `score@<len>`, `<task>`, `n` |
| tools | `accuracy` (mean over categories) | `<category>_acc`, `n` |

While a generation suite runs, finished items are appended to `results/<name>/<suite>.partial.jsonl`
(first line `{"fingerprint"}` of suite + settings); it is deleted when the final json is written.

## Data

Benchmarks are fetched on first use with `huggingface_hub.hf_hub_download(..., repo_type="dataset")`
into the normal HF cache (dataset repos only). `--smoke` uses `evals/fixtures/*.jsonl` (same raw schema,
no network). Long-context tasks are generated, not downloaded.

## Tests

`uv run pytest tests/evals -q` builds tiny random checkpoints of the real `qwen4_exp` / `glm5_next`
classes with native MTP heads and the real tokenizers (`tests/evals/tiny_models.py`), and runs every
suite end to end in smoke mode on both, plus unit tests for parsing, sandboxing, BFCL checking, KL math,
context building, resume and the weight guard.
