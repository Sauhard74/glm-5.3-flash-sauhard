# calib_data contract

```
uv run python -m calib_data.build --model {glm,qwen} --out artifacts/<model>/data/
uv run python -m calib_data.build --model qwen --out /tmp/smoke --smoke     # offline fixtures, ~5 s
```

## Outputs (in `--out`)

| file | content |
|---|---|
| `calib.jsonl` | calibration samples, grouped by domain (code, agentic, reasoning, longctx) |
| `eval.jsonl` | held-out samples; no source row/document is shared with `calib.jsonl` |
| `manifest.json` | args, tokenizer hashes, pinned source revisions, per-domain/per-recipe token stats |
| `preview/` | only with `--preview`: decoded text of the first sample of every recipe, per split |

One JSON object per line:

```json
{"id": "calib-agentic-0007-1a2b3c4d", "domain": "agentic", "source": "openhands",
 "src": "SWE-Gym/OpenHands-SFT-Trajectories:data/...parquet:0:123", "tokens": [248045, 8678, ...]}
```

- `id`: unique across both files. `domain` is one of `code|agentic|reasoning|longctx`.
- `source`: recipe name (below). `src`: stable key of the first source row used (for dedup/debugging).
- `tokens`: the full sequence, already chat-templated with the target model's own `chat_template.jinja`
  and tokenized with `add_special_tokens=False` (the template emits `[gMASK]<sop>` for GLM). Every id is
  `< vocab_size` from the model's `config.json`. No padding; lengths vary.
- Samples are complete conversations (assistant turns included) unless they end in a user turn,
  in which case the generation prompt (`<|assistant|><think>` / `<|im_start|>assistant\n<think>\n`) is
  appended. Overlong conversations are cut at the last turn boundary that fits; a single overlong turn is
  hard-truncated. Consumers should treat every token as a regular input position (no loss mask).

## Defaults

| domain | calib n | eval n | length (tokens) | approx calib tokens |
|---|---|---|---|---|
| code | 192 | 64 | <= 6144 | 0.38-0.41 M |
| agentic | 160 | 62 | <= 8192 | 0.74-0.77 M |
| reasoning | 144 | 62 | <= 8192 | 0.59-0.62 M |
| longctx | 16 | 12 | 16384-65536 | 0.54-0.56 M |
| total | 512 | 200 | | ~2.3 M calib, ~1.0 M eval |

Knobs: `--samples`, `--eval-samples` (scale the mix), `--mix`/`--eval-mix` (`code=192,agentic=160,...`,
relative weights, sample counts are largest-remainder rounded), `--max-len code=...,agentic=...,reasoning=...`,
`--longctx LO-HI`, `--seed`.

Determinism: output is byte-identical for the same args, tokenizer and seed. The calib/eval split does
not depend on the seed: a source row (narrativeqa: a document) goes to eval iff
`sha1("<source>:<key>") % 1000 < 300`, so changing seeds or counts never leaks eval rows into calib.
The seed only changes which rows are drawn and how they are composed.

## Recipes (all sources ungated, pinned to a commit in `calib_data/sources.py`)

| recipe | domain | source | what it produces |
|---|---|---|---|
| `gh_files` | code | codeparrot/github-code-clean (1 row group, 10k files, ~20 languages) | 1-4 real files + coding task (review, tests, refactor, port...); ends at generation prompt |
| `commit_edit` | code | bigcode/commitpackft (1 MB prefix of 15 language files) | 2-4 real commits as a session of edit requests; replies are the full new file or a unified diff |
| `openhands` | agentic | SWE-Gym/OpenHands-SFT-Trajectories | SWE coding-agent trajectories converted to native tool calls (`execute_bash`, `str_replace_editor`, `finish`; schemas parsed from the system prompt) |
| `toucan_multi` / `toucan_single` | agentic | Agent-Ark/Toucan-1.5M (Kimi-K2 subset) | real MCP tool-use conversations, `available_tools` as the tool schema |
| `openr1_math` | reasoning | open-r1/OpenR1-Math-220k | math with the R1 trace as `reasoning_content` |
| `openthoughts_science` / `_code` | reasoning | open-thoughts/OpenThoughts-114k (metadata) | physics/chemistry/biology/puzzle and competitive-coding traces |
| `narrativeqa` | longctx | deepmind/narrativeqa (test, 45 docs) | full book / movie script + 1-4 question/answer turns |
| `arxiv_multi` | longctx | ccdv/arxiv-summarization (test) | several papers concatenated as retrieved context, asks for one paper's abstract |
| `code_repo` | longctx | github-code-clean (disjoint half from `gh_files`) | many same-language files concatenated + a codebase question |

Tool calls are given to the templates as OpenAI-style `tool_calls` with dict `arguments` and rendered by
each model's own template: Qwen `<tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>...`,
GLM `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...`, tool results as
`<tool_response>` (Qwen in a user turn, GLM after `<|observation|>`). Reasoning traces go in
`reasoning_content` (`<think>...</think>`); assistant turns without a trace get the template's empty
think block. Reasoning samples whose full trace exceeds the cap are skipped 75% of the time
(the rest are truncated). Tool outputs are clipped to 12k characters (head + tail).
Templates run with default kwargs (GLM `Reasoning Effort: Max`, Qwen `xhigh`).

## Caching / network

- Source slices are read by byte range (one parquet row group with only the needed columns, or a jsonl
  prefix) through `HfFileSystem` and cached as `--cache-dir/<slug>-<hash>.jsonl.gz`
  (default `artifacts/cache/calib_data`; ~90 MB downloaded, ~200 MB on disk). `--offline` uses only the cache.
- The tokenizer is read from `--tokenizer` or `artifacts/tokenizers/<model>`; if missing,
  `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json`, ... are fetched from the
  model repo into that directory (never into the HF hub cache).
- No model weights are loaded; the build is CPU-only (~20 s per model once cached).

## Smoke

`--smoke` uses synthetic rows with the real schemas (`calib_data/fixtures.py`), 2 calib + 1 eval sample
per domain, caps code 1024 / agentic 2048 / reasoning 2048, longctx 2048-4096. `--smoke --source hf`
runs the same tiny sizes on the real sources.
