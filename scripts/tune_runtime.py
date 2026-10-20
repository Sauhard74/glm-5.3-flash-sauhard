"""Load once, sweep MTP speculative settings on coding/agentic/reasoning prompts, pick the fastest."""

import json
import os
import sys
import time

import mlx.core as mx

GB = 1 << 30
_set_wired = mx.set_wired_limit
mx.set_wired_limit = lambda n: _set_wired(0)
mx.set_memory_limit(76 * GB)
mx.set_cache_limit(2 * GB)

from mlx_vlm import generate, load  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.speculative.drafters import load_drafter  # noqa: E402

MODEL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/runtime_tune.json"
MAX_TOK = int(os.environ.get("MAX_TOK", "384"))

PROMPTS = {
    "code": "Implement an LRU cache in Python with O(1) get/put, thread safety, and a TTL per entry. Include pytest tests.",
    "refactor": "Refactor this to async with proper error handling and retries:\n\n```python\nimport requests\n\ndef fetch_all(urls):\n    out = []\n    for u in urls:\n        r = requests.get(u)\n        out.append(r.json())\n    return out\n```",
    "agentic": "You are a coding agent. The test `test_parse_date` fails with `ValueError: unconverted data remains: Z`. Explain the likely root cause in a datetime.strptime call and give the minimal patch as a unified diff.",
    "reasoning": "A train leaves at 9:40 at 72 km/h; a second leaves the same station at 10:15 at 96 km/h on the same track. When and where does the second catch the first? Show the steps briefly.",
}

model, processor = load(MODEL)
print(f"loaded: active={mx.get_active_memory()/GB:.1f} GiB", flush=True)
DRAFTER = os.environ.get("DRAFTER", MODEL)
drafter, kind = load_drafter(DRAFTER, kind="mtp")
print(f"drafter: {kind}", flush=True)

configs = [("plain", None, None), ("mtp-auto", drafter, None)] + [
    (f"mtp-b{b}", drafter, b) for b in (2, 3, 5, 6)
]

results = []
for name, draft, block in configs:
    row = {"config": name, "per_prompt": {}}
    for pname, text in PROMPTS.items():
        prompt = apply_chat_template(processor, model.config, text, num_images=0, enable_thinking=False)
        kw = dict(max_tokens=MAX_TOK, temperature=0.0, verbose=False)
        if draft is not None:
            kw.update(draft_model=draft, draft_kind="mtp")
            if block:
                kw["draft_block_size"] = block
        t = time.perf_counter()
        r = generate(model, processor, prompt, **kw)
        row["per_prompt"][pname] = {
            "gen_tps": round(r.generation_tps, 1), "prompt_tps": round(r.prompt_tps, 1),
            "gen_tokens": r.generation_tokens, "wall_s": round(time.perf_counter() - t, 2),
            "peak_gb": round(r.peak_memory, 1), "head": r.text[:80],
        }
    tps = [p["gen_tps"] for p in row["per_prompt"].values()]
    row["mean_gen_tps"] = round(sum(tps) / len(tps), 1)
    row["peak_gb"] = max(p["peak_gb"] for p in row["per_prompt"].values())
    results.append(row)
    print(json.dumps({k: row[k] for k in ("config", "mean_gen_tps", "peak_gb")}), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
best = max(results, key=lambda r: r["mean_gen_tps"])
print("BEST", best["config"], best["mean_gen_tps"], "tok/s")
