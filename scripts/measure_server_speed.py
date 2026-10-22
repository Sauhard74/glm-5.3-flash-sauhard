"""Time GLM 5.3 Flash-sauhard through the running server and write site/build/speed.json.

Sends one request per prompt length to the server on :8080 (start the glm profile from the dashboard first), reads
the server's own prefill/decode rates from logs/server.log, and adds the bare-model rows from results/<ship>/speed.json.

uv run python scripts/measure_server_speed.py
"""

import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODEL = "models/glm-5.3-flash-sauhard"
SHIP = "glm-reap153-76-lsq"
LOG = ROOT / "logs/server.log"
KEY = (ROOT / "control/api_key").read_text().strip()
DONE = re.compile(r"Request completed: .*prompt_tokens=(\d+) generated_tokens=(\d+) .*prefill=([\d.]+) tok/s decode=([\d.]+) tok/s")

PARA = ("The scheduler assigns each job to the least loaded worker, records the assignment in the ledger, and retries "
        "failed jobs with exponential backoff up to five times before marking them dead. ")
TASK = "Explain, step by step and in about 400 words, how this scheduler could lose a job, and how to fix it."


def ask(n_words: int) -> None:
    prompt = (PARA * max(1, n_words // len(PARA.split())))[: n_words * 7] + "\n\n" + TASK
    httpx.post("http://127.0.0.1:8080/v1/chat/completions", timeout=900,
               headers={"Authorization": f"Bearer {KEY}"},
               json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200,
                     "temperature": 1.0, "top_p": 0.95}).raise_for_status()


ask(30)  # warm-up, not recorded
time.sleep(1)
warm = LOG.stat().st_size
for words in (60, 750, 6000, 24000):
    ask(words)
time.sleep(1)
with LOG.open() as f:
    f.seek(warm)
    rows = [dict(zip(("prompt_tokens", "generated_tokens", "prefill_tps", "decode_tps"), (int(m[1]), int(m[2]), float(m[3]), float(m[4]))))
            for m in map(DONE.search, f) if m]

bare = json.loads((ROOT / f"results/{SHIP}/speed.json").read_text())
harness = [{"ctx": int(ctx), "prefill_tps": b["prefill_tps_mtp_off"], "ttft_s": b["ttft_s_mtp_off"], "decode_tps": b["decode_tps_mtp_off"],
            "peak_gb": b["peak_mlx_gb_mtp_off"]} for ctx, b in bare["breakdown"].items()]
out = {"server": rows, "harness": harness, "harness_decode": bare["settings"]["decode_tokens"], "resident_gb": bare["summary"]["resident_gb"]}
(ROOT / "site/build/speed.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
