"""Write site/build/bench.json: every benchmark run on GLM 5.3 Flash-sauhard and its sibling builds (summaries only).

uv run python site/build/export_bench.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SHIP = "glm-reap153-76-lsq"

# KL A/B: every candidate build, against full-precision GLM-5.3-Flash on 48 held-out samples
BUILDS = [
    ("glm-candidate-auto76", "REAP 153, plain rounding", 76),
    ("glm-mass153-76", "Mass-scored 153, plain rounding", 76),
    ("glm-massauto-80", "Mass-scored 162, plain rounding", 80),
    ("glm-massauto-80-lsq", "Mass-scored 162, LSQ", 80),
    (SHIP, "REAP 153, LSQ (shipped)", 76),
    ("glm-reap153-76-lsq-dwq", "Shipped + DWQ (KL loss)", 76),
    ("glm-reap153-76-lsq-dwqmse", "Shipped + DWQ (MSE loss)", 76),
]


def load(path):
    return json.loads(path.read_text())


kl = []
for name, label, budget in BUILDS:
    s = load(RES / f"{name}-l12/kl.json")["summary"]
    kl.append({
        "build": name, "label": label, "budget_gb": budget,
        **{k: round(s[k], 4) for k in ("assistant_kl", "assistant_top1", "reasoning_kl", "tool_kl", "user_kl", "system_kl",
                                      "code_kl", "agentic_kl", "longctx_kl", "top1_agree", "ppl_ratio")},
        "tokens": s["tokens"], "assistant_tokens": s.get("assistant_tokens"),
    })

ship_kl = load(RES / f"{SHIP}-l12/kl.json")
domains = {d: {k: round(v, 4) for k, v in b.items() if isinstance(v, float)} | {"tokens": b["tokens"], "samples": b["samples"]}
           for d, b in ship_kl["breakdown"].items()}

tasks = {}
for build in (SHIP, "glm-reap153-76-lsq-dwqmse"):
    tasks[build] = {}
    for suite in ("code", "math", "tools"):
        d = load(RES / build / f"{suite}.json")
        st = d["settings"]
        tasks[build][suite] = {
            "summary": d["summary"],
            "breakdown": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in d["breakdown"].items()},
            "settings": {k: st[k] for k in ("thinking_budget", "max_tokens", "sampling") if k in st},
        }

speed_path = RES / SHIP / "speed.json"
speed = load(speed_path)["summary"] if speed_path.exists() else None

summary = load(ROOT / f"models/{SHIP}/build_summary.json")
out = {
    "ship": SHIP,
    "runtime": ship_kl["runtime"]["versions"],
    "kl_eval": {"samples": ship_kl["summary"]["samples"], "tokens": ship_kl["summary"]["tokens"], "ref_topk": ship_kl["settings"]["ref_meta"]["topk"]},
    "kl": kl,
    "domains": domains,
    "tasks": tasks,
    "speed": speed,
    "build": {"avg_bits_per_weight": round(summary["avg_bits_per_weight"], 3), "written_gib": {k: round(v / 2**30, 2) for k, v in summary["written_bytes"].items()},
              "bits_by_params": summary["bits_histogram_params"], "rounding": summary["rounding"]["method"]},
}
(Path(__file__).parent / "bench.json").write_text(json.dumps(out, indent=1))
print("wrote bench.json", {k: len(v) if isinstance(v, (list, dict)) else v for k, v in out.items()})
