# GLM 5.3 Flash-sauhard

GLM-5.3-Flash (321B mixture-of-experts) rebuilt to run on a 128 GB Apple Silicon Mac in 71.6 GB, with room left to
work. This repo holds the server, the control dashboard, and every step that made the weights.

It reads images too: the original BF16 vision tower is included (+1.13 GB, 73–76 GB resident during image requests),
and both APIs accept them (OpenAI `image_url` parts, Anthropic `image` blocks). Vision is spot-checked, not
benchmarked; see the [model card](https://huggingface.co/Sauhard74/GLM-5.3-Flash-sauhard-MLX).

- Write-up: https://glm-flash-sauhard.vercel.app
- Benchmarks: https://glm-flash-sauhard.vercel.app/report/
- Weights: https://huggingface.co/Sauhard74/GLM-5.3-Flash-sauhard-MLX

## Run it

Needs an Apple Silicon Mac with 128 GB of unified memory, macOS 14+, and 80 GB of free disk.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/Sauhard74/glm-5.3-flash-sauhard.git && cd glm-5.3-flash-sauhard
uv sync
uv run hf download Sauhard74/GLM-5.3-Flash-sauhard-MLX --local-dir models/glm-5.3-flash-sauhard
openssl rand -hex 24 > control/api_key
CAP_GB=84 MTP=0 scripts/quickserve.sh models/glm-5.3-flash-sauhard
```

The server speaks the OpenAI API (`http://127.0.0.1:8080/v1`) and the Anthropic API (`/v1/messages`); the `model`
field is the path above and the key is in `control/api_key`. For Claude Code:

```sh
MODEL_ID=models/glm-5.3-flash-sauhard scripts/claude-local.sh
```

`scripts/install-control.sh` installs the dashboard at http://127.0.0.1:8090 as a launchd service (start, stop,
memory, speed). The full recipe with troubleshooting is at https://glm-flash-sauhard.vercel.app/run/.

## Layout

| Path | What it does |
|---|---|
| `calib_data/` | builds the calibration and held-out eval sets from public Hugging Face datasets |
| `calib/` | streams the original model layer by layer, recording expert routing and per-tensor sensitivity |
| `build/` | prunes experts (REAP), allocates bits to a memory budget, rounds with LSQ, writes MLX weights |
| `recover/` | optional distillation passes (DWQ); measured, not shipped |
| `evals/` | KL against the original, code (HumanEval+, MBPP+), math (MATH-500), tools (BFCL), speed |
| `scripts/` | server launch (`quickserve.sh`, `run_server.py`), Claude Code wrapper, build and eval runs |
| `control/` | the dashboard and its daemon |
| `site/` | the write-up, benchmark report and recipe pages |
| `docs/` | research notes and the build plan |

## Rebuild from the original

About 600 GB of temporary disk for the source checkpoint, 5.5 hours of calibration on an M5 Max, 10 minutes to build.

```sh
uv run hf download zai-org/GLM-5.3-Flash --local-dir models/src-glm-5.3-flash
uv run python -m calib_data.build --model glm --out artifacts/glm/data
uv run python -m calib.run --model glm --src models/src-glm-5.3-flash --data artifacts/glm/data --out artifacts/glm/calib
uv run python -m build.make --model glm --src models/src-glm-5.3-flash --calib artifacts/glm/calib \
  --prune 153 --score reap --budget-gb 76 --drop-vision --rounding lsq --out models/glm-5.3-flash-sauhard
uv run python -m evals.kl --model models/glm-5.3-flash-sauhard
```

`--drop-vision` keeps the build text-only. The published weights add the original vision tower back, unquantized,
as a new directory that symlinks the language weights:

```sh
uv run python -m build.add_vision --src models/src-glm-5.3-flash --base models/glm-5.3-flash-sauhard \
  --out models/glm-5.3-flash-sauhard-vl
```

## License

MIT. Built from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
