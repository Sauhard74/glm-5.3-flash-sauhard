---
license: mit
base_model: zai-org/GLM-5.3-Flash
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - moe
  - reap
  - mixed-precision
  - glm
language:
  - en
  - zh
---

# GLM 5.3 Flash-sauhard (MLX)

GLM-5.3-Flash (321B mixture-of-experts) rebuilt to run on a **128 GB Apple Silicon Mac** with room left for your
editor, builds and Claude Code. 71.6 GB resident while serving.

- **Pruned:** 153 of 288 routed experts per layer, chosen by REAP on calibration data from code, agent, math and
  long-document work.
- **Quantized:** a bit width per tensor from 3 to 8 bits (3.48 bits per weight on average), allocated by measured
  sensitivity, with LSQ rounding (each scale refit against calibration activations).
- **Text-only:** the vision tower is dropped. GLM's MTP draft head is included (1.6 GB) but speculative decoding has
  not been verified yet.

Write-up: https://glm-flash-sauhard.vercel.app · Benchmarks: https://glm-flash-sauhard.vercel.app/report/ ·
Setup: https://glm-flash-sauhard.vercel.app/run/ · Code: https://github.com/Sauhard74/glm-5.3-flash-sauhard

## Results

Measured on a MacBook Pro M5 Max (128 GB). Thinking on, temperature 1.0, top-p 0.95, one sample per problem.
These are subsets, so ranges are wide.

| Benchmark | Problems | Score |
|---|---|---|
| HumanEval+ | 40 | 95.0% |
| MBPP+ | 40 | 92.5% |
| MATH-500 | 40 | 97.5% |
| BFCL function calling | 105 | 84.8% |

Fidelity against full-precision GLM-5.3-Flash on 48 held-out conversations (522K tokens): KL divergence **0.108** and
**92.4%** top-1 agreement on the tokens the model writes.

Speed and memory: see the [benchmark report](https://glm-flash-sauhard.vercel.app/report/#speed).

## Run it

Needs an Apple Silicon Mac with 128 GB of unified memory and the serving scripts from the code repo (they cap memory
and disable a 22 GB long-context prefill buffer that would otherwise push the model past 84 GB).

```sh
git clone https://github.com/Sauhard74/glm-5.3-flash-sauhard.git && cd glm-5.3-flash-sauhard
uv sync
uv run hf download Sauhard74/GLM-5.3-Flash-sauhard-MLX --local-dir models/glm-5.3-flash-sauhard
openssl rand -hex 24 > control/api_key
CAP_GB=84 MTP=0 scripts/quickserve.sh models/glm-5.3-flash-sauhard
```

OpenAI API at `http://127.0.0.1:8080/v1`, Anthropic API at `http://127.0.0.1:8080/v1/messages`. The `model` field
is the path above. Claude Code: `MODEL_ID=models/glm-5.3-flash-sauhard scripts/claude-local.sh`.

Tested with MLX 0.32.4 and mlx-vlm 0.7.9 (`model_type: glm5_next`).

## Limits

- Not run on public leaderboards. The original GLM-5.3-Flash scores 42 on the Artificial Analysis index (medium
  effort); this build is expected to land near 40–41, but that is an estimate.
- Long-context retrieval beyond 64K tokens has not been tested.
- Calibration data is public Hugging Face datasets only; no private data was used.

## License

MIT, as the base model. Built from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
