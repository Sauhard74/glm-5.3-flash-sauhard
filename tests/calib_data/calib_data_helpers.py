from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def tokenizer_dir(model):
    for d in (ROOT / "artifacts" / "tokenizers" / model, ROOT / "tests" / "fixtures" / "tokenizers" / model):
        if (d / "tokenizer.json").exists() and (d / "chat_template.jinja").exists():
            return d
    pytest.skip(f"no local tokenizer for {model} (run calib_data.build once to fetch it)")
