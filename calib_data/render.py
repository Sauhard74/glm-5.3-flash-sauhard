"""Chat-template rendering with the target model's own tokenizer + template."""

import json
from pathlib import Path

REPOS = {"glm": "zai-org/GLM-5.3-Flash", "qwen": "Qwen/Qwen3.8-Flash-Next"}
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
                   "generation_config.json", "vocab.json", "merges.txt"]


def ensure_tokenizer(model: str, path: Path | None, offline=False) -> Path:
    path = Path(path) if path else Path("artifacts/tokenizers") / model
    if (path / "tokenizer.json").exists():
        return path
    if offline:
        raise FileNotFoundError(f"no tokenizer at {path}")
    from huggingface_hub import snapshot_download

    snapshot_download(REPOS[model], allow_patterns=TOKENIZER_FILES, local_dir=path)
    return path


class Renderer:
    def __init__(self, model: str, tokenizer_dir: Path):
        from transformers import AutoTokenizer

        self.model = model
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.tok.model_max_length = 1 << 30
        cfg = Path(tokenizer_dir) / "config.json"
        self.vocab_size = len(self.tok)
        if cfg.exists():
            c = json.loads(cfg.read_text())
            self.vocab_size = max(self.vocab_size, c.get("text_config", c).get("vocab_size") or 0)

    def text(self, messages, tools=None, gen=False) -> str:
        return self.tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                            add_generation_prompt=gen)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def render(self, messages, tools=None, gen=False) -> list[int]:
        return self.encode(self.text(messages, tools, gen))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def clip(self, text: str, n: int) -> str:
        ids = self.encode(text)
        return text if len(ids) <= n else self.tok.decode(ids[:n])

    def fit(self, messages, tools=None, max_len=8192, min_msgs=2):
        """Longest message prefix (ending on an assistant/tool turn) that renders within max_len.
        A trailing user turn gets a generation prompt. Falls back to hard token truncation."""
        n = len(messages)
        ends = [k for k in range(min_msgs, n + 1) if messages[k - 1]["role"] in ("assistant", "tool")]
        if not ends or ends[-1] != n:
            ends.append(n)

        def go(k):
            return self.render(messages[:k], tools, gen=messages[k - 1]["role"] != "assistant")

        full = go(n)
        if len(full) <= max_len:
            return full
        lo, hi, best = 0, len(ends) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            ids = go(ends[mid])
            if len(ids) <= max_len:
                best, lo = ids, mid + 1
            else:
                hi = mid - 1
        return best if best is not None else full[:max_len]
