import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import resolve_model_path
from .parsing import THINK_CLOSE, prompt_opens_thinking, split_reasoning


def add_generation_args(
    p: argparse.ArgumentParser,
    *,
    thinking_budget: int,
    answer_tokens: int,
    thinking: bool = True,
    mtp: str = "auto",
) -> None:
    g = p.add_argument_group("generation")
    g.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=thinking)
    g.add_argument("--thinking-budget", type=int, default=thinking_budget)
    g.add_argument(
        "--answer-tokens",
        type=int,
        default=answer_tokens,
        help="tokens allowed after reasoning; max_tokens = budget + answer unless --max-tokens",
    )
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--temperature", type=float, help="default: model generation_config.json")
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--min-p", type=float, default=0.0)
    g.add_argument("--greedy", action="store_true", help="temperature 0")
    g.add_argument("--reasoning-effort", help="passed to the chat template if set")
    g.add_argument("--mtp", choices=["auto", "on", "off"], default=mtp)
    g.add_argument("--draft-model", help="MTP drafter path (default: --model, native mtp weights)")
    g.add_argument("--prefill-step-size", type=int, default=2048)


# Early-exit text from the Qwen3 report's thinking-budget recipe.
BUDGET_CLOSE = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n" + THINK_CLOSE + "\n\n"
)


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0


# Upstream generation_config.json values, used when a built model dir lacks one.
FAMILY_SAMPLING = {
    "qwen": Sampling(temperature=1.0, top_p=0.95, top_k=20),
    "glm": Sampling(temperature=1.0, top_p=0.95, top_k=0),
}
GENERIC_SAMPLING = Sampling(temperature=0.6, top_p=0.95, top_k=20)


def model_sampling(model_path: Path) -> Sampling:
    """Recommended sampling: the checkpoint's generation_config.json, else the
    upstream defaults for the model family."""
    from .common import model_family, read_config

    path = Path(model_path) / "generation_config.json"
    try:
        fallback = FAMILY_SAMPLING.get(model_family(read_config(Path(model_path))), GENERIC_SAMPLING)
    except OSError:
        fallback = GENERIC_SAMPLING
    if not path.exists():
        return Sampling(**asdict(fallback))
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("do_sample") is False and "temperature" not in cfg:
        return Sampling()
    return Sampling(
        temperature=float(cfg.get("temperature", fallback.temperature)),
        top_p=float(cfg.get("top_p", fallback.top_p)),
        top_k=int(cfg.get("top_k", fallback.top_k) or 0),
    )


def resolve_sampling(args, model_path: Path) -> Sampling:
    s = model_sampling(model_path)
    if args.temperature is not None:
        s.temperature = args.temperature
    if args.top_p is not None:
        s.top_p = args.top_p
    if args.top_k is not None:
        s.top_k = args.top_k
    s.min_p = args.min_p
    if args.greedy or getattr(args, "smoke", False):
        s = Sampling(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0)
    return s


def load_mtp(model, model_path: Path, draft_path: Optional[str] = None) -> Tuple[Any, str]:
    """Return (drafter, kind) or (None, reason)."""
    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

    try:
        drafter, kind = load_drafter(str(draft_path or model_path))
        if kind != "mtp":
            return None, f"drafter kind {kind!r} is not mtp"
        validate_drafter_compatibility(model, drafter, kind)
        return drafter, kind
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@dataclass
class GenOutput:
    text: str
    reasoning: str
    answer: str
    thinking_closed: bool
    prompt_tokens: int
    generation_tokens: int
    finish_reason: Optional[str]
    ttft_s: float
    total_s: float
    generation_tps: float
    prompt_tps: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def record(self, keep_text: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not keep_text:
            d.pop("text")
            d.pop("reasoning")
        return d


class Generator:
    def __init__(
        self,
        model,
        processor,
        model_path: Path,
        *,
        thinking: bool = True,
        thinking_budget: Optional[int] = None,
        answer_tokens: int = 1024,
        max_tokens: Optional[int] = None,
        sampling: Optional[Sampling] = None,
        reasoning_effort: Optional[str] = None,
        draft_model=None,
        draft_kind: Optional[str] = None,
        prefill_step_size: int = 2048,
        mtp_status: str = "off",
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.model_path = Path(model_path)
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.answer_tokens = answer_tokens
        self.max_tokens = max_tokens or (
            (thinking_budget or 0) + answer_tokens if thinking else answer_tokens
        )
        self.sampling = sampling or Sampling()
        self.reasoning_effort = reasoning_effort
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.prefill_step_size = prefill_step_size
        self.mtp_status = mtp_status

    @classmethod
    def from_args(cls, args, model, processor) -> "Generator":
        model_path = resolve_model_path(args.model)
        budget, answer = args.thinking_budget, args.answer_tokens
        if args.smoke:
            budget, answer = min(budget, 16), min(answer, 24)
        draft, kind, status = None, None, "off"
        if args.mtp != "off":
            draft, info = load_mtp(model, model_path, args.draft_model)
            if draft is None:
                if args.mtp == "on":
                    raise SystemExit(f"--mtp on but no MTP drafter: {info}")
                status = f"unavailable ({info})"
            else:
                kind, status = info, "on"
        max_tokens = args.max_tokens
        if args.smoke and max_tokens:
            max_tokens = min(max_tokens, budget + answer)
        return cls(
            model,
            processor,
            model_path,
            thinking=args.thinking,
            thinking_budget=budget,
            answer_tokens=answer,
            max_tokens=max_tokens,
            sampling=resolve_sampling(args, model_path),
            reasoning_effort=args.reasoning_effort,
            draft_model=draft,
            draft_kind=kind,
            prefill_step_size=args.prefill_step_size,
            mtp_status=status,
        )

    def settings(self) -> Dict[str, Any]:
        return {
            "thinking": self.thinking,
            "thinking_budget": self.thinking_budget if self.thinking else None,
            "max_tokens": self.max_tokens,
            "sampling": asdict(self.sampling),
            "reasoning_effort": self.reasoning_effort,
            "mtp": self.mtp_status,
            "prefill_step_size": self.prefill_step_size,
        }

    def render(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[int]]:
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self.thinking,
        }
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        text = self.tokenizer.apply_chat_template(messages, **kwargs)
        if not self.thinking and prompt_opens_thinking(text):
            text += THINK_CLOSE
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text, ids

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> GenOutput:
        prompt, ids = self.render(messages, tools)
        return self.generate_ids(prompt, ids, max_tokens=max_tokens)

    def _stream(self, prompt: str, ids: List[int], max_tokens: int) -> Dict[str, Any]:
        import mlx.core as mx
        from mlx_vlm import stream_generate

        kwargs: Dict[str, Any] = dict(
            input_ids=mx.array([ids]),
            max_tokens=max_tokens,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            top_k=self.sampling.top_k,
            min_p=self.sampling.min_p,
            prefill_step_size=self.prefill_step_size,
        )
        if self.draft_model is not None:
            kwargs.update(draft_model=self.draft_model, draft_kind=self.draft_kind)
        self.tokenizer.stopping_criteria.reset(self.model.config.eos_token_id)
        text, last, ttft = "", None, None
        start = time.perf_counter()
        for response in stream_generate(self.model, self.processor, prompt, **kwargs):
            if ttft is None:
                ttft = time.perf_counter() - start
            text += response.text
            last = response
        total = time.perf_counter() - start
        token_ids = list(last.token_ids or []) if last else []
        if last is not None and last.finish_reason == "stop" and token_ids:
            token_ids = token_ids[:-1]
        return {
            "text": text,
            "token_ids": token_ids,
            "n": last.generation_tokens if last else 0,
            "finish_reason": last.finish_reason if last else None,
            "ttft": ttft if ttft is not None else total,
            "total": total,
            "prompt_tps": last.prompt_tps if last else 0.0,
        }

    def generate_ids(self, prompt: str, ids: List[int], max_tokens: Optional[int] = None) -> GenOutput:
        """Generate with a hard thinking budget: if the first ``thinking_budget``
        tokens do not close the <think> block, append BUDGET_CLOSE and continue
        for ``answer_tokens`` (re-prefilling prompt + reasoning). Done outside
        mlx-vlm so it also holds on the speculative (MTP) path."""
        opened = prompt_opens_thinking(prompt)
        max_tokens = max_tokens or self.max_tokens
        budgeted = self.thinking and opened and self.thinking_budget is not None
        first = self._stream(prompt, ids, min(max_tokens, self.thinking_budget) if budgeted else max_tokens)
        parts, n, total, forced = [first], first["n"], first["total"], False
        if budgeted and first["finish_reason"] == "length" and n < max_tokens:
            if THINK_CLOSE in first["text"]:
                cont, remaining = "", max_tokens - n
            else:
                cont, remaining, forced = BUDGET_CLOSE, self.answer_tokens, True
            prefix = prompt + first["text"] + cont
            cont_ids = self.tokenizer.encode(cont, add_special_tokens=False) if cont else []
            second = self._stream(prefix, ids + first["token_ids"] + cont_ids, remaining)
            parts.append(dict(second, text=cont + second["text"]))
            n += second["n"]
            total += second["total"]
        text = "".join(p["text"] for p in parts)
        reasoning, answer, closed = split_reasoning(text, opened)
        gen_time = sum(p["total"] - p["ttft"] for p in parts)
        return GenOutput(
            text=text,
            reasoning=reasoning,
            answer=answer,
            thinking_closed=closed,
            prompt_tokens=len(ids),
            generation_tokens=n,
            finish_reason=parts[-1]["finish_reason"],
            ttft_s=round(first["ttft"], 4),
            total_s=round(total, 4),
            generation_tps=round(n / gen_time, 2) if gen_time > 0 else 0.0,
            prompt_tps=round(first["prompt_tps"], 2),
            extra={"budget_forced": forced, "passes": len(parts)},
        )
