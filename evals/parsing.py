import re
from fractions import Fraction
from typing import List, Optional, Tuple

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
END_MARKERS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|user|>",
    "<|observation|>",
    "<|assistant|>",
    "<|system|>",
)


def strip_end_markers(text: str) -> str:
    for marker in END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text


def prompt_opens_thinking(prompt: str) -> bool:
    """True when the rendered prompt leaves a <think> block open for the model."""
    last_open = prompt.rfind(THINK_OPEN)
    return last_open != -1 and prompt.find(THINK_CLOSE, last_open) == -1


def split_reasoning(text: str, opened: bool) -> Tuple[str, str, bool]:
    """Split generated text into (reasoning, answer, thinking_closed).

    ``opened`` says whether the prompt already opened a <think> block. If the
    block never closes (budget exhausted / max tokens) the answer is empty.
    """
    text = strip_end_markers(text)
    stripped = text.lstrip()
    if stripped.startswith(THINK_OPEN):
        text = stripped[len(THINK_OPEN) :]
        opened = True
    if opened:
        if THINK_CLOSE not in text:
            return text.strip(), "", False
        reasoning, answer = text.split(THINK_CLOSE, 1)
        return reasoning.strip(), answer.strip(), True
    if THINK_CLOSE in text:
        reasoning, answer = text.rsplit(THINK_CLOSE, 1)
        return reasoning.replace(THINK_OPEN, "").strip(), answer.strip(), True
    return "", text.strip(), True


_CODE_BLOCK = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the last python fenced block (any fenced block as fallback)."""
    blocks = [(lang.lower(), body) for lang, body in _CODE_BLOCK.findall(text)]
    if not blocks:
        return text.strip()
    py = [body for lang, body in blocks if lang in ("python", "py", "python3")]
    chosen = py or [body for lang, body in blocks if not lang] or [b for _, b in blocks]
    best = [b for b in chosen if "def " in b or "import " in b]
    return (best or chosen)[-1].strip("\n")


def last_boxed(text: str) -> Optional[str]:
    """Contents of the last \\boxed{...} / \\fbox{...} with balanced braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx == -1:
        return None
    i = idx
    while i < len(text) and text[i] != "{":
        if text[i] == " " and text[idx:i].strip() == "\\boxed":
            rest = text[i + 1 :].split("$")[0].split()
            return rest[0] if rest else None
        i += 1
    if i >= len(text):
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


_FINAL_PATTERNS = [
    re.compile(r"final answer\s*(?:is|:)\s*\$?(.+?)\$?(?:\.\s|\.$|\n|$)", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:)\s*\$?(.+?)\$?(?:\.\s|\.$|\n|$)", re.IGNORECASE),
]


def extract_math_answer(text: str) -> Optional[str]:
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    return None


def extract_integer(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    cleaned = strip_math(text)
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    nums = re.findall(r"-?\d+", cleaned.replace(",", ""))
    return int(nums[-1]) if nums else None


def _fix_fracs(s: str) -> str:
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        elif len(sub) >= 2:
            a, b = sub[0], sub[1]
            if b != "{":
                out += "{" + a + "}{" + b + "}" + sub[2:]
            else:
                out += "{" + a + "}" + b + sub[2:]
        else:
            out += sub
    return out


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def _fix_a_slash_b(s: str) -> str:
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" if m else s


def strip_math(s: str) -> str:
    """Normalize a LaTeX answer (Hendrycks MATH style, slightly extended)."""
    s = str(s).strip()
    s = s.replace("\n", "").replace("\\!", "")
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\[,;: ]", "", s)
    s = s.replace("\\displaystyle", "")
    for unit in ("degrees", "degree", "units", "unit", "cm", "inches", "inch", "meters", "dollars"):
        s = re.sub(rf"\s*(?:\\mbox{{\s*)?{unit}\}}?\s*$", "", s)
    s = s.strip().rstrip(".")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    s = _fix_a_slash_b(s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _to_number(s: str) -> Optional[Fraction]:
    s = s.strip()
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        sign = -1 if m.group(1) else 1
        den = int(m.group(3))
        return sign * Fraction(int(m.group(2)), den) if den else None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _split_top_level(s: str) -> List[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def math_equal(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    a, b = strip_math(pred), strip_math(gt)
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    if na is not None and nb is not None:
        return abs(float(na) - float(nb)) <= 1e-6 * max(1.0, abs(float(nb)))
    for wrap in ("()", "[]", "{}"):
        if a[:1] == b[:1] == wrap[0] and a[-1:] == b[-1:] == wrap[1]:
            pa, pb = _split_top_level(a[1:-1]), _split_top_level(b[1:-1])
            if len(pa) == len(pb) and len(pa) > 1:
                return all(math_equal(x, y) for x, y in zip(pa, pb))
    pa, pb = _split_top_level(a), _split_top_level(b)
    if len(pa) == len(pb) > 1:
        return sorted(pa) == sorted(pb)
    return False
