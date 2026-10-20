"""RULER-style synthetic long-context suite: single needle, multi-key needle with
distractors, variable tracking, and QA over many similar synthetic documents.

uv run python -m evals.longctx --model models/<name> [--lengths 32768,65536,131072] [--limit 8]
"""

import random
import time
from typing import Any, Callable, Dict, List, Tuple

from .common import Checkpoint, base_parser, load_model, mean, read_config, resolve_model_path, write_result

TASKS = ("niah_single", "niah_multikey", "variable_tracking", "qa")
DEFAULT_LENGTHS = "32768,65536,131072"
SMOKE_LENGTHS = "1024,2048"

_ADJ = "quiet amber hollow brisk crimson distant gentle humble ivory jagged keen lucid mellow narrow olive placid rustic silent tidal vivid woven".split()
_NOUN = "harbor lantern meadow orchard quarry ridge summit thicket valley willow beacon canyon delta estuary fjord glacier island jetty kiln lagoon".split()
_SUBJ = "The committee,A traveling merchant,The old librarian,Every apprentice,The river guide,A young cartographer,The village council,Most farmers,The night watchman,A visiting scholar,The ferry captain,Several weavers".split(",")
_VERB = "recorded,questioned,repaired,described,ignored,celebrated,measured,catalogued,debated,painted,traded,inspected".split(",")
_OBJ = "the northern bridge,a ledger of harvests,the broken sundial,an unusual storm,the price of salt,the abandoned mill,a map of the coast,the spring festival,the lighthouse lamp,a dispute over water rights,the new granary,the winter road".split(",")
_TAIL = "before the first frost,after a long silence,with great patience,despite the rain,for reasons nobody remembered,during the market week,without telling anyone,as the tide came in,late into the evening,in the presence of witnesses,while the bells rang,at the edge of town".split(",")
_SYL = "ka lo mi ren tor vas pel dun shi gor bel nya quo zen ari mor tes lin fa ul".split()
_FIRST = "Mira Tomas Ilse Joran Petra Anselm Kaito Lena Oskar Rhea Dorian Saskia Viggo Nadia Emre Freya".split()
_LAST = "Tollensen Varga Okafor Lindqvist Moreau Halvorsen Castell Brandt Ivers Nakamura Quill Seidel Abernathy Marsh Petrov Duval".split()
_KIND = "bridge observatory library monastery shipyard theater academy museum lighthouse brewery foundry archive".split()
_FEATURE = "a spiral staircase,a copper dome,a clock with no hands,a hidden courtyard,stained glass windows,a garden of black roses,an underground canal,a bell cast from cannons,a mosaic floor,a tower of white stone".split(",")


def filler_sentences(rng: random.Random, n: int) -> List[str]:
    out = []
    for _ in range(n):
        s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} near the {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_TAIL)}."
        out.append(s)
    return out


def _key(rng: random.Random) -> str:
    return f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}-{rng.randint(10, 99)}"


def _number(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def _var(rng: random.Random) -> str:
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))


def _entity(rng: random.Random) -> str:
    name = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))).capitalize()
    return f"the {name} {rng.choice(_KIND)}"


class ContextBuilder:
    """Fills a prompt to an exact-ish token length around inserted lines."""

    def __init__(self, tokenizer, render: Callable[[str], Tuple[str, List[int]]]):
        self.tokenizer = tokenizer
        self.render = render

    def ntok(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def build(
        self,
        target: int,
        make_prompt: Callable[[str], str],
        inserts: List[Tuple[float, str]],
        filler: List[str],
    ) -> Tuple[str, List[int]]:
        """``inserts`` are (depth in [0,1], text); ``filler`` units are joined by spaces.
        Whole units get within one unit of ``target``; a partial unit tops it up."""
        lens = [self.ntok(" " + s) for s in filler[:256]]
        per_unit = max(1.0, sum(lens) / len(lens))
        overhead = len(self.render(make_prompt(" ".join(t for _, t in inserts)))[1])
        tol = max(16, int(0.01 * target))

        def attempt(n_units: int, extra_words: int = 0):
            units = [filler[i % len(filler)] for i in range(n_units)]
            if extra_words:
                units.append(" ".join(filler[n_units % len(filler)].split()[:extra_words]))
            return self.render(make_prompt(self._assemble(units, inserts)))

        n_units = max(1, int((target - overhead) / per_unit))
        prompt, ids = attempt(n_units)
        for _ in range(6):
            err = target - len(ids)
            if 0 <= err <= tol:
                return prompt, ids
            step = int(err / per_unit) or (1 if err > 0 else -1)
            n_units = max(1, n_units + step)
            prompt, ids = attempt(n_units)
            if 0 <= target - len(ids) < per_unit:
                break
        gap = target - len(ids)
        if gap > tol:
            words = filler[n_units % len(filler)].split()
            k = min(len(words), max(1, int(gap / per_unit * len(words))))
            for _ in range(4):
                cand = attempt(n_units, k)
                if len(cand[1]) <= target:
                    prompt, ids = cand
                    break
                k -= 1
        return prompt, ids

    @staticmethod
    def _assemble(units: List[str], inserts: List[Tuple[float, str]]) -> str:
        placed = sorted(inserts, key=lambda x: x[0])
        out, j = [], 0
        for i, u in enumerate(units):
            while j < len(placed) and placed[j][0] * len(units) <= i:
                out.append(placed[j][1])
                j += 1
            out.append(u)
        out.extend(t for _, t in placed[j:])
        return " ".join(out)


NIAH_PREFIX = (
    "Some special magic numbers are hidden within the following text. Make sure to memorize them. "
    "I will quiz you about the numbers afterwards.\n\n"
)


def make_task(task: str, rng: random.Random, n_filler: int = 4000) -> Dict[str, Any]:
    """Return {inserts, make_prompt, answers, filler}."""
    filler = filler_sentences(rng, n_filler)
    if task == "niah_single":
        key, val = _key(rng), _number(rng)
        inserts = [(rng.random(), f"The special magic number for {key} is: {val}.")]
        q = f"What is the special magic number for {key} mentioned in the provided text? Answer with just the number."
        return dict(inserts=inserts, answers=[val], filler=filler, make_prompt=lambda c: f"{NIAH_PREFIX}{c}\n\n{q}")
    if task == "niah_multikey":
        keys = []
        while len(keys) < 5:
            k = _key(rng)
            if k not in keys:
                keys.append(k)
        vals = [_number(rng) for _ in keys]
        inserts = [(rng.random(), f"The special magic number for {k} is: {v}.") for k, v in zip(keys, vals)]
        q = f"What is the special magic number for {keys[0]} mentioned in the provided text? Answer with just the number."
        return dict(inserts=inserts, answers=[vals[0]], filler=filler, make_prompt=lambda c: f"{NIAH_PREFIX}{c}\n\n{q}")
    if task == "variable_tracking":
        val, other = str(rng.randint(10000, 99999)), str(rng.randint(10000, 99999))
        names = []
        while len(names) < 10:
            v = _var(rng)
            if v not in names:
                names.append(v)
        chain, distract = names[:5], names[5:]
        depths = sorted(rng.random() for _ in range(10))
        lines = [f"VAR {chain[0]} = {val}."] + [f"VAR {chain[i]} = VAR {chain[i - 1]}." for i in range(1, 5)]
        lines += [f"VAR {distract[0]} = {other}."] + [f"VAR {distract[i]} = VAR {distract[i - 1]}." for i in range(1, 5)]
        order = list(range(10))
        rng.shuffle(order)
        chain_depths = sorted(depths[i] for i in order[:5])
        distract_depths = sorted(depths[i] for i in order[5:])
        inserts = list(zip(chain_depths, lines[:5])) + list(zip(distract_depths, lines[5:]))
        q = (
            f"Memorize and track the chain(s) of variable assignment hidden in the text above. "
            f"Find all variables that are assigned the value {val}. List all of their names."
        )
        prefix = "Below is a long text containing variable assignments (lines starting with VAR).\n\n"
        return dict(inserts=inserts, answers=chain, filler=filler, make_prompt=lambda c: f"{prefix}{c}\n\n{q}")
    if task == "qa":
        docs, seen = [], set()
        while len(docs) < max(64, n_filler // 2):
            ent = _entity(rng)
            if ent in seen:
                continue
            seen.add(ent)
            founder = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            year = rng.randint(1500, 1950)
            docs.append((ent, founder, year, rng.choice(_FEATURE)))
        target = docs[0]
        text = [
            f"{e[0].upper() + e[1:]} was founded in {y} by {f}. Its best-known feature is {feat}."
            for e, f, y, feat in docs[1:]
        ]
        ent, founder, year, feat = target
        needle = f"{ent[0].upper() + ent[1:]} was founded in {year} by {founder}. Its best-known feature is {feat}."
        ask_year = rng.random() < 0.5
        q = (
            f"According to the documents, in which year was {ent} founded? Answer with just the year."
            if ask_year
            else f"According to the documents, who founded {ent}? Answer with just the name."
        )
        prefix = "Answer the question based on the following collection of short documents.\n\n"
        return dict(
            inserts=[(rng.random(), needle)],
            answers=[str(year)] if ask_year else [founder],
            filler=text,
            make_prompt=lambda c: f"{prefix}{c}\n\nQuestion: {q}",
        )
    raise ValueError(task)


def score(answers: List[str], text: str) -> float:
    low = text.lower()
    return sum(a.lower() in low for a in answers) / len(answers)


def main(argv=None):
    from .generation import Generator, add_generation_args

    p = base_parser("longctx", __doc__)
    p.add_argument("--lengths", help=f"prompt lengths in tokens (default {DEFAULT_LENGTHS})")
    p.add_argument("--tasks", default=",".join(TASKS))
    add_generation_args(p, thinking_budget=1024, answer_tokens=256)
    args = p.parse_args(argv)
    started = time.time()

    lengths = [int(x) for x in (args.lengths or (SMOKE_LENGTHS if args.smoke else DEFAULT_LENGTHS)).split(",")]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    n = args.limit if args.limit is not None else (1 if args.smoke else 8)

    cfg = read_config(resolve_model_path(args.model))
    max_pos = (cfg.get("text_config") or cfg).get("max_position_embeddings")
    model, processor = load_model(args)
    gen = Generator.from_args(args, model, processor)
    builder = ContextBuilder(gen.tokenizer, lambda content: gen.render([{"role": "user", "content": content}]))

    settings = {"lengths": lengths, "tasks": tasks, "per_cell": n, **gen.settings()}
    ck = Checkpoint(args, "longctx", settings)
    samples: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for length in lengths:
        if max_pos and length > max_pos:
            skipped.append({"length": length, "reason": f"> max_position_embeddings {max_pos}"})
            continue
        for task in tasks:
            for i in range(n):
                key = f"{task}/{length}/{i}"
                if ck.get(key) is not None:
                    samples.append(ck.get(key))
                    continue
                rng = random.Random(f"{args.seed}-{task}-{length}-{i}")
                t = make_task(task, rng, n_filler=max(400, length // 12))
                prompt, ids = builder.build(length, t["make_prompt"], t["inserts"], t["filler"])
                out = gen.generate_ids(prompt, ids)
                scored_text = out.answer if out.thinking_closed else out.text
                s = score(t["answers"], scored_text)
                rec = {"task": task, "length": length, "i": i, "score": s, "answers": t["answers"], **out.record()}
                ck.add(key, rec)
                samples.append(rec)
                print(
                    f"[longctx] {task} len={length} ({out.prompt_tokens} tok) #{i} score={s:.2f} "
                    f"ttft={out.ttft_s:.1f}s",
                    flush=True,
                )

    breakdown: Dict[str, Any] = {}
    for length in lengths:
        rows = [s for s in samples if s["length"] == length]
        if not rows:
            continue
        b = {task: mean(r["score"] for r in rows if r["task"] == task) for task in tasks}
        b["mean"] = mean(r["score"] for r in rows)
        b["mean_prompt_tokens"] = mean(r["prompt_tokens"] for r in rows)
        b["mean_ttft_s"] = mean(r["ttft_s"] for r in rows)
        breakdown[str(length)] = b
    summary: Dict[str, Any] = {f"score@{k}": v["mean"] for k, v in breakdown.items()}
    for task in tasks:
        summary[f"{task}"] = mean(s["score"] for s in samples if s["task"] == task)
    summary["score"] = mean(s["score"] for s in samples)
    summary["n"] = len(samples)
    summary["headline"] = "score"
    write_result(
        args,
        "longctx",
        summary=summary,
        breakdown=breakdown,
        samples=samples,
        settings=settings,
        started=started,
        extra={"skipped": skipped},
    )
    ck.finish()


if __name__ == "__main__":
    main()
