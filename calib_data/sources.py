"""Pinned HF dataset slices, fetched by byte range (one parquet row group / one jsonl prefix) and cached."""

import gzip
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Parquet:
    repo: str
    rev: str
    path: str
    row_group: int
    columns: tuple

    def fetch(self):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(f"datasets/{self.repo}@{self.rev}/{self.path}", "rb", block_size=1 << 20) as f:
            table = pq.ParquetFile(f).read_row_group(self.row_group, columns=list(self.columns))
        return table.to_pylist()


@dataclass(frozen=True)
class JsonlPrefix:
    repo: str
    rev: str
    path: str
    nbytes: int

    def fetch(self):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(f"datasets/{self.repo}@{self.rev}/{self.path}", "rb", block_size=0) as f:
            buf = f.read(self.nbytes)
        lines = buf.split(b"\n")
        if len(buf) == self.nbytes:
            lines = lines[:-1]
        return [json.loads(line) for line in lines if line.strip()]


REV = {
    "codeparrot/github-code-clean": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
    "bigcode/commitpackft": "fc56fe33c030c6daa414c2b112c932b8eed085e6",
    "SWE-Gym/OpenHands-SFT-Trajectories": "4aaa5a4a4b5861f4799d2336908760c190ac3b17",
    "Agent-Ark/Toucan-1.5M": "0df3cf37f2abefb380370cfb02eabea2a35ae782",
    "open-r1/OpenR1-Math-220k": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
    "open-thoughts/OpenThoughts-114k": "bd093c3994fd54d2390985b66988ddf282a55eb6",
    "deepmind/narrativeqa": "2e643e7363944af1c33a652d1c87320d0871c4e4",
    "ccdv/arxiv-summarization": "240aaf1a969b3f8cd0ade6986bfad0cd730ee288",
}


def _pq(repo, path, row_groups, columns):
    return [Parquet(repo, REV[repo], path, g, tuple(columns)) for g in row_groups]


COMMIT_LANGS = ("python", "javascript", "typescript", "go", "rust", "java", "c++", "c", "c#",
                "ruby", "php", "shell", "kotlin", "swift", "scala")

SOURCES = {
    "github_code": _pq("codeparrot/github-code-clean", "data/train-00000-of-00880.parquet", [0],
                       ["code", "repo_name", "path", "language"]),
    "commitpack": [JsonlPrefix("bigcode/commitpackft", REV["bigcode/commitpackft"], f"data/{lang}/data.jsonl", 1 << 20)
                   for lang in COMMIT_LANGS],
    "openhands": _pq("SWE-Gym/OpenHands-SFT-Trajectories", "data/train.success.oss-00000-of-00001.parquet", [0],
                     ["messages"]),
    "toucan_multi": _pq("Agent-Ark/Toucan-1.5M", "Kimi-K2/train-00030-of-00040.parquet", [0],
                        ["uuid", "subset_name", "messages", "available_tools"]),
    "toucan_single": _pq("Agent-Ark/Toucan-1.5M", "Kimi-K2/train-00003-of-00040.parquet", [0],
                         ["uuid", "subset_name", "messages", "available_tools"]),
    "openr1_math": _pq("open-r1/OpenR1-Math-220k", "all/default-00000-of-00010.parquet", [0],
                       ["uuid", "problem", "messages", "correctness_math_verify"]),
    "openthoughts_code": _pq("open-thoughts/OpenThoughts-114k", "metadata/train-00011-of-00012.parquet", [3],
                             ["problem", "deepseek_reasoning", "deepseek_solution", "domain"]),
    "openthoughts_science": _pq("open-thoughts/OpenThoughts-114k", "metadata/train-00011-of-00012.parquet",
                                [5, 6, 7, 8], ["problem", "deepseek_reasoning", "deepseek_solution", "domain"]),
    "narrativeqa": _pq("deepmind/narrativeqa", "data/test-00000-of-00008.parquet", [0, 1],
                       ["document.id", "document.kind", "document.text", "question.text", "answers"]),
    "arxiv": _pq("ccdv/arxiv-summarization", "document/test-00000-of-00001.parquet", [0], ["article", "abstract"]),
}


def log(msg):
    print(f"[calib_data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cache_path(cache_dir: Path, part) -> Path:
    key = hashlib.sha1(json.dumps(asdict(part), sort_keys=True).encode()).hexdigest()[:12]
    slug = part.repo.split("/")[-1] + "-" + Path(part.path).stem.replace("/", "_")
    return cache_dir / f"{slug}-{key}.jsonl.gz"


def load_part(part, cache_dir: Path, offline=False, retries=3):
    path = _cache_path(cache_dir, part)
    if path.exists():
        with gzip.open(path, "rt") as f:
            return [json.loads(line) for line in f]
    if offline:
        raise FileNotFoundError(f"{path} not cached and --offline set ({part})")
    for attempt in range(retries):
        try:
            t = time.time()
            rows = part.fetch()
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            log(f"fetch failed ({e!r}); retrying")
            time.sleep(5 * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    log(f"fetched {part.repo}/{part.path}"
        f"{f' rg{part.row_group}' if isinstance(part, Parquet) else ''}: {len(rows)} rows in {time.time() - t:.0f}s")
    return rows


def load(name, cache_dir: Path, offline=False):
    rows = []
    for part in SOURCES[name]:
        for i, r in enumerate(load_part(part, cache_dir, offline)):
            r["_src"] = f"{part.repo}:{part.path}:{getattr(part, 'row_group', 0)}:{i}"
            rows.append(r)
    return rows
