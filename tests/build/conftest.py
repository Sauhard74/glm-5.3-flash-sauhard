import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tests" / "fixtures"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_tiny  # noqa: E402


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    cache = {}

    def get(family):
        if family not in cache:
            d = tmp_path_factory.mktemp(f"tiny-{family}")
            expected, expected_mtp = build_tiny.make_checkpoint(family, d / "src")
            calib = build_tiny.make_calib(family, d / "calib", style="json")
            calib_npz = build_tiny.make_calib(family, d / "calib_npz", style="npz", seed=1)
            cache[family] = {
                "src": d / "src",
                "calib": calib,
                "calib_npz": calib_npz,
                "expected": expected,
                "expected_mtp": expected_mtp,
                "root": d,
            }
        return cache[family]

    return get
