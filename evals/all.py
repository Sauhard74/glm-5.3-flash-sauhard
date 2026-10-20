"""Run several suites one after another (one subprocess each, so memory is
returned between suites), then print the comparison table.

uv run python -m evals.all --model models/<name> [--suites kl,speed,code,math,longctx,tools] [--smoke] [-- extra args]
"""

import argparse
import subprocess
import sys

from .compare import SUITES


def main(argv=None):
    p = argparse.ArgumentParser(prog="evals.all", description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--results-dir", default="results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="continue after a failing suite")
    args, extra = p.parse_known_args(argv)
    extra = [a for a in extra if a != "--"]
    failed = []
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        cmd = [sys.executable, "-m", f"evals.{suite}", "--model", args.model, "--results-dir", args.results_dir]
        cmd += (["--smoke"] if args.smoke else []) + extra
        print(f"[all] {' '.join(cmd)}", flush=True)
        if subprocess.call(cmd) != 0:
            failed.append(suite)
            if not args.keep_going:
                break
    compare_cmd = [sys.executable, "-m", "evals.compare", "--results-dir", args.results_dir]
    subprocess.call(compare_cmd + (["--include-smoke"] if args.smoke else []))
    if failed:
        raise SystemExit(f"failed suites: {', '.join(failed)}")


if __name__ == "__main__":
    main()
