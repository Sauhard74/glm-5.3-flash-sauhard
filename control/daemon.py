"""Control daemon: owns the model server process, persists desired state, serves the dashboard.

Run under launchd (scripts/install-control.sh). Dashboard: http://127.0.0.1:8090
"""

import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "control" / "state.json"
PID_FILE = ROOT / "control" / "server.pid"
SERVER_LOG = ROOT / "logs" / "server.log"
DASHBOARD = ROOT / "control" / "dashboard.html"
GB = 1 << 30

DEFAULTS = {
    "desired": "stopped",
    "model": "models/qwen-flash-next-reap288",
    "mtp": True,
    "draft_block": 2,
    "cap_gb": 80,
    "prefill": 2048,
    "port": 8080,
    "keep_awake": False,
}

lock = threading.RLock()
caffeinate: subprocess.Popen | None = None
started_at: float | None = None


def load_state() -> dict:
    try:
        return {**DEFAULTS, **json.loads(STATE_FILE.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def server_proc() -> psutil.Process | None:
    """The live server process, adopted from the pid file or found by command line."""
    try:
        p = psutil.Process(int(PID_FILE.read_text()))
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            return p
    except (OSError, ValueError, psutil.Error):
        pass
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if "scripts/run_server.py" in cmd and "uv run" not in cmd:
            PID_FILE.write_text(str(p.pid))
            return p
    return None


def server_tree_rss(p: psutil.Process) -> int:
    try:
        procs = [p] + p.children(recursive=True)
        return sum(c.memory_info().rss for c in procs if c.is_running())
    except psutil.Error:
        return 0


def api_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server() -> None:
    global started_at
    with lock:
        if server_proc():
            return
        s = load_state()
        env = {
            **os.environ,
            "MTP": "1" if s["mtp"] else "0",
            "DRAFT_BLOCK": str(s["draft_block"]),
            "CAP_GB": str(s["cap_gb"]),
            "PREFILL": str(s["prefill"]),
            "PORT": str(s["port"]),
        }
        SERVER_LOG.parent.mkdir(exist_ok=True)
        log = open(SERVER_LOG, "ab")
        log.write(f"\n===== control: start {time.ctime()} =====\n".encode())
        proc = subprocess.Popen(
            [str(ROOT / "scripts" / "quickserve.sh"), s["model"]],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        # quickserve.sh execs `uv run`, which spawns the Python server; record the Python pid.
        for _ in range(50):
            time.sleep(0.2)
            kids = [c for c in psutil.Process(proc.pid).children(recursive=True)
                    if "run_server.py" in " ".join(c.cmdline())] if psutil.pid_exists(proc.pid) else []
            if kids:
                PID_FILE.write_text(str(kids[-1].pid))
                break
        else:
            PID_FILE.write_text(str(proc.pid))
        started_at = time.time()


def stop_server() -> None:
    global started_at
    with lock:
        p = server_proc()
        if not p:
            return
        try:
            group = [p] + p.children(recursive=True)
            parents = [q for q in p.parents() if "uv" in q.name() or "quickserve" in " ".join(q.cmdline())]
            for q in group + parents:
                q.send_signal(signal.SIGTERM)
            _, alive = psutil.wait_procs(group, timeout=20)
            for q in alive:
                q.kill()
        except psutil.Error:
            pass
        PID_FILE.unlink(missing_ok=True)
        started_at = None


def apply_keep_awake(on: bool) -> None:
    global caffeinate
    if on and (caffeinate is None or caffeinate.poll() is not None):
        # -i: no idle sleep, -s: no system sleep on AC power. Lid-close on battery still sleeps.
        caffeinate = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-s"])
    elif not on and caffeinate is not None:
        caffeinate.terminate()
        caffeinate = None


def last_stats() -> dict:
    out = {}
    try:
        with open(SERVER_LOG, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 200_000))
            tail = f.read().decode(errors="replace").splitlines()
    except OSError:
        return out
    for line in reversed(tail):
        if "decode_tps" not in out and "Decode completed" in line:
            m = re.search(r"generated_tokens=(\d+).*rate=([\d.]+) tok/s", line)
            if m:
                out["decode_tokens"], out["decode_tps"] = int(m[1]), float(m[2])
                out["last_request"] = line[:19]
        if "prefill_tps" not in out and "Prefill completed" in line:
            m = re.search(r"prompt_tokens=(\d+) cached_tokens=(\d+).*rate=([\d.]+) tok/s", line)
            if m:
                out["prompt_tokens"], out["cached_tokens"], out["prefill_tps"] = int(m[1]), int(m[2]), float(m[3])
        if "error" not in out and ("Traceback" in line or " ERROR " in line):
            out["error"] = line[-300:]
        if len(out) >= 7:
            break
    out["log_tail"] = tail[-12:]
    return out


def memory_info() -> dict:
    vm = psutil.virtual_memory()
    try:
        pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5).stdout
        free_pct = int(re.search(r"free percentage: (\d+)%", pressure)[1])
    except Exception:
        free_pct = None
    return {"total_gb": round(vm.total / GB, 1), "available_gb": round(vm.available / GB, 1),
            "used_gb": round((vm.total - vm.available) / GB, 1), "free_pct": free_pct,
            "swap_gb": round(psutil.swap_memory().used / GB, 1)}


def status() -> dict:
    s = load_state()
    p = server_proc()
    if p is None:
        phase = "stopped" if s["desired"] == "stopped" else "starting"
    else:
        phase = "running" if api_ready(s["port"]) else "loading"
    return {
        "phase": phase, "desired": s["desired"], "settings": s,
        "pid": p.pid if p else None,
        "server_rss_gb": round(server_tree_rss(p) / GB, 1) if p else 0,
        "uptime_s": int(time.time() - p.create_time()) if p else 0,
        "keep_awake_active": caffeinate is not None and caffeinate.poll() is None,
        "memory": memory_info(), "stats": last_stats(),
        "endpoint": f"http://127.0.0.1:{s['port']}",
    }


def supervisor() -> None:
    """Reconcile actual vs desired state; restart the server if it crashed while desired=running."""
    crashes: list[float] = []
    while True:
        s = load_state()
        apply_keep_awake(s["keep_awake"] and s["desired"] == "running")
        if s["desired"] == "running" and server_proc() is None:
            crashes = [t for t in crashes if time.time() - t < 600] + [time.time()]
            if len(crashes) <= 3:
                start_server()
            else:
                s["desired"] = "stopped"
                save_state(s)
        time.sleep(5)


app = FastAPI()


class Settings(BaseModel):
    model: str | None = None
    mtp: bool | None = None
    draft_block: int | None = None
    cap_gb: int | None = None
    prefill: int | None = None
    keep_awake: bool | None = None


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD.read_text()


@app.get("/api/status")
def api_status():
    return status()


@app.post("/api/start")
def api_start():
    s = load_state(); s["desired"] = "running"; save_state(s)
    threading.Thread(target=start_server, daemon=True).start()
    return {"ok": True}


@app.post("/api/stop")
def api_stop():
    s = load_state(); s["desired"] = "stopped"; save_state(s)
    stop_server()
    return {"ok": True}


@app.post("/api/restart")
def api_restart():
    s = load_state(); s["desired"] = "running"; save_state(s)
    def go():
        stop_server(); start_server()
    threading.Thread(target=go, daemon=True).start()
    return {"ok": True}


@app.post("/api/settings")
def api_settings(body: Settings):
    s = load_state()
    upd = body.model_dump(exclude_none=True)
    if "model" in upd and not (ROOT / upd["model"] / "config.json").exists():
        raise HTTPException(400, f"no model at {upd['model']}")
    if "cap_gb" in upd and not 40 <= upd["cap_gb"] <= 110:
        raise HTTPException(400, "cap_gb must be 40-110")
    if "draft_block" in upd and not 1 <= upd["draft_block"] <= 8:
        raise HTTPException(400, "draft_block must be 1-8")
    s.update(upd)
    save_state(s)
    return {"ok": True, "restart_needed": any(k != "keep_awake" for k in upd)}


@app.get("/api/models")
def api_models():
    models = sorted(str(p.parent.relative_to(ROOT)) for p in (ROOT / "models").glob("*/config.json")
                    if "drafter" not in p.parent.name)
    return {"models": models}


if __name__ == "__main__":
    threading.Thread(target=supervisor, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CONTROL_PORT", "8090")), log_level="warning")
