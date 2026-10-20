"""Control daemon: owns the model server process, persists desired state, serves the dashboard.

Run under launchd (scripts/install-control.sh). Dashboard: http://127.0.0.1:8090
"""

import ctypes
import json
import os
import re
import secrets
import signal
import struct
import subprocess
import threading
import time
import urllib.request
from pathlib import Path, PurePosixPath

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "control" / "state.json"
PID_FILE = ROOT / "control" / "server.pid"
SERVER_LOG = ROOT / "logs" / "server.log"
DASHBOARD = ROOT / "control" / "dashboard.html"
GB = 1 << 30
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8090"))
ALLOWED_HOSTS = {f"127.0.0.1:{CONTROL_PORT}", f"localhost:{CONTROL_PORT}"}
ALLOWED_ORIGINS = {f"http://{h}" for h in ALLOWED_HOSTS}


def _secret(name: str) -> str:
    """Per-install secret stored mode 0600 next to the daemon (gitignored)."""
    path = ROOT / "control" / name
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32))
    except FileExistsError:
        pass
    st = path.lstat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077 or not path.is_file() or path.is_symlink():
        raise SystemExit(f"{path} must be a regular file owned by you with mode 0600")
    return path.read_text().strip()


CONTROL_TOKEN = _secret("control_token")
MODEL_API_KEY = _secret("api_key")

PROFILES_FILE = ROOT / "control" / "profiles.json"
PROFILE_KEYS = ("name", "detail", "model", "drafter", "mtp", "draft_block", "cap_gb", "prefill", "expected_gb")
DEFAULTS = {"desired": "stopped", "active": "qwen", "port": 8080, "keep_awake": False, "overrides": {}}
HEADROOM_GB = 8

lock = threading.RLock()
caffeinate: subprocess.Popen | None = None
started_at: float | None = None


def load_profiles(overrides: dict | None = None) -> dict:
    """profiles.json (defaults, committed) with the user's dashboard edits from state.json on top."""
    base = json.loads(PROFILES_FILE.read_text())
    overrides = overrides if overrides is not None else _raw_state().get("overrides", {})
    return {pid: {**p, **overrides.get(pid, {})} for pid, p in base.items()}


def _raw_state() -> dict:
    try:
        raw = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        raw = {}
    return {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}


def load_state() -> dict:
    """Persistent state plus the active profile's settings flattened in (model, mtp, ...)."""
    s = _raw_state()
    profiles = load_profiles(s["overrides"])
    if s["active"] not in profiles:
        s["active"] = next(iter(profiles))
    return {**s, **profiles[s["active"]]}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: state[k] for k in DEFAULTS}, indent=2))
    tmp.replace(STATE_FILE)


def resolve_drafter(s: dict) -> str:
    """Explicit drafter, else the `mtp/` drafter dir build/ writes next to the weights, else none."""
    if s.get("drafter"):
        return s["drafter"]
    return f"{s['model']}/mtp" if (ROOT / s["model"] / "mtp" / "config.json").exists() else ""


def model_disk_gb(model: str) -> float | None:
    path = ROOT / model
    if not (path / "config.json").exists():
        return None
    total = sum(f.stat().st_size for f in path.resolve().glob("*.safetensors"))
    return round(total / 1e9, 1)


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


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def phys_footprint(pid: int) -> int:
    """macOS phys_footprint (what Activity Monitor shows). Unlike RSS it counts MLX's Metal buffers."""
    buf = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(pid, 2, buf) != 0:  # RUSAGE_INFO_V2
        return 0
    return struct.unpack_from("<Q", buf, 72)[0]  # uuid[16] + 7 u64 fields precede ri_phys_footprint


def server_tree_rss(p: psutil.Process) -> int:
    try:
        procs = [p] + p.children(recursive=True)
        return sum(phys_footprint(c.pid) for c in procs if c.is_running())
    except psutil.Error:
        return 0


def api_ready(port: int) -> bool:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                 headers={"Authorization": f"Bearer {MODEL_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server() -> None:
    global started_at
    with lock:
        if server_proc():
            return
        s = load_state()
        drafter = resolve_drafter(s)
        env = {
            **os.environ,
            "MTP": "1" if s["mtp"] and drafter else "0",
            "DRAFTER": drafter,
            "DRAFT_BLOCK": str(s["draft_block"]),
            "CAP_GB": str(s["cap_gb"]),
            "PREFILL": str(s["prefill"]),
            "PORT": str(s["port"]),
            "MLX_VLM_SERVER_API_KEY": MODEL_API_KEY,
        }
        SERVER_LOG.parent.mkdir(exist_ok=True)
        log = open(SERVER_LOG, "ab")
        log.write(f"\n===== control: start {s['active']} ({s['model']}) {time.ctime()} =====\n".encode())
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
        group = [p]
        try:
            group += p.children(recursive=True)
            parent = p.parent()
            if parent is not None and parent.name() == "uv":
                group.append(parent)
        except psutil.Error:
            pass
        for q in group:
            try:
                q.send_signal(signal.SIGTERM)
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(group, timeout=20)
        for q in alive:
            try:
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
        "phase": phase, "desired": s["desired"], "settings": s, "active": s["active"],
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


@app.middleware("http")
async def guard(request: Request, call_next):
    # Host check defeats DNS rebinding; Origin + token header defeat cross-site requests from browser pages.
    if request.headers.get("host") not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "bad host"}, status_code=403)
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "bad origin"}, status_code=403)
    if request.url.path.startswith("/api/") and not secrets.compare_digest(
        request.headers.get("x-control-token", ""), CONTROL_TOKEN
    ):
        return JSONResponse({"detail": "missing control token"}, status_code=403)
    return await call_next(request)


def valid_model_path(model: str) -> bool:
    p = PurePosixPath(model)
    return (len(p.parts) == 2 and p.parts[0] == "models" and p.parts[1] not in ("..", ".")
            and (ROOT / p / "config.json").exists())


class Settings(BaseModel):
    profile: str | None = None
    model: str | None = None
    mtp: bool | None = None
    draft_block: int | None = None
    cap_gb: int | None = None
    prefill: int | None = None
    keep_awake: bool | None = None


class StartReq(BaseModel):
    profile: str | None = None
    force: bool = False


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD.read_text().replace("__CONTROL_TOKEN__", CONTROL_TOKEN)


@app.get("/api/status")
def api_status():
    return status()


def profiles_view() -> list[dict]:
    s = load_state()
    p = server_proc()
    running_gb = server_tree_rss(p) / GB if p else 0.0
    avail = psutil.virtual_memory().available / GB
    out = []
    for pid, prof in load_profiles().items():
        disk = model_disk_gb(prof["model"])
        # memory the model could get: what's free now plus what the running model would release
        room = avail + running_gb - HEADROOM_GB
        out.append({
            "id": pid, **prof, "built": disk is not None, "disk_gb": disk,
            "drafter_resolved": resolve_drafter(prof) if disk is not None else "",
            "active": pid == s["active"], "running": pid == s["active"] and p is not None,
            "fits": prof["expected_gb"] <= room, "room_gb": round(room, 1),
        })
    return out


@app.get("/api/profiles")
def api_profiles():
    return {"profiles": profiles_view()}


@app.post("/api/start")
def api_start(body: StartReq | None = None):
    body = body or StartReq()
    s = load_state()
    target = body.profile or s["active"]
    view = {v["id"]: v for v in profiles_view()}
    if target not in view:
        raise HTTPException(404, f"unknown profile {target}")
    if not view[target]["built"]:
        raise HTTPException(409, f"{view[target]['name']} is not built yet ({view[target]['model']})")
    if not view[target]["fits"] and not body.force:
        raise HTTPException(409, f"needs ~{view[target]['expected_gb']} GB, only ~{view[target]['room_gb']} GB free "
                                 f"after {HEADROOM_GB} GB headroom; close apps or force")
    switching = target != s["active"]
    s["active"], s["desired"] = target, "running"
    save_state(s)

    def go():
        if switching:
            stop_server()
        start_server()
    threading.Thread(target=go, daemon=True).start()
    return {"ok": True, "switched": switching}


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
    pid = upd.pop("profile", None) or s["active"]
    if pid not in load_profiles():
        raise HTTPException(404, f"unknown profile {pid}")
    if "model" in upd and not valid_model_path(upd["model"]):
        raise HTTPException(400, "model must be models/<name> containing config.json")
    if "cap_gb" in upd and not 40 <= upd["cap_gb"] <= 110:
        raise HTTPException(400, "cap_gb must be 40-110")
    if "draft_block" in upd and not 1 <= upd["draft_block"] <= 8:
        raise HTTPException(400, "draft_block must be 1-8")
    if "keep_awake" in upd:
        s["keep_awake"] = upd.pop("keep_awake")
    s["overrides"] = {**s["overrides"], pid: {**s["overrides"].get(pid, {}), **upd}}
    save_state(s)
    return {"ok": True, "restart_needed": bool(upd) and pid == s["active"] and server_proc() is not None}


if __name__ == "__main__":
    threading.Thread(target=supervisor, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=CONTROL_PORT, log_level="warning")
