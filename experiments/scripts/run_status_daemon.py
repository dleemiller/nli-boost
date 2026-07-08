#!/usr/bin/env python
"""Status daemon for hypothesis-vectorizer runs — a real daemon, not a throwaway loop.

Watches one or more run roots and renders a compact, tail-friendly status block to a log every
INTERVAL seconds. Study-agnostic and pipeline-agnostic: it auto-detects the format of each run dir.

  * learning-curve runs  (results/raw/<id>/results.jsonl)  -> rows vs shots×seeds×systems
  * tree-evolve runs     (<runs>/<id>/log.jsonl)           -> rounds vs pool.tree.rounds

Run roots:
  * always: experiments/results/raw            (the learning-curve harness)
  * plus:   any path listed in experiments/results/logs/watch_roots.txt (one per line, re-read each
            cycle) — so you can add a worktree's runs/ dir live without restarting the service.

Meant to run under systemd --user (Restart=on-failure, survives logout via linger). Handles
SIGTERM/SIGINT cleanly, guards against a second instance with a pidfile, and NEVER self-terminates
(no idle timeout — lifecycle is systemd's job).

    systemctl --user start  hv-status      # start
    systemctl --user status hv-status      # is it up?
    journalctl --user -u hv-status -f      # service-level logs
    tail -f experiments/results/logs/run_status.log   # the status board
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import signal
import subprocess
import time
from datetime import datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_LC_ROOT = REPO / "experiments" / "results" / "raw"
STATE_DIR = REPO / "experiments" / "results" / "logs"
DEFAULT_LOG = STATE_DIR / "run_status.log"
WATCH_ROOTS_FILE = STATE_DIR / "watch_roots.txt"
PIDFILE = STATE_DIR / "status_daemon.pid"
RECENT_SECS = 150  # results file touched within this => active even if no process match (slow fits)

_STOP = False


def _handle(signum, _frame):
    global _STOP
    _STOP = True


signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)


# ---------------------------------------------------------------------------- shared helpers
def run_alive(run_id: str) -> bool:
    """A process referencing this run_id is up — excluding watcher shells / this daemon."""
    out = subprocess.run(["pgrep", "-af", run_id], capture_output=True, text=True).stdout
    for line in out.splitlines():
        low = line.lower()
        if any(t in low for t in ("kill -0", "until ", "grep ", "pgrep", "run_status_daemon")):
            continue
        return True
    return False


def gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        util, used, total = [x.strip() for x in out.split(",")]
        return f"GPU {util}% util  {int(used)/1024:.1f}/{int(total)/1024:.0f}GB"
    except Exception:
        return "GPU n/a"


def bar(frac: float, width: int = 10) -> str:
    f = max(0.0, min(frac, 1.0))
    filled = int(round(f * width))
    return "█" * filled + "░" * (width - filled)


def _tree_rounds_total(run_dir: pathlib.Path) -> int | None:
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return None
    try:  # prefer real yaml (venv has it); fall back to a scoped line scan
        import yaml
        c = yaml.safe_load(cfg.read_text())
        return int(c["pool"]["tree"]["rounds"])
    except Exception:
        in_tree = False
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("tree:"):
                in_tree = True
            elif in_tree and "rounds:" in line:
                try:
                    return int(line.split("rounds:")[1].strip())
                except Exception:
                    return None
        return None


# ---------------------------------------------------------------------------- per-format scanners
def scan_lc(run_dir: pathlib.Path) -> dict:
    f = run_dir / "results.jsonl"
    rows, last, nerr, cells = 0, None, 0, collections.defaultdict(int)
    for line in f.open():
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            r = json.loads(line)
            last = r
            cells[(r.get("shots"), r.get("seed"))] += 1
            if r.get("error"):
                nerr += 1
        except Exception:
            pass
    total = None
    man = run_dir / "manifest.json"
    if man.exists() and cells:
        try:
            cfg = json.loads(man.read_text()).get("config", {})
            total = len(cfg["shots"]) * int(cfg["seeds"]) * max(cells.values())
        except Exception:
            total = None
    pos = ""
    if last:
        pos = f"k={last.get('shots')!s:>3} s={last.get('seed')} {str(last.get('system',''))[:24]}"
    return {"count": rows, "total": total, "pos": pos, "nerr": nerr,
            "mtime": f.stat().st_mtime, "unit": "rows",
            "done_extra": None}


def scan_tree(run_dir: pathlib.Path) -> dict:
    f = run_dir / "log.jsonl"
    rounds, last = 0, None
    for line in f.open():
        line = line.strip()
        if not line:
            continue
        rounds += 1
        try:
            last = json.loads(line)
        except Exception:
            pass
    total = _tree_rounds_total(run_dir)
    pos = ""
    if last:
        ig = last.get("info_gain")
        pos = f"rnd {last.get('round')} leaf {last.get('leaf')} IG={ig:.3f}" if ig is not None else \
              f"rnd {last.get('round')}"
    # a finished tree run has metrics.json with pool_cv
    done_extra = None
    met = run_dir / "metrics.json"
    if met.exists():
        try:
            r = json.loads(met.read_text()).get("results", {}).get("pool_cv", {})
            if r:
                done_extra = f"acc={r.get('accuracy')} f1={r.get('macro_f1')}"
        except Exception:
            pass
    return {"count": rounds, "total": total, "pos": pos, "nerr": 0,
            "mtime": f.stat().st_mtime, "unit": "rnds", "done_extra": done_extra}


def detect_and_scan(run_dir: pathlib.Path):
    if (run_dir / "results.jsonl").exists():
        return "lc", scan_lc(run_dir)
    if (run_dir / "log.jsonl").exists():
        return "tree", scan_tree(run_dir)
    return None, None


# ---------------------------------------------------------------------------- roots + pidfile
def discover_roots() -> list[pathlib.Path]:
    roots = [DEFAULT_LC_ROOT]
    if WATCH_ROOTS_FILE.exists():
        for line in WATCH_ROOTS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                p = pathlib.Path(line).expanduser()
                if p not in roots:
                    roots.append(p)
    return roots


def claim_pidfile() -> None:
    if PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
            os.kill(old, 0)  # alive?
            raise SystemExit(f"[status daemon] already running (pid {old}); refusing to start a second.")
        except (ValueError, ProcessLookupError):
            pass  # stale
    PIDFILE.write_text(str(os.getpid()))


# ---------------------------------------------------------------------------- main loop
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    ap.add_argument("--interval", type=float, default=20.0)
    args = ap.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    claim_pidfile()
    with args.log.open("a") as fh:
        fh.write(f"\n[status daemon] started pid {os.getpid()} at {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    history: dict = collections.defaultdict(lambda: collections.deque(maxlen=15))
    try:
        while not _STOP:
            now = time.time()
            run_dirs = []
            for root in discover_roots():
                if root.exists():
                    run_dirs += [p for p in sorted(root.iterdir()) if p.is_dir()]

            lines = [f"\n── {datetime.now():%H:%M:%S} ─ {gpu()} " + "─" * 18]
            active, idle = [], []
            for rd in run_dirs:
                fmt, s = detect_and_scan(rd)
                if not fmt:
                    continue
                rid = rd.name
                key = str(rd)
                history[key].append((now, s["count"]))
                proc = run_alive(rid)
                recent = s["mtime"] and (now - s["mtime"]) < RECENT_SECS
                is_active = proc or recent
                total = s["total"]
                frac = min(s["count"] / total, 1.0) if total else 0.0

                if is_active:
                    state = "RUNNING"
                elif total and s["count"] >= total:
                    state = "done   "
                elif s["done_extra"]:  # tree run with metrics.json but rounds<total (stopped on patience)
                    state = "done   "
                elif s["count"] == 0:
                    state = "waiting"
                elif not total:
                    state = "idle   "
                else:
                    state = "stopped"

                eta = ""
                h = history[key]
                if is_active and len(h) >= 2 and h[-1][1] > h[0][1] and total:
                    dt, dr = h[-1][0] - h[0][0], h[-1][1] - h[0][1]
                    rpm = dr / dt * 60 if dt > 0 else 0
                    if rpm > 0:
                        eta = f"  {rpm:.1f} {s['unit']}/min  ETA ~{(total - s['count']) / rpm:.0f}m"

                tot_s = str(total) if total else "?"
                tag = f"[{fmt}]"
                extra = f"  {s['done_extra']}" if (state.strip() == "done" and s["done_extra"]) else ""
                errtag = f"  [{s['nerr']} err]" if s["nerr"] else ""
                pos = f"  {s['pos']}" if s["pos"] else ""
                line = (f" {tag:<6} {rid:<28} {state}  [{bar(frac)}] "
                        f"{s['count']:>4}/{tot_s:<4} {frac*100:>3.0f}%{pos}{eta}{extra}{errtag}")
                (active if is_active else idle).append((s["mtime"] or 0, line))

            for _, ln in sorted(active, key=lambda x: -x[0]):
                lines.append(ln)
            for _, ln in sorted(idle, key=lambda x: -x[0])[:6]:
                lines.append(ln)
            if not active:
                lines.append(" (nothing active)")

            with args.log.open("a") as fh:
                fh.write("\n".join(lines) + "\n")

            # responsive to SIGTERM: sleep in short ticks
            slept = 0.0
            while slept < args.interval and not _STOP:
                time.sleep(0.5)
                slept += 0.5
    finally:
        with args.log.open("a") as fh:
            fh.write(f"[status daemon] stopped at {datetime.now():%H:%M:%S} (signal={_STOP}).\n")
        try:
            if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
