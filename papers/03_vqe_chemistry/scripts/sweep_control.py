"""Control interface for the running (or previously run) run_sweep_v2.py:
pause/resume/stop it without killing the process or losing in-flight
work, and check its progress at any time.

  pause   -- stop submitting new cells; in-flight cells finish normally,
             their results are saved, then the sweep idles until resumed
  resume  -- go back to submitting new cells
  stop    -- like pause, but the sweep process exits once in-flight
             cells finish (rerun run_sweep_v2.py later to resume -- it
             skips everything already in the results CSV)
  status  -- process alive? current control state? cells done/remaining?
             recent throughput and a rough ETA? tail of the log

Usage:
    python papers/03_vqe_chemistry/scripts/sweep_control.py status
    python papers/03_vqe_chemistry/scripts/sweep_control.py pause
    python papers/03_vqe_chemistry/scripts/sweep_control.py resume
    python papers/03_vqe_chemistry/scripts/sweep_control.py stop
"""

import csv
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "optimizer_sweep_v2.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "sweep_v2.log")
CONTROL_PATH = os.path.join(RESULTS_DIR, "sweep_control.json")
PID_PATH = os.path.join(RESULTS_DIR, "sweep_v2.pid")

N_REPEATS = 5


def _pid_alive():
    if not os.path.exists(PID_PATH):
        return None
    try:
        with open(PID_PATH) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # signal 0: existence check only, doesn't kill
        return pid
    except (OSError, ValueError):
        return None


def _set_state(state: str):
    with open(CONTROL_PATH, "w") as f:
        json.dump({"state": state}, f)
    print(f"Control state set to '{state}'.")
    pid = _pid_alive()
    print(f"Sweep process {'is running (pid ' + str(pid) + ')' if pid else 'is NOT currently running'}.")


def _completed_counts():
    counts = {}
    if not os.path.exists(RESULTS_PATH):
        return counts
    with open(RESULTS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["molecule"], row["optimizer"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def status():
    from molecules import MOLECULES
    from run_sweep_v2 import OPTIMIZERS

    total_cells = len(MOLECULES) * len(OPTIMIZERS)
    done = _completed_counts()
    cells_done = sum(1 for v in done.values() if v >= N_REPEATS)
    rows_done = sum(min(v, N_REPEATS) for v in done.values())
    total_rows = total_cells * N_REPEATS

    pid = _pid_alive()
    try:
        with open(CONTROL_PATH) as f:
            state = json.load(f).get("state", "run")
    except (FileNotFoundError, json.JSONDecodeError):
        state = "run (default, no control file yet)"

    print("=" * 70)
    print(f"Process:  {'RUNNING (pid ' + str(pid) + ')' if pid else 'NOT RUNNING'}")
    print(f"Control:  {state}")
    print(f"Cells:    {cells_done}/{total_cells} complete "
          f"({total_cells - cells_done} remaining)")
    print(f"Repeats:  {rows_done}/{total_rows} rows written")
    print("=" * 70)

    if not os.path.exists(LOG_PATH):
        print("(no log file yet)")
        return

    # Recent throughput / ETA from the last progress lines in the log.
    ts_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \[\s*(\d+)/(\d+)\]")
    recent = []
    with open(LOG_PATH) as f:
        for line in f:
            m = ts_re.match(line)
            if m:
                t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                recent.append((t, int(m.group(2)), int(m.group(3))))

    if len(recent) >= 2:
        window = recent[-20:]
        dt = window[-1][0] - window[0][0]
        dn = window[-1][1] - window[0][1]
        if dt > 0 and dn > 0:
            rate_per_hour = dn / (dt / 3600)
            remaining = window[-1][2] - window[-1][1]
            eta_hours = remaining / rate_per_hour if rate_per_hour > 0 else float("inf")
            print(f"Recent rate: {rate_per_hour:.1f} cells/hour  "
                  f"(from last {len(window)} completions)")
            print(f"Rough ETA:   {eta_hours:.1f}h remaining at this rate")

    print("\nLast 10 log lines:")
    with open(LOG_PATH) as f:
        lines = f.readlines()
    for line in lines[-10:]:
        print("  " + line.rstrip())


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("status", "pause", "resume", "stop"):
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "status":
        status()
    elif cmd == "pause":
        _set_state("pause")
    elif cmd == "resume":
        _set_state("run")
    elif cmd == "stop":
        _set_state("stop")


if __name__ == "__main__":
    main()
