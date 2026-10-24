"""Seed farm: train 30 members (recipes B, C, M x seeds 6-15) via existing modules.

  python -m student_work.experiments.seed_farm --list            # print one shell command per job
  python -m student_work.experiments.seed_farm --run --jobs 6    # run all missing jobs, 6 at a time
Logs go to outputs/exp_farm_logs/, models to outputs/exp_farm-<R>_s<S>/model.pt.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SEEDS = range(6, 16)
RECIPES = {
    "B": ["-m", "student_work.experiments.combo", "--reward", "heavy", "--mask-cap", "0"],
    "C": ["-m", "student_work.experiments.combo", "--reward", "medium", "--mask-cap", "300"],
    "M": ["-m", "student_work.experiments.abides_like", "--family", "mixed"],
}


def jobs():
    for r, args in RECIPES.items():
        for s in SEEDS:
            out = Path(f"outputs/exp_farm-{r}_s{s}/model.pt")
            yield r, s, out, [sys.executable, *args, "--seed", str(s), "--out", str(out)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    todo = [j for j in jobs() if not j[2].exists()]
    if a.list:
        for r, s, out, cmd in todo:
            print(f"{r}_s{s}\t{' '.join(cmd)}")
        return
    if a.run:
        logs = Path("outputs/exp_farm_logs"); logs.mkdir(parents=True, exist_ok=True)
        running = []
        while todo or running:
            while todo and len(running) < a.jobs:
                r, s, out, cmd = todo.pop(0)
                f = open(logs / f"farm-{r}_s{s}.log", "w")
                running.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f))
            running = [(pr, f) for pr, f in running if pr.poll() is None or f.close()]
            import time; time.sleep(5)


if __name__ == "__main__":
    main()
