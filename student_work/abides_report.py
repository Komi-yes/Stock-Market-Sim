"""Summarise abides_eval JSON results per model, with a rubric-like ABIDES SCORE (orchestrator-owned).

Usage:
  python -m student_work.abides_report outputs/abides_eval/r5_diag_g11.json [more.json ...] [--add-hold] [--mode individual]

The SCORE reuses student_work.evaluate.composite, so it is rank-normalised inside the pool of models listed:
  R  mean return % over every run (template x seed)
  D  mean max drawdown %
  C  rank(worst template mean return) and rank(share of profitable runs)
  O  skipped orders per 100 wakeups + share of runs with |inventory| > 400 + 10 x share of runs with max DD >= 100%
--add-hold adds an exact always-HOLD row per (template, seed) (HOLD is verified to give 0 return and 0 DD in ABIDES).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from student_work.evaluate import composite


def load_rows(paths, mode=None) -> list[dict]:
    rows = []
    for f in paths:
        rows += json.loads(Path(f).read_text())["rows"]
    if mode:
        rows = [r for r in rows if r["mode"] == mode]
    return rows


def add_hold(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["template"], r["seed"], r["mode"]) for r in rows})
    hold = [dict(name="HOLD", template=t, seed=s, mode=m, return_pct=0.0, max_drawdown_pct=0.0,
                 skipped_total=0, wakeups=1, max_abs_inventory=0, n_fills=0) for t, s, m in keys]
    return [r for r in rows if r["name"] != "HOLD"] + hold


def templates_of(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["template"] not in seen:
            seen.append(r["template"])
    return seen


def summarise(rs: list[dict]) -> dict:
    ret = np.array([r["return_pct"] for r in rs])
    dd = np.array([r["max_drawdown_pct"] for r in rs])
    per_t = {t: float(np.mean([r["return_pct"] for r in rs if r["template"] == t])) for t in templates_of(rs)}
    per_t_dd = {t: float(np.max([r["max_drawdown_pct"] for r in rs if r["template"] == t])) for t in templates_of(rs)}
    skips = sum(r["skipped_total"] for r in rs)
    wakes = max(1, sum(r["wakeups"] for r in rs))
    return dict(
        n=len(rs), R=float(ret.mean()), median_R=float(np.median(ret)), D=float(dd.mean()), max_DD=float(dd.max()),
        worst_scenario=float(min(per_t.values())), profitable_share=float((ret > 0).mean()),
        O=float(100.0 * skips / wakes + np.mean([r["max_abs_inventory"] > 400 for r in rs])
                + 10.0 * np.mean(dd >= 100.0)),
        skipped_per_run=float(skips / len(rs)), max_abs_inventory=int(max(r["max_abs_inventory"] for r in rs)),
        fills_per_run=float(np.mean([r.get("n_fills", 0) for r in rs])),
        by_template=per_t, max_dd_by_template=per_t_dd,
    )


def pool_of(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    return {n: summarise(rs) for n, rs in by.items()}


def print_report(pool: dict[str, dict]) -> None:
    comp = composite(pool)
    order = sorted(pool, key=lambda k: -comp[k]["score"])
    print(f"\n{'name':22s} {'SCORE':>6s} {'n':>4s} {'R%':>7s} {'medR%':>7s} {'DD%':>6s} {'maxDD':>6s} {'worstT':>7s} "
          f"{'prof':>5s} {'O':>6s} {'skip':>5s} {'maxInv':>6s} {'fills':>6s}")
    for n in order:
        x = pool[n]
        print(f"{n:22s} {comp[n]['score']:6.3f} {x['n']:4d} {x['R']:7.2f} {x['median_R']:7.2f} {x['D']:6.2f} "
              f"{x['max_DD']:6.1f} {x['worst_scenario']:7.2f} {x['profitable_share']:5.2f} {x['O']:6.2f} "
              f"{x['skipped_per_run']:5.1f} {x['max_abs_inventory']:6d} {x['fills_per_run']:6.0f}")
    tpls = list(next(iter(pool.values()))["by_template"])
    for label, key in (("mean return %", "by_template"), ("max DD %", "max_dd_by_template")):
        print(f"\nPer-template {label}:")
        print(f"{'template':32s} " + " ".join(f"{n[:11]:>11s}" for n in order))
        for t in tpls:
            print(f"{t:32s} " + " ".join(f"{pool[n][key].get(t, float('nan')):11.2f}" for n in order))


def main():
    p = argparse.ArgumentParser(description="Summarise abides_eval results with a rubric-like SCORE.")
    p.add_argument("json", nargs="+", type=Path)
    p.add_argument("--add-hold", action="store_true", help="add exact always-HOLD rows")
    p.add_argument("--mode", choices=["individual", "competitive"], default=None)
    p.add_argument("--only", nargs="*", default=None, help="keep only these model names")
    a = p.parse_args()
    rows = load_rows(a.json, a.mode)
    if a.only:
        rows = [r for r in rows if r["name"] in a.only]
    if a.add_hold:
        rows = add_hold(rows)
    print_report(pool_of(rows))


if __name__ == "__main__":
    main()
