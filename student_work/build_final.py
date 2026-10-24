"""Rebuild the selected final policy (see student_work/EXPERIMENTS.md sections 6 and 7) and write it to outputs/model.pt.

The final model (v2, stage 2) is a TorchScript SafePolicy around BookNeutral(Ensemble) of 9 DQN members:
  * 5 from round 2 (student_work/experiments/combo.py; random practice regimes; heavy or medium risk reward)
  * 2 from round 3 (student_work/experiments/abides_like.py; mixed practice + ABIDES-like regimes; heavy reward)
  * 2 from round 6 (student_work/experiments/seed_farm.py; combo recipe C: medium reward + training mask 300)
BookNeutral feeds the members practice means of the order-book features; the guards read the raw observation.
`--v1` rebuilds the stage-1 model instead (7 members, resting_mode 0, no BookNeutral).

To retrain the members from scratch first (each takes about 3 minutes on CPU):
  python -m student_work.experiments.combo --reward medium --mask-cap 300 --seed 1 --out outputs/exp_combo-C_s1/model.pt
  python -m student_work.experiments.combo --reward medium --mask-cap 300 --seed 4 --out outputs/exp_combo-C_s4/model.pt
  python -m student_work.experiments.combo --reward medium --mask-cap 0   --seed 4 --out outputs/exp_combo-A_s4/model.pt
  python -m student_work.experiments.combo --reward heavy  --mask-cap 0   --seed 5 --out outputs/exp_combo-B_s5/model.pt
  python -m student_work.experiments.combo --reward heavy  --mask-cap 0   --seed 2 --out outputs/exp_combo-B_s2/model.pt
  python -m student_work.experiments.abides_like --family mixed --seed 3 --out outputs/exp_abides_like-mixed_s3/model.pt
  python -m student_work.experiments.abides_like --family mixed --seed 5 --out outputs/exp_abides_like-mixed_s5/model.pt
  python -m student_work.experiments.combo --reward medium --mask-cap 300 --seed 12 --out outputs/exp_farm-C_s12/model.pt
  python -m student_work.experiments.combo --reward medium --mask-cap 300 --seed 7  --out outputs/exp_farm-C_s7/model.pt
Then: python -m student_work.build_final [--out outputs/model.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from student_work.evaluate import check_contract
from student_work.model import BookNeutral, Ensemble, SafePolicy

MEMBERS = [   # v1's members (stage 1)
    "outputs/exp_combo-C_s1/model.pt",
    "outputs/exp_combo-C_s4/model.pt",
    "outputs/exp_combo-A_s4/model.pt",
    "outputs/exp_combo-B_s5/model.pt",
    "outputs/exp_combo-B_s2/model.pt",
    "outputs/exp_abides_like-mixed_s3/model.pt",
    "outputs/exp_abides_like-mixed_s5/model.pt",
]
V2_EXTRA = [  # added in stage 2 by greedy selection on tuning (EXPERIMENTS.md round 7)
    "outputs/exp_farm-C_s12/model.pt",
    "outputs/exp_farm-C_s7/model.pt",
]
GUARD = dict(
    max_inventory=400.0,    # share cap (the contract limit is 500)
    min_cash_ratio=0.7,     # no buys below 70% of starting cash: long exposure <= ~30% of NAV at any price
    max_cash_ratio=1.3,     # no new shorts above 130% of starting cash: short exposure <= ~30% of NAV
    brake_nav=-0.05,        # below -5% NAV return, only de-risk (forced market orders toward flat)
    max_spread_obs=0.5,     # spread > 100 bps (broken book): no exposure-growing actions
)
V2_RESTING_MODE = 3         # while limit orders rest, no exposure-growing LARGE actions


def build(v1: bool = False):
    members = MEMBERS if v1 else MEMBERS + V2_EXTRA
    base = Ensemble([torch.jit.load(m, map_location="cpu").eval() for m in members])
    if v1:
        return torch.jit.script(SafePolicy(base, **GUARD).eval()), len(members)
    return torch.jit.script(SafePolicy(BookNeutral(base), **GUARD, resting_mode=V2_RESTING_MODE).eval()), len(members)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/model.pt"))
    p.add_argument("--v1", action="store_true", help="rebuild the stage-1 model (v1) instead of v2")
    a = p.parse_args()
    model, n = build(a.v1)
    check_contract(model)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(a.out))
    print(f"Saved final policy {'v1' if a.v1 else 'v2'} ({n} members, guard {GUARD}) to {a.out}")


if __name__ == "__main__":
    main()
