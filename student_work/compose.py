"""Build a TorchScript policy from saved TorchScript models: optional seed ensemble plus SafePolicy guard.

  python -m student_work.compose --base outputs/exp_a_s1/model.pt --base outputs/exp_a_s2/model.pt \
      --max-inventory 300 --out outputs/exp_a_ens_safe300/model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from student_work.evaluate import check_contract
from student_work.model import BookNeutral, Ensemble, SafePolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", action="append", required=True, type=Path)
    p.add_argument("--max-inventory", type=float, default=None, help="add SafePolicy with this cap")
    p.add_argument("--min-cash-ratio", type=float, default=0.1)
    p.add_argument("--max-cash-ratio", type=float, default=1e9,
                   help="block growing a short once cash exceeds this multiple of starting cash")
    p.add_argument("--brake-nav", type=float, default=-1e9,
                   help="below this NAV return (e.g. -0.05) only de-risk, forcing market orders toward flat")
    p.add_argument("--max-spread-obs", type=float, default=2.0,
                   help="block exposure-growing actions while the spread feature exceeds this (0.5 = 100 bps)")
    p.add_argument("--resting-mode", type=int, default=0, choices=[0, 1, 2, 3],
                   help="while orders rest (x[12] > 0) block exposure growth: 1 limit, 2 all, 3 LARGE actions")
    p.add_argument("--neutral-book", action="store_true",
                   help="feed the members practice means of the book features x[5..7] (guards still see raw x)")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    bases = [torch.jit.load(str(b), map_location="cpu").eval() for b in a.base]
    model = bases[0] if len(bases) == 1 else Ensemble(bases)
    if a.neutral_book:
        model = BookNeutral(model)
    if a.max_inventory is not None:
        model = SafePolicy(model, a.max_inventory, a.min_cash_ratio, a.max_cash_ratio, a.brake_nav,
                           a.max_spread_obs, a.resting_mode)
    scripted = model if isinstance(model, torch.jit.ScriptModule) else torch.jit.script(model.eval())
    check_contract(scripted)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(a.out))
    print(f"Saved composed policy to {a.out}")


if __name__ == "__main__":
    main()
