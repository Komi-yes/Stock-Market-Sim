"""Round-6 W9: SafePolicy2 = SafePolicy + resting-order awareness + broken-book brake handling.

With all new options off it is bit-identical to student_work.model.SafePolicy.
  resting_mode: 0 off | 1 demote exposure-growing LIMIT actions while x[12] > 0 | 2 demote ALL exposure-growing actions
  broken_brake: 0 off (force market order, as SafePolicy) | 1 HOLD when braking and book broken | 2 force the LIMIT shrinking order instead
  near_cash_margin (a3): with resting orders, also demote LARGE exposure-growing actions when cash is within margin of a bound
CLI: python -m student_work.experiments.guard2 --variant a1|a2|b1|b2|a2b1|a2b1c65|a2b1c60|equiv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from student_work.build_final import GUARD, MEMBERS
from student_work.evaluate import check_contract
from student_work.model import Ensemble


class SafePolicy2(nn.Module):
    def __init__(self, base: nn.Module, max_inventory: float = 500.0, min_cash_ratio: float = 0.1,
                 max_cash_ratio: float = 1e9, brake_nav: float = -1e9, max_spread_obs: float = 2.0,
                 resting_mode: int = 0, broken_brake: int = 0, near_cash_margin: float = 0.0):
        super().__init__()
        self.base = base
        self.max_inventory = float(max_inventory)
        self.min_cash_ratio = float(min_cash_ratio)
        self.max_cash_ratio = float(max_cash_ratio)
        self.brake_nav = float(brake_nav)
        self.max_spread_obs = float(max_spread_obs)
        self.resting_mode = int(resting_mode)
        self.broken_brake = int(broken_brake)
        self.near_cash_margin = float(near_cash_margin)
        self.register_buffer("signed_qty", torch.tensor([[0.0, 10.0, -10.0, 50.0, -50.0, 10.0, -10.0, 50.0, -50.0]]))
        self.register_buffer("action_index", torch.arange(9, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("is_limit", torch.tensor([[0.0, 0, 0, 0, 0, 1, 1, 1, 1]]))
        self.register_buffer("is_large", torch.tensor([[0.0, 0, 0, 1, 1, 0, 0, 1, 1]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.base(x)
        inv = x[:, 8:9] * 500.0
        new_inv = inv + self.signed_qty
        grows = new_inv.abs() > inv.abs()
        too_big = (new_inv.abs() > self.max_inventory) & grows
        cash = x[:, 9:10] * 2.0
        no_cash = (cash < self.min_cash_ratio) & (self.signed_qty > 0.0)
        over_short = (cash > self.max_cash_ratio) & (new_inv < 0.0) & grows
        bad = too_big | no_cash | over_short
        braking = (x[:, 10:11] * 0.2) < self.brake_nav
        broken = x[:, 4:5] > self.max_spread_obs
        bad = bad | ((braking | broken) & grows)
        resting = x[:, 12:13] > 0.0
        if self.resting_mode == 1:
            bad = bad | (resting & grows & (self.is_limit > 0.5))
        elif self.resting_mode == 2:
            bad = bad | (resting & grows)
        if self.near_cash_margin > 0.0:
            near = (cash < self.min_cash_ratio + self.near_cash_margin) | (cash > self.max_cash_ratio - self.near_cash_margin)
            bad = bad | (resting & near & grows & (self.is_large > 0.5))
        floor = q.min(dim=1, keepdim=True).values - 1.0
        out = torch.where(bad, floor, q)
        big = inv.abs() >= 50.0
        idx = torch.where(inv > 0.0, torch.where(big, torch.full_like(inv, 4.0), torch.full_like(inv, 2.0)),
                          torch.where(big, torch.full_like(inv, 3.0), torch.full_like(inv, 1.0)))
        force = braking & (inv.abs() > 0.5)
        top = out.max(dim=1, keepdim=True).values + 1.0
        if self.broken_brake == 1:
            hold_now = force & broken
            force = force & ~broken
            out = torch.where(hold_now & (self.action_index == 0.0), top, out)
            top = out.max(dim=1, keepdim=True).values + 1.0
        elif self.broken_brake == 2:
            idx = torch.where(broken, idx + 4.0, idx)
        hit = self.action_index == idx
        return torch.where(force & hit, top, out)


VARIANTS = {
    "a1": dict(resting_mode=1),
    "a2": dict(resting_mode=2),
    "b1": dict(broken_brake=1),
    "b2": dict(broken_brake=2),
    "a2b1": dict(resting_mode=2, broken_brake=1),
    "a2b1c65": dict(resting_mode=2, broken_brake=1, min_cash_ratio=0.65, max_cash_ratio=1.35),
    "a2b1c60": dict(resting_mode=2, broken_brake=1, min_cash_ratio=0.6, max_cash_ratio=1.4),
    "a1b2": dict(resting_mode=1, broken_brake=2),
    "equiv": dict(),
}


def build(variant: str):
    g = dict(GUARD)
    g.update(VARIANTS[variant])
    bases = [torch.jit.load(m, map_location="cpu").eval() for m in MEMBERS]
    return torch.jit.script(SafePolicy2(Ensemble(bases), **g).eval()), g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    a = p.parse_args()
    model, g = build(a.variant)
    check_contract(model)
    if a.variant == "equiv":
        ref = torch.jit.load("outputs/backup_v1/model.pt", map_location="cpu").eval()
        torch.manual_seed(0)
        x = torch.cat([torch.rand(2000, 16) * 2 - 1, torch.zeros(1, 16)])
        with torch.no_grad():
            o1, o2 = model(x), ref(x)
        print("max|diff| =", (o1 - o2).abs().max().item(), "| argmax equal:", bool((o1.argmax(1) == o2.argmax(1)).all()))
        return
    out = Path(f"outputs/exp_guard2-{a.variant}/model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print("saved", out, g)


if __name__ == "__main__":
    main()
