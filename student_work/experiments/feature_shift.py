"""Round-6 W8: clamp / neutralise out-of-range market features before the Q-ensemble (SafePolicy sees RAW obs).

  python -m student_work.experiments.feature_shift measure   # writes outputs/exp_fshift/ranges.json
  python -m student_work.experiments.feature_shift build     # writes outputs/exp_fshift-<variant>/model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from student_work.build_final import GUARD, MEMBERS
from student_work.evaluate import check_contract
from student_work.model import Ensemble, SafePolicy

RANGES = Path("outputs/exp_fshift/ranges.json")
ALL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14]   # x[4] spread excluded from clamping by default? see variants
VARIANTS = {
    "a": dict(clamp=[0, 1, 2, 3, 4, 5, 6, 7, 13, 14]),
    "b": dict(clamp=[5, 6, 7]),
    "c": dict(neutral=[5, 6, 7]),           # x5 -> 0, x6/x7 -> practice mean
    "d": dict(clamp=[5, 6, 7], neutral=[]),  # placeholder, replaced below
}
VARIANTS["d"] = dict(neutral=[5, 6, 7], clamp=[0, 1, 2, 3, 13, 14])  # neutralise book + clamp return/vol features (not spread)


class ClampedBase(nn.Module):
    def __init__(self, base: nn.Module, lo: torch.Tensor, hi: torch.Tensor, neutral_mask: torch.Tensor, neutral_val: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.register_buffer("nmask", neutral_mask)
        self.register_buffer("nval", neutral_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.max(torch.min(x, self.hi), self.lo)
        c = torch.where(self.nmask > 0.5, self.nval.expand_as(c), c)
        return self.base(c)


def measure(n_scn_seeds=40, n_regimes=300):
    from shared.practice_market import PRACTICE_SCENARIOS, PracticeMarketEnv
    from student_work.envs import RandomRegimeEnv
    m = torch.jit.load("outputs/backup_v1/model.pt", map_location="cpu").eval()

    def run(env):
        obs, done, out = env.reset(), False, []
        while not done:
            out.append(obs)
            with torch.inference_mode():
                a = int(m(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax())
            obs, _, done, _ = env.step(a)
        return out
    allobs = []
    for scn in PRACTICE_SCENARIOS:
        for s in range(100, 100 + n_scn_seeds):
            allobs += run(PracticeMarketEnv(scn, steps=300, seed=s))
    for s in range(1000, 1000 + n_regimes):
        allobs += run(RandomRegimeEnv(steps=300, seed=s, family="practice"))
    A = np.asarray(allobs, dtype=np.float64)
    res = dict(n=len(A), p1=np.percentile(A, 1, 0).tolist(), p99=np.percentile(A, 99, 0).tolist(),
               mean=A.mean(0).tolist(), min=A.min(0).tolist(), max=A.max(0).tolist())
    RANGES.parent.mkdir(parents=True, exist_ok=True)
    RANGES.write_text(json.dumps(res, indent=1))
    for i in range(16):
        print(f"x[{i:2d}] p1={res['p1'][i]:9.4f} p99={res['p99'][i]:9.4f} mean={res['mean'][i]:9.4f} min={res['min'][i]:9.4f} max={res['max'][i]:9.4f}")


def build(variants):
    r = json.loads(RANGES.read_text())
    for v in variants:
        cfg = VARIANTS[v]
        lo = torch.full((1, 16), -1e9)
        hi = torch.full((1, 16), 1e9)
        for i in cfg.get("clamp", []):
            lo[0, i], hi[0, i] = r["p1"][i], r["p99"][i]
        nm = torch.zeros(1, 16)
        nv = torch.zeros(1, 16)
        for i in cfg.get("neutral", []):
            nm[0, i] = 1.0
            nv[0, i] = 0.0 if i == 5 else r["mean"][i]
        bases = [torch.jit.load(p, map_location="cpu").eval() for p in MEMBERS]
        model = torch.jit.script(SafePolicy(ClampedBase(Ensemble(bases), lo, hi, nm, nv), **GUARD).eval())
        check_contract(model)
        out = Path(f"outputs/exp_fshift-{v}/model.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print("saved", out, cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["measure", "build"])
    p.add_argument("--variants", nargs="*", default=["a", "b", "c", "d"])
    a = p.parse_args()
    measure() if a.cmd == "measure" else build(a.variants)
