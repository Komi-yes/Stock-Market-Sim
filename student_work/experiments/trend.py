"""Round-6 trend: DQN members trained on trend-rich, direction-symmetric regimes with a light inventory penalty.

p(trend)=0.7 (half up, half down), drift U(0.0002,0.0008); drawdown penalty kept, inv_quad much lighter.
Reuses combo.train; only swaps the env class (combo.py / envs.py / reward_risk.py untouched).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from student_work.envs import ExtendedMarketEnv  # noqa: E402
from student_work.experiments import combo  # noqa: E402
from student_work.experiments.reward_risk import VARIANTS  # noqa: E402

# (dd_coef, inv_quad, trade_cost, breach_pen); added at runtime
VARIANTS["trend"] = (2.0, 0.2, 0.05, 1.0)
VARIANTS["trend2"] = (2.0, 0.4, 0.05, 1.0)


def sample_trend_regime(rng: np.random.Generator) -> dict:
    if rng.random() < 0.7:
        drift = float(rng.choice([1.0, -1.0]) * rng.uniform(0.0002, 0.0008))
    else:
        drift = 0.0
    return dict(
        drift=drift,
        volatility=float(np.exp(rng.uniform(np.log(0.0005), np.log(0.007)))),
        spread_bps=float(np.exp(rng.uniform(np.log(3.0), np.log(50.0)))),
        depth=int(rng.uniform(120, 1000)),
        shock_prob=float(rng.choice([0.0, rng.uniform(0.002, 0.03)], p=[0.5, 0.5])),
        shock_scale=float(rng.uniform(0.01, 0.04)),
        fill_prob=float(rng.uniform(0.2, 0.6)),
        adverse=float(rng.uniform(0.0, 0.5)),
        impact_bps=float(rng.uniform(0.0, 8.0)),
        open_order_window=int(rng.integers(0, 8)),
    )


class TrendRegimeEnv(ExtendedMarketEnv):
    def __init__(self, steps=300, seed=0, random_length: bool = True):
        rng = np.random.default_rng(20_000_033 + int(seed))
        if random_length:
            steps = int(rng.integers(150, 1001))
        super().__init__(scenario="random", steps=steps, seed=seed, cfg=sample_trend_regime(rng))


class TrendShapedEnv(combo.ShapedRegimeEnv):
    def __init__(self, env_seed: int, params):
        self.env = TrendRegimeEnv(seed=env_seed, random_length=True)
        self.params = params
        self.steps = self.env.steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--variant", choices=["trend", "trend2"], default="trend")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert a.seed < 5000  # env seeds seed*1000+episode stay below the evaluation seeds (10000+)
    combo.ShapedRegimeEnv = TrendShapedEnv
    combo.train(a.variant, 0, a.steps, a.seed, a.out)


if __name__ == "__main__":
    main()
