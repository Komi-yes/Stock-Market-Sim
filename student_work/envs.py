"""Extended practice markets for training and evaluation (orchestrator-owned; workers import, never edit).

`shared/practice_market.py` stays untouched; these classes subclass it. Compared with the base env they add:
  * stress scenarios the base env lacks (down-trends, crashes, thin books, choppy markets);
  * a configurable limit-order fill probability and adverse selection (passive orders fill more often when
    the price is about to move against them, as in a real book);
  * permanent market impact for market orders;
  * a non-zero `open_orders` feature (recently unfilled limit orders count as resting).

With the default knobs (fill_prob=0.55, adverse=0, impact_bps=0, no open orders) the dynamics match the base
env, except that the price move is drawn before the limit fill, so the random stream differs.
"""
from __future__ import annotations

import math

import numpy as np

from shared.contract import (
    ACTION_SIZE, POSITION_LIMIT, STARTING_CASH_CENTS, action_to_order, build_observation,
)
from shared.practice_market import PRACTICE_SCENARIOS, PracticeMarketEnv

_DEFAULTS = dict(fill_prob=0.55, adverse=0.0, impact_bps=0.0, open_order_window=0,
                 price=10_000.0, imbalance_noise=0.0)
# price: starting mid in cents (ABIDES templates use ~100,000 = $1000, so cash caps long exposure near 100 shares).
# imbalance_noise: if > 0, bid/ask depth = depth * (1 +/- u), u ~ U(-k, k), giving the wide book imbalance seen in ABIDES.
# The base env uses +/-100 shares of noise (imbalance about +/-0.1).

STRESS_SCENARIOS = {
    "crash": dict(drift=-0.0006, volatility=0.0035, spread_bps=12.0, depth=500, shock_prob=0.03,
                  shock_scale=0.03, fill_prob=0.45, adverse=0.3, impact_bps=2.0, open_order_window=5),
    "downtrend": dict(drift=-0.00035, volatility=0.0012, spread_bps=6.0, depth=750, shock_prob=0.0,
                      shock_scale=0.0, fill_prob=0.55, adverse=0.2, impact_bps=0.5, open_order_window=5),
    "low_liquidity": dict(drift=0.0, volatility=0.0020, spread_bps=40.0, depth=150, shock_prob=0.005,
                          shock_scale=0.02, fill_prob=0.25, adverse=0.4, impact_bps=8.0, open_order_window=5),
    "choppy": dict(drift=0.0, volatility=0.0060, spread_bps=20.0, depth=400, shock_prob=0.01,
                   shock_scale=0.02, fill_prob=0.50, adverse=0.3, impact_bps=2.0, open_order_window=5),
}

ALL_SCENARIOS = {**{k: {**v, **_DEFAULTS} for k, v in PRACTICE_SCENARIOS.items()}, **STRESS_SCENARIOS}


class ExtendedMarketEnv(PracticeMarketEnv):
    """PracticeMarketEnv with an explicit config dict (any key of ALL_SCENARIOS entries)."""

    def __init__(self, scenario="stable", steps=300, seed=0, cfg: dict | None = None):
        if cfg is None:
            if scenario not in ALL_SCENARIOS:
                raise ValueError(f"Unknown scenario {scenario}. Choose {list(ALL_SCENARIOS)}")
            cfg = ALL_SCENARIOS[scenario]
        self.scenario = scenario
        self.steps = int(steps)
        self.seed = int(seed)
        self.cfg = {**_DEFAULTS, **cfg}
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self):
        self._unfilled: list[int] = []  # step index of each recent unfilled limit order
        self.insolvent = False
        obs = super().reset()
        if float(self.cfg["price"]) != 10_000.0:
            self.mid = float(self.cfg["price"])
            self.history = [self.mid]
            obs = self._obs()
        return obs

    def _book(self):
        k = float(self.cfg["imbalance_noise"])
        if k <= 0:
            return super()._book()
        spread = max(1.0, self.mid * self.cfg["spread_bps"] / 10_000.0)
        bid = int(round(self.mid - spread / 2.0))
        ask = int(round(self.mid + spread / 2.0))
        if ask <= bid:
            ask = bid + 1
        u = float(self.rng.uniform(-k, k))
        bid_depth = max(10, int(self.cfg["depth"] * (1.0 + u)))
        ask_depth = max(10, int(self.cfg["depth"] * (1.0 - u)))
        return bid, ask, bid_depth, ask_depth

    def _open_orders(self) -> int:
        w = int(self.cfg["open_order_window"])
        if w <= 0:
            return 0
        self._unfilled = [t for t in self._unfilled if self.t - t < w]
        return len(self._unfilled)

    def _obs(self):
        bid, ask, bd, ad = self._book()
        return build_observation(
            self.history, bid, ask, bd, ad, self.inventory, self.cash,
            time_progress=self.t / max(1, self.steps), open_orders=self._open_orders(),
        )

    def step(self, action: int):
        if not 0 <= int(action) < ACTION_SIZE:
            raise ValueError("Action must be an integer 0..8")
        cfg = self.cfg
        bid, ask, bd, ad = self._book()
        order = action_to_order(action, bid, ask)

        # Draw the price move first so limit fills can be adversely selected against it.
        shock = 0.0
        if self.rng.random() < cfg["shock_prob"]:
            shock = self.rng.choice([-1.0, 1.0]) * cfg["shock_scale"]
        ret = cfg["drift"] + self.rng.normal(0.0, cfg["volatility"]) + shock

        filled = False
        qty = int(order["quantity"])
        is_buy = order["side"] == "BUY"
        if order["kind"] == "MARKET":
            filled = qty > 0
            price = ask if is_buy else bid
        elif order["kind"] == "LIMIT":
            against = (ret < 0) if is_buy else (ret > 0)
            p = cfg["fill_prob"] * (1.0 + cfg["adverse"] if against else 1.0 - cfg["adverse"])
            filled = self.rng.random() < min(1.0, max(0.0, p))
            price = int(order["limit_price"])
            if not filled:
                self._unfilled.append(self.t)
        else:
            price = int(round(self.mid))

        if filled:
            signed = qty if is_buy else -qty
            proposed = self.inventory + signed
            if abs(proposed) <= POSITION_LIMIT and (signed < 0 or self.cash >= signed * price):
                self.inventory = proposed
                self.cash -= signed * price
                self.trade_count += 1
                if order["kind"] == "MARKET" and cfg["impact_bps"] > 0:
                    ret += math.copysign(cfg["impact_bps"] / 10_000.0 * qty / 50.0, signed)

        self.mid = max(100.0, self.mid * math.exp(ret))
        self.history.append(self.mid)
        self.t += 1

        nav = self.cash + self.inventory * self.mid
        pnl_change = nav - self.prev_nav
        self.prev_nav = nav
        if nav <= 0:
            self.insolvent = True
        reward = pnl_change / 1000.0 - 0.0005 * abs(self.inventory)
        done = self.t >= self.steps or self.insolvent
        info = {
            "nav_cents": float(nav), "cash_cents": int(self.cash),
            "inventory": int(self.inventory), "mid_cents": float(self.mid),
            "trade_count": int(self.trade_count), "scenario": self.scenario,
            "insolvent": self.insolvent,
        }
        return self._obs(), float(reward), done, info


def sample_regime(rng: np.random.Generator) -> dict:
    """Random market regime for domain-randomised training. Covers up/down trends, calm to crisis."""
    drift = float(rng.choice([0.0, 1.0, -1.0], p=[0.4, 0.3, 0.3]) * rng.uniform(0.0001, 0.0007))
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


def sample_abides_regime(rng: np.random.Generator) -> dict:
    """ABIDES-NG-like regime, from the round-1 harness feature comparison.

    It has a $1000 share price (cash binds long exposure), calm markets 5-10x quieter than practice,
    sub-bp to few-bp spreads, wide book imbalance, and an occasional extreme stress regime.
    """
    stress = rng.random() < 0.25
    drift = float(rng.choice([0.0, 1.0, -1.0], p=[0.5, 0.25, 0.25]) * rng.uniform(0.00002, 0.0004))
    if stress:
        vol = float(np.exp(rng.uniform(np.log(0.003), np.log(0.02))))
        spread = float(np.exp(rng.uniform(np.log(5.0), np.log(150.0))))
        shock_prob, shock_scale = float(rng.uniform(0.005, 0.04)), float(rng.uniform(0.02, 0.08))
    else:
        vol = float(np.exp(rng.uniform(np.log(0.00008), np.log(0.0015))))
        spread = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        shock_prob, shock_scale = float(rng.choice([0.0, rng.uniform(0.001, 0.01)], p=[0.6, 0.4])), float(rng.uniform(0.005, 0.02))
    return dict(
        drift=drift, volatility=vol, spread_bps=spread,
        depth=int(rng.uniform(60, 1100)), imbalance_noise=float(rng.uniform(0.3, 0.9)),
        shock_prob=shock_prob, shock_scale=shock_scale,
        fill_prob=float(rng.uniform(0.15, 0.6)), adverse=float(rng.uniform(0.0, 0.5)),
        impact_bps=float(rng.uniform(0.0, 5.0)), open_order_window=int(rng.integers(0, 8)),
        price=float(np.exp(rng.uniform(np.log(5_000.0), np.log(200_000.0)))),
    )


class RandomRegimeEnv(ExtendedMarketEnv):
    """One randomly sampled regime (and optionally episode length) per instance, deterministic in `seed`.

    family: "practice" (default; the round-1/2 sampler), "abides" (sample_abides_regime) or
    "mixed" (50/50 between the two).
    """

    def __init__(self, steps=300, seed=0, random_length: bool = False, family: str = "practice"):
        rng = np.random.default_rng(10_000_019 + int(seed))
        if random_length:
            steps = int(rng.integers(150, 1001))
        if family == "practice":
            cfg = sample_regime(rng)
        elif family == "abides":
            cfg = sample_abides_regime(rng)
        elif family == "mixed":
            cfg = sample_abides_regime(rng) if rng.random() < 0.5 else sample_regime(rng)
        else:
            raise ValueError(f"unknown regime family {family!r}")
        super().__init__(scenario="random", steps=steps, seed=seed, cfg=cfg)


def make_env(scenario: str, steps: int = 300, seed: int = 0):
    """Official scenarios use the untouched base env; stress scenarios use ExtendedMarketEnv."""
    if scenario in PRACTICE_SCENARIOS:
        return PracticeMarketEnv(scenario=scenario, steps=steps, seed=seed)
    return ExtendedMarketEnv(scenario=scenario, steps=steps, seed=seed)
