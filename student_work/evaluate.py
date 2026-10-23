"""Fixed evaluation protocol for every experiment (orchestrator-owned; workers import, never edit).

Usage:
  python -m student_work.evaluate --set tuning --model mine=outputs/exp_x_s1/model.pt [--model ...] --out outputs/exp_x_eval
  python -m student_work.evaluate --compare a.json b.json ...      # re-pool saved results without re-running

Scoring mirrors the lecturer's rubric. Each component is rank-normalised inside the comparison pool, which always
contains the HOLD and rule-momentum baselines:
  50% R  mean return % over every episode (scenario x length x seed)
  20% D  mean max drawdown % (lower is better)
  20% C  robustness: mean of rank(worst scenario mean return) and rank(share of profitable episodes)
  10% O  operational: invalid-order attempts per 100 steps + share of steps with |inv| > 400 + 10 x insolvent share
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from shared.contract import LARGE_ORDER, OBSERVATION_SIZE, ACTION_SIZE, POSITION_LIMIT, SMALL_ORDER, STARTING_CASH_CENTS
from student_work.envs import make_env

SCENARIOS = ["stable", "trending", "volatile", "crash", "downtrend", "low_liquidity", "choppy"]
LENGTHS = [300, 1000]
SEED_SETS = {
    "tuning": list(range(10_000, 10_010)),       # workers tune against these
    "validation": list(range(20_000, 20_010)),   # orchestrator only, between rounds
    "holdout": list(range(30_000, 30_020)),      # used once, for the v1 final pick (spent)
    "holdout2": list(range(40_000, 40_020)),     # used once, for the stage-2 final gate (v2 vs v1)
    "smoke": [10_000, 10_001],                   # quick check while developing
}
WEIGHTS = dict(R=0.5, D=0.2, C=0.2, O=0.1)
_SIGNED_QTY = [0, SMALL_ORDER, -SMALL_ORDER, LARGE_ORDER, -LARGE_ORDER,
               SMALL_ORDER, -SMALL_ORDER, LARGE_ORDER, -LARGE_ORDER]

Policy = Callable[[np.ndarray], int]


def hold_policy(obs: np.ndarray) -> int:
    return 0


def rule_policy(obs: np.ndarray) -> int:
    """The momentum rule from examples/demo_02."""
    momentum_5, inventory = float(obs[1]), float(obs[8])
    if momentum_5 > 0.015 and inventory < 0.8:
        return 1
    if momentum_5 < -0.015 and inventory > -0.8:
        return 2
    return 0


@torch.inference_mode()  # same mode as shared/test_policy; mixing grad and inference calls trips the JIT executor
def check_contract(model) -> None:
    """Raise if the TorchScript model breaks the tournament contract."""
    if not isinstance(model, torch.jit.ScriptModule):
        raise TypeError("model is not TorchScript")
    for x in (torch.zeros(1, OBSERVATION_SIZE), torch.rand(4, OBSERVATION_SIZE) * 2 - 1):
        y = model(x.float())
        if y.dtype != torch.float32 or tuple(y.shape) != (x.shape[0], ACTION_SIZE):
            raise ValueError(f"bad output {y.dtype} {tuple(y.shape)}")
        if not torch.isfinite(y).all():
            raise ValueError("non-finite output")


def torchscript_policy(path: str | Path) -> Policy:
    model = torch.jit.load(str(path), map_location="cpu").eval()
    check_contract(model)

    @torch.inference_mode()
    def act(obs: np.ndarray) -> int:
        return int(model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)).argmax(dim=1).item())
    return act


def run_episode(policy: Policy, scenario: str, steps: int, seed: int) -> dict:
    env = make_env(scenario, steps=steps, seed=seed)
    obs = env.reset()
    navs = [float(STARTING_CASH_CENTS)]
    invalid = exposed = 0
    actions = np.zeros(ACTION_SIZE, dtype=np.int64)
    done, info, n = False, {}, 0
    while not done:
        a = policy(obs)
        actions[a] += 1
        signed = _SIGNED_QTY[a]
        if signed:  # attempt that the market would reject (reads env state only; no RNG use)
            if abs(env.inventory + signed) > POSITION_LIMIT or (signed > 0 and env.cash < signed * env.mid):
                invalid += 1
        obs, _, done, info = env.step(a)
        n += 1
        navs.append(info["nav_cents"])
        if abs(info["inventory"]) > 0.8 * POSITION_LIMIT:
            exposed += 1
    nav = np.asarray(navs)
    peak = np.maximum.accumulate(nav)
    return dict(
        scenario=scenario, steps=steps, seed=seed,
        return_pct=float((nav[-1] / STARTING_CASH_CENTS - 1.0) * 100.0),
        max_dd_pct=float(((peak - nav) / peak).max() * 100.0),
        trades=int(info["trade_count"]), final_inventory=int(info["inventory"]),
        invalid=int(invalid), exposed_frac=float(exposed / n),
        insolvent=bool(nav.min() <= 0), n_steps=n,
        action_counts=actions.tolist(),
    )


def evaluate_policy(policy: Policy, seed_set: str) -> list[dict]:
    return [run_episode(policy, sc, L, s) for sc in SCENARIOS for L in LENGTHS for s in SEED_SETS[seed_set]]


def summarise(episodes: list[dict]) -> dict:
    r = np.array([e["return_pct"] for e in episodes])
    by_scen = {}
    for sc in sorted({e["scenario"] for e in episodes}, key=SCENARIOS.index):
        es = [e for e in episodes if e["scenario"] == sc]
        by_scen[sc] = dict(
            return_pct=float(np.mean([e["return_pct"] for e in es])),
            return_std=float(np.std([e["return_pct"] for e in es])),
            max_dd_pct=float(np.mean([e["max_dd_pct"] for e in es])),
            trades=float(np.mean([e["trades"] for e in es])),
            abs_final_inv=float(np.mean([abs(e["final_inventory"]) for e in es])),
        )
    steps = sum(e["n_steps"] for e in episodes)
    actions = np.sum([e["action_counts"] for e in episodes], axis=0)
    return dict(
        R=float(r.mean()),
        D=float(np.mean([e["max_dd_pct"] for e in episodes])),
        worst_scenario=float(min(v["return_pct"] for v in by_scen.values())),
        profitable_share=float((r > 0).mean()),
        O=float(100.0 * sum(e["invalid"] for e in episodes) / steps
                + np.mean([e["exposed_frac"] for e in episodes])
                + 10.0 * np.mean([e["insolvent"] for e in episodes])),
        return_std=float(r.std()),
        trades=float(np.mean([e["trades"] for e in episodes])),
        hold_share=float(actions[0] / max(1, actions.sum())),
        action_share=(actions / max(1, actions.sum())).round(3).tolist(),
        by_scenario=by_scen,
    )


def _rank01(values: list[float], higher_better: bool) -> list[float]:
    """Average-rank normalisation to [0, 1]; 1 is best. Ties share the same score."""
    n = len(values)
    if n == 1:
        return [1.0]
    v = np.asarray(values) * (1 if higher_better else -1)
    order = v.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(n, dtype=float)
    for val in np.unique(v):  # average ties
        ranks[v == val] = ranks[v == val].mean()
    return (ranks / (n - 1)).tolist()


def composite(pool: dict[str, dict]) -> dict[str, dict]:
    names = list(pool)
    s = [pool[n] for n in names]
    rR = _rank01([x["R"] for x in s], True)
    rD = _rank01([x["D"] for x in s], False)
    rC = [(a + b) / 2 for a, b in zip(_rank01([x["worst_scenario"] for x in s], True),
                                      _rank01([x["profitable_share"] for x in s], True))]
    rO = _rank01([x["O"] for x in s], False)
    return {n: dict(score=WEIGHTS["R"] * a + WEIGHTS["D"] * b + WEIGHTS["C"] * c + WEIGHTS["O"] * d,
                    rR=a, rD=b, rC=c, rO=d)
            for n, a, b, c, d in zip(names, rR, rD, rC, rO)}


def print_pool(pool: dict[str, dict]) -> None:
    comp = composite(pool)
    print(f"\n{'name':28s} {'SCORE':>6s} {'R%':>7s} {'DD%':>6s} {'worst%':>7s} {'prof':>5s} "
          f"{'O':>6s} {'trades':>7s} {'hold':>5s}")
    for n in sorted(pool, key=lambda k: -comp[k]["score"]):
        x = pool[n]
        print(f"{n:28s} {comp[n]['score']:6.3f} {x['R']:7.3f} {x['D']:6.3f} {x['worst_scenario']:7.3f} "
              f"{x['profitable_share']:5.2f} {x['O']:6.3f} {x['trades']:7.1f} {x['hold_share']:5.2f}")
    print("\nPer-scenario mean return % (all lengths and seeds):")
    print(f"{'name':28s} " + " ".join(f"{sc[:9]:>9s}" for sc in SCENARIOS))
    for n in sorted(pool, key=lambda k: -comp[k]["score"]):
        bs = pool[n]["by_scenario"]
        print(f"{n:28s} " + " ".join(f"{bs[sc]['return_pct']:9.3f}" if sc in bs else f"{'-':>9s}" for sc in SCENARIOS))


def main():
    p = argparse.ArgumentParser(description="Evaluate policies with the fixed protocol and rubric-like score.")
    p.add_argument("--set", choices=list(SEED_SETS), default="tuning")
    p.add_argument("--model", action="append", default=[], help="name=path/to/model.pt (repeatable)")
    p.add_argument("--out", type=Path, default=None, help="directory for <name>_<set>.json results")
    p.add_argument("--compare", nargs="*", type=Path, default=None, help="pool saved JSON results instead")
    a = p.parse_args()
    torch.set_num_threads(1)

    pool: dict[str, dict] = {}
    if a.compare:
        sets = set()
        for f in a.compare:
            d = json.loads(f.read_text())
            sets.add(d["set"])
            pool[d["name"]] = d["summary"]
        if len(sets) > 1:
            raise SystemExit(f"Refusing to pool results from different seed sets: {sets}")
    else:
        policies: dict[str, Policy] = {"HOLD": hold_policy, "rule-momentum": rule_policy}
        for spec in a.model:
            name, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--model needs name=path, got {spec!r}")
            policies[name] = torchscript_policy(path)
        for name, pol in policies.items():
            episodes = evaluate_policy(pol, a.set)
            pool[name] = summarise(episodes)
            if a.out is not None:
                a.out.mkdir(parents=True, exist_ok=True)
                (a.out / f"{name}_{a.set}.json").write_text(json.dumps(
                    dict(name=name, set=a.set, summary=pool[name], episodes=episodes), indent=1))
            print(f"evaluated {name} on {a.set} ({len(episodes)} episodes)", flush=True)
    print_pool(pool)
    comp = composite(pool)
    if "HOLD" in pool:
        losers = [n for n in pool if n != "HOLD" and comp[n]["score"] <= comp["HOLD"]["score"]]
        if losers:
            print(f"\nDoes not beat HOLD on the composite (reject): {', '.join(losers)}")


if __name__ == "__main__":
    main()
