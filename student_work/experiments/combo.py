"""Round-2 combo: starter DQN + randomised regimes (a_random) + risk-shaped reward + training-time action mask.

Flags: --reward {none,medium,heavy}  --mask-cap N (0 = off)  --steps  --seed  --out

Training env: always RandomRegimeEnv(seed=seed*1000+episode, random_length=True) (from regime_rand's a_random).
Reward: reward_risk.VARIANTS coefficients (dd_coef, inv_quad, trade_cost, breach_pen) applied on top of the
regime env (same formula as reward_risk.RiskShapedEnv, which is hard-wired to PracticeMarketEnv).
Mask (cap > 0): the SAME rule as student_work.model.SafePolicy: an action is invalid if it would make
|inv+signed| > cap while increasing |inv|, or it is a buy with cash < 10% of starting cash. HOLD is always valid.
Used for greedy action, random exploration, and the max over next-state actions in the TD target.
The exported model is the raw QNetwork (torch.jit.script); compose it with SafePolicy separately.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

from shared.contract import ACTION_SIZE, POSITION_LIMIT  # noqa: E402
from student_work.envs import RandomRegimeEnv  # noqa: E402
from student_work.experiments.regime_rand import Replay  # noqa: E402
from student_work.experiments.reward_risk import VARIANTS  # noqa: E402
from student_work.model import QNetwork  # noqa: E402

SIGNED = torch.tensor([[0.0, 10.0, -10.0, 50.0, -50.0, 10.0, -10.0, 50.0, -50.0]])
SIGNED_LIST = SIGNED[0].tolist()


def valid_mask(obs: torch.Tensor, cap: float, min_cash_ratio: float = 0.1) -> torch.Tensor:
    """obs [B,16] -> bool [B,9], True = action allowed (mirrors SafePolicy)."""
    inv = obs[:, 8:9] * 500.0
    new_inv = inv + SIGNED
    too_big = (new_inv.abs() > cap) & (new_inv.abs() > inv.abs())
    no_cash = (obs[:, 9:10] * 2.0 < min_cash_ratio) & (SIGNED > 0.0)
    return ~(too_big | no_cash)


class ShapedRegimeEnv:
    """RandomRegimeEnv with reward_risk-style shaped reward (reward 'none' keeps the raw starter reward)."""

    def __init__(self, env_seed: int, params):
        self.env = RandomRegimeEnv(seed=env_seed, random_length=True)
        self.params = params
        self.steps = self.env.steps

    def reset(self):
        s = self.env.reset()
        self.peak = self.env.prev_nav
        self.dd = 0.0
        self.last_trades = 0
        return s

    def step(self, action):
        e = self.env
        dd_coef, inv_quad, trade_cost, breach_pen = self.params
        signed = SIGNED_LIST[action]
        breach = action != 0 and abs(e.inventory + signed) > POSITION_LIMIT
        s, raw_r, done, info = e.step(action)
        if self.params == (0.0, 0.0, 0.0, 0.0):
            return s, float(raw_r), done, dict(info, raw_reward=raw_r, breach=bool(breach))
        nav = info["nav_cents"]
        pnl_term = raw_r + 0.0005 * abs(info["inventory"])
        self.peak = max(self.peak, nav)
        dd = self.peak - nav
        dd_inc = max(0.0, dd - self.dd)
        self.dd = dd
        filled = info["trade_count"] > self.last_trades
        self.last_trades = info["trade_count"]
        inv_n = info["inventory"] / POSITION_LIMIT
        r = (pnl_term - 0.0005 * abs(info["inventory"]) - inv_quad * inv_n * inv_n
             - dd_coef * dd_inc / 1000.0 - trade_cost * float(filled) - breach_pen * float(breach))
        return s, float(r), done, dict(info, raw_reward=raw_r, breach=bool(breach))


def train(reward: str, mask_cap: float, steps: int, seed: int, out: Path):
    params = VARIANTS[reward] if reward != "none" else (0.0, 0.0, 0.0, 0.0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    q = QNetwork(); target = QNetwork()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    replay = Replay()
    gamma, batch_size = 0.995, 64
    global_step = episode = 0
    nan_seen = False
    t0 = time.time()

    while global_step < steps:
        env = ShapedRegimeEnv(seed * 1000 + episode, params)
        state = env.reset()
        done = False
        ep_r = ep_raw = 0.0
        ep_breach = max_inv = 0
        ep_hold = ep_n = 0
        info = {"nav_cents": 0.0, "trade_count": 0}
        while not done and global_step < steps:
            epsilon = max(0.05, 1.0 - global_step / max(1, steps * 0.75))
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            vm = valid_mask(x, mask_cap)[0] if mask_cap > 0 else None
            if random.random() < epsilon:
                if vm is None:
                    action = random.randrange(ACTION_SIZE)
                else:
                    action = random.choice(torch.nonzero(vm).flatten().tolist())
            else:
                with torch.no_grad():
                    qv = q(x)
                    if vm is not None:
                        qv = torch.where(vm.unsqueeze(0), qv, torch.full_like(qv, -1e9))
                    action = int(qv.argmax(dim=1).item())
            next_state, reward_v, done, info = env.step(action)
            replay.add(state, action, reward_v, next_state, done)
            state = next_state
            ep_r += reward_v; ep_raw += info["raw_reward"]
            ep_breach += int(info["breach"])
            max_inv = max(max_inv, abs(info["inventory"]))
            ep_hold += int(action == 0); ep_n += 1
            global_step += 1

            if len(replay) >= batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    tq = target(ns)
                    if mask_cap > 0:
                        tq = torch.where(valid_mask(ns, mask_cap), tq, torch.full_like(tq, -1e9))
                    y = r + gamma * (1.0 - d) * tq.max(dim=1).values
                loss = nn.functional.smooth_l1_loss(pred, y)
                if not torch.isfinite(loss):
                    nan_seen = True
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
            if global_step % 250 == 0:
                target.load_state_dict(q.state_dict())
        episode += 1
        print(f"episode={episode:03d} len={env.steps:4d} steps={global_step:6d}/{steps} eps={epsilon:.2f} "
              f"shaped={ep_r:9.2f} raw={ep_raw:9.2f} nav=${info['nav_cents']/100:,.0f} trades={info['trade_count']:3d} "
              f"maxinv={max_inv:3d} breach={ep_breach:3d} hold={ep_hold/max(1,ep_n):.2f} [{time.time()-t0:.0f}s]",
              flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    q.eval()
    for p in q.parameters():
        if not torch.isfinite(p).all():
            nan_seen = True
    torch.jit.script(q.cpu()).save(str(out))
    print(f"NaN/Inf seen: {nan_seen}")
    print(f"Saved TorchScript policy to: {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=["none", "medium", "heavy"], default="medium")
    p.add_argument("--mask-cap", type=float, default=0)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert a.seed < 5000
    train(a.reward, a.mask_cap, a.steps, a.seed, a.out)


if __name__ == "__main__":
    main()
