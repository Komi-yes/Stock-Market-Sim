"""W1 reward_risk: starter DQN (same algorithm/architecture/scenarios) with a risk-shaped TRAINING reward only.

shaped_r = pnl_change/1000                                   (raw NAV change, dominant term)
           - 0.0005*|inv|                                     (starter linear inventory penalty, kept)
           - inv_quad * (inv/500)^2                           (nonlinear inventory penalty)
           - dd_coef  * max(0, dd_t - dd_{t-1}) / 1000        (increase of running-peak drawdown, cents)
           - trade_cost * [a trade was filled this step]
           - breach_pen * [chosen order would push |inv| past 500 or buy beyond cash]
Only the reward seen by the learner changes; policy evaluation uses the unchanged evaluator.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

from shared.contract import ACTION_SIZE, POSITION_LIMIT, action_to_order  # noqa: E402
from shared.practice_market import PracticeMarketEnv  # noqa: E402
from student_work.model import QNetwork  # noqa: E402

SCENARIOS = ["stable", "trending", "volatile"]

VARIANTS = {
    # name: (dd_coef, inv_quad, trade_cost, breach_pen)
    "starter": (0.0, 0.0, 0.0, 0.0),   # sanity: reproduces the starter reward
    "light":   (0.5, 0.2, 0.02, 0.1),
    "medium":  (1.0, 0.5, 0.05, 0.3),
    "heavy":   (2.0, 1.0, 0.15, 1.0),
}


class RiskShapedEnv:
    """Wraps PracticeMarketEnv; replaces the reward by the shaped one. Observations/dynamics unchanged."""

    def __init__(self, scenario, steps, seed, params):
        self.env = PracticeMarketEnv(scenario=scenario, steps=steps, seed=seed)
        self.dd_coef, self.inv_quad, self.trade_cost, self.breach_pen = params

    def reset(self):
        s = self.env.reset()
        self.peak = self.env.prev_nav
        self.dd = 0.0
        self.last_trades = 0
        return s

    def step(self, action):
        e = self.env
        inv_before = e.inventory
        # Would the chosen order break the position limit? (deterministic from action + inventory)
        order_qty = 0 if action == 0 else (50 if action in (3, 4, 7, 8) else 10)
        signed = 0 if action == 0 else (order_qty if action in (1, 3, 5, 7) else -order_qty)
        breach = action != 0 and abs(inv_before + signed) > POSITION_LIMIT
        s, raw_r, done, info = e.step(action)
        nav = info["nav_cents"]
        # raw pnl change recovered from raw reward: raw_r = pnl/1000 - 0.0005*|inv|
        pnl_term = raw_r + 0.0005 * abs(info["inventory"])
        self.peak = max(self.peak, nav)
        dd = self.peak - nav
        dd_inc = max(0.0, dd - self.dd)
        self.dd = dd
        filled = info["trade_count"] > self.last_trades
        self.last_trades = info["trade_count"]
        inv_n = info["inventory"] / POSITION_LIMIT
        r = (pnl_term - 0.0005 * abs(info["inventory"])
             - self.inv_quad * inv_n * inv_n
             - self.dd_coef * dd_inc / 1000.0
             - self.trade_cost * float(filled)
             - self.breach_pen * float(breach))
        info = dict(info, raw_reward=raw_r, breach=bool(breach), filled=bool(filled))
        return s, float(r), done, info


class Replay:
    def __init__(self, capacity=50_000):
        self.data = deque(maxlen=capacity)

    def add(self, *row):
        self.data.append(row)

    def sample(self, n):
        batch = random.sample(self.data, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)


def train(variant: str, steps: int, seed: int, out: Path):
    params = VARIANTS[variant]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    q = QNetwork()
    target = QNetwork()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    replay = Replay()
    gamma, batch_size = 0.995, 64
    global_step = episode = 0
    nan_seen = False
    t0 = time.time()
    act_counts = np.zeros(ACTION_SIZE, dtype=np.int64)

    while global_step < steps:
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        env = RiskShapedEnv(scenario, 250, seed + episode, params)
        state = env.reset()
        done = False
        ep_r = ep_raw = 0.0
        ep_breach = 0
        max_inv = 0
        ep_acts = np.zeros(ACTION_SIZE, dtype=np.int64)
        while not done and global_step < steps:
            epsilon = max(0.05, 1.0 - global_step / max(1, steps * 0.75))
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    action = int(q(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            replay.add(state, action, reward, next_state, done)
            state = next_state
            ep_r += reward; ep_raw += info["raw_reward"]
            ep_breach += int(info["breach"])
            max_inv = max(max_inv, abs(info["inventory"]))
            ep_acts[action] += 1
            global_step += 1

            if len(replay) >= batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    y = r + gamma * (1.0 - d) * target(ns).max(dim=1).values
                loss = nn.functional.smooth_l1_loss(pred, y)
                if not torch.isfinite(loss):
                    nan_seen = True
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
            if global_step % 250 == 0:
                target.load_state_dict(q.state_dict())
        episode += 1
        act_counts += ep_acts
        print(f"episode={episode:03d} scen={scenario:9s} steps={global_step:6d}/{steps} eps={epsilon:.2f} "
              f"shaped={ep_r:9.2f} raw={ep_raw:9.2f} nav=${info['nav_cents']/100:,.0f} trades={info['trade_count']:3d} "
              f"maxinv={max_inv:3d} breach={ep_breach:3d} hold={ep_acts[0]/max(1,ep_acts.sum()):.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    q.eval()
    for p in q.parameters():
        if not torch.isfinite(p).all():
            nan_seen = True
    torch.jit.script(q.cpu()).save(str(out))
    print(f"NaN/Inf seen: {nan_seen}")
    print(f"Saved TorchScript policy to: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    assert a.seed < 5000
    out = a.out or Path(f"outputs/exp_reward_risk-{a.variant}_s{a.seed}/model.pt")
    train(a.variant, a.steps, a.seed, out)


if __name__ == "__main__":
    main()
