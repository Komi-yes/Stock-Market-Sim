"""W2 regime_rand: starter DQN (unchanged algorithm / architecture / reward) trained on randomised market regimes.

Variants (--variant):
  a_random : every episode is a RandomRegimeEnv(random_length=True)
  b_mix    : even episodes = official scenario (stable/trending/volatile cycle, 250 steps, as starter),
             odd episodes = RandomRegimeEnv(random_length=True)
  c_curric : random regimes whose parameters are interpolated from a narrow calm regime to the full
             sample_regime ranges (width w goes 0.25 -> 1.0 over the first 60% of steps)

Only the training environment distribution differs from student_work/train.py.
Env seeds are seed*1000 + episode (< 5000 for seeds 1-3).
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.contract import ACTION_SIZE
from shared.practice_market import PracticeMarketEnv
from student_work.envs import ExtendedMarketEnv, RandomRegimeEnv, sample_regime
from student_work.model import QNetwork

SCENARIOS = ["stable", "trending", "volatile"]

# narrow "calm" anchor for the curriculum
_NARROW = dict(drift=0.0, volatility=0.001, spread_bps=5.0, depth=800, shock_prob=0.0, shock_scale=0.0,
               fill_prob=0.55, adverse=0.0, impact_bps=0.0, open_order_window=0)


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


def curriculum_env(env_seed: int, width: float):
    rng = np.random.default_rng(10_000_019 + int(env_seed))
    steps = int(rng.integers(150, 1001))
    full = sample_regime(rng)
    cfg = {}
    for k, v in full.items():
        cfg[k] = _NARROW[k] + width * (v - _NARROW[k])
    cfg["depth"] = int(cfg["depth"])
    cfg["open_order_window"] = int(round(cfg["open_order_window"]))
    return ExtendedMarketEnv(scenario="random", steps=steps, seed=env_seed, cfg=cfg), "curric"


def make_train_env(variant: str, seed: int, episode: int, global_step: int, total: int):
    env_seed = seed * 1000 + episode
    if variant == "a_random":
        return RandomRegimeEnv(seed=env_seed, random_length=True), "random"
    if variant == "b_mix":
        if episode % 2 == 0:
            sc = SCENARIOS[(episode // 2) % len(SCENARIOS)]
            return PracticeMarketEnv(scenario=sc, steps=250, seed=env_seed), sc
        return RandomRegimeEnv(seed=env_seed, random_length=True), "random"
    if variant == "c_curric":
        width = min(1.0, 0.25 + 0.75 * global_step / max(1, 0.6 * total))
        return curriculum_env(env_seed, width)
    raise ValueError(variant)


def train(variant: str, steps: int, seed: int, out: Path):
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    q = QNetwork()
    target = QNetwork()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    replay = Replay()
    gamma, batch_size = 0.995, 64
    global_step = 0
    episode = 0

    while global_step < steps:
        env, tag = make_train_env(variant, seed, episode, global_step, steps)
        state = env.reset()
        done = False
        ep_reward = 0.0
        info = {"nav_cents": 0.0}
        while not done and global_step < steps:
            epsilon = max(0.05, 1.0 - global_step / max(1, steps * 0.75))
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = int(q(x).argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            replay.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            global_step += 1

            if len(replay) >= batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nxt = target(ns).max(dim=1).values
                    y = r + gamma * (1.0 - d) * nxt
                loss = nn.functional.smooth_l1_loss(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {global_step}")
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

            if global_step % 250 == 0:
                target.load_state_dict(q.state_dict())
        episode += 1
        print(f"episode={episode:03d} env={tag:9s} len={env.steps:4d} steps={global_step:6d}/{steps} "
              f"reward={ep_reward:9.3f} nav=${info['nav_cents']/100:,.2f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    q.eval()
    for p in q.parameters():
        assert torch.isfinite(p).all(), "non-finite weights"
    torch.jit.script(q.cpu()).save(str(out))
    print(f"Saved TorchScript policy to: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Starter DQN trained on randomised regimes.")
    p.add_argument("--variant", choices=["a_random", "b_mix", "c_curric"], required=True)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or Path(f"outputs/exp_regime_rand-{args.variant}_s{args.seed}/model.pt")
    train(args.variant, args.steps, args.seed, out)


if __name__ == "__main__":
    main()
