"""W3 experiment "dqn_plus": Double DQN + dueling head + n-step returns + LayerNorm MLP + lr decay + soft target.

Variants (see VARIANTS):
  full : 128x128 LayerNorm MLP, dueling, Double DQN, 3-step returns, lr 1e-3 -> 1e-4 linear, soft target (tau=0.005)
  dd   : ablation. 64x64 plain MLP, dueling, Double DQN, 1-step, constant lr 1e-3, periodic target sync (250), like the starter.

Everything else (replay 50k, batch 64, gamma 0.995, eps 1.0->0.05 over 75% steps, smooth-L1, clip 5, 250-step
episodes cycling stable/trending/volatile, env seed = seed + episode, original reward) is the starter's.

Run:  .venv\\Scripts\\python.exe -W ignore -m student_work.experiments.dqn_plus --variant full --seed 1 --steps 60000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.contract import OBSERVATION_SIZE, ACTION_SIZE
from shared.practice_market import PracticeMarketEnv

SCENARIOS = ["stable", "trending", "volatile"]

VARIANTS = {
    "full": dict(hidden=128, layernorm=True, dueling=True, double=True, nstep=3,
                 lr0=1e-3, lr1=1e-4, tau=0.005, sync=0),
    "dd": dict(hidden=64, layernorm=False, dueling=True, double=True, nstep=1,
               lr0=1e-3, lr1=1e-3, tau=0.0, sync=250),
}


class DuelingQNet(nn.Module):
    """Trunk (Linear[-LayerNorm]-ReLU x2) + value / advantage streams. Plain layers only; deterministic."""

    def __init__(self, hidden: int = 128, layernorm: bool = True, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        layers = [nn.Linear(OBSERVATION_SIZE, hidden)]
        if layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, hidden))
        if layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.adv = nn.Linear(hidden, ACTION_SIZE)
        self.val = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        a = self.adv(h)
        if self.dueling:
            v = self.val(h)
            return v + a - a.mean(dim=1, keepdim=True)
        return a


class Replay:
    def __init__(self, capacity=50_000):
        self.data = deque(maxlen=capacity)

    def add(self, *row):
        self.data.append(row)

    def sample(self, n):
        batch = random.sample(self.data, n)
        s, a, r, ns, d, g = zip(*batch)
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            torch.tensor(g, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)


def flush_nstep(buf, replay, gamma, final_next, final_done, all_remaining):
    """Emit n-step transitions from buf (deque of (s,a,r)). If all_remaining, emit every pending one
    (episode ended), otherwise emit only the oldest."""
    while buf:
        R, g = 0.0, 1.0
        for (_, _, r) in buf:
            R += g * r
            g *= gamma
        s0, a0, _ = buf[0]
        replay.add(s0, a0, R, final_next, float(final_done), g)
        buf.popleft()
        if not all_remaining:
            break


def train(variant: str, steps: int, seed: int, out: Path):
    torch.set_num_threads(1)
    cfg = VARIANTS[variant]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    q = DuelingQNet(cfg["hidden"], cfg["layernorm"], cfg["dueling"])
    target = DuelingQNet(cfg["hidden"], cfg["layernorm"], cfg["dueling"])
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg["lr0"])
    replay = Replay()
    gamma, batch_size, n = 0.995, 64, cfg["nstep"]
    global_step = episode = 0
    t0 = time.time()
    log = []
    losses, qmaxs = [], []

    while global_step < steps:
        scenario = SCENARIOS[episode % len(SCENARIOS)]
        env = PracticeMarketEnv(scenario=scenario, steps=250, seed=seed + episode)
        state = env.reset()
        done = False
        ep_reward = 0.0
        buf = deque()
        while not done and global_step < steps:
            epsilon = max(0.05, 1.0 - global_step / max(1, steps * 0.75))
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = int(q(x).argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            buf.append((state, action, reward))
            if done:
                flush_nstep(buf, replay, gamma, next_state, True, True)
            elif len(buf) == n:
                flush_nstep(buf, replay, gamma, next_state, False, False)
            state = next_state
            ep_reward += reward
            global_step += 1

            # lr schedule (linear)
            frac = min(1.0, global_step / steps)
            lr = cfg["lr0"] + (cfg["lr1"] - cfg["lr0"]) * frac
            for g_ in opt.param_groups:
                g_["lr"] = lr

            if len(replay) >= batch_size:
                s, a, r, ns, d, gn = replay.sample(batch_size)
                qs = q(s)
                pred = qs.gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if cfg["double"]:
                        a_star = q(ns).argmax(dim=1, keepdim=True)
                        nxt = target(ns).gather(1, a_star).squeeze(1)
                    else:
                        nxt = target(ns).max(dim=1).values
                    y = r + gn * (1.0 - d) * nxt
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.item()))
                qmaxs.append(float(qs.max(dim=1).values.mean().item()))
                if not np.isfinite(losses[-1]):
                    raise RuntimeError("non-finite loss")

            if cfg["tau"] > 0:
                with torch.no_grad():
                    for tp, p in zip(target.parameters(), q.parameters()):
                        tp.mul_(1 - cfg["tau"]).add_(p, alpha=cfg["tau"])
            elif global_step % cfg["sync"] == 0:
                target.load_state_dict(q.state_dict())
        episode += 1
        ml = float(np.mean(losses)) if losses else float("nan")
        mq = float(np.mean(qmaxs)) if qmaxs else float("nan")
        log.append(dict(episode=episode, scenario=scenario, step=global_step, reward=ep_reward,
                        nav=info["nav_cents"] / 100, loss=ml, qmax=mq, eps=epsilon, lr=lr))
        losses, qmaxs = [], []
        print(f"episode={episode:03d} {scenario:9s} steps={global_step:6d}/{steps} reward={ep_reward:8.3f} "
              f"nav=${info['nav_cents']/100:,.2f} loss={ml:.5f} qmax={mq:.3f} eps={epsilon:.2f} lr={lr:.1e} "
              f"t={time.time()-t0:.0f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    q.eval()
    scripted = torch.jit.script(q.cpu())
    # sanity: finite output, right shape
    with torch.no_grad():
        o = scripted(torch.zeros(4, OBSERVATION_SIZE))
        assert o.shape == (4, ACTION_SIZE) and torch.isfinite(o).all()
    scripted.save(str(out))
    (out.parent / "train_log.json").write_text(json.dumps(dict(variant=variant, cfg=cfg, seed=seed,
                                                                steps=steps, log=log)))
    print(f"Saved TorchScript policy to: {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or Path(f"outputs/exp_dqn_plus-{a.variant}_s{a.seed}/model.pt")
    train(a.variant, a.steps, a.seed, out)


if __name__ == "__main__":
    main()
