"""W4: clean PPO actor-critic on the practice market (official 3 training scenarios, original reward).

Run (repo root):
  .venv\\Scripts\\python.exe -W ignore -m student_work.experiments.ppo --name main --seed 1 --steps 150000
Outputs: outputs/exp_ppo-<name>_s<seed>/model.pt (+ train_log.csv). Exported module = actor logits only.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.contract import ACTION_SIZE, OBSERVATION_SIZE
from shared.practice_market import PracticeMarketEnv

torch.set_num_threads(1)

SCENARIOS = ["stable", "trending", "volatile"]


class Actor(nn.Module):
    """Exported policy: float32 [B,16] -> logits [B,9]. Plain Linear/Tanh only."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RunningStd:
    """Running std of discounted returns, used to scale rewards (no mean shift)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 1e-4, 0.0, 0.0

    def update(self, x: np.ndarray):
        for v in x.ravel():
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    @property
    def std(self) -> float:
        return float(max(1e-3, (self.m2 / self.n) ** 0.5))


def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out_dir = Path("outputs") / f"exp_ppo-{a.name}_s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    actor, critic = Actor(a.hidden), Critic(a.hidden)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=a.lr, eps=1e-5)
    ret_rms = RunningStd()

    env_id = 0

    def new_env():
        nonlocal env_id
        sc = SCENARIOS[env_id % 3]
        env = PracticeMarketEnv(scenario=sc, steps=a.ep_len, seed=a.seed * 1000 + env_id)  # < 5000
        env_id += 1
        return env, env.reset()

    env, obs = new_env()
    ep_ret, ep_navs, finished = 0.0, [], []  # finished: (scenario, return, nav)
    ep_disc_ret = 0.0
    n_updates = a.steps // a.rollout
    log = []
    step_count = 0
    for upd in range(n_updates):
        frac = upd / max(1, n_updates - 1)
        ent_coef = a.ent_start + (a.ent_end - a.ent_start) * frac
        for g in opt.param_groups:
            g["lr"] = a.lr * (1.0 - 0.9 * frac)
        B_obs = np.zeros((a.rollout, OBSERVATION_SIZE), np.float32)
        B_act = np.zeros(a.rollout, np.int64)
        B_logp = np.zeros(a.rollout, np.float32)
        B_val = np.zeros(a.rollout, np.float32)
        B_rew = np.zeros(a.rollout, np.float32)
        B_done = np.zeros(a.rollout, np.float32)
        B_nextval = np.zeros(a.rollout, np.float32)  # value of s' for non-terminal steps (unused when done)
        ep_rets, ep_scn = [], []
        for t in range(a.rollout):
            x = torch.from_numpy(np.asarray(obs, np.float32)).unsqueeze(0)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=actor(x))
                act = dist.sample()
                logp = dist.log_prob(act)
                val = critic(x)
            nobs, r, done, info = env.step(int(act.item()))
            B_obs[t] = obs; B_act[t] = act.item(); B_logp[t] = logp.item(); B_val[t] = val.item()
            B_rew[t] = r; B_done[t] = float(done)
            ep_ret += r
            ep_disc_ret = ep_disc_ret * a.gamma + r
            ret_rms.update(np.array([ep_disc_ret]))
            obs = nobs
            step_count += 1
            if done:
                ep_rets.append(ep_ret); ep_scn.append(env.scenario)
                ep_ret, ep_disc_ret = 0.0, 0.0
                env, obs = new_env()
        with torch.no_grad():
            last_val = critic(torch.from_numpy(np.asarray(obs, np.float32)).unsqueeze(0)).item()
        rew = B_rew / ret_rms.std
        adv = np.zeros(a.rollout, np.float32)
        lastgae = 0.0
        for t in reversed(range(a.rollout)):
            nv = last_val if t == a.rollout - 1 else B_val[t + 1]
            nonterm = 1.0 - B_done[t]
            delta = rew[t] + a.gamma * nv * nonterm - B_val[t]
            lastgae = delta + a.gamma * a.lam * nonterm * lastgae
            adv[t] = lastgae
        ret = adv + B_val
        o = torch.from_numpy(B_obs); ac = torch.from_numpy(B_act); lp0 = torch.from_numpy(B_logp)
        advt = torch.from_numpy(adv); rett = torch.from_numpy(ret)
        for _ in range(a.epochs):
            perm = torch.randperm(a.rollout)
            for i in range(0, a.rollout, a.minibatch):
                idx = perm[i:i + a.minibatch]
                dist = torch.distributions.Categorical(logits=actor(o[idx]))
                lp = dist.log_prob(ac[idx])
                ratio = (lp - lp0[idx]).exp()
                ad = advt[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean()
                vl = 0.5 * (critic(o[idx]) - rett[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pl + a.vf_coef * vl - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
        with torch.no_grad():
            lg = actor(o)
            d = torch.distributions.Categorical(logits=lg)
            ent_now = d.entropy().mean().item()
            greedy = np.bincount(lg.argmax(1).numpy(), minlength=ACTION_SIZE) / a.rollout
            finite = bool(torch.isfinite(lg).all())
        row = dict(update=upd + 1, steps=step_count, mean_ep_return=float(np.mean(ep_rets)) if ep_rets else float("nan"),
                   entropy=ent_now, hold_greedy=float(greedy[0]), greedy_max=float(greedy.max()),
                   greedy_share=" ".join(f"{g:.2f}" for g in greedy), finite=finite)
        for sc in SCENARIOS:
            v = [r for r, s in zip(ep_rets, ep_scn) if s == sc]
            row[f"ret_{sc}"] = float(np.mean(v)) if v else float("nan")
        log.append(row)
        if (upd + 1) % 5 == 0 or upd == 0 or upd == n_updates - 1:
            print(f"upd={upd+1:3d}/{n_updates} steps={step_count} ret={row['mean_ep_return']:8.3f} "
                  f"(st {row['ret_stable']:.2f} tr {row['ret_trending']:.2f} vo {row['ret_volatile']:.2f}) "
                  f"ent={ent_now:.3f} greedy=[{row['greedy_share']}] finite={finite}", flush=True)
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    actor.eval()
    scripted = torch.jit.script(actor)
    scripted.save(str(out_dir / "model.pt"))
    print(f"Saved actor TorchScript to {out_dir / 'model.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--ep-len", type=int, default=250)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-start", type=float, default=0.01)
    p.add_argument("--ent-end", type=float, default=0.002)
    train(p.parse_args())


if __name__ == "__main__":
    main()
