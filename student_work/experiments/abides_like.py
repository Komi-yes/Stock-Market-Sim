"""Round-3 abides_like: config B (heavy reward, no mask, 60k steps) trained on ABIDES-like regime families.

Flags: --family {practice,abides,mixed}  --steps  --seed  --out
Reuses student_work.experiments.combo.train unchanged; only swaps the env class it builds so that the
RandomRegimeEnv gets family=... (combo.py is not edited).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

torch.set_num_threads(1)

from student_work.envs import RandomRegimeEnv  # noqa: E402
from student_work.experiments import combo  # noqa: E402


def make_env_cls(family: str):
    class FamilyShapedEnv(combo.ShapedRegimeEnv):
        def __init__(self, env_seed: int, params):
            self.env = RandomRegimeEnv(seed=env_seed, random_length=True, family=family)
            self.params = params
            self.steps = self.env.steps

    return FamilyShapedEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["practice", "abides", "mixed"], required=True)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert a.seed < 5000
    combo.ShapedRegimeEnv = make_env_cls(a.family)
    combo.train("heavy", 0, a.steps, a.seed, a.out)


if __name__ == "__main__":
    main()
