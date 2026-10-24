"""Greedy forward selection of ensemble members on a practice seed set (orchestrator-owned).

Each step tries adding every remaining candidate to the current ensemble, evaluates every trial ensemble
(wrapped in SafePolicy with the given guard) on the seed set, and keeps the addition with the best composite
SCORE in a pool of {current, all trials, HOLD, rule-momentum}. It stops when no trial beats the current
ensemble in that pool, or at --max-members.

Usage:
  python -m student_work.select_ensemble --set tuning --start v1 --cand outputs/exp_farm-B_s6/model.pt ... --jobs 8
  (--start v1 = the 7 members in student_work.build_final.MEMBERS; --start none = empty)

ABIDES is too slow for the inner loop; check the shortlisted ensembles with abides_eval afterwards.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from student_work.build_final import GUARD, MEMBERS
from student_work.evaluate import composite, evaluate_policy, hold_policy, rule_policy, summarise
from student_work.model import Ensemble, SafePolicy


def _policy(members: list[str], guard: dict):
    bases = [torch.jit.load(m, map_location="cpu").eval() for m in members]
    model = torch.jit.script(SafePolicy(Ensemble(bases), **guard).eval())

    @torch.inference_mode()
    def act(obs: np.ndarray) -> int:
        return int(model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)).argmax(dim=1).item())
    return act


def _eval_members(args) -> dict:
    members, guard, seed_set = args
    torch.set_num_threads(1)
    return summarise(evaluate_policy(_policy(members, guard), seed_set))


def _eval_fixed(args) -> dict:
    name, seed_set = args
    return summarise(evaluate_policy(hold_policy if name == "HOLD" else rule_policy, seed_set))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="tuning")
    p.add_argument("--start", default="v1", help="'v1' (build_final.MEMBERS), 'none', or comma-separated paths")
    p.add_argument("--cand", nargs="+", required=True, help="candidate member model paths")
    p.add_argument("--max-members", type=int, default=15)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--guard", default=None, help="JSON dict overriding build_final.GUARD")
    p.add_argument("--out", type=Path, required=True, help="JSON log of every step")
    a = p.parse_args()
    guard = {**GUARD, **(json.loads(a.guard) if a.guard else {})}
    current = list(MEMBERS) if a.start == "v1" else ([] if a.start == "none" else a.start.split(","))
    cands = [c for c in a.cand if c not in current]
    log = dict(set=a.set, guard=guard, start=list(current), steps=[])

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        base = dict(zip(["HOLD", "rule-momentum"], ex.map(_eval_fixed, [("HOLD", a.set), ("rule", a.set)])))
        cur_summary = next(ex.map(_eval_members, [(current, guard, a.set)])) if current else None
        while cands and len(current) < a.max_members:
            trials = [current + [c] for c in cands]
            sums = list(ex.map(_eval_members, [(t, guard, a.set) for t in trials]))
            pool = dict(base)
            if cur_summary is not None:
                pool["current"] = cur_summary
            for c, s in zip(cands, sums):
                pool[c] = s
            comp = composite(pool)
            best = max(cands, key=lambda c: comp[c]["score"])
            cur_score = comp["current"]["score"] if cur_summary is not None else -1.0
            step = dict(n=len(current), current_score=cur_score, best=best, best_score=comp[best]["score"],
                        trials={c: dict(score=comp[c]["score"], R=pool[c]["R"], D=pool[c]["D"],
                                        worst=pool[c]["worst_scenario"], prof=pool[c]["profitable_share"])
                                for c in cands})
            log["steps"].append(step)
            print(f"step {len(log['steps'])}: current({len(current)}) {cur_score:.3f} | best +{best} "
                  f"{comp[best]['score']:.3f} R {pool[best]['R']:.3f} DD {pool[best]['D']:.3f} "
                  f"worst {pool[best]['worst_scenario']:.3f}", flush=True)
            if comp[best]["score"] <= cur_score:
                print("no trial beats the current ensemble: stop", flush=True)
                break
            current.append(best)
            cur_summary = pool[best]
            cands.remove(best)
            log["final"] = list(current)
            a.out.write_text(json.dumps(log, indent=1))
    log["final"] = list(current)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(log, indent=1))
    print("final members:", *current, sep="\n  ")


if __name__ == "__main__":
    main()
