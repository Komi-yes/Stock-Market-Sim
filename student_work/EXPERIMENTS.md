# Experiment Record

Running log of the whole process, kept as the workshop guide requires (section 17).
Newest entries go at the bottom of each section. Never delete failed experiments: failures are evidence.

## Rules for this file
- Orchestrator (Opus 5.5) owns this file. Sonnet 5.5 workers send results in their reports.
- Every result must say which seeds were used and whether they were **tuning seeds** or **held-out seeds**.
- Held-out seeds are never used for tuning or for choosing between variants until the final check.

## 1. Setup log
| Date | What | Notes |
|---|---|---|
| 2026-10-16 | `.venv` rebuilt with Python 3.12 | The old `.venv` was Python 3.11; ABIDES-NG needs 3.12+ |
| 2026-10-16 | `requirements.txt` now installs ABIDES-NG from GitHub | `abides-ng` is not on PyPI |
| 2026-10-16 | ABIDES smoke test (`stable_day`, seed 42) runs | torch 2.14.1+cpu, numpy 2.5.4 |

## 2. Evaluation protocol
Fill in once and keep fixed so results stay comparable.
- Scenarios: stable, trending, volatile (practice market); ABIDES templates for spot checks
- Tuning seeds: _TBD_
- Held-out seeds: _TBD_ (not used until the final model check)
- Episode length: _TBD_
- Metrics per scenario and seed: final NAV / PnL, max drawdown, trade count, final inventory

## 3. Baselines
| Name | Description | Scenario | Seeds | PnL | Max DD | Trades | Final inv |
|---|---|---|---|---|---|---|---|
| rule-momentum | `examples/demo_02` 5-step momentum | | | | | | |
| starter-dqn | Starter DQN, 5,000 steps, seed 7 | | | | | | |

## 4. Experiments
Copy this template for each experiment.

### EXP-000: template
- **Date / worker:**
- **Hypothesis / question:**
- **Change** (model / learning / reward / training regime / exploration / state):
- **Architecture:**
- **Algorithm and key hyperparameters:**
- **Reward function:**
- **Training scenarios and seeds:**
- **Training diagnostics** (loss, reward curve, instability, NaN/Inf):
- **Results by scenario and seed:**
- **Compared with baseline:**
- **Conclusion** (keep / drop / combine):
- **Limitation or failure mode observed:**
- **Files / output path:** `outputs/exp_<name>/`

## 5. Combinations
Which winning ideas were combined and how the combination performed against the parts.

## 6. Final model selection
- **Chosen model:**
- **Reason for selection** (generalisation across scenarios and held-out seeds, not single-seed profit):
- **Held-out results:**
- **ABIDES-NG check:**
- **Validator output** (`VALID` on the exact ZIP):
- **Known limitation / failure mode:**

## 7. Answers to the guide's analysis questions (optional, for coursework)
