# Stock-Market-Sim: MARL Stock Exchange Workshop

Goal: train a trading policy (RL) in a simulated limit-order market and export it as `TEAM_XX.zip`
for a tournament run by the lecturer in ABIDES-NG. The practice market in `shared/practice_market.py`
is only a fast sandbox; the assessed market is different, with hidden seeds and extra scenarios
(crisis, low liquidity, other students' agents). Generalisation matters more than one-seed profit.

## Hard rules (never break)

1. **Never edit anything in `shared/`**, above all `shared/contract.py`. It is the common competition
   interface and the lecturer uses the same one. This also covers `practice_market.py`, `test_policy.py`,
   `export_submission.py`, `validate_submission.py` and `abides_smoke_test.py`.
   If an experiment needs a different environment or reward, subclass or wrap it in `student_work/`
   or in a new experiment file. Do not patch `shared/`.
2. **The model contract is fixed:** input float32 `[batch, 16]`, output float32 `[batch, 9]`, finite outputs,
   CPU-only, TorchScript (`torch.jit.script` or `trace`). The tournament takes `argmax(output)` as the action.
   Observation order (`OBSERVATION_NAMES`), the 9 actions, the order sizes (10/50), the position limit (500)
   and `CONTRACT_VERSION = "1.0"` must not change.
3. **Edit only** `student_work/` (`model.py`, `train.py`) and new experiment files of our own
   (for example `student_work/experiments/`). Do not touch the workshop `.docx`, the `.bat` files,
   `examples/` or `DELIVERABLE_SPECIFICATION.txt`.
4. **The submission ZIP contains exactly `model.pt` and `metadata.json` at the ZIP root.** Nothing else:
   no source, no checkpoints, no folders. Always build it with `shared.export_submission` and confirm
   `shared.validate_submission` prints `VALID` on that exact ZIP before calling it done.
5. **Do not select a model from a single seed or scenario.** Evaluate on stable, trending and volatile,
   across several seeds. Keep a held-out seed set that is never used for tuning. Seeds you tune against
   count as training data.
6. Do not export, overwrite `outputs/model.pt` with an unvetted model, or create anything in `submission/`
   unless asked. Experiments write to their own paths, for example `outputs/exp_<name>/model.pt`.

## Commands (Windows; use the venv)

- Setup once: `SETUP_ONCE.bat` (creates `.venv`, installs `requirements.txt`). `.venv` must be Python 3.12+
  (ABIDES-NG needs it). `abides-ng` is not on PyPI, so `requirements.txt` installs it from GitHub and needs `git`.
- Menu: `START_HERE.bat`
- Train: `.venv\Scripts\python.exe -m student_work.train --steps 5000 --seed 7 --out outputs\model.pt`
- Test: `.venv\Scripts\python.exe -m shared.test_policy --model outputs\model.pt --scenario all --seed 101`
- Export: `.venv\Scripts\python.exe -m shared.export_submission --team-id TEAM_XX --team-name "..." --members "A,B" --algorithm DQN --model outputs\model.pt`
- Validate: `.venv\Scripts\python.exe -m shared.validate_submission submission\TEAM_XX.zip`
- ABIDES check: `.venv\Scripts\python.exe -m shared.abides_smoke_test --scenario stable_day --seed 42`

Run modules from the repo root with `-m` (imports are `shared.*` and `student_work.*`).

Our own tools are in `student_work/`, owned by the orchestrator; workers import them and never edit them:
- `evaluate`: the fixed protocol and rubric-like SCORE. Example: `-m student_work.evaluate --set tuning --model n=path [--model ...]`.
  The seed sets are tuning, validation and holdout.
- `envs`: extra stress scenarios and `RandomRegimeEnv(family=practice|abides|mixed)`.
- `abides_eval`: runs our TorchScript policy inside ABIDES-NG, alone or `--competitive`. See below for which seeds are spent.
- `compose`: builds a seed ensemble and wraps it in `SafePolicy` guards.
- `build_final`: rebuilds the selected final model into `outputs\model.pt`.
- `abides_report` (ABIDES SCORE tables; `--add-hold`), `select_ensemble` (greedy member selection), `make_figures`
  (presentation PNGs), and `make_docx.py` (Spanish explanation; run it with `outputsenv_torch22`).
  `abides_eval` has the `G11`/`S6` grids, `a+b` overlays and `--resume`. ABIDES seeds 1–70 and practice `holdout`/`holdout2` are spent; use seeds 71+ for any new check.

## Workflow (in order)

1. **Setup, once:** `SETUP_ONCE.bat`, check `.venv` is Python 3.12+, run the ABIDES smoke test (menu 6).
2. **Baselines:** train the starter DQN and run the rule baseline. Record them in `student_work/EXPERIMENTS.md`
   with the fixed evaluation protocol (tuning vs held-out seeds).
3. **Improvement rounds:** see the multi-agent section below. Experiments use module commands with their own
   `--out` paths (menu 7 and 8 are hardwired to `outputs\model.pt`). No export or validate during this stage.
4. **Final model:** the orchestrator picks the winner on held-out seeds, copies it to `outputs\model.pt`
   and runs an ABIDES check.
5. **Release, only when told:** export the ZIP (menu 9 or `shared.export_submission`; it validates automatically),
   then run validate (V) on the exact ZIP to upload. Not done until the validator prints `VALID`.

## Model improvement stage: multi-agent experimentation

When improving the model, use parallel subagents to explore different solutions, combinations and paths.

- **Orchestrator and planner: Opus 5.5** (the main session). It designs the experiment plan,
  decides which paths to try and how they can be combined, launches the agents, compares results
  and picks the final model.
- **Experiment workers: Sonnet 5.5**. Each is spawned with the Agent tool and `model: "sonnet"`.
  Each worker implements and trains one variant, evaluates it and reports back. Workers do not plan the
  overall strategy or choose the final model.
- Candidate paths come from the workshop guide:
  - **Model:** wider or deeper MLP, duelling head, recurrent memory.
  - **Learning:** Double DQN, prioritised replay, PPO or actor-critic.
  - **Reward:** inventory, drawdown and turnover penalties, without collapsing to permanent HOLD.
  - **Training regimes:** curriculum or randomised scenarios.
  - **Exploration:** different epsilon schedules.
  - **State use:** feature ablations or temporal history.
  Also try combinations of the winners.
- **Isolation:** workers run in parallel, so they must not collide. Give each worker `isolation: "worktree"`,
  or a unique experiment name with its own file (`student_work/experiments/<name>.py`) and its own output
  directory (`outputs/exp_<name>/`). Workers never edit the same file and never write to `outputs/model.pt`.
- **Fair comparison:** give every worker the same evaluation protocol: the same scenarios, the same fixed
  seed list, and the same metrics (final NAV or PnL, max drawdown, trade count, final inventory, per scenario
  and per seed). The orchestrator also re-evaluates the top candidates itself on held-out seeds before trusting a worker's claim.
- **Worker report** (short and structured): what was changed and why, hyperparameters, training scenarios and
  seeds, per-scenario and per-seed results, any NaN/Inf or instability, and one limitation or failure mode.
- **Merging:** only the orchestrator merges a winning variant into `student_work/model.py` and `student_work/train.py`.
  Keep the merged result contract-compliant, then validate the exported model.
- Do not spawn more workers than the plan needs. Prefer a small number of well-separated experiments per round,
  then a second round that combines the winners.

## Experiment record (the guide requires keeping this)

The log is `student_work/EXPERIMENTS.md`. The orchestrator (Opus 5.5) keeps it up to date: update it after every
experiment round, record the evaluation protocol (tuning vs held-out seeds) before the first experiment,
and never delete failed experiments. Each entry has: algorithm and architecture, reward function,
training scenarios and seeds, key hyperparameters, training curve or diagnostics, performance by scenario and seed,
the reason the final model was selected, and one observed limitation or failure mode.

## Code notes

- `build_observation` normalises everything to roughly [-1, 1] or [0, 1]; the last feature is a constant bias.
- The practice reward is `pnl_change / 1000 - 0.0005 * abs(inventory)`. Changing it in training is allowed
  and encouraged, but do it in `student_work/`, not in `shared/`.
- The practice market fills limit orders with 55% probability and market orders always. ABIDES-NG is more
  realistic, so check any promising model with the ABIDES smoke test before trusting it.
- The saved model must be self-contained TorchScript with no dependency on training code.
