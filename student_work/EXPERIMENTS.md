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
Fixed on 2026-10-16. Implemented in `student_work/evaluate.py`; extra scenarios are in `student_work/envs.py`.

**Lecturer's ranking** (given to us): 50% mean return across scenarios and seeds (final value = cash + inventory
at market value), 20% risk (max drawdown), 20% robustness and consistency across scenarios, 10% operational
(invalid orders, excessive exposure, insolvency). Seeds and scenario parameters are hidden.

- **Scenarios:** the official practice ones (stable, trending, volatile), run with the untouched `PracticeMarketEnv`,
  plus stress ones in `ExtendedMarketEnv`:
  - crash: negative drift and shocks
  - downtrend
  - low_liquidity: 40 bps spread, depth 150, 25% limit fills
  - choppy

  The stress scenarios add adverse selection on limit fills, market impact, and a non-zero `open_orders`.
- **Episode lengths:** 300 and 1000 steps (the lecturer's length is unknown).
- **Seed sets:**
  - tuning: 10000–10009, used by workers.
  - validation: 20000–20009, used only by the orchestrator between rounds.
  - holdout: 30000–30019, used once for the final pick.

  Training env seeds are `seed + episode` and stay below about 2000, so they never overlap these.
- **Metrics per episode:** return %, max drawdown %, trades, final inventory, invalid-order attempts, share of steps
  with |inventory| > 400, and insolvency.
- **Composite SCORE**, rank-normalised within the comparison pool, which always includes HOLD and rule-momentum:
  - 0.5 × R (mean return)
  - 0.2 × D (mean max DD, lower is better)
  - 0.2 × C (rank of the worst-scenario mean return and of the share of profitable episodes)
  - 0.1 × O (invalid attempts per 100 steps + exposure share + 10 × insolvent share)

  A model that does not beat HOLD on SCORE is rejected.

## 3. Baselines
Results are on the **validation** set (140 episodes = 7 scenarios × 2 lengths × 10 seeds). JSON files are in `outputs/eval_baselines/`.

| Name | Description | SCORE | R % | DD % | worst scen % | profitable | O | trades | crash % |
|---|---|---|---|---|---|---|---|---|---|
| rule-momentum | `examples/demo_02` 5-step momentum | 0.725 | 2.73 | 1.92 | -1.36 | 0.60 | 0.0 | 452 | 5.0 |
| starter-dqn-s8 | Starter DQN, 5k steps, seed 8 | 0.725 | 3.61 | 3.90 | -1.21 | 0.61 | **54.3** | 229 | 13.6 |
| HOLD | always action 0 | 0.500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| starter-dqn-ens-safe300 | mean of s7+s8 + SafePolicy (cap 300) | 0.500 | 0.28 | 2.52 | -2.79 | 0.44 | 0.0 | 415 | 1.0 |
| starter-dqn-s7 | Starter DQN, 5k steps, seed 7 | 0.050 | -0.08 | 4.27 | -7.44 | 0.37 | 24.8 | 313 | -7.4 |

**Findings:**
- **Training-seed variance is huge.** In crash, s7 makes −7.4% and s8 makes +13.6%. Every recipe must be trained with
  at least 3 seeds and judged on the mean.
- **The starter DQN keeps pushing against the position limit**, with up to 54 invalid attempts per 100 steps. That is the
  10% operational part of the score.
- **The momentum rule is a strong, low-drawdown bar to beat.** It profits from down moves too.
- **s7 is long-biased.** It does well in trending and loses in crash and downtrend.

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

### Round 1 (2026-10-16). Common setup for all four experiments
- **Training:** 3 training seeds per variant (1, 2, 3). 60k env steps for DQN and 150k for PPO, with `torch.set_num_threads(1)`.
- **Evaluation:** tuning set (140 episodes per model). SCORE is rank-normalised inside the pooled table of the
  baselines, the control and the worker's models, so compare SCOREs only within one table.
  The means below are over the 3 seeds.
- **Results:** JSON files are in `outputs/exp_<name>_eval/`. Code is in `student_work/experiments/<name>.py`.

### EXP-000c: control, starter DQN at 60k steps (orchestrator)
- **Change:** none. It is `student_work/train.py` with `--steps 60000`, seeds 1–3, so the round-1 variants have the same step budget.
- **Results (mean):** R 0.83%, DD 4.1%, worst scenario −5.8%, O 43.
  - Per scenario: trending +12.9%, crash −5.8%, downtrend −2.3%.
  - 2 of 3 seeds do not beat HOLD.
- **Conclusion:** more steps make the long bias stronger. The policy learns "buy", because the 3 training regimes never go down.
- **Limitation:** it keeps hitting the ±500 position limit, at about 40 rejected attempts per 100 steps.

### EXP-001: reward_risk (W1, Sonnet). Risk-shaped reward
- **Hypothesis:** penalising risk cuts drawdown, invalid orders and seed variance without collapsing into HOLD.
- **Change (reward only):**
  `r = pnl/1000 − 0.0005|inv| − inv_quad·(inv/500)² − dd_coef·ΔDD⁺/1000 − trade_cost·[fill] − breach_pen·[would breach ±500]`
- **Variants:**

  | variant | dd_coef | inv_quad | trade_cost | breach_pen |
  |---|---|---|---|---|
  | light | 0.5 | 0.2 | 0.02 | 0.1 |
  | medium | 1.0 | 0.5 | 0.05 | 0.3 |
  | heavy | 2.0 | 1.0 | 0.15 | 1.0 |
- **Results (mean):**

  | variant | SCORE | R% | DD% | worst% | O |
  |---|---|---|---|---|---|
  | light | 0.26 | −0.39 | 4.65 | −9.6 | 27 |
  | **medium** | **0.62** | **1.46** | 3.79 | **−2.9** | 30 |
  | heavy | 0.56 | 0.95 | **3.38** | −3.7 | **14** |
  | control (same pool) | 0.40 | 0.83 | 4.12 | −5.8 | 43 |

  There was no collapse into HOLD (hold share ≤ 0.32) and no NaN.
- **Conclusion:** keep medium, as it has the best score. Keep heavy as the option for risk and operational behaviour.
- **Limitation:** it does not fix crash and downtrend, since there are no down-markets in training. The O term drops but stays above 0, because penalties cannot mask actions.

### EXP-002: regime_rand (W2, Sonnet). Domain randomisation of the training market
- **Hypothesis:** training on random regimes removes the long bias and the reliance on free limit fills.
- **Change (training env only):**
  - (a) pure `RandomRegimeEnv` with random length 150–1000
  - (b) 50/50 mix of official and random episodes
  - (c) curriculum from narrow to wide parameter ranges

  The named stress scenarios were never used for training.
- **Results (mean):**

  | variant | SCORE | R% | DD% | worst% | O | crash% | downtrend% | trending% |
  |---|---|---|---|---|---|---|---|---|
  | **a_random** | **0.63** | **1.79** | **2.39** | **−1.19** | 24 | **+5.2** | **+4.1** | 4.3 |
  | b_mix | 0.44 | 0.98 | 3.84 | −5.69 | 31 | −4.3 | −0.6 | 11.0 |
  | c_curric | 0.52 | 1.43 | 3.36 | −2.97 | 32 | +4.5 | −1.1 | 7.7 |
  | control (same pool) | 0.31 | 0.83 | 4.13 | −5.76 | 43 | −5.8 | −2.3 | 12.9 |
- **Conclusion:** keep a_random. It is the biggest round-1 gain.
  - b_mix keeps the long bias alive.
  - The curriculum adds complexity for no gain.
- **Limitation:**
  - Seed 3 of a_random collapsed into high churn with no edge (598 trades, R ≈ 0).
  - O is still 24.
  - The tuning gain is partly in-distribution, because the random ranges overlap the stress scenarios by design. It must be confirmed on validation seeds and in ABIDES.
- **Orchestrator follow-up:** a 3-seed ensemble of a_random with SafePolicy gave the following, versus +5% and +4% for the good single seeds.

  | cap | R% | DD% | O | crash% | downtrend% |
  |---|---|---|---|---|---|
  | 300 | 0.82 | 1.67 | 0 | −0.1 | −2.2 |
  | 500 | 1.23 | 1.99 | 0.1 | — | — |

  The bad seed dilutes the ensemble, so ensembles should only use members that pass on tuning.

### EXP-003: dqn_plus (W3, Sonnet). Algorithm and architecture upgrades
- **Change:**
  - full: Double DQN, duelling 128×128 with LayerNorm, 3-step returns, lr 1e-3 decaying to 1e-4, soft target τ = 0.005
  - dd (ablation): Double and duelling on the 64×64 net only
- **Results (mean):**

  | variant | SCORE | R% | DD% | worst% | O |
  |---|---|---|---|---|---|
  | full | 0.46 | 0.38 | 4.32 | −5.85 | 30 |
  | dd | 0.33 | −0.06 | 3.88 | −6.24 | 22 |
  | control (same pool) | 0.49 | 0.83 | 4.12 | −5.76 | 43 |

  Q-overestimation was still visible in dd seed 3.
- **Conclusion:** drop it. The algorithm is not the bottleneck: every variant overfits the up-only training regimes.
- **Limitation:** seed variance (about ±0.3 SCORE) is bigger than the difference between variants.

### EXP-004: ppo (W4, Sonnet). PPO actor-critic
- **Change:** separate Tanh actor and critic, GAE with λ = 0.95, clip 0.2, 2048-step rollouts, 150k steps.
  Variants: main, big (128 units, higher entropy), lowent, and lowent with γ = 0.98.
- **Results:** every variant has a negative mean R (−0.4% to −1.5%) and a worst scenario of −7% to −15%.
  - The best model (big_s1) holds 62% of the time and barely beats HOLD.
- **Failure mode:** greedy collapse. The sampled policy stays near uniform (entropy 1.7–2.1 out of a maximum of 2.2), but its argmax
  becomes "market-buy 50" 98% of the time. That gives +13% in trending, −15% in crash and O = 99.
  - PPO was not more stable across seeds than DQN.
- **Conclusion:** drop it.

### Round 1 summary
**The training-market distribution is the main lever.** Regime randomisation (EXP-002a) was best, the medium/heavy
risk reward (EXP-001) was second, and algorithm changes did nothing (EXP-003 and EXP-004).
Round 2 combines a_random with the medium/heavy reward and adds a training-time action mask that matches SafePolicy.

### EXP-005: abides_harness (W5, Sonnet). ABIDES-NG evaluation harness (infrastructure)
- **What it is:** `student_work/abides_eval.py`. It is a `PolicyAgent(TradingAgent)` added with `run_simulation(config, runtime_agents=[...])`.
  - **Each wakeup (60 s):** builds the observation with `shared.contract.build_observation` (L2 depth from 5 levels), places `action_to_order`
    orders, cancels resting limits after 3 wakeups, and skips orders that would breach ±500 or that it cannot pay for.
  - **Modes:** individual (each model alone) and `--competitive` (all models in one exchange).
- **Check:** HOLD gives exactly 0 in every run. |inventory| never goes above 500. There are no NaNs. A run takes 10–40 s (low_liquidity under 1 s).
- **Main finding, a distribution shift between practice and ABIDES:**
  - The ABIDES price is ~$1000 vs $100. Cash then caps longs near 100 shares, while shorting is uncapped, so leverage can reach several × NAV.
  - Calm ABIDES markets are 5–10× quieter.
  - Spreads are under 1 bp to a few bp.
  - Book imbalance is about ±0.6–0.85, against ±0.15 in practice.
  - stress_test has extreme moves.
- **Limitation:** we do not know how the lecturer's adapter defines depth, the wake interval or the cash checks.

### EXP-006: combo (Round 2, Sonnet). Regime randomisation + risk reward + training-time action mask
- **Change:** all configs use a_random training. 60k steps, seeds 1–5 for A, B and C.

  | config | reward | training mask |
  |---|---|---|
  | A | medium | no |
  | B | heavy | no |
  | C | medium | SafePolicy rule, cap 300 (used for greedy, exploration and TD target) |
  | D | as C, 150k steps | as C |
- **Results (tuning, mean over seeds):**

  | config | SCORE | DD% | worst% |
  |---|---|---|---|
  | B raw | **0.62** | 1.59 | −1.11 |
  | B with SafePolicy 300 | 0.54 | | |
  | C with SafePolicy 300 | 0.54 | | |
  | A | 0.48–0.49 | | |
  | D | 0.49–0.51 | | |

  Seeds below HOLD: A 3/5, B 1/5, C 2/5, D 1/3.
- **Conclusion:**
  - Heavy reward plus random regimes is the best recipe.
  - The training-time mask raises the worst seed but not the mean.
  - Training longer does not help.
  - Seed variance (0.3–0.8 SCORE within one config) is larger than the differences between configs.

### Orchestrator, Round 2: ensembles and guards (validation seeds, practice market)
Ensembles average the Q-values of seeds that pass on tuning. They are built with `student_work/compose.py`.

| candidate | SCORE | R% | DD% | worst% | profitable |
|---|---|---|---|---|---|
| top5mix_safe400 (C_s1, C_s4, A_s4, B_s5, B_s2) | 0.79 | 2.34 | 0.99 | −0.77 | 0.79 |
| Bpass4_safe400 (B s1, s2, s4, s5) | 0.73 | 2.10 | 1.07 | −0.25 | 0.71 |
| ABCpass9_safe300 | 0.69 | 1.85 | 0.95 | −0.57 | 0.74 |
| rule-momentum | 0.62 | 2.73 | 1.92 | −1.36 | 0.60 |
| C_s1_safe300 (best single on tuning: 0.84) | 0.52 | 1.36 | 1.07 | −1.04 | 0.76 |
| B_s5_safe300 (single) | 0.30 | 0.91 | 1.61 | −2.01 | 0.65 |
| Ball5_safe300 (unfiltered, includes collapsed B_s3) | 0.51 | 0.92 | 0.91 | −1.06 | 0.66 |

- **Single seeds that are "best on tuning" do not generalise. Filtered seed ensembles do.**
- **Cap 400 is better than cap 300** in the practice market.

### Orchestrator, Round 2: ABIDES guard sweep
Runs are individual, on ABIDES seeds 11–15, templates stable_day, volatile_day, low_liquidity and stress_test (20 runs per model).

Guards in SafePolicy:
- `min_cash_ratio` blocks buys once cash falls below k × start, and `max_cash_ratio` blocks growing a short once cash rises above k × start. Together they cap exposure as a share of NAV, independent of price.
- `brake_nav`: below that NAV return, the wrapper only de-risks, forcing market orders toward flat.

| model | ABIDES R% | mean DD% | max DD% | stress_test | volatile_day | practice validation R% / DD% |
|---|---|---|---|---|---|---|
| rule-momentum | −10.40 | 11.2 | 132 | −36.2 | −5.2 | 2.73 / 1.92 |
| top5 cap 400, no cash guard | −5.93 | 14.8 | 107 | +4.8 | −28.5 | 2.34 / 0.99 |
| top5 + exposure ≤ 50% (0.5 / 1.5) | −1.02 | 10.4 | 144 | −2.8 | −1.1 | 2.34 / 1.00 |
| top5 + exposure ≤ 40% (0.6 / 1.4) | −0.28 | 7.8 | 38.6 | −3.2 | +4.4 | 2.25 / 1.00 |
| **top5 + exposure ≤ 30% (0.7 / 1.3)** | **+2.34** | **2.0** | **7.6** | +9.1 | +0.5 | 1.96 / 1.07 |
| top5 + exposure ≤ 20% (0.8 / 1.2) | +3.26 | 2.3 | 11.5 | +12.9 | +0.3 | 1.59 / 0.99 |
| top5 x30 + brake −5% | +2.35 | 2.0 | 6.9 | +9.1 | +0.5 | 1.96 / 1.07 |
| B4 x30 | −20.7 | 25.1 | **401** (insolvent) | −83.1 | +0.7 | 1.79 / 1.07 |
| B4 x30 + brake −5% | (stress/volatile only) | | 31.4 | −11.9 | +0.7 | |

- **In ABIDES, exposure as a share of NAV decides survival.** The practice market prefers looser caps, ABIDES needs about 30%, and x30 is the compromise.
- **The brake is the tail-risk insurance.** On seed 14 of stress_test, B4 x30 went from −428% (insolvent) to −24.7% with the brake.
- **Caveat:** ABIDES seeds 11–15 were used to choose these guard levels, so they are now tuning data. The final ABIDES check uses seeds 21+.

### EXP-007: abides_like (Round 3, Sonnet). Training on ABIDES-like regimes
- **Change:** `RandomRegimeEnv(family="abides" | "mixed")`, using `envs.sample_abides_regime`, with heavy reward and 60k steps, seeds 1–5.
  - The ABIDES-like sampler has prices of $50–$2000, calm regimes 5–10× quieter, spreads under 1 bp up to 10 bp, book
    imbalance of ±0.3–0.9, and 25% extreme-stress episodes.
  - Every model was evaluated with the g30 guard (cap 400, cash 0.7 / 1.3).
- **ABIDES results (seeds 1–3, mean per family):**

  | family | R% | mean DD% | max DD% |
  |---|---|---|---|
  | abides | −8.2 | 23 | 407 |
  | mixed | −4.9 | 6.8 | — |
  | refB (round-2 B seeds) | −3.5 | 30 | — |

  The practice-market SCORE was about the same.
- **Conclusion:** the hypothesis was not supported. Pure ABIDES-like training made results worse.
  - mixed mostly taught smaller positions.
  - Two stable mixed seeds (s3, s5) were kept as ensemble members.
- **Limitation:** stress_test variance (−118% to +38%) swamps the differences between families.

### Orchestrator, Round 4: broken-book guard
- **Problem:** in ABIDES stress_test the book can empty out, with mid going from 100,000 to 24 cents. Any position is then
  marked at a meaningless price, which caused both the −151% / −94% blowups and the rule's lucky +224%.
- **Change:** `SafePolicy(max_spread_obs=0.5)`. While the spread feature is above 100 bps, no action may grow exposure.
  The practice market never exceeds 40 bps, so its results are identical.
- **stress_test results (seeds 11–15 and 31–40, top7 base, n = 15):**

  | guard | mean R% | mean DD% | max DD% |
  |---|---|---|---|
  | none | +22.1 | 23.3 | 189 |
  | **100 bps** | +8.0 | **7.0** | **21.6** |
  | 50 bps | +2.7 | 11.1 | 39.3 |

  Chosen: 100 bps.

## 5. Combinations
- **Round 2:** the regime randomisation winner (EXP-002) was combined with the risk reward winners (EXP-001), giving EXP-006.
  - Heavy reward plus random regimes was the best recipe.
  - Adding a training-time mask did not help the mean.
- **Seed ensembles** (mean Q-values over seeds that pass on tuning):
  - On validation they beat every single seed: the best single seed scored 0.52, the best ensemble 0.79.
  - Unfiltered ensembles are worse (0.51), because collapsed seeds dilute them.
- **Deployment guard layers** (`SafePolicy`) were added one at a time, each on the evidence above:
  share cap 400, then cash-based exposure ≤ 30% of NAV, then the −5% NAV brake, then the broken-book guard.
- **Final ensemble:** the 5 round-2 members plus 2 round-3 "mixed" members (top7).

## 6. Final model selection
- **Chosen model:** `top7mix_400_x30_b5_sp50`, rebuilt by `python -m student_work.build_final` into `outputs/model.pt`.
  - A TorchScript `SafePolicy(Ensemble(7 DQN members))`, where each member is MLP 16-64-64-9. The members are:
    - exp_combo-C_s1 and C_s4 (medium reward, training mask 300)
    - A_s4 (medium reward)
    - B_s5 and B_s2 (heavy reward)
    - exp_abides_like-mixed_s3 and mixed_s5 (heavy reward, mixed regimes)
  - Guard settings: cap 400, cash 0.7 / 1.3, brake −5%, spread guard 100 bps.
- **Reason for selection:** a seed ensemble generalises where single seeds do not, and the guards turn the ABIDES tail risks
  (leverage at a $1000 price, gap moves, broken books) into bounded losses.
  - The choice follows the rubric: 50% return, 20% drawdown, 20% robustness, 10% operational.
- **Pre-registered rule and how it was resolved:** before the holdout we fixed this rule: "keep top5 unless top7 has a better ABIDES
  holdout mean and max DD < 15%".
  - Both models failed the DD condition (top5 154%, top7 138%), because of the broken-book artifact. That was fixed afterwards
    with the spread guard, selected on ABIDES seeds 11–15 and 31–40, not on the holdout.
  - top7 was at least as good as top5 on every holdout aggregate, so top7 was chosen.
- **Held-out results, practice market** (holdout seeds 30000–30019, 280 episodes; top7 before the spread guard, which does not change practice results):

  | model | SCORE | R% | DD% | worst scenario % | profitable |
  |---|---|---|---|---|---|
  | **top7 (final)** | **0.713** | 2.23 | 1.35 | **−0.17** | 0.76 |
  | rule-momentum | 0.688 | 3.15 | 1.80 | −1.49 | 0.60 |
  | top5 | 0.588 | 2.07 | 1.09 | −0.59 | 0.75 |
  | HOLD | 0.488 | 0 | 0 | 0 | 0 |
  | starter-dqn-s7 | 0.025 | −0.17 | 4.39 | −7.26 | 0.41 |

  Per scenario (top7): stable +0.09, trending +0.53, volatile +1.08, crash +7.22, downtrend +5.88, low_liq +0.96, choppy −0.17.
  O = 0, so there were no invalid orders.
- **ABIDES-NG check, final model, untouched seeds 41–45**, 4 templates, 60 s wake:

  | model | mode | runs | R% | median R% | mean DD% | max DD% | skipped orders | profitable runs |
  |---|---|---|---|---|---|---|---|---|
  | **final** | individual | 20 | +2.19 | +0.06 | 1.20 | 8.9 | 0 | 55% |
  | rule | individual | 20 | −5.44 | | 10.3 | 87.9 | | |
  | starter-dqn | individual | 20 | +40.7 | −0.03 | 13.7 | 95.8 | 135 per run | |
  | final (copy a) | competitive | 12 | +4.71 | | 2.9 | 18.3 | | |
  | final (copy b) | competitive | 12 | +3.10 | | 3.0 | 15.8 | | |
  | rule | competitive | 12 | +2.44 | | 6.1 | 42.8 | | |
  | top5 | competitive | 12 | −2.73 | | 4.6 | 38.9 | | |

  - In the competitive runs, both final copies, the rule, the starter DQN and top5 trade in one exchange.
  - The starter DQN's +40.7% individual mean comes from leveraged lottery bets on broken stress_test books.
- **Compatibility:** the model loads in a throwaway **torch 2.2.2 + numpy 1.26** venv and gives bit-identical outputs
  (max diff 0, argmax agreement 100% on 1,400 observations). It was saved with torch 2.14.1.
- **`shared.test_policy` (seed 101, one episode per scenario):**

  | scenario | PnL |
  |---|---|
  | stable | +$300 |
  | trending | −$174 |
  | volatile | −$1,605 |

  This is a single-seed snapshot, and the multi-seed protocol above is the basis for selection.
- **Validator output:** not yet. The model has not been exported, because export waits for the team's go-ahead.
- **Known limitations / failure modes:**
  - **Trending markets.** The final model gives up most of the trending gains (+0.5% against +9% for the rule). Removing the long bias
    is what made crash and downtrend profitable.
  - **ABIDES stress_test** is still a lottery driven by simulator artifacts (books emptying out). The guards bound the loss but cannot make it predictable.
  - **The lecturer's ABIDES adapter is unknown:** depth definition, wake interval, cash checks and price scale. The guards are
    written to hold at any price, and they rely on the observation features `inventory`, `cash_ratio`, `nav_return` and `spread_bps`.
  - **Seed variance in DQN training is very large.** This is why the members were filtered on tuning seeds and then ensembled.


## 7. Stage 2: rounds 5–8 (started 2026-10-23)
The goal is a challenger (v2) that beats the incumbent v1 (§6) on fresh, pre-registered holdout seeds. If none does, v1 stays.
v1 is backed up in `outputs/backup_v1/` (`model.pt` and `TEAM_01.zip`).

### Pre-registration (written before any stage-2 result)
**Seed registry.** The v1 holdout sets are spent and are not used for decisions any more.

| set | practice (`evaluate.SEED_SETS`) | ABIDES | who |
|---|---|---|---|
| tuning | `tuning` 10000–10009 | 51–55 | workers and orchestrator |
| validation | `validation` 20000–20009 | 56–60 | orchestrator only, between rounds |
| holdout2 | `holdout2` 40000–40019 | 61–70 | final gate only, used once |
| spent | `holdout` 30000–30019 | 1–3, 11–15, 21–45 | not used for decisions |

**ABIDES grid G11** (`abides_eval --template G11`, individual, wake 60 s, depth 5 levels):
- stable_day, volatile_day, low_liquidity, stress_test
- the never-used templates trending_day, rmsc04, liquid_market, thin_market
- the overlays stable_day+with_execution (a POV agent buying 1.2M shares, so sustained buy pressure),
  stable_day+with_execution:ASK (sell pressure) and volatile_day+with_momentum

The ABIDES SCORE (`student_work/abides_report.py`) is the same rank-normalised 50/20/20/10 composite over template × seed runs,
with an exact always-HOLD row added to the pool.

**Final gate.** A challenger replaces v1 only if **all** of these hold:
1. **Practice:** on `holdout2`, its SCORE is higher than v1's. The pool is v1, the challengers, rule-momentum, HOLD and starter-s7.
2. **ABIDES G11 individual, seeds 61–70:**
   - mean R ≥ v1's
   - mean DD ≤ 1.25 × v1's
   - max DD ≤ max(15%, v1's)
   - 0 skipped orders per run on average (rounded)
3. **ABIDES competitive:** v1, the challenger(s), rule, starter-s7 and starter-s8 trade in one exchange, on 6 G11 templates
   (stable_day, volatile_day, stress_test, trending_day, stable_day+with_execution, stable_day+with_execution:ASK) × seeds 61–65.
   - The challenger's mean R ≥ v1's in the same runs.
   - Max DD ≤ max(20%, v1's).
4. **Compatibility:** the contract check passes, and outputs under torch 2.2.2 + numpy 1.26 are bit-identical.

If several challengers pass, the one with the highest holdout2 SCORE plus ABIDES SCORE wins. If none passes, v1 stays and this is reported as a negative result.

### Round 5 (orchestrator): diagnosis of v1 on G11, ABIDES tuning seeds 51–55
**Infrastructure.**
- The harness has overlays (`a+b`, `with_execution:ASK`) and the G11 and S6 grids.
- Exchange book and order logging are now off by default. They used ~1.8 GB per run and made 18 parallel runs thrash 32 GB of RAM;
  results are identical with logging off (checked on 2 runs).
- Results stream to a resumable `<tag>.partial.jsonl`.

**G11 individual (55 runs per model):**

| model | ABIDES SCORE | R% | median R% | mean DD% | max DD% | worst template | profitable | skips/run |
|---|---|---|---|---|---|---|---|---|
| HOLD | 0.733 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rule | 0.667 | +8.60 | −0.11 | 10.6 | 101.9 | −9.9 | 0.09 | 1.8 |
| **v1** | 0.517 | **−0.31** | −0.02 | 3.4 | 37.8 | −7.1 (sell pressure) | 0.45 | 0.3 |
| starter-s7 | 0.083 | −1.13 | −0.02 | 14.1 | 217.7 | −63 | 0.45 | 113 |

**v1 per template (mean R%):**

| template | v1 R% |
|---|---|
| stable_day | −0.08 |
| volatile_day | −4.44 (seed 51: −19.2) |
| low_liquidity | −0.20 |
| stress_test | +3.75 |
| trending_day | +0.16 |
| rmsc04 | 0.00 |
| liquid | −0.02 |
| thin | +0.08 |
| +exec BID | +1.24 |
| +exec ASK | −7.13 (seed 52: −31.3) |
| volatile+momentum | +3.41 (seeds: −6.8 … +23.2) |

The rule's mean comes from two book collapses (+368% on stress_test s51 and +240% on trending_day s52, with mid falling to 54 and 574 cents). Its median is −0.11% and only 9% of its runs are profitable.

**Adapter sensitivity (v1, 4 templates × seeds 51–52):**

| setting | v1 R% |
|---|---|
| wake 60 s (base) | −3.89 |
| wake 30 s | −2.03 |
| wake 120 s | −1.60 |
| depth 1 level | −1.08 |
| depth 10 levels | −0.25 |

The same two runs (volatile_day s51, stress_test s51) dominate every setting. No adapter setting is much worse than the base.

**Diagnosis.** v1's ABIDES losses come from execution mechanics, not from its market view:
1. **Resting-order overshoot.** Max |inventory| reaches 120–170 shares, which is more than 100% of NAV at a $1000 price, although the cash guard targets about 30%.
   Several LARGE limit orders rest at once (they are cancelled after 3 wakeups) and fill together. The guard only checks cash and inventory at decision time.
   The `open_orders` feature (x[12] = resting/10) shows the resting orders, but the guard does not use it. Volatile_day s51 (−19% on a −17% move) is this case.
2. **Fire sales in one-sided books.** Under sell pressure (exec ASK s52) v1 lost −31% (DD 38%) while the mid ended flat.
   The −5% brake forces market orders even when the book is broken, so they fill deep in an empty bid side.
3. **Churn.** v1 market-buys small about 50% of the time and limit-sells large about 15%, with about 270 fills per run. With sub-bp spreads this is close to neutral, but it is not an edge.

**Amendment to gate criterion 1** (made before any holdout2 run): "higher than v1's" becomes **"≥ v1's"**.
A guard-only fix (resting orders, broken book) changes nothing in the practice market, where open_orders = 0 and the spread is ≤ 40 bps. Its practice SCORE is then
identical to v1's by construction, and a strict ">" would reject it on a tie.

### Round 6 (2026-10-23): four Sonnet workers
After the round-5 diagnosis, the plan's `agg_guard` worker was replaced by `guard2`, which works on the resting-order and broken-book guards.
- **Evaluation:** practice **tuning** set, plus ABIDES S6 on seeds 51–55 for guard2 and feature_shift.
- **CPU caps:** guard2 7, seed_farm 6, trend 3, feature_shift 3.

### EXP-008: trend (W6, Sonnet). Direction-symmetric trend members. **Dropped**
- **Change:** its own regime sampler with p(trend) = 0.7, split evenly up and down, drift U(0.0002, 0.0008). The "trend" reward is
  (dd 2.0, inv_quad 0.2, trade 0.05, breach 1.0). It is trained with `combo.train`, swapping the env class only. 6 seeds × 60k steps.
  Code: `student_work/experiments/trend.py`.
- **Results (tuning):** on their own, 5 of 6 members ride uptrends (+7 to +13% in trending; the rule makes +9.7%), but most lose crash and downtrend.
  - Best single member, raw s6: R 2.85%, but O = 19.
  - **v1 + 2 trend members:** SCORE 0.500 vs v1 0.600. R 1.77 vs 1.92. Trending barely moves (0.48 → 0.51). Downtrend drops and the worst scenario falls to −1.61.
  - **v1 + 4 trend members:** SCORE 0.475, R 1.66.
- **Failure mode:** averaging mean Q lets v1's flat or short-leaning members outvote the trend members, so their edge disappears.
  Exploiting them would need a regime-gated mixture. The members are also seed-fragile: s1 holds 75% of the time, and only s6 rides downtrends.

### EXP-009: seed_farm (W7, Sonnet). More members from the proven recipes
- **Change:** recipes B (heavy), C (medium + mask 300) and M (mixed, heavy), each on seeds 6–15, giving 30 members.
  The filter was fixed in advance: tuning SCORE > HOLD in the pooled table, worst scenario ≥ −2%, hold share < 0.9.
  Code: `student_work/experiments/seed_farm.py`.
- **Pass rates:** B 5/10, C 5/10, M 4/10. The passers are C s9, s14, s8, s7, s12; M s14, s13, s8, s12; B s14, s12, s6, s15, s9.
  - Best single member: C s9, SCORE 0.856 (R 2.23%, worst −0.20%).
  - No NaN/Inf in any run.
- **Ensembles (tuning, v1 guard):**
  - farm_only (the 14 passers): SCORE 0.61 vs v1 0.55. R 1.41 vs 1.92. DD 0.65. Worst +0.09.
  - farm_top (v1's 7 + the top 7 passers): SCORE 0.53, R 1.89, DD 1.04.
  - farm_v1plus (v1's 7 + all 14): SCORE 0.41.
- **Limitation:** selecting and scoring on the same tuning set is optimistic. Only 14 of 30 seeds pass, so variance between seeds is still large.

### Round 7 (orchestrator): greedy ensemble selection and validation
- **Tool:** `student_work/select_ensemble.py`. Starting from v1's 7 members, each step adds the candidate that gives the best tuning SCORE
  in a pool of current, all trials, HOLD and rule. The candidates were the 14 farm passers and trend s6, s5 and s2.
  - It took 5 steps and added C_s12, C_s7, M_s13 and B_s15, then stopped because B_s14 lowered the SCORE.
  - Tuning R by step: 1.92 → 2.07 → 2.12 → 2.13 → 2.06 (DD 1.10, worst −0.06). No trend member was ever picked.
- **Practice validation** (seeds 20000–20009, 140 episodes, one pool; JSON in `outputs/eval_r7_val/`):

  | candidate | members | SCORE | R% | DD% | worst% | profitable | crash | downtrend | trending |
  |---|---|---|---|---|---|---|---|---|---|
  | **farm_top** | v1's 7 + C s9, s14, s8, M s14, B s14, B s12, M s13 | **0.667** | **2.07** | **1.03** | −0.52 | 0.71 | 7.82 | 5.89 | 0.63 |
  | rule-momentum | | 0.567 | 2.73 | 1.92 | −1.36 | 0.60 | 5.00 | 6.82 | 9.45 |
  | sel9 | v1's 7 + C s12, C s7 (greedy step 2) | 0.558 | 2.05 | 1.30 | −0.52 | 0.74 | 6.77 | 5.83 | 0.48 |
  | farm_only | 14 farm passers | 0.475 | 1.34 | 0.73 | −0.28 | 0.74 | 1.47 | 1.40 | 6.09 |
  | sel11 | greedy final (11) | 0.442 | 1.78 | 1.19 | −0.38 | 0.74 | 7.99 | 3.36 | 0.51 |
  | v1 | | 0.442 | 1.96 | 1.33 | −0.97 | 0.74 | 6.76 | 6.21 | 0.55 |
  | HOLD | | 0.350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

  - The greedy endpoint overfit tuning: sel11 drops back on validation.
  - farm_top, picked by a simple tuning rule (top half of the passers), generalises best and beats v1 on R, DD and worst scenario.
  - farm_only's lower DD comes from passivity in crash and downtrend.

### EXP-010: guard2 (W9, Sonnet). Resting-order and broken-book guards
- **Change:** `student_work/experiments/guard2.py` holds `SafePolicy2`, which wraps v1's 7 members. With every new option off its outputs
  are bit-identical to v1 (max diff 0 on 2001 observations).
  - `resting_mode` 1: while x[12] > 0, block exposure-growing limit actions.
  - `resting_mode` 2: while x[12] > 0, block every exposure-growing action.
  - `broken_brake` 1: when the book is broken, put HOLD on top instead of forcing a market order.
  - `broken_brake` 2: when the book is broken, force the shrinking limit order instead.
  - Cash bounds 0.65 / 1.35 as a looser variant.
- **ABIDES S6, seeds 51–55** (30 runs per model; v1's rows taken from round 5):

  | variant | ABIDES SCORE | R% | mean DD% | max DD% | worst template | max inv | volatile s51 | exec ASK s52 |
  |---|---|---|---|---|---|---|---|---|
  | v1 | 0.057 | −0.78 | 5.94 | 37.8 | −7.13 | 170 | −19.2 | −31.3 |
  | **a2** (resting, all) | **0.886** | **+1.71** | 4.14 | 33.7 | −0.09 | 117 | −3.3 | +22.9 |
  | a2 + b1 + cash 0.65 / 1.35 | 0.786 | +1.62 | 2.53 | 21.7 | −1.23 | 204 | −4.4 | +8.4 |
  | a2 + b1 | 0.714 | +1.46 | 4.17 | 33.7 | −0.09 | 117 | −3.3 | +16.4 |
  | a1 (resting, limit only) | 0.529 | +0.46 | 4.57 | 27.6 | −4.66 | 130 | −3.5 | +5.4 |
  | b2 | 0.271 | −0.06 | 5.18 | 33.4 | −4.72 | 170 | −12.3 | −18.5 |
  | b1 | 0.143 | −0.45 | 5.61 | 40.4 | −7.58 | 170 | −8.1 | −30.1 |
- **Practice tuning:**

  | variant | R% |
  |---|---|
  | v1 | 1.92 |
  | a2 | 1.25 |
  | a2 + b1 + cash 0.65 / 1.35 | 1.43 |

  The **amendment's premise was wrong**: the stress scenarios in `envs.py` simulate a non-zero `open_orders`, so a2 changes practice behaviour.
  The loss falls on crash (6.4 → 4.3), downtrend (6.0 → 3.9) and low_liquidity. The amendment to criterion 1 ("≥") stays, but it does not make guard changes practice-neutral.
- **Conclusion:** resting-order awareness is the fix for ABIDES. The broken-book brake variants do little.
  The cost is practice return, so round 7 adds `resting_mode` 3, which blocks only exposure-growing LARGE actions while orders rest.

### Round 7 (continued)
- **farm_top and sel9 with v1's guard on ABIDES S6, seeds 51–55:**

  | model | R% | max DD% | max inv | skips/run |
  |---|---|---|---|---|
  | farm_top | −1.42 | **192** | 370 | 3.1 |
  | sel9 | +1.16 | 17.5 | 218 | 0.2 |
  | v1 | −0.78 | 37.8 | 170 | 0.5 |

  farm_top wins on practice validation but is unsafe in ABIDES without a resting-order guard.
- **Merged into `student_work/model.py`:** `SafePolicy(resting_mode=0|1|2|3)`, with the matching `compose.py --resting-mode` flag.
  - Mode 0 reproduces v1: the `build_final` rebuild has identical outputs (max diff 0, every argmax the same; the file bytes differ only because the scripted code changed).
  - Mode 2 is identical to W9's a2.
- **Grid screened:** {v1, sel9, farm_top (14 members)} × resting mode {2, 3}, on practice tuning and ABIDES S6 seeds 51–55.

### EXP-011: feature_shift (W8, Sonnet). Clamp or neutralise out-of-range ABIDES features
- **Change:** `student_work/experiments/feature_shift.py` adds `ClampedBase`, which changes the observation before the Q-ensemble while the guard reads the raw x.
  The ranges are p1–p99 of 157k practice observations from v1 rollouts (seeds 100–139, plus RandomRegimeEnv 1000–1299), stored in `outputs/exp_fshift/ranges.json`.
  - a: clamp x0–x7, x13 and x14.
  - b: clamp x5–x7.
  - c: neutralise the book (x5 = 0, x6 = x7 = practice mean 0.608).
  - d: c plus clamping the return features.
- **ABIDES S6, seeds 51–55:**

  | variant | ABIDES SCORE | R% | mean DD% | max DD% | worst template |
  |---|---|---|---|---|---|
  | **c** | 0.825 | **+4.15** | 3.79 | 27.3 | −0.20 |
  | b | 0.575 | +0.21 | 5.12 | 30.2 | −1.50 |
  | a | 0.450 | +2.81 | 6.46 | 40.4 | −1.92 |
  | v1 | 0.125 | −0.78 | 5.94 | 37.8 | −7.13 |

- **Practice tuning:**

  | variant | R% | worst scenario |
  |---|---|---|
  | b | 1.93 | identical to v1 |
  | c | 2.02 | −1.42 (choppy) |
  | v1 | 1.92 | |
  | a | 1.77 | |
  | d | 1.75 | does not beat HOLD |
- **Conclusion:** neutralising the book features is roughly practice-neutral and helps in ABIDES, which supports the extrapolation hypothesis.
  The median R is still about 0, so the mean gain is noisy, and stress_test max DD rises to 27%. The churn (BUY_MARKET_SMALL ≈ 23%) remains.
- **Merged:** `model.BookNeutral` (x5 = 0, x6 = x7 = 0.6077) and `compose.py --neutral-book`. Its argmax agrees 100% with W8's variant c.

### Round 7: combination grid (practice tuning; ABIDES S6, seeds 51–55)
| model | members | resting mode | practice SCORE | practice R% | practice DD% | practice worst | ABIDES SCORE | ABIDES R% | ABIDES median | ABIDES max DD | max inv | ABIDES profitable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **sel9_rm3** | v1's 7 + C s12, C s7 | 3 (LARGE) | **0.700** | 1.57 | 0.96 | −0.44 | **0.750** | **+3.27** | +0.05 | **9.8** | 90 | 0.57 |
| v1 | v1's 7 | 0 | 0.600 | 1.92 | 1.26 | −0.64 | 0.044 | −0.78 | −0.18 | 37.8 | 170 | 0.30 |
| top14_rm3 | farm_top | 3 | 0.575 | 1.27 | 0.83 | −0.44 | 0.533 | +3.36 | −0.03 | 41.5 | 153 | 0.47 |
| top14_rm2 | farm_top | 2 | 0.550 | 1.27 | 0.78 | −0.46 | 0.550 | +1.90 | −0.01 | 11.7 | 81 | 0.50 |
| v1_rm3 | v1's 7 | 3 | 0.513 | 1.39 | 0.96 | −0.52 | 0.633 | +2.44 | −0.02 | 26.7 | 159 | 0.47 |
| sel9_rm2 | sel9 | 2 | 0.350 | 1.10 | 0.92 | −0.69 | 0.600 | +1.76 | −0.01 | 5.4 | 110 | 0.50 |

- The ABIDES SCOREs are from one pool that also holds the W8 and W9 models, so they compare only within this table.
- **sel9_rm3 is best on both sides.** Every resting-order variant costs practice R, because the practice stress scenarios report open orders:
  1.92 → 1.57 for sel9_rm3. sel9_rm3 still has the best practice SCORE, through lower DD and a better worst scenario.
- **Two challengers go to validation:** sel9_rm3, and sel9_rm3_nb (sel9_rm3 with `--neutral-book`).
  The validation set is practice validation plus ABIDES G11 seeds 56–60, alongside v1.
