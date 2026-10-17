"""Run our TorchScript policies as trading agents inside ABIDES-NG.

The practice market in ``shared/practice_market.py`` is only a sandbox. This
harness injects one ``PolicyAgent`` per model into an ABIDES-NG template through
the supported ``run_simulation(config, runtime_agents=[...])`` hook, so the
policy sees the same observation contract against the real limit order book.

Examples (run from the repo root):

    .venv\\Scripts\\python.exe -m student_work.abides_eval ^
        --model HOLD --model rule --model dqn=outputs\\exp_baseline_s7\\model.pt ^
        --template stable_day --template volatile_day --seeds 1 2 --wake 60

    # all models in the same exchange
    .venv\\Scripts\\python.exe -m student_work.abides_eval --competitive ^
        --model HOLD --model rule --model dqn=outputs\\exp_baseline_s7\\model.pt ^
        --template stable_day --seeds 1

    # compare observation feature ranges, ABIDES vs the practice market
    .venv\\Scripts\\python.exe -m student_work.abides_eval --model rule ^
        --template stable_day --seeds 1 --practice-compare

Nothing here edits ``shared/``; everything is written under ``outputs/abides_eval/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from shared.contract import (
    ACTION_NAMES, ACTION_SIZE, OBSERVATION_NAMES, OBSERVATION_SIZE,
    POSITION_LIMIT, STARTING_CASH_CENTS, action_to_order, build_observation,
)

torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "abides_eval"
SYMBOL = "ABM"
DEFAULT_TEMPLATES = ["stable_day", "volatile_day", "low_liquidity", "stress_test"]
NS = 1_000_000_000


# ---------------------------------------------------------------------------
# Policies (pseudo-models + TorchScript)
# ---------------------------------------------------------------------------

class Policy:
    """Maps a 16-d observation to an action 0..8."""

    def __init__(self, name: str, spec: str):
        self.name = name
        self.spec = spec
        self.module = None
        self.nonfinite = 0
        if spec.upper() == "HOLD":
            self.kind = "hold"
        elif spec.lower() == "rule":
            self.kind = "rule"
        else:
            self.kind = "torchscript"
            path = Path(spec)
            if not path.is_absolute():
                path = REPO_ROOT / path
            self.module = torch.jit.load(str(path), map_location="cpu").eval()

    def act(self, obs: np.ndarray) -> int:
        if self.kind == "hold":
            return 0
        if self.kind == "rule":
            if obs[1] > 0.015 and obs[8] < 0.8:
                return 1
            if obs[1] < -0.015 and obs[8] > -0.8:
                return 2
            return 0
        with torch.no_grad():
            out = self.module(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0))
        out = out.detach().cpu().numpy().reshape(-1)
        if out.shape[0] != ACTION_SIZE or not np.all(np.isfinite(out)):
            self.nonfinite += 1
            return 0
        return int(np.argmax(out))


def parse_model_arg(arg: str) -> tuple[str, str]:
    """'name=path' | 'HOLD' | 'rule' | 'path.pt' -> (name, spec)."""
    if arg.upper() == "HOLD":
        return "HOLD", "HOLD"
    if arg.lower() == "rule":
        return "rule", "rule"
    if "=" in arg:
        name, spec = arg.split("=", 1)
        return name, spec
    return Path(arg).parent.name or Path(arg).stem, arg


# ---------------------------------------------------------------------------
# ABIDES agent
# ---------------------------------------------------------------------------

def _make_policy_agent_class():
    """Defer the abides imports so ``--help`` works without the package."""
    from abides_markets.agents.trading_agent import TradingAgent
    from abides_markets.messages.query import QuerySpreadResponseMsg
    from abides_markets.orders import LimitOrder, Side

    class PolicyAgent(TradingAgent):
        """Trades according to a ``Policy`` once per ``wake_ns`` of simulated time.

        Cycle: wakeup -> QuerySpread(depth=L2 levels) -> receive response -> build
        observation -> policy -> place order -> schedule next wakeup.
        Orders are placed with the TradingAgent API (place_market_order /
        place_limit_order); resting limit orders are tracked in ``self.resting``
        and cancelled after ``stale_wakeups`` wakeups.
        """

        VALID_STATES = frozenset({"AWAITING_WAKEUP", "AWAITING_SPREAD"})

        def __init__(self, id, symbol, starting_cash, name, random_state, policy,
                     wake_ns, depth_levels=5, stale_wakeups=3, query_depth=10):
            super().__init__(id, name, "PolicyAgent", random_state, starting_cash,
                             log_orders=False)
            self.symbol = symbol
            self.policy = policy
            self.wake_ns = int(wake_ns)
            self.depth_levels = int(depth_levels)
            self.query_depth = max(int(query_depth), self.depth_levels)
            self.stale_wakeups = int(stale_wakeups)
            self.state = "AWAITING_WAKEUP"
            self.category = "policy"
            # own bookkeeping
            self.mid_history: list[float] = []
            self.resting: dict[int, dict] = {}     # order_id -> {side, qty, wake}
            self.cancel_sent: dict[int, int] = {}
            self.wake_idx = 0
            self.stats = dict(
                wakeups=0, no_quote=0, orders_placed=0, market_orders=0, limit_orders=0,
                skipped_position=0, skipped_cash=0, rejected=0, cancels=0,
                fills=0, filled_shares=0, max_abs_inventory=0,
            )
            self.action_counts = [0] * ACTION_SIZE
            self.nav_series: list[float] = []
            self.mid_series: list[float] = []
            self.inv_series: list[int] = []
            self.obs_log: list[np.ndarray] = []
            self.raw_depth: list[tuple[int, int, int, int, int, int]] = []
            self.spread_cents: list[int] = []
            self.final_mid: float | None = None
            self.final_nav: float | None = None

        # ---- scheduling -------------------------------------------------
        def get_wake_frequency(self):
            return self.wake_ns          # first wakeup = market open + one interval

        def wakeup(self, current_time):
            can_trade = super().wakeup(current_time)
            if not can_trade:
                return
            self.wake_idx += 1
            self._purge_ghost_market_orders()
            self.get_current_spread(self.symbol, depth=self.query_depth)
            self.state = "AWAITING_SPREAD"

        def _purge_ghost_market_orders(self):
            # ABIDES never answers an unfilled/partially filled market order, so the
            # copy kept in self.orders would linger forever. Drop anything that is
            # not a tracked resting limit order.
            for oid in [o for o in self.orders if o not in self.resting]:
                del self.orders[oid]

        # ---- messages ---------------------------------------------------
        def receive_message(self, current_time, sender_id, message):
            super().receive_message(current_time, sender_id, message)
            if self.state == "AWAITING_SPREAD" and isinstance(message, QuerySpreadResponseMsg):
                self.state = "AWAITING_WAKEUP"
                self._decide(current_time)
                nxt = current_time + self.wake_ns
                if not self.mkt_closed and nxt <= self.mkt_close:
                    self.set_wakeup(nxt)

        def order_executed(self, order):
            super().order_executed(order)
            self.stats["fills"] += 1
            self.stats["filled_shares"] += int(order.quantity)
            inv = self.get_holdings(self.symbol)
            self.stats["max_abs_inventory"] = max(self.stats["max_abs_inventory"], abs(inv))
            info = self.resting.get(order.order_id)
            if info is not None:
                info["qty"] -= int(order.quantity)
                if info["qty"] <= 0:
                    self.resting.pop(order.order_id, None)
                    self.cancel_sent.pop(order.order_id, None)

        def order_cancelled(self, order):
            self.resting.pop(order.order_id, None)
            self.cancel_sent.pop(order.order_id, None)
            self.orders.pop(order.order_id, None)    # avoid base-class warning noise
            self.stats["cancels"] += 1

        def on_order_rejected(self, order_id, reason):
            self.stats["rejected"] += 1
            self.resting.pop(order_id, None)
            self.cancel_sent.pop(order_id, None)
            super().on_order_rejected(order_id, reason)

        # ---- decision ---------------------------------------------------
        def _depth(self, levels):
            return sum(q for _, q in levels[: self.depth_levels])

        def _decide(self, now):
            st = self.stats
            st["wakeups"] += 1
            bids = self.known_bids.get(self.symbol) or []
            asks = self.known_asks.get(self.symbol) or []
            inv = self.get_holdings(self.symbol)
            cash = self.holdings["CASH"]

            # stale resting orders -> cancel
            for oid, info in list(self.resting.items()):
                if self.wake_idx - info["wake"] >= self.stale_wakeups:
                    sent = self.cancel_sent.get(oid)
                    if sent is None or self.wake_idx - sent >= 2:
                        o = self.orders.get(oid)
                        if o is not None:
                            self.cancel_order(o)
                            self.cancel_sent[oid] = self.wake_idx
                        else:
                            self.resting.pop(oid, None)

            if not bids or not asks:
                st["no_quote"] += 1
                last = self.last_trade.get(self.symbol)
                mid = float(last) if last else (self.mid_series[-1] if self.mid_series else None)
                if mid is not None:
                    self._record_nav(mid, inv, cash)
                return
            bid, ask = int(bids[0][0]), int(asks[0][0])
            if ask <= bid:                    # crossed/locked should not happen; be safe
                ask = bid + 1
            mid = (bid + ask) / 2.0
            self.mid_history.append(mid)
            self.spread_cents.append(ask - bid)
            bid_depth, ask_depth = self._depth(bids), self._depth(asks)
            self.raw_depth.append((int(bids[0][1]), int(asks[0][1]),
                                   sum(q for _, q in bids[:5]), sum(q for _, q in asks[:5]),
                                   sum(q for _, q in bids[:10]), sum(q for _, q in asks[:10])))
            progress = (now - self.mkt_open) / max(1, self.mkt_close - self.mkt_open)
            obs = build_observation(
                self.mid_history, bid, ask, bid_depth, ask_depth, inv, cash,
                time_progress=progress, open_orders=len(self.resting),
            )
            self.obs_log.append(obs)
            self._record_nav(mid, inv, cash)

            action = self.policy.act(obs)
            self.action_counts[action] += 1
            order = action_to_order(action, bid, ask)
            if order["kind"] == "HOLD":
                return
            qty = int(order["quantity"])
            is_buy = order["side"] == "BUY"
            # Worst case: every resting order on the SAME side fills (opposite-side
            # resting orders must not be netted, they may never fill).
            same = sum(i["qty"] for i in self.resting.values() if i["side"] == order["side"])
            projected = inv + (same + qty if is_buy else -(same + qty))
            if abs(projected) > POSITION_LIMIT:
                st["skipped_position"] += 1
                return
            if is_buy:
                price = ask if order["kind"] == "MARKET" else int(order["limit_price"])
                pending_cost = sum(i["qty"] * i["price"] for i in self.resting.values()
                                   if i["side"] == "BUY")
                if cash - pending_cost < qty * price:
                    st["skipped_cash"] += 1
                    return
            side = Side.BID if is_buy else Side.ASK
            if order["kind"] == "MARKET":
                self.place_market_order(self.symbol, qty, side)
                st["market_orders"] += 1
            else:
                price = int(order["limit_price"])
                before = set(self.orders)
                self.place_limit_order(self.symbol, qty, side, price)
                new = [o for o in self.orders if o not in before]
                if not new:
                    st["skipped_position"] += 1
                    return
                self.resting[new[0]] = dict(side=order["side"], qty=qty, wake=self.wake_idx,
                                            price=price)
                st["limit_orders"] += 1
            st["orders_placed"] += 1

        def _record_nav(self, mid, inv, cash):
            self.mid_series.append(float(mid))
            self.inv_series.append(int(inv))
            self.nav_series.append(float(cash + inv * mid))
            self.final_mid = float(mid)
            self.final_nav = float(cash + inv * mid)

    return PolicyAgent


# ---------------------------------------------------------------------------
# One simulation
# ---------------------------------------------------------------------------

def _agent_summary(agent, abides_agent) -> dict:
    nav = np.asarray(agent.nav_series, dtype=np.float64)
    start = float(STARTING_CASH_CENTS)
    if nav.size:
        peaks = np.maximum.accumulate(np.concatenate([[start], nav]))
        dd = (peaks - np.concatenate([[start], nav])) / peaks
        max_dd = float(dd.max() * 100.0)
        final_nav = float(nav[-1])
    else:
        max_dd, final_nav = 0.0, start
    total_actions = max(1, sum(agent.action_counts))
    inv = int(agent.get_holdings(agent.symbol))
    d = dict(
        name=agent.name,
        return_pct=(final_nav / start - 1.0) * 100.0,
        max_drawdown_pct=max_dd,
        final_nav_cents=final_nav,
        final_inventory=inv,
        final_cash_cents=int(agent.holdings["CASH"]),
        n_fills=agent.stats["fills"],
        filled_shares=agent.stats["filled_shares"],
        orders_placed=agent.stats["orders_placed"],
        skipped_position=agent.stats["skipped_position"],
        skipped_cash=agent.stats["skipped_cash"],
        skipped_total=agent.stats["skipped_position"] + agent.stats["skipped_cash"],
        rejected=agent.stats["rejected"],
        no_quote=agent.stats["no_quote"],
        cancels=agent.stats["cancels"],
        wakeups=agent.stats["wakeups"],
        max_abs_inventory=int(max([agent.stats["max_abs_inventory"]] + [abs(i) for i in agent.inv_series])),
        action_share={ACTION_NAMES[i]: round(c / total_actions, 4)
                      for i, c in enumerate(agent.action_counts) if c},
        nonfinite_outputs=agent.policy.nonfinite,
    )
    if abides_agent is not None:
        d["abides_pnl_cents"] = int(abides_agent.pnl_cents)
        d["abides_pnl_pct"] = float(abides_agent.pnl_pct)
    return d


def feature_stats(obs_list: list[np.ndarray]) -> dict:
    if not obs_list:
        return {}
    a = np.stack(obs_list)
    return {
        n: dict(min=float(a[:, i].min()), p5=float(np.percentile(a[:, i], 5)),
                mean=float(a[:, i].mean()), p95=float(np.percentile(a[:, i], 95)),
                max=float(a[:, i].max()))
        for i, n in enumerate(OBSERVATION_NAMES)
    }


def run_one(task: dict) -> dict:
    """Run one ABIDES simulation. Top-level so it can run in a worker process."""
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    from abides_markets.config_system import SimulationBuilder
    from abides_markets.simulation import run_simulation

    template, seed = task["template"], int(task["seed"])
    wake_ns = int(task["wake"]) * NS
    PolicyAgent = _make_policy_agent_class()
    agents = []
    for i, (name, spec) in enumerate(task["models"]):
        agents.append(PolicyAgent(
            id=-1, symbol=SYMBOL, starting_cash=STARTING_CASH_CENTS, name=name,
            random_state=np.random.RandomState(seed * 1000 + 17 + i),
            policy=Policy(name, spec), wake_ns=wake_ns,
            depth_levels=task["depth_levels"], stale_wakeups=task["stale"],
        ))
    builder = SimulationBuilder().apply_template(template).ticker(SYMBOL).seed(seed)
    if task.get("r_bar"):      # optional experiment: rescale the fundamental price level
        mkt = builder._market
        builder.oracle(mkt.oracle.model_copy(update={"r_bar": int(task["r_bar"])}))
    config = builder.build()
    log_dir = str(OUT_DIR / "logs" / f"{template}_{seed}_{int(time.time()*1000)}")
    t0 = time.time()
    result = run_simulation(config, log_dir=log_dir, runtime_agents=agents)
    runtime_s = time.time() - t0
    by_id = {a.agent_id: a for a in result.agents}
    rows = []
    for ag in agents:
        row = _agent_summary(ag, by_id.get(ag.id))
        row.update(template=template, seed=seed, mode="competitive" if task["competitive"] else "individual",
                   wake_s=int(task["wake"]), runtime_s=runtime_s)
        row["features"] = feature_stats(ag.obs_log)
        row["raw_depth"] = _depth_summary(ag.raw_depth)
        sc = np.asarray(ag.spread_cents, dtype=np.float64)
        row["spread_cents"] = (dict(min=float(sc.min()), median=float(np.median(sc)),
                                    p95=float(np.percentile(sc, 95)), max=float(sc.max()))
                               if sc.size else {})
        row["mid_start"] = ag.mid_series[0] if ag.mid_series else None
        row["mid_end"] = ag.mid_series[-1] if ag.mid_series else None
        row["_obs"] = [o.tolist() for o in ag.obs_log] if task.get("keep_obs") else None
        rows.append(row)
    # market context
    l1 = result.markets[SYMBOL].l1_close
    ctx = dict(template=template, seed=seed, runtime_s=runtime_s,
               n_background=len(result.agents) - len(agents),
               pct_time_no_bid=result.markets[SYMBOL].liquidity.pct_time_no_bid,
               pct_time_no_ask=result.markets[SYMBOL].liquidity.pct_time_no_ask,
               total_volume=result.markets[SYMBOL].liquidity.total_exchanged_volume)
    return dict(rows=rows, context=ctx)


def _depth_summary(raw) -> dict:
    if not raw:
        return {}
    a = np.asarray(raw, dtype=np.float64)
    labels = ["bid_L1", "ask_L1", "bid_L5", "ask_L5", "bid_L10", "ask_L10"]
    return {l: dict(min=float(a[:, i].min()), median=float(np.median(a[:, i])),
                    p95=float(np.percentile(a[:, i], 95)), max=float(a[:, i].max()))
            for i, l in enumerate(labels)}


# ---------------------------------------------------------------------------
# Practice-market feature stats (for the ABIDES comparison)
# ---------------------------------------------------------------------------

def practice_feature_stats(policy_specs, seeds, steps=300) -> dict:
    from shared.practice_market import PRACTICE_SCENARIOS, PracticeMarketEnv
    out: dict = {}
    for name, spec in policy_specs:
        pol = Policy(name, spec)
        per_scn = {}
        for scn in PRACTICE_SCENARIOS:
            obs_all = []
            for s in seeds:
                env = PracticeMarketEnv(scn, steps=steps, seed=int(s))
                obs = env.reset()
                done = False
                while not done:
                    obs_all.append(obs)
                    obs, _, done, _ = env.step(pol.act(obs))
            per_scn[scn] = feature_stats(obs_all)
        out[name] = per_scn
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_tasks(args, models):
    tasks = []
    for tpl in args.template:
        for seed in args.seeds:
            base = dict(template=tpl, seed=seed, wake=args.wake, depth_levels=args.depth_levels,
                        stale=args.stale, keep_obs=False, r_bar=args.r_bar)
            if args.competitive:
                tasks.append(dict(base, models=models, competitive=True))
            else:
                for m in models:
                    tasks.append(dict(base, models=[m], competitive=False))
    return tasks


def print_table(rows):
    hdr = (f"{'template':14s}{'seed':>5s}{'mode':>6s} {'model':14s}{'ret%':>8s}{'maxDD%':>8s}"
           f"{'fills':>6s}{'shs':>6s}{'inv':>5s}{'|inv|max':>9s}{'skipP':>6s}{'skipC':>6s}{'rej':>4s}{'wk':>5s}"
           f"{'abidesPnL%':>11s}{'sim_s':>7s}  actions")
    print(hdr)
    for r in rows:
        acts = " ".join(f"{k.replace('_MARKET','M').replace('_LIMIT','L').replace('_SMALL','s').replace('_LARGE','l')}:{v:.2f}"
                        for k, v in r["action_share"].items())
        print(f"{r['template']:14s}{r['seed']:>5d}{r['mode'][:4]:>6s} {r['name'][:14]:14s}"
              f"{r['return_pct']:8.3f}{r['max_drawdown_pct']:8.3f}{r['n_fills']:6d}{r['filled_shares']:6d}"
              f"{r['final_inventory']:5d}{r['max_abs_inventory']:9d}{r['skipped_position']:6d}{r['skipped_cash']:6d}{r['rejected']:4d}"
              f"{r['wakeups']:5d}{r.get('abides_pnl_pct', float('nan')):11.3f}{r['runtime_s']:7.1f}  {acts}")


def print_feature_comparison(rows, practice, model_name):
    """p5/mean/p95 per feature: practice scenarios vs ABIDES templates for ONE model."""
    sel = [r for r in rows if r["name"] == model_name]
    templates = sorted({r["template"] for r in sel}, key=[r["template"] for r in sel].index)
    cols = []   # (label, {feature: stat})
    if practice and model_name in practice:
        for scn, st in practice[model_name].items():
            cols.append(("P:" + scn[:6], st))
    for tpl in templates:
        pooled = {n: _pool([r["features"][n] for r in sel if r["template"] == tpl])
                  for n in OBSERVATION_NAMES}
        cols.append(("A:" + tpl[:8], pooled))
    print(f"\nFeature p5 / mean / p95 for model '{model_name}'  (P = practice market, A = ABIDES template)")
    print(f"{'':15s}" + "".join(f"{lab:>22s}" for lab, _ in cols))
    for n in OBSERVATION_NAMES:
        print(f"{n:15s}" + "".join(
            f"{st[n]['p5']:7.3f}/{st[n]['mean']:6.3f}/{st[n]['p95']:6.3f}" for _, st in cols))
    print("\nABIDES raw book (median [p95] shares, summed over levels) and spread in cents:")
    for tpl in templates:
        rr = [r for r in sel if r["template"] == tpl]
        d = rr[0]["raw_depth"]
        if d:
            print(f"  {tpl:14s} mid~{rr[0]['mid_start']:.0f}c  spread_c med={rr[0]['spread_cents']['median']:.0f} "
                  f"p95={rr[0]['spread_cents']['p95']:.0f} max={rr[0]['spread_cents']['max']:.0f} | depth "
                  + " ".join(f"{k}={d[k]['median']:.0f}[{d[k]['p95']:.0f}]" for k in
                             ("bid_L1", "bid_L5", "bid_L10")))


def _pool(lst):
    return dict(min=min(s["min"] for s in lst), p5=float(np.mean([s["p5"] for s in lst])),
                mean=float(np.mean([s["mean"] for s in lst])),
                p95=float(np.mean([s["p95"] for s in lst])), max=max(s["max"] for s in lst))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                   help="name=path.pt | path.pt | HOLD | rule (repeatable)")
    p.add_argument("--template", action="append", default=None,
                   help=f"ABIDES template (repeatable). Default: {' '.join(DEFAULT_TEMPLATES)}")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--wake", type=int, default=60, help="seconds of simulated time between wakeups")
    p.add_argument("--competitive", action="store_true",
                   help="all models trade in the same simulation (default: each alone)")
    p.add_argument("--depth-levels", type=int, default=5,
                   help="number of L2 price levels summed into bid_depth/ask_depth (default 5)")
    p.add_argument("--stale", type=int, default=3,
                   help="cancel resting limit orders older than this many wakeups (default 3)")
    p.add_argument("--r-bar", type=int, default=None,
                   help="override the oracle fundamental price in cents (templates use 100000 = $1000; "
                        "the practice market starts at 10000). Experiment only: other price-denominated "
                        "template parameters are NOT rescaled.")
    p.add_argument("--jobs", type=int, default=1, help="parallel simulations (processes)")
    p.add_argument("--practice-compare", action="store_true",
                   help="also compute practice-market feature ranges with the same policies")
    p.add_argument("--features", action="store_true", help="print the per-feature ranges")
    p.add_argument("--features-model", default=None,
                   help="model name used for the feature comparison (default: last model given)")
    p.add_argument("--tag", default=None, help="output file tag")
    args = p.parse_args(argv)
    args.template = args.template or DEFAULT_TEMPLATES

    models = [parse_model_arg(m) for m in args.model]
    names = [n for n, _ in models]
    if len(set(names)) != len(names):
        p.error(f"duplicate model names: {names}")
    tasks = build_tasks(args, models)
    print(f"{len(tasks)} simulation(s); templates={args.template} seeds={args.seeds} wake={args.wake}s "
          f"mode={'competitive' if args.competitive else 'individual'}", flush=True)

    t0 = time.time()
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for res in ex.map(run_one, tasks):
                results.append(res)
    else:
        for i, t in enumerate(tasks, 1):
            res = run_one(t)
            results.append(res)
            print(f"  [{i}/{len(tasks)}] {t['template']} seed={t['seed']} "
                  f"{[m[0] for m in t['models']]} sim={res['context']['runtime_s']:.1f}s", flush=True)
    rows = [r for res in results for r in res["rows"]]
    contexts = [res["context"] for res in results]
    print()
    print_table(rows)

    practice = None
    if args.practice_compare or args.features:
        if args.practice_compare:
            practice = practice_feature_stats(models, args.seeds + [101, 102, 103])
        print_feature_comparison(rows, practice, args.features_model or models[-1][0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = OUT_DIR / f"{tag}.json"
    for r in rows:
        r.pop("_obs", None)
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), models=models, rows=rows, contexts=contexts,
                       practice_features=practice, total_wall_s=time.time() - t0), f, indent=1)
    print(f"\nsaved {out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
