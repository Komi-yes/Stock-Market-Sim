from __future__ import annotations
import torch
from torch import nn
from shared.contract import OBSERVATION_SIZE, ACTION_SIZE

class QNetwork(nn.Module):
    """Starter network. You may redesign the internal architecture.

    The only fixed requirement is:
      input  : float32 tensor [batch, 16]
      output : float32 tensor [batch, 9]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Ensemble(nn.Module):
    """Averages the scores of several policies (e.g. the same recipe trained with different seeds)."""
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [m(x) for m in self.members]
        return torch.stack(outs, dim=0).mean(dim=0)


class BookNeutral(nn.Module):
    """Feeds the base policy the practice-market means of the order-book features (stage 2, EXP-011).

    x[5] book imbalance becomes 0 and x[6], x[7] (bid and ask depth) become 0.6077. ABIDES books show imbalance of
    about +/-0.6-0.85 against +/-0.15 in practice, so the members extrapolate there. Wrap the Q-ensemble with this
    and put SafePolicy outside it, so the guards still read the raw observation.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer("book", torch.tensor([[0.0, 0.6077, 0.6077]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(torch.cat([x[:, :5], self.book.expand(x.shape[0], 3), x[:, 8:]], dim=1))


class SafePolicy(nn.Module):
    """Wraps any policy and demotes actions that would break market constraints.

    Uses only the observation: x[:, 8] is inventory / 500 and x[:, 9] is cash / (2 * starting cash).
    An action is demoted below every other action when it would grow |inventory| past `max_inventory`,
    buy while cash is below `min_cash_ratio` of the starting cash, or grow a short position while cash is
    above `max_cash_ratio` of the starting cash. Short sales add cash, so the last rule caps short exposure
    at roughly (max_cash_ratio - 1) x starting cash whatever the share price is. ABIDES uses a ~10x higher price
    than the practice market, so the share cap alone allows heavy leverage there. HOLD is never demoted, so the
    argmax is always a legal action. Outputs stay finite.

    Drawdown brake (`brake_nav`, e.g. -0.05): while NAV return (x[:, 10] / 5) is below it, exposure-growing
    actions are demoted and a market order that shrinks the position is forced. A gap move against an open
    position cannot be stopped by order checks alone; this caps it (ABIDES stress_test showed insolvency without it).

    Broken-book guard (`max_spread_obs`, e.g. 0.5 = 100 bps): while the spread feature x[:, 4] is above it,
    exposure-growing actions are demoted. ABIDES stress_test books can empty out, with mid going to cents,
    and any position then gets marked at a meaningless price.

    Resting-order guard (`resting_mode`, stage 2): while x[:, 12] (open orders / 10) is above 0, the checks above
    cannot see what the resting limit orders will add when they fill. In ABIDES several resting LARGE orders filled
    together and took |inventory| past 100% of NAV. Mode 1 demotes exposure-growing limit actions, mode 2 every
    exposure-growing action, and mode 3 exposure-growing LARGE actions only. Mode 0 (default) is the v1 behaviour.
    """
    def __init__(self, base: nn.Module, max_inventory: float = 500.0, min_cash_ratio: float = 0.1,
                 max_cash_ratio: float = 1e9, brake_nav: float = -1e9, max_spread_obs: float = 2.0,
                 resting_mode: int = 0):
        super().__init__()
        self.base = base
        self.resting_mode = int(resting_mode)
        self.max_inventory = float(max_inventory)
        self.min_cash_ratio = float(min_cash_ratio)
        self.max_cash_ratio = float(max_cash_ratio)
        self.brake_nav = float(brake_nav)
        self.max_spread_obs = float(max_spread_obs)
        self.register_buffer("signed_qty", torch.tensor([[0.0, 10.0, -10.0, 50.0, -50.0, 10.0, -10.0, 50.0, -50.0]]))
        self.register_buffer("action_index", torch.arange(9, dtype=torch.float32).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.base(x)
        inv = x[:, 8:9] * 500.0
        new_inv = inv + self.signed_qty
        too_big = (new_inv.abs() > self.max_inventory) & (new_inv.abs() > inv.abs())
        cash = x[:, 9:10] * 2.0
        no_cash = (cash < self.min_cash_ratio) & (self.signed_qty > 0.0)
        over_short = (cash > self.max_cash_ratio) & (new_inv < 0.0) & (new_inv.abs() > inv.abs())
        bad = too_big | no_cash | over_short
        braking = (x[:, 10:11] * 0.2) < self.brake_nav
        broken = x[:, 4:5] > self.max_spread_obs
        bad = bad | ((braking | broken) & (new_inv.abs() > inv.abs()))
        if self.resting_mode > 0:
            grows_resting = (x[:, 12:13] > 0.0) & (new_inv.abs() > inv.abs())
            if self.resting_mode == 1:
                grows_resting = grows_resting & (self.action_index > 4.5)        # limit actions 5..8
            elif self.resting_mode == 3:
                grows_resting = grows_resting & (self.signed_qty.abs() > 25.0)   # LARGE actions
            bad = bad | grows_resting
        floor = q.min(dim=1, keepdim=True).values - 1.0
        out = torch.where(bad, floor, q)
        # Forced de-risking: market sell (long) or buy (short), large if |inv| >= 50.
        big = inv.abs() >= 50.0
        idx = torch.where(inv > 0.0, torch.where(big, torch.full_like(inv, 4.0), torch.full_like(inv, 2.0)),
                          torch.where(big, torch.full_like(inv, 3.0), torch.full_like(inv, 1.0)))
        force = braking & (inv.abs() > 0.5)
        hit = self.action_index == idx
        top = out.max(dim=1, keepdim=True).values + 1.0
        return torch.where(force & hit, top, out)
