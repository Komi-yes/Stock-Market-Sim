"""Charts for the presentation (Spanish labels), written to student_work/figures/*.png.

Usage: python -m student_work.make_figures
Reads the saved stage-2 results; a figure whose input file is missing is skipped with a note.
  fig1  practice holdout2: mean return per scenario (v1, v2, rule)
  fig2  ABIDES final gate, individual: mean return and max drawdown per template (v1 vs v2)
  fig3  ABIDES validation ablation: what each component of v2 contributes (mean return and max drawdown)
  fig4  ABIDES final gate, competitive: risk vs return of every agent in the same market
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from student_work.abides_report import load_rows, pool_of  # noqa: E402

OUT = Path("student_work/figures")
EVAL = Path("outputs/eval_gate_holdout2")
ABIDES = Path("outputs/abides_eval")
V2 = "sel9_rm3_nb"

# Validated categorical slots (dataviz reference palette, light mode): v2 blue, v1 orange, rule aqua.
COLOR = {"v2": "#2a78d6", "v1": "#eb6834", "rule": "#1baf7a", "other": "#8a8985"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCEN_ES = {"stable": "Estable", "trending": "Tendencia alcista", "volatile": "Volátil", "crash": "Crash",
           "downtrend": "Tendencia bajista", "low_liquidity": "Baja liquidez", "choppy": "Errático"}
TPL_ES = {"stable_day": "Día estable", "volatile_day": "Día volátil", "low_liquidity": "Baja liquidez",
          "stress_test": "Estrés", "trending_day": "Día con tendencia", "rmsc04": "Referencia (rmsc04)",
          "liquid_market": "Mercado líquido", "thin_market": "Mercado poco profundo",
          "stable_day+with_execution": "Presión compradora", "stable_day+with_execution:ASK": "Presión vendedora",
          "volatile_day+with_momentum": "Volátil + momentum"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def _hbars(ax, labels, series, fmt="{:+.2f}"):
    """Grouped horizontal bars. series: list of (legend, color, values)."""
    n = len(series)
    h = 0.8 / n
    y = np.arange(len(labels))
    for i, (name, color, vals) in enumerate(series):
        pos = y - 0.4 + h * (i + 0.5)
        ax.barh(pos, vals, height=h * 0.86, color=color, label=name, edgecolor=SURFACE, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(v, p, " " + fmt.format(v) + " ", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(y, labels)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.margins(x=0.12)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.grid(axis="y", visible=False)


def fig1():
    names = {"v1": "v1", V2: "v2", "rule-momentum": "rule"}
    files = {k: EVAL / f"{k}_holdout2.json" for k in names}
    if not all(f.exists() for f in files.values()):
        return print("fig1 skipped: holdout2 results missing")
    data = {names[k]: json.loads(f.read_text())["summary"] for k, f in files.items()}
    scen = list(SCEN_ES)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    _hbars(ax, [SCEN_ES[s] for s in scen], [
        ("v2 (modelo nuevo)", COLOR["v2"], [data["v2"]["by_scenario"][s]["return_pct"] for s in scen]),
        ("v1 (modelo anterior)", COLOR["v1"], [data["v1"]["by_scenario"][s]["return_pct"] for s in scen]),
        ("Regla de momentum", COLOR["rule"], [data["rule"]["by_scenario"][s]["return_pct"] for s in scen]),
    ])
    ax.set_xlabel("Rentabilidad media por episodio (%)")
    ax.set_title("Mercado de práctica, semillas holdout2 (280 episodios)", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "fig1_practica_holdout2.png")


def fig2():
    f = ABIDES / "gate_g11_ind.json"
    if not f.exists():
        return print("fig2 skipped: gate_g11_ind.json missing")
    pool = pool_of(load_rows([f]))
    tpls = [t for t in TPL_ES if t in pool["v1"]["by_template"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained")
    for ax, key, title, fmt in ((axes[0], "by_template", "Rentabilidad media (%)", "{:+.2f}"),
                                (axes[1], "max_dd_by_template", "Drawdown máximo (%)", "{:.1f}")):
        _hbars(ax, [TPL_ES[t] for t in tpls], [
            ("v2 (modelo nuevo)", COLOR["v2"], [pool[V2][key][t] for t in tpls]),
            ("v1 (modelo anterior)", COLOR["v1"], [pool["v1"][key][t] for t in tpls]),
        ], fmt=fmt)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8.5)
    n = pool["v1"]["n"] // max(1, len(tpls))
    fig.suptitle(f"ABIDES-NG, prueba final: 11 escenarios × {n} semillas nunca vistas", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    _save(fig, "fig2_abides_prueba_final.png")


def fig3():
    files = [ABIDES / "r7_val_g11.json", ABIDES / "r7_val2_g11.json"]
    if not all(f.exists() for f in files):
        return print("fig3 skipped: validation results missing")
    pool = pool_of(load_rows(files))
    steps = [("v1", "v1 (7 redes)"), ("sel9", "9 redes (v1 + 2 nuevas)"),
             ("sel9_rm3", "9 redes + guardia de órdenes en espera"), ("sel9_nb", "9 redes + libro neutralizado"),
             (V2, "v2 = 9 redes + ambas mejoras")]
    labels = [lab for _, lab in steps]
    colors = [COLOR["v2"] if k == V2 else (COLOR["v1"] if k == "v1" else COLOR["other"]) for k, _ in steps]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, layout="constrained")
    for ax, key, title, fmt in ((axes[0], "R", "Rentabilidad media (%)", "{:+.2f}"),
                                (axes[1], "max_DD", "Drawdown máximo (%)", "{:.1f}")):
        vals = [pool[k][key] for k, _ in steps]
        y = np.arange(len(steps))
        ax.barh(y, vals, height=0.6, color=colors, edgecolor=SURFACE, linewidth=1)
        for p, v in zip(y, vals):
            ax.text(v, p, " " + fmt.format(v) + " ", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK2)
        ax.set_yticks(y, labels)
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.margins(x=0.15)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=11)
    fig.suptitle("ABIDES-NG, validación (55 simulaciones por modelo): qué aporta cada pieza", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    _save(fig, "fig3_abides_ablacion.png")


def fig4():
    f = ABIDES / "gate_comp.json"
    if not f.exists():
        return print("fig4 skipped: gate_comp.json missing")
    pool = pool_of(load_rows([f]))
    label = {"v1": ("v1", COLOR["v1"]), V2: ("v2", COLOR["v2"]), "rule": ("Regla de momentum", COLOR["rule"]),
             "starter_s7": ("DQN inicial (semilla 7)", COLOR["other"]),
             "starter_s8": ("DQN inicial (semilla 8)", COLOR["other"])}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k, (lab, color) in label.items():
        if k not in pool:
            continue
        x, yv = pool[k]["max_DD"], pool[k]["R"]
        ax.scatter([x], [yv], s=90, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{lab}\n{yv:+.2f} %, DD máx {x:.0f} %", (x, yv), textcoords="offset points",
                    xytext=(8, 4), fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.margins(x=0.12, y=0.18)
    ax.set_xlabel("Drawdown máximo (%)  ← mejor")
    ax.set_ylabel("Rentabilidad media (%)  mejor →")
    ax.set_title("ABIDES-NG competitivo: todos los agentes en el mismo mercado", loc="left")
    _save(fig, "fig4_abides_competitivo.png")


if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4):
        fn()
