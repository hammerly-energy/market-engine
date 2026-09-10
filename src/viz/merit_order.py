"""M0 figure: supply stack, market clearing, and the price-demand relation.

Panel (a) is the merit order at a single demand level. Areas are quantities:
solid blocks left of the demand line are production cost, the hatched wedge is
producer surplus, and the two together are the load payment.

Panel (b) sweeps demand across the fleet. Every point is a solve, not a drawn
line. Open markers sit at capacity breakpoints, where the dual is not unique.

Palette: categorical slots 1-3 of the reference data-viz palette (blue, orange,
aqua), validated all-pairs in light and dark mode.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.model.dispatch import solve_dispatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def merit_order(c, Pmax):
    """Generators cheapest first, with the MW interval each one occupies."""
    blocks, cum = [], 0.0
    for g in sorted(c, key=lambda g: c[g]):
        blocks.append({"gen": g, "cost": c[g], "lo": cum, "hi": cum + Pmax[g]})
        cum += Pmax[g]
    return blocks


def _marginal_unit(res, Pmax):
    for g, mw in res["p"].items():
        if 1e-6 < mw < Pmax[g] - 1e-6:
            return g
    return None


def _panel_stack(ax, c, Pmax, D, res):
    blocks = merit_order(c, Pmax)
    lmbda = res["lmbda"]
    total = sum(Pmax.values())
    ymax = max(c.values()) * 1.35
    marginal = _marginal_unit(res, Pmax)

    for i, b in enumerate(blocks):
        color = SERIES[i % len(SERIES)]
        run = res["p"][b["gen"]]
        edge = b["lo"] + run

        if edge < b["hi"]:
            ax.fill_between([edge, b["hi"]], 0, b["cost"], color=color, alpha=0.12,
                            edgecolor=SURFACE, linewidth=1.5, zorder=2)
        if run > 0:
            ax.fill_between([b["lo"], edge], 0, b["cost"], color=color, alpha=0.88,
                            edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if lmbda > b["cost"]:
                ax.fill_between([b["lo"], edge], b["cost"], lmbda, facecolor="none",
                                hatch="//", edgecolor=color, linewidth=0.0,
                                alpha=0.32, zorder=3)

        ax.plot([b["lo"], b["hi"]], [b["cost"]] * 2, color=color, lw=2, zorder=4,
                solid_capstyle="butt")

        name = f"{b['gen']} (marginal)" if b["gen"] == marginal else b["gen"]
        ax.text(b["lo"] + total * 0.013, b["cost"] + ymax * 0.022, name,
                ha="left", va="bottom", fontsize=9, color=INK_2, zorder=5)

    ax.plot([0, total], [lmbda, lmbda], color=INK, lw=1.4, ls=(0, (5, 3)), zorder=6)
    ax.plot([D, D], [0, lmbda], color=INK_MUTED, lw=1.4, ls=(0, (2, 3)), zorder=6)
    ax.plot([D], [lmbda], marker="o", ms=7, color=INK, zorder=7)

    ax.text(total * 0.985, lmbda + ymax * 0.022, rf"$\lambda$ = {lmbda:.0f}",
            ha="right", va="bottom", fontsize=9, color=INK, zorder=7)

    payment = D * lmbda
    ax.text(0.985, 0.965,
            f"production cost   {res['cost']:>7,.0f}\n"
            f"producer surplus  {payment - res['cost']:>7,.0f}\n"
            f"load payment      {payment:>7,.0f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK_2, family="monospace", linespacing=1.5)

    ax.set_xlim(0, total)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Cumulative capacity (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel(r"Offer price (\$/MWh)", fontsize=9.5, color=INK_2)
    ax.set_title("(a)  Supply stack at D = 150 MW", fontsize=10.5, color=INK,
                 loc="left", pad=10)
    ax.legend(
        handles=[
            Patch(facecolor=INK_MUTED, alpha=0.5, label="Production cost"),
            Patch(facecolor="none", hatch="//", edgecolor=INK_MUTED,
                  label="Producer surplus"),
            Patch(facecolor=INK_MUTED, alpha=0.12, label="Idle capacity"),
        ],
        loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2,
        handlelength=1.6, handleheight=1.1, borderpad=0.2,
    )


def _panel_price_curve(ax, c, Pmax, D, step=1.0):
    total = sum(Pmax.values())
    blocks = merit_order(c, Pmax)
    ymax = max(c.values()) * 1.35

    xs, ys, d = [], [], step
    while d <= total + 1e-9:
        xs.append(d)
        ys.append(solve_dispatch(c, Pmax, d)["lmbda"])
        d += step
    ax.plot(xs, ys, color=SERIES[0], lw=1.8, zorder=4, solid_capstyle="butt")

    for b in blocks[:-1]:
        nxt = next(x["cost"] for x in blocks if x["lo"] == b["hi"])
        ax.plot([b["hi"]], [solve_dispatch(c, Pmax, b["hi"])["lmbda"]],
                marker="o", ms=7, mfc=SURFACE, mec=SERIES[1], mew=1.8, zorder=6)
        ax.text(b["hi"] + total * 0.018, b["cost"] + (nxt - b["cost"]) * 0.25,
                rf"$\lambda \in$ [{b['cost']:.0f}, {nxt:.0f}]",
                fontsize=8.5, color=SERIES[1], ha="left", va="center")

    lmbda = solve_dispatch(c, Pmax, D)["lmbda"]
    ax.plot([D, D], [0, lmbda], color=INK_MUTED, lw=1.4, ls=(0, (2, 3)), zorder=3)
    ax.plot([D], [lmbda], marker="o", ms=7, color=INK, zorder=7)
    ax.text(D, lmbda + ymax * 0.045, rf"D = {D:.0f},  $\lambda$ = {lmbda:.0f}",
            fontsize=9, color=INK, ha="center", va="bottom")

    ax.set_xlim(0, total)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Demand (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel(r"Clearing price $\lambda$ (\$/MWh)", fontsize=9.5, color=INK_2)
    ax.set_title("(b)  Clearing price against demand", fontsize=10.5, color=INK,
                 loc="left", pad=10)


def figure_m0(c, Pmax, D):
    res = solve_dispatch(c, Pmax, D)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)

    _panel_stack(axes[0], c, Pmax, D, res)
    _panel_price_curve(axes[1], c, Pmax, D)
    fig.tight_layout(w_pad=3.0)
    return fig, res


if __name__ == "__main__":
    import datetime as _dt
    import pathlib
    import shutil

    COST = {"g1": 20.0, "g2": 35.0, "g3": 80.0}
    PMAX = {"g1": 100.0, "g2": 100.0, "g3": 100.0}
    DEMAND = 150.0

    run_dir = pathlib.Path("runs") / _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    fig, res = figure_m0(COST, PMAX, DEMAND)
    out = run_dir / "m0_merit_order.png"
    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(run_dir / "m0_merit_order.pdf", facecolor=SURFACE, bbox_inches="tight")

    if pathlib.Path("configs/m0.yaml").exists():
        shutil.copy("configs/m0.yaml", run_dir / "config.yaml")
    (run_dir / "results.txt").write_text(
        "\n".join(
            [f"demand_mw: {DEMAND}", f"lambda_usd_per_mwh: {res['lmbda']}",
             f"production_cost_usd: {res['cost']}",
             f"load_payment_usd: {DEMAND * res['lmbda']}"]
            + [f"dispatch_{g}_mw: {mw}" for g, mw in res["p"].items()]
        )
        + "\n"
    )
    print(f"wrote {out}")
