"""M0 visualization: the supply stack, where it clears, and what that costs.

Two panels, because M0 has two distinct ideas in it:

  A. The supply stack at one demand level. Area under the stack left of D is
     production cost. The rectangle between cost and lambda is producer surplus.
     Together they are the load payment.

  B. lambda as demand sweeps 0 -> total capacity. The price is a staircase, and
     the risers sit exactly at the capacity breakpoints where the dual stops
     being unique.

Palette: categorical slots 1-3 from the reference data-viz palette (blue /
orange / aqua), which validate all-pairs in both light and dark mode.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.model.dispatch import solve_dispatch

# --- design tokens -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1, 2, 3


def merit_order(c, Pmax):
    """Generators cheapest first, with the MW interval each one occupies."""
    blocks, cum = [], 0.0
    for g in sorted(c, key=lambda g: c[g]):
        blocks.append({"gen": g, "cost": c[g], "lo": cum, "hi": cum + Pmax[g]})
        cum += Pmax[g]
    return blocks


def _panel_stack(ax, c, Pmax, D, res):
    """Panel A: the supply stack, the demand line, and the clearing price."""
    blocks = merit_order(c, Pmax)
    lmbda = res["lmbda"]
    ymax = max(c.values()) * 1.35
    total = sum(Pmax.values())

    for i, b in enumerate(blocks):
        color = SERIES[i % len(SERIES)]
        run = res["p"][b["gen"]]
        edge = b["lo"] + run  # boundary between dispatched and idle capacity

        # idle capacity: present, priced, but not running
        if edge < b["hi"]:
            ax.fill_between([edge, b["hi"]], 0, b["cost"], color=color, alpha=0.13,
                            edgecolor=SURFACE, linewidth=1.5, zorder=2)
        # dispatched capacity: area = this unit's production cost
        if run > 0:
            ax.fill_between([b["lo"], edge], 0, b["cost"], color=color, alpha=0.90,
                            edgecolor=SURFACE, linewidth=1.5, zorder=3)
            # producer surplus: paid lambda, cost c -> the wedge in between
            if lmbda > b["cost"]:
                ax.fill_between([b["lo"], edge], b["cost"], lmbda, facecolor="none",
                                hatch="////", edgecolor=color, linewidth=0.0,
                                alpha=0.55, zorder=3)

        # offer step outline
        ax.plot([b["lo"], b["hi"]], [b["cost"]] * 2, color=color, lw=2, zorder=4,
                solid_capstyle="round")

        # direct label -- required, three slots sit under 3:1 on a light surface
        ax.text(b["lo"] + total * 0.012, b["cost"] + ymax * 0.03,
                f"{b['gen']}  \\${b['cost']:.0f}/MWh", ha="left", va="bottom",
                fontsize=9.5, color=INK_2, zorder=5)

    # clearing price
    ax.plot([0, D], [lmbda, lmbda], color=INK, lw=2, ls=(0, (5, 3)), zorder=6)
    ax.text(total * 0.012, lmbda + ymax * 0.02,
            f"$\\lambda$ = \\${lmbda:.0f}/MWh",
            ha="left", va="bottom", fontsize=10, color=INK, fontweight="bold", zorder=7)

    # demand
    ax.plot([D, D], [0, ymax * 0.93], color=INK_2, lw=2, ls=(0, (2, 2)), zorder=6)
    ax.text(D, ymax * 0.95, f"D = {D:.0f} MW", ha="center", va="bottom",
            fontsize=10, color=INK_2, fontweight="bold")

    # name the unit that sets the price
    marginal = [g for g, mw in res["p"].items() if 1e-6 < mw < Pmax[g] - 1e-6]
    if marginal:
        g = marginal[0]
        b = next(x for x in blocks if x["gen"] == g)
        ax.annotate(
            f"{g} is marginal\n(strictly between 0 and {Pmax[g]:.0f} MW)\nso {g} sets the price",
            xy=(b["lo"] + res["p"][g] / 2, b["cost"]),
            xytext=(b["lo"] - 6, ymax * 0.62),
            fontsize=9, color=INK, ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.5,
                            connectionstyle="arc3,rad=-0.2"),
        )

    ax.set_xlim(0, total)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Cumulative capacity (MW)", fontsize=10, color=INK_2)
    ax.set_ylabel("Offer price (\\$/MWh)", fontsize=10, color=INK_2)
    ax.set_title("A.  The supply stack clears where demand meets it  \u2014  $\\lambda$ is the dual on energy balance",
                 fontsize=12, color=INK, pad=14, loc="left")

    cost, payment = res["cost"], D * lmbda
    ax.legend(
        handles=[
            Patch(facecolor=INK_MUTED, alpha=0.55, label=f"Production cost   \\${cost:,.0f}"),
            Patch(facecolor="none", hatch="////", edgecolor=INK_MUTED,
                  label=f"Producer surplus  \\${payment - cost:,.0f}"),
            Patch(facecolor=INK_MUTED, alpha=0.13, label="Idle capacity"),
        ],
        loc="upper left", fontsize=9, frameon=False, labelcolor=INK_2,
        bbox_to_anchor=(0.01, 0.99),
    )


def _panel_price_curve(ax, c, Pmax, D, step=1.0):
    """Panel B: lambda swept across demand. Solved, not drawn by hand."""
    total = sum(Pmax.values())
    blocks = merit_order(c, Pmax)

    xs, ys = [], []
    d = step
    while d <= total + 1e-9:
        xs.append(d)
        ys.append(solve_dispatch(c, Pmax, d)["lmbda"])
        d += step
    ax.plot(xs, ys, color=SERIES[0], lw=2, zorder=4, solid_capstyle="round")

    # the risers are exactly the capacity breakpoints
    for b in blocks[:-1]:
        lo, hi = b["cost"], next(x["cost"] for x in blocks if x["lo"] == b["hi"])
        ax.plot([b["hi"]], [solve_dispatch(c, Pmax, b["hi"])["lmbda"]],
                marker="o", ms=9, mfc=SURFACE, mec=SERIES[1], mew=2, zorder=6)
        ax.annotate(f"$\\lambda \\in$ [{lo:.0f}, {hi:.0f}]\nno unique price",
                    xy=(b["hi"], (lo + hi) / 2), xytext=(b["hi"] + 12, (lo + hi) / 2),
                    fontsize=8.5, color=SERIES[1], va="center",
                    arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=1, alpha=0.6))

    lmbda = solve_dispatch(c, Pmax, D)["lmbda"]
    ax.plot([D, D], [0, lmbda], color=INK_2, lw=2, ls=(0, (2, 2)), zorder=3)
    ax.plot([D], [lmbda], marker="o", ms=9, color=INK, zorder=7)
    ax.text(D - 6, lmbda + 5, f"D = {D:.0f} MW\n$\\lambda$ = \\${lmbda:.0f}", fontsize=9,
            color=INK, ha="right", va="bottom", fontweight="bold")

    ax.set_xlim(0, total)
    ax.set_ylim(0, max(c.values()) * 1.35)
    ax.set_xlabel("Demand (MW)", fontsize=10, color=INK_2)
    ax.set_ylabel("Clearing price $\\lambda$  (\\$/MWh)", fontsize=10, color=INK_2)
    ax.set_title("B.  Price is a staircase; the risers are where the dual breaks down",
                 fontsize=12, color=INK, pad=14, loc="left")


def figure_m0(c, Pmax, D):
    res = solve_dispatch(c, Pmax, D)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    _panel_stack(axes[0], c, Pmax, D, res)
    _panel_price_curve(axes[1], c, Pmax, D)

    fig.suptitle("M0  ·  Single bus, single hour  ·  The price is the dual on energy balance",
                 fontsize=14, color=INK, x=0.035, ha="left", y=0.975, fontweight="bold")
    fig.text(0.035, 0.917,
             f"Load pays \\${D * res['lmbda']:,.0f}.   Generators are paid "
             f"\\${sum(res['p'].values()) * res['lmbda']:,.0f}.   "
             f"Equal, because one bus has no lines to congest \u2014 the settlement identity with zero congestion rent.",
             fontsize=10, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig, res


if __name__ == "__main__":
    import datetime as _dt
    import pathlib
    import shutil

    COST = {"g1": 20.0, "g2": 35.0, "g3": 80.0}
    PMAX = {"g1": 100.0, "g2": 100.0, "g3": 100.0}
    DEMAND = 150.0

    # convention: every run writes its config alongside its output
    run_dir = pathlib.Path("runs") / _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    fig, res = figure_m0(COST, PMAX, DEMAND)
    # publication figure: 300 dpi raster plus a vector copy
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
