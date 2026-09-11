"""M1 figure: the 24-hour solve and the separability it rests on.

Panel (a) stacks dispatch in merit order, hour by hour. The top of the stack
is the load, because energy balance is an equality.

Panel (b) is the clearing price. It is a staircase in the same three offers as
M0 -- each hour prices independently off its own marginal unit, so the shape of
the price is the shape of the load passed through the supply stack.

Panel (c) is the milestone. Each hour's price is plotted against that hour's
load, over the single-hour sweep from M0, shaded light-to-dark by hour so the
day can be walked in place. All 24 points sit on the curve, so
lambda depends on D[t] and nothing else -- the 24-hour problem is 24 problems.
That stops being true at M6, when startup cost and min up/down time put
entries off the block diagonal and the points lift off the curve.

Every panel is computed from solves, not drawn.

Palette: categorical slots 1-3 of the reference data-viz palette (blue, orange,
aqua), validated all-pairs in light mode -- 0 FAIL, 1 WARN. The WARN is the
aqua's 2.74:1 contrast against the surface, which obliges a visible label
rather than color alone; all three bands are directly labeled.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.model.dispatch import marginal_unit, solve_dispatch, solve_dispatch_day

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# Sequential ramp for hour-of-day in panel (c). One hue, light to dark, so it
# survives grayscale and reads as an ordering rather than as four identities.
# Ordinal-validated: adjacent delta L >= 0.06, lightest step 2.06:1 -- 0 FAIL.
HOUR_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
# Stack order doubles as a lightness ramp so the bands survive grayscale.
ALPHA = [0.92, 0.66, 0.40]


def _steps(hours, values):
    """Hourly values are constant across [t, t+1). Draw them as such.

    Returns edges of length n+1 so step(where="post") closes the last block
    instead of dropping it.
    """
    return list(hours) + [hours[-1] + 1], list(values) + [values[-1]]


def _panel_dispatch(ax, c, Pmax, D, res, hours):
    floor = [0.0] * len(hours)
    for i, g in enumerate(sorted(c, key=lambda g: c[g])):
        top = [floor[j] + res["p"][g, t] for j, t in enumerate(hours)]
        xs, lo = _steps(hours, floor)
        _, hi = _steps(hours, top)
        ax.fill_between(xs, lo, hi, step="post", color=SERIES[i], alpha=ALPHA[i],
                        linewidth=0.0, zorder=2)
        # 2 px surface gap between segments, per the mark spec
        ax.step(xs, hi, where="post", color=SURFACE, lw=1.6, zorder=3)

        # direct label in the hour where this unit's band is thickest
        j = max(range(len(hours)), key=lambda j: top[j] - floor[j])
        if top[j] - floor[j] > 12.0:
            # the deep first band takes surface-colored text; the lighter
            # bands above it take ink -- white on the 0.40 aqua is unreadable
            label_ink = SURFACE if i == 0 else INK
            ax.text(hours[j] + 0.5, (top[j] + floor[j]) / 2, g, ha="center",
                    va="center", fontsize=9, color=label_ink, zorder=5)
        floor = top

    ax.set_ylim(0, sum(Pmax.values()))
    ax.set_ylabel("Dispatch (MW)", fontsize=9.5, color=INK_2)
    ax.set_title("(a)  Dispatch by hour, in merit order", fontsize=10.5,
                 color=INK, loc="left", pad=10)
    ax.legend(
        handles=[Patch(facecolor=SERIES[i], alpha=ALPHA[i], label=g)
                 for i, g in enumerate(sorted(c, key=lambda g: c[g]))],
        loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2,
        handlelength=1.6, handleheight=1.1, borderpad=0.2, ncol=3,
        columnspacing=1.2,
    )


def _panel_price(ax, c, Pmax, D, res, hours):
    xs, ys = _steps(hours, [res["lmbda"][t] for t in hours])
    ax.step(xs, ys, where="post", color=SERIES[0], lw=1.8, zorder=4)

    # one label per plateau, placed on the widest run at that price
    runs, start = [], 0
    for j in range(1, len(hours) + 1):
        if j == len(hours) or abs(ys[j] - ys[start]) > 1e-9:
            runs.append((start, j, ys[start]))
            start = j
    seen = {}
    for lo, hi, v in runs:
        if hi - lo > seen.get(round(v, 6), (0, None))[0]:
            seen[round(v, 6)] = (hi - lo, (lo + hi) / 2)
    for v, (_, mid) in seen.items():
        unit = next(g for g in c if abs(c[g] - v) < 1e-6)
        ax.text(mid, v + max(c.values()) * 0.035,
                rf"$\lambda$ = {v:.0f}   {unit}", ha="center", va="bottom",
                fontsize=8.5, color=INK_2, zorder=5,
                bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none",
                          pad=1.5))

    ax.set_ylim(0, max(c.values()) * 1.35)
    ax.set_ylabel(r"Clearing price $\lambda$ (\$/MWh)", fontsize=9.5, color=INK_2)
    ax.set_title("(b)  Clearing price by hour", fontsize=10.5, color=INK,
                 loc="left", pad=10)


def _panel_separability(ax, c, Pmax, D, res, hours):
    """Price against that hour's load, over the M0 single-hour sweep.

    Separability is a null result -- nothing happens in 23 hours -- and a
    panel of 23 zeros spends a third of the figure showing nothing. Stated
    positively it has evidence in it: if the hours were coupled, an hour's
    price would depend on the rest of the day and the 24 points would
    scatter off the single-hour curve. They do not, so lambda is a function
    of D[t] alone. At M6 these points lift off the curve.
    """
    total = sum(Pmax.values())
    sweep_d = [d * 1.0 for d in range(0, int(total) + 1)]
    sweep_l = [solve_dispatch(c, Pmax, d)["lmbda"] for d in sweep_d]
    ax.plot(sweep_d, sweep_l, color=INK_MUTED, lw=1.4, zorder=3,
            label="Single-hour sweep")

    # Hour is ordered data, so it gets the sequential ramp, not a hue per
    # hour. Reading light-to-dark walks the day: the pale points are the
    # overnight trough, the darkest are the late evening.
    cmap = LinearSegmentedColormap.from_list("hour", HOUR_RAMP)
    sc = ax.scatter([D[t] for t in hours], [res["lmbda"][t] for t in hours],
                    c=hours, cmap=cmap, norm=Normalize(hours[0], hours[-1]),
                    s=54, edgecolor=SURFACE, linewidth=1.4, zorder=4)

    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045, shrink=0.82,
                            ticks=[h for h in hours if h % 6 == 0])
    cb.set_label("Hour index t", fontsize=9, color=INK_2)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)
    cb.outline.set_visible(False)

    ax.set_xlim(0, total)
    ax.set_ylim(0, max(c.values()) * 1.35)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("Hourly load (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel(r"Clearing price $\lambda$ (\$/MWh)", fontsize=9.5, color=INK_2)
    ax.set_title("(c)  Clearing price against hourly load", fontsize=10.5,
                 color=INK, loc="left", pad=10)
    # The colorbar identifies the points; the legend still has to name the
    # two series, so the dot entry carries a mid-ramp swatch.
    ax.legend(
        handles=[
            Line2D([], [], color=INK_MUTED, lw=1.4, label="Single-hour sweep"),
            Line2D([], [], color=HOUR_RAMP[1], marker="o", ms=7, ls="none",
                   mec=SURFACE, mew=1.4, label="Joint 24-hour solve"),
        ],
        loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2,
        handlelength=1.6, borderpad=0.2,
    )
    return None


def figure_m1(c, Pmax, D):
    hours = sorted(D)
    res = solve_dispatch_day(c, Pmax, D)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        # Vertical gridlines land on the 6-hour ticks in (a) and (b) and on
        # the capacity breakpoints in (c) -- both give the eye a rhythm to
        # count against instead of tracing back to the axis.
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)
    for ax in axes[:2]:  # (c) is indexed by load, not by hour
        ax.set_xlim(hours[0], hours[-1] + 1)
        ax.set_xticks([h for h in hours if h % 6 == 0] + [hours[-1] + 1])
        ax.set_xlabel("Hour index t", fontsize=9.5, color=INK_2)

    _panel_dispatch(axes[0], c, Pmax, D, res, hours)
    _panel_price(axes[1], c, Pmax, D, res, hours)
    delta = _panel_separability(axes[2], c, Pmax, D, res, hours)
    fig.tight_layout(w_pad=2.6)
    return fig, res, delta


if __name__ == "__main__":
    import datetime as _dt
    import pathlib
    import shutil

    COST = {"g1": 20.0, "g2": 35.0, "g3": 80.0}
    PMAX = {"g1": 100.0, "g2": 100.0, "g3": 100.0}
    LOAD = dict(enumerate([
         62,  58,  55,  54,  57,  68,  92, 118, 146, 162, 171, 178,
        183, 186, 181, 176, 188, 214, 247, 263, 238, 192, 141,  88,
    ]))

    run_dir = pathlib.Path("runs") / _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    fig, res, delta = figure_m1(COST, PMAX, LOAD)
    out = run_dir / "m1_day_profile.png"
    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(run_dir / "m1_day_profile.pdf", facecolor=SURFACE, bbox_inches="tight")

    if pathlib.Path("configs/m1.yaml").exists():
        shutil.copy("configs/m1.yaml", run_dir / "config.yaml")

    energy = sum(LOAD.values())
    payment = sum(LOAD[t] * res["lmbda"][t] for t in LOAD)
    (run_dir / "results.txt").write_text(
        "\n".join(
            [f"hours: {len(LOAD)}",
             f"energy_mwh: {energy}",
             f"production_cost_usd: {res['cost']}",
             f"load_payment_usd: {payment}",
             f"load_weighted_price_usd_per_mwh: {payment / energy}"]
            + [f"lambda_t{t:02d}_usd_per_mwh: {res['lmbda'][t]}" for t in sorted(LOAD)]
            + [f"marginal_unit_t{t:02d}: {marginal_unit(res, COST, PMAX, t)}"
               for t in sorted(LOAD)]
        )
        + "\n"
    )
    print(f"wrote {out}")
