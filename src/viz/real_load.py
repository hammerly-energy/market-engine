"""M2 figure: the same fleet, cleared against real ERCOT demand.

Panel (a) is the ingested series itself, in system GW on a UTC axis -- the
thing that crossed the boundary, before any scaling. The window is one local
ERCOT day resolved into UTC, so it starts at 05:00 UTC and the overnight
trough sits in the middle of the panel rather than at its edge.

Panel (b) stacks dispatch in merit order, as at M1. g1's band is a flat slab
across all 24 hours: it runs at 100 MW and never moves.

Panel (c) is the finding. Each hour's price is plotted against that hour's
load, over the same M0 single-hour sweep used in the M1 figure. The 24 hours
are confined to a narrow band on the right and never reach the $20 step: the
sweep is drawn across the full 0-300 MW range precisely so that the part of
the supply curve the real day never visits stays visible. ERCOT swings
1.53 : 1, and a linear rescale slides that band left or right but cannot
stretch it, so no choice of scale factor puts g1 on margin.

Every panel is computed from solves, not drawn.

Palette: categorical slots 1-3 of the reference data-viz palette (blue,
orange, aqua) for the merit-order bands, matching the M1 figure so the two
are readable side by side; the sequential blue ramp for hour-of-day in (c).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.model.dispatch import solve_dispatch, solve_dispatch_day

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
HOUR_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
ALPHA = [0.92, 0.66, 0.40]

def _steps(positions, values):
    """Hourly values are constant across [t, t+1). Draw them as such."""
    return list(positions) + [positions[-1] + 1], list(values) + [values[-1]]


def _utc_label(iso):
    """"2024-08-19T05:00:00+00:00" -> "05:00"."""
    return iso[11:16]


def _panel_system(ax, hours, system_mw, positions):
    """The ingested series, unscaled, in system GW."""
    gw = [system_mw[t] / 1000.0 for t in hours]
    xs, ys = _steps(positions, gw)
    ax.step(xs, ys, where="post", color=SERIES[0], lw=1.8, zorder=4)
    ax.fill_between(xs, 0, ys, step="post", color=SERIES[0], alpha=0.10,
                    linewidth=0.0, zorder=2)

    peak_j = max(positions, key=lambda j: gw[j])
    trough_j = min(positions, key=lambda j: gw[j])
    # Both annotations anchored the same way -- above the point they describe.
    for j, text in (
        (peak_j, f"Peak {gw[peak_j]:.1f}"),
        (trough_j, f"Trough {gw[trough_j]:.1f}"),
    ):
        ax.text(j + 0.5, gw[j] + 7.0, text, ha="center", va="bottom",
                fontsize=8.5, color=INK_2, zorder=5,
                bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none",
                          pad=1.5))

    ratio = max(gw) / min(gw)
    ax.text(0.5, 8.0, f"Peak / trough = {ratio:.2f}", ha="left", va="bottom",
            fontsize=8.5, color=INK_2, zorder=5,
            bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none", pad=1.5))

    ax.set_ylim(0, max(gw) * 1.30)
    ax.set_ylabel("ERCOT demand (GW)", fontsize=9.5, color=INK_2)
    ax.set_title("(a)  Ingested system demand", fontsize=10.5, color=INK,
                 loc="left", pad=10)


def _panel_dispatch(ax, c, Pmax, res, hours, positions):
    floor = [0.0] * len(hours)
    order = sorted(c, key=lambda g: c[g])
    for i, g in enumerate(order):
        top = [floor[j] + res["p"][g, t] for j, t in enumerate(hours)]
        xs, lo = _steps(positions, floor)
        _, hi = _steps(positions, top)
        ax.fill_between(xs, lo, hi, step="post", color=SERIES[i], alpha=ALPHA[i],
                        linewidth=0.0, zorder=2)
        ax.step(xs, hi, where="post", color=SURFACE, lw=1.6, zorder=3)

        # A flat band is equally thick everywhere and max() would return
        # hour 0, putting the label on the spine. Tie-break to mid-day.
        mid = (len(positions) - 1) / 2.0
        j = max(positions,
                key=lambda j: (round(top[j] - floor[j], 6), -abs(j - mid)))
        if top[j] - floor[j] > 12.0:
            label_ink = SURFACE if i == 0 else INK
            ax.text(j + 0.5, (top[j] + floor[j]) / 2, g, ha="center",
                    va="center", fontsize=9, color=label_ink, zorder=5)
        floor = top

    ax.set_ylim(0, sum(Pmax.values()))
    ax.set_ylabel("Dispatch (MW)", fontsize=9.5, color=INK_2)
    ax.set_title("(b)  Dispatch by hour, in merit order", fontsize=10.5,
                 color=INK, loc="left", pad=10)
    ax.legend(
        handles=[Patch(facecolor=SERIES[i], alpha=ALPHA[i], label=g)
                 for i, g in enumerate(order)],
        loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2,
        handlelength=1.6, handleheight=1.1, borderpad=0.2, ncol=3,
        columnspacing=1.2,
    )


def _panel_reach(ax, c, Pmax, D, res, hours, positions):
    """Price against hourly load, over the M0 single-hour sweep."""
    total = sum(Pmax.values())
    sweep_d = [float(d) for d in range(0, int(total) + 1)]
    sweep_l = [solve_dispatch(c, Pmax, d)["lmbda"] for d in sweep_d]
    ax.plot(sweep_d, sweep_l, color=INK_MUTED, lw=1.4, zorder=3)

    cmap = LinearSegmentedColormap.from_list("hour", HOUR_RAMP)
    sc = ax.scatter([D[t] for t in hours], [res["lmbda"][t] for t in hours],
                    c=positions, cmap=cmap, norm=Normalize(0, len(hours) - 1),
                    s=54, edgecolor=SURFACE, linewidth=1.4, zorder=5)

    ticks = [j for j in positions if j % 6 == 0]
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045, shrink=0.82,
                            ticks=ticks)
    cb.ax.set_yticklabels([_utc_label(hours[j]) for j in ticks])
    cb.set_label("Hour (UTC)", fontsize=9, color=INK_2)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)
    cb.outline.set_visible(False)

    # The band the real day occupies. Terse, and set clear of the points.
    lo, hi = min(D.values()), max(D.values())
    ax.text((lo + hi) / 2, max(c.values()) * 1.20,
            rf"$D_t \in$ [{lo:.0f}, {hi:.0f}]", ha="center", va="center",
            fontsize=8.5, color=INK_2, zorder=6,
            bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none", pad=1.5))

    ax.set_xlim(0, total)
    ax.set_ylim(0, max(c.values()) * 1.35)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("Hourly load (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel(r"Clearing price $\lambda$ (\$/MWh)", fontsize=9.5, color=INK_2)
    ax.set_title("(c)  Clearing price against hourly load", fontsize=10.5,
                 color=INK, loc="left", pad=10)
    ax.legend(
        handles=[
            Line2D([], [], color=INK_MUTED, lw=1.4, label="Single-hour sweep"),
            Line2D([], [], color=HOUR_RAMP[1], marker="o", ms=7, ls="none",
                   mec=SURFACE, mew=1.4, label="m2 ERCOT day"),
        ],
        loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2,
        handlelength=1.6, borderpad=0.2,
    )


def figure_m2(scenario, system_mw):
    """Returns (fig, res). system_mw is {iso_hour: unscaled system MW}."""
    c, Pmax, D = scenario.cost(), scenario.pmax(), scenario.demand()
    hours = scenario.hours
    positions = list(range(len(hours)))
    res = solve_dispatch_day(c, Pmax, D)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)

    # (a) and (b) are indexed by hour; ticks every 6 so the day has a rhythm.
    for ax in axes[:2]:
        ax.set_xlim(0, len(hours))
        ticks = [j for j in positions if j % 6 == 0] + [len(hours)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [_utc_label(hours[j]) for j in ticks[:-1]] + [_utc_label(hours[0])]
        )
        ax.set_xlabel("Hour (UTC)", fontsize=9.5, color=INK_2)

    _panel_system(axes[0], hours, system_mw, positions)
    _panel_dispatch(axes[1], c, Pmax, res, hours, positions)
    _panel_reach(axes[2], c, Pmax, D, res, hours, positions)
    fig.tight_layout(w_pad=2.6)
    return fig, res


if __name__ == "__main__":
    import datetime as _dt
    import json
    import pathlib
    import shutil

    from src.ingest import eia930
    from src.ingest.scenario import build_scenario
    from src.model.dispatch import marginal_unit

    CONFIG = "configs/m2.yaml"
    scenario = build_scenario(CONFIG)
    system, _ = eia930.load_demand(
        scenario.provenance["respondent"], scenario.provenance["local_date"]
    )
    system_mw = {t.isoformat(): float(v) for t, v in system.items()}

    run_dir = pathlib.Path("runs") / _dt.datetime.now(
        _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    fig, res = figure_m2(scenario, system_mw)
    out = run_dir / "m2_real_load.png"
    fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(run_dir / "m2_real_load.pdf", facecolor=SURFACE,
                bbox_inches="tight")

    shutil.copy(CONFIG, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(
        json.dumps(scenario.provenance, indent=2) + "\n"
    )
    eia930.write_processed(system, f"{scenario.provenance['respondent'].lower()}"
                                   f"_{scenario.provenance['local_date']}_demand")

    D, c, Pmax = scenario.demand(), scenario.cost(), scenario.pmax()
    energy = sum(D.values())
    payment = sum(D[t] * res["lmbda"][t] for t in scenario.hours)
    (run_dir / "results.txt").write_text(
        "\n".join(
            [f"hours: {len(scenario.hours)}",
             f"window_utc: {scenario.provenance['start_utc']}"
             f" .. {scenario.provenance['end_utc_exclusive']}",
             f"scale_factor: {scenario.provenance['scale_factor']}",
             f"energy_mwh: {energy}",
             f"production_cost_usd: {res['cost']}",
             f"load_payment_usd: {payment}",
             f"load_weighted_price_usd_per_mwh: {payment / energy}"]
            + [f"lambda_{_utc_label(t)}_usd_per_mwh: {res['lmbda'][t]}"
               for t in scenario.hours]
            + [f"marginal_unit_{_utc_label(t)}: "
               f"{marginal_unit(res, c, Pmax, t)}" for t in scenario.hours]
        )
        + "\n"
    )
    print(f"wrote {out}")
