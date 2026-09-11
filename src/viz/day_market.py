r"""M4 figures: one network, twenty-four hours.

Three figures, in the order the argument runs.

    figure_context          the system every later panel is about
                            (a) the five buses, six lines and two ratings
                            (b) the offer stack against the day's load range
      |
      v
    figure_day_prices       what the day looks like from the demand side
                            (a) system load, congested hours shaded
                            (b) the five LMPs, converging and separating
      |
      v
    figure_day_dispatch     what the fleet and the network actually do
                            (a) dispatch by unit, stacked in merit order
                            (b) the DE corridor against its 240 MW limit
      |
      v
    figure_day_settlement   the acceptance criterion, drawn
                            (a) payments, revenue and congestion rent
                            (b) the settlement residual, hour by hour

The prices figure is the argument for M4 in one image. M3 could only ever show
one column of it: a single hour is either congested or it is not. Across a day
the DE corridor binds in the morning and releases at night, so the five prices
sit on top of each other for seven hours and fan out for the other seventeen.
The gap between those two states is what M4 exists to produce.

Nothing here is hand-placed. Every number comes from clear() on
configs/m4.yaml -- the same call the tests make -- so a figure that disagrees
with the model is a figure that cannot be drawn.

Panel (b) of the settlement figure is the M4 goal made visible: the residual of

    sum(load payments) - sum(generator revenue) == sum_l mu[l] * limit[l]

in every hour separately, against the 1e-6 tolerance the tests assert. It is
drawn on a log axis because the interesting claim is about ORDERS of magnitude
-- the residuals land around 1e-12, six decades under tolerance, and a linear
axis would draw all of them flat on zero and prove nothing.

Palette: Okabe-Ito, assigned to buses by position in configs/m4.yaml and never
cycled, so bus A is the same hue here as in the M3 figures. Generators take
their bus's hue, because a unit's location is what the network cares about.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# The network panel is imported rather than redrawn. It is the SAME network as
# M3 -- same buses, same reactances, same two ratings -- and a second copy of
# the drawing code would be a second thing to keep in step with configs/. If
# the M3 figure and the M4 context figure ever disagreed about the topology,
# one of them would be lying and neither would say which.
from src.viz.network import _ink_on, _panel_network

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"

# Okabe-Ito, in config bus order. Identity, not a ramp. Matches src/viz/network.py.
BUS_HUE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"]

# The congested band's fill. Kept very light: it sits behind data lines and
# must not compete with them, and it has to survive a grayscale print as a
# shade rather than as a block.
SHADE = "#e8e4da"

# A 24-hour series carries ticks and gridlines every 6 hours, so the day has a
# visible rhythm and the evening ramp can be found without counting.
HOUR_TICKS = [0, 6, 12, 18, 23]


def _bare(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)
    ax.set_facecolor(SURFACE)


def _title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=10.5, color=INK, pad=8)


def _hours(ax, hours):
    ax.set_xlim(min(hours) - 0.4, max(hours) + 0.4)
    ax.set_xticks(HOUR_TICKS)
    ax.set_xlabel("Hour", fontsize=9.5, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _display(name):
    """Config key -> display name. park_city -> Park City."""
    return name.replace("_", " ").title()


def _lighten(hex_color, amount):
    """Mix a hue toward the surface. amount 0 keeps it, 1 erases it."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    sr, sg, sb = (int(SURFACE[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c, s_: round(c + (s_ - c) * amount)
    return "#%02x%02x%02x" % (mix(r, sr), mix(g, sg), mix(b, sb))


def _unit_hues(fleet, buses):
    """{generator: colour}. Hue by BUS, lightness by unit within that bus.

    Colour carries location, because location is what the network prices. But
    case5 puts Alta and Park City at the same bus, and two identical blues
    stacked on each other read as one band with a stray white line through it.
    So the second and subsequent units at a bus are mixed toward the surface,
    which keeps "bus A is blue" true while making the two units separable --
    one hue, light to dark, exactly as a sequential scale should be.
    """
    base = {b: BUS_HUE[i] for i, b in enumerate(buses)}
    order = sorted(fleet, key=lambda g: g["cost"])
    seen, out = {}, {}
    for g in order:
        n = seen.get(g["bus"], 0)
        seen[g["bus"]] = n + 1
        out[g["name"]] = _lighten(base[g["bus"]], 0.0 if n == 0 else 0.34 * n)
    return out


def _bbox():
    """For any annotation that may land on a gridline or a data line."""
    return dict(facecolor=SURFACE, alpha=0.75, edgecolor="none", pad=1.5)


def _congested_spans(hours, congested):
    """Contiguous runs of congested hours, as (start, end) in axis coordinates.

    Drawn as spans rather than as one shaded patch per hour so the band reads
    as a period of the day, and so its edges fall between hours instead of
    through them.
    """
    spans, run = [], []
    for t in hours:
        if t in congested:
            run.append(t)
        elif run:
            spans.append((run[0] - 0.5, run[-1] + 0.5))
            run = []
    if run:
        spans.append((run[0] - 0.5, run[-1] + 0.5))
    return spans


def _shade_congested(ax, hours, congested, label_y=None, label=True):
    spans = _congested_spans(hours, congested)
    for x0, x1 in spans:
        ax.axvspan(x0, x1, color=SHADE, lw=0, zorder=0)
    if label and spans and label_y is not None:
        x0, x1 = spans[0]
        ax.text((x0 + x1) / 2, label_y, "DE binding", ha="center", va="top",
                fontsize=8.5, color=INK_2, bbox=_bbox(), zorder=6)
    return spans


# ------------------------------------------------------------ figure 1

def _panel_load(ax, hours, load, congested):
    lo, hi = min(load.values()), max(load.values())
    pad = (hi - lo) * 0.28
    ax.set_ylim(lo - pad * 0.5, hi + pad)

    _shade_congested(ax, hours, congested, label_y=hi + pad * 0.92)

    ax.plot(hours, [load[t] for t in hours], color=INK, lw=1.6, zorder=4)
    ax.scatter(hours, [load[t] for t in hours], s=16, color=INK,
               edgecolor=SURFACE, linewidth=0.9, zorder=5)

    peak = max(hours, key=lambda t: load[t])
    ax.annotate(f"{load[peak]:,.0f} MW", (peak, load[peak]),
                textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=8.5, color=INK, bbox=_bbox(), zorder=6)
    trough = min(hours, key=lambda t: load[t])
    ax.annotate(f"{load[trough]:,.0f} MW", (trough, load[trough]),
                textcoords="offset points", xytext=(0, -14), ha="center",
                fontsize=8.5, color=INK_2, bbox=_bbox(), zorder=6)

    _bare(ax)
    _hours(ax, hours)
    ax.set_ylabel("System load (MW)", fontsize=9.5, color=INK_2)
    _title(ax, "a", "System load across the day")


def _panel_lmp(ax, hours, buses, lmp, congested):
    values = [lmp[b, t] for b in buses for t in hours]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.22
    ax.set_ylim(lo - pad, hi + pad)

    _shade_congested(ax, hours, congested, label_y=hi + pad * 0.96)

    for i, b in enumerate(buses):
        ax.plot(hours, [lmp[b, t] for t in hours], color=BUS_HUE[i], lw=1.7,
                zorder=4, label=b)

    # Direct labels in the middle of the congested run, where the five series
    # are furthest apart. Labelling at the right edge would stack all five on
    # one point -- at hour 23 nothing binds and every price is $10.
    anchor = sorted(congested)[len(congested) // 2] if congested else hours[-1]
    for i, b in enumerate(buses):
        ax.text(anchor, lmp[b, anchor], f" {b} ", ha="center", va="center",
                fontsize=8.5, color=BUS_HUE[i], bbox=_bbox(), zorder=6)

    _bare(ax)
    _hours(ax, hours)
    ax.set_ylabel("LMP ($/MWh)", fontsize=9.5, color=INK_2)
    _title(ax, "b", "Locational prices at each bus")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=5,
                    handlelength=1.2, columnspacing=1.0, borderpad=0.2)
    for text in leg.get_texts():
        text.set_color(INK_2)


def figure_day_prices(hours, buses, load, lmp, congested):
    """Load and the five prices, side by side, over one day."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE,
                             constrained_layout=True)
    _panel_load(axes[0], hours, load, congested)
    _panel_lmp(axes[1], hours, buses, lmp, congested)
    return fig


# ------------------------------------------------------------ figure 2

def _panel_stack(ax, hours, fleet, dispatch, buses, congested):
    """Dispatch stacked in merit order, cheapest at the bottom.

    Merit order, not config order: the stack is an argument about which units
    the market reaches for first, so the bands have to be in the order the
    market would reach for them.
    """
    order = sorted(fleet, key=lambda g: g["cost"])
    hue = _unit_hues(fleet, buses)

    bands, bottom = {}, np.zeros(len(hours))
    for g in order:
        band = np.array([dispatch[g["name"], t] for t in hours])
        ax.fill_between(hours, bottom, bottom + band, color=hue[g["name"]],
                        lw=0.8, edgecolor=SURFACE, zorder=3)
        bands[g["name"]] = (bottom.copy(), band)
        bottom = bottom + band

    top = bottom.max() * 1.30
    ax.set_ylim(0, top)

    # Direct-label a band only where it is thick enough ON THE AXIS to hold
    # text. Alta is 40 MW in a 1,000 MW stack: measured against its own height
    # it looks labellable, and measured against the page it is a sliver. The
    # legend is what carries the thin ones.
    mid = len(hours) // 2
    for g in order:
        base, band = bands[g["name"]]
        if band[mid] > top * 0.09:
            ax.text(hours[mid], base[mid] + band[mid] / 2,
                    f"{_display(g['name'])} ({g['bus']})", ha="center",
                    va="center", fontsize=8.5, color=SURFACE, zorder=5)

    _bare(ax)
    _hours(ax, hours)
    _shade_congested(ax, hours, congested, label_y=top * 0.985)
    ax.set_ylabel("Dispatch (MW)", fontsize=9.5, color=INK_2)
    _title(ax, "a", "Dispatch by unit, stacked in merit order")

    # Every unit in the fleet, in merit order, including any that never runs.
    # Sundance at $40 is never reached on this day, and a legend entry with no
    # band next to it says that more clearly than its absence would.
    handles = [Patch(facecolor=hue[g["name"]], edgecolor=SURFACE, lw=0.8,
                     label=f"{_display(g['name'])} ({g['bus']})") for g in order]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=8.5, ncol=2, handlelength=1.1,
                    columnspacing=1.0, borderpad=0.2, labelspacing=0.35)
    for text in leg.get_texts():
        text.set_color(INK_2)


def _panel_corridor(ax, hours, flow, limit, congested, line="DE"):
    f = [abs(flow[line, t]) for t in hours]
    ax.set_ylim(0, max(max(f), limit) * 1.26)
    _shade_congested(ax, hours, congested, label_y=max(max(f), limit) * 1.24)

    ax.axhline(limit, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.text(0.2, limit, f" Limit {limit:,.0f} MW", ha="left", va="bottom",
            fontsize=8.5, color=INK_2, bbox=_bbox(), zorder=6)

    ax.plot(hours, f, color=INK, lw=1.7, zorder=4)
    ax.scatter(hours, f, s=16, color=INK, edgecolor=SURFACE,
               linewidth=0.9, zorder=5)

    _bare(ax)
    _hours(ax, hours)
    ax.set_ylabel(f"{line} flow (MW)", fontsize=9.5, color=INK_2)
    _title(ax, "b", f"The {line} corridor against its limit")


def figure_day_dispatch(hours, buses, fleet, dispatch, flow, limit, congested):
    """Who runs, and the one line that stops them."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), facecolor=SURFACE,
                             constrained_layout=True)
    _panel_stack(axes[0], hours, fleet, dispatch, buses, congested)
    _panel_corridor(axes[1], hours, flow, limit, congested)
    return fig


# ------------------------------------------------------------ figure 3

def _panel_money(ax, hours, settlement, congested):
    pay = [settlement[t]["payments"] for t in hours]
    rev = [settlement[t]["revenue"] for t in hours]
    rent = [settlement[t]["mu_times_limit"] for t in hours]

    top = max(pay) * 1.2
    ax.set_ylim(0, top)
    _shade_congested(ax, hours, congested, label_y=top * 0.985)

    series = [
        ("Load payment", pay, INK, 1.8),
        ("Generator revenue", rev, BUS_HUE[2], 1.6),
        ("Congestion rent", rent, BUS_HUE[1], 1.6),
    ]
    for label, ys, color, lw in series:
        ax.plot(hours, ys, color=color, lw=lw, zorder=4, label=label)

    _bare(ax)
    _hours(ax, hours)
    ax.set_ylabel("Money ($/h)", fontsize=9.5, color=INK_2)
    _title(ax, "a", "Payments, revenue and congestion rent")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                    handlelength=1.4, borderpad=0.2)
    for text in leg.get_texts():
        text.set_color(INK_2)


def _panel_residual(ax, hours, settlement, congested, tol=1e-6):
    """|residual| per hour on a log axis, against the asserted tolerance.

    An exact zero has no place on a log axis, so it is drawn on the floor of
    the plot and counted in the annotation instead of being dropped -- an hour
    silently missing from this panel would be the one hour nobody checked.

    The congested band is shaded here too, and it earns its place: the hours
    with a non-zero residual are almost all inside it. That is the expected
    result rather than a worry. An uncongested hour settles as one price times
    one quantity and the arithmetic is exact; a congested hour multiplies five
    different LMPs by five loads and adds a mu times a limit, so it accumulates
    a few ulps. Error tracking the amount of arithmetic done is floating point
    behaving normally. Error tracking anything ELSE -- the hour of day, the
    load level, one particular bus -- would not be.
    """
    resid = [abs(settlement[t]["residual"]) for t in hours]
    nonzero = [r for r in resid if r > 0]
    floor = min(nonzero) / 20 if nonzero else tol / 1e9
    ax.set_yscale("log")
    ax.set_ylim(floor, tol * 40)
    _shade_congested(ax, hours, congested, label_y=tol * 30)

    ax.axhline(tol, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.text(0.2, tol * 1.6, f" Tolerance {tol:.0e} ", ha="left", va="bottom",
            fontsize=8.5, color=INK_2, bbox=_bbox(), zorder=6)

    exact = [t for t, r in zip(hours, resid) if r == 0]
    ax.scatter([t for t, r in zip(hours, resid) if r > 0],
               [r for r in resid if r > 0],
               s=26, color=INK, edgecolor=SURFACE, linewidth=0.9, zorder=5)
    if exact:
        ax.scatter(exact, [floor * 1.6] * len(exact), s=26, marker="v",
                   color=INK_MUTED, edgecolor=SURFACE, linewidth=0.9, zorder=5)
        ax.text(22.6, floor * 2.6, f"{len(exact)} hours exactly zero",
                ha="right", va="bottom", fontsize=8.5, color=INK_2,
                bbox=_bbox(), zorder=6)

    _bare(ax)
    _hours(ax, hours)
    ax.set_ylabel("|Residual| ($/h)", fontsize=9.5, color=INK_2)
    _title(ax, "b", "Settlement identity residual, by hour")


def figure_day_settlement(hours, settlement, congested):
    """The money, and the identity that has to balance it in every hour."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE,
                             constrained_layout=True)
    _panel_money(axes[0], hours, settlement, congested)
    _panel_residual(axes[1], hours, settlement, congested)
    return fig


# ------------------------------------------------------------ figure 0

def _panel_offer_band(ax, buses, fleet, trough, peak):
    """The merit order, with the band of load the day sweeps across.

    This is the stack with NO network -- the order a single-bus solve would
    follow. It is drawn first and on its own precisely so the later figures
    have something to contradict: the day never actually dispatches this way,
    because the DE corridor stops Brighton short of what its offer entitles it
    to. The gap between this panel and panel (a) of the dispatch figure is the
    whole of M3 and M4 in one comparison.

    M3's version of this panel marks a single load. A day has a range, so the
    line becomes a band, and what the band shows is which offers the day
    reaches into at all. Sundance sits entirely to the right of it and never
    runs -- not because the network forbids it, but because the day never gets
    that expensive.
    """
    hue = dict(zip(buses, BUS_HUE))
    order = sorted(fleet, key=lambda g: g["cost"])
    top = max(g["cost"] for g in fleet) * 1.3

    # Two rules and a label, NOT a shaded span. SHADE means "DE binding" in
    # every other panel of this set, and a reader flipping between figures
    # should never have to ask which sense a tint is being used in. A region
    # between two dashed rules reads as a range without borrowing the token.
    left = 0.0
    for g in order:
        ax.bar(left, g["cost"], width=g["pmax"], align="edge", bottom=0,
               color=hue[g["bus"]], edgecolor=SURFACE, linewidth=0.8, zorder=2)
        # Upright inside a narrow block. The type scale is fixed, so a
        # collision is resolved by turning or moving the label, never by
        # shrinking it.
        ax.text(left + g["pmax"] / 2, g["cost"] / 2,
                f"{_display(g['name'])} ({g['bus']})",
                ha="center", va="center", rotation=90 if g["pmax"] < 250 else 0,
                fontsize=8.5, color=_ink_on(hue[g["bus"]]), zorder=5)
        left += g["pmax"]

    for x in (trough, peak):
        ax.axvline(x, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.annotate("", xy=(trough, top * 0.86), xytext=(peak, top * 0.86),
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8), zorder=5)
    ax.text((trough + peak) / 2, top * 0.88,
            f"Load {trough:,.0f}-{peak:,.0f} MW", ha="center", va="bottom",
            fontsize=8.5, color=INK, bbox=_bbox(), zorder=6)

    _bare(ax)
    ax.set_xlim(0, left)
    ax.set_ylim(0, top)
    ax.set_xlabel("Cumulative capacity (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Offer ($/MWh)", fontsize=9.5, color=INK_2)


def figure_context(buses, branches, slack, fleet, peak_load, trough_mw, peak_mw):
    """The system the rest of the day's figures are about.

    peak_load is {bus: MW} at the peak hour, so panel (a) shows the network
    loaded at its hardest moment -- the hour where the rating actually bites.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.0, 1.15]},
                             constrained_layout=True)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    capacity = {b: sum(g["pmax"] for g in fleet if g["bus"] == b) for b in buses}
    _panel_network(axes[0], buses, branches, slack, capacity, peak_load)
    _title(axes[0], "a", "Network, capacity and peak load (MW)")

    _panel_offer_band(axes[1], buses, fleet, trough_mw, peak_mw)
    _title(axes[1], "b", "Offer stack and the day's load range")
    return fig


if __name__ == "__main__":
    from datetime import datetime, timezone
    from pathlib import Path

    from src.ingest.scenario import build_scenario, load_config
    from src.model.clearing import binding_lines, clear

    root = Path(__file__).parents[2]
    config_path = root / "configs" / "m4.yaml"
    config = load_config(config_path)

    # One call. Everything on these pages comes out of it, so a figure that
    # disagrees with the model is a figure that cannot be drawn.
    scenario = build_scenario(config_path)
    day = clear(scenario)

    hours = day["hours"]
    buses = day["buses"]
    fleet = [
        {"name": n, "bus": s["bus"], "cost": float(s["cost_usd_per_mwh"]),
         "pmax": float(s["pmax_mw"])}
        for n, s in config["fleet"].items()
    ]
    D = scenario.demand_by_bus()
    load = {t: sum(D[b][t] for b in buses) for t in hours}
    congested = {t for t in hours if binding_lines(day, t)}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(config_path.read_text())

    # ------------------------------------------------------------- report
    rows = [
        f"{config['name']}  --  PJM 5-bus, {len(hours)} hours",
        "",
        f"  slack {day['slack']}   buses {len(buses)}   lines {len(day['lines'])}"
        f"   peak {max(load.values()):,.0f} MW   trough {min(load.values()):,.0f} MW",
        f"  congested hours {len(congested)} of {len(hours)}",
        "",
        "  Hour      Load   Binding   lambda  " + "  ".join(f"{b:>7}" for b in buses)
        + "     Residual",
    ]
    for t in hours:
        bind = ",".join(sorted(binding_lines(day, t))) or "--"
        rows.append(
            f"  {t:>4}  {load[t]:8,.0f}   {bind:>7}  {day['lmbda'][t]:7.2f}  "
            + "  ".join(f"{day['lmp'][b, t]:7.2f}" for b in buses)
            + f"   {day['settlement'][t]['residual']:10.2e}"
        )

    worst = max(hours, key=lambda t: abs(day["settlement"][t]["residual"]))
    rows += [
        "",
        "  Settlement identity",
        f"    Worst hour             {worst}",
        f"    Worst |residual|       {abs(day['settlement'][worst]['residual']):.2e} $/h",
        f"    Tolerance asserted     1.00e-06 $/h",
        "",
        f"  Day production cost      {day['cost']:,.2f} $",
        "",
    ]
    report = "\n".join(rows)
    print(report)
    (out / "results.txt").write_text(report)

    # ------------------------------------------------------------ figures
    from src.model.inputs import Branch

    branches = [
        Branch(name=n, from_bus=b["from"], to_bus=b["to"],
               reactance_pu=float(b["reactance_pu"]), limit_mw=float(b["limit_mw"]))
        for n, b in config["network"]["branches"].items()
    ]
    peak_hour = max(hours, key=lambda t: load[t])
    peak_load = {b: D[b][peak_hour] for b in buses}

    for name, fig in [
        ("m4_0_context", figure_context(
            buses, branches, day["slack"], fleet, peak_load,
            min(load.values()), max(load.values()))),
        ("m4_1_prices", figure_day_prices(hours, buses, load, day["lmp"], congested)),
        ("m4_2_dispatch", figure_day_dispatch(
            hours, buses, fleet, day["dispatch"], day["flows"],
            day["limits"]["DE"], congested)),
        ("m4_3_settlement", figure_day_settlement(hours, day["settlement"], congested)),
    ]:
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
        print(out / f"{name}.png")