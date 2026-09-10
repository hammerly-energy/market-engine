r"""M3 figures: inputs, processing, results.

Three figures, in the order the code runs.

    figure_inputs        what configs/m3.yaml declares
                         (a) network, capacity and load
                         (b) the fleet as an offer stack
      |
      v
    figure_processing    what src/network/topology.py builds
                         (a) incidence A      pure topology, no reactance yet
                         (b) susceptance B    reactance arrives, rows sum to 0
      |
      v
    figure_results       what src/network/ptdf.py returns
                         (a) the PTDF matrix, slack column identically zero
                         (b) the one row that binds this case
      |
      v
    figure_flows         PTDF applied to the merit-order dispatch
                         (a) where the power goes
                         (b) which line cannot carry it

The flows figure is the argument for M3 in one image. The dispatch it draws
is M0's single-bus solve -- the cheapest way to serve 1000 MW if transmission
were free -- pushed through the shift factors to see what it would ask the
network to do. DE is asked for 283 MW across a 240 MW line, so the answer is
infeasible and the LP in Stage 3 has to find a more expensive one. Nothing
here is hand-placed: the dispatch comes from solve_dispatch and the flows
from ptdf.

Panel (b) of the inputs figure is the fleet with no network at all -- the
merit order a single-bus solve would follow. The whole of M3 is the gap
between that stack and what the network permits, so it is drawn first and on
its own.

Matrix panels print their values, because a diverging fill is the one thing
that does not survive a grayscale print.

Palette: Okabe-Ito, assigned to buses by position in configs/m3.yaml and never
cycled, so bus A is the same hue in every panel and every later figure. The
matrices use its blue and vermillion as a diverging pair about a neutral
surface midpoint. Two hues, never a rainbow.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.network.ptdf import ptdf
from src.network.topology import b_bus, incidence

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e7e6e2"

# Okabe-Ito, in config bus order. Identity, not a ramp.
BUS_HUE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"]

DIVERGING = LinearSegmentedColormap.from_list(
    "flow", ["#104281", "#5598e7", SURFACE, "#f2a07a", "#a8380f"]
)

# Coordinates matching the ASCII sketch in configs/m3.yaml, so the figure and
# the config show the same network in the same orientation. The second entry
# is where a bus's annotation hangs, chosen once so nothing collides.
LAYOUT = {
    "A": ((0.00,  0.00), ("right", "center")),
    "B": ((0.00,  1.00), ("right", "center")),
    "C": ((1.25,  1.00), ("left",  "center")),
    "D": ((1.25,  0.00), ("left",  "center")),
    "E": ((0.62, -1.05), ("center", "top")),
}


def _bare(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)


def _title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=10.5, color=INK, pad=8)


def _display(name):
    """Config key -> display name. park_city -> Park City.

    configs/m3.yaml keys are snake_case because they are dict keys; its prose
    calls the units Brighton and Park City. Figures follow the prose.
    """
    return name.replace("_", " ").title()


def _cell(value, fmt):
    """Exact zero prints as a bare 0. A signed +0.00 reads as a rounded value."""
    return "0" if value == 0 else fmt.format(value)


def _ink_on(fill):
    """Contrasting ink for text sitting on a solid fill. Colour, never weight."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return SURFACE if luma < 0.55 else INK


def _ink_for(value, norm):
    """Contrasting ink for text sitting on the rendered fill, never bold."""
    return SURFACE if abs(norm(value) - 0.5) > 0.34 else INK


def _matrix(ax, M, rows, cols, fmt="{:.0f}", vlim=None):
    span = vlim if vlim is not None else max(abs(M).max(), 1e-9)
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, _cell(M[i, j], fmt), ha="center", va="center",
                    fontsize=8.5, color=_ink_for(M[i, j], norm))

    ax.set_xticks(range(len(cols)), cols, fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(rows)), rows, fontsize=8.5, color=INK_2)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.4)
    ax.tick_params(which="both", length=0)


# --------------------------------------------------------------- inputs ----

def _panel_network(ax, buses, branches, slack, capacity, load):
    hue = dict(zip(buses, BUS_HUE))

    for br in branches:
        (x0, y0), _ = LAYOUT[br.from_bus]
        (x1, y1), _ = LAYOUT[br.to_bus]
        rated = np.isfinite(br.limit_mw)
        ax.plot([x0, x1], [y0, y1], color=INK_2 if rated else INK_MUTED,
                linewidth=2.2 if rated else 1.2, solid_capstyle="round", zorder=1)

        # Reactance always; a thermal rating only where one exists, and named,
        # because a bare "400 MW" beside a reactance says nothing about which
        # quantity is capped.
        label = f"x = {br.reactance_pu:g}"
        if rated:
            label += f"\nRated {br.limit_mw:.0f} MW"
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center",
                fontsize=8.5, color=INK if rated else INK_2,
                bbox=dict(facecolor=SURFACE, alpha=0.9, edgecolor="none", pad=1.5),
                zorder=3)

    for name in buses:
        (x, y), (ha, va) = LAYOUT[name]
        ax.scatter([x], [y], s=430, zorder=4, facecolor=hue[name],
                   edgecolor=SURFACE, linewidth=1.4)
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5,
                color=SURFACE, zorder=5)

        lines = []
        if capacity.get(name):
            lines.append(f"Gen {capacity[name]:.0f}")
        if load.get(name):
            lines.append(f"Load {load[name]:.0f}")
        if name == slack:
            lines.append("Slack")
        dx = {"right": -0.14, "left": 0.14, "center": 0.0}[ha]
        dy = {"center": 0.0, "top": -0.16}[va]
        ax.text(x + dx, y + dy, "\n".join(lines), ha=ha, va=va,
                fontsize=8.5, color=INK_2, linespacing=1.35, zorder=5)

    ax.set_xlim(-0.95, 2.2)
    ax.set_ylim(-1.75, 1.35)
    ax.set_axis_off()


def _panel_offer_stack(ax, buses, fleet, total_load):
    """The merit order with no network. What a single-bus solve would do."""
    hue = dict(zip(buses, BUS_HUE))
    order = sorted(fleet, key=lambda g: g["cost"])

    left = 0.0
    for g in order:
        ax.bar(left, g["cost"], width=g["pmax"], align="edge", bottom=0,
               color=hue[g["bus"]], edgecolor=SURFACE, linewidth=0.8, zorder=2)
        # Inside the block, not above it. Above, alta's 40 MW step is too
        # narrow to hold a label without running into park_city's, and
        # solitude's lands on the load line. A block narrower than its own
        # label turns the label upright rather than shrinking it -- the type
        # scale is fixed, so collisions are resolved by moving text.
        ax.text(left + g["pmax"] / 2, g["cost"] / 2,
                f"{_display(g['name'])} ({g['bus']})",
                ha="center", va="center", rotation=90 if g["pmax"] < 250 else 0,
                fontsize=8.5, color=_ink_on(hue[g["bus"]]), zorder=5)
        left += g["pmax"]

    ax.axvline(total_load, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
    ax.text(total_load - 24, 49, f"Load {total_load:.0f} MW", ha="right", va="top",
            fontsize=8.5, color=INK, zorder=5,
            bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none", pad=1.5))

    _bare(ax)
    ax.set_xlim(0, left)
    ax.set_ylim(0, 52)
    ax.set_xlabel("Cumulative capacity (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Offer ($/MWh)", fontsize=9.5, color=INK_2)


def figure_inputs(buses, branches, slack, fleet, load):
    """What configs/m3.yaml declares. Returns a Figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 1.15]},
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    capacity = {b: sum(g["pmax"] for g in fleet if g["bus"] == b) for b in buses}
    _panel_network(axes[0], buses, branches, slack, capacity, load)
    _title(axes[0], "a", "Network, capacity and load (MW)")

    _panel_offer_stack(axes[1], buses, fleet, sum(load.values()))
    _title(axes[1], "b", "Generators")
    return fig


# ----------------------------------------------------------- processing ----

def figure_processing(buses, branches):
    """What src/network/topology.py builds. Returns a Figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    names = [b.name for b in branches]

    _matrix(axes[0], incidence(buses, branches), names, buses, "{:+.0f}")
    _title(axes[0], "a", "Incidence A")
    axes[0].set_xlabel("Bus", fontsize=9.5, color=INK_2)
    axes[0].set_ylabel("Branch", fontsize=9.5, color=INK_2)

    _matrix(axes[1], b_bus(buses, branches), buses, buses, "{:.0f}")
    _title(axes[1], "b", "Susceptance B (p.u.)")
    axes[1].set_xlabel("Bus", fontsize=9.5, color=INK_2)
    axes[1].set_ylabel("Bus", fontsize=9.5, color=INK_2)
    return fig


# --------------------------------------------------------------- results ---

def _panel_shift_row(ax, buses, branches, P, line):
    hue = dict(zip(buses, BUS_HUE))
    row = P[[b.name for b in branches].index(line)]
    ax.bar(buses, row, color=[hue[b] for b in buses],
           edgecolor=SURFACE, linewidth=0.8, width=0.66)
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)

    for name, v in zip(buses, row):
        ax.text(name, v + (0.03 if v >= 0 else -0.03), _cell(v, "{:.2f}"),
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, color=INK_2)

    _bare(ax)
    ax.set_ylim(min(row.min(), 0) - 0.13, max(row.max(), 0) + 0.09)
    ax.set_xlabel("Bus injecting 1 MW", fontsize=9.5, color=INK_2)
    ax.set_ylabel(f"Flow on {line} (MW)", fontsize=9.5, color=INK_2)


def figure_results(buses, branches, slack, line="DE"):
    """What src/network/ptdf.py returns. Returns a Figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7),
                             gridspec_kw={"width_ratios": [1.25, 1.0]},
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    P = ptdf(buses, branches, slack)
    names = [b.name for b in branches]

    _matrix(axes[0], P, names, buses, "{:+.2f}", vlim=1.0)
    _title(axes[0], "a", f"Shift factors, slack {slack} (MW/MW)")
    axes[0].set_xlabel("Bus injecting", fontsize=9.5, color=INK_2)
    axes[0].set_ylabel("Branch", fontsize=9.5, color=INK_2)

    _panel_shift_row(axes[1], buses, branches, P, line)
    _title(axes[1], "b", f"Sensitivity of {line}")
    return fig


# ----------------------------------------------------------------- flows ---

OVER = "#d55e00"   # Okabe-Ito vermillion, reserved for a violated rating


def line_flows(buses, branches, slack, injection):
    """MW on each branch, positive along from_bus -> to_bus."""
    inj = np.array([injection[b] for b in buses])
    return ptdf(buses, branches, slack) @ inj


def _panel_flow_network(ax, buses, branches, slack, injection, flows):
    hue = dict(zip(buses, BUS_HUE))

    for br, mw in zip(branches, flows):
        (x0, y0), _ = LAYOUT[br.from_bus]
        (x1, y1), _ = LAYOUT[br.to_bus]
        over = abs(mw) > br.limit_mw
        rated = np.isfinite(br.limit_mw)

        # Width carries loading, colour carries the violation. Both, so the
        # overload is not colour alone and survives a grayscale print.
        width = 1.0 + 3.2 * min(abs(mw) / 320.0, 1.0)
        ax.plot([x0, x1], [y0, y1], color=OVER if over else INK_2,
                linewidth=width, solid_capstyle="round", zorder=1)

        # Direction is stated in the label rather than drawn as an arrowhead.
        head, tail = (br.from_bus, br.to_bus) if mw >= 0 else (br.to_bus, br.from_bus)
        label = f"{head} to {tail}\n{abs(mw):.0f}"
        if rated:
            label += f" / {br.limit_mw:.0f} MW"
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center",
                fontsize=8.5, color=OVER if over else INK, linespacing=1.3,
                bbox=dict(facecolor=SURFACE, alpha=0.9, edgecolor="none", pad=1.5),
                zorder=3)

    for name in buses:
        (x, y), (ha, va) = LAYOUT[name]
        ax.scatter([x], [y], s=430, zorder=4, facecolor=hue[name],
                   edgecolor=SURFACE, linewidth=1.4)
        ax.text(x, y, name, ha="center", va="center", fontsize=9.5,
                color=SURFACE, zorder=5)

        net_mw = injection[name]
        lines = [f"{net_mw:+.0f}"]
        if name == slack:
            lines.append("Slack")
        dx = {"right": -0.14, "left": 0.14, "center": 0.0}[ha]
        dy = {"center": 0.0, "top": -0.16}[va]
        ax.text(x + dx, y + dy, "\n".join(lines), ha=ha, va=va, fontsize=8.5,
                color=INK_2, linespacing=1.35, zorder=5)

    ax.set_xlim(-0.95, 2.2)
    ax.set_ylim(-1.75, 1.35)
    ax.set_axis_off()


def _panel_loading(ax, branches, flows):
    names = [b.name for b in branches]
    mag = np.abs(flows)
    over = [abs(mw) > br.limit_mw for br, mw in zip(branches, flows)]
    y = np.arange(len(branches))

    ax.barh(y, mag, color=[OVER if o else INK_2 for o in over],
            edgecolor=SURFACE, linewidth=0.8, height=0.62, zorder=2)

    for i, (br, mw) in enumerate(zip(branches, flows)):
        if np.isfinite(br.limit_mw):
            # The rating as a gate the bar has to pass through, not a bar of
            # its own -- two bars per line would read as two measurements.
            ax.plot([br.limit_mw, br.limit_mw], [i - 0.36, i + 0.36],
                    color=INK, linewidth=1.4, zorder=4)
        ax.text(abs(mw) + 9, i, f"{abs(mw):.0f}", ha="left", va="center",
                fontsize=8.5, color=OVER if over[i] else INK_2)

    # Label the topmost gate only. Naming each one repeats a unit and a word
    # the reader needs once.
    first = next(i for i, br in enumerate(branches) if np.isfinite(br.limit_mw))
    ax.text(branches[first].limit_mw, first - 0.55, "Rating", ha="center",
            va="bottom", fontsize=8.5, color=INK)

    _bare(ax)
    ax.set_yticks(y, names, fontsize=8.5, color=INK_2)
    ax.set_ylim(len(branches) - 0.5, -1.05)
    ax.set_xlim(0, max(mag.max(), 400) * 1.16)
    ax.set_xlabel("Flow (MW)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Branch", fontsize=9.5, color=INK_2)


def figure_flows(buses, branches, slack, injection):
    """Where the merit-order dispatch would push power. Returns a Figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 0.92]},
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    flows = line_flows(buses, branches, slack, injection)

    _panel_flow_network(axes[0], buses, branches, slack, injection, flows)
    _title(axes[0], "a", "Flows and net injection (MW)")

    _panel_loading(axes[1], branches, flows)
    _title(axes[1], "b", "Flow against rating")
    return fig


# ------------------------------------------------------------ settlement ---

def _panel_lmp(ax, buses, lmp, offer_range):
    hue = dict(zip(buses, BUS_HUE))
    ax.bar(buses, [lmp[b] for b in buses], color=[hue[b] for b in buses],
           edgecolor=SURFACE, linewidth=0.8, width=0.62, zorder=3)

    lo, hi = offer_range
    ax.axhspan(lo, hi, color=GRID, zorder=1)
    # Set inside the band at its left end, where no bar reaches.
    ax.text(-0.42, hi - 1.2, "Offer range", ha="left", va="top",
            fontsize=8.5, color=INK_2, zorder=4)

    for b in buses:
        ax.text(b, lmp[b] + 0.9, f"{lmp[b]:.2f}", ha="center", va="bottom",
                fontsize=8.5, color=INK_2, zorder=4)

    _bare(ax)
    ax.set_ylim(0, max(max(lmp.values()), hi) * 1.18)
    ax.set_xlabel("Bus", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Local price ($/MWh)", fontsize=9.5, color=INK_2)


def _panel_revenue(ax, buses, fleet, dispatch, lmp):
    """Revenue at the local price, split at the unit's own offer.

    The lower block is what the unit would need to break even; the upper is
    what the local price pays on top. A unit at a congested bus can earn a
    surplus that has nothing to do with being cheap.
    """
    hue = dict(zip(buses, BUS_HUE))
    order = sorted(fleet, key=lambda g: -dispatch[g["name"]])
    names = [_display(g["name"]) for g in order]

    for i, g in enumerate(order):
        mw, price = dispatch[g["name"]], lmp[g["bus"]]
        cost = mw * g["cost"]
        ax.bar(i, cost, color=hue[g["bus"]], edgecolor=SURFACE, linewidth=0.8,
               width=0.62, zorder=3)
        ax.bar(i, mw * price - cost, bottom=cost, color=hue[g["bus"]], alpha=0.42,
               edgecolor=SURFACE, linewidth=0.8, width=0.62, zorder=3)
        ax.text(i, mw * price + 260, f"{mw:.0f} MW\n{price:.2f} $/MWh",
                ha="center", va="bottom", fontsize=8.5, color=INK_2,
                linespacing=1.3, zorder=4)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=INK_2, edgecolor=SURFACE),
        plt.Rectangle((0, 0), 1, 1, facecolor=INK_2, alpha=0.42, edgecolor=SURFACE),
    ]
    ax.legend(handles, ["Offer cost", "Above offer"], frameon=False,
              fontsize=8.5, labelcolor=INK_2, loc="upper right",
              handlelength=1.3, borderpad=0.2)

    top = max(dispatch[g["name"]] * lmp[g["bus"]] for g in fleet)
    _bare(ax)
    ax.set_xticks(range(len(names)), names, fontsize=8.5, color=INK_2)
    ax.set_ylim(0, top * 1.34)
    ax.set_xlabel("Generator", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Revenue ($/h)", fontsize=9.5, color=INK_2)


def figure_settlement(buses, fleet, dispatch, lmp):
    """Who gets paid what, at which local price. Returns a Figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [0.78, 1.0]},
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    offers = [g["cost"] for g in fleet]
    _panel_lmp(axes[0], buses, lmp, (min(offers), max(offers)))
    _title(axes[0], "a", "Price by bus")

    _panel_revenue(axes[1], buses, fleet, dispatch, lmp)
    _title(axes[1], "b", "Generator revenue")
    return fig


if __name__ == "__main__":
    from datetime import datetime, timezone
    from pathlib import Path

    import yaml

    from src.model.inputs import Branch

    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / "configs" / "m3.yaml").read_text())
    net = config["network"]
    buses, slack = net["buses"], net["slack"]
    branches = [
        Branch(name=n, from_bus=s["from"], to_bus=s["to"],
               reactance_pu=float(s["reactance_pu"]), limit_mw=float(s["limit_mw"]))
        for n, s in net["branches"].items()
    ]
    fleet = [
        {"name": n, "bus": s["bus"], "cost": float(s["cost_usd_per_mwh"]),
         "pmax": float(s["pmax_mw"])}
        for n, s in config["fleet"].items()
    ]
    load = {b: float(mw) for b, mw in config["load"]["mw"].items()}

    # The dispatch M0 would choose if transmission were free. Solved, not
    # assumed -- the figure has to be able to be wrong.
    from src.model.dispatch import solve_dispatch

    res = solve_dispatch({g["name"]: g["cost"] for g in fleet},
                         {g["name"]: g["pmax"] for g in fleet},
                         sum(load.values()))
    injection = {
        b: sum(res["p"][g["name"]] for g in fleet if g["bus"] == b) - load.get(b, 0.0)
        for b in buses
    }
    assert abs(sum(injection.values())) < 1e-6, "injections must net to zero"

    # ---------------------------------------------------------------------
    # HAND-DERIVED, AND TEMPORARY. Stage 3's LP and Stage 4's pricing.py
    # replace every line of this block. It is here so the figures have prices
    # to draw before the solver exists, and so the LP has something to be
    # checked against -- if solve_dispatch_network_day disagrees with these
    # numbers, one of the two is wrong and that is the point.
    #
    # It works only because this case has exactly two marginal units and one
    # binding line, so lambda and mu fall out of two equations. Nothing about
    # it generalises; do not grow it.
    # ---------------------------------------------------------------------
    P = ptdf(buses, branches, slack)
    de = [b.name for b in branches].index("DE")
    i_e, i_c = buses.index("E"), buses.index("C")

    # brighton (E) and solitude (C) are both off their bounds, so each prices
    # at its own offer. Two equations, two duals.
    mu = (10.0 - 30.0) / (P[de, i_e] - P[de, i_c])
    lam = 30.0 - P[de, i_c] * mu
    lmp = {b: lam + P[de, i] * mu for i, b in enumerate(buses)}

    # Redispatch off E until DE sits exactly on its rating.
    over = abs(line_flows(buses, branches, slack, injection)[de]) - 240.0
    shift = over / (P[de, i_c] - P[de, i_e])
    feasible = dict(res["p"])
    feasible["brighton"] -= shift
    feasible["solitude"] += shift

    at_bus = {g["name"]: g["bus"] for g in fleet}
    payment = sum(mw * lmp[b] for b, mw in load.items())
    revenue = sum(mw * lmp[at_bus[n]] for n, mw in feasible.items())
    print(f"congestion rent  {payment - revenue:12,.2f} $/h "
          f"(mu x limit = {mu * 240.0:,.2f})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text((root / "configs" / "m3.yaml").read_text())

    for name, fig in [
        ("m3_1_inputs", figure_inputs(buses, branches, slack, fleet, load)),
        ("m3_2_processing", figure_processing(buses, branches)),
        ("m3_3_results", figure_results(buses, branches, slack)),
        ("m3_4_flows", figure_flows(buses, branches, slack, injection)),
        ("m3_5_settlement", figure_settlement(buses, fleet, feasible, lmp)),
    ]:
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
        print(out / f"{name}.png")
