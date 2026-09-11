"""One call from a Scenario to a cleared, priced, settled market.

Everything here already existed, scattered across a __main__ block and a test
helper: build the PTDF, solve the LP, assemble the prices, check the identity.
Nothing new is formulated. What is new is that the whole pipeline has ONE name,
so a caller that is not a figure -- a parameter sweep, an HTTP handler, a
notebook -- can run a market without reassembling the sequence by hand and
getting one step subtly wrong.

    Scenario --> ptdf() --> solve_dispatch_network_day() --> lmps() --> settle()
       |           |                    |                      |          |
     buses      PTDF[l,i]           p, lambda, mu           LMP[i]    residual
     branches                                                          == 0

The slack is an ARGUMENT, not a Scenario field. It is a choice this layer
makes, and per CLAUDE.md trap 2 nothing physical depends on it: changing it
shifts every LMP by a constant and moves no dispatch, no flow, and no price
DIFFERENCE. It defaults to whatever the config recorded so that the common
case needs no second argument.
"""

import numpy as np

from src.model.dispatch import solve_dispatch_network_day
from src.model.pricing import congestion_prices, lmps
from src.network.ptdf import ptdf
from src.settle.settlement import settle


def clear(scenario, slack=None, limits=None):
    """Clear, price, and settle every hour of a Scenario.

    Args:
        scenario: a Scenario carrying buses and branches. Single-bus scenarios
                  are rejected -- they have no network, and M0's solve_dispatch
                  is the right call for them.
        slack:    bus name. Defaults to scenario.provenance["slack"].
        limits:   {line: MW} overriding the branch ratings. The lever a sweep
                  or a UI slider pulls: pass {"DE": 240.0} to re-solve one
                  network at a different rating without rebuilding a Scenario.
                  Lines omitted keep their declared limit; np.inf is allowed
                  and means unlimited.

    Returns:
        A dict keyed by hour under "hours", with everything a caller needs to
        draw or serve the result. Dispatch, flows, mu and lmp are keyed by
        (name, hour) exactly as the solver returns them -- not re-nested --
        because a caller that wants one snapshot indexes once, and a caller
        that wants the day iterates, and neither has to undo a shape the other
        preferred.

    Raises:
        ValueError on a scenario this cannot price; RuntimeError from the
        solver if the LP is infeasible or unbounded.
    """
    buses = [b.name for b in scenario.buses]
    branches = list(scenario.branches)
    if len(buses) < 2 or not branches:
        raise ValueError(
            f"{scenario.name}: nodal clearing needs a network; got "
            f"{len(buses)} bus(es) and {len(branches)} branch(es)"
        )

    if slack is None:
        slack = scenario.provenance.get("slack")
    if slack is None:
        raise ValueError("no slack bus: pass slack= or record network.slack")
    if slack not in buses:
        raise ValueError(f"slack {slack!r} is not a bus in {buses}")

    lines = [br.name for br in branches]
    # Row order is branches, column order is buses, and both are fixed HERE and
    # passed down. Every consumer of the PTDF re-derives its index from these
    # two lists rather than from its own ordering assumption.
    PTDF = ptdf(buses, branches, slack)

    Fmax = {br.name: br.limit_mw for br in branches}
    if limits:
        unknown = set(limits) - set(Fmax)
        if unknown:
            raise ValueError(f"limit override for unknown line(s) {sorted(unknown)}")
        Fmax.update({l: float(v) for l, v in limits.items()})

    gen_bus = {g.name: g.bus for g in scenario.generators}
    D = scenario.demand_by_bus()

    res = solve_dispatch_network_day(
        c=scenario.cost(),
        Pmax=scenario.pmax(),
        D=D,
        gen_bus=gen_bus,
        buses=buses,
        PTDF=PTDF,
        Fmax=Fmax,
    )

    mu = congestion_prices(res)
    lmp = lmps(res, buses, lines, PTDF)

    # Settlement is per hour because the identity is per hour. Summing the day
    # first would let a positive residual in one hour cancel a negative one in
    # another and report a clean zero over a broken solve.
    settlement = {}
    for t in scenario.hours:
        gen_mw = {b: 0.0 for b in buses}
        for g, bus in gen_bus.items():
            gen_mw[bus] += res["p"][g, t]
        settlement[t] = settle(
            lmp={b: lmp[b, t] for b in buses},
            load_mw={b: D[b][t] for b in buses},
            gen_mw=gen_mw,
            mu={l: mu[l, t] for l in lines},
            limits=Fmax,
        )

    return {
        "buses": buses,
        "lines": lines,
        "slack": slack,
        "hours": list(scenario.hours),
        "limits": Fmax,
        "gen_bus": gen_bus,
        "dispatch": res["p"],      # {(gen, hour): MW}
        "flows": res["f"],         # {(line, hour): MW}
        "mu": mu,                  # {(line, hour): $/MWh}
        "lmp": lmp,                # {(bus, hour): $/MWh}
        "lmbda": res["lmbda"],     # {hour: $/MWh}
        "cost": res["cost"],       # $/h over the horizon
        "settlement": settlement,  # {hour: {...}}
        "PTDF": PTDF,
    }


def binding_lines(cleared, hour, tol=1e-9):
    """Line names whose flow limit is holding the dispatch back, at one hour.

    A line is binding when its dual is non-zero, not when its flow is near its
    rating. Those usually coincide, but the dual is the one that answers the
    question a market asks -- would one more MW of this line change the cost --
    and it is the number the price is assembled from.
    """
    return {l for l in cleared["lines"] if abs(cleared["mu"][l, hour]) > tol}
