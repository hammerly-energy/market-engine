"""Pass 2 of 2: economic dispatch as an LP with binaries fixed.

Produces dispatch levels AND the duals that pricing needs:
  lambda -- dual on system energy balance
  mu[l]  -- dual on each line flow limit

M0: single bus, single hour, no network. lambda is the whole price.
M1: single bus, 24 hours. One balance constraint per hour, one dual per hour.
    Nothing couples the hours, so the problem is block diagonal.
"""

import pyomo.environ as pyo


def solve_dispatch(c, Pmax, D):
    """Clear a single-bus, single-hour energy market.

    c     -- {gen: marginal cost $/MWh}
    Pmax  -- {gen: capacity MW}
    D     -- demand MW

    Returns {"p": {gen: MW}, "lmbda": $/MWh, "cost": $}.
    Raises RuntimeError if the solve is not optimal.
    """
    m = pyo.ConcreteModel()

    # 1. index set
    m.G = pyo.Set(initialize=list(c))

    # 2. one variable per generator, bounded 0..Pmax
    m.p = pyo.Var(m.G, bounds=lambda m, g: (0, Pmax[g]))

    # 3. minimize total cost
    m.cost = pyo.Objective(expr=sum(c[g] * m.p[g] for g in m.G), sense=pyo.minimize)

    # 4. energy balance -- EQUALITY. this is the one that carries the price
    m.balance = pyo.Constraint(expr=sum(m.p[g] for g in m.G) == D)

    # 5. ask for duals BEFORE solving. omit this and m.dual is empty
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # 6. solve
    #    load_solutions=False so an infeasible model reports its status
    #    instead of raising while trying to load a solution that is not there
    res = pyo.SolverFactory("appsi_highs").solve(m, load_solutions=False)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"solve not optimal: {tc}")
    m.solutions.load_from(res)

    return {
        "p": {g: pyo.value(m.p[g]) for g in m.G},
        "lmbda": m.dual[m.balance],
        "cost": pyo.value(m.cost),
    }


def _check_day_inputs(c, Pmax, D):
    """Reject malformed inputs before Pyomo turns them into a confusing model.

    Catches the mistakes that otherwise surface as a wrong answer rather
    than an error: a generator priced but not sized (it silently vanishes from
    the fleet), a negative capacity (an inverted bound, infeasible for reasons
    that look like a modelling bug), a negative demand.

    Deliberately does NOT check that supply can meet demand. Whether the day
    clears is the solver's finding, not a precondition -- the caller learns it
    from the termination condition, the same way it learns about any other
    infeasibility. A capacity precheck here would also start lying the moment
    a network or a commitment constraint can make a feasible-looking day
    infeasible.
    """
    if not D:
        raise ValueError("D is empty: no hours to solve")
    missing = set(c) ^ set(Pmax)
    if missing:
        raise ValueError(f"c and Pmax disagree on the fleet: {sorted(missing)}")
    for g, cap in Pmax.items():
        if cap < 0:
            raise ValueError(f"negative capacity for {g}: {cap}")
    for t, mw in D.items():
        if mw < 0:
            raise ValueError(f"negative demand in hour {t}: {mw}")


def solve_dispatch_day(c, Pmax, D):
    """Clear a single-bus energy market jointly across a set of hours.

    c     -- {gen: marginal cost $/MWh}          same in every hour
    Pmax  -- {gen: capacity MW}                  same in every hour
    D     -- {t: demand MW}                      t is any hashable label

    Returns {"p": {(gen, t): MW}, "lmbda": {t: $/MWh}, "cost": $}.
    Raises RuntimeError if the solve is not optimal.

    Same objective and same bounds as solve_dispatch, indexed over T as well
    as G. The energy balance becomes one constraint per hour, so it needs
    rule= rather than expr=, and the dual is m.dual[m.balance[t]].

    Nothing here may reference two different t. That is the milestone.
    """
    _check_day_inputs(c, Pmax, D)

    m = pyo.ConcreteModel()

    # 1. index sets. G is the fleet, T is the horizon.
    #    T is built from D's keys, so hour labels are whatever the caller
    #    used -- 0..23 now, UTC timestamps at M2. Sorted for determinism:
    #    build order decides which vertex simplex reports at a degenerate
    #    hour, so a stable order keeps a degenerate price reproducible.
    m.G = pyo.Set(initialize=list(c))
    m.T = pyo.Set(initialize=sorted(D), ordered=True)

    # 2. one variable per generator PER HOUR, bounded 0..Pmax
    #    Two index sets, so the bounds rule takes two indices. Pmax[g] does
    #    not depend on t -- the same machine, 24 times over.
    m.p = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (0, Pmax[g]))

    # 3. minimize total cost over the whole horizon
    #    A sum over both index sets. No discounting and no weighting: every
    #    hour is one hour, so the day's cost is the sum of the hours' costs.
    #    That is what makes the objective separable -- it is already a sum of
    #    24 independent terms, and step 4 decides whether the constraints
    #    keep it that way.
    m.cost = pyo.Objective(
        expr=sum(c[g] * m.p[g, t] for g in m.G for t in m.T),
        sense=pyo.minimize,
    )

    # 4. energy balance -- one EQUALITY PER HOUR. these carry the prices.
    #    An indexed Constraint needs rule=, not expr=. The rule takes the
    #    model and the index: def _balance(m, t): return ... == D[t]
    #
    #    Only p[g, t] and D[t] appear inside the rule. The moment a t-1 or a
    #    t+1 appears, the blocks fuse and M1's separability tests are lying.
    def _balance(m, t):
        return sum(m.p[g, t] for g in m.G) == D[t]

    m.balance = pyo.Constraint(m.T, rule=_balance)

    # 5. ask for duals BEFORE solving. omit this and m.dual is empty
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # 6. solve
    #    load_solutions=False so an infeasible model reports its status
    #    instead of raising while trying to load a solution that is not there
    res = pyo.SolverFactory("appsi_highs").solve(m, load_solutions=False)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        # One status for the whole day. A single infeasible hour takes all 24
        # down with it -- there is no partial answer to hand back.
        raise RuntimeError(f"solve not optimal: {tc}")
    m.solutions.load_from(res)

    # 7. unpack. p is keyed by the (gen, hour) pair; lmbda by hour alone.
    #    m.dual is indexed by the CONSTRAINT OBJECT, so it is m.balance[t],
    #    not m.balance. Indexing it with a bare t is a KeyError, and looking
    #    up m.balance gets you an IndexedConstraint that is not a dual key.
    return {
        "p": {(g, t): pyo.value(m.p[g, t]) for g in m.G for t in m.T},
        "lmbda": {t: m.dual[m.balance[t]] for t in m.T},
        "cost": pyo.value(m.cost),
    }


def marginal_unit(res, c, Pmax, t):
    """The unit strictly between 0 and its cap in hour t, or None.

    "Marginal" means free to move in both directions, so it is the unit whose
    offer the dual should equal. A unit pinned at its cap or at zero cannot
    respond to one more MW of demand and never sets the price. None means the
    hour sits on a breakpoint and the dual is not unique -- see the M0 and M1
    degeneracy tests.
    """
    for g in c:
        if 1e-6 < res["p"][g, t] < Pmax[g] - 1e-6:
            return g
    return None


def day_summary(res, c, Pmax, D):
    """Per-hour rollup: dispatch, price, cost, and who set it.

    Reporting only -- no model logic here. The settlement identity lives in
    src/settle/, not in a print helper.
    """
    rows = []
    for t in sorted(D):
        served = sum(res["p"][g, t] for g in c)
        rows.append({
            "t": t,
            "load_mw": D[t],
            "served_mw": served,
            "lmbda": res["lmbda"][t],
            "marginal_unit": marginal_unit(res, c, Pmax, t),
            "cost": sum(c[g] * res["p"][g, t] for g in c),
            "load_payment": D[t] * res["lmbda"][t],
            "dispatch": {g: res["p"][g, t] for g in c},
        })
    return rows


def _print_day(rows, c):
    fleet = list(c)
    head = ["t", "Load", "lambda", "Marg"] + fleet
    widths = [3, 7, 8, 6] + [7] * len(fleet)
    print("  ".join(h.rjust(w) for h, w in zip(head, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = [
            f"{r['t']}",
            f"{r['load_mw']:.0f}",
            f"{r['lmbda']:.2f}",
            r["marginal_unit"] or "--",
        ] + [f"{r['dispatch'][g]:.1f}" for g in fleet]
        print("  ".join(cell.rjust(w) for cell, w in zip(cells, widths)))


if __name__ == "__main__":
    c = {"g1": 20.0, "g2": 35.0, "g3": 80.0}       # marginal cost, $/MWh
    Pmax = {"g1": 100.0, "g2": 100.0, "g3": 100.0}  # capacity, MW

    # M0: one hour.
    r = solve_dispatch(c, Pmax, 150.0)
    print("M0 -- single hour, D = 150 MW")
    for g, mw in r["p"].items():
        print(f"  {g}: {mw:.1f} MW")
    print(f"  lambda ${r['lmbda']:.2f}/MWh, cost ${r['cost']:,.2f}")

    # M1: the same fleet across a synthetic day. Mirrors configs/m1.yaml.
    load = dict(enumerate([
         62,  58,  55,  54,  57,  68,  92, 118, 146, 162, 171, 178,
        183, 186, 181, 176, 188, 214, 247, 263, 238, 192, 141,  88,
    ]))

    day = solve_dispatch_day(c, Pmax, load)
    rows = day_summary(day, c, Pmax, load)

    print("\nM1 -- 24 hours, joint solve")
    _print_day(rows, c)

    energy = sum(load.values())
    payment = sum(r["load_payment"] for r in rows)
    print(f"\n  Energy served       {energy:>12,.0f} MWh")
    print(f"  Production cost     {day['cost']:>12,.2f} $")
    print(f"  Load payment        {payment:>12,.2f} $")
    print(f"  Producer surplus    {payment - day['cost']:>12,.2f} $")
    print(f"  Load-weighted price {payment / energy:>12,.2f} $/MWh")
