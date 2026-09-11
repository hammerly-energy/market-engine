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


def capacity(Pmax, g, t):
    """The upper bound on generator g in hour t.

    Pmax[g] is either a float -- a machine whose capacity is the same in every
    hour -- or a {hour: MW} mapping, which is what a wind farm, a solar plant
    or a hydro schedule needs. Both live in the same argument because to the LP
    they are the same thing: a number on the right of p[g,t] <= .

    This is the ONLY coupling M5 adds, and it is not a coupling between hours.
    An hourly cap still constrains each hour independently, so the block
    diagonal structure M1 established survives intact. What breaks separability
    is a constraint that spans two t -- ramping, min up time, state of charge --
    and none of those are here yet.
    """
    cap = Pmax[g]
    if isinstance(cap, dict):
        try:
            return cap[t]
        except KeyError:
            raise ValueError(f"{g}: no capacity for hour {t!r}") from None
    return cap


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
        # A profiled unit carries one number per hour, so the check walks the
        # profile. A single negative hour in a weather file is exactly the kind
        # of thing that would otherwise invert one bound out of thousands.
        hourly = cap.values() if isinstance(cap, dict) else [cap]
        for mw in hourly:
            if mw < 0:
                raise ValueError(f"negative capacity for {g}: {mw}")
    for t, mw in D.items():
        if mw < 0:
            raise ValueError(f"negative demand in hour {t}: {mw}")



def solve_dispatch_network_day(c, Pmax, D, gen_bus, buses, PTDF, Fmax):
    """Clear a NODAL energy market across a set of hours, on a DC network.

    c        -- {gen: marginal cost $/MWh}      same in every hour
    Pmax     -- {gen: capacity MW}              same in every hour
    D        -- {bus: {t: MW}}                  from Scenario.demand_by_bus()
    gen_bus  -- {gen: bus}
    buses    -- ordered bus names; fixes the PTDF COLUMN order
    PTDF     -- (L, N) array; rows ordered as Fmax's keys
    Fmax     -- {line: MW}, .inf allowed        fixes the PTDF ROW order

    Returns {"p": {(gen, t): MW}, "lmbda": {t: $/MWh}, "cost": $,
             "f": {(line, t): MW}, "mu_up"/"mu_dn": {(line, t): $/MWh}}.
    Raises RuntimeError if the solve is not optimal.

    Sits ALONGSIDE solve_dispatch_day, not in place of it. That one clears a
    single bus and its lambda is the whole price; this one adds the network,
    and lambda becomes only the energy component. Keeping both means M1's
    separability tests still have their model to run against.

    The two orderings above are the load-bearing part. PTDF[l, i] means
    nothing without knowing which l and which i, and a transposed row here
    reappears later as an inexplicable price at the wrong bus.

    Prices are NOT assembled here. This returns the two dual families raw;
    src/model/pricing.py combines them into
    LMP[i, t] = lmbda[t] + sum_l PTDF[l, i] * mu[l, t].

    Nothing here may reference two different t. That is still M1's rule --
    the network couples BUSES within an hour, never hours to each other.
    """
    m = pyo.ConcreteModel()

    # 1. index sets. G is the fleet, T is the horizon, B and L the network.
    #    D is keyed by bus now, so the hours are one level down and have to
    #    be gathered across buses -- sorted(D) would give bus names. Sorted
    #    for determinism: build order decides which vertex simplex reports at
    #    a degenerate hour, so a stable order keeps a degenerate price
    #    reproducible.
    #
    #    B and L take their order from the caller, not from sorting, because
    #    that order IS the PTDF's column and row order.
    m.G = pyo.Set(initialize=list(c))
    m.T = pyo.Set(initialize=sorted({t for per_bus in D.values() for t in per_bus}),
                  ordered=True)
    m.B = pyo.Set(initialize=list(buses), ordered=True)
    m.L = pyo.Set(initialize=list(Fmax), ordered=True)

    # 2. one variable per generator PER HOUR, bounded 0..Pmax
    #    Unchanged from the single-bus model. A generator's capacity does not
    #    depend on where it sits -- the network constrains the FLOW between
    #    buses, not the machine.
    m.p = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (0, capacity(Pmax, g, t)))

    # 3. minimize total cost over the whole horizon
    #    No transmission term. A DC line is lossless, so moving power costs
    #    nothing; congestion shows up as a binding constraint, never as a
    #    price in the objective.
    m.cost = pyo.Objective(
        expr=sum(c[g] * m.p[g, t] for g in m.G for t in m.T),
        sense=pyo.minimize,
    )

    # 4. energy balance -- one EQUALITY PER HOUR. these carry lambda.
    #    Still SYSTEM-wide, not per bus: total generation equals total load.
    #    The network decides where the power can physically go, and that is
    #    step 6's job; this only says the books balance. One dual per hour,
    #    the same at every bus, which is exactly the energy component.
    def _balance(m, t):
        return sum(m.p[g, t] for g in m.G) == sum(D[i][t] for i in buses)

    m.balance = pyo.Constraint(m.T, rule=_balance)

    # 5. net injection and line flow. EXPRESSIONS, not constraints.
    #    An Expression names a sum; it asserts nothing and carries no dual.
    #    inj is generation at a bus minus load there -- positive for a net
    #    source. f is the PTDF definition applied to it: each line's flow is
    #    a weighted sum of every injection in the system, which is what makes
    #    a network different from a single bus.
    #
    #    Writing these as constraints instead would work, but it would create
    #    dual families that mean nothing and clutter the settlement identity.
    def _inj(m, i, t):
        return sum(m.p[g, t] for g in m.G if gen_bus[g] == i) - D[i][t]

    m.inj = pyo.Expression(m.B, m.T, rule=_inj)

    # Position lookups, because PTDF is a plain array and indexes by integer.
    col = {b: j for j, b in enumerate(buses)}
    row = {l: k for k, l in enumerate(Fmax)}

    def _flow(m, l, t):
        return sum(PTDF[row[l], col[i]] * m.inj[i, t] for i in m.B)

    m.f = pyo.Expression(m.L, m.T, rule=_flow)

    # 6. line limits -- TWO ONE-SIDED CONSTRAINTS PER LINE, never one ranged.
    #    A line is rated the same in both directions, so -Fmax <= f <= Fmax
    #    is the physics. Written as a single ranged constraint Pyomo returns
    #    ONE dual, and its sign no longer says which direction bound. Split
    #    into two, at most one of them binds, and the sign is unambiguous.
    #    This is the top cause of a broken settlement identity.
    #
    #    An .inf limit builds a trivially-true constraint whose dual is zero.
    #    That is deliberate -- every line keeps a row, so the caller never has
    #    to check whether a key exists.
    m.mu_up = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: m.f[l, t] <= Fmax[l])
    m.mu_dn = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: -m.f[l, t] <= Fmax[l])

    # 7. ask for duals BEFORE solving. omit this and m.dual is empty
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # 8. solve
    #    load_solutions=False so an infeasible model reports its status
    #    instead of raising while trying to load a solution that is not there.
    #    Infeasible now has a second cause it did not have at M1: enough
    #    capacity in total, but no way to deliver it through the network.
    res = pyo.SolverFactory("appsi_highs").solve(m, load_solutions=False)
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"solve not optimal: {tc}")
    m.solutions.load_from(res)

    # 9. unpack. m.dual is indexed by the CONSTRAINT OBJECT -- m.balance[t],
    #    not a bare t. Expressions evaluate after the solve like anything
    #    else, so f comes back through pyo.value.
    #
    #    The mu are returned RAW, in the solver's own sign convention, and
    #    are not yet the congestion prices. HiGHS reports a binding <=
    #    constraint with a non-positive dual, so a congested line shows up as
    #    a negative mu_dn. pricing.py is where mu = mu_up - mu_dn resolves
    #    that into one signed number per line and the settlement identity
    #    becomes checkable.
    return {
        "p": {(g, t): pyo.value(m.p[g, t]) for g in m.G for t in m.T},
        "lmbda": {t: m.dual[m.balance[t]] for t in m.T},
        "cost": pyo.value(m.cost),
        "f": {(l, t): pyo.value(m.f[l, t]) for l in m.L for t in m.T},
        "mu_up": {(l, t): m.dual[m.mu_up[l, t]] for l in m.L for t in m.T},
        "mu_dn": {(l, t): m.dual[m.mu_dn[l, t]] for l in m.L for t in m.T},
    }



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
    m.p = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (0, capacity(Pmax, g, t)))

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
        if 1e-6 < res["p"][g, t] < capacity(Pmax, g, t) - 1e-6:
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
