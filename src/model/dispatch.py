"""Pass 2 of 2: economic dispatch as an LP with binaries fixed.

Produces dispatch levels AND the duals that pricing needs:
  lambda -- dual on system energy balance
  mu[l]  -- dual on each line flow limit

M0: single bus, single hour, no network. lambda is the whole price.
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


if __name__ == "__main__":
    c = {"g1": 20, "g2": 35, "g3": 80}       # marginal cost, $/MWh
    Pmax = {"g1": 100, "g2": 100, "g3": 100}  # capacity, MW
    D = 150                                   # demand, MW

    r = solve_dispatch(c, Pmax, D)

    print("Dispatch:")
    for g, mw in r["p"].items():
        print(f"  {g}: {mw:.1f} MW")
    print(f"System price lambda: ${r['lmbda']:.2f}/MWh")
    print(f"Total production cost: ${r['cost']:.2f}")
    print(f"Load payment:   ${D * r['lmbda']:.2f}")
    print(f"Gen revenue:    ${sum(r['p'].values()) * r['lmbda']:.2f}")
