"""Scenario and its parts: Generator, Bus, Branch, Load.

The boundary between ingest and model. Nothing in src/model/ may call a data
source; it reads a Scenario and nothing else.

That rule is what this module exists to enforce. A Scenario is inert: frozen
dataclasses, plain floats, a UTC-timestamped load, and a provenance dict that
records where the numbers came from without being able to go get more. There
is no fetch, no path, no API key, and no pandas dependency below this line --
the solver cannot reach a data source even by accident.

At M2 the network is empty. Bus and Branch are declared because the shape of
the object should not change when M3 fills them in, and because `buses = one`
is a claim worth being able to see rather than an omission.

              ingest/                     |            model/
    EIA-930 --> parse --> validate --> Scenario --> dispatch --> duals
    RTS-GMLC ->  ...  -->   ...    -->    ^
                                          |
                        the boundary. one direction only.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Bus:
    """A node. At M2 there is exactly one and it carries no information."""
    name: str


@dataclass(frozen=True)
class Branch:
    """A transmission line with a thermal limit.

    reactance_pu is strictly positive, and that is not a style rule. The DC
    model inverts it -- b = 1/x -- and then builds every matrix in
    src/network/ out of the result:

        x > 0    b > 0     B_bus is positive semidefinite. One zero
                           eigenvalue, the angle reference, which ptdf()
                           removes on purpose. This is the physical case.

        x = 0    b = inf   ZeroDivisionError inside b_branch().

        x < 0    b < 0     B_bus is INDEFINITE. It still inverts, the LP
                           still solves, and it returns prices that look
                           entirely plausible and mean nothing. A negative
                           reactance is a line that carries power uphill.

    The third case is why this check exists here rather than being left to
    the linear algebra. A bad reactance does not raise anywhere downstream;
    it produces a confident wrong answer, which is the one failure mode no
    test catches and no user notices.
    """
    name: str
    from_bus: str
    to_bus: str
    reactance_pu: float
    limit_mw: float

    def __post_init__(self):
        if self.from_bus == self.to_bus:
            # Incidence would put +1 and -1 in the same column, giving an
            # all-zero row: a line whose flow is identically zero and which
            # contributes nothing to B_bus. Harmless arithmetic, meaningless
            # network.
            raise ValueError(
                f"{self.name}: from_bus and to_bus are both {self.from_bus!r}"
            )
        if not self.reactance_pu > 0:
            raise ValueError(
                f"{self.name}: reactance_pu must be > 0, got {self.reactance_pu}"
            )
        if not self.limit_mw > 0:
            # inf passes: an unlimited line is a modelling choice this repo
            # uses deliberately, and inf > 0 is True. Zero does not, because a
            # line that can carry nothing is a line that should be deleted --
            # keeping it makes every solve infeasible for a reason the user
            # cannot see on screen.
            raise ValueError(
                f"{self.name}: limit_mw must be > 0, got {self.limit_mw}"
            )


@dataclass(frozen=True)
class Generator:
    """An offer: a price and a quantity, located at a bus.

    cost_usd_per_mwh is a single marginal cost, not yet a curve. It becomes
    fuel_price * heat_rate + VOM at M5 and a real ERCOT offer curve later.
    pmin_mw is carried but not enforced -- an LP has no way to honour it
    (trap 4), which is the whole reason M6 exists.
    """
    name: str
    bus: str
    cost_usd_per_mwh: float
    pmax_mw: float
    pmin_mw: float = 0.0

    def __post_init__(self):
        if self.pmax_mw < 0:
            raise ValueError(f"{self.name}: negative pmax_mw {self.pmax_mw}")
        if not 0 <= self.pmin_mw <= self.pmax_mw:
            raise ValueError(
                f"{self.name}: pmin_mw {self.pmin_mw} outside [0, {self.pmax_mw}]"
            )


@dataclass(frozen=True)
class Load:
    """Demand at one bus, keyed by UTC hour.

    mw maps an ISO-8601 UTC timestamp STRING to MW. A string, not a pandas
    Timestamp, so that the model layer needs no pandas and a Scenario stays
    trivially serializable into a run directory. Ingest formats it once; the
    hours sort correctly as strings because ISO-8601 UTC does.
    """
    bus: str
    mw: Dict[str, float]


@dataclass(frozen=True)
class Scenario:
    """Everything one solve needs, and nothing about where it came from.

    provenance is the exception that proves the rule: it is a record FOR the
    run directory, opaque to the solver, and it must never contain anything
    callable.
    """
    name: str
    generators: Tuple[Generator, ...]
    loads: Tuple[Load, ...]
    buses: Tuple[Bus, ...] = ()
    branches: Tuple[Branch, ...] = ()
    provenance: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Bus names are the column order of every matrix in src/network/ and
        # the keys of every price. A repeat gives B_bus two identical rows,
        # which is singular for a reason that has nothing to do with the angle
        # reference ptdf() expects to remove -- so it surfaces as "Singular
        # matrix" from inside numpy, indistinguishable from a network that was
        # simply cut in half.
        bus_names = [b.name for b in self.buses]
        if len(bus_names) != len(set(bus_names)):
            dupes = sorted({n for n in bus_names if bus_names.count(n) > 1})
            raise ValueError(f"duplicate bus names: {dupes}")

        # A Branch validates itself, but it cannot see the bus list, so this is
        # the only place an endpoint typo can be caught. Left alone it becomes
        # a bare KeyError two modules away in incidence().
        branch_names = [br.name for br in self.branches]
        if len(branch_names) != len(set(branch_names)):
            dupes = sorted({n for n in branch_names if branch_names.count(n) > 1})
            raise ValueError(f"duplicate branch names: {dupes}")
        known = set(bus_names)
        for br in self.branches:
            for end, bus in (("from_bus", br.from_bus), ("to_bus", br.to_bus)):
                if bus not in known:
                    raise ValueError(
                        f"branch {br.name}: {end} {bus!r} is not a declared bus "
                        f"{sorted(known)}"
                    )

        names = [g.name for g in self.generators]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate generator names in {names}")
        if not self.generators:
            raise ValueError(f"{self.name}: no generators")
        if not self.loads:
            raise ValueError(f"{self.name}: no loads")
        spans = {tuple(sorted(l.mw)) for l in self.loads}
        if len(spans) > 1:
            raise ValueError("loads disagree on the hour index")

    # -- views the solver consumes. plain dicts, no pandas, no I/O. --

    @property
    def hours(self) -> Sequence[str]:
        """The horizon, as sorted ISO-8601 UTC strings."""
        return sorted(self.loads[0].mw)

    def cost(self) -> Dict[str, float]:
        """{gen: $/MWh}, the c argument to solve_dispatch_day."""
        return {g.name: g.cost_usd_per_mwh for g in self.generators}

    def pmax(self) -> Dict[str, float]:
        """{gen: MW}, the Pmax argument to solve_dispatch_day."""
        return {g.name: g.pmax_mw for g in self.generators}

    def demand(self) -> Dict[str, float]:
        """{hour: MW}, summed over buses. The D argument to solve_dispatch_day.

        Single-bus at M2, so this sum is over one term. At M3 the solver stops
        wanting a system total and starts wanting demand per bus, and this
        method gets a sibling rather than a rewrite.
        """
        return {
            t: sum(l.mw[t] for l in self.loads)
            for t in self.hours
        }

    @property
    def total_capacity_mw(self) -> float:
        return sum(g.pmax_mw for g in self.generators)

    @property
    def peak_load_mw(self) -> float:
        d = self.demand()
        return max(d.values())

    def demand_by_bus(self) -> Dict[str, Dict[str, float]]:
        """{bus: {hour: MW}}. Sibling to demand(), not a replacement.

        Every bus gets an entry, including the ones carrying no load. The
        network solver needs D[i, t] for every i to form an injection, and a
        bus that is simply absent from the mapping is a KeyError rather than
        a zero.

        Loads are summed, not assigned, because nothing stops two Load objects
        sitting at the same bus -- and a dict comprehension would silently
        keep only the last of them.
        """
        out = {b.name: {t: 0.0 for t in self.hours} for b in self.buses}
        for l in self.loads:
            if l.bus not in out:
                raise ValueError(f"load at unknown bus {l.bus!r}")
            for t, mw in l.mw.items():
                out[l.bus][t] += mw
        return out
