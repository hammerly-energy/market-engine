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
    """A transmission line with a thermal limit. Unused until M3."""
    name: str
    from_bus: str
    to_bus: str
    reactance_pu: float
    limit_mw: float


@dataclass(frozen=True)
class Generator:
    """An offer: a price and a quantity, located at a bus.

    cost_usd_per_mwh is a single marginal cost, not yet a curve. It becomes
    fuel_price * heat_rate + VOM at M4 and a real ERCOT offer curve later.
    pmin_mw is carried but not enforced -- an LP has no way to honour it
    (trap 4), which is the whole reason M5 exists.
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
