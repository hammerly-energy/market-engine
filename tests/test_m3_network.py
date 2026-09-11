r"""M3 network tests: incidence, susceptance, PTDF, and the nodal clearing.

Everything here is hand-checkable. The point of M3 is that exactly one new
failure surface opens -- the network -- so these tests pin the linear algebra
down before an LP is ever asked to trust it. A transposed row or a flipped
sign caught here is an afternoon; caught at M5 it is an inexplicable LMP.

Three reference networks, chosen because each has an answer you can write
down without a computer:

    2-bus            one line, nowhere else to go        PTDF entry is 1
    A --AB-- B

    radial           no loop, so no choices to make      entries are 0 or +-1
    A --AB-- B --BC-- C

    3-line loop      equal reactance, symmetric split    2/3 direct, 1/3 long
    A --AB-- B
     \        \
      AC       BC
       \        \
        ---- C ----

TestNetworkInvariants holds forever. b_bus stays symmetric and singular at
M5, at M8, and on any network anyone ever hands this repo -- those are
properties of the DC formulation, not of case5.

TestCase5Transcription is the provenance guard. configs/m3.yaml is a
hand-typed view of tests/fixtures/case5.m, and MATPOWER numbers buses while
this repo names them A..E. That rename is where a silent transposition would
enter. The .m file is the source of truth; the config is checked against it.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ingest.scenario import build_scenario, load_config, scenario_from_config
from src.model.clearing import binding_lines, clear
from src.model.dispatch import solve_dispatch_network_day
from src.model.inputs import Branch
from src.model.pricing import congestion_prices, lmps
from src.network.ptdf import ptdf
from src.network.topology import (
    b_bus, b_branch, b_flow, components, incidence)
from src.settle.settlement import settle

CASE5_M = Path(__file__).parent / "fixtures" / "case5.m"
CONFIG = Path(__file__).parents[1] / "configs" / "m3.yaml"

# MATPOWER bus number -> the letter this repo uses. The one mapping the whole
# transcription test turns on; it is stated once, here.
BUS_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def branch(name, f, t, x, limit=np.inf):
    return Branch(name=name, from_bus=f, to_bus=t, reactance_pu=x, limit_mw=limit)


# --- the three reference networks -----------------------------------------

TWO_BUS = (["A", "B"], [branch("AB", "A", "B", 0.1)])

RADIAL = (
    ["A", "B", "C"],
    [branch("AB", "A", "B", 0.1), branch("BC", "B", "C", 0.2)],
)

LOOP = (
    ["A", "B", "C"],
    [
        branch("AB", "A", "B", 0.1),
        branch("BC", "B", "C", 0.1),
        branch("AC", "A", "C", 0.1),
    ],
)


# --- case5, read off the config -------------------------------------------

def _case5():
    net = yaml.safe_load(CONFIG.read_text())["network"]
    buses = net["buses"]
    branches = [
        branch(n, s["from"], s["to"], float(s["reactance_pu"]), float(s["limit_mw"]))
        for n, s in net["branches"].items()
    ]
    return buses, branches, net["slack"]


def _matpower_rows(block):
    """Numeric rows of one mpc.<block> = [ ... ]; assignment in case5.m."""
    text = CASE5_M.read_text()
    body = re.search(rf"mpc\.{block}\s*=\s*\[(.*?)\];", text, re.S).group(1)
    rows = []
    for line in body.splitlines():
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in line.split()])
    return rows


class TestNetworkInvariants:
    """True of any DC network, at every milestone. Never deleted."""

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_incidence_rows_sum_to_zero(self, net):
        # +1 at from, -1 at to. A line that both leaves and arrives somewhere
        # nets out; a row that does not sum to zero is a self-loop or a typo.
        A = incidence(*net)
        assert np.allclose(A.sum(axis=1), 0.0)

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_incidence_has_two_nonzeros_per_row(self, net):
        A = incidence(*net)
        assert np.all(np.count_nonzero(A, axis=1) == 2)

    def test_b_branch_is_reciprocal_reactance(self):
        buses, branches = RADIAL
        assert np.allclose(np.diag(b_branch(branches)), [1 / 0.1, 1 / 0.2])

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_b_bus_is_symmetric(self, net):
        B = b_bus(*net)
        assert np.allclose(B, B.T)

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_b_bus_is_singular(self, net):
        # Rows sum to zero because adding a constant to every angle moves no
        # power. That singularity IS the slack; ptdf() is where it is resolved.
        B = b_bus(*net)
        assert np.allclose(B.sum(axis=1), 0.0)
        assert np.allclose(B @ np.ones(B.shape[0]), 0.0)

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_b_flow_composes_incidence_and_susceptance(self, net):
        buses, branches = net
        assert np.allclose(b_flow(*net), b_branch(branches) @ incidence(*net))

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_ptdf_slack_column_is_zero(self, net):
        # Inject at the slack, withdraw at the slack: nothing moves.
        buses, _ = net
        for slack in buses:
            P = ptdf(*net, slack)
            assert np.allclose(P[:, buses.index(slack)], 0.0)

    @pytest.mark.parametrize("net", [TWO_BUS, RADIAL, LOOP], ids=["2bus", "radial", "loop"])
    def test_ptdf_shape_matches_network(self, net):
        buses, branches = net
        assert ptdf(*net, buses[0]).shape == (len(branches), len(buses))

    @pytest.mark.parametrize("net", [RADIAL, LOOP], ids=["radial", "loop"])
    def test_ptdf_differences_are_slack_invariant(self, net):
        """CLAUDE.md trap 2, in matrix form.

        A transfer from i to j moves PTDF[l,i] - PTDF[l,j] onto line l, and
        that difference cannot depend on which bus was called the reference.
        The whole matrix shifts by a per-row constant; differences do not.
        """
        buses, _ = net
        base = ptdf(*net, buses[0])
        base = base - base[:, [0]]
        for slack in buses[1:]:
            P = ptdf(*net, slack)
            assert np.allclose(P - P[:, [0]], base)


class TestTwoBus:
    """One line. Every MW that leaves A arrives at B; there is no alternative."""

    def test_incidence(self):
        assert np.allclose(incidence(*TWO_BUS), [[1, -1]])

    def test_b_flow(self):
        # b = 1/0.1 = 10, so 1 radian of angle difference is 10 MW.
        assert np.allclose(b_flow(*TWO_BUS), [[10, -10]])

    def test_b_bus(self):
        assert np.allclose(b_bus(*TWO_BUS), [[10, -10], [-10, 10]])

    def test_ptdf(self):
        assert np.allclose(ptdf(*TWO_BUS, "B"), [[1, 0]])

    def test_ptdf_flips_sign_with_slack(self):
        # Slack A means "inject at B, withdraw at A", which runs against the
        # A -> B positive direction incidence() declared.
        assert np.allclose(ptdf(*TWO_BUS, "A"), [[0, -1]])


class TestRadial:
    """No loop, so power has no choice. Every entry is 0 or +-1."""

    def test_ptdf_entries_are_zero_or_unit(self):
        P = ptdf(*RADIAL, "A")
        assert np.allclose(np.abs(P), np.round(np.abs(P)))
        assert set(np.round(P.ravel(), 9)) <= {0.0, 1.0, -1.0}

    def test_ptdf_is_independent_of_reactance(self):
        """The defining property of a radial network.

        Reactance decides how power SPLITS between parallel paths. With no
        parallel path there is nothing to split, so the answer is topology
        alone. Doubling a reactance must change nothing.
        """
        buses, branches = RADIAL
        stretched = [
            branch(b.name, b.from_bus, b.to_bus, b.reactance_pu * 7.3)
            for b in branches
        ]
        assert np.allclose(ptdf(buses, branches, "A"), ptdf(buses, stretched, "A"))

    def test_injection_at_c_loads_both_lines(self):
        # C -> A must traverse BC then AB, in the reverse of both declared
        # positive directions.
        buses, _ = RADIAL
        P = ptdf(*RADIAL, "A")
        assert P[0, buses.index("C")] == pytest.approx(-1.0)
        assert P[1, buses.index("C")] == pytest.approx(-1.0)


class TestLoop:
    """Three identical lines. The only case where the split is a real choice."""

    def test_transfer_splits_two_thirds_one_third(self):
        """A -> C: one line direct, two lines the long way round.

        Flow divides inversely to path reactance, so the direct path carries
        2/3 and the A-B-C path carries 1/3. Parallel springs, sharing a load
        inversely to their compliance.
        """
        buses, _ = LOOP
        P = ptdf(*LOOP, "C")
        a = buses.index("A")
        assert P[2, a] == pytest.approx(2 / 3)   # AC, direct
        assert P[0, a] == pytest.approx(1 / 3)   # AB, the long way
        assert P[1, a] == pytest.approx(1 / 3)   # BC, the long way

    def test_flow_into_the_slack_sums_to_one(self):
        """Conservation. Every injected MW arrives at the slack.

        Lines AC and BC are the two that terminate at C, and both point into
        it, so their PTDF entries add to exactly 1 MW delivered.
        """
        buses, _ = LOOP
        P = ptdf(*LOOP, "C")
        a = buses.index("A")
        assert P[1, a] + P[2, a] == pytest.approx(1.0)

    def test_stiffer_direct_path_takes_more(self):
        """Halve AC's reactance and it takes more of the transfer.

        b = 1/x, so half the reactance is twice the susceptance -- a stiffer
        spring. Direct path b = 20 against the series pair's b = 5, so the
        split moves from 2/3 to 4/5.
        """
        buses, branches = LOOP
        stiff = [
            branch(b.name, b.from_bus, b.to_bus,
                   b.reactance_pu / 2 if b.name == "AC" else b.reactance_pu)
            for b in branches
        ]
        P = ptdf(buses, stiff, "C")
        assert P[2, buses.index("A")] == pytest.approx(0.8)


class TestCase5Transcription:
    """configs/m3.yaml against tests/fixtures/case5.m. Provenance, not physics.

    MATPOWER numbers the buses and this repo names them after ski resorts.
    A single row transposed in that rename would surface at M3 as a PTDF sign
    nobody can explain, so it is caught here instead, against the file that
    was fetched verbatim.
    """

    def test_config_branches_match_case5(self):
        _, branches, _ = _case5()
        rows = _matpower_rows("branch")
        assert len(branches) == len(rows)
        for br, row in zip(branches, rows):
            f, t, x = BUS_LETTER[int(row[0])], BUS_LETTER[int(row[1])], row[3]
            assert (br.from_bus, br.to_bus) == (f, t)
            assert br.reactance_pu == pytest.approx(x)

    def test_config_limits_match_case5_rate_a(self):
        """rateA = 0 in MATPOWER means UNLIMITED, not zero-capacity.

        The config deliberately writes .inf instead, so it can be read without
        a lookup table in hand. This is the one place the two files are
        allowed to disagree literally, and the translation is asserted.
        """
        _, branches, _ = _case5()
        for br, row in zip(branches, _matpower_rows("branch")):
            rate_a = row[5]
            expected = np.inf if rate_a == 0 else rate_a
            assert br.limit_mw == expected

    def test_config_fleet_matches_case5(self):
        config = yaml.safe_load(CONFIG.read_text())
        fleet = list(config["fleet"].values())
        gen, cost = _matpower_rows("gen"), _matpower_rows("gencost")
        assert len(fleet) == len(gen) == len(cost)
        for spec, g, c in zip(fleet, gen, cost):
            assert spec["bus"] == BUS_LETTER[int(g[0])]
            assert float(spec["pmax_mw"]) == pytest.approx(g[8])
            # gencost model 2, n = 2: [model, startup, shutdown, n, c1, c0].
            assert float(spec["cost_usd_per_mwh"]) == pytest.approx(c[4])

    def test_config_load_matches_case5(self):
        config = yaml.safe_load(CONFIG.read_text())
        declared = {b: float(mw) for b, mw in config["load"]["mw"].items()}
        expected = {
            BUS_LETTER[int(row[0])]: row[2]
            for row in _matpower_rows("bus")
            if row[2] > 0
        }
        assert declared == expected

    def test_slack_is_the_matpower_reference(self):
        """case5.m marks bus 4 as type 3. Inherited, and nothing depends on it.

        TestNetworkInvariants.test_ptdf_differences_are_slack_invariant is
        what proves the choice does not leak into any answer.
        """
        _, _, slack = _case5()
        types = {BUS_LETTER[int(r[0])]: int(r[1]) for r in _matpower_rows("bus")}
        assert types[slack] == 3


class TestCase5Network:
    """The real thing. Values pinned so a refactor cannot drift them."""

    def test_shape(self):
        buses, branches, slack = _case5()
        assert ptdf(buses, branches, slack).shape == (6, 5)

    def test_slack_invariance_on_case5(self):
        buses, branches, _ = _case5()
        base = ptdf(buses, branches, "D")
        base = base - base[:, [0]]
        for slack in buses:
            P = ptdf(buses, branches, slack)
            assert np.allclose(P - P[:, [0]], base)

    def test_brighton_pushes_power_down_the_de_corridor(self):
        """The line the whole case turns on.

        Brighton at E is the cheapest unit and the furthest from load. 1 MW
        injected there sends 0.48 MW down DE toward D -- negative against the
        declared D -> E direction -- and DE is rated 240 MW. That is what
        stops Brighton from displacing the fleet.
        """
        buses, branches, slack = _case5()
        P = ptdf(buses, branches, slack)
        de = [b.name for b in branches].index("DE")
        assert P[de, buses.index("E")] == pytest.approx(-0.4805, abs=1e-4)


# --------------------------------------------------------------------------
# From here down a solver runs. Everything above is linear algebra with a
# hand-checkable answer; everything below asks an LP to agree with one.


def _clear(limits="config"):
    """Solve case5 and price it. limits="config" or "none".

    "none" raises every limit to infinity, which must collapse the case to a
    plain merit-order stack with one system price. That is the test that
    separates a topology bug from a pricing bug: if the uncongested case is
    already wrong, no amount of staring at mu will help.
    """
    scenario = build_scenario(CONFIG)
    buses = [b.name for b in scenario.buses]
    branches = list(scenario.branches)
    lines = [br.name for br in branches]
    P = ptdf(buses, branches, scenario.provenance["slack"])

    Fmax = {br.name: (np.inf if limits == "none" else br.limit_mw)
            for br in branches}
    gen_bus = {g.name: g.bus for g in scenario.generators}
    res = solve_dispatch_network_day(
        c={g.name: g.cost_usd_per_mwh for g in scenario.generators},
        Pmax={g.name: g.pmax_mw for g in scenario.generators},
        D=scenario.demand_by_bus(),
        gen_bus=gen_bus,
        buses=buses,
        PTDF=P,
        Fmax=Fmax,
    )
    return {
        "res": res,
        "buses": buses,
        "lines": lines,
        "PTDF": P,
        "Fmax": Fmax,
        "gen_bus": gen_bus,
        "demand": scenario.demand_by_bus(),
        "hour": scenario.hours[0],
    }


class TestClearingInvariants:
    """Holds forever, on any network, congested or not.

    These are the properties M5 will run on RTS-GMLC and M6 will run with
    binaries fixed. Nothing here mentions case5's numbers.
    """

    @pytest.mark.parametrize("limits", ["none", "config"])
    def test_energy_balances(self, limits):
        s = _clear(limits)
        t = s["hour"]
        served = sum(mw for (g, h), mw in s["res"]["p"].items() if h == t)
        load = sum(d[t] for d in s["demand"].values())
        assert served == pytest.approx(load)

    @pytest.mark.parametrize("limits", ["none", "config"])
    def test_no_line_exceeds_its_rating(self, limits):
        s = _clear(limits)
        for (l, t), mw in s["res"]["f"].items():
            assert abs(mw) <= s["Fmax"][l] + 1e-6

    @pytest.mark.parametrize("limits", ["none", "config"])
    def test_settlement_identity(self, limits):
        """The primary correctness test (CLAUDE.md).

            sum(load payments) - sum(generator revenue) == sum_l mu[l]*limit[l]

        It is an identity, not a coincidence: it falls out of LP duality, so
        a failure means PTDF construction, a sign convention, or dual
        extraction is wrong -- never that the market "didn't settle".

        An unlimited line contributes nothing because its mu is exactly zero,
        so inf * 0 has to be skipped rather than evaluated.
        """
        s = _clear(limits)
        t = s["hour"]
        mu = congestion_prices(s["res"])
        lmp = lmps(s["res"], s["buses"], s["lines"], s["PTDF"])

        payments = sum(s["demand"][i][t] * lmp[i, t] for i in s["buses"])
        revenue = sum(mw * lmp[s["gen_bus"][g], t]
                      for (g, h), mw in s["res"]["p"].items() if h == t)
        rent = sum(mu[l, t] * s["Fmax"][l]
                   for l in s["lines"] if np.isfinite(s["Fmax"][l]))

        assert payments - revenue == pytest.approx(rent, abs=1e-6)

        # The arithmetic above is deliberately hand-rolled, so the identity is
        # checked independently of the module that reports it. settle() must
        # then agree with it -- a wrapper that quietly computed something else
        # would otherwise pass every test that only ever calls the wrapper.
        gen_mw = {b: 0.0 for b in s["buses"]}
        for (g, h), mw in s["res"]["p"].items():
            if h == t:
                gen_mw[s["gen_bus"][g]] += mw
        money = settle(
            lmp={b: lmp[b, t] for b in s["buses"]},
            load_mw={b: s["demand"][b][t] for b in s["buses"]},
            gen_mw=gen_mw,
            mu={l: mu[l, t] for l in s["lines"]},
            limits=s["Fmax"],
        )
        assert money["payments"] == pytest.approx(payments)
        assert money["revenue"] == pytest.approx(revenue)
        assert money["mu_times_limit"] == pytest.approx(rent)
        assert money["residual"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("limits", ["none", "config"])
    def test_at_most_one_direction_binds_per_line(self, limits):
        """Why the limit is TWO one-sided constraints and not one ranged.

        Flow cannot sit at +Fmax and -Fmax at once, so at most one of the
        pair can bind and the other's dual is exactly zero. A ranged
        constraint would collapse them into a single dual whose sign no
        longer says which bound was active -- the top cause of a broken
        settlement identity.
        """
        s = _clear(limits)
        for key in s["res"]["mu_up"]:
            assert (abs(s["res"]["mu_up"][key]) < 1e-9
                    or abs(s["res"]["mu_dn"][key]) < 1e-9)

    def test_congestion_is_zero_when_nothing_binds(self):
        """M0's settlement identity, restated rather than deleted.

        At M0 the congestion term was pinned at zero because there was no
        network to congest. Here there is one, and it still comes to zero
        when no line binds -- so every LMP collapses back to lambda and the
        nodal market reduces to the single-bus market it generalises.
        """
        s = _clear("none")
        t = s["hour"]
        mu = congestion_prices(s["res"])
        lmp = lmps(s["res"], s["buses"], s["lines"], s["PTDF"])
        for l in s["lines"]:
            assert mu[l, t] == pytest.approx(0.0, abs=1e-9)
        for i in s["buses"]:
            assert lmp[i, t] == pytest.approx(s["res"]["lmbda"][t])


class TestCase5Uncongested:
    """Both limits raised to infinity. The control case.

    configs/m3.yaml states this expectation in prose; it is asserted here so
    the config and the code cannot drift apart silently.
    """

    def test_merit_order_stack(self):
        """1000 MW served cheapest-first, ignoring the network entirely.

        Brighton 600 at $10, Park City 170 at $15, Alta 40 at $14 -- both of
        the A units are cheaper than Solitude, so they fill before it -- and
        Solitude picks up the remaining 190 of its 520. Sundance at $40 never
        starts.
        """
        s = _clear("none")
        t = s["hour"]
        p = {g: mw for (g, h), mw in s["res"]["p"].items() if h == t}
        assert p == pytest.approx({
            "brighton": 600.0,
            "park_city": 170.0,
            "alta": 40.0,
            "solitude": 190.0,
            "sundance": 0.0,
        })

    def test_one_price_everywhere(self):
        """Solitude is marginal, so lambda is its offer -- at every bus."""
        s = _clear("none")
        t = s["hour"]
        lmp = lmps(s["res"], s["buses"], s["lines"], s["PTDF"])
        for i in s["buses"]:
            assert lmp[i, t] == pytest.approx(30.0)


class TestCase5Congested:
    """DE limited to 240 MW. The milestone's headline result.

    Dispatch is checked against MATPOWER's own OPF solution, carried in the
    Pg column of tests/fixtures/case5.m -- an answer produced by a different
    program, which is the only kind worth regressing against.
    """

    def test_dispatch_matches_matpower_opf(self):
        """Brighton backed down to 466.51, Solitude up to 323.49.

        Read out of the .m file rather than typed here, so the fixture stays
        the single source of truth. Those are the only two units that move:
        Alta and Park City are already at their caps and Sundance is still
        too expensive to start.
        """
        s = _clear("config")
        t = s["hour"]
        p = {g: mw for (g, h), mw in s["res"]["p"].items() if h == t}

        # mpc.gen columns: bus, Pg, ... -- the dispatch MATPOWER's OPF found.
        pg = [row[1] for row in _matpower_rows("gen")]
        expected = dict(zip(["alta", "park_city", "solitude", "sundance",
                             "brighton"], pg))
        assert p == pytest.approx(expected, abs=0.01)

    def test_de_is_the_only_binding_line(self):
        s = _clear("config")
        t = s["hour"]
        mu = congestion_prices(s["res"])
        assert mu["DE", t] == pytest.approx(62.32, abs=0.01)
        for l in s["lines"]:
            if l != "DE":
                assert mu[l, t] == pytest.approx(0.0, abs=1e-9)

    def test_de_flow_is_pinned_at_its_rating(self):
        """Negative because power flows E -> D, against the declared
        direction. The magnitude is what the rating constrains."""
        s = _clear("config")
        t = s["hour"]
        assert s["res"]["f"]["DE", t] == pytest.approx(-240.0)

    def test_nodal_prices(self):
        """The five LMPs. What M3 exists to produce.

        Derived by hand from one PTDF row before the LP was written, which is
        why they are written out rather than recomputed:

            LMP[i] = lambda + PTDF[DE, i] * mu[DE]

        E prices at Brighton's own offer: it is fenced in behind a saturated
        line, so an extra MW of load there is served by the cheapest thing
        trapped on that side. D prices at 39.94, nearly Sundance's $40,
        because relief has to come from the expensive side instead.
        """
        s = _clear("config")
        t = s["hour"]
        lmp = lmps(s["res"], s["buses"], s["lines"], s["PTDF"])
        assert {i: lmp[i, t] for i in s["buses"]} == pytest.approx({
            "A": 16.98,
            "B": 26.38,
            "C": 30.00,
            "D": 39.94,
            "E": 10.00,
        }, abs=0.01)

    def test_prices_separate_across_the_constraint(self):
        """The structural claim, independent of the exact numbers.

        Congestion is what makes a nodal market nodal. If DE binds and every
        bus still prices the same, the congestion component is not reaching
        the LMPs at all -- and the test above would pass on a coincidence.
        """
        s = _clear("config")
        t = s["hour"]
        lmp = lmps(s["res"], s["buses"], s["lines"], s["PTDF"])
        assert lmp["D", t] - lmp["E", t] > 1.0

    def test_congestion_costs_the_system_money(self):
        """The constrained solve is more expensive. It has to be.

        Same fleet, same load, strictly fewer feasible points -- so the
        optimum cannot improve. A congested case that clears cheaper than the
        unconstrained one means the limits were built with the wrong sign.
        """
        assert _clear("config")["res"]["cost"] > _clear("none")["res"]["cost"]


class TestClearEntryPoint:
    """src.model.clearing.clear() must agree with the hand-assembled pipeline.

    clear() exists so that a caller which is not a figure can run a market in
    one call. The risk in collapsing five steps into one name is that the
    wrapper drifts from the sequence it replaced -- a stale PTDF, a bus order
    recovered from the wrong list -- and no existing test would notice, because
    every existing test assembles the pipeline itself.

    So these compare the two routes rather than re-asserting case5's numbers.
    The numbers are already pinned in TestCase5Congested; what is unpinned is
    whether the shortcut takes the same road.
    """

    def test_matches_the_hand_assembled_pipeline(self):
        by_hand = _clear("config")
        t = by_hand["hour"]
        auto = clear(build_scenario(CONFIG))

        assert auto["buses"] == by_hand["buses"]
        assert auto["lines"] == by_hand["lines"]
        assert auto["slack"] == build_scenario(CONFIG).provenance["slack"]
        np.testing.assert_allclose(auto["PTDF"], by_hand["PTDF"])

        for g in by_hand["gen_bus"]:
            assert auto["dispatch"][g, t] == pytest.approx(
                by_hand["res"]["p"][g, t])
        for l in by_hand["lines"]:
            assert auto["flows"][l, t] == pytest.approx(by_hand["res"]["f"][l, t])

        want = lmps(by_hand["res"], by_hand["buses"], by_hand["lines"],
                    by_hand["PTDF"])
        for b in by_hand["buses"]:
            assert auto["lmp"][b, t] == pytest.approx(want[b, t])

    def test_settlement_identity_holds(self):
        """The primary correctness test, now reported by clear() itself.

        Every hour separately. A day-level sum could hide a positive residual
        in one hour cancelling a negative one in another.
        """
        cleared = clear(build_scenario(CONFIG))
        for t in cleared["hours"]:
            assert cleared["settlement"][t]["residual"] == pytest.approx(0.0, abs=1e-6)

    def test_limits_override_reaches_the_solver(self):
        """The lever a sweep or a slider pulls.

        Raising DE well above its unconstrained flow must un-bind it and
        collapse the case to one system price -- the same outcome as
        _clear("none"), reached by overriding a single line instead of all of
        them.
        """
        cleared = clear(build_scenario(CONFIG), limits={"DE": 1e4})
        t = cleared["hours"][0]
        assert binding_lines(cleared, t) == set()
        prices = {round(cleared["lmp"][b, t], 6) for b in cleared["buses"]}
        assert len(prices) == 1

    def test_tightening_a_limit_costs_more(self):
        """Monotonicity through the override, not through the config.

        Strictly fewer feasible points cannot produce a cheaper optimum. This
        is the invariant a UI slider will exercise thousands of times, so it is
        worth pinning at the entry point and not only at the solver.
        """
        loose = clear(build_scenario(CONFIG), limits={"DE": 1e4})
        tight = clear(build_scenario(CONFIG), limits={"DE": 100.0})
        assert tight["cost"] > loose["cost"]

    def test_unknown_line_override_is_rejected(self):
        """A typo in a line name must not silently clear the base case.

        The override is the one argument that will arrive from outside this
        repo. Ignoring an unrecognised key would answer a question nobody
        asked, and it would look like a correct answer.
        """
        with pytest.raises(ValueError, match="unknown line"):
            clear(build_scenario(CONFIG), limits={"NOPE": 100.0})

    def test_slack_choice_moves_prices_by_a_constant(self):
        """CLAUDE.md trap 2, asserted at the entry point.

        Changing the slack shifts every LMP by one constant and moves no
        dispatch and no flow. Price DIFFERENCES are what the market means.
        """
        base = clear(build_scenario(CONFIG), slack="D")
        alt = clear(build_scenario(CONFIG), slack="A")
        t = base["hours"][0]

        for g in base["gen_bus"]:
            assert alt["dispatch"][g, t] == pytest.approx(base["dispatch"][g, t])
        for l in base["lines"]:
            assert alt["flows"][l, t] == pytest.approx(base["flows"][l, t])

        shifts = {round(alt["lmp"][b, t] - base["lmp"][b, t], 6)
                  for b in base["buses"]}
        assert len(shifts) == 1

    def test_single_bus_scenario_is_refused(self):
        """M0's scenarios have no network, and clear() must say so.

        Falling through to the network solver with zero branches would build a
        PTDF with no rows and return a price that looks fine and means nothing.
        """
        config = load_config(CONFIG)
        config.pop("network")
        config["fleet"] = {n: {k: v for k, v in spec.items() if k != "bus"}
                           for n, spec in config["fleet"].items()}
        config["load"] = {"source": "static", "mw": {"bus1": 600.0}}
        with pytest.raises(ValueError, match="needs a network"):
            clear(scenario_from_config(config))


class TestConfigFromDict:
    """scenario_from_config must be the same function build_scenario is.

    The split exists so a config can arrive as a dict from a POST body or a
    sweep. If the dict path ever diverged from the file path, every such caller
    would be testing something other than what this repo's tests cover.
    """

    def test_dict_and_file_paths_agree(self):
        from_file = build_scenario(CONFIG)
        from_dict = scenario_from_config(load_config(CONFIG))

        assert from_dict.generators == from_file.generators
        assert from_dict.loads == from_file.loads
        assert from_dict.buses == from_file.buses
        assert from_dict.branches == from_file.branches

    def test_origin_is_recorded(self):
        """Provenance still says where the numbers came from with no file."""
        assert str(CONFIG) in build_scenario(CONFIG).provenance["config"]
        assert scenario_from_config(
            load_config(CONFIG), origin="POST /clear"
        ).provenance["config"] == "POST /clear"


class TestBranchValidation:
    """A Branch that cannot exist must not be constructible. Holds forever.

    These guard the one failure mode the rest of this file cannot catch. Every
    other test here compares a computed number against a known one, which only
    works when the inputs are physical. A negative reactance is not physical,
    raises nothing anywhere downstream, and yields prices that look ordinary --
    so the check has to sit at construction, where the bad value enters.

    Written at M3 because M3 is where Branch stops being a placeholder, but
    nothing here is about case5 or about the DC model specifically. They stay
    true at M5 on RTS-GMLC and at M8 on a real fleet.
    """

    def _branch(self, **kw):
        spec = dict(name="AB", from_bus="A", to_bus="B",
                    reactance_pu=0.0281, limit_mw=400.0)
        spec.update(kw)
        return Branch(**spec)

    def test_a_good_branch_still_builds(self):
        """The guard must not reject the case5 branches it ships alongside."""
        assert self._branch().reactance_pu == 0.0281

    @pytest.mark.parametrize("x", [-0.0281, -1.0])
    def test_negative_reactance_is_refused(self, x):
        """The silent-wrong-answer case.

        b = 1/x goes negative, B_bus becomes indefinite rather than positive
        semidefinite, and the LP solves happily on a network where power flows
        uphill. Nothing downstream raises; the prices are simply wrong.
        """
        with pytest.raises(ValueError, match="reactance_pu must be > 0"):
            self._branch(reactance_pu=x)

    def test_zero_reactance_is_refused(self):
        """Caught here rather than as a ZeroDivisionError inside b_branch.

        The arithmetic failure is real but names no branch, and a caller
        editing a network sees a traceback through two modules instead of the
        line they typed.
        """
        with pytest.raises(ValueError, match="reactance_pu must be > 0"):
            self._branch(reactance_pu=0.0)

    @pytest.mark.parametrize("mw", [0.0, -1.0])
    def test_non_positive_limit_is_refused(self, mw):
        """A line that can carry nothing is a line that should be deleted.

        Left in place it makes the solve infeasible, and "infeasible" does not
        tell anyone which of their edits caused it.
        """
        with pytest.raises(ValueError, match="limit_mw must be > 0"):
            self._branch(limit_mw=mw)

    def test_unlimited_line_is_allowed(self):
        """inf is a deliberate modelling choice, not a missing value.

        _clear("none") depends on it: raising every limit to infinity is how
        this suite separates a topology bug from a pricing bug.
        """
        assert np.isinf(self._branch(limit_mw=np.inf).limit_mw)

    def test_self_loop_is_refused(self):
        """Both endpoints at one bus gives an all-zero incidence row.

        The flow is then identically zero and the branch contributes nothing
        to B_bus -- valid arithmetic describing no line at all.
        """
        with pytest.raises(ValueError, match="both 'A'"):
            self._branch(to_bus="A")


class TestComponents:
    """Connected components. A pure graph property, true on any network.

    Uses the three reference networks at the top of this file plus the cuts
    that make them fall apart, because a components() that cannot tell a
    3-line loop from three isolated buses is worse than no check at all.
    """

    def test_a_connected_network_is_one_component(self):
        buses, branches = LOOP
        assert components(buses, branches) == [["A", "B", "C"]]

    def test_a_bus_with_no_branches_stands_alone(self):
        buses, branches = RADIAL
        assert components(buses + ["D"], branches) == [["A", "B", "C"], ["D"]]

    def test_cutting_a_radial_line_splits_it(self):
        """A --AB-- B --BC-- C, with BC removed. The M3 failure a UI produces."""
        buses, branches = RADIAL
        kept = [br for br in branches if br.name != "BC"]
        assert components(buses, kept) == [["A", "B"], ["C"]]

    def test_cutting_one_loop_line_keeps_it_whole(self):
        """A loop survives losing any single line. That is what a loop is for."""
        buses, branches = LOOP
        for drop in [br.name for br in branches]:
            kept = [br for br in branches if br.name != drop]
            assert components(buses, kept) == [["A", "B", "C"]], drop

    def test_no_branches_means_every_bus_is_its_own_island(self):
        buses, _ = LOOP
        assert components(buses, []) == [["A"], ["B"], ["C"]]

    def test_order_follows_the_bus_order_given(self):
        """Reproducible, not set-ordered.

        The bus list fixes the column order of every matrix in src/network/,
        so anything derived from it that reorders silently is a trap.
        """
        _, branches = RADIAL
        assert components(["C", "B", "A"], branches) == [["C", "B", "A"]]

    def test_case5_is_one_component(self):
        """CLAUDE.md's PTDF note: one slack per connected component, and
        case5 is one. Asserted rather than assumed."""
        scenario = build_scenario(CONFIG)
        assert len(components([b.name for b in scenario.buses],
                              list(scenario.branches))) == 1


class TestScenarioTopologyValidation:
    """A Scenario whose network cannot be priced must say so in a sentence.

    Every case here used to reach numpy or a bare KeyError two modules away.
    Duplicate bus names and a network cut in half produced the SAME message --
    "Singular matrix" -- from two unrelated mistakes, which is the worst thing
    an error can do to someone editing topology.

    Note where each check lives, because the split is deliberate:

        Branch.__post_init__   what one line can check alone (reactance, limit)
        Scenario.__post_init__ what needs the bus list (duplicates, endpoints)
        clear()                what needs the slack (connectivity)

    Connectivity is not a Scenario invariant. Two islands are a real power
    system; they are simply two markets, each needing its own slack and its
    own energy balance. This repo prices one, and clear() is where that
    assumption is made.
    """

    def _config(self):
        return load_config(CONFIG)

    def test_duplicate_bus_name_is_named(self):
        config = self._config()
        config["network"]["buses"].append("E")
        with pytest.raises(ValueError, match=r"duplicate bus names: \['E'\]"):
            scenario_from_config(config)

    def test_branch_endpoint_must_be_a_declared_bus(self):
        config = self._config()
        config["network"]["branches"]["DE"]["to"] = "Q"
        with pytest.raises(ValueError, match="branch DE: to_bus 'Q' is not a declared bus"):
            scenario_from_config(config)

    def test_islanded_bus_names_the_bus_and_the_slack(self):
        """Delete every line touching E. The single most likely UI edit.

        Checked at ptdf() rather than further up, because ptdf() owns the
        inverse that fails and already holds the slack. Anything that builds a
        PTDF gets the diagnosis, not just the one entry point that happened to
        remember to ask.
        """
        config = self._config()
        branches = config["network"]["branches"]
        for name in [n for n, spec in branches.items()
                     if "E" in (spec["from"], spec["to"])]:
            del branches[name]
        scenario = scenario_from_config(config)
        with pytest.raises(ValueError, match=r"disconnected.*\['E'\].*slack 'D'"):
            ptdf([b.name for b in scenario.buses], list(scenario.branches), "D")

    def test_a_valid_network_passes_all_of_it(self):
        """The guards must not reject the case they ship alongside."""
        assert len(build_scenario(CONFIG).branches) == 6


class TestSettle:
    """settle() on numbers small enough to check by hand. Holds forever.

    The case5 test above proves settle() agrees with the solver. These prove
    it is doing the arithmetic the identity names, on inputs with no LP behind
    them -- so a sign error cannot hide behind a case where both halves happen
    to be zero.
    """

    def test_no_congestion_means_no_rent(self):
        """One price everywhere: load pays exactly what generation collects."""
        money = settle(
            lmp={"A": 30.0, "B": 30.0},
            load_mw={"A": 0.0, "B": 100.0},
            gen_mw={"A": 100.0, "B": 0.0},
            mu={"AB": 0.0},
            limits={"AB": 400.0},
        )
        assert money["payments"] == 3000.0
        assert money["revenue"] == 3000.0
        assert money["congestion_rent"] == 0.0
        assert money["residual"] == 0.0

    def test_congestion_rent_is_the_price_gap_times_the_flow(self):
        """Two buses, one binding line, every number hand-checkable.

        Load at B pays $40, generation at A collects $10, and the line carries
        100 MW at its limit. The $30 gap on 100 MW is $3000, and mu on a
        binding line is exactly that gap -- so both halves of the identity
        come to 3000 by different routes.
        """
        money = settle(
            lmp={"A": 10.0, "B": 40.0},
            load_mw={"A": 0.0, "B": 100.0},
            gen_mw={"A": 100.0, "B": 0.0},
            mu={"AB": 30.0},
            limits={"AB": 100.0},
        )
        assert money["payments"] == 4000.0
        assert money["revenue"] == 1000.0
        assert money["congestion_rent"] == 3000.0
        assert money["mu_times_limit"] == 3000.0
        assert money["residual"] == 0.0

    def test_unlimited_line_is_skipped_not_multiplied(self):
        """inf * 0 is nan, and one nan destroys the residual silently.

        The guard is the reason this function takes limits at all rather than
        being handed a rent total, so it is worth its own test.
        """
        money = settle(
            lmp={"A": 30.0, "B": 30.0},
            load_mw={"A": 0.0, "B": 100.0},
            gen_mw={"A": 100.0, "B": 0.0},
            mu={"AB": 0.0},
            limits={"AB": np.inf},
        )
        assert money["mu_times_limit"] == 0.0
        assert not np.isnan(money["residual"])

    def test_a_broken_price_shows_up_as_a_residual(self):
        """The failure mode the identity exists to catch.

        Same congested case, but the price at B is wrong by $5. Nothing raises
        -- the arithmetic is all valid -- and the residual is the only thing
        that says so. That is exactly how a PTDF sign error presents.
        """
        money = settle(
            lmp={"A": 10.0, "B": 45.0},
            load_mw={"A": 0.0, "B": 100.0},
            gen_mw={"A": 100.0, "B": 0.0},
            mu={"AB": 30.0},
            limits={"AB": 100.0},
        )
        assert money["residual"] == pytest.approx(500.0)

    def test_a_bus_carrying_both_load_and_generation_nets_correctly(self):
        """Generation and load at one bus are separate terms, not a net.

        They are settled at the same price so the arithmetic agrees either
        way here -- but netting them first would silently lose the gross
        quantities a settlement statement has to report.
        """
        money = settle(
            lmp={"A": 20.0},
            load_mw={"A": 60.0},
            gen_mw={"A": 60.0},
            mu={},
            limits={},
        )
        assert money["payments"] == 1200.0
        assert money["revenue"] == 1200.0
        assert money["residual"] == 0.0
