r"""M3 network tests: incidence, susceptance, PTDF. No solver, no prices yet.

Everything here is hand-checkable. The point of M3 is that exactly one new
failure surface opens -- the network -- so these tests pin the linear algebra
down before an LP is ever asked to trust it. A transposed row or a flipped
sign caught here is an afternoon; caught at M4 it is an inexplicable LMP.

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
M4, at M7, and on any network anyone ever hands this repo -- those are
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

from src.model.inputs import Branch
from src.network.ptdf import ptdf
from src.network.topology import b_bus, b_branch, b_flow, incidence

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
