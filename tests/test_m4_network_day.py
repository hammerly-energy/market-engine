r"""M4 acceptance tests: the M3 network across a full day.

M4 is the cross product of two things already proved. M1 established that 24
hours on one bus are separable; M3 established the network in a single hour.
Nothing new is formulated here, so the risk is not that the formulation is
wrong -- it is that joining the two quietly breaks one of them:

    a constraint accidentally written across all t
        -> the hours couple through the flow limits, M1's separability dies,
           and every price is subtly wrong in a way no single hour reveals

    a PTDF built once and indexed per hour with a stale bus order
        -> prices land at the wrong buses, consistently, in all 24 hours

    a settlement identity checked on the DAY instead of per HOUR
        -> a positive residual in one hour cancels a negative one in another
           and a broken solve reports a clean zero

So the tests below are mostly the M1 and M3 tests re-run against a day, plus
the one property that is genuinely new: congestion appears in some hours and
not others, and the price regime changes with it.

              M1                    M3                    M4
        24 hours, 1 bus  x   1 hour, 5 buses   =   24 hours, 5 buses
        p[g,t]               p[g], f[l]            p[g,t], f[l,t]
        lambda[t]            lambda + PTDF.mu      lambda[t] + PTDF.mu[t]

Grouped by lifespan, not by milestone. TestForever holds on any network in any
milestone. TestSeparability is expected to FAIL at M6, for the same reason M1's
copy of it is -- startup cost and min up/down time couple the hours on purpose.
"""

import numpy as np
import pytest

from src.ingest.scenario import build_scenario, load_config
from src.model.clearing import binding_lines, clear
from src.model.dispatch import solve_dispatch_network_day

M3_CONFIG = "configs/m3.yaml"
M4_CONFIG = "configs/m4.yaml"

# The hour where shape == 1.0. Not discovered at runtime: the whole point is
# that this hour is m3's case, and a test that located it by looking for the
# maximum would still pass if the shape were rescaled away from m3 entirely.
PEAK_HOUR = 18


@pytest.fixture(scope="module")
def day():
    return clear(build_scenario(M4_CONFIG))


@pytest.fixture(scope="module")
def scenario():
    return build_scenario(M4_CONFIG)


# ------------------------------------------------------------ the M4 goal

class TestSettlementIdentity:
    """The primary correctness test, per hour. The M4 acceptance criterion.

        sum(load payments) - sum(gen revenue) == sum_l mu[l] * limit[l]

    Per hour is the whole point. M0 through M3 had one hour or no network, so
    there was nothing for a day-level sum to hide. M4 is the first milestone
    where summing first could cancel a real error, and the first where getting
    this wrong would still look right.
    """

    def test_holds_in_every_hour(self, day):
        for t in day["hours"]:
            assert day["settlement"][t]["residual"] == pytest.approx(0.0, abs=1e-6), (
                f"hour {t}: residual {day['settlement'][t]['residual']}"
            )

    def test_congestion_rent_is_zero_exactly_when_nothing_binds(self, day):
        """The two halves of the identity move together, or neither does.

        An hour with no binding line must have zero congestion rent AND equal
        payments and revenue. An hour with a binding line must have positive
        rent -- rent is what the identity balances, so a binding constraint
        earning nothing means mu was extracted with the wrong sign.
        """
        for t in day["hours"]:
            rent = day["settlement"][t]["congestion_rent"]
            if binding_lines(day, t):
                assert rent > 0, f"hour {t}: DE binds but rent is {rent}"
            else:
                assert rent == pytest.approx(0.0, abs=1e-9)


class TestCongestionSwitches:
    """What M4 buys that M3 could not: a day with more than one price regime.

    M3 is one hour, so congestion is either on or off for the whole scenario.
    A day crosses the threshold, and crossing it is the new observable.
    """

    def test_some_hours_congested_and_some_not(self, day):
        congested = [t for t in day["hours"] if binding_lines(day, t)]
        clear_hours = [t for t in day["hours"] if not binding_lines(day, t)]
        assert congested, "no hour congested: the shape never crosses the threshold"
        assert clear_hours, "every hour congested: the trough is too high"

    def test_uncongested_hours_have_one_price(self, day):
        """No binding line means no congestion component, so LMP == lambda."""
        for t in day["hours"]:
            if binding_lines(day, t):
                continue
            for b in day["buses"]:
                assert day["lmp"][b, t] == pytest.approx(day["lmbda"][t])

    def test_congested_hours_separate(self, day):
        """A binding line must actually move prices apart.

        mu > 0 with every LMP still equal would mean the congestion component
        was computed and then dropped on the floor.
        """
        for t in day["hours"]:
            if not binding_lines(day, t):
                continue
            prices = {round(day["lmp"][b, t], 6) for b in day["buses"]}
            assert len(prices) > 1, f"hour {t}: line binds but all LMPs equal"

    def test_congestion_follows_load(self, day, scenario):
        """Congested hours are the heavy ones. Nothing else drives it here.

        No ramp limits, no commitment, no hourly capacity -- load is the only
        thing that varies across the day, so the congested set must be exactly
        the hours above some load threshold. A congested trough hour would mean
        something other than demand is moving, which at M4 is a bug.
        """
        D = scenario.demand_by_bus()
        total = {t: sum(D[b][t] for b in D) for t in day["hours"]}
        congested = [total[t] for t in day["hours"] if binding_lines(day, t)]
        clear_hours = [total[t] for t in day["hours"] if not binding_lines(day, t)]
        assert min(congested) > max(clear_hours)


class TestSeparability:
    """The joint 24-hour solve must equal 24 independent one-hour solves.

    M1's property, re-established with a network in the way. Nothing couples
    the hours yet -- the network couples BUSES within an hour and never one
    hour to the next -- so this must hold exactly, not approximately.

    Expected to FAIL at M6. Startup cost and min up/down time couple the hours
    deliberately, and that failure is an M6 acceptance criterion rather than a
    regression. Superseded there, not deleted.
    """

    def test_hours_solve_independently(self, day, scenario):
        c, Pmax = scenario.cost(), scenario.pmax()
        gen_bus = {g.name: g.bus for g in scenario.generators}
        D = scenario.demand_by_bus()

        for t in day["hours"]:
            alone = solve_dispatch_network_day(
                c=c, Pmax=Pmax,
                D={b: {t: D[b][t]} for b in D},
                gen_bus=gen_bus, buses=day["buses"],
                PTDF=day["PTDF"], Fmax=day["limits"],
            )
            for g in gen_bus:
                assert alone["p"][g, t] == pytest.approx(day["dispatch"][g, t]), (
                    f"hour {t}, {g}: joint and single-hour dispatch differ"
                )
            assert alone["lmbda"][t] == pytest.approx(day["lmbda"][t])

    def test_day_cost_is_the_sum_of_hourly_costs(self, day, scenario):
        """The objective is additive across hours because nothing links them."""
        c = scenario.cost()
        by_hand = sum(
            c[g] * day["dispatch"][g, t]
            for g in c for t in day["hours"]
        )
        assert day["cost"] == pytest.approx(by_hand)


# -------------------------------------------------- M3 carried forward

class TestM3RegressionAtPeak:
    """Hour 18 IS configs/m3.yaml, and must still produce M3's published answer.

    This is why the shape peaks at exactly 1.0. It keeps the PJM 5-bus
    reference LMPs a live test inside M4 rather than a historical note in M3 --
    if the day-indexed code path breaks the network, it breaks here, at a bus
    and an hour with a published number next to it.
    """

    def test_config_network_matches_m3(self):
        """Drift guard on the duplicated block. See the m4 config description."""
        assert load_config(M4_CONFIG)["network"] == load_config(M3_CONFIG)["network"]

    def test_config_fleet_matches_m3(self):
        assert load_config(M4_CONFIG)["fleet"] == load_config(M3_CONFIG)["fleet"]

    def test_peak_hour_load_matches_m3(self):
        """shape[18] == 1.0, so the peak hour's demand is m3's, bus by bus."""
        m3_load = load_config(M3_CONFIG)["load"]["mw"]
        D = build_scenario(M4_CONFIG).demand_by_bus()
        for bus, mw in m3_load.items():
            assert D[bus][PEAK_HOUR] == pytest.approx(float(mw))

    def test_peak_hour_reproduces_m3_prices(self, day):
        """The published PJM 5-bus LMPs, transcribed from the m3 notes."""
        expected = {"A": 16.98, "B": 26.38, "C": 30.00, "D": 39.94, "E": 10.00}
        for bus, lmp in expected.items():
            assert day["lmp"][bus, PEAK_HOUR] == pytest.approx(lmp, abs=5e-3)
        assert day["lmbda"][PEAK_HOUR] == pytest.approx(39.94, abs=5e-3)

    def test_peak_hour_reproduces_m3_dispatch(self, day):
        """MATPOWER's own OPF answer for case5, via the m3 notes."""
        assert day["dispatch"]["brighton", PEAK_HOUR] == pytest.approx(466.51, abs=1e-2)
        assert day["dispatch"]["solitude", PEAK_HOUR] == pytest.approx(323.49, abs=1e-2)

    def test_peak_hour_matches_a_direct_m3_solve(self, day):
        """Not just the transcribed numbers -- the same code on the same config.

        The numbers above could both be wrong in the same way. This compares
        M4's hour 18 against M3 solved directly, so the two paths have to agree
        with each other as well as with the publication.
        """
        m3 = clear(build_scenario(M3_CONFIG))
        t3 = m3["hours"][0]
        for b in m3["buses"]:
            assert day["lmp"][b, PEAK_HOUR] == pytest.approx(m3["lmp"][b, t3])
        for l in m3["lines"]:
            assert day["flows"][l, PEAK_HOUR] == pytest.approx(m3["flows"][l, t3])


# ------------------------------------------------- invariants. hold forever

class TestForever:
    """True on any network, in any hour, at any milestone. Nothing supersedes."""

    def test_energy_balance_every_hour(self, day, scenario):
        D = scenario.demand_by_bus()
        for t in day["hours"]:
            gen = sum(day["dispatch"][g, t] for g in day["gen_bus"])
            load = sum(D[b][t] for b in D)
            assert gen == pytest.approx(load)

    def test_dispatch_within_capacity(self, day, scenario):
        Pmax = scenario.pmax()
        for g in day["gen_bus"]:
            for t in day["hours"]:
                assert -1e-9 <= day["dispatch"][g, t] <= Pmax[g] + 1e-9

    def test_flows_within_limits(self, day):
        for l in day["lines"]:
            limit = day["limits"][l]
            for t in day["hours"]:
                assert abs(day["flows"][l, t]) <= limit + 1e-6

    def test_lmp_equals_the_assembly_formula(self, day):
        """LMP[i,t] = lambda[t] + sum_l PTDF[l,i] * mu[l,t], independently.

        Recomputed here from the raw duals rather than trusting pricing.py, so
        a sign convention that flipped inside the assembly would surface as a
        mismatch instead of as prices that merely look odd.
        """
        idx_b = {b: i for i, b in enumerate(day["buses"])}
        for t in day["hours"]:
            for b in day["buses"]:
                congestion = sum(
                    day["PTDF"][li, idx_b[b]] * day["mu"][l, t]
                    for li, l in enumerate(day["lines"])
                )
                assert day["lmp"][b, t] == pytest.approx(
                    day["lmbda"][t] + congestion, abs=1e-9
                )

    def test_flows_match_ptdf_times_injection(self, day, scenario):
        """The network the prices came from is the network the flows obey.

        f = PTDF * (generation - load), per hour. If this fails, the solver's
        flow variable and the PTDF used to assemble prices are two different
        networks, and the LMPs are describing a grid that is not there.
        """
        D = scenario.demand_by_bus()
        idx_b = {b: i for i, b in enumerate(day["buses"])}
        for t in day["hours"]:
            inj = np.zeros(len(day["buses"]))
            for g, bus in day["gen_bus"].items():
                inj[idx_b[bus]] += day["dispatch"][g, t]
            for b in day["buses"]:
                inj[idx_b[b]] -= D[b][t]
            f = day["PTDF"] @ inj
            for li, l in enumerate(day["lines"]):
                assert day["flows"][l, t] == pytest.approx(f[li], abs=1e-6)

    def test_no_negative_dispatch(self, day):
        for g in day["gen_bus"]:
            for t in day["hours"]:
                assert day["dispatch"][g, t] >= -1e-9


class TestProfileLoadSource:
    """The config idiom M4 introduces: peak_mw x shape.

    Written at M4 because M4 is where it appears, but the guards matter most
    later -- RTS-GMLC uses the same shape (an area profile times a bus
    participation factor) at M5, with 73 buses instead of three to eyeball.
    """

    def test_shape_must_peak_at_one(self):
        """A shape topping out below 1.0 silently rescales every bus.

        It would still solve, and it would move hour 18 off m3's case without
        any test noticing -- so the M3 regression above would be checking a
        scenario that is no longer m3.
        """
        from src.ingest.scenario import scenario_from_config
        config = load_config(M4_CONFIG)
        config["load"]["shape"] = [0.5, 0.95]
        with pytest.raises(ValueError, match="peak at exactly 1.0"):
            scenario_from_config(config)

    def test_zero_hour_is_rejected(self):
        from src.ingest.scenario import scenario_from_config
        config = load_config(M4_CONFIG)
        config["load"]["shape"] = [1.0, 0.0]
        with pytest.raises(ValueError, match="must be > 0"):
            scenario_from_config(config)

    def test_every_bus_scales_by_the_same_shape(self, scenario):
        """One shape, applied uniformly. Bus shares stay fixed all day.

        M4 varies the LEVEL of load, not its distribution. If the ratio between
        two buses drifted across the day, congestion would be responding to
        something the config never declared.
        """
        D = scenario.demand_by_bus()
        ratios = {
            t: D["B"][t] / D["D"][t] for t in scenario.hours
        }
        assert len(set(round(r, 12) for r in ratios.values())) == 1

    def test_provenance_records_the_shape(self, scenario):
        p = scenario.provenance
        assert p["load_source"] == "profile"
        assert p["horizon_hours"] == 24
        assert p["shape_peak_hour"] == PEAK_HOUR
