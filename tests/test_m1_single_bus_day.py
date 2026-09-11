"""M1 acceptance tests: same single-bus fleet, 24 hours, joint solve.

The milestone is separability. Nothing written so far couples one hour to the
next, so the constraint matrix is block diagonal:

           p[.,0]  p[.,1]  p[.,2]   ...   p[.,23]
         +--------------------------------------+
   bal 0 | ####                                 |  = D[0]   -> lambda[0]
   bal 1 |        ####                          |  = D[1]   -> lambda[1]
   bal 2 |               ####                   |  = D[2]   -> lambda[2]
     :   |                        .             |
   bal23 |                              ####    |  = D[23]  -> lambda[23]
         +--------------------------------------+
            no entry ever lands off the diagonal

A joint 24-hour solve must therefore reproduce 24 independent single-hour
solves to the last digit. TestSeparability asserts exactly that.

TestSeparability is expected to FAIL at M6. Startup cost and min up/down time
put entries off the diagonal, the blocks fuse, and hour 17's dual starts to
carry information about hour 3. That failure is an M6 acceptance criterion,
not a regression -- do not delete this class when it goes red, and be able to
explain which constraint broke it.

Load shape is hardcoded and synthetic; M2 replaces it with EIA-930 across the
ingest boundary. Every value sits off a capacity breakpoint (100, 200, 300 MW),
so every hour has a unique marginal unit and a unique dual.
"""

import pytest

from src.model.dispatch import solve_dispatch, solve_dispatch_day

COST = {"g1": 20.0, "g2": 35.0, "g3": 80.0}     # $/MWh
PMAX = {"g1": 100.0, "g2": 100.0, "g3": 100.0}  # MW
TOTAL_CAPACITY = sum(PMAX.values())             # 300 MW

HOURS = list(range(24))

#            00   01   02   03   04   05   06   07   08   09   10   11
LOAD_MW = {t: mw for t, mw in enumerate([
             62,  58,  55,  54,  57,  68,  92, 118, 146, 162, 171, 178,
            183, 186, 181, 176, 188, 214, 247, 263, 238, 192, 141,  88,
#            12   13   14   15   16   17   18   19   20   21   22   23
])}

# Duals from two different LPs agree to solver noise, not to the bit.
EXACT = dict(rel=0, abs=1e-9)


@pytest.fixture(scope="module")
def day():
    """One joint solve, reused by every test in the module."""
    return solve_dispatch_day(COST, PMAX, LOAD_MW)


@pytest.fixture(scope="module")
def hourly():
    """Twenty-four independent single-hour solves. The reference answer."""
    return {t: solve_dispatch(COST, PMAX, LOAD_MW[t]) for t in HOURS}


# ---------------------------------------------------------------- separability

class TestSeparability:
    """The M1 goal. Superseded at M6 -- see the module docstring."""

    def test_dispatch_matches_independent_hourly_solves(self, day, hourly):
        for t in HOURS:
            for g in COST:
                assert day["p"][g, t] == pytest.approx(hourly[t]["p"][g], **EXACT), \
                    f"hour {t}, unit {g}"

    def test_duals_match_independent_hourly_solves(self, day, hourly):
        for t in HOURS:
            assert day["lmbda"][t] == pytest.approx(hourly[t]["lmbda"], **EXACT), \
                f"hour {t}"

    def test_total_cost_is_the_sum_of_hourly_costs(self, day, hourly):
        assert day["cost"] == pytest.approx(
            sum(hourly[t]["cost"] for t in HOURS), **EXACT
        )

    def test_one_hour_horizon_reduces_to_the_single_hour_solve(self):
        """The 24-hour model with T = {0} is the M0 model."""
        r = solve_dispatch_day(COST, PMAX, {0: 150.0})
        assert r["p"] == pytest.approx({("g1", 0): 100.0, ("g2", 0): 50.0,
                                        ("g3", 0): 0.0})
        assert r["lmbda"][0] == pytest.approx(35.0)

    def test_perturbing_one_hour_leaves_every_other_hour_untouched(self, day):
        """No constraint spans two hours, so hour 19 cannot reach hour 3.

        This is the strongest form of the claim: it holds even when the
        perturbed hour is the system peak.
        """
        bumped = {**LOAD_MW, 19: LOAD_MW[19] - 40.0}
        r = solve_dispatch_day(COST, PMAX, bumped)
        for t in HOURS:
            if t == 19:
                continue
            assert r["lmbda"][t] == pytest.approx(day["lmbda"][t], **EXACT), \
                f"hour {t} moved when only hour 19 changed"
            for g in COST:
                assert r["p"][g, t] == pytest.approx(day["p"][g, t], **EXACT)

    def test_hours_may_be_solved_in_any_order(self):
        """t is a label on a block, not a position in a sequence.

        Reversing the horizon relabels the blocks and nothing else. When ramp
        limits arrive, this stops being true -- time acquires a direction.
        """
        reversed_load = {t: LOAD_MW[23 - t] for t in HOURS}
        r = solve_dispatch_day(COST, PMAX, reversed_load)
        forward = solve_dispatch_day(COST, PMAX, LOAD_MW)
        for t in HOURS:
            assert r["lmbda"][t] == pytest.approx(forward["lmbda"][23 - t], **EXACT)


# -------------------------------------------------------------------- pricing

class TestPricing:
    """Every hour prices independently off its own marginal unit."""

    def test_dual_equals_the_marginal_unit_cost_in_every_hour(self, day):
        for t in HOURS:
            marginal = [g for g in COST
                        if 1e-6 < day["p"][g, t] < PMAX[g] - 1e-6]
            assert len(marginal) == 1, f"hour {t}: no unique marginal unit"
            assert day["lmbda"][t] == pytest.approx(COST[marginal[0]])

    def test_price_shape_follows_the_load_shape(self, day):
        """Overnight prices at g1's offer, the peak at g3's."""
        assert day["lmbda"][3] == pytest.approx(20.0)    # trough, 54 MW
        assert day["lmbda"][13] == pytest.approx(35.0)   # shoulder, 186 MW
        assert day["lmbda"][19] == pytest.approx(80.0)   # peak, 263 MW

    def test_equal_load_hours_clear_at_equal_prices(self):
        """Two hours with the same demand are the same LP block.

        Price is a function of the hour's demand alone. Nothing about position
        in the day enters. This is exactly what M6 removes.
        """
        flat = {t: 150.0 for t in HOURS}
        r = solve_dispatch_day(COST, PMAX, flat)
        assert all(r["lmbda"][t] == pytest.approx(35.0, **EXACT) for t in HOURS)


# ----------------------------------------------------------------- invariants

class TestInvariants:
    """Hold forever, at every milestone. Nothing here is superseded."""

    def test_energy_balance_holds_in_every_hour(self, day):
        for t in HOURS:
            assert sum(day["p"][g, t] for g in COST) == pytest.approx(LOAD_MW[t])

    def test_no_generator_exceeds_its_capacity_or_goes_negative(self, day):
        for t in HOURS:
            for g in COST:
                assert -1e-9 <= day["p"][g, t] <= PMAX[g] + 1e-9

    def test_production_cost_matches_dispatch(self, day):
        assert day["cost"] == pytest.approx(
            sum(COST[g] * day["p"][g, t] for g in COST for t in HOURS)
        )

    def test_settlement_identity_holds_in_every_hour(self, day):
        """sum(load payments) - sum(gen revenue) == sum(mu[l] * limit[l])

        Still one bus, so the congestion term is exactly zero and the two
        sides must match hour by hour. M3 replaces that zero with a real
        congestion rent.
        """
        for t in HOURS:
            load_payment = LOAD_MW[t] * day["lmbda"][t]
            gen_revenue = sum(day["p"][g, t] for g in COST) * day["lmbda"][t]
            congestion_rent = 0.0  # no branches at M1
            assert load_payment - gen_revenue == pytest.approx(congestion_rent)

    def test_settlement_identity_holds_over_the_whole_day(self, day):
        """The daily identity is the sum of the hourly ones, and must not be
        the only thing checked -- two hours can cancel each other's error."""
        load_payment = sum(LOAD_MW[t] * day["lmbda"][t] for t in HOURS)
        gen_revenue = sum(day["p"][g, t] * day["lmbda"][t]
                          for g in COST for t in HOURS)
        assert load_payment - gen_revenue == pytest.approx(0.0)

    def test_daily_energy_matches_daily_load(self, day):
        assert sum(day["p"].values()) == pytest.approx(sum(LOAD_MW.values()))

    def test_every_hour_has_a_dual(self, day):
        """Guards the Suffix declaration and the indexed-constraint rule.

        A Constraint built with expr= instead of rule= over m.T yields one
        constraint, one dual, and 23 silently missing hours.
        """
        assert set(day["lmbda"]) == set(HOURS)
        assert all(isinstance(day["lmbda"][t], float) for t in HOURS)


# ---------------------------------------------------------------- edge cases

class TestEdgeCases:
    """Degeneracy and infeasibility, met deliberately."""

    def test_price_at_a_breakpoint_is_not_unique(self):
        """Hour 5 sits exactly on g1's cap: no unit is marginal there.

        Same degeneracy as M0, now one hour among 24. The joint solve is free
        to return a different vertex than the single-hour solve would, which
        is why the load shape above avoids breakpoints entirely -- otherwise
        TestSeparability would be asserting solver pivot order.
        """
        load = {**LOAD_MW, 5: 100.0}
        r = solve_dispatch_day(COST, PMAX, load)
        assert r["p"]["g1", 5] == pytest.approx(100.0)
        assert r["p"]["g2", 5] == pytest.approx(0.0)
        assert 20.0 - 1e-6 <= r["lmbda"][5] <= 35.0 + 1e-6

    def test_one_infeasible_hour_fails_the_whole_day(self):
        """A joint solve has one status. Hour 19 alone above capacity makes
        all 24 hours unsolved -- there is no partial answer to return.

        Real markets shed load at an administrative value of lost load rather
        than going infeasible. Not modeled here.
        """
        load = {**LOAD_MW, 19: TOTAL_CAPACITY + 50.0}
        with pytest.raises(RuntimeError, match="not optimal"):
            solve_dispatch_day(COST, PMAX, load)

    def test_zero_load_hour_clears_at_or_below_the_cheapest_offer(self):
        load = {**LOAD_MW, 2: 0.0}
        r = solve_dispatch_day(COST, PMAX, load)
        assert all(r["p"][g, 2] == pytest.approx(0.0) for g in COST)
        assert r["lmbda"][2] <= 20.0 + 1e-6

    def test_non_contiguous_hour_labels_are_accepted(self):
        """T is a set, not a range. M2 hands over UTC timestamps with gaps
        possible; the model must not assume 0..23.
        """
        load = {3: 54.0, 17: 214.0, 19: 263.0}
        r = solve_dispatch_day(COST, PMAX, load)
        assert set(r["lmbda"]) == {3, 17, 19}
        assert r["lmbda"][19] == pytest.approx(80.0)
