"""M0 acceptance tests: 3 generators, 1 bus, 1 hour.

The milestone is "verify by hand that the dual equals the marginal unit's
cost." These tests are that verification, made permanent.

Fleet under test:

    $/MWh
     80 |                              +------------ g3
        |                              |
     35 |              +---------------+              g2
        |              |
     20 |--------------+                              g1
        |
      0 +------+-------+-------+-------+------> MW
              50     100     150     250
"""

import pytest

from src.model.dispatch import solve_dispatch

COST = {"g1": 20.0, "g2": 35.0, "g3": 80.0}   # $/MWh
PMAX = {"g1": 100.0, "g2": 100.0, "g3": 100.0}  # MW
TOTAL_CAPACITY = sum(PMAX.values())            # 300 MW


def clear(demand):
    return solve_dispatch(COST, PMAX, demand)


# ---------------------------------------------------------------- merit order

class TestMeritOrder:
    """Cheapest capacity fills first. No exceptions without a network."""

    def test_only_cheapest_runs_below_its_capacity(self):
        r = clear(50.0)
        assert r["p"] == pytest.approx({"g1": 50.0, "g2": 0.0, "g3": 0.0})

    def test_second_unit_picks_up_the_remainder(self):
        r = clear(150.0)
        assert r["p"] == pytest.approx({"g1": 100.0, "g2": 50.0, "g3": 0.0})

    def test_third_unit_starts_only_when_first_two_are_full(self):
        r = clear(250.0)
        assert r["p"] == pytest.approx({"g1": 100.0, "g2": 100.0, "g3": 50.0})


# -------------------------------------------------------------------- pricing

class TestPricing:
    """The dual on energy balance equals the marginal unit's cost.

    "Marginal" means strictly between its bounds -- free to move up or down.
    A unit at its cap or at zero cannot respond to a small change in demand,
    so it never sets the price.
    """

    @pytest.mark.parametrize(
        "demand, expected_lmbda, marginal_unit",
        [
            (50.0, 20.0, "g1"),
            (150.0, 35.0, "g2"),
            (250.0, 80.0, "g3"),
        ],
    )
    def test_dual_equals_marginal_unit_cost(self, demand, expected_lmbda, marginal_unit):
        r = clear(demand)
        assert r["lmbda"] == pytest.approx(expected_lmbda)
        # the unit setting the price is strictly between 0 and its cap
        assert 0.0 < r["p"][marginal_unit] < PMAX[marginal_unit]
        assert r["lmbda"] == pytest.approx(COST[marginal_unit])

    def test_duals_were_actually_imported(self):
        """Guards the Suffix declaration. Without it the dual is missing."""
        r = clear(150.0)
        assert r["lmbda"] is not None
        assert isinstance(r["lmbda"], float)

    def test_price_does_not_depend_on_generator_ordering(self):
        """Merit order is economics, not dict insertion order."""
        reversed_cost = {k: COST[k] for k in reversed(list(COST))}
        reversed_pmax = {k: PMAX[k] for k in reversed(list(PMAX))}
        r = solve_dispatch(reversed_cost, reversed_pmax, 150.0)
        assert r["lmbda"] == pytest.approx(35.0)
        assert r["p"] == pytest.approx({"g1": 100.0, "g2": 50.0, "g3": 0.0})

    def test_inframarginal_unit_earns_rent(self):
        """g1 offers at 20, is paid 35, keeps the spread on every MW.

        This is how capital cost is recovered. It is the design, not a flaw.
        """
        r = clear(150.0)
        rent = r["p"]["g1"] * (r["lmbda"] - COST["g1"])
        assert rent == pytest.approx(100.0 * 15.0)


# ----------------------------------------------------------------- invariants

class TestInvariants:
    """Must hold at every demand level. These carry forward to M2+."""

    DEMANDS = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 299.0, 300.0]

    @pytest.mark.parametrize("demand", DEMANDS)
    def test_energy_balance_holds(self, demand):
        r = clear(demand)
        assert sum(r["p"].values()) == pytest.approx(demand)

    @pytest.mark.parametrize("demand", DEMANDS)
    def test_no_generator_exceeds_its_capacity_or_goes_negative(self, demand):
        r = clear(demand)
        for g, mw in r["p"].items():
            assert -1e-9 <= mw <= PMAX[g] + 1e-9

    @pytest.mark.parametrize("demand", DEMANDS)
    def test_production_cost_matches_dispatch(self, demand):
        r = clear(demand)
        assert r["cost"] == pytest.approx(sum(COST[g] * mw for g, mw in r["p"].items()))

    @pytest.mark.parametrize("demand", DEMANDS)
    def test_settlement_identity(self, demand):
        """sum(load payments) - sum(gen revenue) == sum(mu[l] * limit[l])

        One bus has no lines, so the congestion term is exactly zero and the
        two sides must match. At M2 they stop matching, and the gap is
        congestion rent.
        """
        r = clear(demand)
        load_payment = demand * r["lmbda"]
        gen_revenue = sum(r["p"].values()) * r["lmbda"]
        congestion_rent = 0.0  # no branches at M0
        assert load_payment - gen_revenue == pytest.approx(congestion_rent)

    def test_production_cost_is_below_load_payment(self):
        """The wedge between the two is total producer surplus."""
        r = clear(150.0)
        assert r["cost"] < 150.0 * r["lmbda"]


# ---------------------------------------------------------------- edge cases

class TestEdgeCases:
    """Degeneracy and infeasibility. Meet them deliberately, not by accident."""

    def test_price_at_a_breakpoint_is_not_unique(self):
        """D = 100 sits exactly on g1's cap, so no unit is marginal.

        Every value in [20, 35] is an optimal dual. Simplex returns whichever
        vertex its pivot rule reaches, and that can change with the solver,
        the version, or a rounding error in Pmax.

        This is why no test asserts an exact price at a breakpoint.
        """
        r = clear(100.0)
        assert r["p"] == pytest.approx({"g1": 100.0, "g2": 0.0, "g3": 0.0})
        assert 20.0 - 1e-6 <= r["lmbda"] <= 35.0 + 1e-6

    def test_zero_demand_price_is_capped_by_the_cheapest_offer(self):
        """At D = 0 nothing constrains the price from below.

        Any lambda <= 20 is optimal. Only the upper bound is meaningful.
        """
        r = clear(0.0)
        assert r["p"] == pytest.approx({"g1": 0.0, "g2": 0.0, "g3": 0.0})
        assert r["lmbda"] <= 20.0 + 1e-6

    def test_demand_at_exactly_total_capacity_clears(self):
        r = clear(TOTAL_CAPACITY)
        assert r["p"] == pytest.approx({"g1": 100.0, "g2": 100.0, "g3": 100.0})

    def test_demand_above_capacity_fails_loudly(self):
        """No scarcity pricing yet, so this must raise rather than return junk.

        Real markets shed load at an administrative value of lost load instead
        of going infeasible. That is an M1 addition.
        """
        with pytest.raises(RuntimeError, match="not optimal"):
            clear(TOTAL_CAPACITY + 100.0)
