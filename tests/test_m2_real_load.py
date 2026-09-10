"""M2 acceptance tests: real ERCOT demand across the ingest boundary.

The milestone is the boundary and the UTC discipline, not a new formulation.
solve_dispatch_day is byte-for-byte the M1 code; only where D comes from has
changed. So these tests are about the DATA and the SEAM:

    EIA-930 JSON --> parse --> validate --> scale --> Scenario --> dispatch
                       ^          ^           ^          ^
                   TestParse  TestIndex   TestScaling  TestBoundary

TestIndex is the milestone's stated goal: row count asserted against hours
requested, UTC index, no gaps, no duplicates.

Network tests hit the EIA API and are marked `network`. Everything else runs
against a committed fixture, so the suite is offline and deterministic:

    pytest                     # offline, fixture-backed
    pytest -m network          # also re-pull from EIA

The fixture is a real EIA payload, trimmed to the fields the parser reads.
"""

import json

import pandas as pd
import pytest

from src.ingest import eia930
from src.ingest.scenario import build_scenario
from src.model.dispatch import solve_dispatch_day
from src.model.inputs import Generator, Load, Scenario

FIXTURE = "tests/fixtures/eia930_erco_2024-08-19.json"

COST = {"g1": 20.0, "g2": 35.0, "g3": 80.0}     # $/MWh, unchanged from m1
PMAX = {"g1": 100.0, "g2": 100.0, "g3": 100.0}  # MW,    unchanged from m1
TOTAL_CAPACITY = 300.0

START_UTC = pd.Timestamp("2024-08-19T05:00:00Z")
END_UTC = pd.Timestamp("2024-08-20T05:00:00Z")   # exclusive
SYSTEM_PEAK_MW = 85263.0
SYSTEM_TROUGH_MW = 55616.0


@pytest.fixture(scope="module")
def series():
    """The committed EIA payload, parsed. No network."""
    return eia930.parse(FIXTURE)


@pytest.fixture(scope="module")
def scenario():
    """A Scenario built from the fixture, bypassing fetch."""
    return _scenario_from(FIXTURE)


def _scenario_from(path, peak_fraction=0.90):
    s = eia930.validate(eia930.parse(path), START_UTC, END_UTC)
    scaled, _ = eia930.scale_to_fleet(s, TOTAL_CAPACITY, peak_fraction)
    return Scenario(
        name="m2-test",
        generators=tuple(
            Generator(g, "bus1", COST[g], PMAX[g]) for g in COST
        ),
        loads=(Load("bus1", {t.isoformat(): float(v) for t, v in scaled.items()}),),
    )


# ------------------------------------------------------------------- windowing

class TestMarketDayWindow:
    """A UTC calendar day is not a market day. Resolve it once, at the edge."""

    def test_local_day_maps_to_the_right_utc_instants(self):
        start, end = eia930.market_day_window("ERCO", "2024-08-19")
        assert start == START_UTC          # 00:00 CDT == 05:00 UTC
        assert end == END_UTC
        assert str(start.tz) == "UTC" and str(end.tz) == "UTC"

    def test_window_is_half_open_so_consecutive_days_tile(self):
        _, end_19 = eia930.market_day_window("ERCO", "2024-08-19")
        start_20, _ = eia930.market_day_window("ERCO", "2024-08-20")
        assert end_19 == start_20

    def test_summer_and_winter_days_have_different_utc_offsets(self):
        """CDT is UTC-5, CST is UTC-6. A fixed offset would break one of them."""
        summer, _ = eia930.market_day_window("ERCO", "2024-08-19")
        winter, _ = eia930.market_day_window("ERCO", "2024-01-15")
        assert summer.hour == 5
        assert winter.hour == 6

    def test_spring_forward_day_is_23_hours(self):
        """Trap 1. 2024-03-10 has no 02:00 CST. Nothing may assume 24."""
        start, end = eia930.market_day_window("ERCO", "2024-03-10")
        assert (end - start) == pd.Timedelta(hours=23)

    def test_fall_back_day_is_25_hours(self):
        """Trap 1. 2024-11-03 has 01:00 CDT and 01:00 CST -- two of them."""
        start, end = eia930.market_day_window("ERCO", "2024-11-03")
        assert (end - start) == pd.Timedelta(hours=25)

    def test_unknown_respondent_fails_loudly(self):
        with pytest.raises(ValueError, match="no timezone known"):
            eia930.market_day_window("NOPE", "2024-08-19")


# ---------------------------------------------------------------------- parse

class TestParse:
    """EIA hands back strings. Types are fixed here and nowhere else."""

    def test_values_are_floats_not_strings(self, series):
        assert series.dtype == float

    def test_series_is_named_and_indexed_for_downstream_use(self, series):
        assert series.name == "load_mw"
        assert series.index.name == "period_utc"

    def test_empty_payload_fails_loudly(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"response": {"data": []}}))
        with pytest.raises(ValueError, match="no rows"):
            eia930.parse(path)


# ---------------------------------------------------------------------- index
# The M2 goal, stated in the milestone table, asserted here.

class TestIndex:
    """Row count against hours requested; UTC; no gaps; no duplicates."""

    def test_row_count_matches_the_window_requested(self, series):
        expected = len(pd.date_range(START_UTC, END_UTC, freq="h",
                                     inclusive="left"))
        assert expected == 24
        assert len(series) == expected

    def test_index_is_timezone_aware_utc(self, series):
        assert series.index.tz is not None
        assert str(series.index.tz) == "UTC"

    def test_index_spans_exactly_the_window(self, series):
        assert series.index[0] == START_UTC
        assert series.index[-1] == END_UTC - pd.Timedelta(hours=1)

    def test_no_duplicate_hours(self, series):
        assert not series.index.has_duplicates

    def test_no_gaps(self, series):
        deltas = series.index.to_series().diff().dropna().unique()
        assert list(deltas) == [pd.Timedelta(hours=1)]

    def test_no_missing_or_negative_values(self, series):
        assert not series.isna().any()
        assert (series > 0).all()

    def test_validate_accepts_the_real_series(self, series):
        assert eia930.validate(series, START_UTC, END_UTC) is series


class TestValidateRejects:
    """Each failure mode would otherwise surface later as a wrong price."""

    def test_naive_index_is_rejected(self, series):
        naive = series.copy()
        naive.index = naive.index.tz_localize(None)
        with pytest.raises(ValueError, match="timezone-naive"):
            eia930.validate(naive)

    def test_non_utc_index_is_rejected(self, series):
        local = series.tz_convert("America/Chicago")
        with pytest.raises(ValueError, match="expected UTC"):
            eia930.validate(local)

    def test_duplicated_hour_is_rejected(self, series):
        doubled = pd.concat([series, series.iloc[[3]]]).sort_index()
        with pytest.raises(ValueError, match="duplicated hours"):
            eia930.validate(doubled)

    def test_gap_is_rejected(self, series):
        """A dropped hour silently deletes a balance constraint."""
        with pytest.raises(ValueError, match="not hourly"):
            eia930.validate(series.drop(series.index[5]))

    def test_nan_is_rejected(self, series):
        holed = series.copy()
        holed.iloc[7] = float("nan")
        with pytest.raises(ValueError, match="missing values"):
            eia930.validate(holed)

    def test_short_window_is_rejected(self, series):
        with pytest.raises(ValueError, match="expected"):
            eia930.validate(series.iloc[:20], START_UTC, END_UTC)


# -------------------------------------------------------------------- scaling

class TestScaling:
    """The factor is declared, reproducible, and shape-preserving."""

    def test_peak_lands_at_the_declared_fraction_of_capacity(self, series):
        scaled, _ = eia930.scale_to_fleet(series, TOTAL_CAPACITY, 0.90)
        assert scaled.max() == pytest.approx(270.0)

    def test_factor_is_what_the_config_documents(self, series):
        _, factor = eia930.scale_to_fleet(series, TOTAL_CAPACITY, 0.90)
        assert factor == pytest.approx((300.0 * 0.90) / SYSTEM_PEAK_MW)

    def test_rescale_preserves_shape_exactly(self, series):
        """Linear, so every ratio between hours survives untouched.

        scaled/series is one constant to floating-point, not to the bit --
        the division rounds differently at different magnitudes.
        """
        scaled, factor = eia930.scale_to_fleet(series, TOTAL_CAPACITY, 0.90)
        ratios = scaled / series
        assert ratios.min() == pytest.approx(ratios.max())
        assert ratios.mean() == pytest.approx(factor)
        assert (scaled.max() / scaled.min()) == pytest.approx(
            series.max() / series.min()
        )

    def test_rescale_cannot_change_the_peak_to_trough_ratio(self, series):
        """The M2 finding. No factor can stretch a flat day.

        Whatever fraction is chosen, the ratio is fixed by the data at 1.53.
        Putting the trough below 100 MW while the peak stays above 200 would
        need a ratio above 2.0, so with 100/100/100 MW blocks g1 can never be
        marginal on this day. That is a fact about ERCOT, not about scaling,
        and the fleet was not resized to hide it.
        """
        ratio = SYSTEM_PEAK_MW / SYSTEM_TROUGH_MW
        assert ratio == pytest.approx(1.533, abs=1e-3)
        for fraction in (0.5, 0.75, 0.90, 1.0):
            scaled, _ = eia930.scale_to_fleet(series, TOTAL_CAPACITY, fraction)
            assert (scaled.max() / scaled.min()) == pytest.approx(ratio)
            assert ratio < 2.0

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_nonsense_fraction_is_rejected(self, series, fraction):
        with pytest.raises(ValueError, match="peak_fraction"):
            eia930.scale_to_fleet(series, TOTAL_CAPACITY, fraction)


# ------------------------------------------------------------------- boundary
# "The model layer must not know where data came from."

class TestBoundary:
    """The Scenario is inert. The solver cannot reach a data source."""

    def test_hours_are_iso_utc_strings_not_timestamps(self, scenario):
        for t in scenario.hours:
            assert isinstance(t, str)
            assert t.endswith("+00:00")

    def test_hours_sort_chronologically_as_strings(self, scenario):
        assert scenario.hours == sorted(scenario.hours)
        assert scenario.hours[0].startswith("2024-08-19T05")
        assert scenario.hours[-1].startswith("2024-08-20T04")

    def test_scenario_carries_no_callables(self, scenario):
        """provenance is a record, not a handle. Nothing in it can fetch."""
        for key, value in scenario.provenance.items():
            assert not callable(value), key

    def test_scenario_is_frozen(self, scenario):
        with pytest.raises(Exception):
            scenario.name = "mutated"

    def test_model_layer_imports_no_data_source(self):
        """The rule, enforced. src/model/ may not import src/ingest/."""
        import pathlib
        offenders = []
        for path in pathlib.Path("src/model").rglob("*.py"):
            text = path.read_text()
            if "src.ingest" in text or "import requests" in text:
                offenders.append(str(path))
        assert offenders == []

    def test_actuals_never_reach_a_scenario(self):
        """src/ingest/actuals.py is a held-out answer key."""
        import pathlib
        offenders = []
        for path in list(pathlib.Path("src/model").rglob("*.py")) + \
                    [pathlib.Path("src/ingest/scenario.py")]:
            if "actuals" in path.read_text():
                offenders.append(str(path))
        assert offenders == []

    def test_scenario_views_feed_dispatch_unchanged(self, scenario):
        """The M1 solver takes the Scenario with no adapter in between."""
        res = solve_dispatch_day(
            scenario.cost(), scenario.pmax(), scenario.demand()
        )
        assert set(res["lmbda"]) == set(scenario.hours)


# ----------------------------------------------------------------- invariants
# Carried forward from M0/M1. These hold forever; nothing below supersedes.

class TestInvariants:
    """Same assertions as M0 and M1, now against real load."""

    @pytest.fixture(scope="class")
    def solved(self, scenario):
        return scenario, solve_dispatch_day(
            scenario.cost(), scenario.pmax(), scenario.demand()
        )

    def test_energy_balance_holds_every_hour(self, solved):
        scenario, res = solved
        D = scenario.demand()
        for t in scenario.hours:
            served = sum(res["p"][g, t] for g in scenario.cost())
            assert served == pytest.approx(D[t]), t

    def test_no_generator_exceeds_capacity_or_goes_negative(self, solved):
        scenario, res = solved
        pmax = scenario.pmax()
        for (g, t), mw in res["p"].items():
            assert -1e-9 <= mw <= pmax[g] + 1e-9, (g, t)

    def test_production_cost_matches_dispatch(self, solved):
        scenario, res = solved
        c = scenario.cost()
        assert res["cost"] == pytest.approx(
            sum(c[g] * mw for (g, _), mw in res["p"].items())
        )

    def test_settlement_identity(self, solved):
        """sum(load payments) - sum(gen revenue) == sum(mu[l] * limit[l])

        Still one bus, so the congestion term is still exactly zero. M3
        replaces that zero with a real term; this test is not deleted then.
        """
        scenario, res = solved
        D = scenario.demand()
        for t in scenario.hours:
            payment = D[t] * res["lmbda"][t]
            revenue = sum(res["p"][g, t] for g in scenario.cost()) * res["lmbda"][t]
            congestion_rent = 0.0  # no branches at M2
            assert payment - revenue == pytest.approx(congestion_rent), t

    def test_price_is_always_one_of_the_offers(self, solved):
        """Single bus, no network: lambda can only be an offer price."""
        scenario, res = solved
        offers = set(scenario.cost().values())
        for t, lmbda in res["lmbda"].items():
            assert any(abs(lmbda - o) < 1e-6 for o in offers), (t, lmbda)


# -------------------------------------------------------------- the M2 finding
# Superseded when the fleet or the day changes. Kept as the record of what
# real load did to a fleet sized for a synthetic shape.

class TestRealLoadIsFlatterThanSynthetic:
    """g1 never sets the price on this day. Recorded, not fixed."""

    @pytest.fixture(scope="class")
    def solved(self, scenario):
        return scenario, solve_dispatch_day(
            scenario.cost(), scenario.pmax(), scenario.demand()
        )

    def test_cheapest_unit_runs_flat_out_all_day(self, solved):
        scenario, res = solved
        for t in scenario.hours:
            assert res["p"]["g1", t] == pytest.approx(100.0), t

    def test_cheapest_unit_is_never_marginal(self, solved):
        scenario, res = solved
        for t in scenario.hours:
            assert res["lmbda"][t] != pytest.approx(20.0), t

    def test_price_takes_only_two_values(self, solved):
        _, res = solved
        assert {round(v, 6) for v in res["lmbda"].values()} == {35.0, 80.0}


# --------------------------------------------------------------------- network

@pytest.mark.network
class TestLiveApi:
    """Re-pull from EIA. Skipped by default; `pytest -m network` runs it."""

    def test_live_pull_matches_the_committed_fixture(self):
        import os
        if not os.environ.get("EIA_API_KEY"):
            pytest.skip("EIA_API_KEY not set")
        live, prov = eia930.load_demand("ERCO", "2024-08-19")
        fixture = eia930.parse(FIXTURE)
        assert prov["hours"] == 24
        pd.testing.assert_series_equal(live, fixture)

    def test_build_scenario_end_to_end(self):
        import os
        if not os.environ.get("EIA_API_KEY"):
            pytest.skip("EIA_API_KEY not set")
        s = build_scenario("configs/m2.yaml")
        assert len(s.hours) == 24
        assert s.peak_load_mw == pytest.approx(270.0)
        assert s.provenance["scale_factor"] == pytest.approx(
            270.0 / SYSTEM_PEAK_MW
        )
