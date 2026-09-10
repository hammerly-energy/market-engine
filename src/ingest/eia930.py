"""EIA-930 Hourly Grid Monitor (API v2) -> hourly load MW, UTC index.

This is the ingest boundary. Everything downstream of here is UTC and MW and
knows nothing about EIA, HTTP, or Central Prevailing Time.

Three responsibilities, in order:

  1. fetch    -- HTTP -> a JSON file in data/raw/, byte-for-byte as returned.
                 Append-only: an existing file is never rewritten.
  2. parse    -- JSON -> a pandas Series, tz-aware UTC index, float MW.
  3. validate -- assert the index is what was asked for, before anyone uses it.

The window
----------
EIA-930's `period` field on the region-data endpoint is UTC. A UTC calendar
day is NOT a market day: 2024-08-19T00Z is 19:00 Aug 18 in Central, so the
UTC day straddles two ERCOT days and splits the evening ramp in half.

    UTC   00 ......................... 23
    CPT      19 (Aug 18) ....... 18 (Aug 19)
             |<-- yesterday -->|<-- today -->|

So the local market day is resolved ONCE, here, by market_day_window(), which
converts a local calendar date into the pair of UTC instants that bound it.
The returned index is still UTC. The timezone is used at the boundary and
then discarded -- it never reaches a Scenario.

DST is handled by construction rather than by arithmetic: the window is built
from localized midnights, so the spring-forward day yields 23 hours and the
fall-back day 25. Nothing here assumes 24.
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
RAW_DIR = Path("data/raw/eia930")
PROCESSED_DIR = Path("data/processed")

# EIA-930 series type. "D" is metered demand -- what load actually was.
# "DF" is the day-ahead forecast and "NG" is net generation; neither is load.
DEMAND = "D"

# Balancing authority -> the local clock its market runs on. Used only to
# resolve a calendar date into UTC instants; see market_day_window.
BA_TIMEZONE = {
    "ERCO": "America/Chicago",   # ERCOT, Central Prevailing Time
    "PJM": "America/New_York",
    "CISO": "America/Los_Angeles",
    "MISO": "America/New_York",
    "ISNE": "America/New_York",
    "NYIS": "America/New_York",
}


# ------------------------------------------------------------------ 0. window

def market_day_window(respondent, date):
    """The UTC instants bounding one local market day.

    respondent -- BA code, e.g. "ERCO"
    date       -- local calendar date, "YYYY-MM-DD"

    Returns (start_utc, end_utc) as tz-aware Timestamps, start inclusive and
    end EXCLUSIVE. Half-open so that consecutive days tile without overlap.

    The length is whatever the local day actually is. Ask for 2024-03-10 in
    ERCOT and you get 23 hours, because 02:00 CST that morning does not exist.
    """
    tz = BA_TIMEZONE.get(respondent)
    if tz is None:
        raise ValueError(
            f"no timezone known for respondent {respondent!r}; "
            f"add it to BA_TIMEZONE"
        )
    start_local = pd.Timestamp(date, tz=tz)
    # Add a calendar day in local time, not 24 hours. On a DST boundary these
    # differ, and the calendar day is the market day.
    end_local = (start_local.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(tz)
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


# ------------------------------------------------------------------- 1. fetch

def raw_path(respondent, start_utc, end_utc):
    """Deterministic filename for one pull. Same request -> same file."""
    fmt = "%Y%m%dT%H%MZ"
    return RAW_DIR / (
        f"eia930_{respondent}_{DEMAND}_"
        f"{start_utc.strftime(fmt)}_{end_utc.strftime(fmt)}.json"
    )


def fetch(respondent, start_utc, end_utc, api_key=None, force=False):
    """Download one window of hourly demand into data/raw/. Returns the path.

    data/raw/ is append-only: if the file already exists this returns it
    untouched and makes no HTTP call, so a re-run is free and reproducible.
    force=True is for a known-bad download and will refuse to run silently --
    it renames the old file aside rather than overwriting it.

    The API's `end` is INCLUSIVE and hour-resolution, so the half-open
    [start, end) window becomes [start, end - 1h] on the wire.
    """
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "no EIA API key: set EIA_API_KEY in the environment or pass api_key="
        )

    path = raw_path(respondent, start_utc, end_utc)
    if path.exists() and not force:
        return path
    if path.exists() and force:
        path.rename(path.with_suffix(".json.superseded"))

    last_hour = end_utc - pd.Timedelta(hours=1)
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": DEMAND,
        "start": start_utc.strftime("%Y-%m-%dT%H"),
        "end": last_hour.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    if "response" not in payload:
        raise RuntimeError(f"unexpected EIA payload: {sorted(payload)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


# ------------------------------------------------------------------- 2. parse

def parse(path):
    """Raw EIA JSON -> Series of MW on a tz-aware UTC index, named "load_mw".

    EIA returns `period` as "YYYY-MM-DDTHH" with no offset, and `value` as a
    STRING. Both are converted here and nowhere else.
    """
    payload = json.loads(Path(path).read_text())
    rows = payload["response"]["data"]
    if not rows:
        raise ValueError(f"no rows in {path}: the window returned nothing")

    frame = pd.DataFrame(rows)
    # utc=True is what makes the index tz-aware rather than naive-and-hoping.
    index = pd.to_datetime(frame["period"], format="%Y-%m-%dT%H", utc=True)
    series = pd.Series(
        frame["value"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(index, name="period_utc"),
        name="load_mw",
    )
    return series.sort_index()


# ---------------------------------------------------------------- 3. validate

def validate(series, start_utc=None, end_utc=None):
    """Assert the index is a clean, gapless, hourly UTC range. Returns it.

    This is the M2 acceptance criterion in code. Every failure mode here is
    one that would otherwise surface much later as a wrong price:

      - a naive index silently compared against a tz-aware one
      - a duplicated hour (fall-back DST, or a paginated pull stitched twice)
      - a missing hour, which drops a balance constraint with no error
      - a NaN, which makes an hour's demand undefined and the LP infeasible
    """
    if series.index.tz is None:
        raise ValueError("index is timezone-naive; ingest must produce UTC")
    if str(series.index.tz) != "UTC":
        raise ValueError(f"index tz is {series.index.tz}, expected UTC")
    if not series.index.is_monotonic_increasing:
        raise ValueError("index is not sorted")

    dupes = series.index[series.index.duplicated()]
    if len(dupes):
        raise ValueError(f"duplicated hours: {list(dupes)}")

    gaps = series.index.to_series().diff().dropna()
    bad = gaps[gaps != pd.Timedelta(hours=1)]
    if len(bad):
        raise ValueError(f"index is not hourly; gaps at {list(bad.index)}")

    if series.isna().any():
        raise ValueError(f"missing values at {list(series.index[series.isna()])}")
    if (series < 0).any():
        raise ValueError(f"negative demand at {list(series.index[series < 0])}")

    if start_utc is not None and series.index[0] != start_utc:
        raise ValueError(f"starts at {series.index[0]}, expected {start_utc}")
    if end_utc is not None:
        expected_last = end_utc - pd.Timedelta(hours=1)
        if series.index[-1] != expected_last:
            raise ValueError(f"ends at {series.index[-1]}, expected {expected_last}")
        want = len(pd.date_range(start_utc, expected_last, freq="h", tz="UTC"))
        if len(series) != want:
            raise ValueError(f"got {len(series)} hours, window spans {want}")

    return series


# ------------------------------------------------------------------ 4. scale

def scale_to_fleet(series, fleet_capacity_mw, peak_fraction):
    """Map system demand onto a toy fleet, preserving shape exactly.

    ERCOT is ~85,000 MW and the M1/M2 fleet is 300 MW, so real load cannot be
    fed in unscaled. The factor is chosen so the day's PEAK lands at
    peak_fraction of total fleet capacity:

        factor = (fleet_capacity_mw * peak_fraction) / max(series)

    This is a linear rescale, so it preserves the shape and every ratio in it.
    It cannot change the peak-to-trough RATIO, and that is the point worth
    holding on to: ERCOT on 2024-08-19 swings 1.53:1 while M1's synthetic day
    swings 4.9:1. Real load is much flatter than the synthetic shape, so with
    breakpoints at 100/200/300 MW the cheapest unit runs flat out all day and
    never sets the price. That is a finding about real load, not a scaling
    artifact -- no factor can fix it, and it is not tuned away.

    peak_fraction is the one free knob and it is declared per scenario in the
    config, never defaulted here.
    """
    if not 0 < peak_fraction <= 1:
        raise ValueError(f"peak_fraction must be in (0, 1], got {peak_fraction}")
    peak = float(series.max())
    if peak <= 0:
        raise ValueError("series peak is not positive; nothing to scale")
    factor = (fleet_capacity_mw * peak_fraction) / peak
    return series * factor, factor


# ----------------------------------------------------------------- 5. persist

def write_processed(series, name):
    """Canonical parquet in data/processed/. One schema, UTC index."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    series.to_frame().to_parquet(path)
    return path


def load_demand(respondent, date, api_key=None):
    """fetch -> parse -> validate, for one local market day. The usual entry.

    Returns (series, provenance). The series is unscaled system MW on a UTC
    index; scaling is a scenario choice and happens in scale_to_fleet.
    """
    start_utc, end_utc = market_day_window(respondent, date)
    path = fetch(respondent, start_utc, end_utc, api_key=api_key)
    series = validate(parse(path), start_utc, end_utc)
    provenance = {
        "source": "EIA-930 Hourly Grid Monitor, API v2",
        "respondent": respondent,
        "type": DEMAND,
        "local_date": date,
        "local_timezone": BA_TIMEZONE[respondent],
        "start_utc": start_utc.isoformat(),
        "end_utc_exclusive": end_utc.isoformat(),
        "hours": len(series),
        "raw_file": str(path),
    }
    return series, provenance


if __name__ == "__main__":
    series, prov = load_demand("ERCO", "2024-08-19")
    print(f"{prov['respondent']} {prov['local_date']} ({prov['local_timezone']})")
    print(f"  {prov['start_utc']} -> {prov['end_utc_exclusive']}  {len(series)} hours")
    print(f"  peak {series.max():,.0f} MW   trough {series.min():,.0f} MW"
          f"   ratio {series.max() / series.min():.2f}")
    print(f"  written to {write_processed(series, 'erco_2024-08-19_demand')}")
