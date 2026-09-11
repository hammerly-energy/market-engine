"""RTS-GMLC (NREL) -> network topology, generator fleet, load and renewable profiles.

The M5 ingest. Everything here reads CSVs out of data/raw/rts_gmlc/ and returns
plain dicts; nothing here imports from src/model/. scenario.py is what turns
these dicts into a Scenario, exactly as it does for EIA-930.

    bus.csv     --+
    branch.csv  --+--> read_network()  --> buses, branches, load shares
    gen.csv     ----> read_fleet()     --> offers, and which units are profiled
    DAY_AHEAD_* --+--> read_profiles() --> {gen: {hour: MW available}}
                  +--> read_load()     --> {bus: {hour: MW}}

Why this is a source module and not a config
--------------------------------------------
M3's network lives inline in configs/m3.yaml because five buses and six lines
fit on a page and every number is worth reading. 73 buses, 120 branches and 158
generators do not. Transcribing them would produce a config nobody proofreads
and a second copy of the data to drift from the first. So M5's config declares
a SOURCE and a date, and the numbers stay in data/raw/ where they were
downloaded.

Time
----
RTS-GMLC is a synthetic system. Its timeseries carry a calendar date and an
hour-of-day (Period 1..24) and no timezone at all, because there is no real
place for them to be local to. They are read as UTC and provenance records that
choice. This is the one place in the repo where a UTC label is a convention
rather than a conversion, and calling it anything else would be inventing a
timezone for a grid that does not have one.

Units that are read and then dropped
------------------------------------
    SYNC_COND (3)  PMax 0. A synchronous condenser supplies reactive power,
                   which the DC approximation does not model. Carrying it would
                   add three generators that can never produce a MW.
    STORAGE  (1)   Needs state of charge across hours. That is M7, and the
                   hours do not couple yet.
    CSP      (1)   Has its own timeseries directory and a thermal storage model
                   behind it. Not a flat availability profile, so it does not
                   fit the renewable path, and it is not dispatchable, so it
                   does not fit the thermal path either.

Each is dropped explicitly and counted in provenance, so the fleet that reaches
the solver can be reconciled against the 158 rows of gen.csv by subtraction
rather than by trust.
"""

import csv
from pathlib import Path

RTS_ROOT = Path("data/raw/rts_gmlc")

# Timeseries files whose columns are GEN UIDs and whose values are MW available
# in that hour. Wind and PV are weather. Hydro is a schedule, not weather, but
# it enters the LP the same way: an hourly cap the unit may produce up to.
PROFILE_FILES = {
    "wind": "DAY_AHEAD_wind.csv",
    "pv": "DAY_AHEAD_pv.csv",
    "rtpv": "DAY_AHEAD_rtpv.csv",
    "hydro": "DAY_AHEAD_hydro.csv",
}

LOAD_FILE = "DAY_AHEAD_regional_Load.csv"

# Dropped, with the reason kept next to the name so the provenance line that
# reports the count can also report why.
DROPPED_TYPES = {
    "SYNC_COND": "reactive support only; PMax 0 under the DC approximation",
    "STORAGE": "needs state of charge across hours; M7",
    "CSP": "thermal storage behind the profile; neither flat renewable nor dispatchable",
}

# The DC model is on a 100 MVA base, matching reactance_pu in branch.csv.
BASE_MVA = 100.0


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _hour_key(row):
    """A row of a timeseries file -> an ISO-8601 UTC hour string.

    Period is 1..24 and hour-of-day is 0..23, so Period 1 is 00:00. Getting
    this off by one would shift every renewable profile against the load it is
    supposed to serve by an hour, which looks like a plausible solar shape
    sitting in the wrong place and does not raise anywhere.
    """
    period = int(row["Period"])
    if not 1 <= period <= 24:
        raise ValueError(f"Period {period} outside 1..24")
    return "%04d-%02d-%02dT%02d:00:00+00:00" % (
        int(row["Year"]), int(row["Month"]), int(row["Day"]), period - 1
    )


def _day_rows(path, date):
    """The 24 rows of a timeseries file belonging to one calendar date.

    date is 'YYYY-MM-DD'. Asserts 24 of them: a short day means the file does
    not cover the date asked for, and a silent 23-row horizon would propagate
    into a Scenario whose hours look fine until an index is missing.
    """
    y, m, d = (int(p) for p in date.split("-"))
    rows = [r for r in _rows(path)
            if (int(r["Year"]), int(r["Month"]), int(r["Day"])) == (y, m, d)]
    if len(rows) != 24:
        raise ValueError(
            f"{Path(path).name}: expected 24 rows for {date}, found {len(rows)}"
        )
    return rows


# ------------------------------------------------------------------ network

def read_network(root=RTS_ROOT):
    """buses, branches, and each bus's share of its area's load.

    Bus names are the Bus ID as a string -- '101', not 'Abel' -- because that
    is what branch.csv references and what the RTS papers cite. The friendly
    name is carried alongside for figures and never used as a key.

    Returns (buses, branches, load_share, bus_names, area_of) where

        buses      ['101', '102', ...]  the column order of every matrix
        branches   {uid: {from, to, reactance_pu, limit_mw}}
        load_share {bus: fraction of its area's load}
        bus_names  {bus: 'Abel'}
        area_of    {bus: area id}
    """
    root = Path(root)
    bus_rows = _rows(root / "bus.csv")

    buses = [r["Bus ID"] for r in bus_rows]
    bus_names = {r["Bus ID"]: r["Bus Name"] for r in bus_rows}
    area_of = {r["Bus ID"]: r["Area"] for r in bus_rows}

    # RTS publishes load per AREA per hour, and a static MW Load per bus. The
    # bus figure is a participation factor, not a demand: it says how the
    # area's load is distributed, and the hourly area total says how much there
    # is. Multiplying the two is how RTS itself builds nodal load.
    area_total = {}
    for r in bus_rows:
        area_total[r["Area"]] = area_total.get(r["Area"], 0.0) + float(r["MW Load"])
    load_share = {}
    for r in bus_rows:
        total = area_total[r["Area"]]
        load_share[r["Bus ID"]] = float(r["MW Load"]) / total if total else 0.0

    branches = {}
    for r in _rows(root / "branch.csv"):
        limit = float(r["Cont Rating"])
        if limit <= 0:
            # Not a MATPOWER file, so the rateA = 0 convention does not apply
            # here and a zero would be a data problem rather than "unlimited".
            raise ValueError(f"branch {r['UID']}: Cont Rating {limit} is not > 0")
        branches[r["UID"]] = {
            "from": r["From Bus"],
            "to": r["To Bus"],
            # X is already per-unit on the 100 MVA base. Transformers carry a
            # Tr Ratio, which the DC approximation ignores: it scales voltage
            # magnitude, and DC holds every magnitude at 1.0 pu by assumption.
            "reactance_pu": float(r["X"]),
            "limit_mw": limit,
        }

    known = set(buses)
    for uid, br in branches.items():
        for end in ("from", "to"):
            if br[end] not in known:
                raise ValueError(f"branch {uid}: {end} bus {br[end]!r} not in bus.csv")

    return buses, branches, load_share, bus_names, area_of


# -------------------------------------------------------------------- fleet

def average_heat_rate(row):
    """MMBtu per MWh at full output, from the 4-point curve in gen.csv.

    gen.csv gives the curve as one AVERAGE heat rate at the lowest output point
    and three INCREMENTAL rates for the segments above it, in Btu/kWh. Total
    fuel at full output is the first point's average times its MW, plus each
    segment's incremental times that segment's width:

        fuel(MMBtu/h)
              |                                  * Pmax
              |                          HR_incr_3
              |                  * ---
              |          HR_incr_2
              |      * ---
              |  HR_incr_1
              |* ---  <- Output_pct_0 * Pmax, at HR_avg_0
              |
              +-------------------------------------- MW
             0

    Dividing that total by Pmax gives one average rate. It INCLUDES the fuel a
    unit burns just to sit at its minimum, which the incremental rates exclude
    -- and that no-load burn is most of why a peaker is expensive. An offer
    built from HR_incr_1 alone makes every unit look cheaper than it is, and
    peakers cheaper by the most, which reorders the stack in the wrong
    direction.

    The distortion that remains: this is an average, not a marginal cost, so a
    unit at part load offers above its true next-MW cost. Fixing that properly
    means a no-load term plus a piecewise offer, which is M6/M7. The segments
    are already in gen.csv waiting for it.
    """
    pmax = float(row["PMax MW"])
    pcts = [float(row["Output_pct_%d" % i]) for i in range(4)]
    mw = [p * pmax for p in pcts]

    # Btu/kWh -> MMBtu/MWh is a factor of 1000, not 1e6: kWh -> MWh eats the
    # other three orders. Getting this wrong is a 1000x price, which is at
    # least loud.
    heat = mw[0] * float(row["HR_avg_0"]) / 1000.0
    for k in range(1, 4):
        heat += (mw[k] - mw[k - 1]) * float(row["HR_incr_%d" % k]) / 1000.0
    return heat / pmax


def read_fleet(root=RTS_ROOT, profiled=()):
    """{uid: {bus, cost_usd_per_mwh, pmax_mw, pmin_mw, unit_type, profiled}}.

    profiled is the set of GEN UIDs that have an hourly availability profile.
    Those offer at $0 and are capped hour by hour; everything else offers at
    fuel * heat rate + VOM against a fixed capacity.

    A renewable at $0 is not a claim that wind is free. It is the claim that
    its opportunity cost is zero, so it runs whenever the network lets it and
    is curtailed only when the network does not -- which is the behaviour a
    market actually produces, and is what lets a curtailed hour price at zero
    rather than at the last thermal unit.
    """
    root = Path(root)
    profiled = set(profiled)

    fleet, dropped = {}, {}
    for r in _rows(root / "gen.csv"):
        uid = r["GEN UID"]
        unit_type = r["Unit Type"]
        if unit_type in DROPPED_TYPES:
            dropped[uid] = unit_type
            continue

        pmax = float(r["PMax MW"])
        pmin = float(r["PMin MW"])
        is_profiled = uid in profiled

        if is_profiled:
            cost = 0.0
        else:
            cost = (average_heat_rate(r) * float(r["Fuel Price $/MMBTU"])
                    + float(r["VOM"]))

        fleet[uid] = {
            "bus": r["Bus ID"],
            "cost_usd_per_mwh": cost,
            "pmax_mw": pmax,
            # Carried, not enforced: an LP has no way to honour a minimum
            # (trap 4). It is here so M6 does not have to re-read gen.csv.
            "pmin_mw": pmin,
            "unit_type": unit_type,
            "fuel": r["Fuel"],
            "profiled": is_profiled,
        }

    return fleet, dropped


# ----------------------------------------------------------------- profiles

def read_profiles(root=RTS_ROOT, date=None):
    """{gen uid: {hour: MW available}} for one day, across all four files.

    A UID appearing in two files would silently keep whichever was read last,
    so the collision is checked rather than assumed away.
    """
    root = Path(root)
    out = {}
    for kind, fname in PROFILE_FILES.items():
        rows = _day_rows(root / fname, date)
        uids = [c for c in rows[0] if c not in ("Year", "Month", "Day", "Period")]
        clash = set(uids) & set(out)
        if clash:
            raise ValueError(f"{fname}: UIDs already profiled elsewhere: {sorted(clash)}")
        for uid in uids:
            out[uid] = {}
        for r in rows:
            t = _hour_key(r)
            for uid in uids:
                mw = float(r[uid])
                if mw < 0:
                    raise ValueError(f"{uid} at {t}: negative availability {mw}")
                out[uid][t] = mw
    return out


def read_load(root=RTS_ROOT, date=None, load_share=None, area_of=None):
    """{bus: {hour: MW}} -- hourly area load, distributed over that area's buses.

    RTS publishes demand per AREA per hour and a static MW Load per bus. The
    bus figure is a participation factor: it says how an area's load is spread,
    not how much there is. Nodal load is the product, which is how RTS itself
    builds it.

        area 1 load at 18:00  x  bus 101's share of area 1  =  load at 101

    Every bus gets an entry, including the ones with no load at all, because
    the network solver needs an injection at every bus and a missing key is a
    KeyError rather than a zero.
    """
    root = Path(root)
    rows = _day_rows(root / LOAD_FILE, date)
    areas = [c for c in rows[0] if c not in ("Year", "Month", "Day", "Period")]

    missing = set(area_of.values()) - set(areas)
    if missing:
        raise ValueError(f"bus.csv has areas with no load column: {sorted(missing)}")

    out = {bus: {} for bus in area_of}
    for r in rows:
        t = _hour_key(r)
        for bus, area in area_of.items():
            out[bus][t] = float(r[area]) * load_share[bus]
    return out
