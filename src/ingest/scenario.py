"""Config + data source -> Scenario. The only module that knows both.

This sits on the ingest side of the boundary described in src/model/inputs.py.
Something has to hold a config path in one hand and an API key in the other;
putting it here keeps that knowledge out of src/model/ entirely, and keeps any
single ingest module (eia930, rts_gmlc, ...) from having to know what a
Scenario is.

    configs/m2.yaml ---+
                       +--> build_scenario() --> Scenario --> src/model/
    src/ingest/eia930 -+
"""

from pathlib import Path

import yaml

from src.ingest import eia930
from src.model.inputs import Branch, Bus, Generator, Load, Scenario

SINGLE_BUS = "bus1"


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def _fleet(config):
    return tuple(
        Generator(
            name=name,
            bus=spec.get("bus", SINGLE_BUS),
            cost_usd_per_mwh=float(spec["cost_usd_per_mwh"]),
            pmax_mw=float(spec["pmax_mw"]),
            pmin_mw=float(spec.get("pmin_mw", 0.0)),
        )
        for name, spec in config["fleet"].items()
    )


STATIC_HOUR = "static"


def _loads_eia930(spec, capacity, api_key):
    """Real hourly demand at a single bus. The M2 path, unchanged.

    The scaling guards live here rather than in build_scenario so that adding
    a second source cannot quietly make them optional: an unjustified rescale
    of real load is the thing M2 exists to make explicit.
    """
    if "scaling" not in spec:
        raise ValueError(
            "load.scaling is required; real load must not be rescaled onto a "
            "toy fleet without the choice being written down"
        )
    scaling = spec["scaling"]
    if scaling.get("method") != "peak_fraction_of_fleet_capacity":
        raise ValueError(f"unsupported scaling method {scaling.get('method')!r}")

    system_mw, provenance = eia930.load_demand(
        spec["respondent"], spec["local_date"], api_key=api_key
    )
    scaled_mw, factor = eia930.scale_to_fleet(
        system_mw, capacity, float(scaling["peak_fraction"])
    )

    provenance = dict(provenance)
    provenance.update({
        "scaling_method": scaling["method"],
        "peak_fraction": float(scaling["peak_fraction"]),
        "scale_factor": factor,
        "system_peak_mw": float(system_mw.max()),
        "system_trough_mw": float(system_mw.min()),
        "system_peak_trough_ratio": float(system_mw.max() / system_mw.min()),
        "scaled_peak_mw": float(scaled_mw.max()),
        "scaled_trough_mw": float(scaled_mw.min()),
    })

    # pandas Timestamps become ISO-8601 UTC strings HERE. Past this line the
    # model layer has no pandas dependency and no timezone to get wrong.
    loads = (Load(bus=SINGLE_BUS,
                  mw={t.isoformat(): float(mw) for t, mw in scaled_mw.items()}),)
    return loads, provenance


def _loads_static(spec):
    """Demand declared per bus in the config. One snapshot, no clock.

    M3 changes the network and holds the load source still, so that the
    milestone adds exactly one new failure surface. There are no timestamps to
    carry, so the single hour is labelled "static" rather than given a
    fabricated UTC time -- a made-up timestamp would look like real data and
    would sort alongside M2's hours as though it belonged there.

    The hour label is opaque to the solver: solve_dispatch_day already accepts
    any hashable, and Scenario.hours sorts whatever it is given.
    """
    mw = spec["mw"]
    if not mw:
        raise ValueError("load.mw is empty: no demand to serve")
    loads = tuple(
        Load(bus=bus, mw={STATIC_HOUR: float(v)}) for bus, v in mw.items()
    )
    return loads, {"load_source": "static", "static_load_mw": dict(mw)}


def _loads_profile(spec):
    """Per-bus demand across a horizon, declared in the config. The M4 path.

    M3's static load is one snapshot; this is the same buses across a day. The
    shape is a list of fractions and peak_mw is the per-bus demand at the top
    of it, so bus b in hour t carries

        load[b][t] = peak_mw[b] * shape[t]

    Writing it as peak x shape rather than as a 24-entry list per bus is the
    RTS idiom (an area profile times a bus participation factor), and it buys
    one property worth having: the hour where shape == 1.0 is EXACTLY M3's
    static case. So M4 contains M3 as its peak hour, and the published 5-bus
    LMPs stay a live regression test instead of becoming a historical note.

    That is why shape must peak at 1.0 and not merely at its own maximum --
    a shape topping out at 0.95 would still solve, and would quietly move the
    anchor hour off the case it is supposed to reproduce.

    Hours are the integers 0..23, as at M1. Not UTC strings: case5 is a
    textbook network with no location and no clock, and stamping a timezone on
    it would make a fabricated instant look like ingested data. The UTC
    discipline belongs to real sources, and it arrives with RTS-GMLC at M5.
    """
    shape = spec["shape"]
    peak = spec["peak_mw"]
    if not shape:
        raise ValueError("load.shape is empty: no hours to solve")
    if not peak:
        raise ValueError("load.peak_mw is empty: no demand to serve")

    shape = [float(v) for v in shape]
    for t, v in enumerate(shape):
        if v <= 0:
            raise ValueError(f"load.shape[{t}] = {v}; must be > 0")
    top = max(shape)
    if abs(top - 1.0) > 1e-9:
        raise ValueError(
            f"load.shape must peak at exactly 1.0, got {top}. peak_mw is read "
            "as the demand at the top of the shape, so a shape that peaks "
            "anywhere else silently rescales every bus."
        )

    loads = tuple(
        Load(bus=bus, mw={t: float(mw) * shape[t] for t in range(len(shape))})
        for bus, mw in peak.items()
    )
    provenance = {
        "load_source": "profile",
        "horizon_hours": len(shape),
        "peak_load_mw": dict(peak),
        "shape_peak_hour": shape.index(top),
        "shape_min": min(shape),
        "peak_trough_ratio": top / min(shape),
    }
    return loads, provenance


def _network(config):
    """Buses and branches. Empty before M3, and empty is a claim, not a gap."""
    net = config.get("network")
    if net is None:
        return (Bus(SINGLE_BUS),), ()

    # Bus order is the config's, and it is load-bearing: it fixes the column
    # order of every matrix in src/network/.
    buses = tuple(Bus(name) for name in net["buses"])
    branches = tuple(
        Branch(
            name=name,
            from_bus=spec["from"],
            to_bus=spec["to"],
            reactance_pu=float(spec["reactance_pu"]),
            limit_mw=float(spec["limit_mw"]),
        )
        for name, spec in net.get("branches", {}).items()
    )
    return buses, branches


def scenario_from_config(config, api_key=None, origin="<dict>"):
    """Assemble the Scenario a config DICT declares. No solving, no plotting.

    Dispatches on load.source. Each source owns its own validation, so a new
    one cannot inherit another's guards by accident.

    Takes a parsed dict rather than a path so that a config can arrive from
    somewhere other than the filesystem -- a POST body, a parameter sweep that
    varies one line limit per solve -- without that caller having to write a
    temporary YAML file. build_scenario is the thin file-reading wrapper.

    origin is recorded as provenance["config"]. It is a label, not something
    anything reads back, and it exists so a run directory can still say where
    its numbers came from when there was no file.
    """
    generators = _fleet(config)
    capacity = sum(g.pmax_mw for g in generators)

    spec = config["load"]
    source = spec.get("source")
    if source == "eia930":
        loads, provenance = _loads_eia930(spec, capacity, api_key)
    elif source == "static":
        loads, provenance = _loads_static(spec)
    elif source == "profile":
        loads, provenance = _loads_profile(spec)
    else:
        raise ValueError(f"unsupported load source {source!r}")

    buses, branches = _network(config)
    known = {b.name for b in buses}
    for g in generators:
        if g.bus not in known:
            raise ValueError(f"generator {g.name} at unknown bus {g.bus!r}")

    provenance = dict(provenance)
    provenance.update({
        "config": str(origin),
        "fleet_capacity_mw": capacity,
    })
    if "slack" in config.get("network", {}):
        # Not a Scenario field: the slack is a choice the pricing code makes,
        # and trap 2 says nothing physical depends on it. Recorded so a run
        # can be reproduced, not so the solver can read it back.
        provenance["slack"] = config["network"]["slack"]

    return Scenario(
        name=config["name"],
        generators=generators,
        loads=loads,
        buses=buses,
        branches=branches,
        provenance=provenance,
    )


def build_scenario(config_path, api_key=None):
    """Read a config file and assemble its Scenario.

    The filesystem half of scenario_from_config, kept as its own name because
    every caller in this repo -- tests, figures, the __main__ blocks -- holds
    a path and nothing else.
    """
    return scenario_from_config(
        load_config(config_path), api_key=api_key, origin=config_path
    )
