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
from src.model.inputs import Bus, Generator, Load, Scenario

SINGLE_BUS = "bus1"


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def _fleet(config):
    return tuple(
        Generator(
            name=name,
            bus=SINGLE_BUS,
            cost_usd_per_mwh=float(spec["cost_usd_per_mwh"]),
            pmax_mw=float(spec["pmax_mw"]),
            pmin_mw=float(spec.get("pmin_mw", 0.0)),
        )
        for name, spec in config["fleet"].items()
    )


def build_scenario(config_path, api_key=None):
    """Assemble the Scenario a config declares. No solving, no plotting.

    The scale factor is computed here, from the fleet in the same config, and
    recorded in provenance. It is never hardcoded and never defaulted: a
    config without a scaling block is an error, because an unjustified
    rescale of real load is exactly the thing M2 exists to make explicit.
    """
    config = load_config(config_path)
    generators = _fleet(config)
    capacity = sum(g.pmax_mw for g in generators)

    spec = config["load"]
    if spec.get("source") != "eia930":
        raise ValueError(f"unsupported load source {spec.get('source')!r}")
    if "scaling" not in spec:
        raise ValueError(
            f"{config_path}: load.scaling is required; real load must not be "
            f"rescaled onto a toy fleet without the choice being written down"
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
        "config": str(config_path),
        "scaling_method": scaling["method"],
        "peak_fraction": float(scaling["peak_fraction"]),
        "scale_factor": factor,
        "fleet_capacity_mw": capacity,
        "system_peak_mw": float(system_mw.max()),
        "system_trough_mw": float(system_mw.min()),
        "system_peak_trough_ratio": float(system_mw.max() / system_mw.min()),
        "scaled_peak_mw": float(scaled_mw.max()),
        "scaled_trough_mw": float(scaled_mw.min()),
    })

    # pandas Timestamps become ISO-8601 UTC strings HERE. Past this line the
    # model layer has no pandas dependency and no timezone to get wrong.
    load = Load(
        bus=SINGLE_BUS,
        mw={t.isoformat(): float(mw) for t, mw in scaled_mw.items()},
    )

    return Scenario(
        name=config["name"],
        generators=generators,
        loads=(load,),
        buses=(Bus(SINGLE_BUS),),
        branches=(),           # no network until M3
        provenance=provenance,
    )
