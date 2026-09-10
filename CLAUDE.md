# CLAUDE.md

Context for working in this repo. Read fully before writing code.

## What this is

An electricity market clearing engine, built from scratch. It answers: given generator offers, a load forecast, and a transmission network with limits, who runs, at what output, and what is a MWh worth at each bus.

Scope: nodal energy market, DC approximation, 24-hour day-ahead solve. No AC power flow, no reactive power, no financial transmission rights.

## Purpose (read this before "helping")

This is a learning project. The author is a mechanical engineer transitioning into power markets. The point is to build the formulations by hand and understand where prices come from.

Therefore:

- **Do not reach for PyPSA, GenX, or pandapower to solve the problem.** They are reference implementations to check answers against, not starting points. Using them defeats the exercise.
- **Do not hand over a finished module** when the author is on a milestone they haven't attempted. Explain the formulation, then let them write it.
- Do write scaffolding, data ingestion, tests, and plotting code — that's not the part worth learning by hand.
- When explaining a new concept, include a diagram or visual, not just prose.

## Core mechanics a contributor must understand

**Prices are dual variables.** The dual on the energy balance constraint is the marginal cost of serving one more MW. That is the price. It is not computed by a pricing rule; it falls out of the optimization.

**The two-pass structure is mandatory.** Unit commitment is a MILP. A MILP has no meaningful duals, so it cannot produce a price. The sequence is:

1. Solve UC as a MILP → binary on/off schedule
2. Fix the binaries at those values
3. Re-solve dispatch as an LP → dispatch levels AND valid duals
4. Assemble prices from the duals

Real ISOs do exactly this. Never try to read prices off the MILP.

**LMP assembly:**

```
LMP[i] = lambda + sum_over_lines( PTDF[line, i] * mu[line] )  [+ loss term]
```

- `lambda` = dual on system energy balance. Same at every bus. The energy component.
- `mu[line]` = dual on that line's flow limit. Zero unless binding. Positive only when congested.
- The sum is the congestion component.

Sign convention depends on how the flow constraint was written. Verify against a case with a published answer rather than trusting the formula.

**The settlement identity is the primary correctness test:**

```
sum(load payments) - sum(generator revenue) == sum_over_lines( mu[line] * limit[line] )
```

Must hold to floating-point tolerance. If it doesn't, the bug is in PTDF construction, a sign convention, or dual extraction. Run this check on every solve.

## Repo layout

```
market-engine/
├── data/
│   ├── raw/                 exactly as downloaded, never edited, gitignored
│   ├── interim/             parsed, not yet aligned
│   └── processed/           canonical parquet, one schema, UTC index
├── src/
│   ├── ingest/              one module per source
│   │   ├── eia930.py
│   │   ├── eia_fuels.py
│   │   ├── nrel_profiles.py
│   │   ├── rts_gmlc.py
│   │   └── actuals.py       gridstatus pulls, HELD OUT from model inputs
│   ├── network/
│   │   ├── topology.py      buses, branches, susceptance matrix
│   │   └── ptdf.py          shift factors
│   ├── model/
│   │   ├── inputs.py        Generator, Bus, Branch, Load, Scenario
│   │   ├── unit_commitment.py
│   │   ├── dispatch.py
│   │   └── pricing.py
│   ├── settle/settlement.py
│   ├── validate/compare.py
│   └── viz/
├── configs/                 yaml, one file per scenario, fully declarative
├── runs/                    one dir per solve, config snapshot copied in
├── tests/                   test_<milestone>_<context>.py
└── notebooks/               exploration only, nothing importable
```

## Conventions

- **The model layer must not know where data came from.** Ingest produces a `Scenario` object; the solver consumes it. Never call a data source from inside `src/model/`.
- **Everything is UTC internally.** Convert at the ingest boundary only.
- Units: MW for power, MWh for energy, $/MWh for prices, $/MMBtu for fuel.
- Every run writes its config into `runs/<timestamp>/` so results are reproducible.
- `data/raw/` is append-only. Never edit a downloaded file.
- Actuals in `src/ingest/actuals.py` are a held-out answer key. Never let them reach a `Scenario`.
- **Test files are named `test_<milestone>_<context>.py`** — e.g. `test_m0_single_bus.py`, `test_m2_network.py`. The milestone says *when* a test was written and ties it to the table below; the context says *what physical setup it covers*, which is what still means something at M5. A milestone alone (`test_m0.py`) ages into a date stamp. A context alone loses the spine of the repo.
  Inside the file, group tests by lifespan, not by milestone: invariants that hold forever (energy balance, capacity bounds, cost consistency) belong in their own class, separate from the special cases a later milestone supersedes. The settlement identity at M0 is the example — it holds with congestion pinned at zero, and M2 replaces that zero with a real congestion term rather than deleting the test.

## Visualization

**Every plot is a publication figure.** Assume it will be printed in a journal
article, at column width, in grayscale, next to a caption. Nothing in this repo
gets a default-styled throwaway chart.

Non-negotiable:

- **The title states the finding, not the variables.** "Price is a staircase;
  the risers are where the dual breaks down" — not "lambda vs demand."
- **Axes are labeled with units.** `$/MWh`, `MW`, `MWh`, UTC timestamps. Always.
- **Colorblind-safe palette, validated, not eyeballed.** Assign hues by identity
  in a fixed order; never cycle. Sequential data gets one hue light-to-dark;
  diverging data gets two hues with a neutral midpoint. Never a rainbow.
- **Direct-label series** wherever they can be placed without collision. A legend
  is present for two or more series; direct labels supplement it, not replace it.
- **One y-axis.** Never a dual-axis chart. Two measures of different scale get
  two panels or an indexed common base.
- **Recessive grid and axes.** No top or right spine, no heavy gridlines, no
  chartjunk, no 3-D, no drop shadows.
- **300 dpi raster plus a vector copy** (PDF or SVG). Raster alone is not
  publishable.
- **Render it and look at it before accepting it.** Silent failures are the norm
  in matplotlib: unescaped `$` becomes mathtext, labels collide, text overflows
  the axes. None of these raise. Open the file and inspect it every time.

Figures are written into `runs/<timestamp>/` alongside the config snapshot that
produced them, so any figure can be traced back to the exact solve behind it.

Plotting code lives in `src/viz/` and is importable: a function returns a
`Figure`, and only `__main__` writes files. Never bury a `savefig` inside model
or ingest code.

## Data sources

| Input | Source |
|---|---|
| Network + fleet + profiles | RTS-GMLC (NREL, GitHub) — primary test system |
| Small test networks | MATPOWER cases (case14, case118) |
| Hourly load | EIA-930 Hourly Grid Monitor, API v2 |
| Fuel prices | EIA API v2 (Henry Hub, SoCal Citygate, Waha) |
| Generator specs | EIA Form 860, Form 923, EPA CAMD |
| Wind/solar shapes | NREL NSRDB, PVWatts, WIND Toolkit |
| Real offer curves | ERCOT 60-Day SCED Disclosure |
| Validation LMPs | `gridstatus` Python library |

Offers start as a cost proxy: `fuel_price * heat_rate + VOM`. Upgrade to real ERCOT offer curves later.

## Tooling

- Pyomo + HiGHS (`pip install pyomo highspy`). Julia/JuMP is a valid alternative and is where NREL's Sienna stack lives.
- If UC solve time explodes, set a MIP gap of 0.1% rather than chasing optimality. Production systems do the same.

## Milestones

Start at M0.

| | Goal | Est. |
|---|---|---|
| M0 | 3 generators, 1 bus, 1 hour. Verify by hand that the dual equals the marginal unit's cost. | 45 min |
| M1 | Same fleet, 24 hours, real EIA-930 load shape. | a weekend |
| M2 | PJM 5-bus example with DC network. Reproduce published LMPs exactly, commit as a test. | 1 week |
| M3 | RTS-GMLC, full DC OPF, one day, settlement identity check passing. | 2 weeks |
| M4 | Unit commitment: startup cost, min up/down, min output. Two-pass structure. | 2 weeks |
| M5 | Storage plus energy/reserve co-optimization. | 2 weeks |
| M6 | Real fleet and load, validate against published LMPs, write up. | 2 weeks |

Do not skip to a later milestone. Each one's test suite is the foundation for the next.

## Known traps

1. **Time zones.** EIA-930 is UTC. ERCOT market time is Central Prevailing Time with DST — one 23-hour and one 25-hour day per year, including a duplicated hour. CAISO is Pacific Prevailing. Assert 8760 or 8784 rows in a year.
2. **PTDF slack bus.** PTDF is defined relative to a slack. Changing the slack shifts all prices by a constant. Price *differences* are invariant; that's what matters.
3. **LP degeneracy.** When multiple optimal bases exist, duals are arbitrary and prices flip between values on near-identical inputs. Not a bug.
4. **Min output without binaries.** A pure LP will run a 600 MW coal unit at 4 MW. Physically impossible. This is why UC exists.
5. **ERCOT has no loss component in its LMPs** — losses are socialized to load. A model with a loss term will not match ERCOT prices.

## Validation stance

Absolute price levels will be wrong; the fleet is synthetic and offers are cost-based. What should match published data is structure: when prices separate across the system, when the evening ramp bites, which hours go negative.

When the model says $40 and the market cleared at $180, the gap is usually a real market feature not yet modeled. Investigate it, don't tune it away.

## Reference

Kirschen & Strbac, *Fundamentals of Power System Economics*. Read chapters as the corresponding milestone comes up.
