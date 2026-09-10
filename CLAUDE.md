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
- **Never commit automatically.** Make the changes, run the tests, report what
  changed, and stop. The author decides what gets committed and when, and writes
  the commit. Do not run `git commit`, `git push`, or `git add` unless explicitly
  asked in that message.
- **Test files are named `test_<milestone>_<context>.py`** — e.g. `test_m0_single_bus.py`, `test_m3_network.py`. The milestone says *when* a test was written and ties it to the table below; the context says *what physical setup it covers*, which is what still means something at M7. A milestone alone (`test_m0.py`) ages into a date stamp. A context alone loses the spine of the repo.
  Inside the file, group tests by lifespan, not by milestone: invariants that hold forever (energy balance, capacity bounds, cost consistency) belong in their own class, separate from the special cases a later milestone supersedes. The settlement identity at M0 is the example — it holds with congestion pinned at zero, and M3 replaces that zero with a real congestion term rather than deleting the test.

## Visualization

**Every plot is a publication figure.** Assume it will be printed in a journal
article, at column width, in grayscale, next to a caption. Nothing in this repo
gets a default-styled throwaway chart.

### Register

A figure labels; it does not narrate. The argument lives in the caption and in
prose — the image carries the evidence.

- **Titles are nominal phrases, not claims.** "Clearing price against demand" —
  not "Price is a staircase; the risers are where the dual breaks down." If a
  sentence is worth saying, say it in the caption.
- **Multi-panel figures use `(a)`, `(b)` labels**, set left, in the panel title.
  No figure-level suptitle and no subtitle sentence.
- **Annotations are terse.** `λ = 35`, `λ ∈ [20, 35]`. Not `λ = $35/MWh — the
  dual on energy balance`, not `no unique price`.
- **Units appear once, on the axis label.** Never repeated on every mark.
- **Capitalize every piece of display text.** Titles, panel labels, axis labels,
  legend entries, annotations, and inset tables all begin with a capital letter.
  Sentence case, not Title Case. Identifiers that come from the data (`g1`, bus
  names, `λ`) keep their literal form.
- **No bold, restrained type scale.** Roughly 8.5–10.5 pt. Emphasis comes from
  position and whitespace, not weight. This holds for text sitting on a shaded
  fill too — fix its legibility with font *color*, never with weight.
- **One typographic hierarchy, strictly ordered.** Panel title (10.5 pt) >
  axis label (9.5 pt) > annotation (8.5–9 pt) > tick label (8.5 pt). There is
  no tier above panel title, because there is no figure-level title.
- **Text over a shaded area takes a contrasting ink.** Surface-white on a deep
  fill, dark ink on a light one. Check it against the *rendered* fill, which is
  the color times its alpha over the surface, not the nominal hex.
- **Units keep their real capitalization.** `MW`, `MWh`, `$/MWh`, `$/MMBtu` —
  never `mw`, `mwh`. Case is meaning: `m` is milli and `M` is mega.
- **Identifiers match the prose and the equations.** If the text says `g1` and
  `λ`, the figure says `g1` and `λ` — not `G1`, not `Lambda`.
- **No arrow annotations.** Arrows collide with the labels they point at and
  almost always mean the text is in the wrong place. Put the text on empty
  ground adjacent to what it describes.
- **Resolve collisions by moving text, not by shrinking it.** If nothing fits,
  the figure has too much in it — cut content or split the panel.

### Construction

- **Axes are labeled with units, in parentheses, at the end.** `Dispatch (MW)`,
  `Clearing price λ ($/MWh)`, UTC timestamps. Always.
- **Colorblind-safe palette, validated, not eyeballed.** Assign hues by identity
  in a fixed order; never cycle. Sequential data gets one hue light-to-dark;
  diverging data gets two hues with a neutral midpoint. Never a rainbow.
- **Direct-label series** wherever they can be placed without collision. A legend
  is present for two or more series; direct labels supplement it, not replace it.
- **One y-axis.** Never a dual-axis chart. Two measures of different scale get
  two panels or an indexed common base.
- **Recessive grid and axes.** No top or right spine, no heavy gridlines, no
  chartjunk, no 3-D, no drop shadows.
- **Gridlines earn their place by helping the eye reach an axis.** A 24-hour
  series carries ticks and vertical gridlines every 6 hours, so the day has a
  visible rhythm and a reader can find the evening ramp without counting.
- **Annotations over a gridline or a data line get a bbox.**
  `bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none", pad=1.5)` — a
  line struck through a label is a silent failure, and moving the text is not
  always possible. Anchor a set of related annotations the same way relative to
  the feature each describes; do not alternate above and below to dodge.
- **Overlapping marks get a thin surface-colored edge** so a cluster reads as
  several points rather than one blob. Scatter, stacked fills, and adjacent
  bars alike.
- **Let a layout manager do the spacing.** `constrained_layout=True` or
  `tight_layout()`, never hand-tuned margins. A colorbar is allocated its own
  width and padding; it must not squeeze the panel it belongs to.
- **Curves are computed, not drawn.** A price-vs-demand staircase is many solves,
  not a hand-placed polyline. If the figure asserts something about the model,
  the model has to produce it.
- **300 dpi raster plus a vector copy** (PDF or SVG). Raster alone is not
  publishable.

### Before accepting a figure

**Render it and look at it. Every time.** Silent failures are the norm in
matplotlib: an unescaped `$` swallows the rest of the string into mathtext,
labels overlap, text overflows the axes. None of these raise, and no test
catches them. Open the file and inspect it.

Expect to iterate. Moving one label routinely creates a collision somewhere
else, so re-render and re-inspect after every adjustment.

### Placement

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

Start at M0. Each milestone adds exactly one new way to be wrong — one new
failure surface — so that when a result looks wrong there is only one place to
look. A milestone is done when its **Goal** column is demonstrably true, not
when the code runs.

| | Milestone | Description | Goal — done when this is true | Est. |
|---|---|---|---|---|
| **M0** ✓ | Single-bus clearing | 3 generators, 1 bus, 1 hour. LP dispatch, minimize offer cost subject to one energy-balance equality. Price read off the dual on that constraint. | λ equals the marginal unit's offer, verified by hand and asserted in tests. Degenerate breakpoints assert bounds, not values. Settlement identity holds with the congestion term at zero. | 45 min |
| **M1** | Time indexing | Same fleet, 24 hours, hardcoded synthetic load shape. `p[g,t]`, 24 balance constraints, 24 duals. No data source, no timezone, no ramp limits. | A joint 24-hour solve reproduces 24 independent single-hour solves *exactly*. Hours are provably separable, because nothing yet couples them. | an evening |
| **M2** | Real load data | Replace the synthetic shape with EIA-930 hourly demand, API v2. Establishes the ingest → `Scenario` boundary and the UTC discipline. | Row count asserted against hours requested; index is UTC with no gaps or duplicates; the scaling choice from system load to fleet capacity is written down and justified. | a weekend |
| **M3** | DC network and congestion | PJM 5-bus example. Susceptance matrix, PTDF relative to a slack, line flow limits. `LMP[i] = λ + Σ PTDF[l,i]·μ[l]`. | Published PJM 5-bus LMPs reproduced exactly and committed as a regression test. Price *differences* invariant to slack choice. | 1 week |
| **M4** | Full DC OPF at scale | RTS-GMLC network, fleet, and profiles. One full day, every hour, real topology. | Settlement identity holds to floating-point tolerance on every hour: `Σ load payments − Σ gen revenue == Σ μ[l]·limit[l]`. | 2 weeks |
| **M5** | Unit commitment | Startup cost, min up/down time, min stable output. MILP for the binaries, then re-solve as an LP with binaries fixed. | The two-pass structure produces valid duals where the MILP alone cannot. No unit runs below its minimum. M1's separability test now *fails*, and you can explain exactly why. | 2 weeks |
| **M6** | Storage and reserves | Energy and reserve co-optimization. Storage state of charge, charge/discharge, round-trip efficiency. | Reserve price appears as its own dual. Storage arbitrages the price spread without being told to. Energy and reserve prices are jointly consistent. | 2 weeks |
| **M7** | Validation and write-up | Real fleet and real load. Compare against published LMPs pulled by `gridstatus` — the held-out answer key, touched here for the first time. | Structural agreement with published prices: price separation events, the evening ramp, negative hours. Every residual gap is explained as a market feature not yet modeled, not tuned away. | 2 weeks |

Do not skip to a later milestone. Each one's test suite is the foundation for the
next, and the invariants established early are what catch the subtle failures
later.

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
