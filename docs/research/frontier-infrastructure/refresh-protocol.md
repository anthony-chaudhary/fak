# Refresh and rumor protocol

The corpus is a dated snapshot, not a timeless market map. Refresh it without turning
search snippets or copied leaks into facts.

## Cadence

- **Weekly:** frontier-lab/hyperscaler releases, major funding, acquisitions, cancellations,
  incidents, export controls, and material rumors.
- **Monthly:** startup status, serving/runtime releases, capacity/site announcements, power,
  cooling, network, HBM/packaging, and construction signals.
- **Quarterly:** public-company filings, capex/lease/purchase obligations, revenue/losses,
  backlog, customer concentration, installed capacity, and delivery schedules.
- **On new production-trace research:** extract population, window, fields, fitted
  distributions, quantiles, drift, anonymization, and limitations immediately.

## Search ledger

For each refresh, record the exact date and search families used. Minimum families:

1. entity + `infrastructure`, `cluster`, `GPU`, `TPU`, `Trainium`, `datacenter`, `power`;
2. entity + `inference`, `batching`, `KV cache`, `prefill decode`, `routing`, `SLO`;
3. `LLM production trace`, `workload characterization`, `token distribution`, `prefix reuse`,
   `tenant skew`, `Zipf`, `burst`, `seasonality`, `agent trace`;
4. `AI datacenter` + `grid`, `water`, `cooling`, `permitting`, `opposition`, `delay`,
   `canceled`, `turbine`, `transformer`, `optics`, `HBM`, `CoWoS`;
5. company/project + `funding`, `debt`, `bankruptcy`, `shutdown`, `acquisition`, `license`,
   `partnership`, `release`, `generally available`;
6. company/project + `rumor`, `reportedly`, `sources say`, `leak`, followed by a search for
   the earliest origin and independent corroboration.

Search snippets are discovery aids only. Open the source and record the exact publication
and event dates before adding an entry.

## Rumor lifecycle

A rumor entry requires:

- earliest known origin and whether it had direct sourcing;
- every claimed independent corroborator, with copied/syndicated reports collapsed;
- concrete proposition and expected resolution date;
- confidence and reasons, not just a label;
- expiry or next-review date;
- later state: `unresolved`, `partially_confirmed`, `confirmed`, `refuted`, or `expired`;
- official confirmation/refutation source when one appears.

Rumor entries never support an unqualified architecture or roadmap fact. They can justify
watching a seam, preparing a reversible experiment, or assigning probability in a scenario.

## Contradiction handling

Do not overwrite the older entry when a later source disagrees. Add the new source and
record:

- whether the boundary changed (global vs U.S., IT vs facility power, physical GPUs vs
  H100-equivalents, announced vs online);
- whether the date changed the state;
- whether the estimate methodology differs;
- whether one source is a vendor claim and the other a measured observation;
- the remaining unresolved range.

## Completion bar for a slice

A slice is complete only when it has:

- an explicit actor/topic roster;
- primary-source queries for every actor where primary sources can exist;
- credible secondary reporting for private facts and rumors;
- dated entries with evidence/confidence classes;
- contradictions and missing measurements;
- implications tied to a fak seam or benchmark assumption;
- a validator pass and a visible next-refresh date.
