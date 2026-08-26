---
title: "Work-done query contract"
description: "fak info --work-done-json is the automation and future-UI seam for the same accounting object rendered by the TUI. It emits fak.info.work-done-query/1;"
---
# Work-done query contract

`fak info --work-done-json` is the automation and future-UI seam for the same accounting object rendered by the TUI. It emits `fak.info.work-done-query/1`; consumers must use stable enum and field values, not terminal labels.

Session total:

```console
fak info --gateway-url http://127.0.0.1:PORT --work-done-json
```

Bounded interval (samples cumulative gateway counters at both endpoints):

```console
fak info --gateway-url http://127.0.0.1:PORT --work-done-json --work-done-window 30s
```

The envelope contains `generated_at`, a window descriptor, and `work_done`. A `session_total` window has one endpoint. A `bounded` window carries start/end timestamps and an integer nanosecond duration. If cumulative counters regress or the baseline fingerprint changes between endpoints, `reset` is true, `reset_reason` names the boundary, and all deltas are explicitly unavailable; the query never converts a reset into zero savings.

Metric records carry unit, numeric value, optional integer-safe value for count/token consumers, evidence, baseline ID, basis, and `unavailable_reason`. Durations are seconds in the accounting metric and nanoseconds in the interval descriptor. Sources retain the versioned provenance and exclusivity contract documented in [`work-done-sources.md`](work-done-sources.md). Baselines retain the compatibility contract in [`work-done-baselines.md`](work-done-baselines.md).

## Stability

- Schema names are versioned; incompatible changes require a new schema version.
- Unknown JSON fields must be ignored by consumers.
- Source ordering is deterministic but consumers should key by stable source ID.
- Missing evidence is represented with `available: false` and a reason, never an implicit zero.
- NaN and infinity are never emitted.
- Display strings and TUI row wording are not API.

`fak info --json` remains the broad diagnostic snapshot and includes `work_done` for compatibility. New integrations should prefer `--work-done-json` because it excludes unrelated gateway state and supports bounded deltas.
