---
title: "Next-50 item 5 evidence: measured anchors outrank synthetic Zipf"
description: "Completion witness for DEFAULT-ENABLEMENT-NEXT-50 item 5 — whenever a provider snapshot exists, the observed family distribution is the default anchor source and synthetic Zipf is visibly the FORECAST plane."
---

# Next-50 item 5 — measured anchors outrank synthetic Zipf

Backlog row (see [`DEFAULT-ENABLEMENT-NEXT-50.md`](DEFAULT-ENABLEMENT-NEXT-50.md), line 103):

> **Scoring — Make measured anchors outrank synthetic Zipf whenever a snapshot
> exists.** Default/evidence target: synthetic is visibly `FORECAST`; observed
> family distribution is the default.

Tracked as issue #1523 (parent context #1490).

## Verdict: satisfied by shipped behavior

`fak vcache score` already flips its anchor source to the **measured** observed
family distribution whenever a provider snapshot has turns, and demotes the
synthetic-Zipf workload to a clearly-labeled `FORECAST` plane. This note is the
binding witness; no new scoring code was required.

## Promotion evidence (what makes measured the default)

`cmd/fak/vcache.go` reads the persisted per-turn provider-cache window (the
snapshot a finished `fak manage`/`fak serve` session leaves at the well-known
path) when no `--telemetry`/`--anchors-file` is given. When that snapshot has
turns it ranks the workload from the observed rows and sets
`in.AnchorSource = vcachescore.AnchorSourceMeasured`
(`cmd/fak/vcache.go`, the snapshot branch around the `ProviderTelemetryTurns`
read). Live proof:

```text
$ fak vcache score --snapshot <window.jsonl>
active source: telemetry
anchor source: measured (turns observed 2)
planes: provider=OBSERVED kernel=MISSING context=MISSING external=MISSING forecast=FORECAST
```

The observed family distribution is the active source; the synthetic Zipf
workload has been outranked.

## Demotion / retirement evidence (synthetic stays visibly FORECAST)

When the snapshot is absent, empty, or explicitly disabled with `--snapshot off`,
the score falls open to the synthetic-Zipf workload
(`vcachescore.SyntheticZipfWorkload`, anchor source `synthetic`) and the forecast
plane keeps provenance `FORECAST` — it is never blended into the OBSERVED planes.
Live proof:

```text
$ fak vcache score --snapshot off
active source: planned
anchor source: synthetic (turns observed 0)
planes: provider=MISSING kernel=MISSING context=MISSING external=MISSING forecast=FORECAST
```

`internal/vcachescore/readiness.go` fails closed if the forecast plane is ever
anything other than a separate `FORECAST` plane, so provider/kernel/context/
forecast provenance stays separate as the item requires.

## Repo witnesses (tests)

- `cmd/fak/vcache_test.go` — `TestRunVCacheScoreObservedByDefaultFromSnapshot`
  pins measured/2 with a snapshot and synthetic/0 under `--snapshot off`.
- `internal/vcachescore/score_test.go` — default and telemetry cases assert the
  `AnchorSourceSynthetic` / `AnchorSourceMeasured` split and the separate
  `FORECAST` plane provenance.

Both packages pass under `go test ./cmd/fak/ ./internal/vcachescore/ -count=1`.

## Invalidating assumption

This witness assumes the snapshot the default path reads is a *real* observed
provider window (turns with `cache_read_input_tokens`), not a synthetic fixture
staged at the well-known path. If a synthetic window were persisted there,
`anchor source: measured` would report a fabricated family distribution as
observed — the cold-path guarantee is only as honest as the snapshot writer
(`fak manage`/`fak serve`), which is the surface item 6 hardens.
