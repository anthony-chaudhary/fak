# KV capacity normalization

Provider and runtime KV-cache metrics use different native units. `internal/modelperfobs.NormalizeKVCapacity` converts explicit observations into a comparable `fak-kv-capacity/1` snapshot without erasing the source values.

## What the normalizer accepts

The normalizer covers both shipped dialects:

- **Block-oriented runtimes** report total/free/used blocks plus block geometry.
- **Direct runtimes** report token and byte capacity directly.

It derives three comparable unit families when the inputs support them: resident **tokens**, resident **bytes**, and **occupancy**. Reusable/high-water token counters and between-scrape deltas remain separate from capacity.

## Honesty floor

A derived value is emitted only when its denominator and unit are explicit. Missing block geometry, counter resets, runtime-identity changes, impossible occupancy, and disagreeing native units produce typed diagnostics instead of invented values. Observed byte counters take precedence over geometry estimates, and every rendered report places native observations before normalized values.

## Use the Go seam

```go
snapshot := modelperfobs.NormalizeKVCapacity(current, previous)
err := modelperfobs.WriteKVCapacityMarkdown(w, snapshot)
```

`current` and `previous` are `modelperfobs.KVMetricSample` values. Use `previous == nil` for a point-in-time snapshot. The normalization floor and exact dialect fixtures are pinned by `internal/modelperfobs/kv_capacity_test.go`.
