# Disambiguation projected-scale witness — 2026-08-17

**Provenance: OBSERVED.** These are one local benchmark run, not a universal efficiency claim.

- Dataset: 4,096 synthetic valid entries, each with one alias and one contrast.
- Hardware scope: Windows/amd64, AMD Ryzen 9 9950X 16-Core Processor, Go 1.26.6.
- Reproduce: `go test ./internal/disambiguation -run '^$' -bench 'ProjectedIndex$' -benchtime=100ms -count=1`.
- Tuned baselines: canonical sort + JSON encoding for generation; prebuilt map lookup for exact/alias queries; indexed canonical-entry exact-locator scan for reverse lookup.

| Operation | Observed result |
|---|---:|
| Generate deterministic index | 240,770,300 ns/op |
| Exact canonical query | 172.2 ns/op |
| Exact alias query | 105.5 ns/op |
| Reverse source-path query | 2,814,769 ns/op |

The benchmark reports `entries`, `OBSERVED_provenance`, and `tuned_indexed_baseline` metrics and logs operation, dataset, baseline, OS/architecture, and Go version. Re-run rather than treating this snapshot as a promise on another machine.
