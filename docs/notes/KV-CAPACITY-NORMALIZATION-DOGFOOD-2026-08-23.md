# KV capacity normalization dogfood — 2026-08-23

## Verdict

The repository's two committed runtime-dialect captures normalize to the same three comparable capacity families. No defect surfaced.

## Captured run

```text
go test ./internal/modelperfobs -run 'KVCapacityRepositoryDogfood|KVCapacityDialectFixtures' -count=1 -v
PASS
```

| Source capture | Dialect | Resident tokens | Resident bytes | Occupancy |
|---|---|---:|---:|---:|
| `internal/modelperfobs/testdata/kv-capacity-direct.json` | `fak-kv-direct-metrics/1` | 16384 | 2147483648 | 0.5 |
| `internal/modelperfobs/testdata/kv-capacity-block.json` | `fak-kv-block-metrics/1` | 16384 | 2147483648 | 0.5 |

The captures retain their native direct-token/direct-byte and block-geometry observations. `NormalizeKVCapacity` derives tokens, bytes, and occupancy without replacing those source values. The committed dogfood test decodes both files through the production seam, renders both normalized snapshots, and fails if either dialect or any comparable value drifts.

## Defects filed

None. The run met the declared two-dialect, three-unit-family envelope without a refusal or disagreement diagnostic.
