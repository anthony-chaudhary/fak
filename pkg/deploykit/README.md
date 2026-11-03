# pkg/deploykit

Deploy primitives for governed binary rollout: verified content-addressed
candidate slots (`artifact`), atomic swap with rollback slots and a
single-flight lock (`swap`), the smoke/provenance identity parse (`stamp`),
the fleet-side health/readiness/identity/drain route contract (`probe`), and
the deterministic output-coherence probe shared with the engine readiness
gate (`coherence`). The receipt envelope is `fak.deploy.receipt/v1`
(`receipt.go`). Production consumers: `internal/gateway` (readiness + probe
routes) and `cmd/fak serve`.

## Subpackages

| Package | Concern |
|---|---|
| `artifact` | Build-or-fetch a candidate, verify bytes against a pinned digest, store it in an immutable generation+digest slot |
| `coherence` | Count-run decode check that catches fluent-looking garbage from a mismatched kernel set |
| `probe` | Mount/verify the health, readiness, identity, and drain routes a deployed endpoint must serve |
| `stamp` | Parse `version --json` provenance identity from a candidate |
| `swap` | Atomic file swap, content-addressed prior-slot save/list/prune, single-flight lock |

## Debt-lane verification (flow ticket debtwave2-pkg-deploykit, 2026-11-03)

Witness: `go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane
deploykit --json`.

Lane facts moved by this wave (ticket census 2026-11-02 → post-wave):

| Metric | Census baseline | After |
|---|---|---|
| Maturity | 6.5 / 8 | 7.5 / 8 |
| Total debt | 3.69 pts | 1.19 pts |
| Benchmark status | unbenchmarked | benchmarked (coherence + stamp production-path benches, then probe + swap + artifact) |
| God-file / god-func | none (max file 503 lines, max func 47 lines) | none |

Work-item verdicts:

1. **God-construct split** — verified none remain at HEAD: the reader reports
   `max_file_lines: 503`, `max_func_lines: 47`, zero `god_file`/`god_func`
   modularity issues. Nothing to split.
2. **Model-specific hardcoding** — verified absent. The reader's
   `model_hardcoding (coherence.go: cohere)` findings are a detector false
   positive: the keyword list contains `cohere` (a model family), and the
   English word *coherence* (this package's output-coherence probe) contains
   that substring. All five hits are the `coherence` package-clause identifier,
   the `Coherent` verdict constant name, its `"coherent"` value, and two uses
   of the constant. No model family, checkpoint name, or provider token is
   hardcoded anywhere under `pkg/deploykit`. Clearing the flag requires
   renaming the `coherence` subpackage, whose importers live outside this
   ticket's boundary (`internal/gateway/readiness_decode.go`,
   `internal/gateway/readiness_warmup_fold_test.go`,
   `internal/gateway/readiness_coherence_test.go`, `cmd/fak/serve.go`) — filed
   as the follow-on below, not done here.
3. **Witness** — maturity rose 6.5 → 7.5 and total debt fell 3.69 → 1.19,
   satisfying the done condition ("maturity up or total_debt down").

### Open follow-ons (out of this ticket's boundary)

- **Dogfood proof**: the lane still reads `dogfooded: false`; the remaining
  maturity point to 8.5+ is a `runtime-proofs.json` witness row for lane
  `deploykit` in `internal/maturity/runtime-proofs.json` (or
  `platform/maturity/runtime-proofs.json`), e.g. a row whose command executes
  the package's real paths and whose `output_contains` binds to that output.
  This file sits outside `pkg/deploykit/`, so it was not touched here.
- **Rename the `coherence` subpackage** (with its four importers) to a name
  that does not contain a model-family keyword, which clears the
  `model_hardcoding` false positive and the associated modularity penalty.
