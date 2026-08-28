# Issue #9766 — performance RSI refusal-contract witness

Tests in `internal/perfrsiscore/perfrsiscore_test.go` exercise existing input, composition, evidence, cycle, improvement, provenance, learning, hardware, comparison, and loop-turn refusal/error contracts. Each table case asserts both rejection and a diagnostic naming the invalid field or corrective contract. No production code changed.

## Witnesses

Captured in `test-output.txt`:

- `go test -count=1 ./internal/perfrsiscore -run 'Refus|Error|Requir' -v` — passed.
- `go test -count=1 ./internal/perfrsiscore` — passed.
- `go vet ./internal/perfrsiscore` — passed.
- `make test-fast` — reached `architest-gate`, then failed on pre-existing trunk import violations outside issue #9766: `nativeperf -> agent` and `qwen38quantrun -> agent`. Its build, timeout-regression, and repository-vet stages passed first.
