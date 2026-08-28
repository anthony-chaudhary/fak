# Issue #9765 — performance RSI adversarial QA witness

QA-only tests in `internal/perfrsiscore/perfrsiscore_test.go` cover:

- empty and whitespace-only documents;
- truncated, wrong-type, trailing-value, and unknown-field JSON;
- an oversized 10,000-dimension document;
- malformed-file and unreadable-directory loop-turn load errors.

## Witnesses

Captured in `test-output.txt`:

- `go test -count=1 ./internal/perfrsiscore -run 'Edge|Adversarial' -v` — passed.
- `go test -count=1 ./internal/perfrsiscore` — passed.
- `go vet ./internal/perfrsiscore` — passed.
- `make test-fast` — reached `architest-gate`, then failed on pre-existing trunk import violations outside issue #9765: `nativeperf -> agent` and `qwen38quantrun -> agent`. Its build, timeout-regression, and repository vet stages passed first.

No production code changed.
