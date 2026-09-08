<!-- fak-modelbench-key: rawdecode-paired-tokenizer-template-provenance -->
# feat(modelbench): carry paired GGUF tokenizer and template provenance

```routing
lane: modelbench-strix-receipt
paths: ["internal/rawdecode/executor.go", "internal/rawdecode/executor_test.go", "cmd/modelbench/raw_decode.go", "cmd/modelbench/raw_decode_test.go", "docs/tickets/strix-performance/TICKET-modelbench-rawdecode-tokenizer-template-provenance.md"]
expected_steps: 6
```

GitHub: [#12288](https://github.com/anthony-chaudhary/fak/issues/12288)

## Parent context

Parent [#12096](https://github.com/anthony-chaudhary/fak/issues/12096); extends [#5881](https://github.com/anthony-chaudhary/fak/issues/5881) after [#12287](https://github.com/anthony-chaudhary/fak/issues/12287).

## Current state

Rawdecode already hashes and parses one opened GGUF stream and carries artifact, manifest, quantization, and model identity. The canonical receipt remains unavailable because tokenizer and template digests are not carried.

## Working spine

Exact opened GGUF -> same parsed header -> strict paired identity -> rawdecode observation -> all-or-nothing canonical model tuple -> independent receipt gates.

## Core through-line

Derive both digests from the already parsed header, carry them together, and require both in modelbench's existing full model-identity gate. Missing metadata preserves execution diagnostics but leaves the attempt `UNAVAILABLE` and zero-credit.

## Gold-plating boundary

No parser or hash changes, tokenizer execution, prompt rendering, filename inference, sidecars, download, hardware, service, source/device/runtime identity, or performance claim. Never populate a partial model tuple.

## Done condition / witness

`go test ./internal/rawdecode ./cmd/modelbench -count=1` and `go vet ./internal/rawdecode ./cmd/modelbench` prove paired propagation and missing-field refusal.

## Done condition

- [ ] Both digests come from the same parsed header.
- [ ] Complete provenance maps exactly.
- [ ] Missing either digest clears the canonical model tuple and remains zero-credit.

## Verifiable Witness

The focused commands above are deterministic and device-free. No physical or performance claim is made.

## Likely files

- `internal/rawdecode/executor.go`
- `internal/rawdecode/executor_test.go`
- `cmd/modelbench/raw_decode.go`
- `cmd/modelbench/raw_decode_test.go`
- this ticket

## Lane

`modelbench-strix-receipt`; two Go packages, six expected steps.

## Closure binding

The resolving commit cites #12288 and carries `(fak modelbench)`.
