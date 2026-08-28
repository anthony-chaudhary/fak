# Issue #9793 — Apple-Silicon Qwen GDN batch witness

Verdict: **PASS**. Both exact #9626 native Metal tests enumerated and passed on
darwin/arm64 with cgo enabled. There were zero skip events and the package exited
successfully. The test ran from a `git archive` of committed tip
`5e0db65c50288ec16d9f4f6b5a50bf7174f709c9`, so peer-dirty files in the shared
checkout were not visible to the compiler or test binary.

The public-safe machine class is an Apple Silicon laptop with an Apple M3 Pro
GPU and Metal 4 support. No hostname, user, serial number, private path, or
device identifier is recorded.

## Bound source

- Source tree: `06c7665224d214e391633154d96af08d01d32b13`.
- Module: `internal/metalgemm@r70+ga70d7891e`.
- Required ancestors: `2f0fcd0f5353d1d23d5078b22a76421752496b40`
  and `d89e5ae081f8162d3cdf6cd5f414dd732f9346da`.
- Capture environment: `darwin/arm64/1` from
  `go env GOOS GOARCH CGO_ENABLED`; Go 1.26.6.

## Exact result

The command was:

```console
CGO_ENABLED=1 go test -json ./internal/metalgemm -run '^(TestQwen35DecodeBatchIndependentStateSingleFence|TestQwen35FullAttentionDecodeBatchIndependentKVSingleFence)$' -count=1
```

`TestQwen35FullAttentionDecodeBatchIndependentKVSingleFence` passed in 0.30
seconds. `TestQwen35DecodeBatchIndependentStateSingleFence` passed in 2.49
seconds. The package passed in 3.073 seconds; measured wall time was 5 seconds.
The 46,906-byte, 151-line JSONL stdout log had SHA-256
`158676fced1e1b6b79fa74ea69fc819fc8dce5e366a92fa637c167d75e5a8b44`;
stderr was empty.

The machine-readable [receipt](receipt.json) carries the exact test names,
enumeration/PASS status, zero-SKIP count, source ancestry, sanitized machine
probe, timings, raw-log digest, and replay procedure.

## Boundary

This is a native correctness witness for #9626. It makes no throughput, model
artifact, selector, or fallback claim, and it does not close #9626 by itself.
