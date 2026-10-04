<!-- fak-cross-key: oss-refresh-20260929-gguf-offset-bounds -->
# Reject GGUF offsets before unsigned addition wraps

```routing
lane: ggufload
paths: ["internal/ggufload/gguf_config.go", "internal/ggufload/gguf_config_test.go"]
expected_steps: 3
```

## Target repository

Public `anthony-chaudhary/fak`; core runtime mechanism. Dispatch through `fak-flow spawn oss-refresh-20260929-gguf-offset-bounds --route-repo public`. The specification belongs to the public runtime backlog.

## Working spine

Read source-confirmed invariant -> reproduce through the public runtime seam -> bounded correction -> independent regression -> durable green acceptance receipt.

## Current state

ReadConfig checks data+Offset against MaxInt64 after unsigned addition. With data=128 and Offset=MaxUint64-63, the sum wraps to64 and passes. The offset is aligned to32.

## Source and license

llama.cpp merged PR #26979; current source pin a6ea155d3d38b3f6f43d0c0dc29c2d592413ef44 (exact merged change anchor in native cohort study). Source: https://github.com/ggml-org/llama.cpp/pull/26979. Observation 2026-09-29; upstream tracker states are dated. Apache-2.0 (vLLM/LMCache) or MIT (llama.cpp), ADAPT with source attribution retained. No source code is copied by this ticket. The cited upstream PR and revision are the source anchor; this is a bounded source assessment.

## Core through-line

Check the remaining representable range before addition and retain existing valid-file behavior. Use a malformed GGUF fixture through the real reader rather than asserting arithmetic alone.

## Gold-plating boundary

No new inference backend, cache rewrite, rollout or performance claim. No hardware qualification is needed for the CPU correctness invariant. Keep the repair bounded to this outcome and its consuming call path.

## File:Line seam

- `internal/ggufload/gguf_config.go:123` at public baseline `365167aa5c8b5f3252221594eb5cf2a38379f16c`.

## Dedupe and routing

Dedupe key `oss-refresh-20260929-gguf-offset-bounds`. Both-repo capability queries and nearby issues are in the cohort note. The initial 56-contract native census contains no contract with this same outcome. Related tickets remain separate owners; refresh before claim because peer backlog can advance.

## Done condition / witness

All unchecked criteria below must be satisfied through the real consuming entrypoint; preserve correct existing behavior and reject the adversarial case.

## Definition of done

- [x] Reproduce the current gap through the real target entrypoint.
- [x] Implement the bounded change and independent edge-case regression.
- [x] The malformed aligned offset fixture fails on the old code and is rejected after the repair; MaxInt64 boundary cases and normal aligned offsets remain correct.
- [x] Verify `go test ./internal/ggufload -count=1`, public leak/boundary checks, and commit/land through the public native workflow.

Resolved by commit `3f2f32ff1455ef6bfb212a26d5f12ec838fef208`
("fix(ggufload): reject overflowing tensor file offsets (fak ggufload)"), landed and
pushed to `origin/main`. The pre-addition guard
`tensors[i].Offset > uint64(math.MaxInt64)-data` replaces the wrap-prone
`data+tensors[i].Offset > uint64(math.MaxInt64)` at
`internal/ggufload/gguf_config.go:120`, with `TestReadTensorFileOffsetBounds`
covering the wrap, MaxInt64-boundary, and normal-aligned cases. Witness:
`go test ./internal/ggufload -count=1` → ok.

## Witness

```bash
go test ./internal/ggufload -count=1
```

## Process cause

Process cause: verification-gap

Source readback exposes an unguarded invariant; a real-entrypoint regression is required before claiming the repair.

## Acceptance gate

```bash
go test ./internal/ggufload -count=1
```

Acceptance is a future repair gate; this study has not shipped the repair. An unexecuted future test must not be reported green.

## Parent context

OSS refresh campaign GOAL-oss-refresh-20260929. This bounded leaf is grounded by the upstream issue/PR cited above.

## Why this is next

The source-confirmed invariant gap affects trustworthy cache reuse or loading in the public native runtime. Add the smallest independent correctness regression before expanding features.

## Closure binding

Close this native ticket only after its reproduction and acceptance command pass on the resolving signed public commit, with the ticket ID and source attribution recorded.
