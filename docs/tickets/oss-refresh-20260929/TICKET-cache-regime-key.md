<!-- fak-cross-key: oss-refresh-20260929-cache-regime-key -->
# Make cache regime identity injective

```routing
lane: radixkv
paths: ["internal/radixkv/regime.go", "internal/radixkv/regime_test.go"]
expected_steps: 4
```

## Target repository

Public `anthony-chaudhary/fak`; core runtime mechanism. Dispatch through `fak-flow spawn oss-refresh-20260929-cache-regime-key --route-repo public`. The specification belongs to the public runtime backlog.

## Working spine

Read source-confirmed invariant -> reproduce through the public runtime seam -> bounded correction -> independent regression -> durable green acceptance receipt.

## Current state

RegimeKey joins unconstrained strings with delimiters. ModelID=a;sha=b with ModelSHA=c and ModelID=a with ModelSHA=b;sha=c serialize identically. Hashing this ambiguous string preserves the collision.

## Source and license

vLLM proposed PR #51899, head 166608ec098c860e27ab385d0a7450b9b06a238f; vllm/v1/core/kv_cache_utils.py:633. Source: https://github.com/vllm-project/vllm/pull/51899. Observation 2026-09-29; upstream tracker states are dated. Apache-2.0 (vLLM/LMCache) or MIT (llama.cpp), ADAPT with source attribution retained. No source code is copied by this ticket. The cited upstream PR and revision are the source anchor; this is a bounded source assessment.

## Core through-line

Use a versioned injective field encoding, preserve typed Match behavior, and invalidate or migrate persisted namespaces deliberately.

## Gold-plating boundary

No new inference backend, cache rewrite, rollout or performance claim. No hardware qualification is needed for the CPU correctness invariant. Keep the repair bounded to this outcome and its consuming call path.

## File:Line seam

- `internal/radixkv/regime.go:92` at public baseline `365167aa5c8b5f3252221594eb5cf2a38379f16c`.

## Dedupe and routing

Dedupe key `oss-refresh-20260929-cache-regime-key`. Both-repo capability queries and nearby issues are in the cohort note. The initial 56-contract native census contains no contract with this same outcome. Related tickets remain separate owners; refresh before claim because peer backlog can advance.

## Done condition / witness

All unchecked criteria below must be satisfied through the real consuming entrypoint; preserve correct existing behavior and reject the adversarial case.

## Definition of done

- [x] Reproduce the current gap through the real target entrypoint.
- [x] Implement the bounded change and independent edge-case regression.
- [x] Two complete, unequal adversarial regimes produce distinct keys and hashes; normal stable identities remain deterministic; cache insertion and lookup consume the same versioned identity.
- [x] Verify `go test ./internal/radixkv -count=1`, public leak/boundary checks, and commit/land through the public native workflow.

Resolved by commit `e9540090fcfc42b12bc3e224239f272d0d1444c4`
("fix(radixkv): make decode regime keys injective (fak radixkv)"), landed and
pushed to `origin/main`. `Regime.RegimeKey` now emits a versioned `v2;` key whose
string-valued axes (`model`, `sha`, `dtype`, `quant`, `rope.type`) are each
`strconv.Quote`d, so the formerly aliasing fixtures `ModelID="a;sha=b"`,
`ModelSHA="c"` and `ModelID="a"`, `ModelSHA="b;sha=c"` produce distinct keys and
hashes. `TestRegime_DelimiterAdversarialIdentityDoesNotAliasCache`
(`internal/radixkv/regime_test.go:68`) reproduces the collision and then admits
the left regime through `ScopedTree.AdmitPrivateRegime` and looks it up under the
right regime, asserting a clean miss — the same versioned identity is consumed at
both insertion and lookup. Witness: `go test ./internal/radixkv -count=1` → ok.

## Witness

```bash
go test ./internal/radixkv -count=1
```

## Process cause

Process cause: verification-gap

Source readback exposes an unguarded invariant; a real-entrypoint regression is required before claiming the repair.

## Acceptance gate

```bash
go test ./internal/radixkv -count=1
```

Acceptance is a future repair gate; this study has not shipped the repair. An unexecuted future test must not be reported green.

## Parent context

OSS refresh campaign GOAL-oss-refresh-20260929. This bounded leaf is grounded by the upstream issue/PR cited above.

## Why this is next

The source-confirmed invariant gap affects trustworthy cache reuse or loading in the public native runtime. Add the smallest independent correctness regression before expanding features.

## Closure binding

Close this native ticket only after its reproduction and acceptance command pass on the resolving signed public commit, with the ticket ID and source attribution recorded.
