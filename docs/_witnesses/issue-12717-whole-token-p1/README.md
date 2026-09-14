# issue-12717 — whole-token P1 decode closure witness

Closure evidence for `anthony-chaudhary/fak#12717` ("perf(model): amortize Qwen Metal
decode through the whole-token graph").

## Verdict

The requested route is **landed on `origin/main` and default-on**; the issue closes.
The issue body's declared witness test name
(`TestMetalQwen35P32ToP1WholeTokenGraph`) was **stale** — it was written against an
earlier worker draft that was archived to an orphan ref and never landed. The landed
route took a different, strictly better shape and is witnessed by the tests named below.

## Landed commits (all ancestors of `origin/main`)

| commit | what it did |
|---|---|
| `4cc4eb164` | whole-token Qwen3.8 decode graph with P=1 GEMV + per-shape buffer pool |
| `4e3ba8996` | split-KV flash decoding for P=1 Qwen graph attention |
| `8aac1d7c5` | gather Qwen35 prefill embeddings through the packed store |
| `a94371d13` | uncap Qwen ordered-panel attention with online softmax |
| `684ccb9eb` | default the whole-token decode route on (`(fak model)` closure binding) |
| `e32b6ea44` | report the auto-admitted whole-token decode route truthfully |

Verified at `origin/main = 94c8470154ab29d18d96065bd85ccd02f89d53c0`.

## Route shape (why the declared witness name is stale)

The landed route is a dedicated P=1 decoder, `Qwen35MetalDecodeToken`
(`internal/model/metal_prefill_hybrid.go:910`), admitted by
`Session.tryQwen35MetalDecodeWholeToken` (`internal/model/qwen35_decode_block.go:61`)
from `tokenHiddenQ` (`internal/model/quant_forward.go:273-285`). It reuses the existing
whole-model projection graph with P=1 GEMV kernels and the per-shape buffer pool, and
still declines cleanly to the historical `blockStep` per-layer loop when the resident
owner is absent — the exact behavior #12717 asks for.

`Qwen35MetalForwardSequence` (`internal/model/metal_prefill_hybrid.go:723`) intentionally
still requires `len(ids) == 32`; the P=1 continuation is a separate method rather than a
widening of the prefill entry. The archived draft instead widened
`Qwen35MetalForwardSequence` in place and would have **shadowed** the shipped P=1 GEMV
route via a second admission branch in `tokenHiddenQ`; it was deliberately not landed.
The orphan is preserved at `refs/fak/orphaned/fleet-model-468eca59afbe-*` (`02471b9b3`).

## Witnesses (executed on physical Apple M3 Pro, darwin/arm64, macOS 26.6.2, Go 1.26.6)

```
go test ./internal/model -run 'TestQwen35WholeTokenDecodeRecordsExecutedForwardReceipt|TestQwen35DecodeCallerOwnedGraphParityAndSingleSubmission|TestExactQwen38SessionSelectsWholeDecodeBlockAndHooksOnce|TestExactQwen38DecodeBlockPostSubmitFailureIsTerminal' -count=1
# -> ok github.com/anthony-chaudhary/fak/internal/model  78.948s (4/4 PASS)

go test ./internal/model -run 'TestMetalQwen35BackendNilP32WholeSequenceSingleFenceAndDecodeOwner|TestMetalQwen35GDNProductionPrefillContinuityAndDecodeHandoff|TestMetalQwen35BackendNilP32PostSubmitFailureIsTerminal|TestQwen35MetalDecodeBlockFourStepsParityAccountingAndGreedyToken' -count=1
# -> ok github.com/anthony-chaudhary/fak/internal/model  8.750s (4/4 PASS)
#    prefill logits cosine=1.000000000 maxRel=3.29216e-06 greedy=410
#    first decode logits cosine=1.000000000 maxRel=1.60961e-06 greedy=59
```

These cover the issue's Done condition: P32 resident-state handoff into P1, one
command-buffer completion boundary (`CommandBuffers=1, TerminalWaits=1`), zero
intermediate graph readbacks (`IntermediateReadbacks=0`), greedy-token parity, and
fail-closed post-submit behavior without replay.

## Nonblocking remainder

Accelerated/batched Qwen prefill (current 4.9–7.9 tok/s, target >30) is a separate,
larger effort tracked by #11036 under epic #12459. This closure does not claim it.
#12587 and #10193 remain nonblocking coordination.