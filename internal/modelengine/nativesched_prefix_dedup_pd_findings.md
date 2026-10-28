# Clause C findings: in-batch prefix dedup on the P/D and pipeline admission paths

Status: **ABSTAIN**. The P/D / imported-sequence clause of the in-batch prefix dedup
goal (fak#1914 remainder, fak#13661) cannot share the synchronous coalescing seam.
Each of the two named paths needs its own seam, which is a separate feature, not this
issue's remainder. No code is changed in this leaf.

## What the clause asks

fak#13661 acceptance item 2: "PD imported-sequence admission shows the same
coalescing gain, or a documented refusal reason why it cannot share the synchronous
seam." The issue names two entry points: `NativeScheduler.AdmitImported`
(`native_pd.go`) and `PipelineEngine.Admit` (`pipeline.go`).

## Why `NativePDCluster.Admit` cannot share the synchronous seam

1. **The prefill runs inline on a scheduler-less role.** `NativePDCluster.Admit`
   creates a fresh host session and calls `prefillSess.Prefill(prompt)` directly
   (`native_pd.go:358-359`). The prefill role owns no `NativeScheduler`:
   `newNativePDWorker` assigns `w.sched` only to the decode role
   (`native_pd.go:183-185`). The synchronous seam
   (`NativeScheduler.prefillCoalesced`, gated by `inBatchDedupEligible` under
   `s.mu`, `nativesched_prefix_dedup.go:76-86`) exists only inside a scheduler's
   admit path, so there is no scheduler admission moment for the P/D prefill to
   join.
2. **Reusing `PrefixFlightGroup` here means owning a second flight group on the
   cluster** with its own eligibility, ownership, and KV-budget policy — the
   issue's own framing of "needs its own seam", not sharing the synchronous one.
   It would also add a new opt-in surface to the P/D hot path, which acceptance
   item 3 keeps frozen ("The synchronous-path default, semantics, and
   `InBatchPrefixDedupStats` contract are unchanged").

## Why `AdmitImported` has nothing to coalesce

By contract, an `ImportedSequence` arrives already prefilled: `Session` holds the
prompt KV and `Logits` are the prefill logits for the first generated token
(`native_pd.go:42-54`); `AdmitImported` refuses lanes without resident KV or
logits (`native_pd.go:63-68`) and enqueues them straight into the WAITING queue
for decode. The decode half of P/D never observes a cold prompt — that is the
point of the split — so admission-time coalescing has no work to do there.

## Why `PipelineEngine.Admit` cannot share it either

`PipelineEngine` is a stateless network pipeline-parallel head: each admitted
request re-forwards the entire growing sequence through
`model.RunPipelineAcrossWorkers` once per generated token
(`pipeline.go:7-12`, `pipeline.go:128-143`). No `model.Session` or `KVCache`
exists inside the engine for the flight group to hand off — the
`PrefixFlightGroup` payload is a `*model.KVCache` — and the conservative
full-re-forward design is documented as deliberate. Sharing prefixes across
concurrent PP requests would require an inter-worker prefix-sharing protocol:
again its own seam, a separate feature.

## Consequence

The landed synchronous seam (fak#1914 slice) and the aligned chunked checkpoint
reuse (fak#14052) remain the only coalescing surfaces, and both are witnessed
green on trunk (Witness below). The P/D and pipeline paths keep their historical
behavior byte-for-byte. If a future owner wants P/D prefill coalescing, the named
seam is a cluster-owned `PrefixFlightGroup` around `NativePDCluster.Admit`'s
inline prefill call, with its own opt-in knob, a device-session fail-open gate,
and a transfer-path test — file it separately.

## Witness

Run from the public repo at base `213c388555e21363c690b6600b996fedae2fef5e`,
2026-10-27:

```text
go test ./internal/modelengine -run 'TestChunkedPrefillStrictExtensionReusesCompletedPrefix|TestChunkedPrefillDivergentPromptsReuseLastAlignedPrefix' -count=1
go test ./internal/modelengine -run 'TestNativeSchedulerInBatchPrefixDedup|TestNativeSchedulerCanceledPrefixFollowerDoesNotPrefillOrPublish|TestNativeSchedulerPrefixLeaderFailureStillAllowsLiveFollower|TestNativeSchedulerCanceledAdmissionPublicationBoundaries|TestInBatchPrefixDedupReachable|TestChunkedPrefillExtensionReclaimsPressureBeforeNextChunk|TestChunkedPrefillExtensionLeavesHeadroomForEarlierLongFollower' -count=1
```

First command: PASS — the shared prefix's chunks run once (three donor plus one
suffix chunk; four donor plus one suffix chunk in the divergent case) and the
follower's terminal logits are at parity with an unfused control prefill.
Second command: PASS — the synchronous-path default, semantics, and
`InBatchPrefixDedupStats` contract are unchanged.
