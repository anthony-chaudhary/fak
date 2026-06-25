package engine

// capacity_adapter.go — Plank 4 of the hardware-capacity bridge (issue #708): the
// engine adapter that EXECUTES a cachemeta.PlanPlacement demote/spill against the
// kernel-owned KV cache.
//
// internal/cachemeta is the payload-free POLICY plane: PlanPlacement decides WHERE a
// cached span should move (demote one tier colder, spill to disk, evict) and emits a
// KVTransfer directive, but it touches no bytes. internal/compute is the PHYSICAL plane:
// the device allocators that actually hold the memory and panic when it is full. The two
// met only at the METER — PlanPlacement's KVOffload/KVRestore directives flowed into
// CacheEventRecorder as fak_engine_cache_* metrics, which made a placement decision
// LEGIBLE but did not make it HAPPEN (docs/explainers/hardware-limits-and-capacity.md
// §2–4). This adapter is the missing CONTROL path between them: it turns a demote/spill
// DECISION into a real stage-to-the-colder-tier (abi.KVBackend.StageSpan) PLUS an
// eviction from the live KV tier (abi.KVBackend.Evict, the proven re-RoPE/renumber
// primitive kvmmu already enforces a quarantine through), and it records the transition
// through the same CacheEventRecorder so it lands in the SAME cache-entry stream as
// tool/context entries — a staging fault is a typed FAULT, never a silent recompute.
//
// Fail-safe ordering. The span is STAGED to the colder tier before it is EVICTED from
// the live one, so a staging fault cannot lose the span: on a typed MISS/FAULT the live
// copy is retained and the fault is recorded, exactly as a failed restore is. The
// in-process KVBackend default answers StageSpan as a no-op OK (the span is already
// resident locally), so against the default backend a demote reclaims room in the live
// tier by dropping the span — its identity survives on the digest for a later
// RestoreSpan; a remote / disaggregated KV backend overrides StageSpan to serialize the
// fak-owned pre-RoPE rows off-box first, so the demote survives disaggregated.
//
// Honest fence (matching the explainer's). This adapter executes the move against the
// kernel-owned cache; it does not yet compute live TierPressure from a real device (that
// is Plank 3) nor turn an OOM panic into a typed value (Plank 2). It is the control path
// Plank 4 names; a capacity-pressure loop that drives it from real device state is the
// remaining plank.

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// CapacityAdapter is the engine adapter that executes a cachemeta placement decision
// against the kernel-owned KV cache. It drives the demote/spill/evict family — the
// actions that DROP a span from a hot tier — through the live cache (KV) and records
// each transition through the cache-event stream (Recorder). PlanPlacement decided the
// move; this adapter MAKES the move. A nil Recorder is allowed: the move still executes,
// it is simply not folded into the metric stream.
type CapacityAdapter struct {
	KV       abi.KVBackend
	Recorder *CacheEventRecorder
}

// PlacementMove bundles a cachemeta.PlacementDecision with the identity of the span it
// moves. The decision names the TIERS and the ACTION; the adapter needs the SPAN — its
// digest, its [From,From+N) range in the live cache, and the model/tokenizer/position
// mode it was derived under — to act on it. It is the field-only lowering shape a
// capacity-pressure loop builds per pressured entry before handing it to Execute.
type PlacementMove struct {
	Decision     cachemeta.PlacementDecision
	SpanDigest   string
	From         int
	N            int
	ModelID      string
	TokenizerID  string
	PositionMode cachemeta.PositionMode
	Owner        string
	Lease        string
}

// PlacementResult is the typed outcome of executing a PlacementMove. Recorded carries
// the normalized cache-entry and its lookup verdict — so a staging FAULT is observable
// (never a silent recompute). Evicted is the number of positions removed from the live
// KV cache. Applied reports whether the live span was actually moved/dropped: true for a
// completed demote/spill/evict; false when the move was not executed (promote/keep, or a
// demote/spill whose staging faulted and whose live copy was therefore retained).
type PlacementResult struct {
	Recorded CacheEventResult
	Evicted  int
	Applied  bool
}

// Execute carries out a placement move against the live KV cache. For the demote/spill
// family it:
//  1. STAGES the span to the colder tier (KV.StageSpan), addressed by digest so the
//     disaggregated direction survives. The live copy is dropped only on a confirmed OK
//     — a typed MISS/FAULT retains the span and records the fault (fail-safe).
//  2. EVICTS the span from the live KV tier (KV.Evict, the re-RoPE/renumber primitive),
//     reclaiming room in the hot tier.
//  3. RECORDS the offload as a typed CacheEvent so the move enters the cache-entry
//     stream — a staging fault is a FAULT(residency_fault), never silent.
//
// An evict (no colder tier had room) skips staging and drops the span outright — it is
// the recompute-on-demand path. A promote (KVRestore) is the reverse direction and is
// NOT executed here; Execute returns Applied=false so a caller routes it to the restore
// path. A keep is a no-op. A nil KV backend is a typed error rather than a nil-deref.
func (a *CapacityAdapter) Execute(ctx context.Context, mv PlacementMove) (PlacementResult, error) {
	if a == nil || a.KV == nil {
		return PlacementResult{}, fmt.Errorf("engine: capacity adapter has no live KV backend")
	}
	d := mv.Decision
	switch d.Action {
	case cachemeta.ActionKeep, cachemeta.ActionPromote:
		// Not this adapter's control path: keep is a no-op; promote is a page-IN
		// (RestoreSpan), the reverse direction. Surface that it was not applied.
		return PlacementResult{Applied: false}, nil
	case cachemeta.ActionDemote, cachemeta.ActionSpill, cachemeta.ActionCompressDemote, cachemeta.ActionEvict:
		// The drop-from-hot-tier family — executed below. A compress-demote stages a
		// lossy COMPRESSED span to the colder tier (its smaller EstMoveBytes) before the
		// live exact copy is evicted, exactly the fail-safe stage-then-evict ordering a
		// demote/spill uses.
	default:
		return PlacementResult{Applied: false}, fmt.Errorf("engine: unknown placement action %q", d.Action)
	}

	outcome := cachemeta.KVTransferOK
	bytesMoved := d.EstMoveBytes
	faultReason := ""

	// (1) Stage a demote/spill to the colder tier BEFORE evicting the live copy, so a
	// staging fault cannot lose the span. An evict skips staging (recompute on demand).
	if d.Action == cachemeta.ActionDemote || d.Action == cachemeta.ActionSpill ||
		d.Action == cachemeta.ActionCompressDemote {
		staged, err := a.KV.StageSpan(ctx, mv.SpanDigest, mv.From, mv.N)
		if err != nil {
			// A transport error is a typed FAULT: retain the live span, record, never silent.
			return a.record(mv, d, cachemeta.KVTransferFault, bytesMoved, err.Error(), 0, false), nil
		}
		switch staged.Outcome {
		case abi.KVResidencyOK:
			if staged.BytesMoved > 0 {
				bytesMoved = staged.BytesMoved // the real backend byte count wins over the estimate
			}
		case abi.KVResidencyFault:
			// Retain the live span; record the fault — never a silent recompute.
			return a.record(mv, d, cachemeta.KVTransferFault, bytesMoved, staged.Reason, 0, false), nil
		default:
			// A staging MISS / unset outcome is fail-closed: the colder tier declined the
			// span, so the live copy MUST be retained.
			reason := "stage " + staged.Outcome.String() + ": colder tier declined the span"
			return a.record(mv, d, cachemeta.KVTransferFault, bytesMoved, reason, 0, false), nil
		}
	}

	// (2) Evict the span from the live KV cache to reclaim room in the hot tier.
	evicted := a.KV.Evict(mv.From, mv.N)

	// (3) Record the offload as a typed event in the shared cache-entry stream.
	return a.record(mv, d, outcome, bytesMoved, faultReason, evicted, true), nil
}

// record lowers the executed move into a typed CacheEvent (the same stream the
// offload/restore path feeds), records it when a Recorder is set, and folds the verdict
// plus positions-evicted into a PlacementResult. applied is whether the live span was
// actually dropped (true only on a completed stage+evict or a bare evict).
func (a *CapacityAdapter) record(mv PlacementMove, d cachemeta.PlacementDecision, outcome cachemeta.KVTransferOutcome, bytesMoved int64, faultReason string, evicted int, applied bool) PlacementResult {
	res := PlacementResult{Evicted: evicted, Applied: applied}
	if a.Recorder == nil {
		return res
	}
	res.Recorded = a.Recorder.Record(CacheEvent{
		Direction:    cachemeta.KVOffload,
		SpanDigest:   mv.SpanDigest,
		Tokens:       int64(mv.N),
		ModelID:      mv.ModelID,
		TokenizerID:  mv.TokenizerID,
		PositionMode: mv.PositionMode,
		FromTier:     d.FromTier,
		ToTier:       d.ToTier,
		Owner:        mv.Owner,
		Lease:        mv.Lease,
		Outcome:      outcome,
		FaultReason:  faultReason,
		BytesMoved:   bytesMoved,
	})
	return res
}
