package agent

import "testing"

// TestCompactReceiptReconcilesSessionShed is the load-bearing #2787 acceptance: a session's
// compaction attempts each produce exactly one receipt, and the sum of the per-fire receipt shed
// reconciles the session's aggregate shed total (the gateway's AdjudicationSummary.CompactionShedTokens,
// which observeCompaction accumulates only on a FIRE). Bailed attempts still leave a receipt — with
// their reason and zero shed — so silence is recorded, not read as success.
func TestCompactReceiptReconcilesSessionShed(t *testing.T) {
	// One session's compaction attempts, as the byte-splice returned them: three fires and two bails.
	outcomes := []CompactOutcome{
		{Reason: CompactReasonNone, Dropped: 3, ShedTokens: 1200},
		{Reason: CompactReasonUnderBudget}, // bail — nothing shed
		{Reason: CompactReasonNone, Dropped: 2, ShedTokens: 800},
		{Reason: CompactReasonCachedSpan}, // bail — would burst a cached span
		{Reason: CompactReasonNone, Dropped: 5, ShedTokens: 2000},
	}

	var receipts []CompactReceipt
	totalShed := 0 // the gateway aggregate: shed accumulates ONLY on a fire (observeCompaction)
	for _, o := range outcomes {
		receipts = append(receipts, NewCompactReceipt(o))
		if o.Reason == CompactReasonNone {
			totalShed += o.ShedTokens
		}
	}

	// Exactly one receipt per attempt (fire OR bail) — no fire folded away into the aggregate.
	if len(receipts) != len(outcomes) {
		t.Fatalf("receipts=%d, want exactly one per attempt (%d)", len(receipts), len(outcomes))
	}
	// The invariant: the decomposed per-fire receipts add back up to the opaque aggregate.
	if !ReconcileShed(receipts, totalShed) {
		t.Fatalf("receipts shed %d must reconcile aggregate shed %d", SumReceiptShed(receipts), totalShed)
	}
	if got := SumReceiptShed(receipts); got != 4000 {
		t.Fatalf("summed receipt shed = %d, want 4000", got)
	}
	// A mismatched total must NOT reconcile (the check has teeth, not always-true).
	if ReconcileShed(receipts, totalShed+1) {
		t.Fatalf("reconciliation must fail when the aggregate disagrees with the receipts")
	}

	// A bail leaves an auditable receipt: not fired, its reason preserved, zero shed.
	bail := receipts[1]
	if bail.Fired || bail.Reason != CompactReasonUnderBudget || bail.ShedTokens != 0 || bail.DroppedTurns != 0 {
		t.Fatalf("bail receipt = %+v, want !fired reason=under_budget shed=0 dropped=0", bail)
	}
	// A fired receipt carries its kept-window boundary and the discharged prefix_mismatch=0 proof.
	fired := receipts[0]
	if !fired.Fired || fired.Reason != CompactReasonNone || fired.DroppedTurns != 3 || fired.ShedTokens != 1200 {
		t.Fatalf("fired receipt = %+v, want fired reason=\"\" dropped=3 shed=1200", fired)
	}
	for i, r := range receipts {
		if r.PrefixMismatch != 0 {
			t.Fatalf("receipt[%d].PrefixMismatch = %d, want 0 (a fire proved the prefix byte-identical; a bail spliced nothing)", i, r.PrefixMismatch)
		}
	}
}

// TestCompactReceiptObservedUsageStamp proves the OBSERVED provider read/creation are zero for a
// byte-level caller and only appear when a gateway caller stamps them — and that stamping cannot
// perturb the WITNESSED shed the reconciliation depends on.
func TestCompactReceiptObservedUsageStamp(t *testing.T) {
	r := NewCompactReceipt(CompactOutcome{Reason: CompactReasonNone, Dropped: 1, ShedTokens: 500})
	if r.ObservedCacheReadTokens != 0 || r.ObservedCacheCreationTokens != 0 {
		t.Fatalf("a byte-level receipt must leave OBSERVED provider usage zero, got %+v", r)
	}
	stamped := r.WithObservedUsage(4096, 128)
	if stamped.ObservedCacheReadTokens != 4096 || stamped.ObservedCacheCreationTokens != 128 {
		t.Fatalf("WithObservedUsage = %+v, want read=4096 creation=128", stamped)
	}
	if stamped.ShedTokens != 500 || stamped.DroppedTurns != 1 {
		t.Fatalf("stamping OBSERVED usage perturbed the WITNESSED fields: %+v", stamped)
	}
	// The original is unchanged (value receiver) — receipts are append-only, not mutated in place.
	if r.ObservedCacheReadTokens != 0 {
		t.Fatalf("WithObservedUsage must not mutate the receiver, got %+v", r)
	}
}

// TestCompactReceiptFromRealFire wires the receipt to a REAL byte-level compaction fire (the
// head-anchored dormant-shape fire from TestCompactHeadAnchorFiresOnDormantShape): the receipt built
// from the live CompactOutcome carries that fire's shed and dropped turns, and a single-fire session
// reconciles against the outcome's own shed — the receipt is not a re-typed copy of hand-written
// numbers but the actual splice's witnessed output.
func TestCompactReceiptFromRealFire(t *testing.T) {
	raw := headOrderedBody(t, 120, 2)
	opts := CompactOptions{Budget: 1200, Anchor: CompactAnchorHead, TotalTurns: 1000, CurrentTurn: 1}
	_, outcome := CompactAnthropicHistoryWithOptions(raw, opts)
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("fixture must FIRE for this test; got reason=%q (%+v)", outcome.Reason, outcome)
	}

	receipt := NewCompactReceipt(outcome)
	if !receipt.Fired || receipt.ShedTokens != outcome.ShedTokens || receipt.DroppedTurns != outcome.Dropped {
		t.Fatalf("receipt %+v must mirror the live outcome shed=%d dropped=%d", receipt, outcome.ShedTokens, outcome.Dropped)
	}
	if receipt.ShedTokens <= 0 || receipt.DroppedTurns <= 0 {
		t.Fatalf("a real fire must yield a positive shed/dropped receipt, got %+v", receipt)
	}
	// A one-fire session's aggregate shed IS this fire's shed, so the receipt reconciles it exactly.
	if !ReconcileShed([]CompactReceipt{receipt}, outcome.ShedTokens) {
		t.Fatalf("single-fire receipt %d must reconcile the outcome shed %d", receipt.ShedTokens, outcome.ShedTokens)
	}
}
