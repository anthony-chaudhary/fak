package agent

// compact_receipt.go — the per-fire compaction AUDIT receipt (#2787). Parent epic #2783
// (cache-value netting/attribution); relates to #1383 (compaction receipts).
//
// The gateway folds EVERY compaction fire in a session into ONE opaque aggregate
// (AdjudicationSummary.CompactionShedTokens et al., accumulated in metrics_observe.go's
// observeCompaction). An aggregate cannot be decomposed back to the individual fires, so a
// session that FIRED but shed nothing reads identical to one that never fired — silence reads as
// success. This is the append-only per-fire receipt that keeps each fire individually auditable:
// ONE receipt per compaction attempt, carrying the tokens SHED, the kept-window boundary (whole
// middle turns dropped), the prefix_mismatch=0 proof (a FIRED outcome is only returned after
// verifySplicedBody proved the protected prefix bytes byte-identical — a mismatch bails to
// identity and sheds nothing), the OBSERVED downstream provider cache_read / cache_creation the
// fire's turn earned (stamped by a gateway caller; zero for a byte-level caller with no provider
// usage in hand), and the bail REASON. The load-bearing invariant (ReconcileShed): the sum of a
// session's per-fire receipt shed equals the aggregate CompactionShedTokens — the decomposed
// receipts add back up to the opaque aggregate.
//
// This is the PURE primitive, landed in the package that OWNS CompactOutcome, ahead of its durable
// ledger seam — mirroring how internal/rsiloop/firenettune.go (#2817, the sibling per-fire
// NET-SCORE receipt) and internal/rsiloop/forksaving.go landed pure ahead of the leased
// internal/cachevaluereport/** lane. The durable Track-2 row + the live gateway per-fire emission
// land via that lane; this file is the shape + the reconciliation invariant they build on,
// unit-testable with no I/O and no upward import.

// CompactReceipt is one append-only per-fire compaction audit row (#2787): the individual event an
// AdjudicationSummary aggregate folds away. WITNESSED fields (Fired, Reason, ShedTokens,
// DroppedTurns, PrefixMismatch) are derived by construction from the CompactOutcome the byte-splice
// returned; OBSERVED fields (the provider cache_read / cache_creation) are relayed downstream and
// left zero until a gateway caller stamps them (WithObservedUsage) — never treated as fak-witnessed.
// It is the sibling of CompactOutcome (the momentary verdict) made durable and per-event.
type CompactReceipt struct {
	// Fired is true iff the compaction rewrote the body (Reason == CompactReasonNone). A BAILED
	// attempt still gets a receipt — carrying its Reason and zero shed — so the audit trail records
	// WHY a fire did nothing rather than leaving the silence the aggregate cannot tell from success.
	Fired bool `json:"fired"`
	// Reason is the bail reason from the closed CompactReason* vocabulary; "" (CompactReasonNone) on
	// a fire. It is the "silence must not read as success" field the aggregate discards.
	Reason string `json:"reason,omitempty"`
	// ShedTokens is the tokens this fire removed from the outbound body (CompactOutcome.ShedTokens),
	// in the SAME ~4-chars/token currency as the budget and the aggregate CompactionShedTokens the
	// receipts reconcile against — so ReconcileShed compares like with like. Zero on a bail.
	ShedTokens int `json:"shed_tokens"`
	// DroppedTurns is the kept-window boundary: the whole middle turns removed from between the
	// protected prefix and the kept recent window (CompactOutcome.Dropped). It is the window's lower
	// bound — every message after it survived verbatim. Zero on a bail.
	DroppedTurns int `json:"dropped_turns"`
	// PrefixMismatch is the byte-splice cache-safety proof, 0 on every receipt. A FIRED outcome is
	// only returned after compactSpliceVerdict proved verifySplicedBody != spliceVerdictPrefixMismatch
	// (the protected cache prefix bytes are byte-identical to the input); a prefix mismatch bails to
	// identity and sheds nothing, so no receipt can carry a nonzero mismatch. The field records the
	// DISCHARGED proof rather than leaving it implicit — the issue's explicit `prefix_mismatch=0`.
	PrefixMismatch int `json:"prefix_mismatch"`
	// ObservedCacheReadTokens / ObservedCacheCreationTokens are the OBSERVED provider cache_read /
	// cache_creation the fire's turn earned downstream — provider-relayed, never WITNESSED by fak.
	// They are zero for a byte-level caller with no provider usage; a gateway caller stamps them via
	// WithObservedUsage. Kept on the receipt so a reader can put a fire's WITNESSED shed beside the
	// provider read it actually unlocked without joining a second ledger.
	ObservedCacheReadTokens     uint64 `json:"observed_cache_read_tokens,omitempty"`
	ObservedCacheCreationTokens uint64 `json:"observed_cache_creation_tokens,omitempty"`
	InducedCacheCreationTokens  int    `json:"induced_cache_creation_tokens,omitempty"`
	// JoinKey is the event-join coordinate (#2788) a fire shares with the provider usage record
	// for the turn it affected — the (turn sequence, monotonic ts) pair that makes the receipt's
	// WITNESSED shed correlatable 1:1 with the SAME turn's OBSERVED provider cache_read /
	// cache_creation AFTER the fact, across two independently collected streams. It is stamped by a
	// caller that holds the turn coordinate (the gateway, via WithJoinKey) and left ZERO (unstamped)
	// by a byte-level caller with no turn context — an unstamped key is the honest state of a
	// byte-level receipt, never a failed join. Like the OBSERVED fields it is metadata for the
	// join, never WITNESSED by fak, so it cannot perturb the ReconcileShed invariant.
	JoinKey CompactJoinKey `json:"join_key,omitempty"`
}

// NewCompactReceipt builds the per-fire audit receipt from the CompactOutcome one compaction attempt
// returned. The WITNESSED fields (shed, dropped turns, prefix_mismatch=0, bail reason) are all
// derivable from the outcome; the OBSERVED provider read/creation are left zero for a gateway caller
// to stamp with WithObservedUsage. Called once per attempt (fire OR bail) so exactly one receipt
// exists per fire — the "each fire produces exactly one receipt" half of the acceptance.
func NewCompactReceipt(out CompactOutcome) CompactReceipt {
	return CompactReceipt{
		Fired:                      out.Reason == CompactReasonNone,
		Reason:                     out.Reason,
		ShedTokens:                 nonNegInt(out.ShedTokens),
		DroppedTurns:               nonNegInt(out.Dropped),
		InducedCacheCreationTokens: nonNegInt(out.InducedCacheCreationTokens),
		// A fired outcome discharged the prefix-mismatch proof by construction (see the field doc);
		// a bail carries no splice at all. Either way the receipt records a clean prefix.
		PrefixMismatch: 0,
	}
}

// WithObservedUsage stamps the OBSERVED downstream provider cache_read / cache_creation this fire's
// turn earned onto the receipt, returning the updated copy. A gateway caller with the provider usage
// block in hand calls this; a byte-level caller leaves the fields zero. It never touches the
// WITNESSED shed, so it cannot perturb the ReconcileShed invariant. OBSERVED, never WITNESSED.
func (r CompactReceipt) WithObservedUsage(cacheRead, cacheCreation uint64) CompactReceipt {
	r.ObservedCacheReadTokens = cacheRead
	r.ObservedCacheCreationTokens = cacheCreation
	return r
}

// SumReceiptShed totals the shed across per-fire receipts — the left side of the ReconcileShed
// invariant, exposed so a caller can report the reconciled total alongside the boolean verdict.
// Bailed receipts contribute 0 (they shed nothing), so the sum is over the fires alone.
func SumReceiptShed(receipts []CompactReceipt) int {
	sum := 0
	for _, r := range receipts {
		sum += r.ShedTokens
	}
	return sum
}

// ReconcileShed reports whether the per-fire receipts add back up to a session's aggregate shed
// total — the load-bearing #2787 invariant that the decomposed receipts reconcile the opaque
// aggregate. totalShed is the AdjudicationSummary.CompactionShedTokens the gateway folded
// (observeCompaction accumulates ShedTokens only on a FIRE, so the aggregate is the sum of the
// fires' shed). Equal ⇒ every shed token is attributable to exactly one fire; unequal ⇒ a fire's
// shed went unreceipted or was double-counted. Using the SAME ~4-chars/token currency on both sides
// is what keeps the check from spuriously failing (the confusion risk the issue names).
func ReconcileShed(receipts []CompactReceipt, totalShed int) bool {
	return SumReceiptShed(receipts) == totalShed
}

// nonNegInt floors a count at zero so a zero-valued CompactOutcome (a bail) or any defensive
// negative can never make a receipt's shed/dropped go below zero and corrupt the reconciliation.
func nonNegInt(n int) int {
	if n < 0 {
		return 0
	}
	return n
}

func SumReceiptInducedCreation(receipts []CompactReceipt) int {
	total := 0
	for _, r := range receipts {
		if r.Fired && r.InducedCacheCreationTokens > 0 {
			total += r.InducedCacheCreationTokens
		}
	}
	return total
}

type InducedCreationReconciliation struct {
	Fires                  int
	ReconciledFires        int
	InducedTokens          uint64
	ObservedCreationTokens uint64
	DebitTokens            uint64
	WithinObserved         bool
	AttributedFraction     float64
}

func (r InducedCreationReconciliation) Reconciled() bool {
	return r.Fires > 0 && r.ReconciledFires == r.Fires && r.WithinObserved
}
func ReconcileInducedCreation(receipts []CompactReceipt) InducedCreationReconciliation {
	var out InducedCreationReconciliation
	for _, r := range receipts {
		if !r.Fired {
			continue
		}
		out.Fires++
		if r.ObservedCacheCreationTokens == 0 {
			continue
		}
		out.ReconciledFires++
		if r.InducedCacheCreationTokens > 0 {
			out.InducedTokens += uint64(r.InducedCacheCreationTokens)
		}
		out.ObservedCreationTokens += r.ObservedCacheCreationTokens
	}
	if out.ReconciledFires == 0 {
		return out
	}
	out.WithinObserved = out.InducedTokens <= out.ObservedCreationTokens
	out.DebitTokens = out.InducedTokens
	if out.DebitTokens > out.ObservedCreationTokens {
		out.DebitTokens = out.ObservedCreationTokens
	}
	if out.ObservedCreationTokens > 0 {
		out.AttributedFraction = float64(out.DebitTokens) / float64(out.ObservedCreationTokens)
	}
	return out
}
