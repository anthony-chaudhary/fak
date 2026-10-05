package cachemeta

import "testing"

// TestStoreAdmissible pins the exact allow/deny table the lazy-offload store
// admission gate must enforce: a fully identified, continuous-prefix append is
// allowed; a missing span digest or broken/unknown prefix, and an orphaned batch
// without a successful terminal receipt, are refused with distinct stable reasons.
func TestStoreAdmissible(t *testing.T) {
	identified := StoreAdmission{
		CoveredBlocks:    []string{"blk-0", "blk-1"},
		PrefixContinuity: PrefixContinuityKnown,
		Batch:            BatchActive,
	}

	cases := []struct {
		name string
		ev   LMCTransferEvent
		want StoreRefusal
	}{
		{
			name: "nil admission preserves legacy mapping",
			ev:   LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s"},
			want: StoreAllowed,
		},
		{
			name: "fully identified continuous prefix allowed",
			ev:   LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &identified},
			want: StoreAllowed,
		},
		{
			name: "abort alone does not discard a valid buffered append",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityKnown,
				Batch: BatchActive, Aborted: true,
			}},
			want: StoreAllowed,
		},
		{
			name: "missing span digest refused",
			ev: LMCTransferEvent{Kind: LMCAppend, StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityKnown,
			}},
			want: StoreRefusedUnidentified,
		},
		{
			name: "no covered blocks refused",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				PrefixContinuity: PrefixContinuityKnown,
			}},
			want: StoreRefusedUnidentified,
		},
		{
			name: "unasserted prefix refused",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"},
			}},
			want: StoreRefusedPrefixUnknown,
		},
		{
			name: "explicitly broken prefix refused",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityBroken,
			}},
			want: StoreRefusedPrefixBroken,
		},
		{
			name: "orphaned without receipt refused",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityKnown,
				Batch: BatchOrphaned,
			}},
			want: StoreRefusedOrphanNoReceipt,
		},
		{
			name: "orphaned with pending receipt refused",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityKnown,
				Batch: BatchOrphaned, Receipt: ReceiptPending,
			}},
			want: StoreRefusedReceiptPending,
		},
		{
			name: "orphaned with failed receipt refused",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityKnown,
				Batch: BatchOrphaned, Receipt: ReceiptFailed,
			}},
			want: StoreRefusedReceiptFailed,
		},
		{
			name: "orphaned with successful terminal receipt may drain",
			ev: LMCTransferEvent{Kind: LMCAppend, SpanDigest: "s", StoreAdmission: &StoreAdmission{
				CoveredBlocks: []string{"blk-0"}, PrefixContinuity: PrefixContinuityKnown,
				Batch: BatchOrphaned, Receipt: ReceiptSuccess,
			}},
			want: StoreAllowed,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := StoreAdmissible(c.ev); got != c.want {
				t.Fatalf("StoreAdmissible = %q, want %q", got, c.want)
			}
		})
	}
}

// TestStoreRefusalReasonsDistinct witnesses the structural requirement that every
// deny path carries its own closed reason value (never a shared catch-all).
func TestStoreRefusalReasonsDistinct(t *testing.T) {
	refusals := []StoreRefusal{
		StoreRefusedUnidentified,
		StoreRefusedPrefixUnknown,
		StoreRefusedPrefixBroken,
		StoreRefusedOrphanNoReceipt,
		StoreRefusedReceiptPending,
		StoreRefusedReceiptFailed,
	}
	seen := map[StoreRefusal]bool{}
	for _, r := range refusals {
		if r == StoreAllowed {
			t.Fatalf("refusal token is the allowed zero value")
		}
		if seen[r] {
			t.Fatalf("duplicate refusal token %q", r)
		}
		seen[r] = true
		if r.Reason() == ReasonNone {
			t.Fatalf("refusal %q maps to no LookupReason", r)
		}
	}
}

// TestLMCTransferVerdictStoreAdmission witnesses the append wiring: an unsafe
// append carrying admission facts becomes a typed MISS with the decision's
// reason, while a safe append still serves and a no-facts append keeps the
// legacy mapping exactly.
func TestLMCTransferVerdictStoreAdmission(t *testing.T) {
	unsafe := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCAppend, SpanDigest: "s", ToTier: TierDRAM, Outcome: KVTransferOK,
		StoreAdmission: &StoreAdmission{
			PrefixContinuity: PrefixContinuityBroken, CoveredBlocks: []string{"blk-0"},
		},
	})
	if unsafe.Kind != LookupMiss || unsafe.Reason != ReasonPrefixBroken {
		t.Fatalf("unsafe append: got %s/%s, want miss/prefix_discontinuity", unsafe.Kind, unsafe.Reason)
	}
	if unsafe.CanServe() {
		t.Fatalf("unsafe append served")
	}

	safe := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCAppend, SpanDigest: "s", ToTier: TierDRAM, Outcome: KVTransferOK,
		StoreAdmission: &StoreAdmission{
			PrefixContinuity: PrefixContinuityKnown, CoveredBlocks: []string{"blk-0"},
		},
	})
	if !safe.CanServe() {
		t.Fatalf("safe append: CanServe=false, verdict=%+v", safe)
	}

	// Lookup semantics are unchanged: a lookup never consults store admission even
	// when the (append-shaped) facts are present.
	lookup := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCLookup, SpanDigest: "s", ToTier: TierHBM, Outcome: KVTransferOK,
		StoreAdmission: &StoreAdmission{PrefixContinuity: PrefixContinuityBroken},
	})
	if !lookup.CanServe() {
		t.Fatalf("lookup consulted store admission: %+v", lookup)
	}

	// An unidentified append with no admission facts is still refused by the
	// pre-existing empty-digest rule.
	unnamed := LMCTransferVerdict(LMCTransferEvent{Kind: LMCAppend, Outcome: KVTransferOK})
	if unnamed.Kind != LookupMiss || unnamed.Reason != ReasonAbsent {
		t.Fatalf("unnamed append: got %s/%s, want miss/absent", unnamed.Kind, unnamed.Reason)
	}
}
