package vcachechain

import (
	"reflect"
	"testing"
)

func TestAppendCorrectionPreservesBaseAndPriorVersions(t *testing.T) {
	base := []FactRecord{
		{Key: "owner", Value: "alice"},
		{Key: "region", Value: "west"},
	}
	chain := NewCorrectionChain(base)

	firstInput := CorrectionSegment{
		ID:      "correction-1",
		ByteLen: 10,
		Facts:   []FactRecord{{Key: "owner", Value: "bob"}},
	}
	first := chain.AppendCorrection(firstInput)

	// Neither caller-owned input may alias the immutable chain.
	base[0].Value = "tampered-base"
	firstInput.Facts[0].Value = "tampered-correction"
	if got := first.BaseFacts(); got[0].Value != "alice" {
		t.Fatalf("base changed through caller slice: %+v", got)
	}
	if got := first.Corrections(); got[0].Facts[0].Value != "bob" {
		t.Fatalf("correction changed through caller slice: %+v", got)
	}

	second := first.AppendCorrection(CorrectionSegment{
		ID:      "correction-2",
		ByteLen: 12,
		Facts:   []FactRecord{{Key: "region", Value: "east"}},
	})
	if got := len(chain.Corrections()); got != 0 {
		t.Fatalf("original chain gained %d corrections, want 0", got)
	}
	if got := len(first.Corrections()); got != 1 {
		t.Fatalf("prior chain gained %d corrections, want 1", got)
	}
	if got := len(second.Corrections()); got != 2 {
		t.Fatalf("new chain has %d corrections, want 2", got)
	}
	if got := second.BaseFacts(); !reflect.DeepEqual(got, []FactRecord{
		{Key: "owner", Value: "alice"},
		{Key: "region", Value: "west"},
	}) {
		t.Fatalf("append rewrote base facts: %+v", got)
	}

	// Accessors are snapshots too; mutating one must not alter the chain.
	corrections := second.Corrections()
	corrections[0].Facts[0].Value = "tampered-snapshot"
	if got := second.Corrections()[0].Facts[0].Value; got != "bob" {
		t.Fatalf("Corrections returned aliased facts: got %q, want bob", got)
	}
}

func TestEffectiveFactsUsesNewestCorrectionAndStableKeyOrder(t *testing.T) {
	chain := NewCorrectionChain([]FactRecord{
		{Key: "tier", Value: "standard"},
		{Key: "owner", Value: "alice"},
		{Key: "region", Value: "west"},
	})
	chain = chain.AppendCorrection(CorrectionSegment{
		ID: "correction-1",
		Facts: []FactRecord{
			{Key: "owner", Value: "bob"},
			{Key: "tier", Value: "gold"},
		},
	})
	chain = chain.AppendCorrection(CorrectionSegment{
		ID: "correction-2",
		Facts: []FactRecord{
			{Key: "owner", Value: "carol"},
			{Key: "status", Value: "active"},
		},
	})

	want := []FactRecord{
		{Key: "owner", Value: "carol"},
		{Key: "region", Value: "west"},
		{Key: "status", Value: "active"},
		{Key: "tier", Value: "gold"},
	}
	if got := chain.EffectiveFacts(); !reflect.DeepEqual(got, want) {
		t.Fatalf("EffectiveFacts = %+v, want %+v", got, want)
	}
	if got := chain.EffectiveFacts(); !reflect.DeepEqual(got, want) {
		t.Fatalf("second EffectiveFacts call is not deterministic: %+v", got)
	}

	effective := chain.EffectiveFacts()
	effective[0].Value = "tampered-result"
	if got := chain.EffectiveFacts()[0].Value; got != "carol" {
		t.Fatalf("effective result aliases chain state: got %q, want carol", got)
	}
}

func TestRefreshEvaluationTripsAtConfiguredCountOrByteLimit(t *testing.T) {
	chain := NewCorrectionChain(nil).
		AppendCorrection(CorrectionSegment{ID: "correction-1", ByteLen: 4}).
		AppendCorrection(CorrectionSegment{ID: "correction-2", ByteLen: 6})

	tests := []struct {
		name   string
		policy RefreshPolicy
		action RefreshAction
		reason RefreshReason
	}{
		{
			name:   "below both limits",
			policy: RefreshPolicy{MaxCorrectionCount: 3, MaxCorrectionBytes: 11},
			action: RefreshKeepBase,
			reason: RefreshWithinBudget,
		},
		{
			name:   "count limit reached",
			policy: RefreshPolicy{MaxCorrectionCount: 2, MaxCorrectionBytes: 11},
			action: RefreshBase,
			reason: RefreshCorrectionCountLimit,
		},
		{
			name:   "byte limit reached",
			policy: RefreshPolicy{MaxCorrectionCount: 3, MaxCorrectionBytes: 10},
			action: RefreshBase,
			reason: RefreshCorrectionByteLimit,
		},
		{
			name:   "both limits reached",
			policy: RefreshPolicy{MaxCorrectionCount: 2, MaxCorrectionBytes: 10},
			action: RefreshBase,
			reason: RefreshCorrectionCountAndByteLimits,
		},
		{
			name:   "unconfigured limits stay open",
			policy: RefreshPolicy{},
			action: RefreshKeepBase,
			reason: RefreshWithinBudget,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := chain.EvaluateRefresh(tt.policy)
			if got.Action != tt.action || got.Reason != tt.reason {
				t.Fatalf("EvaluateRefresh = %+v, want action=%q reason=%q", got, tt.action, tt.reason)
			}
			if got.CorrectionCount != 2 || got.CorrectionBytes != 10 {
				t.Fatalf("EvaluateRefresh totals = %d/%d, want count=2 bytes=10", got.CorrectionCount, got.CorrectionBytes)
			}
			if got.NeedsRefresh() != (tt.action == RefreshBase) {
				t.Fatalf("NeedsRefresh = %v for action %q", got.NeedsRefresh(), got.Action)
			}
		})
	}
}
