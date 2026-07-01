package memq

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memview"
)

// provenance_test.go — issue #1599: memq's half of the provenance-timeline view.
// (The issue's own witness command targets internal/memview and internal/recall; this
// file is scoped regression coverage for the memq-side translator these packages
// depend on being correct.)

// TestPromotionLedgerTimelineOrdersAndTranslates proves PromotionLedger.Timeline
// renders a cell's FULL promotion history (not just Latest) as a chronologically
// ordered memview.Timeline, with every structured field carried through verbatim —
// no model narration, no lossy summarization.
func TestPromotionLedgerTimelineOrdersAndTranslates(t *testing.T) {
	m := NewMemStore()
	cell := m.AddPromoted("user", "user", DurabilityBounded,
		[]byte("meeting moved to 3pm"), false,
		PromotionMeta{Consent: ConsentExplicit, Producer: "user", Reason: "user stated a scheduling fact", Expiry: "tick:100"})

	// Re-promote/reclassify the SAME cell to durable — the ledger keeps both records
	// (For's own doc comment: "a cell may be re-promoted/reclassified over its life").
	m.Promotions().Record(PromotionRecord{
		CellID:     cell.ID,
		SourceSpan: memview_testSpan(cell.Step, "memq/consolidate", "meeting moved to 3pm"),
		Durability: DurabilityDurable,
		Consent:    ConsentInferred,
		Producer:   "memq/consolidate",
		Reason:     "recurring fact promoted to durable on repeated confirmation",
	})

	tl := m.Promotions().Timeline(cell.ID)
	if tl.CellID != cell.ID {
		t.Fatalf("CellID = %q, want %q", tl.CellID, cell.ID)
	}
	if len(tl.Events) != 2 {
		t.Fatalf("len(Events) = %d, want 2 (both promotion records)", len(tl.Events))
	}
	if tl.Events[0].Seq != 0 || tl.Events[1].Seq != 1 {
		t.Errorf("events must be in ledger insertion order: got Seq %d, %d", tl.Events[0].Seq, tl.Events[1].Seq)
	}
	first, second := tl.Events[0], tl.Events[1]
	if first.Kind != memview.EventPromotion || second.Kind != memview.EventPromotion {
		t.Errorf("both events must be EventPromotion, got %q and %q", first.Kind, second.Kind)
	}
	if first.Durability != DurabilityBounded || second.Durability != DurabilityDurable {
		t.Errorf("durability progression not preserved: got %q then %q", first.Durability, second.Durability)
	}
	if first.Consent != ConsentExplicit || second.Consent != ConsentInferred {
		t.Errorf("consent not preserved: got %q then %q", first.Consent, second.Consent)
	}
	if second.Producer != "memq/consolidate" {
		t.Errorf("Producer = %q, want memq/consolidate", second.Producer)
	}

	rendered := tl.Render()
	if rendered == "" {
		t.Error("Render() must not be empty")
	}
}

// TestPromotionLedgerTimelineUnknownCellIsEmptyNotError proves a cell with no
// promotion history (never promoted, or unknown to this ledger) renders an honest
// empty timeline rather than erroring — the same Found=false posture Explain takes.
func TestPromotionLedgerTimelineUnknownCellIsEmptyNotError(t *testing.T) {
	m := NewMemStore()
	tl := m.Promotions().Timeline("cell:does-not-exist")
	if len(tl.Events) != 0 {
		t.Fatalf("len(Events) = %d, want 0 for an unknown cell", len(tl.Events))
	}
	if got := tl.Render(); got == "" {
		t.Error("Render() must still produce a (non-empty) 'no history' message")
	}
}

// memview_testSpan is a tiny local helper building a SourceSpan for the second,
// manually-recorded PromotionRecord in the ordering test above.
func memview_testSpan(step int, role, descriptor string) SourceSpan {
	return SourceSpan{Step: step, Role: role, Descriptor: descriptor}
}
