package memq

import (
	"github.com/anthony-chaudhary/fak/internal/memview"
)

// provenance.go — issue #1599: the memq-side half of the provenance-timeline view.
// memq is tier 3 (composer); memview is tier 2 (mechanism), so this downward import
// is exactly the direction internal/architest's tier rule allows. This file
// translates PromotionRecord (#1595, promotion.go) into memview's generic
// ProvenanceEvent — a pure, lossless field copy, same posture as
// PromotionLedger.Explain: no model call, no new vocabulary, nothing synthesized
// beyond formatting/ordering (which memview.Timeline.Render performs).

// ProvenanceEvent translates one PromotionRecord into a memview.ProvenanceEvent for
// placement on a cell's provenance timeline. Seq is copied verbatim from the record
// (the ledger's own insertion order), so a timeline built purely from a cell's
// PromotionLedger.For(cellID) history is already in chronological order before any
// stale-fact transitions (internal/recall) are merged in.
func (r PromotionRecord) ProvenanceEvent() memview.ProvenanceEvent {
	return memview.ProvenanceEvent{
		Seq:        r.Seq,
		Step:       r.SourceSpan.Step,
		Kind:       memview.EventPromotion,
		Durability: r.Durability,
		Producer:   r.Producer,
		Consent:    r.Consent,
		Digest:     r.SourceSpan.Digest,
		Descriptor: r.SourceSpan.Descriptor,
		Reason:     r.Reason,
	}
}

// Timeline renders the FULL provenance timeline for a cell: every PromotionRecord
// this ledger holds for cellID, translated to memview.ProvenanceEvent and ordered
// chronologically. A cell with no promotion history yields an empty (but still
// renderable — "no history found") memview.Timeline, mirroring Explain's
// Found=false posture rather than erroring.
func (l *PromotionLedger) Timeline(cellID string) memview.Timeline {
	recs, _ := l.For(cellID) // ok=false -> nil recs -> an empty, honestly-rendered timeline
	events := make([]memview.ProvenanceEvent, 0, len(recs))
	for _, r := range recs {
		events = append(events, r.ProvenanceEvent())
	}
	return memview.BuildTimeline(cellID, events)
}
