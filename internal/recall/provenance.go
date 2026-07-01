package recall

import (
	"github.com/anthony-chaudhary/fak/internal/memview"
)

// provenance.go — issue #1599: the recall-side half of the provenance-timeline view.
// recall is tier 3 (composer); memview is tier 2 (mechanism), so this downward import
// is exactly the direction internal/architest's tier rule allows (recall may import
// memview; memview must never import recall back). This file's only job is a pure,
// lossless TRANSLATION from recall's own stale-fact vocabulary (stalefact.go, #1594)
// into memview's generic ProvenanceEvent — it invents no new fields and reinterprets
// nothing; a caller (e.g. cmd/fak/debug*.go) merges the resulting events with memq's
// promotion events (internal/memq/provenance.go) into one memview.Timeline.

// StaleTransitionEvent translates a StaleFactDecision (over the Page it was computed
// about) into a memview.ProvenanceEvent, for placement on that fact's provenance
// timeline. It reports ok=false for a StaleFactFresh decision: "still fresh" is a
// non-event for a TIMELINE OF TRANSITIONS (the issue's "subsequent staleness
// transitions" language) — nothing changed, so nothing is placed on the timeline.
// seq is the caller-assigned ordering key (memview.ProvenanceEvent.Seq); recall has
// no ledger of its own to assign one, so the caller (which is merging this with a
// promotion ledger's Seq space) supplies it.
func StaleTransitionEvent(p Page, d StaleFactDecision, seq int) (memview.ProvenanceEvent, bool) {
	if d.Outcome == StaleFactFresh {
		return memview.ProvenanceEvent{}, false
	}
	return memview.ProvenanceEvent{
		Seq:        seq,
		Step:       d.Step,
		Kind:       memview.EventStaleTransition,
		Durability: d.Durability,
		Producer:   "recall.DetectStaleFact",
		Digest:     p.Digest,
		Descriptor: p.Descriptor,
		Outcome:    string(d.Outcome),
		Reason:     d.Reason,
	}, true
}
