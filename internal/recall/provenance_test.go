package recall

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memview"
)

// provenance_test.go — issue #1599 witness: `go test ./internal/memview ./internal/recall
// -run Provenance`.

// TestProvenanceStaleTransitionTranslatesToTimelineEvent proves a non-fresh
// StaleFactDecision becomes a well-formed memview.ProvenanceEvent carrying the page's
// safe descriptor/digest (never raw bytes — Page has no raw-body field to begin with)
// plus the decision's outcome/reason verbatim, so it can be merged onto a fact's
// provenance timeline alongside its promotion history.
func TestProvenanceStaleTransitionTranslatesToTimelineEvent(t *testing.T) {
	p := Page{
		Step:       3,
		Durability: durabilityBounded,
		ValidTo:    100,
		Digest:     "deadbeefcafef00d0102030405",
		Descriptor: "refund policy note",
	}
	decision, err := GuardAgainstStaleFact(p, StaleFactCheck{AsOf: 150, Required: true})
	if err == nil {
		t.Fatalf("fixture must produce a stale decision, got fresh")
	}

	ev, ok := StaleTransitionEvent(p, decision, 5)
	if !ok {
		t.Fatalf("a non-fresh decision must translate to an event")
	}
	if ev.Seq != 5 {
		t.Errorf("Seq = %d, want the supplied 5", ev.Seq)
	}
	if ev.Step != p.Step {
		t.Errorf("Step = %d, want %d", ev.Step, p.Step)
	}
	if ev.Kind != memview.EventStaleTransition {
		t.Errorf("Kind = %q, want %q", ev.Kind, memview.EventStaleTransition)
	}
	if ev.Outcome != string(StaleFactExpiredDeny) {
		t.Errorf("Outcome = %q, want %q", ev.Outcome, StaleFactExpiredDeny)
	}
	if ev.Digest != p.Digest || ev.Descriptor != p.Descriptor {
		t.Errorf("event must carry the page's digest/descriptor verbatim, got digest=%q descriptor=%q", ev.Digest, ev.Descriptor)
	}
	if ev.Reason == "" {
		t.Error("event must carry the decision's operator-readable reason")
	}
}

// TestProvenanceFreshDecisionIsNotATimelineEvent proves a StaleFactFresh decision is
// NOT placed on the timeline: "nothing changed" is not a provenance transition, so
// GuardAgainstStaleFact's success path must not spam the timeline with a fresh-event
// per read.
func TestProvenanceFreshDecisionIsNotATimelineEvent(t *testing.T) {
	p := Page{Step: 1, Durability: durabilityDurable()}
	decision, err := GuardAgainstStaleFact(p, StaleFactCheck{AsOf: 10, Required: true})
	if err != nil {
		t.Fatalf("durable fact must stay fresh, got error: %v", err)
	}
	if decision.Outcome != StaleFactFresh {
		t.Fatalf("fixture must decide fresh, got %q", decision.Outcome)
	}
	if _, ok := StaleTransitionEvent(p, decision, 0); ok {
		t.Error("a fresh decision must not translate to a timeline event")
	}
}

// TestProvenanceTimelineMergesPromotionAndStaleEvents is the end-to-end shape #1599
// asks for: a durable fact's timeline shows a promotion-style event (represented
// here as a plain memview.ProvenanceEvent — the memq-side translator is tested in
// internal/memq) chronologically followed by a stale-fact transition, rendered from
// structured data only.
func TestProvenanceTimelineMergesPromotionAndStaleEvents(t *testing.T) {
	promotion := memview.ProvenanceEvent{
		Seq: 0, Step: 2, Kind: memview.EventPromotion,
		Durability: "durable", Producer: "user", Consent: "explicit",
		Descriptor: "refund policy note", Reason: "user said: remember this",
	}
	p := Page{Step: 9, Durability: durabilityBounded, ValidTo: 5, Digest: "abc123", Descriptor: "refund policy note"}
	decision, _ := GuardAgainstStaleFact(p, StaleFactCheck{AsOf: 20, Required: false})
	staleEv, ok := StaleTransitionEvent(p, decision, 1)
	if !ok {
		t.Fatalf("expired bounded fact must yield a transition event")
	}

	tl := memview.BuildTimeline("cell:demo", []memview.ProvenanceEvent{staleEv, promotion})
	if len(tl.Events) != 2 {
		t.Fatalf("len(Events) = %d, want 2", len(tl.Events))
	}
	if tl.Events[0].Kind != memview.EventPromotion || tl.Events[1].Kind != memview.EventStaleTransition {
		t.Errorf("timeline must order promotion (Seq 0) before the stale transition (Seq 1), got %v then %v",
			tl.Events[0].Kind, tl.Events[1].Kind)
	}
	out := tl.Render()
	if out == "" {
		t.Error("Render() must not be empty for a non-empty timeline")
	}
}

// durabilityDurable is a tiny local helper so the test fixture reads plainly (recall
// already exposes DurabilityDurable indirectly via ctxmmu; using the package-level
// class name here keeps the test readable without importing ctxmmu just for this).
func durabilityDurable() string { return "durable" }
