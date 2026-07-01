package memview

import (
	"strings"
	"testing"
)

// timeline_test.go — issue #1599 witness: `go test ./internal/memview ./internal/recall
// -run Provenance`. These tests exercise BuildTimeline/Render purely against the generic
// ProvenanceEvent shape (no memq/recall import — memview is tier 2 and must not import
// either tier-3 package; see timeline.go's header and internal/architest's tier map).
// The tier-3 adapters that translate memq.PromotionRecord / recall.StaleFactDecision
// into a ProvenanceEvent are tested in their own packages (internal/memq, internal/recall).

// TestProvenanceTimelineOrdersChronologically proves BuildTimeline sorts events by Seq
// (falling back to Step) regardless of input order, and Render lists them in that
// order — the "chronological account" the issue's Working spine calls for.
func TestProvenanceTimelineOrdersChronologically(t *testing.T) {
	events := []ProvenanceEvent{
		{Seq: 2, Step: 20, Kind: EventStaleTransition, Durability: "session", Outcome: "expired_must_query", Reason: "session-scoped fact recalled later"},
		{Seq: 0, Step: 5, Kind: EventPromotion, Durability: "durable", Producer: "user", Consent: "explicit", Digest: "abc123def456abcdef", Descriptor: "refund policy", Reason: "user said: remember this"},
		{Seq: 1, Step: 12, Kind: EventPromotion, Durability: "durable", Producer: "memq/consolidate", Consent: "inferred", Reason: "reclassified on consolidation"},
	}
	tl := BuildTimeline("cell:7", events)
	if len(tl.Events) != 3 {
		t.Fatalf("len(Events) = %d, want 3", len(tl.Events))
	}
	wantSeq := []int{0, 1, 2}
	for i, want := range wantSeq {
		if tl.Events[i].Seq != want {
			t.Errorf("Events[%d].Seq = %d, want %d (must be chronological, not input order)", i, tl.Events[i].Seq, want)
		}
	}

	out := tl.Render()
	iPromo := strings.Index(out, "producer=\"user\"")
	iReclass := strings.Index(out, "reclassified on consolidation")
	iStale := strings.Index(out, "expired_must_query")
	if iPromo < 0 || iReclass < 0 || iStale < 0 {
		t.Fatalf("Render() missing expected entries:\n%s", out)
	}
	if !(iPromo < iReclass && iReclass < iStale) {
		t.Errorf("Render() must list events in chronological (Seq) order, got:\n%s", out)
	}
}

// TestProvenanceTimelineNeverLeaksRawPayload proves the rendered timeline carries
// only safe, already-extractive metadata: a digest is truncated to a short prefix and
// the full raw value never appears, matching the issue's done condition ("no raw
// private payload leakage"). ProvenanceEvent has no field a raw body could occupy in
// the first place; this test pins the digest-truncation behavior specifically.
func TestProvenanceTimelineNeverLeaksRawPayload(t *testing.T) {
	longDigest := "0123456789abcdef0123456789abcdef0123456789abcdef"
	tl := BuildTimeline("cell:9", []ProvenanceEvent{
		{Seq: 0, Kind: EventPromotion, Durability: "durable", Producer: "tool", Digest: longDigest, Descriptor: "safe extractive descriptor"},
	})
	out := tl.Render()
	if strings.Contains(out, longDigest) {
		t.Errorf("Render() must truncate the digest, not print it in full:\n%s", out)
	}
	if !strings.Contains(out, longDigest[:12]) {
		t.Errorf("Render() must include the truncated digest prefix:\n%s", out)
	}
	if !strings.Contains(out, "safe extractive descriptor") {
		t.Errorf("Render() must include the safe descriptor:\n%s", out)
	}
}

// TestProvenanceTimelineEmptyIsRenderableNotError proves a cell with no recorded
// provenance renders a plain "no history" line rather than erroring — mirroring
// memq.Explanation's Found=false posture (an absent record is a valid, honest
// answer, not a refusal).
func TestProvenanceTimelineEmptyIsRenderableNotError(t *testing.T) {
	tl := BuildTimeline("cell:unknown", nil)
	if len(tl.Events) != 0 {
		t.Fatalf("len(Events) = %d, want 0", len(tl.Events))
	}
	out := tl.Render()
	if !strings.Contains(out, "no provenance events recorded") {
		t.Errorf("Render() of an empty timeline must say so plainly, got:\n%s", out)
	}
}

// TestProvenanceEventKindClosedVocabulary mirrors recall's
// TestStaleFactOutcomeClosedVocabulary shape: every declared kind is a member of the
// closed set, and an unrecognized value fails closed to "unknown(...)" rather than
// silently rendering as if it were a real kind.
func TestProvenanceEventKindClosedVocabulary(t *testing.T) {
	for _, k := range []ProvenanceEventKind{EventPromotion, EventStaleTransition, EventObserved} {
		if !ValidProvenanceEventKind(k) {
			t.Errorf("%q should be a valid ProvenanceEventKind", k)
		}
		if k.String() != string(k) {
			t.Errorf("String() of a valid kind must be itself, got %q", k.String())
		}
	}
	bogus := ProvenanceEventKind("bogus")
	if ValidProvenanceEventKind(bogus) {
		t.Error("bogus kind must not be valid")
	}
	if bogus.String() != "unknown(bogus)" {
		t.Errorf("String() of an invalid kind = %q, want unknown(bogus)", bogus.String())
	}
	if ProvenanceEventKind("").String() != "(unset)" {
		t.Errorf("String() of empty kind = %q, want (unset)", ProvenanceEventKind("").String())
	}
}
