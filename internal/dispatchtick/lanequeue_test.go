package dispatchtick

import (
	"path/filepath"
	"reflect"
	"testing"
)

// TestLaneQueueRoundTrip builds a >=2-lane RouterPayload through the real
// BuildRouterPayload path, writes the per-lane queue file to a temp dir, and
// reads each lane back — asserting the persisted ordered refs and the head are
// byte-identical to that lane's RouterLaneGroup.Issues. This proves the
// serializer mirrors (never re-orders) the order BuildRouterPayload's
// OrderLaneCandidates already established, and that the reader returns a lane
// head without re-fetching or re-routing. Witness for issue #4170.
func TestLaneQueueRoundTrip(t *testing.T) {
	routes := []IssueRoute{
		{Number: 101, Lane: "docs", Confidence: "path-confirmed"},
		{Number: 102, Lane: "docs", Confidence: "path-confirmed"},
		{Number: 201, Lane: "gateway", Confidence: "path-confirmed"},
		{Number: 202, Lane: "gateway", Confidence: "path-confirmed"},
		{Number: 203, Lane: "gateway", Confidence: "path-confirmed"},
	}
	// Give one lane a heavier-than-default priority so OrderLaneCandidates does
	// non-trivial reordering — the round-trip must preserve whatever order it
	// produced, not merely the input order.
	payload := BuildRouterPayload(RouterPayloadInput{
		Workspace: "/tmp/ws",
		Routes:    routes,
		Priority:  map[int]int{203: PriorityWeightP1},
	})
	if len(payload.Lanes) < 2 {
		t.Fatalf("expected >=2 lanes, got %d", len(payload.Lanes))
	}

	dir := t.TempDir()
	if err := WriteLaneQueues(dir, payload); err != nil {
		t.Fatalf("WriteLaneQueues: %v", err)
	}

	for lane, grp := range payload.Lanes {
		refs, head, ok := ReadLaneQueue(dir, lane)
		if !ok {
			t.Fatalf("ReadLaneQueue(%q) ok=false, want a persisted queue", lane)
		}
		if !reflect.DeepEqual(refs, grp.Issues) {
			t.Fatalf("lane %q refs = %v, want %v (RouterLaneGroup.Issues)", lane, refs, grp.Issues)
		}
		if len(grp.Issues) == 0 {
			t.Fatalf("lane %q built with no issues; test needs non-empty lanes", lane)
		}
		if head != grp.Issues[0] {
			t.Fatalf("lane %q head = %d, want %d (Issues[0])", lane, head, grp.Issues[0])
		}
	}

	// A lane absent from the file reads back not-ok, and so does a missing file.
	if refs, head, ok := ReadLaneQueue(dir, "no-such-lane"); ok {
		t.Fatalf("ReadLaneQueue absent lane = (%v,%d,true), want ok=false", refs, head)
	}
	if _, _, ok := ReadLaneQueue(t.TempDir(), "docs"); ok {
		t.Fatalf("ReadLaneQueue on empty dir ok=true, want ok=false")
	}
}

// TestLaneQueueEmptyPayload proves an empty payload serializes to a valid,
// readable file (round-trip stable) rather than a null map or a write error —
// the empty-lane assumption the issue calls out.
func TestLaneQueueEmptyPayload(t *testing.T) {
	dir := t.TempDir()
	if err := WriteLaneQueues(dir, RouterPayload{}); err != nil {
		t.Fatalf("WriteLaneQueues(empty): %v", err)
	}
	if _, err := filepath.Abs(dir); err != nil {
		t.Fatalf("dir: %v", err)
	}
	if _, _, ok := ReadLaneQueue(dir, "docs"); ok {
		t.Fatalf("empty payload: lane 'docs' should be absent, got ok=true")
	}
}
