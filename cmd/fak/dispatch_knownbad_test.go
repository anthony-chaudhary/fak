package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// twoLanePayload is the routed fixture the scope-hold tests fold: one issue routed to
// lane "foo" over internal/foo, one to lane "bar" over internal/bar, plus a pre-existing
// human-blocked skip so the tests can prove the known-bad hold is a DISTINCT row class.
func twoLanePayload() dispatchtick.RouterPayload {
	return dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 101, Title: "A", Lane: "foo", Confidence: "path-confirmed", Paths: []string{"internal/foo/thing.go"}, ExpectedSteps: 3},
			{Number: 102, Title: "B", Lane: "bar", Confidence: "path-confirmed", Paths: []string{"internal/bar/other.go"}, ExpectedSteps: 2},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"foo": {Tree: []string{"internal/foo/**"}, Count: 1, StepBudget: 3, Issues: []int{101}},
			"bar": {Tree: []string{"internal/bar/**"}, Count: 1, StepBudget: 2, Issues: []int{102}},
		},
		SkippedHumanBlocked: []dispatchtick.SkippedIssue{
			{Number: 200, Title: "waiting on legal", Reason: dispatchtick.ReasonBlockedByHuman},
		},
		Counts: dispatchtick.RouterCounts{
			Open: 3, Routed: 2, RoutedStepBudget: 5, SkippedHumanBlocked: 1,
			SkippedByReason: map[string]int{dispatchtick.ReasonBlockedByHuman: 1},
		},
	}
}

// liveKnownBadOverFoo is one open, never-expiring signature over internal/foo/** at nowUnix.
func liveKnownBadOverFoo(nowUnix int64) []knownbad.Record {
	return []knownbad.Record{
		knownbad.NewRecord("build", []string{"internal/foo/**"}, "shared build break", "agent-1", "", nowUnix, 0),
	}
}

// TestApplyKnownBadHoldScopesToIntersection is the #2716 Done condition: with one live
// known-bad over tree T, exactly the issues whose paths intersect T are held with reason
// BLOCKED_BY_KNOWN_BAD, and every disjoint issue still routes for dispatch.
func TestApplyKnownBadHoldScopesToIntersection(t *testing.T) {
	const now = 1_000_000
	got := applyKnownBadHold(twoLanePayload(), liveKnownBadOverFoo(now), now)

	// The intersecting issue (#101, internal/foo) is HELD: its lane is gone (held empty)
	// and it is no longer offered as a candidate.
	if _, ok := got.Lanes["foo"]; ok {
		t.Errorf("lane foo intersected the known-bad and must be held out, still present: %+v", got.Lanes["foo"])
	}
	for _, iss := range got.Issues {
		if iss.Number == 101 {
			t.Errorf("held issue #101 must not remain a routable candidate, got %+v", iss)
		}
	}

	// The DISJOINT issue (#102, internal/bar) still dispatches from its lane.
	bar, ok := got.Lanes["bar"]
	if !ok || len(bar.Issues) != 1 || bar.Issues[0] != 102 {
		t.Fatalf("disjoint issue #102 must still route in lane bar, got %+v (ok=%v)", bar, ok)
	}

	// #101 is now a BLOCKED_BY_KNOWN_BAD skip carrying the signature id; the human-blocked
	// row (#200) is untouched and distinct.
	held := knownBadBlockedSkipped(got)
	if len(held) != 1 || held[0].Number != 101 {
		t.Fatalf("want exactly one BLOCKED_BY_KNOWN_BAD row for #101, got %+v", held)
	}
	if !strings.Contains(held[0].NextAction, "sha256:") {
		t.Errorf("held row's next action should cite the signature id, got %q", held[0].NextAction)
	}
	if !strings.Contains(held[0].NextAction, "internal/foo/thing.go") {
		t.Errorf("held row's next action should name the intersecting tree, got %q", held[0].NextAction)
	}
	for _, s := range got.SkippedHumanBlocked {
		if s.Number == 200 && s.Reason != dispatchtick.ReasonBlockedByHuman {
			t.Errorf("pre-existing human-blocked row #200 must be preserved, got reason %q", s.Reason)
		}
	}

	// Counts reconcile: one fewer routed, the hold booked under its own reason.
	if got.Counts.Routed != 1 {
		t.Errorf("routed count should drop to 1 after holding #101, got %d", got.Counts.Routed)
	}
	if got.Counts.RoutedStepBudget != 2 {
		t.Errorf("routed step budget should be bar's 2 after the hold, got %d", got.Counts.RoutedStepBudget)
	}
	if got.Counts.SkippedByReason[reasonBlockedByKnownBad] != 1 {
		t.Errorf("SkippedByReason[%s] should be 1, got %d", reasonBlockedByKnownBad, got.Counts.SkippedByReason[reasonBlockedByKnownBad])
	}
	if got.Counts.SkippedHumanBlocked != 2 {
		t.Errorf("total skipped should be 2 (1 human + 1 known-bad), got %d", got.Counts.SkippedHumanBlocked)
	}
}

// TestApplyKnownBadHoldNoOps pins the disjoint/no-signal paths: an empty ledger, a
// signature over a disjoint tree, and an expired (non-live) signature all leave the
// dispatch surface untouched -- the hold never over-reaches into unrelated work.
func TestApplyKnownBadHoldNoOps(t *testing.T) {
	const now = 1_000_000

	// Empty ledger -> nothing held.
	if got := applyKnownBadHold(twoLanePayload(), nil, now); got.Counts.Routed != 2 {
		t.Errorf("empty ledger must hold nothing, routed=%d want 2", got.Counts.Routed)
	}

	// A live signature over a DISJOINT tree (internal/baz) holds nothing here.
	disjoint := []knownbad.Record{knownbad.NewRecord("test", []string{"internal/baz/**"}, "", "a", "", now, 0)}
	if got := applyKnownBadHold(twoLanePayload(), disjoint, now); len(knownBadBlockedSkipped(got)) != 0 {
		t.Errorf("a disjoint signature must hold nothing, held=%d", len(knownBadBlockedSkipped(got)))
	}

	// An EXPIRED signature over internal/foo (ttl in the past) is not live -> no hold.
	expired := []knownbad.Record{knownbad.NewRecord("build", []string{"internal/foo/**"}, "", "a", "", now-100, 10)}
	got := applyKnownBadHold(twoLanePayload(), expired, now)
	if len(knownBadBlockedSkipped(got)) != 0 {
		t.Errorf("an expired (non-live) signature must not hold, held=%d", len(knownBadBlockedSkipped(got)))
	}
	if _, ok := got.Lanes["foo"]; !ok {
		t.Errorf("lane foo must survive an expired signature")
	}
}

// TestRenderSkippedKnownBadCard pins the operator card: a distinct headline from the
// human-blocked one, one row per held issue, with the link/next-action.
func TestRenderSkippedKnownBadCard(t *testing.T) {
	rows := []dispatchtick.SkippedIssue{
		{Number: 101, Title: "A", Reason: reasonBlockedByKnownBad, NextAction: "held: internal/foo intersects live known-bad sha256:abc (build)"},
	}
	card := renderSkippedKnownBadCard(rows, "https://github.com/o/r")
	if !strings.Contains(card, "held by a live known-bad") {
		t.Errorf("card should headline the known-bad hold, got %q", card)
	}
	if !strings.Contains(card, "#101") || !strings.Contains(card, "A") {
		t.Errorf("card should list the held issue row, got %q", card)
	}
	// It must NOT reuse the human-blocked headline (distinct row class).
	if strings.Contains(card, "blocked by a human") {
		t.Errorf("known-bad card must not use the human-blocked headline, got %q", card)
	}
}
