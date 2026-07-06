package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// TestPrintRotationNoCandidateAnchorAware is the message-shape guard for the confusing case: the
// account you are ON has room, and only the OTHER bucket is capped. The line MUST name the roomy
// anchor and steer to "just launch without --rotate" — it must NOT read as the old bare
// "everything is walled", which sent the operator hunting through the account topology to discover
// their active account was fine all along.
func TestPrintRotationNoCandidateAnchorAware(t *testing.T) {
	room := 1.0
	walled := -1.0
	dec := accounts.RotationDecision{
		OK:         false,
		Reason:     accounts.RotationAllOthersWalled,
		Anchor:     accounts.RotationSeat{Name: "july6-netra", Headroom: &room},
		AnchorRoom: true,
		Walled:     []accounts.RotationSeat{{Name: "day26NEW-netra", Headroom: &walled}},
	}

	var buf bytes.Buffer
	printRotationNoCandidate(&buf, "fak accounts launch --rotate", dec)
	got := buf.String()

	for _, want := range []string{
		"july6-netra",       // names the roomy anchor
		"account with room", // says the anchor is fine
		"day26NEW-netra",    // names the capped bucket
		"without --rotate",  // the actionable fix
	} {
		if !strings.Contains(got, want) {
			t.Errorf("anchor-aware message missing %q\ngot: %s", want, got)
		}
	}
	// It must NOT frame the roomy-anchor case as a no-runtime-launchable dead-end (the old wording).
	if strings.Contains(got, "no runtime-launchable account") {
		t.Errorf("roomy-anchor case should not use the dead-end wording\ngot: %s", got)
	}
}

// TestPrintRotationNoCandidateRealDeadEnd covers the genuine "wait for reset" case: the anchor
// itself is walled, so a model/account switch cannot help — the message names the capped buckets
// and the reset advice, and does NOT claim the operator is on a roomy account.
func TestPrintRotationNoCandidateRealDeadEnd(t *testing.T) {
	walled := -1.0
	dec := accounts.RotationDecision{
		OK:         false,
		Reason:     accounts.RotationAllOthersWalled,
		Anchor:     accounts.RotationSeat{Name: "alice", Headroom: &walled},
		AnchorRoom: false,
		Walled:     []accounts.RotationSeat{{Name: "bob", Headroom: &walled}},
	}

	var buf bytes.Buffer
	printRotationNoCandidate(&buf, "fak accounts launch --rotate", dec)
	got := buf.String()

	if !strings.Contains(got, "Wait for reset") {
		t.Errorf("real dead-end should advise waiting for reset\ngot: %s", got)
	}
	// The anchor is walled, so the message must not claim the operator is ALREADY on a roomy
	// account (it may still advise moving the role TO one — that is the fix, not a false claim).
	if strings.Contains(got, "already on the only account with room") {
		t.Errorf("anchor is walled — must not claim the operator is already on a roomy account\ngot: %s", got)
	}
}

// TestPrintRotationNoCandidateOnlyBucket proves the single-bucket case reads as "just launch",
// not a wall — the anchor is named and there is simply nowhere else to go.
func TestPrintRotationNoCandidateOnlyBucket(t *testing.T) {
	dec := accounts.RotationDecision{
		OK:     false,
		Reason: accounts.RotationOnlyBucket,
		Anchor: accounts.RotationSeat{Name: "solo"},
	}
	var buf bytes.Buffer
	printRotationNoCandidate(&buf, "fak accounts next", dec)
	got := buf.String()
	if !strings.Contains(got, "solo") || !strings.Contains(got, "without --rotate") {
		t.Errorf("only-bucket message should name the anchor and steer to plain launch\ngot: %s", got)
	}
}
