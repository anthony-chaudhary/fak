package blockerpost

import (
	"strings"
	"testing"
)

// The three W7 (#2719) render cases the done-condition names: an empty ledger folds to a
// quiet all-clear; a live signature WITH a fixer folds to a muted status card carrying
// the blast frame; an UNCLAIMED-and-overdue signature escalates to a surfaced operator
// page. These assert on the same Text()/Blocks() the Slack feeder posts.

func TestFoldBlastEmptyIsClear(t *testing.T) {
	b := FoldBlast(nil, "https://github.com/o/r")
	if b.Severity != SeverityClear {
		t.Fatalf("no live signatures should fold to clear, got %q", b.Severity)
	}
	if strings.Contains(b.Text(), "<!here>") {
		t.Fatalf("an all-clear card MUST NOT page:\n%s", b.Text())
	}
	if !strings.Contains(b.Text(), "no shared blockers") {
		t.Fatalf("clear fold missing the all-clear headline:\n%s", b.Text())
	}
}

func TestFoldBlastClaimedIsMutedStatusCard(t *testing.T) {
	sigs := []Signature{{
		ID:             "sha256:abcdef0123456789aa",
		Reason:         "build",
		Trees:          []string{"internal/foo/**"},
		Affected:       6,
		Fixer:          "agent-7",
		WitnessPending: true,
	}}
	b := FoldBlast(sigs, "https://github.com/o/r")
	if b.Severity != SeverityStatus {
		t.Fatalf("a claimed, progressing signature must be muted status, got %q", b.Severity)
	}
	got := b.Text()
	if strings.Contains(got, "<!here>") || strings.Contains(got, "<!channel>") {
		t.Fatalf("a contained (fixer-claimed) blast card MUST NOT page:\n%s", got)
	}
	// The blast frame: 6 affected, 1 fixing (@agent-7), 5 parked, witness pending.
	for _, want := range []string{"6 affected", "1 fixing (@agent-7)", "5 parked", "witness: pending"} {
		if !strings.Contains(got, want) {
			t.Fatalf("blast card missing %q:\n%s", want, got)
		}
	}
	// The signature is shortened to scheme + a hex prefix, never the full 64-char digest.
	if strings.Contains(got, "abcdef0123456789aa") {
		t.Fatalf("the full signature body should be truncated in the row:\n%s", got)
	}
	if !strings.Contains(got, "sha256:abcdef012345") {
		t.Fatalf("the shortened signature should carry a hex prefix:\n%s", got)
	}
	if strings.Contains(blocksText(b), "<!here>") {
		t.Fatalf("the Block Kit path must also not page for a status card:\n%s", blocksText(b))
	}
}

func TestFoldBlastNoFixerOverdueSurfacesToOperator(t *testing.T) {
	sigs := []Signature{
		{ID: "sha256:claimed", Reason: "test", Trees: []string{"internal/a/**"}, Affected: 2, Fixer: "agent-1", WitnessPending: true},
		{ID: "sha256:orphan", Reason: "build", Trees: []string{"internal/b/**"}, Affected: 4, WitnessPending: true, NoFixerOverdue: true},
	}
	b := FoldBlast(sigs, "https://github.com/o/r")
	if b.Severity != SeverityOperator {
		t.Fatalf("an unclaimed, overdue signature must surface to operator, got %q", b.Severity)
	}
	got := b.Text()
	if !strings.Contains(got, "<!here>") {
		t.Fatalf("operator blast card must page:\n%s", got)
	}
	if !strings.Contains(got, "NO FIXER") {
		t.Fatalf("the orphan signature should be marked NO FIXER:\n%s", got)
	}
	// Worst-first: the overdue orphan must precede the claimed one.
	orphan := strings.Index(got, "sha256:orphan")
	claimed := strings.Index(got, "sha256:claimed")
	if orphan < 0 || claimed < 0 || orphan > claimed {
		t.Fatalf("overdue-unclaimed signature not listed first (orphan=%d claimed=%d):\n%s", orphan, claimed, got)
	}
	// The operator card links the ledger as its do-this-next affordance.
	if b.ActionURL == "" || !strings.Contains(b.ActionURL, "known-bad.jsonl") {
		t.Fatalf("operator blast card should link the known-bad ledger, got %q", b.ActionURL)
	}
}

// An unclaimed signature that is NOT yet overdue must stay muted — a just-discovered bug
// should not page before the fleet has had a tick to elect a fixer (W5).
func TestFoldBlastUnclaimedNotOverdueStaysStatus(t *testing.T) {
	sigs := []Signature{{ID: "sha256:fresh", Reason: "build", Trees: []string{"internal/c/**"}, Affected: 3, WitnessPending: true}}
	b := FoldBlast(sigs, "")
	if b.Severity != SeverityStatus {
		t.Fatalf("an unclaimed-but-not-overdue signature must stay status, got %q", b.Severity)
	}
	if strings.Contains(b.Text(), "<!here>") {
		t.Fatalf("a not-yet-overdue card MUST NOT page:\n%s", b.Text())
	}
	// With no fixer, all 3 affected are parked (no -1 for a fixer).
	if !strings.Contains(b.Text(), "3 parked") {
		t.Fatalf("no-fixer parked count should equal the affected count:\n%s", b.Text())
	}
}

func TestFoldBlastTruncatesManySignatures(t *testing.T) {
	var sigs []Signature
	for i := 0; i < maxBlastLines+3; i++ {
		sigs = append(sigs, Signature{ID: "sha256:s", Reason: "build", Trees: []string{"internal/x"}, Affected: 1, Fixer: "a", WitnessPending: true})
	}
	b := FoldBlast(sigs, "")
	if !strings.Contains(b.Text(), "and 3 more") {
		t.Fatalf("a large live set should summarize the overflow:\n%s", b.Text())
	}
}
