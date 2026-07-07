package rsiloop

import (
	"path/filepath"
	"testing"
)

// TestCuratorPerDecisionKeepRevert is the #2841 acceptance witness: every curator
// archive/consolidate action is journaled with a STRUCTURED reason, is individually
// revertible, and the read-path answers "why is this skill gone?" from the journal
// alone — reverting one decision never rolls back a sibling (the per-decision, not
// whole-snapshot, guarantee).
func TestCuratorPerDecisionKeepRevert(t *testing.T) {
	path := filepath.Join(t.TempDir(), "curator.jsonl")
	l, err := OpenCuratorLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}

	// Three archive/consolidate decisions, each with a DISTINCT structured reason —
	// the confusion risk #2841 names is that "stale" and "superseded" must not
	// collapse into one generic token.
	staleSeq, err := l.Archive("skill-alpha", CuratorReason{Kind: ReasonStale, StaleDays: 45})
	if err != nil {
		t.Fatalf("archive alpha: %v", err)
	}
	supSeq, err := l.Consolidate("skill-beta", CuratorReason{Kind: ReasonSuperseded, SupersededBy: "skill-gamma"})
	if err != nil {
		t.Fatalf("consolidate beta: %v", err)
	}
	if _, err := l.Archive("skill-delta", CuratorReason{Kind: ReasonSlopScored, SlopScore: 0.82}); err != nil {
		t.Fatalf("archive delta: %v", err)
	}

	// The read-path answers "why is this skill gone?" with the structured reason,
	// and the two archive reasons are genuinely distinct — not one generic reason.
	if r, gone := l.Why("skill-alpha"); !gone || r.Kind != ReasonStale || r.StaleDays != 45 {
		t.Fatalf("alpha reason = %+v gone=%v, want stale/45", r, gone)
	}
	if r, gone := l.Why("skill-beta"); !gone || r.Kind != ReasonSuperseded || r.SupersededBy != "skill-gamma" {
		t.Fatalf("beta reason = %+v gone=%v, want superseded-by-gamma", r, gone)
	}
	if r, _ := l.Why("skill-alpha"); r.String() == mustWhy(t, l, "skill-beta") {
		t.Fatalf("stale and superseded rendered identically (%q) — reasons collapsed", r.String())
	}

	// Per-decision revert: undo ONLY skill-alpha's archive. skill-alpha returns; the
	// sibling decisions (beta, delta) must be untouched — this is what Hermes' coarse
	// whole-snapshot restore cannot do.
	if err := l.Revert(staleSeq); err != nil {
		t.Fatalf("revert alpha: %v", err)
	}
	if _, gone := l.Why("skill-alpha"); gone {
		t.Fatalf("skill-alpha should be restored after per-decision revert")
	}
	if r, gone := l.Why("skill-beta"); !gone || r.Kind != ReasonSuperseded {
		t.Fatalf("sibling skill-beta was rolled back by alpha's revert: gone=%v reason=%+v", gone, r)
	}
	if _, gone := l.Why("skill-delta"); !gone {
		t.Fatalf("sibling skill-delta was rolled back by alpha's revert")
	}

	// Revert is refused when it cannot be scoped to a live decision.
	if err := l.Revert(staleSeq); err == nil {
		t.Fatalf("double-revert of seq %d should be refused", staleSeq)
	}
	if err := l.Revert(9999); err == nil {
		t.Fatalf("revert of unknown seq should be refused")
	}
	_ = supSeq

	// "From the journal alone": reopen the ledger from disk (no in-memory state) and
	// the read-path still reconstructs every decision and the revert.
	reopened, err := OpenCuratorLedger(path)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	if _, gone := reopened.Why("skill-alpha"); gone {
		t.Fatalf("reopened: skill-alpha revert did not survive on disk")
	}
	if r, gone := reopened.Why("skill-delta"); !gone || r.Kind != ReasonSlopScored || r.SlopScore != 0.82 {
		t.Fatalf("reopened: delta reason = %+v gone=%v, want slop-scored/0.82", r, gone)
	}

	// The folded Log surfaces exactly the still-gone skills with their governing reason.
	gone := map[string]CuratorReasonKind{}
	for _, e := range reopened.Log() {
		if e.Gone {
			gone[e.Skill] = e.Reason.Kind
		}
	}
	if len(gone) != 2 || gone["skill-beta"] != ReasonSuperseded || gone["skill-delta"] != ReasonSlopScored {
		t.Fatalf("Log gone-set = %+v, want {beta:superseded, delta:slop_scored}", gone)
	}
}

// TestCuratorRefusesUnstructuredReason guards that a decision without a valid
// structured reason never enters the journal — every archive must be answerable.
func TestCuratorRefusesUnstructuredReason(t *testing.T) {
	l, err := OpenCuratorLedger("")
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	if _, err := l.Archive("skill-x", CuratorReason{}); err == nil {
		t.Fatalf("archive with empty reason should be refused")
	}
	if _, err := l.Archive("skill-x", CuratorReason{Kind: ReasonStale}); err == nil {
		t.Fatalf("stale reason without StaleDays should be refused")
	}
	if _, err := l.Archive("", CuratorReason{Kind: ReasonStale, StaleDays: 1}); err == nil {
		t.Fatalf("archive with empty skill should be refused")
	}
	if len(l.Rows()) != 0 {
		t.Fatalf("refused decisions leaked into the journal: %+v", l.Rows())
	}
}

func mustWhy(t *testing.T, l *CuratorLedger, skill string) string {
	t.Helper()
	r, gone := l.Why(skill)
	if !gone {
		t.Fatalf("expected %s to be gone", skill)
	}
	return r.String()
}
