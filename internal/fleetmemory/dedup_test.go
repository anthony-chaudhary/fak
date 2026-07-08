package fleetmemory

import "testing"

// The #2142 witness: write a fact, then write an equivalent fact and assert the
// second returns DUP_LESSON naming the first; a genuinely new fact still writes.
func TestPublishRefusesDuplicateWithMerge(t *testing.T) {
	l := New(nil)

	first := l.Publish(Lesson{ID: "git-hang", Fact: "Bash git hangs here — use PowerShell"})
	if first.Outcome != OutcomePublished {
		t.Fatalf("first write should PUBLISH, got %q", first.Outcome)
	}
	if l.Len() != 1 {
		t.Fatalf("ledger should hold 1 entry, has %d", l.Len())
	}

	// An equivalent fact (different casing/punctuation) is a duplicate.
	dup := l.Publish(Lesson{ID: "git-hang-again", Fact: "bash GIT hangs here, use powershell!"})
	if !dup.Refused() {
		t.Fatalf("equivalent fact should be refused DUP_LESSON, got %q", dup.Outcome)
	}
	if dup.Existing == nil || dup.Existing.ID != "git-hang" {
		t.Fatalf("DUP_LESSON must name the existing canonical entry (git-hang), got %+v", dup.Existing)
	}
	if dup.Reason == "" {
		t.Fatalf("DUP_LESSON must carry a refuse-with-merge reason")
	}
	if l.Len() != 1 {
		t.Fatalf("a refused duplicate must NOT grow the ledger; len=%d", l.Len())
	}

	// A genuinely new fact still writes.
	fresh := l.Publish(Lesson{ID: "wsl-route", Fact: "native go test is OS-blocked — route through WSL"})
	if fresh.Outcome != OutcomePublished {
		t.Fatalf("a novel fact should PUBLISH, got %q", fresh.Outcome)
	}
	if l.Len() != 2 {
		t.Fatalf("ledger should hold 2 entries after a fresh write, has %d", l.Len())
	}
}

func TestPublishRefusesEmptyFact(t *testing.T) {
	l := New(nil)
	r := l.Publish(Lesson{ID: "blank", Fact: "   "})
	if !r.Refused() {
		t.Fatalf("an empty fact should be refused, got %q", r.Outcome)
	}
	if l.Len() != 0 {
		t.Fatalf("an empty fact must not be stored; len=%d", l.Len())
	}
}

// Publishing against a pre-seeded ledger (the cross-agent case: my write meets a
// peer's already-published lesson) refuses with the peer's entry.
func TestPublishAgainstSeededLedger(t *testing.T) {
	l := New([]Lesson{{ID: "peer-lesson", Fact: "commit by explicit path, never git add -A"}})
	r := l.Publish(Lesson{ID: "my-lesson", Fact: "Commit by explicit path; never git add -A."})
	if !r.Refused() || r.Existing == nil || r.Existing.ID != "peer-lesson" {
		t.Fatalf("expected DUP_LESSON naming peer-lesson, got outcome=%q existing=%+v", r.Outcome, r.Existing)
	}
}
