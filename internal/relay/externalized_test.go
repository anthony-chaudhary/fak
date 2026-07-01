package relay

import "testing"

// Issue #1884 done condition: a planted transcript-only fact is flagged; a fully
// externalized state is clean. These are that witness (run: `go test ./internal/relay -run
// TranscriptOnly`).

// TestTranscriptOnlyFlagsUnbackedFacts plants a mix of facts and asserts only those with no
// durable backing (no ref, or a ref under an unknown kind) are flagged — the rest, backed
// by a commit/issue/memory/ledger/file pointer, are treated as externalized.
func TestTranscriptOnlyFlagsUnbackedFacts(t *testing.T) {
	facts := []LoadBearingFact{
		{Label: "the fix commit", Backing: Artifact{Kind: string(ArtifactCommit), Ref: "abc123"}},
		{Label: "the tracking issue", Backing: Artifact{Kind: string(ArtifactIssue), Ref: "#1884"}},
		{Label: "a decision I only said out loud", Backing: Artifact{}},                            // transcript-only
		{Label: "a pointer with no ref", Backing: Artifact{Kind: string(ArtifactMemory), Ref: ""}}, // transcript-only
		{Label: "a pointer of unknown kind", Backing: Artifact{Kind: "slack-thread", Ref: "T123"}}, // transcript-only
	}

	got := TranscriptOnly(facts)
	if len(got) != 3 {
		t.Fatalf("expected 3 transcript-only facts, got %d: %+v", len(got), got)
	}
	wantLabels := map[string]bool{
		"a decision I only said out loud": true,
		"a pointer with no ref":           true,
		"a pointer of unknown kind":       true,
	}
	for _, f := range got {
		if !wantLabels[f.Label] {
			t.Errorf("unexpected fact flagged: %q", f.Label)
		}
	}
	if FullyExternalized(facts) {
		t.Error("a state with transcript-only facts must not report FullyExternalized")
	}
}

// TestTranscriptOnlyCleanWhenExternalized asserts a fully externalized state produces no
// flags and reports FullyExternalized — one instance per durable kind.
func TestTranscriptOnlyCleanWhenExternalized(t *testing.T) {
	facts := []LoadBearingFact{
		{Label: "commit", Backing: Artifact{Kind: string(ArtifactCommit), Ref: "deadbeef"}},
		{Label: "issue", Backing: Artifact{Kind: string(ArtifactIssue), Ref: "#1"}},
		{Label: "memory", Backing: Artifact{Kind: string(ArtifactMemory), Ref: "relay-notes"}},
		{Label: "ledger", Backing: Artifact{Kind: string(ArtifactLedger), Ref: ".dos/runs/x.jsonl#L1"}},
		{Label: "file", Backing: Artifact{Kind: string(ArtifactFile), Ref: "internal/relay/**"}},
	}
	if got := TranscriptOnly(facts); len(got) != 0 {
		t.Errorf("a fully externalized state must be clean, got %+v", got)
	}
	if !FullyExternalized(facts) {
		t.Error("a fully externalized state must report FullyExternalized")
	}
	// The empty state is trivially clean and safe to rotate.
	if !FullyExternalized(nil) {
		t.Error("an empty fact set must report FullyExternalized")
	}
}
