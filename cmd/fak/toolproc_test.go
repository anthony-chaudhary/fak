package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// TestToolprocHookJournalRoundTrip drives the seam-4 adapter end to end at
// the file level: pre → post → stop firings append journal lines that the
// same fold `fak toolproc ps` uses reads back as a clean table with the
// orphaned survivor flagged.
func TestToolprocHookJournalRoundTrip(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")

	pre := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_a","tool_input":{"command":"make test"}}`
	post := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_a","tool_input":{"command":"make test"},"tool_response":{"is_error":false}}`
	preOrphan := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_b","tool_input":{"command":"tail -f x"}}`
	stop := `{"session_id":"s1"}`

	steps := []struct {
		kind, payload string
		atMS          int64
	}{
		{"pre", pre, 1_000},
		{"post", post, 3_000},
		{"pre", preOrphan, 4_000},
		{"stop", stop, 9_000},
	}
	for _, s := range steps {
		if err := toolprocHookOnce(strings.NewReader(s.payload), s.kind, journal, toolproc.HookEnvelope{}, s.atMS); err != nil {
			t.Fatalf("hook %s: %v", s.kind, err)
		}
	}

	f, err := os.Open(journal)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	events, err := toolproc.ParseEvents(f)
	if err != nil {
		t.Fatalf("journal must be fold-clean: %v", err)
	}
	tab, err := toolproc.Fold(events, 10_000, toolproc.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if tab.Counts.Done != 1 || tab.Counts.Running != 1 || tab.Counts.Orphaned != 1 {
		t.Fatalf("want 1 done + 1 running orphan, got %+v", tab.Counts)
	}
	if !tab.AttentionNeeded {
		t.Error("the orphaned survivor must need attention")
	}
}

// TestToolprocHookFailOpen: garbage stdin, an unknown kind, and a missing
// journal directory all report an error from the inner helper but the hook
// entry point still exits 0 — observation never wedges the harness.
func TestToolprocHookFailOpen(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "j.jsonl")
	if err := toolprocHookOnce(strings.NewReader(`{nope`), "pre", journal, toolproc.HookEnvelope{}, 1_000); err == nil {
		t.Error("garbage payload must error internally")
	}
	var errOut strings.Builder
	if rc := runToolprocHook(strings.NewReader(`{nope`), &errOut, []string{"pre", "--journal", journal}); rc != 0 {
		t.Errorf("hook must exit 0 on failure (fail-open), got %d", rc)
	}
	if !strings.Contains(errOut.String(), "fail-open") {
		t.Errorf("failure must be reported to stderr, got %q", errOut.String())
	}
	if rc := runToolprocHook(strings.NewReader(""), &errOut, nil); rc != 0 {
		t.Errorf("missing kind must still exit 0, got %d", rc)
	}
}
