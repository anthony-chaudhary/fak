package agentsindex

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAgentGuidanceTracksWorkInGitHubBeforeImplementation(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test file")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	raw, err := os.ReadFile(filepath.Join(root, "AGENTS.md"))
	if err != nil {
		t.Fatalf("read AGENTS.md: %v", err)
	}
	guidance := string(raw)

	for _, want := range []string{
		"Use a GitHub issue as the durable tracker for every substantive unit of work whenever\nreasonably possible.",
		"Before editing code or docs, changing configuration, or launching an\nimplementation worker, search open and closed issues for duplicates",
		"claim the matching\nissue or create one",
		"Scoping, reproduction, and read-only triage may precede the issue",
		"Skip advance ticketing only when it would be unreasonable",
		"create or reconcile the issue as soon as the constraint clears",
		"Do not use plans, TODOs, commit messages, or chat as substitutes",
	} {
		if !strings.Contains(guidance, want) {
			t.Errorf("AGENTS.md lost ticket-first guidance %q", want)
		}
	}
}
