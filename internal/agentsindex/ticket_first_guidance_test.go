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

func TestAgentGuidanceDefaultsSubstantiveWorkToWorkers(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test file")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	checks := map[string][]string{
		"AGENTS.md": {
			"Delegate real work; keep the coordinator context clean",
			"Use guarded headless agents or an equivalent isolated worker for every substantive",
			"independently witness worker results",
		},
		"CLAUDE.md": {
			"Delegate substantive work and keep this coordinator context clean",
			"independently verify worker effects",
		},
		"GEMINI.md": {
			"Delegate substantive work and keep the coordinator context clean",
			"compact witnessed evidence in the primary context",
		},
		filepath.Join(".github", "copilot-instructions.md"): {
			"Delegate substantive work and keep the coordinator context clean",
			"compact witnessed evidence in the primary context",
		},
	}

	for rel, wants := range checks {
		body, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		for _, want := range wants {
			if !strings.Contains(string(body), want) {
				t.Errorf("%s must retain worker-first context-hygiene guidance %q", rel, want)
			}
		}
	}
}
