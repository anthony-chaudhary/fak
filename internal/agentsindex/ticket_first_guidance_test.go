package agentsindex

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// guidanceRoot resolves the repository root that holds AGENTS.md. runtime.Caller
// paths are module-relative under -trimpath (the isolated buildcheck/validate
// compile), so the caller-derived root is only a first candidate; walking up from
// the test's working directory (go test runs in the package dir) finds the real
// checkout in both compile modes.
func guidanceRoot(t *testing.T) string {
	t.Helper()
	if _, file, _, ok := runtime.Caller(0); ok {
		root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
		if fileExists(filepath.Join(root, "AGENTS.md")) {
			return root
		}
	}
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("resolve working directory: %v", err)
	}
	for i := 0; i < 8; i++ {
		if fileExists(filepath.Join(dir, "AGENTS.md")) {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("locate AGENTS.md above the package directory")
	return ""
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func TestAgentGuidanceTracksWorkInGitHubBeforeImplementation(t *testing.T) {
	root := guidanceRoot(t)
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
	root := guidanceRoot(t)
	checks := map[string][]string{
		"AGENTS.md": {
			"Delegate real work; keep the coordinator context clean",
			"Use guarded headless agents or an equivalent isolated worker for every substantive",
			"independently witness worker results",
		},
		"CLAUDE.md": {
			"Delegate substantive work and keep this coordinator context clean",
			"independently verify\n  worker effects",
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
