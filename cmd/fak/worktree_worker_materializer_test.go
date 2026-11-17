package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// fak-test:runtime slow est=30s lane=default
// fak-test:justify why=regression when=changed:cmd/fak/worktree_worker.go,changed:internal/witness/**,changed:internal/workerworktree/**
func TestWorkerLandMaterializerConfirmsOverlayAndReportsWarmSnapshots(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "repo")
	peer := filepath.Join(base, "peer")
	for _, dir := range []string{root, peer} {
		if err := os.Mkdir(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(dir, name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write(peer, "go.mod", "module peer\n\ngo 1.22\n")
	worktreeWorkerTestGit(t, root, "init", "-q")
	worktreeWorkerTestGit(t, root, "config", "user.email", "fixture@example.invalid")
	worktreeWorkerTestGit(t, root, "config", "user.name", "fixture")
	write(root, "go.mod", "module fixture\n\ngo 1.22\n")
	write(root, "go.work", "go 1.22\n\nuse (\n .\n ../peer\n)\n")
	write(root, "value.go", "package fixture\nfunc Good() bool { return false }\n")
	worktreeWorkerTestGit(t, root, "add", ".")
	worktreeWorkerTestGit(t, root, "commit", "-qm", "parent")
	parent := worktreeWorkerTestGit(t, root, "rev-parse", "HEAD")
	write(root, "value.go", "package fixture\nfunc Good() bool { return true }\n")
	write(root, "value_test.go", "package fixture\nimport \"testing\"\nfunc TestSnapshotFix(t *testing.T) { if !Good() { t.Fatal(\"missing fix\") } }\n")
	worktreeWorkerTestGit(t, root, "add", ".")
	worktreeWorkerTestGit(t, root, "commit", "-qm", "fix: restore value")
	candidate := worktreeWorkerTestGit(t, root, "rev-parse", "HEAD")
	for attempt := range 2 {
		res := verifyWorkerLandSymptomSelected(root, candidate, nil, []string{"^TestSnapshotFix$"}, 3*time.Minute)
		if !res.OK || res.SymptomVerdict != symptomVerdictConfirmed {
			t.Fatalf("attempt=%d result=%+v", attempt, res)
		}
		_, raw, ok := strings.Cut(res.Detail, " | witness_snapshots=")
		if !ok {
			t.Fatalf("missing native materialization evidence: %+v", res)
		}
		var events []workerworktree.WitnessSnapshotEvent
		if err := json.Unmarshal([]byte(raw), &events); err != nil {
			t.Fatal(err)
		}
		if len(events) != 4 {
			t.Fatalf("events=%+v", events)
		}
		prepared := map[string]string{}
		released := map[string]bool{}
		for _, e := range events {
			if e.Error != "" || e.Duration < 0 || e.Dir == "" {
				t.Fatalf("invalid event=%+v", e)
			}
			if e.Phase == "prepare" {
				prepared[e.Ref] = e.Dir
				if e.Reused != (attempt == 1) {
					t.Fatalf("attempt=%d prepare=%+v", attempt, e)
				}
			} else if e.Phase == "release" {
				released[e.Dir] = true
			} else {
				t.Fatalf("unknown phase=%q", e.Phase)
			}
		}
		if prepared[parent] == "" || prepared[candidate] == "" || prepared[parent] == prepared[candidate] || !released[prepared[parent]] || !released[prepared[candidate]] {
			t.Fatalf("source pins/releases=%+v", events)
		}
	}
	if got := worktreeWorkerTestGit(t, root, "status", "--porcelain"); got != "" {
		t.Fatalf("caller source mutated: %s", got)
	}
}
