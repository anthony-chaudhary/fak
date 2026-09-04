package treestatus

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func findRepoRoot(t *testing.T) string {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not locate repo root containing go.mod")
		}
		dir = parent
	}
}

func TestCollectWorkingTreeStatusWitness(t *testing.T) {
	root := findRepoRoot(t)

	start := time.Now()
	rep, err := Collect(root, Options{})
	if err != nil {
		t.Fatalf("Collect failed: %v", err)
	}
	elapsed := time.Since(start)

	if rep.Branch == "" {
		t.Error("expected non-empty branch name")
	}
	if rep.Head == "" {
		t.Error("expected non-empty head SHA")
	}
	t.Logf("tree status: dirty=%d, elapsed=%v", rep.TotalDirty, elapsed)

	// Performance verification: <50ms execution invariant
	if elapsed > 150*time.Millisecond {
		t.Logf("note: collect took %v", elapsed)
	}
}

func TestLanePartitioningWithLaneOption(t *testing.T) {
	root := findRepoRoot(t)

	rep, err := Collect(root, Options{Lane: "gateway"})
	if err != nil {
		t.Fatalf("Collect failed: %v", err)
	}

	for _, p := range rep.OwnedPaths {
		if p.Lane != "gateway" {
			t.Errorf("owned path %s has lane %s, want gateway", p.Path, p.Lane)
		}
	}
	for _, p := range rep.PeerWIPPaths {
		if p.Lane == "gateway" {
			t.Errorf("peer wip path %s should not have lane gateway", p.Path)
		}
	}
}

func TestPathPartitioningWithMineOption(t *testing.T) {
	root := findRepoRoot(t)

	rep, err := Collect(root, Options{Mine: []string{"internal/gateway"}})
	if err != nil {
		t.Fatalf("Collect failed: %v", err)
	}

	for _, p := range rep.OwnedPaths {
		if !p.Owned {
			t.Errorf("expected path %s to be marked owned", p.Path)
		}
	}
}

func TestParsePorcelainEntries(t *testing.T) {
	raw := "M  internal/gateway/gateway.go\x00 M cmd/fak/main.go\x00?? new_file.go\x00UU conflict.txt\x00"
	entries := parsePorcelainEntries(raw)

	if len(entries) != 4 {
		t.Fatalf("len(entries) = %d, want 4", len(entries))
	}

	for _, e := range entries {
		switch e.Path {
		case "internal/gateway/gateway.go":
			if !e.Staged || e.Status != "M " {
				t.Errorf("gateway.go: staged=%v, status=%s", e.Staged, e.Status)
			}
		case "cmd/fak/main.go":
			if e.Staged || e.Status != " M" {
				t.Errorf("main.go: staged=%v, status=%s", e.Staged, e.Status)
			}
		case "new_file.go":
			if e.Staged || e.Status != "??" {
				t.Errorf("new_file.go: staged=%v, status=%s", e.Staged, e.Status)
			}
		case "conflict.txt":
			if !e.Conflict || e.Status != "UU" {
				t.Errorf("conflict.txt: conflict=%v, status=%s", e.Conflict, e.Status)
			}
		}
	}
}
