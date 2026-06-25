package pythongate

import (
	"os"
	"path/filepath"
	"testing"
)

// TestOffensesAgainst verifies the ratchet core on a synthetic tracked set + allowlist
// (verify the verifier): with baseline {a.py}, the baselined file is clean, adding b.py
// yields exactly one offense, and removing a.py from the tree yields zero offenses (the
// ratchet never complains about a tool that was ported away).
func TestOffensesAgainst(t *testing.T) {
	baseline := map[string]bool{"tools/a.py": true}

	// Only the grandfathered file present: clean.
	if off := offensesAgainst([]string{"tools/a.py"}, baseline); len(off) != 0 {
		t.Fatalf("grandfathered-only tree: want 0 offenses, got %v", off)
	}

	// A NEW file appears alongside the grandfathered one: exactly one offense, naming b.py.
	off := offensesAgainst([]string{"tools/a.py", "tools/b.py"}, baseline)
	if len(off) != 1 {
		t.Fatalf("added b.py: want 1 offense, got %v", off)
	}
	if off[0].Path != "tools/b.py" {
		t.Errorf("offense path = %q, want tools/b.py", off[0].Path)
	}
	if want := "tools/b.py is a NEW python tool; port it to Go instead (NEW_PYTHON_TOOL)"; off[0].String() != want {
		t.Errorf("offense string = %q, want %q", off[0].String(), want)
	}

	// The grandfathered file is removed from the tree (ported away): zero offenses.
	if off := offensesAgainst(nil, baseline); len(off) != 0 {
		t.Fatalf("removed a.py: want 0 offenses, got %v", off)
	}
}

// TestNoNewPythonTools is the live trunk guard: scanning the real repo's tracked
// tools/*.py against the frozen baseline must yield ZERO offenses. The day a stray new
// tools/*.py is added (and not grandfathered), this reds the trunk with its path.
func TestNoNewPythonTools(t *testing.T) {
	root := repoRoot(t)
	offenses, err := ScanTree(root)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(offenses) > 0 {
		t.Errorf("%d NEW python tool(s) not in the grandfathered baseline:", len(offenses))
		for _, o := range offenses {
			t.Errorf("  %s", o)
		}
		t.Errorf("fix: write the new tool in Go (a new internal/<name>/ package + cmd/fak shell), " +
			"not Python. If you legitimately ported-and-deleted a grandfathered .py, " +
			"regenerate internal/pythongate/baseline.go (see doc.go).")
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
