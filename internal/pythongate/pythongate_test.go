package pythongate

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
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

// TestTestCompanionProvenance is the narrow exception witness: every declared row
// is an exact sibling test, was introduced by a pinned commit, and is admitted only
// because its tracked grandfathered module is imported in-process. This prevents a
// generic *_test.py exemption from laundering a new Python capability.
func TestTestCompanionProvenance(t *testing.T) {
	root := repoRoot(t)
	trackedPaths, err := trackedPyTools(root)
	if err != nil {
		t.Fatal(err)
	}
	tracked := make(map[string]bool, len(trackedPaths))
	for _, path := range trackedPaths {
		tracked[path] = true
	}
	baseline := baselineSet()
	admitted := admittedTestCompanions(root, tracked, baseline)
	sha := regexp.MustCompile(`^[0-9a-f]{40}$`)
	for _, companion := range testCompanions {
		if companion.Path != strings.TrimSuffix(companion.Module, ".py")+"_test.py" {
			t.Errorf("%s is not the exact test sibling of %s", companion.Path, companion.Module)
		}
		if !sha.MatchString(companion.IntroducedBy) {
			t.Errorf("%s has invalid introducing commit %q", companion.Path, companion.IntroducedBy)
		} else {
			cmd := exec.Command("git", "-c", "core.quotepath=false", "diff-tree", "--root", "--no-ext-diff", "--no-renames", "--no-commit-id", "--name-status", "-r", companion.IntroducedBy, "--", companion.Path)
			cmd.Dir = root
			out, err := cmd.Output()
			if err != nil || strings.TrimSpace(string(out)) != "A\t"+companion.Path {
				t.Errorf("%s was not introduced by %s: output=%q err=%v", companion.Path, companion.IntroducedBy, out, err)
			}
		}
		if !baseline[companion.Module] || !tracked[companion.Module] {
			t.Errorf("%s owner %s is not tracked and grandfathered", companion.Path, companion.Module)
		}
		if !admitted[companion.Path] {
			t.Errorf("%s did not satisfy its live import provenance", companion.Path)
		}
	}

	// Exactness witness: an undeclared sibling test remains an offense even when its
	// would-be owner is grandfathered. The contract is reviewed rows, not a wildcard.
	allowed := map[string]bool{"tools/owner.py": true}
	offenses := offensesAgainst([]string{"tools/owner.py", "tools/owner_test.py"}, allowed)
	if len(offenses) != 1 || offenses[0].Path != "tools/owner_test.py" {
		t.Fatalf("undeclared test companion escaped the ratchet: %v", offenses)
	}
}

func TestImportsModuleUsesPythonSyntax(t *testing.T) {
	for _, tc := range []struct {
		name   string
		source string
		want   bool
	}{
		{name: "import", source: "import owner\n", want: true},
		{name: "aliased import", source: "import owner as subject\n", want: true},
		{name: "from import", source: "from owner import run\n", want: true},
		{name: "docstring spoof", source: `"""\nimport owner\n"""\n`, want: false},
		{name: "comment spoof", source: "# import owner\n", want: false},
		{name: "different module", source: "import owner_extra\n", want: false},
		{name: "invalid syntax", source: "import owner as\n", want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := importsModule([]byte(tc.source), "tools/owner.py"); got != tc.want {
				t.Fatalf("importsModule=%v, want %v", got, tc.want)
			}
		})
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
