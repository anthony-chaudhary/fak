package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/pythongate"
)

// TestPythonToolGate_LiveTreeClean: on the real tracked tree the pre-push gate must find
// ZERO new python tools — the same end-to-end witness pythongate.TestNoNewPythonTools
// provides, one boundary earlier. If this reds, the trunk's ci-fast subset is about to red
// too (that is exactly the failure mode this gate exists to prevent).
func TestPythonToolGate_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gatePythonToolTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Logf("new python tool(s) on the tracked tree (ci-fast is about to red): %+v", findings)
	}
}

// TestPythonToolGate_AgreesWithRatchet is the anti-rival-authority witness: the pre-push
// gate and the scorecard ratchet it fronts must return the SAME verdict on the live tree.
// If they diverge, one of them has drifted from the shared grandfathered baseline and the
// gate would either wave through a tool the ratchet reds, or refuse one the ratchet allows.
func TestPythonToolGate_AgreesWithRatchet(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := repoRoot(t)
	tree, err := ReadTrackedTree(root)
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	gateFindings, gerr := gatePythonToolTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	ratchet, rerr := pythongate.ScanTree(root)
	if rerr != nil {
		t.Skipf("pythongate.ScanTree: %v", rerr)
	}
	if len(gateFindings) != len(ratchet) {
		t.Fatalf("gate/ratchet disagree on the live tree: gate=%d offenses, ratchet=%d offenses",
			len(gateFindings), len(ratchet))
	}
	// The gate's reason code must match the ratchet's closed-vocabulary reason.
	if reasonNewPythonTool != pythongate.ReasonNewPythonTool {
		t.Fatalf("reason code drift: gate %q != ratchet %q", reasonNewPythonTool, pythongate.ReasonNewPythonTool)
	}
}

func TestPythonToolGateAdmitsOnlyDeclaredSyntaxImportedCompanion(t *testing.T) {
	build := func(testSource string, declare bool) *TrackedTree {
		root := t.TempDir()
		files := map[string]string{
			pythonBaselineFile:    "package pythongate\n\nvar grandfathered = []string{\n\t\"tools/owner.py\",\n}\n",
			"tools/owner.py":      "def run(): pass\n",
			"tools/owner_test.py": testSource,
		}
		if declare {
			files[pythonTestCompanionFile] = "package pythongate\n\nvar testCompanions = []TestCompanion{\n\t{\"tools/owner_test.py\", \"tools/owner.py\", \"0123456789012345678901234567890123456789\"},\n}\n"
		} else {
			files[pythonTestCompanionFile] = "package pythongate\n\nvar testCompanions = []TestCompanion{}\n"
		}
		paths := make([]string, 0, len(files))
		for rel, body := range files {
			p := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			paths = append(paths, rel)
		}
		return &TrackedTree{Root: root, Paths: paths, fileCache: map[string]fileEntry{}}
	}

	if findings, err := gatePythonToolTree(build("import owner\n", true)); err != nil || len(findings) != 0 {
		t.Fatalf("declared syntax import was not admitted: findings=%v err=%v", findings, err)
	}
	for _, tc := range []struct {
		name    string
		source  string
		declare bool
	}{
		{name: "undeclared", source: "import owner\n"},
		{name: "docstring spoof", source: "\"\"\"\nimport owner\n\"\"\"\n", declare: true},
		{name: "invalid syntax", source: "import owner as\n", declare: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			findings, err := gatePythonToolTree(build(tc.source, tc.declare))
			if err != nil || len(findings) != 1 || findings[0].File != "tools/owner_test.py" {
				t.Fatalf("companion escaped: findings=%v err=%v", findings, err)
			}
		})
	}
}

// TestPythonToolGate_FiresOnNewTool: a synthetic tree with a baseline that does NOT list
// tools/newthing.py must produce exactly one NEW_PYTHON_TOOL finding; adding the row clears
// it. An untracked scratch .py (absent from Paths) must NOT fire. The nested tools/**/x.py is
// grandfathered here so this case isolates the top-level new tool — nested paths ARE in scope
// (git pathspec `*` spans `/`); TestPythonToolGate_FiresOnNestedNewTool witnesses that.
func TestPythonToolGate_FiresOnNewTool(t *testing.T) {
	baselineBody := func(declareNew bool) string {
		rows := "\t\"tools/existing.py\",\n\t\"tools/nested/deep.py\",\n"
		if declareNew {
			rows += "\t\"tools/newthing.py\",\n"
		}
		return "package pythongate\n\nvar grandfathered = []string{\n" + rows + "}\n"
	}

	build := func(declareNew bool) *TrackedTree {
		root := t.TempDir()
		write := func(rel, body string) {
			p := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write(pythonBaselineFile, baselineBody(declareNew))
		write("tools/existing.py", "print('ok')\n")
		write("tools/newthing.py", "print('new')\n")
		write("tools/nested/deep.py", "print('nested — in scope, grandfathered above')\n")
		return &TrackedTree{
			Root: root,
			Paths: []string{
				pythonBaselineFile,
				"tools/existing.py",
				"tools/newthing.py",
				"tools/nested/deep.py",
			},
			fileCache: map[string]fileEntry{},
		}
	}

	// Undeclared -> exactly one finding, naming the new tool (the nested one is grandfathered).
	findings, err := gatePythonToolTree(build(false))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 NEW_PYTHON_TOOL finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != "NEW_PYTHON_TOOL" || findings[0].File != "tools/newthing.py" {
		t.Fatalf("finding wrong: %+v", findings[0])
	}

	// Declared -> clean.
	findings, err = gatePythonToolTree(build(true))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("declared tool should be clean, got %+v", findings)
	}
}

// TestPythonToolGate_FiresOnNestedNewTool: git's `*` pathspec magic spans `/`, so the ratchet's
// `git ls-files tools/*.py` matches a NESTED tools/**/x.py too — pythongate.ScanTree reds CI on
// one. The pre-push gate must therefore refuse it as well; if it skips nested paths, a new nested
// Python module sails through pre-push and reds the shared trunk minutes later, which is exactly
// the divergence this gate exists to prevent. Declaring the nested path in the baseline clears it.
func TestPythonToolGate_FiresOnNestedNewTool(t *testing.T) {
	const nested = "tools/grafana/gen_dashboard.py"

	build := func(declareNested bool) *TrackedTree {
		root := t.TempDir()
		rows := "\t\"tools/existing.py\",\n"
		if declareNested {
			rows += "\t\"" + nested + "\",\n"
		}
		body := "package pythongate\n\nvar grandfathered = []string{\n" + rows + "}\n"
		p := filepath.Join(root, filepath.FromSlash(pythonBaselineFile))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return &TrackedTree{
			Root:      root,
			Paths:     []string{pythonBaselineFile, "tools/existing.py", nested},
			fileCache: map[string]fileEntry{},
		}
	}

	// Undeclared nested tool -> exactly one NEW_PYTHON_TOOL finding naming it.
	findings, err := gatePythonToolTree(build(false))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want 1 NEW_PYTHON_TOOL finding for the nested tool, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != reasonNewPythonTool || findings[0].File != nested {
		t.Fatalf("finding wrong: %+v", findings[0])
	}

	// Declared in the baseline -> clean.
	findings, err = gatePythonToolTree(build(true))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("declared nested tool should be clean, got %+v", findings)
	}
}

// TestPythonToolGate_ScopeMatchesRatchetPathspec is the scope-parity witness: every path the
// ratchet's own pathspec (`git ls-files tools/*.py`, run here against the live tree) returns must
// be IN the pre-push gate's scope. It proves scope parity on real data rather than on a synthetic
// path shape, so any future narrowing of the gate reds here instead of on the shared trunk.
//
// The trick that turns "scope" into something observable: point the gate at a synthetic root whose
// baseline grandfathers nothing that can exist, so every in-scope path necessarily surfaces as a
// finding and the finding set IS the gate's scope. The tracked list is read once and used both as
// the input and as the expectation, so concurrent edits by other sessions cannot race it.
func TestPythonToolGate_ScopeMatchesRatchetPathspec(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	lsFiles := exec.Command("git", "ls-files", "tools/*.py")
	lsFiles.Dir = repoRoot(t)
	out, err := lsFiles.Output()
	if err != nil {
		t.Skipf("git ls-files tools/*.py: %v", err)
	}
	var scanned []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			scanned = append(scanned, strings.ReplaceAll(line, "\\", "/"))
		}
	}
	if len(scanned) == 0 {
		t.Skip("no tracked tools/*.py on this tree")
	}

	root := t.TempDir()
	// A path that cannot collide with a real tool, so the baseline grandfathers nothing.
	body := "package pythongate\n\nvar grandfathered = []string{\n\t\"tools/zz_not_a_real_grandfathered_tool.py\",\n}\n"
	p := filepath.Join(root, filepath.FromSlash(pythonBaselineFile))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	tree := &TrackedTree{
		Root:      root,
		Paths:     append([]string{pythonBaselineFile}, scanned...),
		fileCache: map[string]fileEntry{},
	}
	findings, gerr := gatePythonToolTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	inScope := make(map[string]bool, len(findings))
	for _, f := range findings {
		inScope[f.File] = true
	}
	var missed []string
	for _, s := range scanned {
		if !inScope[s] {
			missed = append(missed, s)
		}
	}
	if len(missed) != 0 {
		t.Fatalf("gate skips %d of %d path(s) the ratchet's `git ls-files tools/*.py` scans, so a new "+
			"one there passes pre-push and reds CI: %v", len(missed), len(scanned), missed)
	}
}

// TestPythonToolGate_FailsOpenOnUnreadableBaseline: with no baseline file on the tree, the
// gate returns ErrCouldNotRun (fail open) rather than flagging every tracked tools/*.py.
func TestPythonToolGate_FailsOpenOnUnreadableBaseline(t *testing.T) {
	tree := &TrackedTree{
		Root:      t.TempDir(),
		Paths:     []string{"tools/whatever.py"},
		fileCache: map[string]fileEntry{},
	}
	if _, err := gatePythonToolTree(tree); err != ErrCouldNotRun {
		t.Fatalf("want ErrCouldNotRun on a missing baseline, got %v", err)
	}
}
