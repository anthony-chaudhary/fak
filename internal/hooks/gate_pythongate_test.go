package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
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
		t.Fatalf("new python tool(s) on the tracked tree (ci-fast is about to red): %+v", findings)
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

// TestPythonToolGate_FiresOnNewTool: a synthetic tree with a baseline that does NOT list
// tools/newthing.py must produce exactly one NEW_PYTHON_TOOL finding; adding the row clears
// it. An untracked scratch .py (absent from Paths) and a nested tools/**/x.py must NOT fire.
func TestPythonToolGate_FiresOnNewTool(t *testing.T) {
	baselineBody := func(declareNew bool) string {
		rows := "\t\"tools/existing.py\",\n"
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
		write("tools/nested/deep.py", "print('nested — one level deeper, out of scope')\n")
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

	// Undeclared -> exactly one finding, naming the new tool; the nested one does not fire.
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
