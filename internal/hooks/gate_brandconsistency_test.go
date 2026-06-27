package hooks

import (
	"os/exec"
	"testing"
)

// gate_brandconsistency_test.go — parity for the BRAND_CONSISTENCY gate. Because the Python
// checker is --audit-tree only (no --audit-staged / --root), the staged runParity harness does
// not cover it. Instead we (1) replay the EXACT golden vectors from
// tools/check_brand_consistency_test.py — the oracle's own must-flag / must-not-flag samples —
// against the ported per-line decision, (2) prove the live tracked tree is clean (the twin of
// the Python test_live_tree_is_clean), and (3) run a verdict-level differential against the live
// Python over the real tree.

// brandDrift are primary-descriptor DRIFT lines (fak declared to BE a retired descriptor) — the
// gate MUST flag each. Copied verbatim from check_brand_consistency_test.py DRIFT.
var brandDrift = []string{
	"`fak` is an agent tool firewall: a single Go binary that sits between an agent and its tools",
	"fak — agent tool firewall",
	`fmt.Fprintf(os.Stderr, "fak - Agent Tool Firewall (Fused Agent Kernel, v%s)")`,
	"`fak` is the tool-call policy gateway for your fleet",
	"fak is a tool call policy gateway",
}

// brandAllowed are legitimate secondary uses (synonym lists, "also described as", the named
// asset) — the gate MUST NOT flag any. Copied verbatim from check_brand_consistency_test.py ALLOWED.
var brandAllowed = []string{
	"`fak` is an **agent kernel** (also described as an *agent tool firewall*): an in-process gate",
	"<sub>Topics: agent kernel · agent tool firewall · AI agent security · prompt injection</sub>",
	"  - agent tool firewall",
	`"alternateName": ["the agent kernel", "agent tool firewall"],`,
	`aria-label="fak — the agent tool firewall: a ~44 second explainer reveal">`,
	"`fak` is the Fused Agent Kernel: a single Go binary that sits between an agent and its tools",
	"It is also described as an agent tool firewall.",
}

func TestBrandConsistency_DriftIsFlagged(t *testing.T) {
	for _, s := range brandDrift {
		if !brandLineViolates(s) {
			t.Errorf("should flag primary-descriptor drift: %q", s)
		}
	}
}

func TestBrandConsistency_AllowedNotFlagged(t *testing.T) {
	for _, s := range brandAllowed {
		if brandLineViolates(s) {
			t.Errorf("should NOT flag legitimate secondary use: %q", s)
		}
	}
}

// TestBrandConsistency_FileFilter pins the scan-scope predicate: exempt files/prefixes and
// non-text extensions are skipped; reader-facing text surfaces are scanned. Mirrors audit()'s
// EXEMPT_FILES / EXEMPT_PREFIXES / SCAN_EXT filter.
func TestBrandConsistency_FileFilter(t *testing.T) {
	cases := []struct {
		rel  string
		want bool
	}{
		{"README.md", true},
		{"docs/note.txt", true},
		{"cmd/fak/main.go", true},
		{"index.html", true},
		{"CITATION.cff", true},
		{"tools/check_brand_consistency.py", false},             // exempt file (Python oracle)
		{"tools/gen_structured_data.py", false},                 // exempt file
		{"llms-full.txt", false},                                // exempt file (generated)
		{"internal/hooks/gate_brandconsistency.go", false},      // exempt: this gate's own source
		{"internal/hooks/gate_brandconsistency_test.go", false}, // exempt: this test's golden vectors
		{"visuals/poster.md", false},                            // exempt prefix
		{"tools/foo.py", false},                                 // not a scanned extension
		{"assets/logo.svg", false},                              // not a scanned extension
	}
	for _, c := range cases {
		if got := brandScanned(c.rel); got != c.want {
			t.Errorf("brandScanned(%q) = %v, want %v", c.rel, got, c.want)
		}
	}
}

// TestBrandConsistency_LiveTreeClean asserts the real tracked tree carries no primary-descriptor
// drift — the Go twin of the Python test_live_tree_is_clean. Skipped outside a git checkout.
func TestBrandConsistency_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateBrandConsistencyTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("primary-descriptor drift on the tracked tree: %+v", findings)
	}
}

// TestBrandConsistency_PythonParity is the verdict-level differential: the ported gate and the
// live Python checker must agree (clean vs. violation) over the SAME real tracked tree. The
// Python checker is --audit-tree only and derives its root from __file__, so both sides scan the
// real clone. Skipped under -short or when python/git is absent.
func TestBrandConsistency_PythonParity(t *testing.T) {
	if testing.Short() {
		t.Skip("python parity skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	clone := repoRoot(t)

	tree, err := ReadTrackedTree(clone)
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateBrandConsistencyTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	goBad := len(findings) > 0

	args := append(append([]string{}, pyArgs...), "tools/check_brand_consistency.py", "--audit-tree")
	cmd := exec.Command(py, args...)
	cmd.Dir = clone
	out, _ := cmd.CombinedOutput()
	pyBad := cmd.ProcessState.ExitCode() == 1

	if goBad != pyBad {
		t.Fatalf("VERDICT MISMATCH: go bad=%v (%d findings) vs python bad=%v\npython said: %s\ngo findings: %+v",
			goBad, len(findings), pyBad, out, findings)
	}
}
