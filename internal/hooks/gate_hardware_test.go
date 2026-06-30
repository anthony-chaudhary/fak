package hooks

import (
	"os/exec"
	"testing"
)

// gate_hardware_test.go — parity for the --audit-tree HARDWARE_TELL gate (gateHardwareTreeTell),
// the whole-tree twin of scrub_hardware_names.py --check. The Python --check lint is tree-only (it
// derives its doc set from `git ls-files *.md`, with no --audit-staged / --root), so the staged
// runParity harness does not reach it. Instead we (1) replay the residual_hits golden vectors from
// tools/scrub_hardware_names_test.py against the shared per-line detector, (2) prove the generated
// docs are excluded and a normal doc is flagged, (3) prove the live tracked tree is clean, and
// (4) run a verdict-level differential against the live Python --check.

// hardwareTreeTells are prose lines that still carry a private hardware name — residualHardwareDocHits
// (the --check detector) MUST flag each. Drawn from scrub_hardware_names_test.py.
var hardwareTreeTells = []string{
	"ran on the DGX box",             // bare uppercase DGX
	"ran on the SXM4 board",          // bare SXM4 SKU
	"we ran the eval on dgx3 today",  // digit-suffixed dgxN node label
	"we ran the eval on da33 today",  // da33 CPU host
	"perf: DGX A100 serve 466 tok/s", // DGX phrase
}

// hardwareTreeClean are identifier / fenced / code-span / FQDN forms residual_hits exempts — the
// detector MUST NOT flag any. Drawn verbatim from scrub_hardware_names_test.py's residual_hits
// must-not-flag vectors.
var hardwareTreeClean = []string{
	"| col | dgx3-control | col |",          // channel name (hyphen boundary)
	"host dgx1.example.lab",                 // FQDN shortname
	"schema dgx3-node-state.v1",             // schema id
	"the da33-control channel",              // da33 channel name
	"host da33.example.lab",                 // da33 FQDN
	"use `dgxbridge` here",                  // code span identifier
	"datacenter server (`dgx3`) is the box", // dgx3 masked inside a code span
	"the FAK_DGX_REQ_ marker",               // underscore identifier
	"see `cmd/dgxbridge` for the bridge",    // path identifier in a code span
}

func TestHardwareTell_TellsFlagged(t *testing.T) {
	for _, s := range hardwareTreeTells {
		if len(residualHardwareDocHits(s)) == 0 {
			t.Errorf("should flag prose hardware tell: %q", s)
		}
	}
}

func TestHardwareTell_CleanNotFlagged(t *testing.T) {
	for _, s := range hardwareTreeClean {
		if hits := residualHardwareDocHits(s); len(hits) != 0 {
			t.Errorf("should NOT flag identifier/code form: %q -> %+v", s, hits)
		}
	}
}

// TestHardwareTell_FencedBlockExempt pins the fence handling: a tell inside a ``` fence is exempt
// (it's a code example), and line numbers are 1-based over the whole content. Mirrors the Python
// test_ignores_code_span_and_fence / test_detects_prose_dgx pair.
func TestHardwareTell_FencedBlockExempt(t *testing.T) {
	if hits := residualHardwareDocHits("```\nDGX\n```"); len(hits) != 0 {
		t.Errorf("fenced DGX must be exempt, got %+v", hits)
	}
	hits := residualHardwareDocHits("intro\nran on the DGX box\nend")
	if len(hits) != 1 || hits[0].Line != 2 {
		t.Errorf("want one hit at line 2, got %+v", hits)
	}
}

// TestHardwareTell_GeneratedDocsExcluded pins the default_doc_set() scope: generated artifacts
// (GENERATED_DOCS, GENERATED_DIR_PREFIXES) carrying a tell are IGNORED; a normal tracked .md is
// flagged. A non-.md file is never scanned.
func TestHardwareTell_GeneratedDocsExcluded(t *testing.T) {
	dir := t.TempDir()
	tell := "ran on the DGX box\n"
	writeFile(t, dir, "docs/bench-plan.md", tell)           // GENERATED_DOCS — skipped
	writeFile(t, dir, "docs/industry-scorecard/x.md", tell) // GENERATED_DIR_PREFIX — skipped
	writeFile(t, dir, "notes.txt", tell)                    // not .md — skipped
	writeFile(t, dir, "docs/notes/real.md", tell)           // normal doc — flagged
	tree := &TrackedTree{
		Root: dir,
		Paths: []string{
			"docs/bench-plan.md",
			"docs/industry-scorecard/x.md",
			"notes.txt",
			"docs/notes/real.md",
		},
		fileCache: map[string]fileEntry{},
	}
	findings, err := gateHardwareTreeTell(tree)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 || findings[0].File != "docs/notes/real.md" {
		t.Fatalf("want exactly the non-generated doc flagged, got %+v", findings)
	}
}

// TestHardwareTell_LiveTreeClean asserts the real tracked tree carries no prose hardware tell — the
// Go twin of `make hygiene`'s passing `scrub_hardware_names.py --check`. Skipped outside a checkout.
func TestHardwareTell_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateHardwareTreeTell(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("prose hardware tells on the tracked tree: %+v", findings)
	}
}

// TestHardwareTell_PythonParity is the verdict-level differential: the ported gate and the live
// Python `scrub_hardware_names.py --check` must agree (clean vs. violation) over the SAME real
// tracked tree. The Python checker derives its root from __file__, so both sides scan the real
// clone. Skipped under -short or when python/git is absent.
func TestHardwareTell_PythonParity(t *testing.T) {
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
	findings, gerr := gateHardwareTreeTell(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	goBad := len(findings) > 0

	args := append(append([]string{}, pyArgs...), "tools/scrub_hardware_names.py", "--check")
	cmd := exec.Command(py, args...)
	cmd.Dir = clone
	out, _ := cmd.CombinedOutput()
	pyBad := cmd.ProcessState.ExitCode() == 1

	if goBad != pyBad {
		t.Fatalf("VERDICT MISMATCH: go bad=%v (%d findings) vs python bad=%v\npython said: %s\ngo findings: %+v",
			goBad, len(findings), pyBad, out, findings)
	}
}
