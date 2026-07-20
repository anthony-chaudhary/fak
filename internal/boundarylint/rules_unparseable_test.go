package boundarylint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeGo drops name into dir with the given source, failing the test on an IO error.
func writeGo(t *testing.T, dir, name, src string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return path
}

// brokenSrc is a fragment go/parser cannot read: the function body never closes.
const brokenSrc = "package broken\n\nfunc Broken() {\n\tif true {\n"

// TestScanUnparseableSurfacesBrokenSource is the tell this witness exists for: a file
// the scanner cannot parse must come back as a recorded UNPARSEABLE_SOURCE skip, not as
// the silent zero-findings pass that reads as "clean" (#4070).
func TestScanUnparseableSurfacesBrokenSource(t *testing.T) {
	dir := t.TempDir()
	writeGo(t, dir, "broken.go", brokenSrc)

	findings, err := ScanUnparseable([]string{dir})
	if err != nil {
		t.Fatalf("ScanUnparseable: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("got %d findings, want 1: %v", len(findings), findings)
	}
	f := findings[0]
	if f.Code != CodeUnparseableSource {
		t.Errorf("Code = %q, want %q", f.Code, CodeUnparseableSource)
	}
	if !strings.HasSuffix(f.File, "broken.go") {
		t.Errorf("File = %q, want a path ending in broken.go", f.File)
	}
	if f.Line <= 0 {
		t.Errorf("Line = %d, want the first syntax error's line (>0)", f.Line)
	}
	if f.Detail == "" {
		t.Error("Detail is empty; a recorded skip must say why it was skipped")
	}
}

// TestScanUnparseableIgnoresValidSource pins the other half: a tree that parses cleanly
// reports nothing, so the witness cannot manufacture a skip out of healthy source.
func TestScanUnparseableIgnoresValidSource(t *testing.T) {
	dir := t.TempDir()
	writeGo(t, dir, "ok.go", "package ok\n\nfunc OK() bool { return true }\n")
	writeGo(t, dir, "ok_test.go", "package ok\n\nimport \"testing\"\n\nfunc TestOK(t *testing.T) {}\n")

	findings, err := ScanUnparseable([]string{dir})
	if err != nil {
		t.Fatalf("ScanUnparseable: %v", err)
	}
	if len(findings) != 0 {
		t.Errorf("got %d findings over valid source, want 0: %v", len(findings), findings)
	}
}

// TestScanUnparseableCoversTestFiles proves the walk spans BOTH file classes. Scan skips
// _test.go and ScanTests skips production files, so an unparseable _test.go would fall
// through every other walk in this package — exactly the blind spot being closed.
func TestScanUnparseableCoversTestFiles(t *testing.T) {
	dir := t.TempDir()
	writeGo(t, dir, "broken_test.go", brokenSrc)

	findings, err := ScanUnparseable([]string{dir})
	if err != nil {
		t.Fatalf("ScanUnparseable: %v", err)
	}
	if len(findings) != 1 || findings[0].Code != CodeUnparseableSource {
		t.Fatalf("unparseable _test.go not surfaced: %v", findings)
	}
}

// TestGatingScansStayFailOpen is the load-bearing separation: the gating walks must NOT
// inherit the new finding. Scan/ScanTests feed TestBoundaryPolicy, which errors on every
// finding, so emitting UNPARSEABLE_SOURCE there would red the suite for the whole fleet
// whenever a peer has a half-written file open on this shared trunk.
func TestGatingScansStayFailOpen(t *testing.T) {
	dir := t.TempDir()
	writeGo(t, dir, "broken.go", brokenSrc)
	writeGo(t, dir, "broken_test.go", brokenSrc)

	findings, err := Scan([]string{dir}, DefaultRules())
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(findings) != 0 {
		t.Errorf("Scan emitted %d findings over unparseable source, want 0 (the gate must stay fail-open): %v", len(findings), findings)
	}

	testFindings, err := ScanTests([]string{dir}, DefaultTestRules())
	if err != nil {
		t.Fatalf("ScanTests: %v", err)
	}
	if len(testFindings) != 0 {
		t.Errorf("ScanTests emitted %d findings over unparseable source, want 0: %v", len(testFindings), testFindings)
	}
}

// TestCatalogDocumentsUnparseableSource keeps the code and the documented policy bound
// together, the same contract TestCatalogCoversEnforcedRules holds for enforced rules:
// UNPARSEABLE_SOURCE must be cataloged, and cataloged as SOFT rather than a gate.
func TestCatalogDocumentsUnparseableSource(t *testing.T) {
	for _, e := range Catalog {
		if e.Code != CodeUnparseableSource {
			continue
		}
		if e.Status != StatusSoft {
			t.Errorf("%s cataloged as %q, want %q — it must never gate the build", e.Code, e.Status, StatusSoft)
		}
		return
	}
	t.Errorf("%s has no catalog entry", CodeUnparseableSource)
}
