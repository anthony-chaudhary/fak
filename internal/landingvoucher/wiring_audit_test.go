package landingvoucher

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// TestWiringAuditContract keeps WIRING-AUDIT.md honest against the tree. It
// recomputes which non-test .go files outside internal/landingvoucher carry the
// package import path, then cross-checks the record's machine-readable verdict
// line against that set: ABSENT requires zero references, WIRED requires at
// least one and every reference listed in the doc body.
//
// fak-test:runtime medium est=8s lane=default
func TestWiringAuditContract(t *testing.T) {
	t.Parallel()

	const importPath = "github.com/anthony-chaudhary/fak/internal/landingvoucher"

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed; cannot derive the repository root")
	}
	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))

	// cmd and internal are required (production entry points and the internal
	// hop space the debt-lanes census reads); the rest are scanned when present
	// so the record's "any non-test .go file" claim stays tree-wide.
	roots := []string{"cmd", "internal", "pkg", "platform", "examples", "tools"}
	for _, name := range roots {
		st, err := os.Stat(filepath.Join(repoRoot, name))
		if err != nil {
			if name == "cmd" || name == "internal" {
				t.Fatalf("required scan root %q is missing (%v); the wiring audit cannot run without it", name, err)
			}
			continue // optional scan roots are skipped when absent
		}
		if !st.IsDir() {
			t.Fatalf("scan root %q exists but is not a directory", name)
		}
	}

	var refs []string
	for _, name := range roots {
		root := filepath.Join(repoRoot, name)
		if _, err := os.Stat(root); err != nil {
			continue
		}
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				switch d.Name() {
				case "testdata", "vendor", ".git", "_scratch":
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			rel, err := filepath.Rel(repoRoot, path)
			if err != nil {
				return err
			}
			rel = filepath.ToSlash(rel)
			if strings.HasPrefix(rel, "internal/landingvoucher/") {
				return nil // the package's own files are not external call paths
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if bytes.Contains(data, []byte(importPath)) {
				refs = append(refs, rel)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk of scan root %q failed: %v", name, err)
		}
	}
	sort.Strings(refs)

	docPath := filepath.Join(repoRoot, "internal", "landingvoucher", "WIRING-AUDIT.md")
	doc, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("cannot read WIRING-AUDIT.md beside this test: %v; the executable contract has no record to check", err)
	}
	body := string(doc)

	if !strings.Contains(body, "debtwave2-internal-landingvoucher") {
		t.Fatal("WIRING-AUDIT.md does not name ticket debtwave2-internal-landingvoucher; restore the ticket meta line")
	}

	var verdicts []string
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "verdict:") {
			verdicts = append(verdicts, strings.TrimSpace(strings.TrimPrefix(line, "verdict:")))
		}
	}
	if len(verdicts) != 1 {
		t.Fatalf("WIRING-AUDIT.md must contain exactly one line starting with 'verdict:', found %d; the executable contract parses a single machine-readable verdict", len(verdicts))
	}
	switch verdicts[0] {
	case "ABSENT":
		if len(refs) != 0 {
			t.Fatalf("record says ABSENT but %d non-test .go file(s) now reference %s: %v. The wiring changed; flip WIRING-AUDIT.md to 'verdict: WIRED' and list these seam files.", len(refs), importPath, refs)
		}
	case "WIRED":
		if len(refs) == 0 {
			t.Fatal("record says WIRED but no non-test .go file outside internal/landingvoucher references the import path; land the cmd/ seam or flip WIRING-AUDIT.md back to 'verdict: ABSENT'")
		}
		for _, ref := range refs {
			if !strings.Contains(body, ref) {
				t.Fatalf("record says WIRED but does not list seam file %q; add it to the seam list in WIRING-AUDIT.md", ref)
			}
		}
	default:
		t.Fatalf("WIRING-AUDIT.md verdict must be ABSENT or WIRED, got %q; fix the machine-readable line", verdicts[0])
	}
}
