package providerjobaccounting

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// wiringAuditImportPath is this package's module path. A non-test .go file
// containing it anywhere else in the root module is a production importer.
const wiringAuditImportPath = "github.com/anthony-chaudhary/fak/internal/providerjobaccounting"

// wiringAuditControlImport is a known production import used as a positive
// control: if the scan cannot find it, the walk scope is broken and the
// absence finding below would be vacuous.
const wiringAuditControlImport = "github.com/anthony-chaudhary/fak/internal/modver"

// wiringAuditEnv opts the audit into a run. The scan reads every non-test .go
// file under the module's Go source roots, a multi-second I/O cost on a
// loaded host, so it stays off the default test lane.
const wiringAuditEnv = "FAK_PROVIDER_JOB_ACCOUNTING_WIRING_AUDIT"

// TestWiringAuditRecordsAbsentProductionCallPath pins the 2026-11-02 READ-ONLY
// wiring-audit finding for debt lane internal/providerjobaccounting (flow
// ticket debtwave2-internal-providerjobaccounting): no production call path
// invokes this package. It is the docs/standards #9575 provider-job
// accounting schema-conformance suite; its only invocation surface is
// `go test ./internal/providerjobaccounting/...`, and the debt-lanes reader's
// runtime-proof entry (`fak version modules`) enumerates modules from git
// history rather than a call graph. The reader therefore reports
// integrated=false and holds the lane at the tested ceiling (maturity 5.0 of
// target 8); lifting that cap needs an import edge from reachable production
// code, i.e. files outside this lane's ticket boundary, so the ABSENT finding
// is recorded here instead of forcing new wiring.
//
// If a real production importer lands later, this test fails on purpose:
// update the record above and drop the assertion.
//
// Any I/O failure during the walk fails the test instead of silently
// narrowing scope: an unreadable subtree could hide exactly the importer
// that would falsify the pinned ABSENT finding (this host has documented
// Defender file-access interference), and the positive control alone only
// detects total scope breakage.
//
// fak-test:runtime medium est=10s lane=optin
// Runtime justification: full read of every non-test .go file under cmd/,
// internal/, pkg/, tools/, examples/ (~8.4k files); roughly 4-15s depending
// on host load.
func TestWiringAuditRecordsAbsentProductionCallPath(t *testing.T) {
	if os.Getenv(wiringAuditEnv) != "1" {
		t.Skipf("set %s=1 to run the repository-wide wiring audit", wiringAuditEnv)
	}
	t.Parallel()

	root := filepath.Join("..", "..")
	selfDir := filepath.Join(root, "internal", "providerjobaccounting")
	var (
		importers  []string
		control    []string
		ioProblems []string
	)
	for _, relRoot := range []string{"cmd", "internal", "pkg", "tools", "examples"} {
		dir := filepath.Join(root, relRoot)
		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				ioProblems = append(ioProblems, fmt.Sprintf("walk %s: %v", path, walkErr))
				return nil
			}
			if d.IsDir() {
				switch d.Name() {
				case ".git", "testdata", "vendor", "node_modules", "_scratch":
					return filepath.SkipDir
				}
				if path == selfDir {
					return filepath.SkipDir // self-reference is not an import
				}
				if info, statErr := os.Stat(filepath.Join(path, "go.mod")); statErr == nil && !info.IsDir() {
					return filepath.SkipDir // nested Go module: separate claim space
				}
				return nil
			}
			name := d.Name()
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				return nil
			}
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				ioProblems = append(ioProblems, fmt.Sprintf("read %s: %v", path, readErr))
				return nil
			}
			rel, _ := filepath.Rel(root, path)
			rel = filepath.ToSlash(rel)
			if bytes.Contains(data, []byte(wiringAuditImportPath)) {
				importers = append(importers, rel)
			}
			if bytes.Contains(data, []byte(wiringAuditControlImport)) {
				control = append(control, rel)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", relRoot, err)
		}
	}
	if len(ioProblems) > 0 {
		t.Fatalf("wiring audit incomplete (%d I/O problem(s)); absence finding not provable: %v", len(ioProblems), ioProblems)
	}

	if len(control) == 0 {
		t.Fatal("positive control failed: no non-test .go file imports internal/modver; the scan scope is broken and the absence finding is vacuous")
	}
	if len(importers) > 0 {
		t.Fatalf("production importer(s) of %s now exist: %v; the 2026-11-02 ABSENT wiring-audit finding is stale - update the record in this test", wiringAuditImportPath, importers)
	}
}
