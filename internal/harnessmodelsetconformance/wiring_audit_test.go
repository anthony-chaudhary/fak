package harnessmodelsetconformance_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const laneImportPath = "github.com/anthony-chaudhary/fak/internal/harnessmodelsetconformance"

// 2026-11-02 debtwave2 wiring audit (ticket:
// docs/tickets/debtwave2-internal-harnessmodelsetconformance): finding is
// ABSENT-with-reason - no production call path invokes this evidence-boundary
// lane. Zero non-test files in the repo import it, and the lane ships no
// production symbols (doc.go only). A minimal connection would require new
// production API plus a cmd/fak import, both outside this lane's edit
// boundary, so the audit lands as durable evidence instead of forced wiring.
// The witnessed production surface is the `fak harness model-set` verb in
// cmd/fak/harness_modelset.go; its five internal leaf imports are re-checked
// on every run so the surface cannot silently drift away from this witness.

// fak-test:runtime medium est=22000ms lane=default
func TestWiringAuditNoProductionCallPath(t *testing.T) {
	// The package sits two levels under the module root (internal/<pkg>).
	root := filepath.Join("..", "..")
	assertNoProductionImporters(t, root)
	assertNothingCallableExported(t)
	assertWitnessedSurfaceAlive(t, root)
}

// assertNoProductionImporters fails if any non-test .go file under the repo
// roots cmd, internal, pkg, platform imports the lane. _test.go files are
// excluded because an external test package may import the lane under test
// without that edge being a production call path. Unparseable candidate
// files are skipped rather than failed: the shared trunk is permanently
// peer-dirty, and a syntactically broken sibling is WIP that cannot be proven
// an importer.
func assertNoProductionImporters(t *testing.T, root string) {
	t.Helper()
	var offenders []string
	for _, rel := range []string{"cmd", "internal", "pkg", "platform"} {
		dir := filepath.Join(root, rel)
		if _, err := os.Stat(dir); err != nil {
			continue // an absent root holds no files; nothing to prove
		}
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() {
				switch {
				case strings.HasPrefix(info.Name(), "."),
					info.Name() == "testdata",
					info.Name() == "vendor",
					info.Name() == "_scratch":
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(info.Name(), ".go") || strings.HasSuffix(info.Name(), "_test.go") {
				return nil
			}
			content, err := os.ReadFile(path)
			if err != nil {
				return nil
			}
			// Cheap pre-filter before parsing: most files never mention the lane.
			if !strings.Contains(string(content), laneImportPath) {
				return nil
			}
			fset := token.NewFileSet()
			parsed, err := parser.ParseFile(fset, path, string(content), parser.ImportsOnly)
			if err != nil {
				return nil
			}
			for _, imp := range parsed.Imports {
				if strings.Trim(imp.Path.Value, `"`) == laneImportPath {
					offenders = append(offenders, path)
				}
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", rel, err)
		}
	}
	if len(offenders) > 0 {
		t.Fatalf("production importers of %s exist: %s", laneImportPath, strings.Join(offenders, ", "))
	}
}

// assertNothingCallableExported pins the lane to a doc-only package: exactly
// one non-test .go file, named doc.go, declaring no functions or types. Any
// exported callable would be a production symbol this audit says does not
// exist yet.
func assertNothingCallableExported(t *testing.T) {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	var nonTest []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
			nonTest = append(nonTest, name)
		}
	}
	if len(nonTest) != 1 || nonTest[0] != "doc.go" {
		t.Fatalf("expected exactly one non-test .go file named doc.go, got %v", nonTest)
	}
	src, err := os.ReadFile("doc.go")
	if err != nil {
		t.Fatalf("read doc.go: %v", err)
	}
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, "doc.go", string(src), 0)
	if err != nil {
		t.Fatalf("parse doc.go: %v", err)
	}
	var funcs, types int
	for _, d := range parsed.Decls {
		switch decl := d.(type) {
		case *ast.FuncDecl:
			funcs++
		case *ast.GenDecl:
			if decl.Tok == token.TYPE {
				types += len(decl.Specs)
			}
		}
	}
	if funcs != 0 || types != 0 {
		t.Fatalf("doc.go declares %d FuncDecl and %d TypeDecl; expected zero of each", funcs, types)
	}
}

// assertWitnessedSurfaceAlive proves the production surface this lane
// witnesses still exists and still composes the five model-set leaves.
func assertWitnessedSurfaceAlive(t *testing.T, root string) {
	t.Helper()
	surface := filepath.Join(root, "cmd", "fak", "harness_modelset.go")
	src, err := os.ReadFile(surface)
	if err != nil {
		t.Fatalf("read %s: %v", surface, err)
	}
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, surface, string(src), parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", surface, err)
	}
	have := make(map[string]bool, len(parsed.Imports))
	for _, imp := range parsed.Imports {
		have[strings.Trim(imp.Path.Value, `"`)] = true
	}
	for _, want := range []string{
		"github.com/anthony-chaudhary/fak/internal/harnessmodelset",
		"github.com/anthony-chaudhary/fak/internal/modelinventory",
		"github.com/anthony-chaudhary/fak/internal/modelsetlock",
		"github.com/anthony-chaudhary/fak/internal/modelsetreceipt",
		"github.com/anthony-chaudhary/fak/internal/modelsetresolve",
	} {
		if !have[want] {
			t.Errorf("%s no longer imports %q; the witnessed surface has drifted", surface, want)
		}
	}
}
