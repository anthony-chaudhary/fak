package main

import (
	"reflect"
	"testing"
)

func TestPackageDirsWithTrackedSource(t *testing.T) {
	got := packageDirsWithTrackedSource([]string{
		"cmd/fak/main.go",
		"cmd/fak/main_test.go",       // test file: does not make its dir buildable on its own
		"internal/loopfleet/loop.go", // real source
		"internal/loopfleet/loop_test.go",
		"internal/onlytest/x_test.go", // ONLY a test file -> dir is NOT buildable
		"README.md",                   // non-go: ignored
		"internal\\winpath\\w.go",     // backslash path normalizes
		"  internal/space/s.go  ",     // surrounding whitespace trimmed
	})
	want := map[string]bool{
		"cmd/fak":            true,
		"internal/loopfleet": true,
		"internal/winpath":   true,
		"internal/space":     true,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("packageDirsWithTrackedSource:\n got %v\nwant %v", got, want)
	}
	if got["internal/onlytest"] {
		t.Errorf("a dir with only _test.go files must NOT count as buildable source")
	}
}

func TestParseImportEdges(t *testing.T) {
	// Shape: <rev>:<file>:<lineno>:<content>
	grep := "" +
		"HEAD:cmd/fak/rollup.go:38:\t\"github.com/anthony-chaudhary/fak/internal/loopfleet\"\n" +
		"HEAD:cmd/fak/rollup.go:39:\tmat \"github.com/anthony-chaudhary/fak/internal/maturity\"\n" + // named import
		"HEAD:cmd/fak/blank.go:7:\t_ \"github.com/anthony-chaudhary/fak/internal/driver\"\n" + // blank import
		"HEAD:cmd/fak/rollup.go:38:\t\"github.com/anthony-chaudhary/fak/internal/loopfleet\"\n" + // duplicate -> collapsed
		"HEAD:internal/x/use.go:12:\tpath := \"github.com/anthony-chaudhary/fak/internal/loopfleet\" // a string literal, NOT an import\n" +
		"HEAD:internal/x/call.go:20:\tfoo(\"github.com/anthony-chaudhary/fak/internal/loopfleet\")\n" + // call arg, not an import
		"HEAD:cmd/fak/ext.go:5:\t\"github.com/spf13/cobra\"\n" + // non-module import (no prefix match anyway)
		"HEAD:cmd/fak/rollup_test.go:9:\t\"github.com/anthony-chaudhary/fak/internal/testonly\"\n" + // _test.go importer excluded
		"HEAD:cmd/fak/std.go:3:\t\"strings\"\n" // stdlib

	got := parseImportEdges(grep)
	want := []importEdge{
		{Importer: "cmd/fak/blank.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/driver", PkgDir: "internal/driver"},
		{Importer: "cmd/fak/rollup.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/loopfleet", PkgDir: "internal/loopfleet"},
		{Importer: "cmd/fak/rollup.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/maturity", PkgDir: "internal/maturity"},
	}
	// parseImportEdges preserves first-seen order; sort-independent compare via a set keyed by importer+path.
	if len(got) != len(want) {
		t.Fatalf("edge count: got %d %+v, want %d", len(got), got, len(want))
	}
	gotSet := map[string]string{}
	for _, e := range got {
		gotSet[e.Importer+"|"+e.ImportPath] = e.PkgDir
	}
	for _, w := range want {
		if pd, ok := gotSet[w.Importer+"|"+w.ImportPath]; !ok {
			t.Errorf("missing expected edge %s -> %s", w.Importer, w.ImportPath)
		} else if pd != w.PkgDir {
			t.Errorf("edge %s -> %s: PkgDir got %q want %q", w.Importer, w.ImportPath, pd, w.PkgDir)
		}
	}
	for _, e := range got {
		if e.Importer == "internal/x/use.go" || e.Importer == "internal/x/call.go" {
			t.Errorf("string-literal / call line must not be parsed as an import: %+v", e)
		}
		if e.PkgDir == "internal/testonly" {
			t.Errorf("_test.go importer must be excluded (does not red go build ./...): %+v", e)
		}
	}
}

func TestDetectUncommittedImports(t *testing.T) {
	edges := []importEdge{
		{Importer: "cmd/fak/rollup.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/loopfleet", PkgDir: "internal/loopfleet"},
		{Importer: "cmd/fak/z.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/loopfleet", PkgDir: "internal/loopfleet"},
		{Importer: "cmd/fak/a.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/gateway", PkgDir: "internal/gateway"}, // has source -> fine
		{Importer: "cmd/fak/b.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/maturity", PkgDir: "internal/maturity"},
	}
	pkgDirs := map[string]bool{
		"internal/gateway": true, // committed
		// internal/loopfleet and internal/maturity: NOT committed -> violations
	}
	got := detectUncommittedImports(edges, pkgDirs)
	want := []uncommittedImport{
		// sorted by (PkgDir, Importer): gateway is fine; loopfleet (rollup.go, z.go), then maturity
		{Importer: "cmd/fak/rollup.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/loopfleet", PkgDir: "internal/loopfleet"},
		{Importer: "cmd/fak/z.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/loopfleet", PkgDir: "internal/loopfleet"},
		{Importer: "cmd/fak/b.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/maturity", PkgDir: "internal/maturity"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("detectUncommittedImports:\n got %+v\nwant %+v", got, want)
	}
}

func TestDetectUncommittedImportsAllClean(t *testing.T) {
	edges := []importEdge{
		{Importer: "cmd/fak/a.go", ImportPath: "github.com/anthony-chaudhary/fak/internal/gateway", PkgDir: "internal/gateway"},
	}
	got := detectUncommittedImports(edges, map[string]bool{"internal/gateway": true})
	if len(got) != 0 {
		t.Fatalf("expected no violations when every import has committed source, got %+v", got)
	}
	if importWitnessVerdict(got) != "OK" {
		t.Errorf("verdict for zero violations should be OK, got %q", importWitnessVerdict(got))
	}
}

func TestImportWitnessVerdict(t *testing.T) {
	if v := importWitnessVerdict(nil); v != "OK" {
		t.Errorf("nil -> OK, got %q", v)
	}
	if v := importWitnessVerdict([]uncommittedImport{{Importer: "a.go", PkgDir: "internal/x"}}); v != "IMPORT_OF_UNCOMMITTED_PACKAGE" {
		t.Errorf("non-empty -> IMPORT_OF_UNCOMMITTED_PACKAGE, got %q", v)
	}
}
