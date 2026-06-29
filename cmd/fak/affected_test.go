package main

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
)

// goListObj mirrors the JSON shape `go list -json` emits for the fields parseGoList
// reads. Marshalling real structs (rather than hand-writing JSON) keeps the fixture's
// Windows backslash paths correctly escaped.
type goListObj struct {
	ImportPath   string
	Dir          string
	Module       *struct{ Path, Dir string }
	GoFiles      []string `json:",omitempty"`
	TestGoFiles  []string `json:",omitempty"`
	EmbedFiles   []string `json:",omitempty"`
	Imports      []string `json:",omitempty"`
	TestImports  []string `json:",omitempty"`
	XTestImports []string `json:",omitempty"`
}

func TestParseGoListAndSelectEndToEnd(t *testing.T) {
	modDir := filepath.FromSlash("/work/m")
	mod := &struct{ Path, Dir string }{Path: "example.com/m", Dir: modDir}
	objs := []goListObj{
		{ImportPath: "example.com/m", Dir: modDir, Module: mod,
			GoFiles: []string{"main.go"},
			Imports: []string{"example.com/m/internal/foo"}},
		{ImportPath: "example.com/m/internal/foo", Dir: filepath.Join(modDir, "internal", "foo"), Module: mod,
			GoFiles: []string{"foo.go"},
			Imports: []string{"fmt"}}, // stdlib import must be filtered out of edges
		{ImportPath: "example.com/m/internal/bar", Dir: filepath.Join(modDir, "internal", "bar"), Module: mod,
			GoFiles:     []string{"bar.go"},
			TestGoFiles: []string{"bar_test.go"},
			TestImports: []string{"example.com/m/internal/foo"}}, // bar's TEST imports foo
	}
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	for _, o := range objs {
		if err := enc.Encode(o); err != nil {
			t.Fatal(err)
		}
	}

	fileToPkg, edges, total, err := parseGoList(strings.NewReader(sb.String()))
	if err != nil {
		t.Fatal(err)
	}
	if total != 3 {
		t.Fatalf("total = %d, want 3", total)
	}
	wantFiles := map[string]string{
		"main.go":                  "example.com/m",
		"internal/foo/foo.go":      "example.com/m/internal/foo",
		"internal/bar/bar.go":      "example.com/m/internal/bar",
		"internal/bar/bar_test.go": "example.com/m/internal/bar",
	}
	if !reflect.DeepEqual(fileToPkg, wantFiles) {
		t.Fatalf("fileToPkg = %v, want %v", fileToPkg, wantFiles)
	}
	// foo's only import was stdlib (fmt), so it has no intra-module edge.
	if _, ok := edges["example.com/m/internal/foo"]; ok {
		t.Errorf("foo should have no intra-module edges, got %v", edges["example.com/m/internal/foo"])
	}
	// bar's TEST import of foo must be recorded as an edge (the test-import correctness case).
	if got := edges["example.com/m/internal/bar"]; !reflect.DeepEqual(got, []string{"example.com/m/internal/foo"}) {
		t.Errorf("bar edges = %v, want [foo]", got)
	}

	// End to end: change a file in foo -> foo + everything importing it (root m imports
	// foo; bar's test imports foo).
	changed := affectedtests.ChangedPackages(fileToPkg, []string{"internal/foo/foo.go"})
	if !reflect.DeepEqual(changed, []string{"example.com/m/internal/foo"}) {
		t.Fatalf("changed = %v, want [foo]", changed)
	}
	selected := affectedtests.Select(edges, changed)
	want := []string{"example.com/m", "example.com/m/internal/bar", "example.com/m/internal/foo"}
	if !reflect.DeepEqual(selected, want) {
		t.Fatalf("selected = %v, want %v", selected, want)
	}

	// A top-level non-source change (Makefile / go.mod) selects nothing -- the root-package
	// over-selection guard. main.go (a real root source file) selects just the root.
	if got := affectedtests.ChangedPackages(fileToPkg, []string{"Makefile", "go.mod"}); len(got) != 0 {
		t.Fatalf("non-source change selected %v, want empty", got)
	}
	if got := affectedtests.ChangedPackages(fileToPkg, []string{"main.go"}); !reflect.DeepEqual(got, []string{"example.com/m"}) {
		t.Fatalf("root source change selected %v, want [example.com/m]", got)
	}

	// A docs-only change selects nothing end to end.
	docChanged := affectedtests.ChangedPackages(fileToPkg, []string{"docs/x.md"})
	if len(docChanged) != 0 {
		t.Fatalf("docs-only change selected %v, want empty", docChanged)
	}
	if got := affectedtests.Select(edges, docChanged); len(got) != 0 {
		t.Fatalf("docs-only selection = %v, want empty", got)
	}
}
