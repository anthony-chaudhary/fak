package devindex

import (
	"reflect"
	"testing"
)

func TestParseSymbolID(t *testing.T) {
	got, err := ParseSymbolID("github.com/acme/fak/internal/core.Widget")
	if err != nil {
		t.Fatalf("ParseSymbolID: %v", err)
	}
	if got.Package != "github.com/acme/fak/internal/core" || got.Symbol != "Widget" {
		t.Fatalf("ParseSymbolID = %+v, want package path plus Widget", got)
	}
	if got.String() != "github.com/acme/fak/internal/core.Widget" {
		t.Fatalf("String = %q", got.String())
	}
	for _, bad := range []string{"", "Widget", "internal/core.", "internal/core/Widget"} {
		if _, err := ParseSymbolID(bad); err == nil {
			t.Fatalf("ParseSymbolID(%q) succeeded, want error", bad)
		}
	}
}

func TestBlastRadiusDirectAndTransitiveDependents(t *testing.T) {
	target := SymbolID{Package: "example.com/fak/internal/core", Symbol: "Widget"}
	idx := NewReferenceIndex([]PackageImports{
		{
			ImportPath: "example.com/fak/internal/direct",
			Imports:    []string{"example.com/fak/internal/core"},
		},
		{
			ImportPath:  "example.com/fak/internal/testonly",
			TestImports: []string{"example.com/fak/internal/core"},
		},
		{
			ImportPath:  "example.com/fak/internal/far",
			TestImports: []string{"example.com/fak/internal/direct"},
		},
		{
			ImportPath:   "example.com/fak/cmd/app",
			XTestImports: []string{"example.com/fak/internal/far"},
		},
		{
			ImportPath: "example.com/fak/internal/shortcut",
			Imports: []string{
				"example.com/fak/internal/direct",
				"example.com/fak/internal/far",
			},
		},
		{
			ImportPath: "example.com/fak/internal/other",
			Imports:    []string{"example.com/fak/internal/core"},
		},
		{
			ImportPath: "example.com/fak/internal/unrelated",
		},
	}, []SymbolReference{
		{FromPackage: "example.com/fak/internal/direct", Target: target},
		{FromPackage: "example.com/fak/internal/direct", Target: target}, // duplicate ref, one row
		{FromPackage: "example.com/fak/internal/testonly", Target: target, Test: true},
		{FromPackage: "example.com/fak/internal/other", Target: SymbolID{Package: target.Package, Symbol: "Other"}},
		{FromPackage: target.Package, Target: target}, // defining package is not its own dependent
	})

	got := idx.BlastRadius(target).Packages
	want := []BlastRadiusPackage{
		{ImportPath: "example.com/fak/internal/direct", Distance: 1, Direct: true},
		{ImportPath: "example.com/fak/internal/testonly", Distance: 1, Direct: true},
		{ImportPath: "example.com/fak/internal/far", Distance: 2},
		{ImportPath: "example.com/fak/internal/shortcut", Distance: 2},
		{ImportPath: "example.com/fak/cmd/app", Distance: 3},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BlastRadius packages = %+v, want %+v", got, want)
	}
}

func TestBlastRadiusNoReferences(t *testing.T) {
	got := BlastRadius([]PackageImports{
		{ImportPath: "example.com/fak/internal/importer", Imports: []string{"example.com/fak/internal/core"}},
	}, nil, SymbolID{Package: "example.com/fak/internal/core", Symbol: "Widget"})
	if len(got.Packages) != 0 {
		t.Fatalf("BlastRadius with no symbol refs = %+v, want no dependents", got.Packages)
	}
}
