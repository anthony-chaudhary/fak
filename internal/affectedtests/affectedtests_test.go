package affectedtests

import (
	"reflect"
	"testing"
)

func eq(t *testing.T, got, want []string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestSelectEmptyChangedIsEmpty(t *testing.T) {
	edges := map[string][]string{"a": {"b"}, "b": nil}
	eq(t, Select(edges, nil), []string{})
	eq(t, Select(edges, []string{}), []string{})
}

func TestSelectChangedWithNoImportersIsJustItself(t *testing.T) {
	// a imports b; change a (a leaf importer) -> only a is affected.
	edges := map[string][]string{"a": {"b"}}
	eq(t, Select(edges, []string{"a"}), []string{"a"})
}

func TestSelectChangedLeafPullsItsAncestors(t *testing.T) {
	// a -> b -> c (a imports b imports c). Changing c affects a, b, c.
	edges := map[string][]string{"a": {"b"}, "b": {"c"}}
	eq(t, Select(edges, []string{"c"}), []string{"a", "b", "c"})
	// Changing b affects a and b, but not c.
	eq(t, Select(edges, []string{"b"}), []string{"a", "b"})
	// Changing a affects only a.
	eq(t, Select(edges, []string{"a"}), []string{"a"})
}

func TestSelectDiamond(t *testing.T) {
	// a imports b and c; both b and c import d. Change d -> all four.
	edges := map[string][]string{"a": {"b", "c"}, "b": {"d"}, "c": {"d"}}
	eq(t, Select(edges, []string{"d"}), []string{"a", "b", "c", "d"})
}

func TestSelectCycleTerminates(t *testing.T) {
	// Go forbids real import cycles, but a robust closure must not loop on one.
	edges := map[string][]string{"a": {"b"}, "b": {"a"}}
	eq(t, Select(edges, []string{"a"}), []string{"a", "b"})
}

func TestSelectChangedNodeNotInEdges(t *testing.T) {
	// A changed package that imports nothing intra-module and is imported by nothing
	// (never appears in edges) still selects itself.
	edges := map[string][]string{"a": {"b"}}
	eq(t, Select(edges, []string{"z"}), []string{"z"})
}

func TestSelectMultipleChangedUnions(t *testing.T) {
	// x -> y ; p -> q. Change y and q -> {p,q,x,y}.
	edges := map[string][]string{"x": {"y"}, "p": {"q"}}
	eq(t, Select(edges, []string{"y", "q"}), []string{"p", "q", "x", "y"})
}

func TestSelectTestOnlyEdgeIsHonored(t *testing.T) {
	// The shell folds TestImports into edges, so an importer that only depends on the
	// changed package through its _test.go is still selected -- modeled here by the
	// edge simply being present.
	edges := map[string][]string{"harness": {"target"}} // harness's test imports target
	eq(t, Select(edges, []string{"target"}), []string{"harness", "target"})
}

func TestSelectIsDeterministicAndSorted(t *testing.T) {
	edges := map[string][]string{
		"m": {"k"}, "b": {"k"}, "z": {"k"}, "a": {"k"},
	}
	got := Select(edges, []string{"k"})
	want := []string{"a", "b", "k", "m", "z"}
	eq(t, got, want)
	// Re-run: identical.
	eq(t, Select(edges, []string{"k"}), want)
	// Output is sorted.
	for i := 1; i < len(got); i++ {
		if got[i-1] >= got[i] {
			t.Fatalf("not sorted: %v", got)
		}
	}
}

func TestSelectDuplicateChangedDoesNotDuplicate(t *testing.T) {
	edges := map[string][]string{"a": {"b"}}
	eq(t, Select(edges, []string{"b", "b", "a"}), []string{"a", "b"})
}

func TestChangedPackagesMapsFilesToPackages(t *testing.T) {
	// fileToPkg keys are the actual source files of each package (the shell builds this
	// from go list's GoFiles/TestGoFiles/... lists).
	fileToPkg := map[string]string{
		"internal/foo/foo.go":      "mod/internal/foo",
		"internal/foo/foo_test.go": "mod/internal/foo", // same package, must dedup
		"internal/bar/bar.go":      "mod/internal/bar",
	}
	files := []string{"internal/foo/foo.go", "internal/foo/foo_test.go", "internal/bar/bar.go"}
	eq(t, ChangedPackages(fileToPkg, files), []string{"mod/internal/bar", "mod/internal/foo"})
}

func TestChangedPackagesSkipsNonSourceFiles(t *testing.T) {
	fileToPkg := map[string]string{"internal/foo/a.go": "mod/internal/foo"}
	files := []string{
		"docs/notes/x.md",   // not a source file
		"tools/y.py",        // not a source file
		"README.md",         // top-level, not a source file
		"internal/foo/a.go", // the one real source file
	}
	eq(t, ChangedPackages(fileToPkg, files), []string{"mod/internal/foo"})
}

func TestChangedPackagesEmptyWhenNoCodeChanged(t *testing.T) {
	fileToPkg := map[string]string{"internal/foo/a.go": "mod/internal/foo"}
	files := []string{"docs/a.md", "Makefile"}
	eq(t, ChangedPackages(fileToPkg, files), []string{})
}

func TestChangedPackagesTopLevelNonSourceFileSelectsNothing(t *testing.T) {
	// The root bug guard: with a root package present, a top-level Makefile / README must
	// NOT drag in the root package -- only a real root source file does.
	fileToPkg := map[string]string{"root.go": "mod"}
	eq(t, ChangedPackages(fileToPkg, []string{"Makefile", "go.mod"}), []string{})
	eq(t, ChangedPackages(fileToPkg, []string{"root.go"}), []string{"mod"})
}

func TestChangedPackagesEmbeddedDataFileSelectsItsPackage(t *testing.T) {
	// A changed non-Go file that IS one of the package's embed files (a //go:embed asset)
	// correctly marks that package changed, because the shell put it in fileToPkg.
	fileToPkg := map[string]string{"internal/tmpl/page.html": "mod/internal/tmpl"}
	eq(t, ChangedPackages(fileToPkg, []string{"internal/tmpl/page.html"}), []string{"mod/internal/tmpl"})
}
