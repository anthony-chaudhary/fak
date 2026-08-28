package devindex

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGeneratedDevHandoffManifestIsCurrent(t *testing.T) {
	root := repoRootForSurface(t)
	if os.Getenv("FAK_UPDATE_DEVHANDOFF_MANIFEST") == "1" {
		if err := WriteDevHandoffManifest(root); err != nil {
			t.Fatal(err)
		}
	}
	want, err := RenderDevHandoffManifest(root)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, filepath.FromSlash(DevHandoffManifestPath))
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s drifted; run `fak-dev index ownership --write-manifest --root %s`", DevHandoffManifestPath, root)
	}
}

func TestStudyOperationsAreDevOwnedAndProductStudyStaysRuntime(t *testing.T) {
	rows, err := ExtractDevSourceOwnership(repoRootForSurface(t))
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]SourceOwnership{}
	for _, row := range rows {
		byName[row.Name] = row
		for _, alias := range row.Aliases {
			byName[alias] = row
		}
	}
	for _, name := range []string{
		"study-monitor", "study-inventory", "study-forge", "study-classify",
		"study-link", "study-priority", "study-tickets", "study-adjacency",
		"idea-scout", "borrow-provenance", "customization-index",
	} {
		row, ok := byName[name]
		if !ok {
			t.Errorf("missing generated ownership for %s", name)
			continue
		}
		if row.Owner != OwnerDev || row.Class != SourceDevOnly || row.DispatchTarget != "fak-dev" || row.Handler == "" || !strings.HasPrefix(row.SourceOrigin, "internal/devcmd/") {
			t.Errorf("%s ownership = %+v", name, row)
		}
	}
	if _, ok := byName["study"]; ok {
		t.Fatal("product study command entered the dev-only handoff manifest")
	}
	if tier, ok := TierOf("study"); ok && tier == TierDev {
		t.Fatalf("study tier = %q; product command must remain runtime-owned", tier)
	}
}

func TestSourceHazardsFailClosed(t *testing.T) {
	for name, source := range map[string]string{
		"init":       "package p\nfunc init() {}\n",
		"reflection": "package p\nimport \"reflect\"\nvar _ = reflect.TypeOf(1)\n",
		"cgo":        "package p\nimport \"C\"\n",
		"linkname":   "package p\n//go:linkname x y\nfunc x()\n",
		"embed":      "package p\nimport _ \"embed\"\n//go:embed x\nvar x string\n",
		"self-exec":  "package p\nimport \"os\"\nfunc f(){ _, _ = os.Executable() }\n",
	} {
		t.Run(name, func(t *testing.T) {
			file, err := parser.ParseFile(token.NewFileSet(), name+".go", source, parser.ParseComments)
			if err != nil {
				t.Fatal(err)
			}
			if hazards := sourceHazards(file); len(hazards) == 0 {
				t.Fatal("hazardous source classified as safe")
			}
		})
	}
}

func TestReachableComponentHazardsFailClosedOnSiblingFile(t *testing.T) {
	pkg := parseOwnershipTestPackage(t, map[string]string{
		"handler.go": "package p\nfunc Handler() { helper() }\n",
		"sibling.go": "package p\nimport \"os\"\nfunc helper() { _, _ = os.Executable() }\n",
	})
	hazards := reachableComponentHazards(pkg, "Handler")
	if got := strings.Join(hazards, ", "); !strings.Contains(got, "self-exec (p/sibling.go)") {
		t.Fatalf("reachable sibling hazard was not classified fail-closed: %v", hazards)
	}
}

func TestRuntimeExtractionCandidateFailsClosedOnReachableSibling(t *testing.T) {
	pkg := parseOwnershipTestPackage(t, map[string]string{
		"handler.go": "package p\nfunc Handler() { helper() }\n",
		"sibling.go": "package p\nimport \"os\"\nfunc helper() { _, _ = os.Executable() }\n",
	})
	if class := classifyExtractionCandidate(pkg, "Handler", "cmd/fak/handler.go", OwnerRuntime); class != SourceHazardous {
		t.Fatalf("runtime extraction candidate class = %q, want %q", class, SourceHazardous)
	}
}

func TestAlreadyDevOwnedEmbeddedCommandRemainsManifestable(t *testing.T) {
	pkg := parseOwnershipTestPackage(t, map[string]string{
		"handler.go": "package p\nimport _ \"embed\"\n//go:embed policy.json\nvar policy string\nfunc Handler() {}\n",
	})
	if hazards := reachableComponentHazards(pkg, "Handler"); len(hazards) == 0 {
		t.Fatal("fixture lost its embed hazard")
	}
	if class := classifyExtractionCandidate(pkg, "Handler", "internal/devcmd/handler.go", OwnerDev); class != SourceDevOnly {
		t.Fatalf("already-dev-owned command class = %q, want %q", class, SourceDevOnly)
	}
}

func TestReachableComponentStopsAtImportedLeafPackage(t *testing.T) {
	pkg := parseOwnershipTestPackage(t, map[string]string{
		"handler.go": "package p\nimport \"example.test/leaf\"\nfunc Handler() { leaf.Run() }\n",
	})
	if hazards := reachableComponentHazards(pkg, "Handler"); len(hazards) != 0 {
		t.Fatalf("imported leaf internals must not be classified as same-package source: %v", hazards)
	}
}

func parseOwnershipTestPackage(t *testing.T, sources map[string]string) *vsPkg {
	t.Helper()
	pkg := &vsPkg{
		fset:   token.NewFileSet(),
		funcs:  map[string]*ast.FuncDecl{},
		fileOf: map[string]*ast.File{},
		pathOf: map[*ast.File]string{},
	}
	for name, source := range sources {
		file, err := parser.ParseFile(pkg.fset, name, source, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		pkg.files = append(pkg.files, file)
		pkg.pathOf[file] = "p/" + name
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Body == nil {
				continue
			}
			pkg.funcs[fn.Name.Name] = fn
			pkg.fileOf[fn.Name.Name] = file
		}
	}
	return pkg
}
