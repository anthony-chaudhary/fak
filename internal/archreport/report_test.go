package archreport

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAnalyzeReportsUpwardEdgeAndLegalReverse(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"primitive":1,"composite":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	write("internal/abi/abi.go", "package abi\n")
	write("internal/primitive/primitive.go", `package primitive
import _ "github.com/anthony-chaudhary/fak/internal/composite"
`)
	write("internal/composite/composite.go", `package composite
import _ "github.com/anthony-chaudhary/fak/internal/primitive"
`)
	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if r.Violations != 1 {
		t.Fatalf("violations=%d report=%+v", r.Violations, r)
	}
	if len(r.Leaves) != 3 {
		t.Fatalf("leaves=%d", len(r.Leaves))
	}
	var p, c Leaf
	for _, l := range r.Leaves {
		if l.Name == "primitive" {
			p = l
		}
		if l.Name == "composite" {
			c = l
		}
	}
	if len(p.Violations) != 1 || p.ImportFloor != 2 {
		t.Fatalf("primitive=%+v", p)
	}
	if len(c.Violations) != 0 || c.ImportFloor != 1 {
		t.Fatalf("composite=%+v", c)
	}
}

func TestAnalyzeScopesLeaf(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "architest"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "architest", "architest_test.go"), []byte(`package architest
var tier=map[string]int{"architest":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`), 0644); err != nil {
		t.Fatal(err)
	}
	r, err := Analyze(root, "architest")
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Leaves) != 1 || r.Leaves[0].Name != "architest" || r.Leaves[0].ImportFloor != 1 {
		t.Fatalf("%+v", r)
	}
}

func TestAnalyzeDerivesDirectDependentsAndRanksHotspots(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"alpha":2,"beta":2,"gamma":2,"delta":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	write("internal/abi/abi.go", "package abi\n")
	write("internal/alpha/alpha.go", `package alpha
import (
 _ "github.com/anthony-chaudhary/fak/internal/abi"
 _ "github.com/anthony-chaudhary/fak/internal/beta"
)
`)
	write("internal/beta/beta.go", `package beta
import _ "github.com/anthony-chaudhary/fak/internal/abi"
`)
	write("internal/gamma/gamma.go", `package gamma
import _ "github.com/anthony-chaudhary/fak/internal/beta"
`)
	write("internal/delta/delta.go", "package delta\n")

	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	wantHotspots := []Hotspot{{Name: "abi", FanIn: 2}, {Name: "beta", FanIn: 2}}
	if !reflect.DeepEqual(r.Hotspots, wantHotspots) {
		t.Fatalf("hotspots=%+v want=%+v", r.Hotspots, wantHotspots)
	}
	byName := map[string]Leaf{}
	for _, leaf := range r.Leaves {
		byName[leaf.Name] = leaf
	}
	if got, want := byName["abi"].Dependents, []string{"alpha", "beta"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("abi dependents=%v want=%v", got, want)
	}
	if got, want := byName["beta"].Dependents, []string{"alpha", "gamma"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("beta dependents=%v want=%v", got, want)
	}
	if len(byName["delta"].Dependents) != 0 {
		t.Fatalf("delta dependents=%v", byName["delta"].Dependents)
	}

	scoped, err := Analyze(root, "abi")
	if err != nil {
		t.Fatal(err)
	}
	if len(scoped.Leaves) != 1 || !reflect.DeepEqual(scoped.Leaves[0].Dependents, []string{"alpha", "beta"}) {
		t.Fatalf("scoped=%+v", scoped)
	}
	if len(scoped.Hotspots) != 0 {
		t.Fatalf("scoped hotspots=%+v", scoped.Hotspots)
	}
	if scoped.Violations != 0 {
		t.Fatalf("scoped violations=%d", scoped.Violations)
	}
}

func TestAnalyzeAdversarialInputs(t *testing.T) {
	tests := []struct {
		name       string
		contract   string
		leaf       string
		files      map[string]string
		wantErr    string
		wantDeps   []string
		wantFloor  int
		violations int
	}{
		{
			name:     "malformed contract",
			contract: "package architest\nvar tier = map[string]int{",
			wantErr:  "parse architecture contract",
		},
		{
			name:     "missing tier table",
			contract: "package architest\nvar tierName=[]string{\"root\"}\n",
			wantErr:  "missing tier or tierName",
		},
		{
			name:     "unknown scoped leaf",
			contract: "package architest\nvar tier=map[string]int{\"known\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			leaf:     "hostile-not-declared",
			wantErr:  `leaf "hostile-not-declared" has no tier declaration`,
		},
		{
			name:     "declared directory missing",
			contract: "package architest\nvar tier=map[string]int{\"missing\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			wantErr:  "internal/missing",
		},
		{
			name:     "malformed leaf source",
			contract: "package architest\nvar tier=map[string]int{\"broken\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			files:    map[string]string{"internal/broken/broken.go": "package broken\nimport ("},
			wantErr:  "broken.go",
		},
		{
			name:     "deduplicates and collapses nested imports",
			contract: "package architest\nvar tier=map[string]int{\"source\":1,\"target\":2}\nvar tierName=[]string{\"root\",\"primitive\",\"foundation-composite\"}\n",
			leaf:     "source",
			files: map[string]string{
				"internal/source/a.go":      "package source\nimport _ \"github.com/anthony-chaudhary/fak/internal/target/subpackage\"\n",
				"internal/source/b.go":      "package source\nimport _ \"github.com/anthony-chaudhary/fak/internal/target\"\n",
				"internal/source/a_test.go": "package source\nimport _ \"github.com/anthony-chaudhary/fak/internal/ignored\"\n",
				"internal/target/target.go": "package target\n",
			},
			wantDeps:   []string{"target"},
			wantFloor:  2,
			violations: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			writeArchitectureFixture(t, root, "internal/architest/architest_test.go", tt.contract)
			for path, body := range tt.files {
				writeArchitectureFixture(t, root, path, body)
			}
			r, err := Analyze(root, tt.leaf)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(filepath.ToSlash(err.Error()), tt.wantErr) {
					t.Fatalf("err=%v want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Leaves) != 1 {
				t.Fatalf("leaves=%d report=%+v", len(r.Leaves), r)
			}
			if !reflect.DeepEqual(r.Leaves[0].Dependencies, tt.wantDeps) {
				t.Fatalf("dependencies=%v want=%v", r.Leaves[0].Dependencies, tt.wantDeps)
			}
			if r.Leaves[0].ImportFloor != tt.wantFloor || r.Violations != tt.violations {
				t.Fatalf("floor=%d violations=%d", r.Leaves[0].ImportFloor, r.Violations)
			}
		})
	}
}

func writeArchitectureFixture(t *testing.T, root, path, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}
