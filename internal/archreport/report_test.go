package archreport

import (
	"os"
	"path/filepath"
	"reflect"
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
