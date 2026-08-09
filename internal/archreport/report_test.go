package archreport

import (
	"os"
	"path/filepath"
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
