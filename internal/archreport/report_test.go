package archreport

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
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
	wantEdge := ViolationEdge{From: "primitive", FromTier: 1, FromTierName: "primitive", To: "composite", ToTier: 2, ToTierName: "foundation-composite", TierDistance: 1}
	if len(p.ViolationEdges) != 1 || p.ViolationEdges[0] != wantEdge || !reflect.DeepEqual(p.Violations, []string{"primitive -> composite"}) || p.ImportFloor != 2 {
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
	if got, want := byName["abi"].TransitiveDependents, []string{"alpha", "beta", "gamma"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("abi transitive dependents=%v want=%v", got, want)
	}
	wantPaths := []BlastPath{
		{Dependent: "alpha", Path: []string{"abi", "alpha"}},
		{Dependent: "beta", Path: []string{"abi", "beta"}},
		{Dependent: "gamma", Path: []string{"abi", "beta", "gamma"}},
	}
	if got := byName["abi"].BlastPaths; !reflect.DeepEqual(got, wantPaths) {
		t.Fatalf("abi blast paths=%v want=%v", got, wantPaths)
	}
	if got := byName["abi"].BlastRadius; got != 3 {
		t.Fatalf("abi blast radius=%d want=3", got)
	}
	if got, want := byName["beta"].TransitiveDependents, []string{"alpha", "gamma"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("beta transitive dependents=%v want=%v", got, want)
	}
	if got := byName["delta"].TransitiveDependents; got == nil || len(got) != 0 || byName["delta"].BlastRadius != 0 || byName["delta"].BlastPaths == nil || len(byName["delta"].BlastPaths) != 0 {
		t.Fatalf("delta transitive dependents=%v blast radius=%d blast paths=%v", got, byName["delta"].BlastRadius, byName["delta"].BlastPaths)
	}

	scoped, err := Analyze(root, "abi")
	if err != nil {
		t.Fatal(err)
	}
	if len(scoped.Leaves) != 1 || !reflect.DeepEqual(scoped.Leaves[0].Dependents, []string{"alpha", "beta"}) || !reflect.DeepEqual(scoped.Leaves[0].TransitiveDependents, []string{"alpha", "beta", "gamma"}) || scoped.Leaves[0].BlastRadius != 3 {
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

func TestAnalyzeErrorsNameRecovery(t *testing.T) {
	tests := []struct {
		name     string
		contract string
		leaf     string
		files    map[string]string
		want     []string
	}{
		{
			name:     "contract syntax",
			contract: "package architest\nvar tier = map[string]int{",
			want:     []string{"parse architecture contract", "repair the Go syntax before reporting"},
		},
		{
			name:     "contract declaration",
			contract: "package architest\nvar tierName=[]string{\"root\"}\n",
			want:     []string{"missing tier or tierName", "restore both declarations"},
		},
		{
			name:     "unknown leaf",
			contract: "package architest\nvar tier=map[string]int{\"known\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			leaf:     "unknown",
			want:     []string{"has no tier declaration", "choose a declared leaf", "or add its tier there"},
		},
		{
			name:     "leaf syntax",
			contract: "package architest\nvar tier=map[string]int{\"broken\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			files:    map[string]string{"internal/broken/broken.go": "package broken\nimport ("},
			want:     []string{"parse imports", "broken.go", "repair the Go syntax before reporting"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			writeArchitectureFixture(t, root, "internal/architest/architest_test.go", tt.contract)
			for path, body := range tt.files {
				writeArchitectureFixture(t, root, path, body)
			}
			_, err := Analyze(root, tt.leaf)
			if err == nil {
				t.Fatal("expected error")
			}
			for _, want := range tt.want {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q does not name recovery %q", err, want)
				}
			}
		})
	}
}

func TestAnalyzeRanksSinkCandidatesByTierGapThenName(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"alpha":4,"beta":3,"gamma":3,"delta":2}
var tierName=[]string{"root","primitive","foundation-composite","mechanism","composer"}
`)
	for _, leaf := range []string{"abi", "alpha", "beta", "gamma", "delta"} {
		writeArchitectureFixture(t, root, "internal/"+leaf+"/leaf.go", "package "+leaf+"\n")
	}
	report, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want := []SinkCandidate{
		{Name: "alpha", DeclaredTier: 4, DeclaredTierName: "composer", ImportFloor: 1, ImportFloorName: "primitive", TierGap: 3},
		{Name: "beta", DeclaredTier: 3, DeclaredTierName: "mechanism", ImportFloor: 1, ImportFloorName: "primitive", TierGap: 2},
		{Name: "gamma", DeclaredTier: 3, DeclaredTierName: "mechanism", ImportFloor: 1, ImportFloorName: "primitive", TierGap: 2},
	}
	if !reflect.DeepEqual(report.SinkCandidates, want) {
		t.Fatalf("candidates=%+v want=%+v", report.SinkCandidates, want)
	}
	for _, leaf := range report.Leaves {
		if leaf.TierGap != leaf.DeclaredTier-leaf.ImportFloor {
			t.Fatalf("leaf=%+v", leaf)
		}
	}
	scoped, err := Analyze(root, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	if len(scoped.SinkCandidates) != 0 || len(scoped.Leaves) != 1 || scoped.Leaves[0].TierGap != 3 {
		t.Fatalf("scoped=%+v", scoped)
	}
}

func TestAnalyzeStaleDeclarationDoesNotSuppressHealthyLeaves(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"healthy":1,"stale":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	writeArchitectureFixture(t, root, "internal/healthy/healthy.go", "package healthy\n")

	full, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(full.Leaves) != 1 || full.Leaves[0].Name != "healthy" {
		t.Fatalf("healthy leaves suppressed: %+v", full.Leaves)
	}
	wantDiagnostic := Diagnostic{
		Kind:     DiagnosticStaleTierDeclaration,
		Leaf:     "stale",
		Message:  "declared package directory " + filepath.Join(root, "internal", "stale") + " does not exist",
		Recovery: "create the package or remove its stale tier declaration",
	}
	if !reflect.DeepEqual(full.Diagnostics, []Diagnostic{wantDiagnostic}) {
		t.Fatalf("diagnostics=%+v want=%+v", full.Diagnostics, wantDiagnostic)
	}

	healthy, err := Analyze(root, "healthy")
	if err != nil {
		t.Fatal(err)
	}
	if len(healthy.Leaves) != 1 || healthy.Leaves[0].Name != "healthy" || len(healthy.Diagnostics) != 0 {
		t.Fatalf("healthy scoped report=%+v", healthy)
	}

	stale, err := Analyze(root, "stale")
	if err != nil {
		t.Fatal(err)
	}
	if len(stale.Leaves) != 0 || !reflect.DeepEqual(stale.Diagnostics, []Diagnostic{wantDiagnostic}) {
		t.Fatalf("stale scoped report=%+v", stale)
	}
}

func TestAnalyzeDeterminismUnderConcurrency(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"alpha":2,"beta":2,"gamma":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	writeArchitectureFixture(t, root, "internal/abi/abi.go", "package abi\n")
	writeArchitectureFixture(t, root, "internal/alpha/alpha.go", `package alpha
import (
 _ "github.com/anthony-chaudhary/fak/internal/beta"
 _ "github.com/anthony-chaudhary/fak/internal/abi"
)
`)
	writeArchitectureFixture(t, root, "internal/beta/beta.go", `package beta
import _ "github.com/anthony-chaudhary/fak/internal/abi"
`)
	writeArchitectureFixture(t, root, "internal/gamma/gamma.go", `package gamma
import _ "github.com/anthony-chaudhary/fak/internal/beta"
`)

	wantReport, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want, err := wantReport.JSON()
	if err != nil {
		t.Fatal(err)
	}

	const workers = 16
	results := make(chan []byte, workers)
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			report, err := Analyze(root, "")
			if err != nil {
				errs <- err
				return
			}
			raw, err := report.JSON()
			if err != nil {
				errs <- err
				return
			}
			results <- raw
		}()
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	seen := 0
	for got := range results {
		seen++
		if !bytes.Equal(got, want) {
			t.Fatalf("non-deterministic report\nwant:\n%s\ngot:\n%s", want, got)
		}
	}
	if seen != workers {
		t.Fatalf("results=%d want=%d", seen, workers)
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

func TestAnalyzeSortsTypedViolationEdgesAndKeepsStringCompatibility(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"alpha":1,"zeta":3,"beta":2}
var tierName=[]string{"root","primitive","foundation-composite","mechanism"}
`)
	writeArchitectureFixture(t, root, "internal/alpha/alpha.go", `package alpha
import (
 _ "github.com/anthony-chaudhary/fak/internal/zeta"
 _ "github.com/anthony-chaudhary/fak/internal/beta"
)
`)
	writeArchitectureFixture(t, root, "internal/zeta/zeta.go", "package zeta\n")
	writeArchitectureFixture(t, root, "internal/beta/beta.go", "package beta\n")
	r, err := Analyze(root, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	want := []ViolationEdge{
		{From: "alpha", FromTier: 1, FromTierName: "primitive", To: "zeta", ToTier: 3, ToTierName: "mechanism", TierDistance: 2},
		{From: "alpha", FromTier: 1, FromTierName: "primitive", To: "beta", ToTier: 2, ToTierName: "foundation-composite", TierDistance: 1},
	}
	if len(r.Leaves) != 1 || !reflect.DeepEqual(r.Leaves[0].ViolationEdges, want) || !reflect.DeepEqual(r.Leaves[0].Violations, []string{"alpha -> zeta", "alpha -> beta"}) || r.MaxViolationDistance != 2 {
		t.Fatalf("report=%+v", r)
	}
	raw, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"violation_edges"`, `"from_tier_name"`, `"violations"`, `"tier_distance"`, `"max_violation_distance"`} {
		if !bytes.Contains(raw, []byte(key)) {
			t.Fatalf("JSON missing %s: %s", key, raw)
		}
	}
}
