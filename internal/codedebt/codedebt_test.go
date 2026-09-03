package codedebt

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseDefectPatterns(t *testing.T) {
	cases := []struct {
		kpi      string
		raw      string
		wantKind string
		wantPath string
		wantPkg  string
		wantCat  Category
	}{
		{
			kpi:      "architecture",
			raw:      "god-file cmd/fak/main.go (1600 lines > 1500)",
			wantKind: "god-file",
			wantPath: "cmd/fak/main.go",
			wantPkg:  "cmd/fak",
			wantCat:  CategoryModularity,
		},
		{
			kpi:      "architecture",
			raw:      "god-function internal/gateway/stream.go:handleStream (250 lines > 200)",
			wantKind: "god-function",
			wantPath: "internal/gateway/stream.go",
			wantPkg:  "internal/gateway",
			wantCat:  CategoryModularity,
		},
		{
			kpi:      "tests",
			raw:      "non-trivial package has no _test.go: internal/orphan",
			wantKind: "untested-package",
			wantPath: "internal/orphan",
			wantPkg:  "internal/orphan",
			wantCat:  CategoryInternalCoherence,
		},
		{
			kpi:      "assertion_strength",
			raw:      "zero-assertion test (cannot fail): internal/foo/foo_test.go:42 TestVacuous",
			wantKind: "zero-assertion-test",
			wantPath: "internal/foo/foo_test.go",
			wantPkg:  "internal/foo",
			wantCat:  CategoryInternalCoherence,
		},
		{
			kpi:      "format",
			raw:      "unformatted (run gofmt -w): cmd/fak/bar.go",
			wantKind: "unformatted",
			wantPath: "cmd/fak/bar.go",
			wantPkg:  "cmd/fak",
			wantCat:  CategoryInternalConsistency,
		},
		{
			kpi:      "deps",
			raw:      "external dependency added: github.com/stretchr/testify",
			wantKind: "external-dep",
			wantPath: "go.mod",
			wantPkg:  ".",
			wantCat:  CategoryInternalConsistency,
		},
		{
			kpi:      "honesty",
			raw:      "untagged/double-tagged claim: - [ ] missing tag",
			wantKind: "mis-tagged-claim",
			wantPath: "CLAIMS.md",
			wantPkg:  ".",
			wantCat:  CategoryInternalConsistency,
		},
	}

	for _, tc := range cases {
		d := ParseDefect(tc.kpi, tc.raw)
		if d.Kind != tc.wantKind {
			t.Errorf("raw %q: Kind = %q, want %q", tc.raw, d.Kind, tc.wantKind)
		}
		if d.Path != tc.wantPath {
			t.Errorf("raw %q: Path = %q, want %q", tc.raw, d.Path, tc.wantPath)
		}
		if d.Package != tc.wantPkg {
			t.Errorf("raw %q: Package = %q, want %q", tc.raw, d.Package, tc.wantPkg)
		}
		if len(d.Categories) == 0 || d.Categories[0] != tc.wantCat {
			t.Errorf("raw %q: Category = %v, want %q", tc.raw, d.Categories, tc.wantCat)
		}
	}
}

func TestQueryFilteringAndSummarization(t *testing.T) {
	rep := &Report{
		TotalDebt: 4,
		Defects: []Defect{
			ParseDefect("architecture", "god-file internal/gateway/main.go (1600 lines > 1500)"),
			ParseDefect("architecture", "god-function internal/gateway/sub.go:runIt (220 lines > 200)"),
			ParseDefect("format", "unformatted (run gofmt -w): cmd/fak/foo.go"),
			ParseDefect("tests", "non-trivial package has no _test.go: internal/alpha"),
		},
	}

	// 1. Filter by KPI
	qKPI := rep.Query(QueryOptions{KPI: "architecture"})
	if qKPI.MatchedDebt != 2 {
		t.Fatalf("Query(KPI: architecture) = %d, want 2", qKPI.MatchedDebt)
	}

	// 2. Filter by Category
	qCat := rep.Query(QueryOptions{Category: CategoryModularity})
	if qCat.MatchedDebt != 2 {
		t.Fatalf("Query(Category: modularity) = %d, want 2", qCat.MatchedDebt)
	}

	// 3. Filter by Path / Package
	qPath := rep.Query(QueryOptions{Path: "internal/gateway"})
	if qPath.MatchedDebt != 2 {
		t.Fatalf("Query(Path: internal/gateway) = %d, want 2", qPath.MatchedDebt)
	}
	qPkg := rep.Query(QueryOptions{Package: "cmd/fak"})
	if qPkg.MatchedDebt != 1 {
		t.Fatalf("Query(Package: cmd/fak) = %d, want 1", qPkg.MatchedDebt)
	}

	// 4. Filter by Search text
	qSearch := rep.Query(QueryOptions{Search: "runIt"})
	if qSearch.MatchedDebt != 1 {
		t.Fatalf("Query(Search: runIt) = %d, want 1", qSearch.MatchedDebt)
	}

	// 5. Test FormatSummary and FormatText
	summary := rep.Query(QueryOptions{}).FormatSummary()
	if !strings.Contains(summary, "modularity") || !strings.Contains(summary, "architecture") {
		t.Fatalf("summary missing expected keywords: %s", summary)
	}

	text := qKPI.FormatText()
	if !strings.Contains(text, "god-file") || !strings.Contains(text, "god-function") {
		t.Fatalf("formatted text missing defects: %s", text)
	}
}

func TestParsePayloadFromJSON(t *testing.T) {
	jsonBlob := `{
  "workspace": "/test/workspace",
  "corpus": {
    "score": 85.5,
    "grade": "B",
    "code_debt": 3,
    "debt_by_category": {
      "modularity": 2,
      "internal_consistency": 1
    },
    "breakdown": [
      {"kpi": "architecture", "score": 88, "debt": 2, "detail": "2 god-function(s)"},
      {"kpi": "format", "score": 88, "debt": 1, "detail": "1 unformatted file(s)"}
    ]
  },
  "kpis": [
    {
      "kpi": "architecture",
      "score": 88,
      "defects": [
        "god-function internal/foo/a.go:BigA (250 lines > 200)",
        "god-function internal/foo/b.go:BigB (210 lines > 200)"
      ]
    },
    {
      "kpi": "format",
      "score": 88,
      "defects": [
        "unformatted (run gofmt -w): cmd/bar/bar.go"
      ]
    }
  ]
}`

	rep, err := ParsePayload([]byte(jsonBlob))
	if err != nil {
		t.Fatalf("ParsePayload error: %v", err)
	}
	if rep.TotalDebt != 3 {
		t.Fatalf("TotalDebt = %d, want 3", rep.TotalDebt)
	}
	if rep.DebtByCat[CategoryModularity] != 2 {
		t.Fatalf("modularity debt = %d, want 2", rep.DebtByCat[CategoryModularity])
	}

	// Query package internal/foo
	res := rep.Query(QueryOptions{Package: "internal/foo"})
	if res.MatchedDebt != 2 {
		t.Fatalf("matched internal/foo debt = %d, want 2", res.MatchedDebt)
	}
}

func TestScanTreeNative(t *testing.T) {
	tempDir := t.TempDir()

	// 1. go.mod stdlib-only
	goMod := "module testmod\n\ngo 1.26\n"
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte(goMod), 0o644); err != nil {
		t.Fatal(err)
	}

	// 2. CLAIMS.md with an untagged claim
	claims := "# Claims\n- [SHIPPED] supported feature\n- [ ] untagged claim\n"
	if err := os.WriteFile(filepath.Join(tempDir, "CLAIMS.md"), []byte(claims), 0o644); err != nil {
		t.Fatal(err)
	}

	// 3. Package pkg1 with god-file and god-function (using small thresholds)
	pkg1Dir := filepath.Join(tempDir, "pkg1")
	if err := os.MkdirAll(pkg1Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	sb.WriteString("package pkg1\n\nfunc Oversized() {\n")
	for i := 0; i < 15; i++ {
		sb.WriteString("\t_ = 1\n")
	}
	sb.WriteString("}\n")
	if err := os.WriteFile(filepath.Join(pkg1Dir, "big.go"), []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	// 4. Package pkg2 with zero-assertion test
	pkg2Dir := filepath.Join(tempDir, "pkg2")
	if err := os.MkdirAll(pkg2Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	pkg2Code := "package pkg2\n\nfunc Helper1() {}\nfunc Helper2() {}\nfunc Helper3() {}\n"
	if err := os.WriteFile(filepath.Join(pkg2Dir, "code.go"), []byte(pkg2Code), 0o644); err != nil {
		t.Fatal(err)
	}
	pkg2Test := "package pkg2\n\nimport \"testing\"\n\nfunc TestVacuous(t *testing.T) {\n\tt.Log(\"nothing asserted\")\n}\n"
	if err := os.WriteFile(filepath.Join(pkg2Dir, "code_test.go"), []byte(pkg2Test), 0o644); err != nil {
		t.Fatal(err)
	}

	rep, err := ScanTree(ScanOptions{
		Workspace:     tempDir,
		Deterministic: true,
		FileHardMax:   12, // big.go has ~18 lines -> god-file
		FuncHardMax:   10, // Oversized has ~16 lines -> god-function
		TestMinFuncs:  2,
	})
	if err != nil {
		t.Fatalf("ScanTree error: %v", err)
	}

	// Verify honesty detected
	honestyQuery := rep.Query(QueryOptions{KPI: "honesty"})
	if honestyQuery.MatchedDebt != 1 {
		t.Errorf("honesty debt = %d, want 1", honestyQuery.MatchedDebt)
	}

	// Verify architecture detected
	archQuery := rep.Query(QueryOptions{KPI: "architecture"})
	if archQuery.MatchedDebt != 2 {
		t.Errorf("architecture debt = %d, want 2 (1 god-file + 1 god-function)", archQuery.MatchedDebt)
	}

	// Verify assertion strength detected
	assertQuery := rep.Query(QueryOptions{KPI: "assertion_strength"})
	if assertQuery.MatchedDebt != 1 {
		t.Errorf("assertion_strength debt = %d, want 1", assertQuery.MatchedDebt)
	}
}
