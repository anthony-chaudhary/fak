package codedebt

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var (
	benchDefectSink      Defect
	benchStringSink      string
	benchReportSink      *Report
	benchQueryResultSink QueryResult
	benchDefectsSink     []Defect
	benchMapSink         map[string]int
	benchBoolMapSink     map[string]bool
	benchStringsSink     []string
)

func makeBenchmarkReport(defectCount int) *Report {
	kpis := []string{
		"architecture", "tests", "assertion_strength", "format", "deps", "honesty", "vet", "build",
	}
	categories := []Category{
		CategoryModularity, CategoryInternalConsistency, CategoryInternalCoherence,
	}
	pkgs := []string{
		"cmd/fak", "internal/gateway", "internal/codedebt", "internal/model",
		"internal/ctxmmu", "internal/engine", "internal/scheduler", "internal/storage",
	}

	rep := &Report{
		Workspace:    "fake/workspace",
		TotalDebt:    defectCount,
		DebtByKPI:    make(map[string]int),
		DebtByCat:    make(map[Category]int),
		DebtByPkg:    make(map[string]int),
		Defects:      make([]Defect, 0, defectCount),
		KPISummaries: make(map[string]KPISummary),
	}

	for i := 0; i < defectCount; i++ {
		kpi := kpis[i%len(kpis)]
		cat := categories[i%len(categories)]
		pkg := pkgs[i%len(pkgs)]
		path := fmt.Sprintf("%s/file_%d.go", pkg, i%5)
		raw := fmt.Sprintf("god-function %s:Func%d (%d lines > 200)", path, i, 205+(i%50))

		d := Defect{
			KPI:        kpi,
			Categories: []Category{cat},
			Raw:        raw,
			Path:       path,
			Package:    pkg,
			Kind:       "god-function",
			Line:       10 + i,
		}

		rep.Defects = append(rep.Defects, d)
		rep.DebtByKPI[kpi]++
		rep.DebtByCat[cat]++
		rep.DebtByPkg[pkg]++
	}

	for _, k := range kpis {
		rep.KPISummaries[k] = KPISummary{
			KPI:        k,
			Score:      80,
			Debt:       rep.DebtByKPI[k],
			Detail:     fmt.Sprintf("%d defect(s)", rep.DebtByKPI[k]),
			Categories: KPICategories[k],
		}
	}
	calculateScoreAndGrade(rep)

	return rep
}

func makeScorecardJSON(numDefects int) []byte {
	type summaryItem struct {
		KPI    string `json:"kpi"`
		Score  int    `json:"score"`
		Debt   int    `json:"debt"`
		Detail string `json:"detail"`
	}
	type kpiItem struct {
		KPI     string   `json:"kpi"`
		Score   int      `json:"score"`
		Defects []string `json:"defects"`
		Soft    []string `json:"soft"`
	}

	kpis := []string{"architecture", "format", "deps", "honesty", "tests"}
	var kpiItems []kpiItem
	var breakdown []summaryItem

	defectsPerKPI := (numDefects + len(kpis) - 1) / len(kpis)
	totalEmitted := 0

	for _, k := range kpis {
		var dList []string
		for j := 0; j < defectsPerKPI && totalEmitted < numDefects; j++ {
			var raw string
			switch k {
			case "architecture":
				raw = fmt.Sprintf("god-function internal/pkg%d/file%d.go:BigFunc (250 lines > 200)", j%4, j)
			case "format":
				raw = fmt.Sprintf("unformatted (run gofmt -w): cmd/tool%d/file%d.go", j%3, j)
			case "deps":
				raw = fmt.Sprintf("external dependency added: example.com/dep%d", j)
			case "honesty":
				raw = fmt.Sprintf("untagged/double-tagged claim: - [ ] untagged claim %d", j)
			case "tests":
				raw = fmt.Sprintf("non-trivial package has no _test.go: internal/untested%d", j)
			default:
				raw = fmt.Sprintf("misc defect %d", j)
			}
			dList = append(dList, raw)
			totalEmitted++
		}
		kpiItems = append(kpiItems, kpiItem{
			KPI:     k,
			Score:   85,
			Defects: dList,
			Soft:    []string{"soft advisory warning"},
		})
		breakdown = append(breakdown, summaryItem{
			KPI:    k,
			Score:  85,
			Debt:   len(dList),
			Detail: fmt.Sprintf("%d defect(s)", len(dList)),
		})
	}

	payload := struct {
		Workspace string `json:"workspace"`
		Corpus    struct {
			Score          float64        `json:"score"`
			Grade          string         `json:"grade"`
			CodeDebt       int            `json:"code_debt"`
			DebtByCategory map[string]int `json:"debt_by_category"`
			Breakdown      []summaryItem  `json:"breakdown"`
		} `json:"corpus"`
		KPIs []kpiItem `json:"kpis"`
	}{
		Workspace: "/workspace/bench",
		Corpus: struct {
			Score          float64        `json:"score"`
			Grade          string         `json:"grade"`
			CodeDebt       int            `json:"code_debt"`
			DebtByCategory map[string]int `json:"debt_by_category"`
			Breakdown      []summaryItem  `json:"breakdown"`
		}{
			Score:    85.0,
			Grade:    "B",
			CodeDebt: totalEmitted,
			DebtByCategory: map[string]int{
				"modularity":           totalEmitted / 2,
				"internal_consistency": totalEmitted / 2,
			},
			Breakdown: breakdown,
		},
		KPIs: kpiItems,
	}

	bytes, _ := json.Marshal(payload)
	return bytes
}

func setupBenchWorkspace(tb testing.TB) (string, []string, []string) {
	tempDir := tb.TempDir()

	goMod := "module benchmod\n\ngo 1.26\n"
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte(goMod), 0o644); err != nil {
		tb.Fatal(err)
	}

	claims := "# Claims\n- [SHIPPED] claim one\n- [ ] untagged claim\n"
	if err := os.WriteFile(filepath.Join(tempDir, "CLAIMS.md"), []byte(claims), 0o644); err != nil {
		tb.Fatal(err)
	}

	var srcFiles []string
	var testFiles []string

	for pkgIdx := 0; pkgIdx < 3; pkgIdx++ {
		pkgDir := filepath.Join(tempDir, fmt.Sprintf("pkg%d", pkgIdx))
		if err := os.MkdirAll(pkgDir, 0o755); err != nil {
			tb.Fatal(err)
		}

		var srcContent strings.Builder
		srcContent.WriteString(fmt.Sprintf("package pkg%d\n\n", pkgIdx))
		for f := 0; f < 5; f++ {
			srcContent.WriteString(fmt.Sprintf("func Worker%d() int {\n", f))
			for l := 0; l < 10; l++ {
				srcContent.WriteString("\t_ = 42\n")
			}
			srcContent.WriteString("\treturn 42\n}\n\n")
		}
		srcRel := fmt.Sprintf("pkg%d/worker.go", pkgIdx)
		if err := os.WriteFile(filepath.Join(tempDir, filepath.FromSlash(srcRel)), []byte(srcContent.String()), 0o644); err != nil {
			tb.Fatal(err)
		}
		srcFiles = append(srcFiles, srcRel)

		var testContent strings.Builder
		testContent.WriteString(fmt.Sprintf("package pkg%d\n\nimport \"testing\"\n\n", pkgIdx))
		testContent.WriteString("func TestWorker(t *testing.T) {\n")
		testContent.WriteString("\tif Worker0() != 42 {\n\t\tt.Fatal(\"mismatch\")\n\t}\n}\n")
		testRel := fmt.Sprintf("pkg%d/worker_test.go", pkgIdx)
		if err := os.WriteFile(filepath.Join(tempDir, filepath.FromSlash(testRel)), []byte(testContent.String()), 0o644); err != nil {
			tb.Fatal(err)
		}
		testFiles = append(testFiles, testRel)
	}

	return tempDir, srcFiles, testFiles
}

func BenchmarkParseDefect(b *testing.B) {
	cases := []struct {
		name string
		kpi  string
		raw  string
	}{
		{"GodFile", "architecture", "god-file cmd/fak/main.go (1600 lines > 1500)"},
		{"GodFunction", "architecture", "god-function internal/gateway/stream.go:handleStream (250 lines > 200)"},
		{"UntestedPackage", "tests", "non-trivial package has no _test.go: internal/orphan"},
		{"ZeroAssertion", "assertion_strength", "zero-assertion test (cannot fail): internal/foo/foo_test.go:42 TestVacuous"},
		{"Unformatted", "format", "unformatted (run gofmt -w): cmd/fak/bar.go"},
		{"VetDiagnostic", "vet", "vet: internal/pkg/sub.go:15: unreachable code"},
		{"ExternalDep", "deps", "external dependency added: github.com/stretchr/testify"},
		{"GosumPresent", "deps", "go.sum exists (the zero-dep invariant broke)"},
		{"MisTaggedClaim", "honesty", "untagged/double-tagged claim: - [ ] missing tag"},
		{"BuildFailure", "build", "build failure: exit status 2"},
		{"Misc", "unknown_kpi", "some generic defect message"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchDefectSink = ParseDefect(tc.kpi, tc.raw)
			}
		})
	}

	b.Run("BatchMixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tc := cases[i%len(cases)]
			benchDefectSink = ParseDefect(tc.kpi, tc.raw)
		}
	})
}

func BenchmarkPackageOf(b *testing.B) {
	paths := []struct {
		name string
		path string
	}{
		{"RootGoMod", "go.mod"},
		{"RootClaims", "CLAIMS.md"},
		{"TopLevelGo", "main.go"},
		{"SingleSubdir", "cmd/fak/main.go"},
		{"DeepSubdir", "internal/gateway/v2/transports/grpc/server.go"},
		{"PackageOnly", "internal/codedebt"},
	}

	for _, p := range paths {
		b.Run(p.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchStringSink = PackageOf(p.path)
			}
		})
	}
}

func BenchmarkParsePayload(b *testing.B) {
	smallJSON := makeScorecardJSON(10)
	largeJSON := makeScorecardJSON(200)

	b.Run("Small_10Defects", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep, err := ParsePayload(smallJSON)
			if err != nil {
				b.Fatal(err)
			}
			benchReportSink = rep
		}
	})

	b.Run("Large_200Defects", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep, err := ParsePayload(largeJSON)
			if err != nil {
				b.Fatal(err)
			}
			benchReportSink = rep
		}
	})
}

func BenchmarkReportQuery(b *testing.B) {
	rep := makeBenchmarkReport(200)

	scenarios := []struct {
		name string
		opts QueryOptions
	}{
		{"Unfiltered", QueryOptions{}},
		{"FilterByKPI", QueryOptions{KPI: "architecture"}},
		{"FilterByCategory", QueryOptions{Category: CategoryModularity}},
		{"FilterByPackage", QueryOptions{Package: "internal/codedebt"}},
		{"FilterByPath", QueryOptions{Path: "internal/codedebt"}},
		{"FilterBySearch", QueryOptions{Search: "god-function"}},
		{"FilterCombined", QueryOptions{KPI: "architecture", Package: "internal/codedebt", Search: "Func"}},
		{"WithLimit", QueryOptions{Limit: 15}},
		{"Deterministic", QueryOptions{Deterministic: true}},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchQueryResultSink = rep.Query(sc.opts)
			}
		})
	}
}

func BenchmarkQueryResultFormatText(b *testing.B) {
	sizes := []struct {
		name  string
		count int
	}{
		{"10Defects", 10},
		{"50Defects", 50},
		{"200Defects", 200},
	}

	for _, s := range sizes {
		rep := makeBenchmarkReport(s.count)
		qRes := rep.Query(QueryOptions{})

		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchStringSink = qRes.FormatText()
			}
		})
	}
}

func BenchmarkQueryResultFormatSummary(b *testing.B) {
	sizes := []struct {
		name  string
		count int
	}{
		{"10Defects", 10},
		{"50Defects", 50},
		{"200Defects", 200},
	}

	for _, s := range sizes {
		rep := makeBenchmarkReport(s.count)
		qRes := rep.Query(QueryOptions{})

		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchStringSink = qRes.FormatSummary()
			}
		})
	}
}

func BenchmarkAssembleDefects(b *testing.B) {
	arch := []Defect{
		ParseDefect("architecture", "god-file cmd/fak/main.go (1600 lines > 1500)"),
		ParseDefect("architecture", "god-function internal/gateway/stream.go:handleStream (250 lines > 200)"),
	}
	tests := []Defect{
		ParseDefect("tests", "non-trivial package has no _test.go: internal/orphan"),
	}
	zeroAssert := []Defect{
		ParseDefect("assertion_strength", "zero-assertion test (cannot fail): internal/foo/foo_test.go:42 TestVacuous"),
	}
	format := []Defect{
		ParseDefect("format", "unformatted (run gofmt -w): cmd/fak/bar.go"),
	}
	deps := []Defect{
		ParseDefect("deps", "external dependency added: github.com/stretchr/testify"),
	}
	honesty := []Defect{
		ParseDefect("honesty", "untagged/double-tagged claim: - [ ] missing tag"),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDefectsSink = assembleDefects(arch, tests, zeroAssert, format, deps, honesty)
	}
}

func BenchmarkScanSource(b *testing.B) {
	root, srcFiles, _ := setupBenchWorkspace(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counts, arch, format := scanSource(root, srcFiles, DefaultFileHardMax, DefaultFuncHardMax)
		benchMapSink = counts
		benchDefectsSink = arch
		_ = format
	}
}

func BenchmarkScanTests(b *testing.B) {
	root, _, testFiles := setupBenchWorkspace(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tested, zeroAssert := scanTests(root, testFiles)
		benchBoolMapSink = tested
		benchDefectsSink = zeroAssert
	}
}

func BenchmarkCheckUntestedPackages(b *testing.B) {
	pkgFuncCount := map[string]int{
		"cmd/fak":           12,
		"internal/gateway":  25,
		"internal/codedebt": 18,
		"internal/storage":  8,
		"internal/tiny":     2,
		"internal/untested": 15,
	}
	testedPkgs := map[string]bool{
		"cmd/fak":           true,
		"internal/gateway":  true,
		"internal/codedebt": true,
		"internal/storage":  true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		defects, nonTrivial := checkUntestedPackages(pkgFuncCount, testedPkgs, DefaultTestMinFunc)
		benchDefectsSink = defects
		benchStringsSink = nonTrivial
	}
}

func BenchmarkCalculateScoreAndGrade(b *testing.B) {
	rep := makeBenchmarkReport(50)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculateScoreAndGrade(rep)
	}
}
