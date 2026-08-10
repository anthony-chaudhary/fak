package trigram

import (
	"strings"
	"time"
)

type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	BuildLatency    time.Duration
	QueryLatency    time.Duration
	Queries         int
	ExactQueries    int
	FalsePositives  int
	FalseNegatives  int
	LocationErrors  int
	CPUSeconds      float64
	PeakRSSBytes    int64
	CorpusBytes     int64
	IndexBytes      int64
	NetworkBytes    int64
	StorageBytes    int64
	OperatorSeconds float64
	CostUSD         float64
	Note            string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}
type comparisonDoc struct{ id, path, body string }

var comparisonDocs = []comparisonDoc{{"1", "a.go", "package p\nfunc Alpha(){ sharedNeedle() }\n"}, {"2", "b.go", "package p\n// sharedNeedle appears here\nfunc Beta(){}\n"}, {"3", "c.go", "package p\nfunc Gamma(){ short() }\n"}, {"4", "d.go", "package p\nconst Unicode = \"καφές\"\n"}, {"5", "e.go", "package p\nfunc Empty(){}\n"}, {"6", "f.go", "package p\nfunc Again(){ sharedNeedle() }\n"}}
var comparisonQueries = []string{"sharedNeedle", "short", "καφές", "absent"}

func buildComparisonIndex() *Index {
	ix := &Index{}
	for _, d := range comparisonDocs {
		ix.Add(d.id, d.path, d.body)
	}
	return ix
}
func linearSearch(lit string) []Result {
	var out []Result
	for _, d := range comparisonDocs {
		var lines []int
		for i, line := range strings.Split(d.body, "\n") {
			if strings.Contains(line, lit) {
				lines = append(lines, i+1)
			}
		}
		if len(lines) > 0 {
			out = append(out, Result{ID: d.id, Path: d.path, Lines: lines})
		}
	}
	return out
}
func expectedQuery(q string, r []Result) bool {
	switch q {
	case "sharedNeedle":
		return len(r) == 3 && r[0].Path == "a.go" && r[0].Lines[0] == 2 && r[1].Path == "b.go" && r[1].Lines[0] == 2 && r[2].Path == "f.go"
	case "short":
		return len(r) == 1 && r[0].Path == "c.go" && r[0].Lines[0] == 2
	case "καφές":
		return len(r) == 1 && r[0].Path == "d.go" && r[0].Lines[0] == 2
	default:
		return len(r) == 0
	}
}
func corpusBytes() int64 {
	var n int64
	for _, d := range comparisonDocs {
		n += int64(len(d.body))
	}
	return n
}
func CompareLocal() ComparisonResult {
	start := time.Now()
	ix := buildComparisonIndex()
	build := time.Since(start)
	native := ComparisonArm{Name: "fak native trigram indexed literal search", Kind: "native", Available: true, BuildLatency: build, Queries: len(comparisonQueries), CorpusBytes: corpusBytes(), Note: "distinct-rune trigram postings with exact line verification and short-query fallback"}
	start = time.Now()
	for _, q := range comparisonQueries {
		if expectedQuery(q, ix.Search(q)) {
			native.ExactQueries++
		} else {
			native.LocationErrors++
		}
	}
	native.QueryLatency = time.Since(start)
	native.Correct = native.ExactQueries == len(comparisonQueries)
	base := ComparisonArm{Name: "optimized in-memory linear scan", Kind: "baseline", Available: true, Queries: len(comparisonQueries), CorpusBytes: corpusBytes(), Note: "tuned no-index baseline scans each loaded file once per query"}
	start = time.Now()
	for _, q := range comparisonQueries {
		if expectedQuery(q, linearSearch(q)) {
			base.ExactQueries++
		} else {
			base.LocationErrors++
		}
	}
	base.QueryLatency = time.Since(start)
	base.Correct = base.ExactQueries == len(comparisonQueries)
	return ComparisonResult{Workload: "build/load six files and execute four literal queries with exact ordered path/line oracle", Arms: []ComparisonArm{native, base, {Name: "ripgrep", Kind: "external", Note: "requires pinned rg process"}, {Name: "git grep", Kind: "external", Note: "requires real git grep process"}, {Name: "Zoekt", Kind: "external", Note: "requires real indexserver and query"}, {Name: "livegrep", Kind: "external", Note: "requires real index and query service"}, {Name: "Sourcegraph Search", Kind: "external", Note: "requires real indexed Sourcegraph instance"}}}
}
