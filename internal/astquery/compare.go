package astquery

import (
	"strings"
	"time"
)

const comparisonPattern = `eq($X, $X)`
const comparisonSource = `package fixture
func f() {
	eq(alpha, alpha)
	eq(alpha, beta)
	_ = "eq(decoy, decoy)"
	// eq(comment, comment)
	eq(beta, beta)
	other(beta, beta)
}`

type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	Latency         time.Duration
	TruePositives   int
	FalsePositives  int
	FalseNegatives  int
	BindingErrors   int
	LocationErrors  int
	ParseFailures   int
	InputBytes      int64
	CPUSeconds      float64
	PeakRSSBytes    int64
	NetworkBytes    int64
	OperatorSeconds float64
	CostUSD         float64
	Note            string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func correctMatches(ms []Match) bool {
	return len(ms) == 2 && ms[0].Pos.Line == 3 && ms[0].Bindings["X"] == "alpha" && ms[1].Pos.Line == 7 && ms[1].Bindings["X"] == "beta"
}

func literalMatches(src string) (lines []int) {
	for i, line := range strings.Split(src, "\n") {
		if strings.Contains(line, "eq(") {
			lines = append(lines, i+1)
		}
	}
	return lines
}

func CompareLocal() ComparisonResult {
	start := time.Now()
	ms, err := Search(comparisonSource, comparisonPattern)
	nativeLatency := time.Since(start)
	nativeCorrect := err == nil && correctMatches(ms)
	start = time.Now()
	lines := literalMatches(comparisonSource)
	baselineLatency := time.Since(start)
	return ComparisonResult{Workload: "find exactly two repeated-metavariable Go calls while rejecting inconsistent arguments, comments, strings, and unrelated calls", Arms: []ComparisonArm{
		{Name: "fak native Go AST query", Kind: "native", Available: true, Correct: nativeCorrect, Latency: nativeLatency, TruePositives: len(ms), FalseNegatives: boolInt(!nativeCorrect), ParseFailures: boolInt(err != nil), InputBytes: int64(len(comparisonSource)), Note: "Go AST unification with named-hole back-reference and source bindings"},
		{Name: "literal text search", Kind: "baseline", Available: true, Correct: false, Latency: baselineLatency, TruePositives: 2, FalsePositives: len(lines) - 2, InputBytes: int64(len(comparisonSource)), Note: "tuned single-pass literal scan finds candidate text but cannot enforce syntax or back-references"},
		{Name: "Semgrep", Kind: "external", Note: "requires pinned Semgrep and real pattern execution"},
		{Name: "ast-grep", Kind: "external", Note: "requires pinned ast-grep and real rule execution"},
		{Name: "Comby", Kind: "external", Note: "requires pinned Comby and real matcher execution"},
		{Name: "gogrep", Kind: "external", Note: "requires pinned gogrep and real pattern execution"},
	}}
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
