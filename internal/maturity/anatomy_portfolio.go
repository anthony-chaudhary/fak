package maturity

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// AnatomyPortfolio compares first-class internal leaves without collapsing the
// underlying measurements into an opaque quality score.
type AnatomyPortfolio struct {
	Schema          string                  `json:"schema"`
	Summary         AnatomyPortfolioSummary `json:"summary"`
	Packages        []Anatomy               `json:"packages"`
	MissingPackages []string                `json:"missing_packages,omitempty"`
	Rankings        AnatomyRankings         `json:"rankings"`
	Caveats         []string                `json:"caveats"`
}

type AnatomyPortfolioSummary struct {
	DeclaredPackages     int `json:"declared_packages"`
	Packages             int `json:"packages"`
	ProductionFiles      int `json:"production_files"`
	Functions            int `json:"functions"`
	Statements           int `json:"statements"`
	DecisionPoints       int `json:"decision_points"`
	CyclomaticComplexity int `json:"cyclomatic_complexity"`
	ErrorExits           int `json:"error_exits"`
	SuccessExits         int `json:"success_exits"`
	AmbiguousExits       int `json:"ambiguous_exits"`
	AssumptionComments   int `json:"assumption_comments"`
	ExpectationComments  int `json:"expectation_comments"`
	InvariantComments    int `json:"invariant_comments"`
	RequirementComments  int `json:"requirement_comments"`
	DependencyCycles     int `json:"dependency_cycle_packages"`
	ExportedSymbols      int `json:"exported_symbols"`
	DocumentedExports    int `json:"documented_exports"`
	CLIReachablePackages int `json:"cli_reachable_packages"`
}

type AnatomyRankings struct {
	Complexity             []AnatomyRank `json:"aggregate_complexity"`
	ComplexityDensity      []AnatomyRank `json:"complexity_per_function"`
	MaximumFunction        []AnatomyRank `json:"maximum_function_complexity"`
	DocumentationGap       []AnatomyRank `json:"undocumented_exports"`
	Dependencies           []AnatomyRank `json:"internal_dependencies"`
	Dependents             []AnatomyRank `json:"internal_dependents"`
	Assumptions            []AnatomyRank `json:"assumption_comments"`
	Expectations           []AnatomyRank `json:"expectation_comments"`
	TransitiveDependencies []AnatomyRank `json:"transitive_dependencies"`
	TransitiveDependents   []AnatomyRank `json:"transitive_dependents"`
	ErrorExits             []AnatomyRank `json:"error_exits"`
}

type AnatomyRank struct {
	Package string  `json:"package"`
	Value   float64 `json:"value"`
}

// AnalyzeAnatomyPortfolio analyzes the capability roster declared in dos.toml.
// Limit affects ranking lists only; the package corpus and summary remain full.
func AnalyzeAnatomyPortfolio(root string, limit int) (AnatomyPortfolio, error) {
	if root == "" {
		root = "."
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return AnatomyPortfolio{}, err
	}
	lanes := parseLaneTrees(filepath.Join(absRoot, "dos.toml"))
	if len(lanes) == 0 {
		return AnatomyPortfolio{}, fmt.Errorf("no first-class internal leaves declared in dos.toml")
	}
	graph := internalImportGraph(filepath.Join(absRoot, "internal"))
	reachable := scanReachableWithGraph(absRoot, graph)
	p := AnatomyPortfolio{
		Schema: "fak-maturity-anatomy-portfolio/1",
		Caveats: []string{
			"Rankings are navigation aids, not quality grades or remediation priorities.",
			"Aggregate complexity grows with package size; complexity-per-function is shown separately.",
			"Static production-code counts do not measure runtime frequency or independent executable paths.",
		},
	}
	p.Summary.DeclaredPackages = len(lanes)
	for _, lane := range lanes {
		target := "internal/" + lane
		if info, statErr := os.Stat(filepath.Join(absRoot, "internal", lane)); statErr != nil || !info.IsDir() {
			if os.IsNotExist(statErr) {
				p.MissingPackages = append(p.MissingPackages, target)
				continue
			}
			return AnatomyPortfolio{}, fmt.Errorf("%s: %w", target, statErr)
		}
		hasCode, _, _ := scanLeaf(filepath.Join(absRoot, "internal", lane))
		if !hasCode {
			p.MissingPackages = append(p.MissingPackages, target)
			continue
		}
		a, err := analyzeAnatomy(absRoot, target, graph, reachable)
		if err != nil {
			return AnatomyPortfolio{}, fmt.Errorf("%s: %w", target, err)
		}
		p.Packages = append(p.Packages, a)
		addAnatomySummary(&p.Summary, a)
	}
	sort.Slice(p.Packages, func(i, j int) bool { return p.Packages[i].Package < p.Packages[j].Package })
	sort.Strings(p.MissingPackages)
	p.Summary.Packages = len(p.Packages)
	if limit <= 0 {
		limit = 10
	}
	p.Rankings = buildAnatomyRankings(p.Packages, limit)
	return p, nil
}

func addAnatomySummary(s *AnatomyPortfolioSummary, a Anatomy) {
	s.ProductionFiles += a.Shape.Files - a.Shape.TestFiles
	s.Functions += a.Shape.Functions
	s.Statements += a.Shape.Statements
	s.DecisionPoints += a.Flow.DecisionPoints
	s.CyclomaticComplexity += a.Flow.CyclomaticComplexity
	s.ErrorExits += a.Outcomes.ErrorExits
	s.SuccessExits += a.Outcomes.SuccessExits
	s.AmbiguousExits += a.Outcomes.AmbiguousExits
	s.AssumptionComments += a.Contracts.AssumptionComments
	s.ExpectationComments += a.Contracts.ExpectationComments
	s.InvariantComments += a.Contracts.InvariantComments
	s.RequirementComments += a.Contracts.RequirementComments
	if a.Position.InDependencyCycle {
		s.DependencyCycles++
	}
	s.ExportedSymbols += a.Documentation.ExportedSymbols
	s.DocumentedExports += a.Documentation.DocumentedExports
	if a.Position.CLIReachable {
		s.CLIReachablePackages++
	}
}

func buildAnatomyRankings(packages []Anatomy, limit int) AnatomyRankings {
	rank := func(value func(Anatomy) float64, includeZero bool) []AnatomyRank {
		rows := make([]AnatomyRank, 0, len(packages))
		for _, a := range packages {
			v := value(a)
			if includeZero || v != 0 {
				rows = append(rows, AnatomyRank{Package: a.Package, Value: math.Round(v*100) / 100})
			}
		}
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].Value == rows[j].Value {
				return rows[i].Package < rows[j].Package
			}
			return rows[i].Value > rows[j].Value
		})
		if len(rows) > limit {
			rows = rows[:limit]
		}
		return rows
	}
	return AnatomyRankings{
		Complexity: rank(func(a Anatomy) float64 { return float64(a.Flow.CyclomaticComplexity) }, false),
		ComplexityDensity: rank(func(a Anatomy) float64 {
			if a.Shape.Functions == 0 {
				return 0
			}
			return float64(a.Flow.CyclomaticComplexity) / float64(a.Shape.Functions)
		}, false),
		MaximumFunction: rank(func(a Anatomy) float64 { return float64(a.Flow.MaximumFunction) }, false),
		DocumentationGap: rank(func(a Anatomy) float64 {
			return float64(a.Documentation.ExportedSymbols - a.Documentation.DocumentedExports)
		}, false),
		Dependencies:           rank(func(a Anatomy) float64 { return float64(len(a.Position.InternalDependencies)) }, false),
		Dependents:             rank(func(a Anatomy) float64 { return float64(len(a.Position.InternalDependents)) }, false),
		TransitiveDependencies: rank(func(a Anatomy) float64 { return float64(a.Position.TransitiveDependencies) }, false),
		TransitiveDependents:   rank(func(a Anatomy) float64 { return float64(a.Position.TransitiveDependents) }, false),
		Assumptions:            rank(func(a Anatomy) float64 { return float64(a.Contracts.AssumptionComments) }, false),
		Expectations:           rank(func(a Anatomy) float64 { return float64(a.Contracts.ExpectationComments) }, false),
		ErrorExits:             rank(func(a Anatomy) float64 { return float64(a.Outcomes.ErrorExits) }, false),
	}
}

func RenderAnatomyPortfolioText(w io.Writer, p AnatomyPortfolio) {
	s := p.Summary
	fmt.Fprintf(w, "MATURITY ANATOMY PORTFOLIO  packages=%d/%d cli_reachable=%d missing=%d\n", s.Packages, s.DeclaredPackages, s.CLIReachablePackages, len(p.MissingPackages))
	fmt.Fprintf(w, "shape          production_files=%d functions=%d statements=%d\n", s.ProductionFiles, s.Functions, s.Statements)
	fmt.Fprintf(w, "flow           decisions=%d cyclomatic=%d\n", s.DecisionPoints, s.CyclomaticComplexity)
	fmt.Fprintf(w, "outcomes       success=%d error=%d ambiguous=%d\n", s.SuccessExits, s.ErrorExits, s.AmbiguousExits)
	fmt.Fprintf(w, "contracts      assumptions=%d expectations=%d invariants=%d requirements=%d\n", s.AssumptionComments, s.ExpectationComments, s.InvariantComments, s.RequirementComments)
	fmt.Fprintf(w, "graph          cycle_packages=%d\n", s.DependencyCycles)
	fmt.Fprintf(w, "documentation exported=%d documented=%d\n", s.ExportedSymbols, s.DocumentedExports)
	if len(p.MissingPackages) > 0 {
		fmt.Fprintf(w, "roster_gap      %s\n", strings.Join(p.MissingPackages, ", "))
	}
	renderRanks := func(label string, rows []AnatomyRank) {
		fmt.Fprintf(w, "\n%s\n", label)
		for _, row := range rows {
			fmt.Fprintf(w, "  %-42s %g\n", row.Package, row.Value)
		}
	}
	renderRanks("TOP AGGREGATE COMPLEXITY", p.Rankings.Complexity)
	renderRanks("TOP COMPLEXITY / FUNCTION", p.Rankings.ComplexityDensity)
	renderRanks("TOP MAXIMUM FUNCTION COMPLEXITY", p.Rankings.MaximumFunction)
	renderRanks("TOP UNDOCUMENTED EXPORTS", p.Rankings.DocumentationGap)
	renderRanks("TOP INTERNAL DEPENDENCIES", p.Rankings.Dependencies)
	renderRanks("TOP INTERNAL DEPENDENTS", p.Rankings.Dependents)
	renderRanks("TOP TRANSITIVE DEPENDENCIES", p.Rankings.TransitiveDependencies)
	renderRanks("TOP TRANSITIVE DEPENDENTS", p.Rankings.TransitiveDependents)
	renderRanks("TOP ASSUMPTION COMMENTS", p.Rankings.Assumptions)
	renderRanks("TOP EXPECTATION COMMENTS", p.Rankings.Expectations)
	renderRanks("TOP ERROR EXITS", p.Rankings.ErrorExits)
	fmt.Fprintln(w, "\nnote           rankings navigate evidence; they are not quality grades or runtime path counts")
}

func EncodeAnatomyPortfolioJSON(w io.Writer, p AnatomyPortfolio) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(p)
}
