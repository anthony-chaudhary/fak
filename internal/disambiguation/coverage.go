package disambiguation

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path"
	"sort"
	"strings"
	"testing/fstest"
	"unicode"
)

const CoverageSchemaVersion = "fak-disambiguation-coverage/1"

const (
	CoverageReasonMissingClassification = "MISSING_TERM_CLASSIFICATION"
	CoverageClassCanonical              = "canonical"
	CoverageClassIncidental             = "incidental"
)

type PublicTerminologySurface struct {
	Locator string `json:"locator"`
	Kind    string `json:"kind"`
}

type IncidentalTerm struct {
	Term   string `json:"term"`
	Reason string `json:"reason"`
}

type CoverageFinding struct {
	Surface   string `json:"surface"`
	Term      string `json:"term"`
	Candidate string `json:"candidate"`
	Reason    string `json:"reason"`
}

type CoverageReport struct {
	SchemaVersion string                     `json:"schema_version"`
	Surfaces      []PublicTerminologySurface `json:"surfaces"`
	Findings      []CoverageFinding          `json:"findings"`
	Candidates    int                        `json:"candidates"`
	Canonical     int                        `json:"canonical"`
	Incidental    int                        `json:"incidental"`
	OK            bool                       `json:"ok"`
}

// InventoryCoverage checks only explicitly declared public surfaces. Go packages are
// parsed from the supplied public repository filesystem; no network or private input is used.
func InventoryCoverage(root fs.FS, surfaces []PublicTerminologySurface, index *Index, incidental []IncidentalTerm) (CoverageReport, error) {
	report := CoverageReport{SchemaVersion: CoverageSchemaVersion, Surfaces: append([]PublicTerminologySurface(nil), surfaces...), Findings: []CoverageFinding{}}
	sort.Slice(report.Surfaces, func(i, j int) bool { return report.Surfaces[i].Locator < report.Surfaces[j].Locator })
	canonical := map[string]bool{}
	symbols := map[string]bool{}
	for _, owners := range index.canonical {
		for _, e := range owners {
			canonical[normalizeCoverageTerm(e.Identity.CanonicalTerm)] = true
			for _, alias := range e.Identity.Aliases {
				canonical[normalizeCoverageTerm(alias)] = true
			}
			for _, source := range e.Sources {
				if source.Reference != nil && source.Reference.Kind == ReferenceKindGoSymbol {
					symbols[source.Reference.Name] = true
				}
			}
		}
	}
	classified := map[string]bool{}
	for i, item := range incidental {
		if strings.TrimSpace(item.Term) == "" || strings.TrimSpace(item.Reason) == "" {
			return report, fmt.Errorf("incidental[%d]: term and reason are required", i)
		}
		classified[item.Term] = true
	}
	seen := map[string]bool{}
	for _, surface := range report.Surfaces {
		if surface.Kind != "go_package" {
			return report, fmt.Errorf("surface %q: unsupported kind %q", surface.Locator, surface.Kind)
		}
		entries, err := fs.ReadDir(root, surface.Locator)
		if err != nil {
			return report, fmt.Errorf("surface %q: %w", surface.Locator, err)
		}
		fset := token.NewFileSet()
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			locator := path.Join(surface.Locator, entry.Name())
			file, err := parser.ParseFile(fset, locator, mustReadFS(root, locator), 0)
			if err != nil {
				return report, fmt.Errorf("parse %s: %w", locator, err)
			}
			ast.Inspect(file, func(node ast.Node) bool {
				var name string
				switch n := node.(type) {
				case *ast.TypeSpec:
					name = n.Name.Name
				case *ast.ValueSpec:
					for _, ident := range n.Names {
						addCoverageCandidate(&report, seen, canonical, symbols, classified, surface.Locator, ident.Name)
					}
					return true
				case *ast.FuncDecl:
					name = n.Name.Name
				}
				addCoverageCandidate(&report, seen, canonical, symbols, classified, surface.Locator, name)
				return true
			})
		}
	}
	sort.Slice(report.Findings, func(i, j int) bool {
		if report.Findings[i].Surface != report.Findings[j].Surface {
			return report.Findings[i].Surface < report.Findings[j].Surface
		}
		return report.Findings[i].Term < report.Findings[j].Term
	})
	report.OK = len(report.Findings) == 0
	return report, nil
}

func addCoverageCandidate(report *CoverageReport, seen, canonical, symbols, incidental map[string]bool, surface, name string) {
	if name == "" || !ast.IsExported(name) {
		return
	}
	key := surface + "\x00" + name
	if seen[key] {
		return
	}
	seen[key] = true
	report.Candidates++
	if symbols[name] || canonical[normalizeCoverageTerm(splitExportedName(name))] {
		report.Canonical++
		return
	}
	if incidental[name] {
		report.Incidental++
		return
	}
	report.Findings = append(report.Findings, CoverageFinding{Surface: surface, Term: name, Candidate: splitExportedName(name), Reason: CoverageReasonMissingClassification})
}

func splitExportedName(value string) string {
	var out []rune
	rs := []rune(value)
	for i, r := range rs {
		if i > 0 && unicode.IsUpper(r) && (unicode.IsLower(rs[i-1]) || (i+1 < len(rs) && unicode.IsLower(rs[i+1]))) {
			out = append(out, ' ')
		}
		out = append(out, unicode.ToLower(r))
	}
	return string(out)
}
func normalizeCoverageTerm(value string) string {
	return strings.Join(strings.Fields(strings.ToLower(value)), " ")
}
func mustReadFS(root fs.FS, name string) []byte {
	data, err := fs.ReadFile(root, name)
	if err != nil {
		return nil
	}
	return data
}

type CoverageSelfCheckReport struct {
	SchemaVersion  string `json:"schema_version"`
	DetectedReason string `json:"detected_reason"`
	Detected       bool   `json:"detected"`
	Cleared        bool   `json:"cleared"`
	Passed         bool   `json:"passed"`
}

func CoverageSelfCheck() CoverageSelfCheckReport {
	fixture := fstest.MapFS{"public/widget.go": {Data: []byte("package public\n\ntype AgentKernel struct{}\ntype NewlyExportedTerm struct{}\n")}}
	index := publicIndex
	surfaces := []PublicTerminologySurface{{Locator: "public", Kind: "go_package"}}
	before, beforeErr := InventoryCoverage(fixture, surfaces, index, nil)
	after, afterErr := InventoryCoverage(fixture, surfaces, index, []IncidentalTerm{{Term: "NewlyExportedTerm", Reason: "fixture-only API name"}})
	report := CoverageSelfCheckReport{SchemaVersion: CoverageSchemaVersion, DetectedReason: CoverageReasonMissingClassification}
	report.Detected = beforeErr == nil && len(before.Findings) == 1 && before.Findings[0].Term == "NewlyExportedTerm" && before.Findings[0].Reason == CoverageReasonMissingClassification
	report.Cleared = afterErr == nil && after.OK && len(after.Findings) == 0 && after.Incidental == 1
	report.Passed = report.Detected && report.Cleared
	return report
}
