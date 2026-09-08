package debtlane

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// UnitDetectorResult holds all detected findings and evaluated dimensions for a unit of work.
type UnitDetectorResult struct {
	Lane       string
	UnitOfWork string
	Surface    SurfaceClass
	Findings   []FindingProvenance
	Dimensions []DetectorDimension
}

// InspectUnitDetectors runs all 15 detector dimensions on a unit of work.
func InspectUnitDetectors(lane *DebtLane, unitAbsDir string, compRoots ...string) []FindingProvenance {
	if lane == nil {
		return nil
	}

	surface := classifySurface(lane.UnitOfWork)
	var findings []FindingProvenance

	// 1. Headline 5 dimensions
	if !lane.Evidence.HasTests {
		sev := "warning"
		if lane.Criticality == CriticalityCore {
			sev = "critical"
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimTestStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  sev,
			Message:   "missing unit tests: no passing *_test.go found",
		})
	}

	if lane.Evidence.ExcessComments {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimCommentHygiene),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   fmt.Sprintf("excess comment bloat or formulaic comment gaming (%.1f%% comments)", lane.Evidence.CommentRatio*100),
		})
	}

	if !lane.Evidence.Integrated {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimWiringStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   "disconnected wiring: unreachable from production command graph",
		})
	}

	if !lane.Evidence.Dogfooded && (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling) {
		sev := "warning"
		if lane.Criticality == CriticalityCore {
			sev = "critical"
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimProofStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  sev,
			Message:   "unproven runtime: missing real execution proof or dogfooding receipt in runtime-proofs.json",
		})
	}

	if !lane.Evidence.Benchmarked && (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling) {
		sev := "warning"
		if lane.Criticality == CriticalityCore {
			sev = "critical"
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimBenchmarkStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  sev,
			Message:   "unmeasured performance: missing substantive benchmark function or authority entry",
		})
	}

	// 6. Modularity deficit
	if lane.Evidence.ModularityDeficit {
		msg := "modularity deficit: monolithic god-files (>1500 lines) or god-functions (>200 lines)"
		if len(lane.Evidence.ModularityIssues) > 0 {
			msg = strings.Join(lane.Evidence.ModularityIssues, "; ")
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimModularityDeficit),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "critical",
			Message:   msg,
		})
	}

	// 7. Model coupling
	if lane.Evidence.HasModelHardcoding {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimModelCoupling),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   "model coupling: hardcoded model family tokens outside model/ or hfhub/ packages",
		})
	}

	// 8. Blast radius
	if lane.Evidence.DependentsCount > 25 {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimBlastRadius),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   fmt.Sprintf("high blast radius: %d inbound dependents require strict API stability", lane.Evidence.DependentsCount),
		})
	}

	// 9-15: AST-grounded inspections for Go files
	if unitAbsDir != "" {
		astFindings := inspectGoASTDetectors(lane, unitAbsDir, surface)
		findings = append(findings, astFindings...)
	}

	// Non-Go surface-appropriate checks
	if surface == SurfaceSkills && unitAbsDir != "" {
		skillFindings := inspectSkillSurface(lane, unitAbsDir)
		findings = append(findings, skillFindings...)
	} else if surface == SurfaceWorkflows && unitAbsDir != "" {
		workflowFindings := inspectWorkflowSurface(lane, unitAbsDir)
		findings = append(findings, workflowFindings...)
	} else if surface == SurfaceDocs && unitAbsDir != "" {
		docFindings := inspectDocSurface(lane, unitAbsDir)
		findings = append(findings, docFindings...)
	}

	return findings
}

func inspectGoASTDetectors(lane *DebtLane, unitDir string, surface SurfaceClass) []FindingProvenance {
	var findings []FindingProvenance
	fset := token.NewFileSet()

	entries, err := os.ReadDir(unitDir)
	if err != nil {
		return findings
	}

	hasFuzzOrRace := false
	isProtocolOrConcurrency := strings.Contains(lane.Lane, "gateway") ||
		strings.Contains(lane.Lane, "engine") ||
		strings.Contains(lane.Lane, "sched") ||
		strings.Contains(lane.Lane, "sync") ||
		strings.Contains(lane.Lane, "protocol") ||
		strings.Contains(lane.Lane, "ctxmmu")

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		path := filepath.Join(unitDir, e.Name())
		relPath := filepath.Join(lane.UnitOfWork, e.Name())

		// Check for fuzz test files
		if strings.Contains(e.Name(), "fuzz") || strings.Contains(e.Name(), "race") {
			hasFuzzOrRace = true
		}

		// 9. Thin test detection: inspect test files for real assertions
		if strings.HasSuffix(e.Name(), "_test.go") {
			fileNode, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
			if err == nil {
				thinCount := 0
				testCount := 0
				for _, decl := range fileNode.Decls {
					fn, ok := decl.(*ast.FuncDecl)
					if !ok || fn.Body == nil {
						continue
					}
					if strings.HasPrefix(fn.Name.Name, "Test") {
						testCount++
						if !hasTestAssertions(fn.Body) {
							thinCount++
						}
					}
					if strings.HasPrefix(fn.Name.Name, "Fuzz") {
						hasFuzzOrRace = true
					}
				}
				if testCount > 0 && thinCount == testCount {
					findings = append(findings, FindingProvenance{
						Dimension: string(DimThinTests),
						Surface:   string(surface),
						Lane:      lane.Lane,
						Path:      relPath,
						Severity:  "warning",
						Message:   fmt.Sprintf("thin test hazard: %d test function(s) contain zero assertions", thinCount),
					})
				}
			}
			continue
		}

		// Non-test Go files
		fileNode, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			continue
		}

		// 10. Stub debt detection: TODO/FIXME/unimplemented panics
		stubCount := countStubMarkers(path)
		if stubCount > 0 {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(surface),
				Lane:      lane.Lane,
				Path:      relPath,
				Severity:  "warning",
				Message:   fmt.Sprintf("stub debt: %d unimplemented/TODO marker(s) found", stubCount),
			})
		}

		// 12. Unsafe usage in non-core
		if lane.Criticality != CriticalityCore && lane.Criticality != CriticalityEnabling {
			for _, imp := range fileNode.Imports {
				if imp.Path != nil && imp.Path.Value == `"unsafe"` {
					findings = append(findings, FindingProvenance{
						Dimension: string(DimUnsafeUsage),
						Surface:   string(surface),
						Lane:      lane.Lane,
						Path:      relPath,
						Severity:  "warning",
						Message:   "unsafe usage: importing package unsafe in peripheral/stewardship unit",
					})
					break
				}
			}
		}

		// 13. Subprocess execution in internal libraries
		if surface == SurfaceInternal || surface == SurfacePkg {
			for _, imp := range fileNode.Imports {
				if imp.Path != nil && imp.Path.Value == `"os/exec"` {
					// Allowed in driver/daemon packages, flagged in pure algorithmic/compute leaves
					if !strings.Contains(lane.Lane, "dispatch") && !strings.Contains(lane.Lane, "process") &&
						!strings.Contains(lane.Lane, "flow") && !strings.Contains(lane.Lane, "bridge") {
						findings = append(findings, FindingProvenance{
							Dimension: string(DimSubprocessExec),
							Surface:   string(surface),
							Lane:      lane.Lane,
							Path:      relPath,
							Severity:  "info",
							Message:   "subprocess execution: os/exec imported in library leaf; prefer native Go abstractions",
						})
					}
					break
				}
			}
		}
	}

	// 11. Undocumented exported contracts
	if lane.Evidence.ExportedSymbols > 5 && lane.Evidence.DocumentedExports == 0 {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimUndocumentedExports),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "info",
			Message:   fmt.Sprintf("undocumented exports: %d exported symbol(s) lack godoc documentation", lane.Evidence.ExportedSymbols),
		})
	}

	// 14. Race/Fuzz status for concurrency/protocol leaves
	if isProtocolOrConcurrency && !hasFuzzOrRace && lane.Evidence.HasTests {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimRaceFuzzStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "info",
			Message:   "race/fuzz gap: protocol or concurrency-critical package lacks fuzz or dedicated race tests",
		})
	}

	// 15. Stale or missing performance proof
	if (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling) &&
		lane.Evidence.Integrated && !lane.Evidence.Benchmarked && !lane.Evidence.Dogfooded {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimStalePerfProof),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "critical",
			Message:   "stale performance proof: core/enabling unit missing from both benchmark authority and runtime proofs",
		})
	}

	return findings
}

func hasTestAssertions(body *ast.BlockStmt) bool {
	if body == nil || len(body.List) == 0 {
		return false
	}
	hasAssert := false
	ast.Inspect(body, func(n ast.Node) bool {
		if hasAssert {
			return false
		}
		switch call := n.(type) {
		case *ast.CallExpr:
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				name := sel.Sel.Name
				if name == "Error" || name == "Errorf" || name == "Fatal" || name == "Fatalf" ||
					name == "Fail" || name == "FailNow" || name == "Equal" || name == "True" ||
					name == "False" || name == "NoError" {
					hasAssert = true
					return false
				}
			}
		case *ast.IfStmt:
			// Often `if err != nil { t.Fatal(err) }` or `if got != want`
			hasAssert = true
			return false
		}
		return true
	})
	return hasAssert
}

func countStubMarkers(filePath string) int {
	f, err := os.Open(filePath)
	if err != nil {
		return 0
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lower := strings.ToLower(line)
		if strings.Contains(line, "TODO") || strings.Contains(line, "FIXME") || strings.Contains(line, "XXX") {
			count++
		} else if strings.Contains(lower, `panic("not implemented")`) ||
			strings.Contains(lower, `panic("unimplemented")`) ||
			strings.Contains(lower, `panic("todo")`) {
			count++
		}
	}
	return count
}

func inspectSkillSurface(lane *DebtLane, dir string) []FindingProvenance {
	var findings []FindingProvenance
	skillPath := filepath.Join(dir, "SKILL.md")
	data, err := os.ReadFile(skillPath)
	if err != nil {
		return findings
	}
	content := string(data)

	// Frontmatter check
	if !strings.HasPrefix(content, "---\n") {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimWiringStatus),
			Surface:   string(SurfaceSkills),
			Lane:      lane.Lane,
			Path:      filepath.Join(lane.UnitOfWork, "SKILL.md"),
			Severity:  "critical",
			Message:   "missing YAML frontmatter in SKILL.md",
		})
	}

	// Verification check
	if !strings.Contains(content, "## Verification") && !strings.Contains(content, "## Witness") &&
		!strings.Contains(content, "go test") && !strings.Contains(content, "python tools/") {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimProofStatus),
			Surface:   string(SurfaceSkills),
			Lane:      lane.Lane,
			Path:      filepath.Join(lane.UnitOfWork, "SKILL.md"),
			Severity:  "warning",
			Message:   "missing verification or witness commands in SKILL.md",
		})
	}

	return findings
}

func inspectWorkflowSurface(lane *DebtLane, dir string) []FindingProvenance {
	var findings []FindingProvenance
	entries, err := os.ReadDir(dir)
	if err != nil {
		return findings
	}
	for _, e := range entries {
		if e.IsDir() || (!strings.HasSuffix(e.Name(), ".yml") && !strings.HasSuffix(e.Name(), ".yaml")) {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := string(data)
		if !strings.Contains(content, "timeout-minutes:") {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimBlastRadius),
				Surface:   string(SurfaceWorkflows),
				Lane:      lane.Lane,
				Path:      filepath.Join(lane.UnitOfWork, e.Name()),
				Severity:  "warning",
				Message:   "unbounded CI job timeout: workflow job missing timeout-minutes",
			})
		}
	}
	return findings
}

func inspectDocSurface(lane *DebtLane, dir string) []FindingProvenance {
	var findings []FindingProvenance
	entries, err := os.ReadDir(dir)
	if err != nil {
		return findings
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := strings.TrimSpace(string(data))
		lines := strings.Split(content, "\n")
		if len(lines) < 10 && (strings.Contains(content, "TODO") || strings.Contains(content, "stub")) {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(SurfaceDocs),
				Lane:      lane.Lane,
				Path:      filepath.Join(lane.UnitOfWork, e.Name()),
				Severity:  "warning",
				Message:   "stub doc: documentation file is an unfinished placeholder stub",
			})
		}
	}
	return findings
}
