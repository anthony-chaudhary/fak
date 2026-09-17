package debtlane

// scan_surface_evidence.go holds the per-surface evidence inspectors for the non-Go
// repository surfaces (cmd/, tools/, skills, workflows, examples, docs, and unrecognized
// source-bearing roots). Split out of scan.go to keep that file under the god-file ceiling;
// same package, no API change.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

func inspectCmdEvidence(cmdDir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance
	fset := token.NewFileSet()

	_ = filepath.WalkDir(cmdDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(cmdDir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceCmd),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != cmdDir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(cmdDir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceCmd),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}
		name := d.Name()
		if !strings.HasSuffix(name, ".go") {
			return nil
		}

		if strings.HasSuffix(name, "_test.go") {
			testNode, err := parser.ParseFile(fset, path, data, 0)
			if err == nil {
				hasRealTests := false
				for _, decl := range testNode.Decls {
					fn, ok := decl.(*ast.FuncDecl)
					if !ok {
						continue
					}
					if strings.HasPrefix(fn.Name.Name, "Test") && fn.Body != nil && len(fn.Body.List) > 0 {
						hasRealTests = true
					}
					if strings.HasPrefix(fn.Name.Name, "Benchmark") {
						if isSubstantiveBenchmark(fn) {
							ev.Benchmarked = true
						}
					}
				}
				if hasRealTests {
					ev.HasTests = true
					ev.TestFilesCount++
				}
			}
			return nil
		}

		ev.FilesCount++
		ev.CodeLines += countNonEmptyLines(data)
		ev.HasCode = true
		stubs := countStubMarkers(path)
		if stubs > 0 {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(SurfaceCmd),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("stub debt: %d unimplemented/TODO marker(s) found", stubs),
			})
			ev.ModularityDeficit = true
		}
		return nil
	})

	if ev.HasCode {
		ev.Integrated = true
		if ev.HasTests {
			ev.Dogfooded = true
		}
	}

	return ev, findings
}

func inspectToolEvidence(toolsDir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance
	hasStubs := false
	fset := token.NewFileSet()

	_ = filepath.WalkDir(toolsDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(toolsDir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceTools),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != toolsDir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(toolsDir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceTools),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}

		name := d.Name()
		if strings.HasSuffix(name, ".go") {
			isTest := strings.HasSuffix(name, "_test.go")
			if isTest {
				testNode, err := parser.ParseFile(fset, path, data, 0)
				if err == nil {
					thinCount := 0
					testCount := 0
					for _, decl := range testNode.Decls {
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
					}
					if testCount > 0 && thinCount == testCount {
						findings = append(findings, FindingProvenance{
							Dimension: string(DimThinTests),
							Surface:   string(SurfaceTools),
							Lane:      laneName,
							Path:      filepath.Join(unitOfWork, rel),
							Severity:  "warning",
							Message:   fmt.Sprintf("thin test hazard: %d test function(s) contain zero assertions", thinCount),
						})
					} else if testCount > 0 {
						ev.HasTests = true
						ev.TestFilesCount++
					}
				}
				return nil
			}
			ev.FilesCount++
			ev.CodeLines += countNonEmptyLines(data)
			ev.HasCode = true
			stubs := countStubMarkers(path)
			if stubs > 0 {
				hasStubs = true
				findings = append(findings, FindingProvenance{
					Dimension: string(DimStubDebt),
					Surface:   string(SurfaceTools),
					Lane:      laneName,
					Path:      filepath.Join(unitOfWork, rel),
					Severity:  "warning",
					Message:   fmt.Sprintf("stub debt: %d unimplemented/TODO marker(s) found", stubs),
				})
			}
		} else if strings.HasSuffix(name, ".py") {
			isTest := strings.HasSuffix(name, "_test.py") || strings.HasPrefix(name, "test_")
			if isTest {
				ev.HasTests = true
				ev.TestFilesCount++
				return nil
			}
			ev.FilesCount++
			ev.CodeLines += countNonEmptyLines(data)
			ev.HasCode = true
			content := string(data)
			if strings.Contains(content, "TODO") || strings.Contains(content, "FIXME") || strings.Contains(content, "panic(") {
				hasStubs = true
				findings = append(findings, FindingProvenance{
					Dimension: string(DimStubDebt),
					Surface:   string(SurfaceTools),
					Lane:      laneName,
					Path:      filepath.Join(unitOfWork, rel),
					Severity:  "warning",
					Message:   "stub debt: TODO or placeholder found in tool script",
				})
			}
		} else if strings.HasSuffix(name, ".sh") || strings.HasSuffix(name, ".ps1") {
			ev.FilesCount++
			ev.CodeLines += countNonEmptyLines(data)
			ev.HasCode = true
		}
		return nil
	})

	if ev.HasCode {
		if ev.HasTests && !hasStubs {
			ev.Integrated = true
			ev.Dogfooded = true
		} else if hasStubs {
			ev.ModularityDeficit = true
		}
	}

	return ev, findings
}

func inspectSkillEvidence(skillDir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance

	var skillMdData []byte
	var skillMdFound bool

	_ = filepath.WalkDir(skillDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(skillDir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceSkills),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != skillDir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(skillDir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceSkills),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if strings.EqualFold(d.Name(), "SKILL.md") && strings.Contains(string(data), "generated-by: fak project-assets sync") {
			skillMdData = data
			skillMdFound = true
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}
		ev.FilesCount++
		ev.CodeLines += countNonEmptyLines(data)
		if strings.EqualFold(d.Name(), "SKILL.md") {
			skillMdData = data
			skillMdFound = true
		}
		return nil
	})

	if skillMdFound {
		if strings.Contains(string(skillMdData), "generated-by: fak project-assets sync") {
			ev.Generated = true
		}
		ev.HasCode = true
		content := string(skillMdData)
		if strings.HasPrefix(content, "---\n") {
			ev.Integrated = true
		} else {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimWiringStatus),
				Surface:   string(SurfaceSkills),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, "SKILL.md"),
				Severity:  "critical",
				Message:   "missing YAML frontmatter in SKILL.md",
			})
		}

		if strings.Contains(content, "## Verification") || strings.Contains(content, "## Witness") ||
			strings.Contains(content, "go test") || strings.Contains(content, "python tools/") {
			ev.HasTests = true
			ev.TestFilesCount = 1
			ev.Dogfooded = true
		} else {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimProofStatus),
				Surface:   string(SurfaceSkills),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, "SKILL.md"),
				Severity:  "warning",
				Message:   "missing verification or witness commands in SKILL.md",
			})
		}

		if strings.Contains(content, "TODO") || strings.Contains(content, "FIXME") {
			ev.ModularityDeficit = true
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(SurfaceSkills),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, "SKILL.md"),
				Severity:  "warning",
				Message:   "stub debt: TODO or placeholder found in SKILL.md",
			})
		}
	}

	return ev, findings
}

func inspectWorkflowEvidence(wfDir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance
	hasUnboundedTimeout := false

	_ = filepath.WalkDir(wfDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(wfDir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceWorkflows),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != wfDir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".yml") && !strings.HasSuffix(d.Name(), ".yaml") {
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(wfDir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceWorkflows),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}
		ev.FilesCount++
		ev.CodeLines += countNonEmptyLines(data)
		ev.HasCode = true

		content := string(data)
		if !strings.Contains(content, "timeout-minutes:") {
			hasUnboundedTimeout = true
			findings = append(findings, FindingProvenance{
				Dimension: string(DimBlastRadius),
				Surface:   string(SurfaceWorkflows),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   "unbounded CI job timeout: workflow job missing timeout-minutes",
			})
		}
		return nil
	})

	if ev.HasCode {
		if !hasUnboundedTimeout {
			ev.Integrated = true
			ev.HasTests = true
			ev.Dogfooded = true
			ev.TestFilesCount = ev.FilesCount
		} else {
			ev.ModularityDeficit = true
		}
	}

	return ev, findings
}

func inspectExampleEvidence(exDir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance
	hasStubs := false

	_ = filepath.WalkDir(exDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(exDir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceExamples),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != exDir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(exDir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceExamples),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}
		ev.FilesCount++
		ev.CodeLines += countNonEmptyLines(data)
		ev.HasCode = true

		content := string(data)
		if strings.Contains(content, "TODO") || strings.Contains(content, "panic(\"not implemented\")") {
			hasStubs = true
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(SurfaceExamples),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   "stub debt: unimplemented/TODO marker found in example",
			})
		}
		return nil
	})

	if ev.HasCode {
		if !hasStubs && ev.CodeLines >= 20 {
			ev.Integrated = true
			ev.HasTests = true
			ev.Dogfooded = true
		} else if hasStubs {
			ev.ModularityDeficit = true
		}
	}

	return ev, findings
}

func inspectDocEvidence(docsDir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance
	hasStubs := false

	_ = filepath.WalkDir(docsDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(docsDir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceDocs),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != docsDir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(docsDir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(SurfaceDocs),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".md") && !strings.HasSuffix(d.Name(), ".rst") && !strings.HasSuffix(d.Name(), ".txt") {
			return nil
		}
		ev.FilesCount++
		ev.CodeLines += countNonEmptyLines(data)
		ev.HasCode = true

		content := strings.TrimSpace(string(data))
		lines := strings.Split(content, "\n")
		if len(lines) < 10 && (strings.Contains(content, "TODO") || strings.Contains(content, "stub")) {
			hasStubs = true
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(SurfaceDocs),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   "stub doc: documentation file is an unfinished placeholder stub",
			})
		}
		return nil
	})

	if ev.HasCode {
		ev.Documented = true
		if !hasStubs {
			ev.Integrated = true
			ev.HasTests = true
			ev.Dogfooded = true
		} else {
			ev.ModularityDeficit = true
		}
	}

	return ev, findings
}

func inspectUnsupportedRootEvidence(dir, unitOfWork, laneName string) (Evidence, []FindingProvenance) {
	var ev Evidence
	var findings []FindingProvenance

	surface := classifySurface(unitOfWork)

	findings = append(findings, FindingProvenance{
		Dimension: string(DimCoverageDebt),
		Surface:   string(surface),
		Lane:      laneName,
		Path:      unitOfWork,
		Severity:  "warning",
		Message:   fmt.Sprintf("unsupported surface root: directory %s is not a recognized repository surface", unitOfWork),
	})

	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			rel, _ := filepath.Rel(dir, path)
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(surface),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", walkErr),
			})
			return nil
		}
		if d.IsDir() {
			if path != dir && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		rel, _ := filepath.Rel(dir, path)
		if err != nil {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimCoverageDebt),
				Surface:   string(surface),
				Lane:      laneName,
				Path:      filepath.Join(unitOfWork, rel),
				Severity:  "warning",
				Message:   fmt.Sprintf("unreadable file: %v", err),
			})
			return nil
		}
		if isGeneratedArtifact(path, data) {
			return nil
		}
		ev.FilesCount++
		ev.CodeLines += countNonEmptyLines(data)
		ev.HasCode = true
		return nil
	})

	return ev, findings
}
