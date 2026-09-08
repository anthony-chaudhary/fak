package debtlane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCoverageReceiptBreadthAndDepth(t *testing.T) {
	// Hermetic test verifying >= 9 surface classes and >= 15 detector dimensions
	lanes := []DebtLane{
		{Lane: "core_gate", UnitOfWork: "internal/gate", Criticality: CriticalityCore},
		{Lane: "pub_sdk", UnitOfWork: "pkg/sdk", Criticality: CriticalityEnabling},
		{Lane: "plat_disp", UnitOfWork: "platform/disp", Criticality: CriticalityEnabling},
		{Lane: "cmd_fak", UnitOfWork: "cmd/fak", Criticality: CriticalityEnabling},
		{Lane: "tools_doc", UnitOfWork: "tools/doc", Criticality: CriticalityStewardship},
		{Lane: "skill_audit", UnitOfWork: ".claude/skills/audit", Criticality: CriticalityStewardship},
		{Lane: "wf_ci", UnitOfWork: ".github/workflows", Criticality: CriticalityStewardship},
		{Lane: "ex_demo", UnitOfWork: "examples/demo", Criticality: CriticalityPeripheral},
		{Lane: "doc_guide", UnitOfWork: "docs/guide", Criticality: CriticalityStewardship},
	}

	var findings []FindingProvenance
	receipt := BuildCoverageReceipt("/test/workspace", "fak", lanes, findings, 50)

	if receipt.Schema != CoverageReceiptSchema {
		t.Errorf("expected schema %q, got %q", CoverageReceiptSchema, receipt.Schema)
	}

	// Verify >= 9 surface classes inventoried
	if receipt.ObservedBreadth < 9 {
		t.Errorf("expected at least 9 observed surface classes, got %d: %v",
			receipt.ObservedBreadth, receipt.SurfaceClasses)
	}
	if receipt.BreadthRatio < 3.0 {
		t.Errorf("expected breadth ratio >= 3.0x, got %.2f", receipt.BreadthRatio)
	}

	// Verify >= 15 independent detector dimensions evaluated
	if receipt.ObservedDepth < 15 {
		t.Errorf("expected at least 15 observed detector dimensions, got %d: %v",
			receipt.ObservedDepth, receipt.Dimensions)
	}
	if receipt.DepthRatio < 3.0 {
		t.Errorf("expected depth ratio >= 3.0x, got %.2f", receipt.DepthRatio)
	}

	// Verify all standard surface classes are present
	for _, expected := range StandardSurfaceClasses {
		found := false
		for _, s := range receipt.SurfaceClasses {
			if s == string(expected) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected surface class %q in receipt", expected)
		}
	}

	// Verify all standard detector dimensions are present
	for _, expected := range StandardDetectorDimensions {
		found := false
		for _, d := range receipt.Dimensions {
			if d == string(expected) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected detector dimension %q in receipt", expected)
		}
	}
}

func TestAdversarialCleanFixture(t *testing.T) {
	tmp := t.TempDir()
	cleanDir := filepath.Join(tmp, "internal", "cleanpkg")
	if err := os.MkdirAll(cleanDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Clean Go code: clean comments, exported doc, no TODOs, no unsafe, no os/exec
	code := `// Package cleanpkg provides verified clean functionality.
package cleanpkg

// Engine computes clean calculations.
type Engine struct{}

// Calculate returns a verified integer.
func (e *Engine) Calculate(x int) int {
	return x * 2
}
`
	if err := os.WriteFile(filepath.Join(cleanDir, "clean.go"), []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}

	// Clean test with real assertions and benchmarks
	testCode := `package cleanpkg

import "testing"

func TestCalculate(t *testing.T) {
	e := &Engine{}
	got := e.Calculate(2)
	if got != 4 {
		t.Fatalf("expected 4, got %d", got)
	}
}

func BenchmarkCalculate(b *testing.B) {
	e := &Engine{}
	for i := 0; i < b.N; i++ {
		_ = e.Calculate(i)
	}
}
`
	if err := os.WriteFile(filepath.Join(cleanDir, "clean_test.go"), []byte(testCode), 0o644); err != nil {
		t.Fatal(err)
	}

	lane := DebtLane{
		Lane:        "cleanpkg",
		UnitOfWork:  "internal/cleanpkg",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:        true,
			HasTests:       true,
			TestFilesCount: 1,
			Integrated:     true,
			Dogfooded:      true,
			Benchmarked:    true,
			CodeLines:      25,
			CommentLines:   3,
			CommentRatio:   0.10,
			ExcessComments: false,
		},
	}

	findings := InspectUnitDetectors(&lane, cleanDir)
	if len(findings) != 0 {
		t.Fatalf("adversarial clean fixture must have 0 findings, got %d: %+v", len(findings), findings)
	}
}

func TestSeededMixedSurfaceFixture(t *testing.T) {
	tmp := t.TempDir()

	// 1. Seeded internal Go package with thin tests and stub debt
	pkgDir := filepath.Join(tmp, "internal", "seeded")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}

	seededCode := `package seeded

// TODO: finish this implementation later
func Compute() {
	panic("not implemented")
}
`
	if err := os.WriteFile(filepath.Join(pkgDir, "seeded.go"), []byte(seededCode), 0o644); err != nil {
		t.Fatal(err)
	}

	// Thin test: test function with zero assertions!
	thinTest := `package seeded

import "testing"

func TestCompute(t *testing.T) {
	x := 1
	_ = x
}
`
	if err := os.WriteFile(filepath.Join(pkgDir, "seeded_test.go"), []byte(thinTest), 0o644); err != nil {
		t.Fatal(err)
	}

	laneGo := DebtLane{
		Lane:        "seeded",
		UnitOfWork:  "internal/seeded",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:           true,
			HasTests:          true,
			TestFilesCount:    1,
			Integrated:        true,
			Dogfooded:         false, // unproven!
			Benchmarked:       false, // unbenchmarked!
			ModularityDeficit: true,
			ModularityIssues:  []string{"god_func (>200 lines)"},
		},
	}

	findingsGo := InspectUnitDetectors(&laneGo, pkgDir)

	// Verify typed findings are emitted
	foundThin := false
	foundStub := false
	foundMod := false
	foundBench := false
	foundProof := false

	for _, f := range findingsGo {
		switch f.Dimension {
		case string(DimThinTests):
			foundThin = true
			if !strings.Contains(f.Message, "thin test hazard") {
				t.Errorf("unexpected thin test message: %s", f.Message)
			}
		case string(DimStubDebt):
			foundStub = true
			if !strings.Contains(f.Message, "stub debt") {
				t.Errorf("unexpected stub debt message: %s", f.Message)
			}
		case string(DimModularityDeficit):
			foundMod = true
		case string(DimBenchmarkStatus):
			foundBench = true
		case string(DimProofStatus):
			foundProof = true
		}
	}

	if !foundThin {
		t.Errorf("missing expected DimThinTests finding in: %+v", findingsGo)
	}
	if !foundStub {
		t.Errorf("missing expected DimStubDebt finding in: %+v", findingsGo)
	}
	if !foundMod {
		t.Errorf("missing expected DimModularityDeficit finding in: %+v", findingsGo)
	}
	if !foundBench {
		t.Errorf("missing expected DimBenchmarkStatus finding in: %+v", findingsGo)
	}
	if !foundProof {
		t.Errorf("missing expected DimProofStatus finding in: %+v", findingsGo)
	}

	// 2. Seeded skill surface with missing frontmatter and verification
	skillDir := filepath.Join(tmp, ".claude", "skills", "seeded_skill")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("# Bad Skill Without Frontmatter\nNo verification commands anywhere.\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	laneSkill := DebtLane{
		Lane:        "skill_seeded",
		UnitOfWork:  ".claude/skills/seeded_skill",
		Criticality: CriticalityStewardship,
	}
	findingsSkill := InspectUnitDetectors(&laneSkill, skillDir)

	foundSkillFM := false
	foundSkillVerify := false
	for _, f := range findingsSkill {
		if strings.Contains(f.Message, "missing YAML frontmatter") {
			foundSkillFM = true
		}
		if strings.Contains(f.Message, "missing verification") {
			foundSkillVerify = true
		}
	}
	if !foundSkillFM {
		t.Errorf("missing expected skill frontmatter finding: %+v", findingsSkill)
	}
	if !foundSkillVerify {
		t.Errorf("missing expected skill verification finding: %+v", findingsSkill)
	}

	// 3. Seeded workflow surface with unbounded timeout
	wfDir := filepath.Join(tmp, ".github", "workflows")
	if err := os.MkdirAll(wfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	badWf := `name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: make test
`
	if err := os.WriteFile(filepath.Join(wfDir, "ci.yml"), []byte(badWf), 0o644); err != nil {
		t.Fatal(err)
	}

	laneWf := DebtLane{
		Lane:        "workflows",
		UnitOfWork:  ".github/workflows",
		Criticality: CriticalityStewardship,
	}
	findingsWf := InspectUnitDetectors(&laneWf, wfDir)
	foundTimeout := false
	for _, f := range findingsWf {
		if strings.Contains(f.Message, "unbounded CI job timeout") {
			foundTimeout = true
		}
	}
	if !foundTimeout {
		t.Errorf("missing expected workflow timeout finding: %+v", findingsWf)
	}
}
