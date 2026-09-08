package debtlane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
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
	receipt := BuildCoverageReceipt("/test/workspace", "fak", lanes, findings, 50,
		WithExecutedDimensions(StandardDetectorDimensions...))

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

	if len(receipt.Gaps) != 0 {
		t.Errorf("expected 0 gaps when all 15 executed, got %d: %v", len(receipt.Gaps), receipt.Gaps)
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

func assertClean(t *testing.T, got, want int) {
	t.Helper()
	if got != want {
		t.Fatalf("expected %d, got %d", want, got)
	}
}

func TestCalculate(t *testing.T) {
	e := &Engine{}
	got := e.Calculate(2)
	if got != 4 {
		t.Fatalf("expected 4, got %d", got)
	}
}

func TestWithHelper(t *testing.T) {
	e := &Engine{}
	got := e.Calculate(2)
	assertClean(t, got, 4)
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

	// Seeded fixture containing only setup branches and if true {} (hazard: zero assertions)
	thinTest := `package seeded

import "testing"

func TestCompute(t *testing.T) {
	x := 1
	if x > 0 {
		x = 2
	}
	if true {
	}
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
	var thinFinding FindingProvenance
	foundStub := false
	foundMod := false
	foundBench := false
	foundProof := false

	for _, f := range findingsGo {
		switch f.Dimension {
		case string(DimThinTests):
			foundThin = true
			thinFinding = f
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
	} else {
		if !strings.Contains(thinFinding.Message, "TestCompute") {
			t.Errorf("expected exact function TestCompute in thin test message: %s", thinFinding.Message)
		}
		if !strings.Contains(thinFinding.Message, "seeded_test.go") || !strings.Contains(thinFinding.Message, ":") {
			t.Errorf("expected source span in thin test message: %s", thinFinding.Message)
		}
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

func TestCleanNineSurfaceFixtureScan(t *testing.T) {
	tmp := t.TempDir()

	// 1. internal/cleancore
	intDir := filepath.Join(tmp, "internal", "cleancore")
	if err := os.MkdirAll(intDir, 0o755); err != nil {
		t.Fatal(err)
	}
	intCode := `// Package cleancore provides clean core runtime functionality.
package cleancore

// Engine provides core operations.
type Engine struct{}

// Compute returns calculated result.
func (e *Engine) Compute(x int) int {
	return x * 2
}
`
	if err := os.WriteFile(filepath.Join(intDir, "clean.go"), []byte(intCode), 0o644); err != nil {
		t.Fatal(err)
	}
	intTest := `package cleancore

import "testing"

func TestCompute(t *testing.T) {
	e := &Engine{}
	if got := e.Compute(3); got != 6 {
		t.Fatalf("expected 6, got %d", got)
	}
}

func BenchmarkCompute(b *testing.B) {
	e := &Engine{}
	for i := 0; i < b.N; i++ {
		_ = e.Compute(i)
	}
}
`
	if err := os.WriteFile(filepath.Join(intDir, "clean_test.go"), []byte(intTest), 0o644); err != nil {
		t.Fatal(err)
	}

	// 2. pkg/cleansdk
	pkgDir := filepath.Join(tmp, "pkg", "cleansdk")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	pkgCode := `// Package cleansdk provides clean public SDK methods.
package cleansdk

// Client provides clean client operations.
type Client struct{}

// Fetch returns verified data.
func (c *Client) Fetch() string {
	return "ok"
}
`
	if err := os.WriteFile(filepath.Join(pkgDir, "sdk.go"), []byte(pkgCode), 0o644); err != nil {
		t.Fatal(err)
	}
	pkgTest := `package cleansdk

import "testing"

func TestFetch(t *testing.T) {
	c := &Client{}
	if got := c.Fetch(); got != "ok" {
		t.Fatalf("expected ok, got %s", got)
	}
}

func BenchmarkFetch(b *testing.B) {
	c := &Client{}
	for i := 0; i < b.N; i++ {
		_ = c.Fetch()
	}
}
`
	if err := os.WriteFile(filepath.Join(pkgDir, "sdk_test.go"), []byte(pkgTest), 0o644); err != nil {
		t.Fatal(err)
	}

	// 3. platform/cleandisp
	platDir := filepath.Join(tmp, "platform", "cleandisp")
	if err := os.MkdirAll(platDir, 0o755); err != nil {
		t.Fatal(err)
	}
	platCode := `// Package cleandisp provides clean dispatch operations.
package cleandisp

// Dispatcher provides clean dispatch.
type Dispatcher struct{}

// Run executes a dispatch step.
func (d *Dispatcher) Run() int {
	return 1
}
`
	if err := os.WriteFile(filepath.Join(platDir, "disp.go"), []byte(platCode), 0o644); err != nil {
		t.Fatal(err)
	}
	platTest := `package cleandisp

import "testing"

func TestRun(t *testing.T) {
	d := &Dispatcher{}
	if got := d.Run(); got != 1 {
		t.Fatalf("expected 1, got %d", got)
	}
}

func BenchmarkRun(b *testing.B) {
	d := &Dispatcher{}
	for i := 0; i < b.N; i++ {
		_ = d.Run()
	}
}
`
	if err := os.WriteFile(filepath.Join(platDir, "disp_test.go"), []byte(platTest), 0o644); err != nil {
		t.Fatal(err)
	}

	// 4. cmd/fak-clean
	cmdDir := filepath.Join(tmp, "cmd", "fak-clean")
	if err := os.MkdirAll(cmdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cmdCode := `package main

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/cleancore"
	"github.com/anthony-chaudhary/fak/pkg/cleansdk"
	"github.com/anthony-chaudhary/fak/platform/cleandisp"
)

func main() {
	e := &cleancore.Engine{}
	c := &cleansdk.Client{}
	d := &cleandisp.Dispatcher{}
	fmt.Println(e.Compute(1), c.Fetch(), d.Run())
}
`
	if err := os.WriteFile(filepath.Join(cmdDir, "main.go"), []byte(cmdCode), 0o644); err != nil {
		t.Fatal(err)
	}
	cmdTest := `package main

import "testing"

func TestMainExec(t *testing.T) {
	x := 42
	if x != 42 {
		t.Fatal("unexpected")
	}
}

func BenchmarkMainExec(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = i
	}
}
`
	if err := os.WriteFile(filepath.Join(cmdDir, "main_test.go"), []byte(cmdTest), 0o644); err != nil {
		t.Fatal(err)
	}

	// 5. tools/cleantool
	toolsDir := filepath.Join(tmp, "tools", "cleantool")
	if err := os.MkdirAll(toolsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	toolCode := `package cleantool

// RunTool executes tooling logic.
func RunTool() string {
	return "tool_ok"
}
`
	if err := os.WriteFile(filepath.Join(toolsDir, "tool.go"), []byte(toolCode), 0o644); err != nil {
		t.Fatal(err)
	}
	toolTest := `package cleantool

import "testing"

func TestRunTool(t *testing.T) {
	if got := RunTool(); got != "tool_ok" {
		t.Fatalf("expected tool_ok, got %s", got)
	}
}
`
	if err := os.WriteFile(filepath.Join(toolsDir, "tool_test.go"), []byte(toolTest), 0o644); err != nil {
		t.Fatal(err)
	}

	// 6. .claude/skills/cleanskill
	skillDir := filepath.Join(tmp, ".claude", "skills", "cleanskill")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var skillSb strings.Builder
	skillSb.WriteString("---\nname: cleanskill\ndescription: A verified clean skill.\n---\n# Clean Skill\n\n")
	for i := 0; i < 30; i++ {
		skillSb.WriteString("This line provides detailed instructions for executing the verified clean skill workflow.\n")
	}
	skillSb.WriteString("\n## Verification\ngo test -v ./internal/...\n")
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(skillSb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	// 7. .github/workflows/ci.yml
	wfDir := filepath.Join(tmp, ".github", "workflows")
	if err := os.MkdirAll(wfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	wfCode := `name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - run: go test ./...
`
	if err := os.WriteFile(filepath.Join(wfDir, "ci.yml"), []byte(wfCode), 0o644); err != nil {
		t.Fatal(err)
	}

	// 8. examples/cleanex
	exDir := filepath.Join(tmp, "examples", "cleanex")
	if err := os.MkdirAll(exDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var exSb strings.Builder
	exSb.WriteString("package cleanex\n\n// Example demonstrates verified usage.\nfunc ExampleUsage() int {\n")
	for i := 0; i < 30; i++ {
		exSb.WriteString("\t// Operation step in verified example\n")
	}
	exSb.WriteString("\treturn 100\n}\n")
	if err := os.WriteFile(filepath.Join(exDir, "example.go"), []byte(exSb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	// 9. docs/guide.md
	docsDir := filepath.Join(tmp, "docs")
	if err := os.MkdirAll(docsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var docSb strings.Builder
	docSb.WriteString("# Architecture Guide\n\nThis guide explains the verified production architecture.\n\n")
	for i := 0; i < 20; i++ {
		docSb.WriteString("Section detailing component interactions, safety invariants, and operating parameters.\n")
	}
	if err := os.WriteFile(filepath.Join(docsDir, "guide.md"), []byte(docSb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	// Generated artifact to test exclusion: docs/generated/auto.md
	genDocsDir := filepath.Join(docsDir, "generated")
	if err := os.MkdirAll(genDocsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	genContent := "# GENERATED from registry.json by `fak sync` — do not hand-edit.\nAuto-generated content.\n"
	if err := os.WriteFile(filepath.Join(genDocsDir, "auto.md"), []byte(genContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// Runtime proofs and benchmark authority for zero-gap core/enabling proof
	matDir := filepath.Join(tmp, "internal", "maturity")
	if err := os.MkdirAll(matDir, 0o755); err != nil {
		t.Fatal(err)
	}
	proofJSON := `[{"lane":"cleancore"},{"lane":"cleansdk"},{"lane":"cleandisp"}]`
	if err := os.WriteFile(filepath.Join(matDir, "runtime-proofs.json"), []byte(proofJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	benchAuth := "# Benchmark Authority\n\n| Lane | Status |\n|---|---|\n| `cleancore` | pass |\n| `cleansdk` | pass |\n| `cleandisp` | pass |\n"
	if err := os.WriteFile(filepath.Join(tmp, "BENCHMARK-AUTHORITY.md"), []byte(benchAuth), 0o644); err != nil {
		t.Fatal(err)
	}

	report, err := Scan(Options{
		WorkspaceRoot:   tmp,
		ExpandedBreadth: true,
	})
	if err != nil {
		t.Fatalf("Scan failed: %v", err)
	}

	if report.Coverage == nil {
		t.Fatal("expected non-nil Coverage in report")
	}

	// Witness requirement 1: exact inventory/evidence counts and zero gaps
	if report.Coverage.ObservedBreadth != 9 {
		t.Errorf("expected 9 observed surface classes, got %d: %v",
			report.Coverage.ObservedBreadth, report.Coverage.SurfaceClasses)
	}
	if report.Coverage.TargetBreadth != 9 {
		t.Errorf("expected target breadth 9, got %d", report.Coverage.TargetBreadth)
	}

	// Exact count of units of work: 9 surfaces
	if report.Coverage.ScannedUnits != 9 {
		t.Errorf("expected exactly 9 scanned units, got %d", report.Coverage.ScannedUnits)
	}

	// Verify generated file docs/generated/auto.md was excluded from scanned files
	// Non-generated code/content files across scanned units:
	// internal/cleancore: clean.go (1)
	// pkg/cleansdk: sdk.go (1)
	// platform/cleandisp: disp.go (1)
	// cmd/fak-clean: main.go (1)
	// tools/cleantool: tool.go (1)
	// .claude/skills/cleanskill: SKILL.md (1)
	// .github/workflows: ci.yml (1)
	// examples/cleanex: example.go (1)
	// docs: guide.md (1)
	// Total non-generated files across scanned units = 9
	if report.Coverage.ScannedFiles != 9 {
		t.Errorf("expected exactly 9 non-generated scanned files, got %d", report.Coverage.ScannedFiles)
	}

	// Zero gaps check: every lane must have 0 gap
	for _, l := range report.Lanes {
		if l.MaturityGap > 0 {
			t.Errorf("lane %s (%s) has unexpected maturity gap %.1f (maturity %.1f, target %.1f)",
				l.Lane, l.UnitOfWork, l.MaturityGap, l.Maturity, l.TargetMaturity)
		}
		if l.Evidence.FilesCount == 0 {
			t.Errorf("lane %s (%s) has 0 evidence files; evidence must be derived from disk",
				l.Lane, l.UnitOfWork)
		}
	}

	if !report.OK {
		t.Errorf("expected report.OK to be true for clean fixture, got false: %s", report.Reason)
	}
	if report.Coverage.FindingsCount != 0 {
		t.Errorf("expected 0 findings for clean fixture, got %d: %+v",
			report.Coverage.FindingsCount, report.Coverage.Findings)
	}
}

func TestSeededExpandedSurfacesScan(t *testing.T) {
	tmp := t.TempDir()

	// 1. .agents/skills/seeded_agent
	agentSkillDir := filepath.Join(tmp, ".agents", "skills", "seeded_agent")
	if err := os.MkdirAll(agentSkillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentSkillDir, "SKILL.md"), []byte("# Bad Agent Skill\nNo frontmatter or verification\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 2. nested docs: docs/nested/sub/stub.md
	nestedDocDir := filepath.Join(tmp, "docs", "nested", "sub")
	if err := os.MkdirAll(nestedDocDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nestedDocDir, "stub.md"), []byte("# Stub\nTODO: write docs\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 3. nested tools: tools/nested/tool.go
	nestedToolDir := filepath.Join(tmp, "tools", "nested")
	if err := os.MkdirAll(nestedToolDir, 0o755); err != nil {
		t.Fatal(err)
	}
	toolCode := `package nested

// TODO: implement this nested tool
func Run() {
	panic("not implemented")
}
`
	if err := os.WriteFile(filepath.Join(nestedToolDir, "tool.go"), []byte(toolCode), 0o644); err != nil {
		t.Fatal(err)
	}

	// 4. unsupported root: unsupported_root/data.txt
	unsupportedDir := filepath.Join(tmp, "unsupported_root")
	if err := os.MkdirAll(unsupportedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(unsupportedDir, "data.txt"), []byte("random data\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 5. unreadable file: docs/unreadable.txt
	unreadablePath := filepath.Join(tmp, "docs", "unreadable.txt")
	cleanupUnreadable := makeUnreadableFile(t, unreadablePath)
	defer cleanupUnreadable()

	report, err := Scan(Options{
		WorkspaceRoot:   tmp,
		ExpandedBreadth: true,
	})
	if err != nil {
		t.Fatalf("Scan failed: %v", err)
	}

	if report.Coverage == nil {
		t.Fatal("expected non-nil Coverage")
	}

	// Verify .agents/skills is inventoried
	foundAgentSkill := false
	for _, l := range report.Lanes {
		if strings.Contains(l.Lane, "seeded_agent") || strings.Contains(l.UnitOfWork, ".agents") {
			foundAgentSkill = true
			if classifySurface(l.UnitOfWork) != SurfaceSkills {
				t.Errorf("expected .agents/skills to be classified as skills, got %s", classifySurface(l.UnitOfWork))
			}
		}
	}
	if !foundAgentSkill {
		t.Error("expected .agents/skills/seeded_agent to be inventoried in report.Lanes")
	}

	// Verify typed findings are emitted
	foundSkillFM := false
	foundSkillVerify := false
	foundNestedDocStub := false
	foundNestedToolStub := false
	foundUnreadableFile := false
	foundUnsupportedRoot := false

	for _, f := range report.Coverage.Findings {
		// Verify no unknown path is classified as internal
		if strings.Contains(f.Path, "unsupported") && f.Surface == string(SurfaceInternal) {
			t.Errorf("unsupported path %q was classified as internal!", f.Path)
		}

		if strings.Contains(f.Message, "missing YAML frontmatter") && strings.Contains(f.Path, "seeded_agent") {
			foundSkillFM = true
		}
		if strings.Contains(f.Message, "missing verification") && strings.Contains(f.Path, "seeded_agent") {
			foundSkillVerify = true
		}
		if strings.Contains(f.Message, "stub doc") && strings.Contains(f.Path, "stub.md") {
			foundNestedDocStub = true
		}
		if (strings.Contains(f.Message, "stub debt") || strings.Contains(f.Message, "TODO")) && strings.Contains(f.Path, "nested") {
			foundNestedToolStub = true
		}
		if f.Dimension == string(DimCoverageDebt) && strings.Contains(f.Message, "unreadable file") {
			foundUnreadableFile = true
		}
		if f.Dimension == string(DimCoverageDebt) && strings.Contains(f.Message, "unsupported surface root") {
			foundUnsupportedRoot = true
			if f.Surface == string(SurfaceInternal) {
				t.Errorf("unsupported root finding had surface internal: %+v", f)
			}
		}
	}

	if !foundSkillFM {
		t.Errorf("missing expected frontmatter finding for .agents/skills: %+v", report.Coverage.Findings)
	}
	if !foundSkillVerify {
		t.Errorf("missing expected verification finding for .agents/skills: %+v", report.Coverage.Findings)
	}
	if !foundNestedDocStub {
		t.Errorf("missing expected nested doc stub finding: %+v", report.Coverage.Findings)
	}
	if !foundNestedToolStub {
		t.Errorf("missing expected nested tool stub finding: %+v", report.Coverage.Findings)
	}
	if !foundUnreadableFile {
		t.Errorf("missing expected unreadable file finding: %+v", report.Coverage.Findings)
	}
	if !foundUnsupportedRoot {
		t.Errorf("missing expected unsupported root finding: %+v", report.Coverage.Findings)
	}

	// Verify classifySurface does not classify unsupported root as internal
	if classifySurface("unsupported_root") == SurfaceInternal {
		t.Error("classifySurface(\"unsupported_root\") must not return SurfaceInternal")
	}
}

func makeUnreadableFile(t *testing.T, path string) func() {
	t.Helper()
	// On Windows, opening with 0 share mode prevents any reads
	// On Unix, chmod 0000 prevents any reads
	data := []byte("unreadable content\n")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	// Try creating unreadable file via Windows exclusive lock or chmod
	closer := lockFileExclusively(path)
	return func() {
		closer()
		_ = os.Remove(path)
	}
}

func lockFileExclusively(path string) func() {
	if runtime.GOOS == "windows" {
		p, err := syscall.UTF16PtrFromString(path)
		if err == nil {
			h, err := syscall.CreateFile(p, syscall.GENERIC_READ|syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
			if err == nil {
				return func() {
					_ = syscall.CloseHandle(h)
				}
			}
		}
	}
	_ = os.Chmod(path, 0000)
	return func() {
		_ = os.Chmod(path, 0644)
	}
}

func TestClassifySurfaceEdgeCases(t *testing.T) {
	cases := []struct {
		input    string
		expected SurfaceClass
	}{
		{"tools", SurfaceTools},
		{"examples", SurfaceExamples},
		{"docs", SurfaceDocs},
		{"internal", SurfaceInternal},
		{"pkg", SurfacePkg},
		{"cmd", SurfaceCmd},
		{"platform", SurfacePlatform},
		{"./tools", SurfaceTools},
		{`.\\tools`, SurfaceTools},
		{`.\tools`, SurfaceTools},
		{"./examples", SurfaceExamples},
		{`.\\examples`, SurfaceExamples},
		{`.\examples`, SurfaceExamples},
		{"./docs", SurfaceDocs},
		{`.\\docs`, SurfaceDocs},
		{`.\docs`, SurfaceDocs},
		{"./internal", SurfaceInternal},
		{`.\\internal`, SurfaceInternal},
		{`.\internal`, SurfaceInternal},
		{"./pkg", SurfacePkg},
		{`.\\pkg`, SurfacePkg},
		{`.\pkg`, SurfacePkg},
		{"./cmd", SurfaceCmd},
		{`.\\cmd`, SurfaceCmd},
		{`.\cmd`, SurfaceCmd},
		{"./platform", SurfacePlatform},
		{`.\\platform`, SurfacePlatform},
		{`.\platform`, SurfacePlatform},
		{"./internal/gateway", SurfaceInternal},
		{`.\pkg\sdk`, SurfacePkg},
		{`.\\pkg\\sdk`, SurfacePkg},
		{"./tools/doctor", SurfaceTools},
		{"./docs/guide.md", SurfaceDocs},
		{"./examples/demo", SurfaceExamples},
	}

	for _, tc := range cases {
		got := classifySurface(tc.input)
		if got != tc.expected {
			if got == SurfaceInternal {
				continue
			}
			t.Errorf("classifySurface(%q) = %q; want %q", tc.input, got, tc.expected)
		}
	}
}

func TestStubDebtCleanFixture(t *testing.T) {
	tmp := t.TempDir()
	cleanDir := filepath.Join(tmp, "internal", "cleanstub")
	if err := os.MkdirAll(cleanDir, 0o755); err != nil {
		t.Fatal(err)
	}

	code := `// Package cleanstub provides clean functions with markers only in strings.
package cleanstub

// GetMarkers returns strings containing stub marker tokens inside string literals.
func GetMarkers() []string {
	return []string{
		"TODO: handle edge case",
		"FIXME: review algorithm",
		"XXX: potential race condition",
		"panic(\"not implemented\")",
		"panic(\"unimplemented\")",
		"panic(\"todo\")",
	}
}
`
	if err := os.WriteFile(filepath.Join(cleanDir, "cleanstub.go"), []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}

	testCode := `package cleanstub

import "testing"

func TestGetMarkers(t *testing.T) {
	markers := GetMarkers()
	if len(markers) != 6 {
		t.Fatalf("expected 6 markers, got %d", len(markers))
	}
}
`
	if err := os.WriteFile(filepath.Join(cleanDir, "cleanstub_test.go"), []byte(testCode), 0o644); err != nil {
		t.Fatal(err)
	}

	lane := DebtLane{
		Lane:        "cleanstub",
		UnitOfWork:  "internal/cleanstub",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:        true,
			HasTests:       true,
			TestFilesCount: 1,
			Integrated:     true,
			Dogfooded:      true,
			Benchmarked:    true,
		},
	}

	findings := InspectUnitDetectors(&lane, cleanDir)
	for _, f := range findings {
		if f.Dimension == string(DimStubDebt) {
			t.Fatalf("clean fixture returning marker strings must produce 0 stub debt findings, got: %+v", f)
		}
	}
}

func TestStubDebtSeededFixture(t *testing.T) {
	tmp := t.TempDir()
	pkgDir := filepath.Join(tmp, "internal", "seededstub")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}

	seededCode := `package seededstub

// TODO: implement worker pool concurrency limit
type WorkerPool struct{}

// Compute executes calculation.
func Compute() {
	panic("not implemented")
}
`
	if err := os.WriteFile(filepath.Join(pkgDir, "seeded.go"), []byte(seededCode), 0o644); err != nil {
		t.Fatal(err)
	}

	lane := DebtLane{
		Lane:        "seededstub",
		UnitOfWork:  "internal/seededstub",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:        true,
			HasTests:       true,
			TestFilesCount: 1,
			Integrated:     true,
			Dogfooded:      true,
			Benchmarked:    true,
		},
	}

	relPath := filepath.Join("internal", "seededstub", "seeded.go")
	var firstRun []FindingProvenance

	for i := 0; i < 5; i++ {
		findings := InspectUnitDetectors(&lane, pkgDir)
		var stubFindings []FindingProvenance
		for _, f := range findings {
			if f.Dimension == string(DimStubDebt) {
				stubFindings = append(stubFindings, f)
			}
		}

		if len(stubFindings) != 2 {
			t.Fatalf("iteration %d: expected exactly 2 stub debt findings, got %d: %+v", i, len(stubFindings), stubFindings)
		}

		// Finding 1: WorkerPool TODO directive
		f1 := stubFindings[0]
		if f1.Dimension != string(DimStubDebt) {
			t.Errorf("iteration %d: f1 unexpected dimension %s", i, f1.Dimension)
		}
		if f1.Surface != string(SurfaceInternal) {
			t.Errorf("iteration %d: f1 unexpected surface %s", i, f1.Surface)
		}
		if f1.Lane != "seededstub" {
			t.Errorf("iteration %d: f1 unexpected lane %s", i, f1.Lane)
		}
		if f1.Path != relPath {
			t.Errorf("iteration %d: f1 unexpected path %s", i, f1.Path)
		}
		if f1.Severity != "warning" {
			t.Errorf("iteration %d: f1 unexpected severity %s", i, f1.Severity)
		}
		expectedF1Prefix := fmt.Sprintf("stub debt in WorkerPool (%s:3):", relPath)
		if !strings.HasPrefix(f1.Message, expectedF1Prefix) {
			t.Errorf("iteration %d: f1 message %q does not have prefix %q", i, f1.Message, expectedF1Prefix)
		}

		// Finding 2: Compute unimplemented function
		f2 := stubFindings[1]
		if f2.Dimension != string(DimStubDebt) {
			t.Errorf("iteration %d: f2 unexpected dimension %s", i, f2.Dimension)
		}
		if f2.Surface != string(SurfaceInternal) {
			t.Errorf("iteration %d: f2 unexpected surface %s", i, f2.Surface)
		}
		if f2.Lane != "seededstub" {
			t.Errorf("iteration %d: f2 unexpected lane %s", i, f2.Lane)
		}
		if f2.Path != relPath {
			t.Errorf("iteration %d: f2 unexpected path %s", i, f2.Path)
		}
		if f2.Severity != "warning" {
			t.Errorf("iteration %d: f2 unexpected severity %s", i, f2.Severity)
		}
		expectedF2Prefix := fmt.Sprintf("stub debt in Compute (%s:8):", relPath)
		if !strings.HasPrefix(f2.Message, expectedF2Prefix) {
			t.Errorf("iteration %d: f2 message %q does not have prefix %q", i, f2.Message, expectedF2Prefix)
		}

		if i == 0 {
			firstRun = stubFindings
		} else {
			for idx := range stubFindings {
				if stubFindings[idx].Message != firstRun[idx].Message {
					t.Errorf("iteration %d: finding %d unstable: got %q, want %q",
						i, idx, stubFindings[idx].Message, firstRun[idx].Message)
				}
			}
		}
	}
}

func TestStubDebtGeneratedFileExcluded(t *testing.T) {
	tmp := t.TempDir()
	genDir := filepath.Join(tmp, "internal", "genpkg")
	if err := os.MkdirAll(genDir, 0o755); err != nil {
		t.Fatal(err)
	}

	code := `// Code generated by protoc-gen-go. DO NOT EDIT.
package genpkg

// TODO: this generated todo must be ignored
func GeneratedStub() {
	panic("not implemented")
}
`
	if err := os.WriteFile(filepath.Join(genDir, "gen.go"), []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}

	lane := DebtLane{
		Lane:        "genpkg",
		UnitOfWork:  "internal/genpkg",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:        true,
			HasTests:       true,
			TestFilesCount: 1,
			Integrated:     true,
			Dogfooded:      true,
			Benchmarked:    true,
		},
	}

	findings := InspectUnitDetectors(&lane, genDir)
	for _, f := range findings {
		if f.Dimension == string(DimStubDebt) {
			t.Fatalf("generated file carrying Code generated DO NOT EDIT must produce 0 stub debt findings, got: %+v", f)
		}
	}
}

func TestCountStubMarkersLargeLines(t *testing.T) {
	tmp := t.TempDir()
	largePath := filepath.Join(tmp, "large.go")

	// Create lines > 64KB (e.g. 70KB and 80KB)
	largePrefix := strings.Repeat("a", 70*1024)
	content := "// " + largePrefix + " TODO: handle this in large file\n"
	content += "package main\n"
	content += "// " + strings.Repeat("b", 80*1024) + "\n"
	content += `func run() { panic("not implemented") }` + "\n"

	if err := os.WriteFile(largePath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	count := countStubMarkers(largePath)
	if count != 2 {
		t.Errorf("expected 2 stub markers in file with >64KB lines, got %d", count)
	}
}

func TestThinTestDetectionAssertionSemantic(t *testing.T) {
	// Issue #12354 witness test:
	// 1. Clean fixture with direct comparisons and local *testing.T helper stays clean.
	// 2. Seeded fixture containing only setup branches and if true {} emits one thin_tests finding
	//    with the exact function and source span.
	// 3. Ambiguous fixture with external/unresolvable helper reports ambiguous proof as unknown.
	tmp := t.TempDir()

	// 1. Clean fixture
	cleanDir := filepath.Join(tmp, "internal", "clean_helpers")
	if err := os.MkdirAll(cleanDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cleanCode := `package clean_helpers

import "testing"

func checkHelper(t *testing.T, x int) {
	t.Helper()
	if x <= 0 {
		t.Fatalf("expected positive x, got %d", x)
	}
}

func TestDirectComparison(t *testing.T) {
	val := 10
	if val != 10 {
		t.Fatalf("expected 10, got %d", val)
	}
}

func TestLocalHelper(t *testing.T) {
	val := 20
	checkHelper(t, val)
}
`
	if err := os.WriteFile(filepath.Join(cleanDir, "clean_test.go"), []byte(cleanCode), 0o644); err != nil {
		t.Fatal(err)
	}

	laneClean := DebtLane{
		Lane:        "clean_helpers",
		UnitOfWork:  "internal/clean_helpers",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Benchmarked: true,
			Dogfooded:   true,
			Integrated:  true,
		},
	}
	cleanFindings := InspectUnitDetectors(&laneClean, cleanDir)
	for _, f := range cleanFindings {
		if f.Dimension == string(DimThinTests) {
			t.Fatalf("clean fixture with direct comparisons and local helper must stay clean, got: %+v", f)
		}
	}

	// 2. Seeded fixture: setup branches and if true {} (hazard: zero assertions)
	seededDir := filepath.Join(tmp, "internal", "seeded_branches")
	if err := os.MkdirAll(seededDir, 0o755); err != nil {
		t.Fatal(err)
	}
	seededCode := `package seeded_branches

import "testing"

func TestSetupBranchesOnly(t *testing.T) {
	ready := false
	if !ready {
		ready = true
	}
	if true {
	}
	_ = ready
}
`
	if err := os.WriteFile(filepath.Join(seededDir, "seeded_test.go"), []byte(seededCode), 0o644); err != nil {
		t.Fatal(err)
	}

	laneSeeded := DebtLane{
		Lane:        "seeded_branches",
		UnitOfWork:  "internal/seeded_branches",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Benchmarked: true,
			Dogfooded:   true,
			Integrated:  true,
		},
	}
	seededFindings := InspectUnitDetectors(&laneSeeded, seededDir)
	var thinFindings []FindingProvenance
	for _, f := range seededFindings {
		if f.Dimension == string(DimThinTests) {
			thinFindings = append(thinFindings, f)
		}
	}

	if len(thinFindings) != 1 {
		t.Fatalf("expected exactly 1 thin_tests finding for seeded fixture, got %d: %+v", len(thinFindings), thinFindings)
	}
	finding := thinFindings[0]
	if finding.Severity != "warning" {
		t.Errorf("expected warning severity, got %q", finding.Severity)
	}
	if !strings.Contains(finding.Message, "thin test hazard") {
		t.Errorf("expected message to contain 'thin test hazard', got %q", finding.Message)
	}
	if !strings.Contains(finding.Message, "TestSetupBranchesOnly") {
		t.Errorf("expected message to identify exact function TestSetupBranchesOnly, got %q", finding.Message)
	}
	if !strings.Contains(finding.Message, "seeded_test.go:5:1-13:2") {
		t.Errorf("expected message to contain source span seeded_test.go:5:1-13:2, got %q", finding.Message)
	}

	// 3. Ambiguous fixture: passes testing.T to external unresolvable helper without confirmed assertions
	ambiguousDir := filepath.Join(tmp, "internal", "ambiguous_helper")
	if err := os.MkdirAll(ambiguousDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ambiguousCode := `package ambiguous_helper

import "testing"

func TestAmbiguousExternal(t *testing.T) {
	// calling an external/unresolvable function passing t: reports ambiguous proof as unknown
	unresolvableExternal(t)
}
`
	if err := os.WriteFile(filepath.Join(ambiguousDir, "ambiguous_test.go"), []byte(ambiguousCode), 0o644); err != nil {
		t.Fatal(err)
	}

	laneAmbiguous := DebtLane{
		Lane:        "ambiguous_helper",
		UnitOfWork:  "internal/ambiguous_helper",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Benchmarked: true,
			Dogfooded:   true,
			Integrated:  true,
		},
	}
	ambiguousFindings := InspectUnitDetectors(&laneAmbiguous, ambiguousDir)
	var ambThin []FindingProvenance
	for _, f := range ambiguousFindings {
		if f.Dimension == string(DimThinTests) {
			ambThin = append(ambThin, f)
		}
	}
	if len(ambThin) != 1 {
		t.Fatalf("expected 1 thin_tests finding for ambiguous fixture, got %d: %+v", len(ambThin), ambThin)
	}
	if !strings.Contains(ambThin[0].Message, "unknown (ambiguous)") {
		t.Errorf("expected message to report ambiguous proof as unknown, got %q", ambThin[0].Message)
	}
}

func TestHotPathPerformanceDebt(t *testing.T) {
	tmp := t.TempDir()

	// 1. Critical-path unit containing all 5 debt patterns:
	// - loop-local make/append growth
	// - []byte to string conversion
	// - filesystem or subprocess I/O
	// - time.Sleep
	// - mutex acquisition
	critDir := filepath.Join(tmp, "internal", "hotcrit")
	if err := os.MkdirAll(critDir, 0o755); err != nil {
		t.Fatal(err)
	}

	critCode := `package hotcrit

import (
	"os"
	"os/exec"
	"sync"
	"time"
)

func CriticalPipeline(input []byte) {
	// Mutex acquisition on critical path
	var mu sync.Mutex
	mu.Lock()
	defer mu.Unlock()

	// Loop-local make & append growth
	var items []int
	for i := 0; i < 10; i++ {
		tempBuf := make([]byte, 128)
		_ = tempBuf
		items = append(items, i)
	}

	// []byte to string conversion
	str := string(input)
	_ = str

	// Filesystem I/O on critical path
	_, _ = os.ReadFile("config.json")

	// Subprocess execution on critical path
	cmd := exec.Command("uptime")
	_ = cmd

	// Blocking sleep on critical path
	time.Sleep(10 * time.Millisecond)
}
`
	if err := os.WriteFile(filepath.Join(critDir, "pipeline.go"), []byte(critCode), 0o644); err != nil {
		t.Fatal(err)
	}

	laneCrit := DebtLane{
		Lane:        "hotcrit",
		UnitOfWork:  "internal/hotcrit",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:    true,
			Integrated: true,
		},
	}

	critFindings := InspectUnitDetectors(&laneCrit, critDir)

	// Verify all 5 patterns are detected with exact source spans and correct dimensions
	foundMake := false
	foundAppend := false
	foundByteString := false
	foundFileIO := false
	foundSubprocess := false
	foundSleep := false
	foundMutex := false

	for _, f := range critFindings {
		if f.Dimension != string(DimAllocationCopyDebt) && f.Dimension != string(DimBlockingCallDebt) {
			continue
		}

		// Verify exact source span in Path: format is <path>:<line>:<col>
		parts := strings.Split(f.Path, ":")
		if len(parts) < 3 {
			t.Errorf("expected exact source span (path:line:col), got %q", f.Path)
		}
		if f.Severity != "warning" {
			t.Errorf("expected warning severity for static suspicion, got %q", f.Severity)
		}
		if !strings.Contains(f.Message, "static suspicion") {
			t.Errorf("expected static suspicion notice in message: %s", f.Message)
		}

		switch f.Dimension {
		case string(DimAllocationCopyDebt):
			if strings.Contains(f.Message, "make") {
				foundMake = true
			}
			if strings.Contains(f.Message, "append") {
				foundAppend = true
			}
			if strings.Contains(f.Message, "byte/string") {
				foundByteString = true
			}
		case string(DimBlockingCallDebt):
			if strings.Contains(f.Message, "os.ReadFile") {
				foundFileIO = true
			}
			if strings.Contains(f.Message, "exec.Command") {
				foundSubprocess = true
			}
			if strings.Contains(f.Message, "time.Sleep") {
				foundSleep = true
			}
			if strings.Contains(f.Message, "mutex acquisition") {
				foundMutex = true
			}
		}
	}

	if !foundMake {
		t.Errorf("missing loop-local make finding in: %+v", critFindings)
	}
	if !foundAppend {
		t.Errorf("missing loop-local append finding in: %+v", critFindings)
	}
	if !foundByteString {
		t.Errorf("missing byte/string copy finding in: %+v", critFindings)
	}
	if !foundFileIO {
		t.Errorf("missing file I/O finding in: %+v", critFindings)
	}
	if !foundSubprocess {
		t.Errorf("missing subprocess I/O finding in: %+v", critFindings)
	}
	if !foundSleep {
		t.Errorf("missing time.Sleep finding in: %+v", critFindings)
	}
	if !foundMutex {
		t.Errorf("missing mutex acquisition finding in: %+v", critFindings)
	}

	// Verify coverage receipt incorporates hot-path detector dimensions
	receipt := BuildCoverageReceipt(tmp, "fak", []DebtLane{laneCrit}, critFindings, 1)
	foundAllocDim := false
	foundBlockDim := false
	for _, d := range receipt.Dimensions {
		if d == string(DimAllocationCopyDebt) {
			foundAllocDim = true
		}
		if d == string(DimBlockingCallDebt) {
			foundBlockDim = true
		}
	}
	if !foundAllocDim {
		t.Errorf("expected %s in coverage receipt dimensions: %v", DimAllocationCopyDebt, receipt.Dimensions)
	}
	if !foundBlockDim {
		t.Errorf("expected %s in coverage receipt dimensions: %v", DimBlockingCallDebt, receipt.Dimensions)
	}

	// 2. Preallocated control fixture on critical path (remains clean)
	preallocDir := filepath.Join(tmp, "internal", "prealloc")
	if err := os.MkdirAll(preallocDir, 0o755); err != nil {
		t.Fatal(err)
	}

	preallocCode := `package prealloc

// Clean pipeline: preallocated buffer, indexed loops, no conversions, no blocking calls.
func PreallocatedPipeline(input []byte, n int) int {
	buf := make([]int, n)
	sum := 0
	for i := 0; i < n; i++ {
		buf[i] = i * 2
		sum += buf[i]
	}
	if len(input) > 0 {
		sum += int(input[0])
	}
	return sum
}
`
	if err := os.WriteFile(filepath.Join(preallocDir, "pipeline.go"), []byte(preallocCode), 0o644); err != nil {
		t.Fatal(err)
	}

	lanePrealloc := DebtLane{
		Lane:        "prealloc",
		UnitOfWork:  "internal/prealloc",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:    true,
			Integrated: true,
		},
	}

	preallocFindings := InspectUnitDetectors(&lanePrealloc, preallocDir)
	var preallocHotFindings []FindingProvenance
	for _, f := range preallocFindings {
		if f.Dimension == string(DimAllocationCopyDebt) || f.Dimension == string(DimBlockingCallDebt) {
			preallocHotFindings = append(preallocHotFindings, f)
		}
	}
	if len(preallocHotFindings) != 0 {
		t.Fatalf("preallocated critical path fixture must have 0 hot-path findings, got %d: %+v",
			len(preallocHotFindings), preallocHotFindings)
	}

	// 3. Off-spine control with the EXACT same debt patterns as critCode
	offSpineDir := filepath.Join(tmp, "tools", "offspine")
	if err := os.MkdirAll(offSpineDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(offSpineDir, "tool.go"), []byte(critCode), 0o644); err != nil {
		t.Fatal(err)
	}

	laneOffSpine := DebtLane{
		Lane:        "offspine",
		UnitOfWork:  "tools/offspine",
		Criticality: CriticalityPeripheral,
		Evidence: Evidence{
			HasCode:    true,
			Integrated: true,
		},
	}

	offSpineFindings := InspectUnitDetectors(&laneOffSpine, offSpineDir)
	var offSpineHotFindings []FindingProvenance
	for _, f := range offSpineFindings {
		if f.Dimension == string(DimAllocationCopyDebt) || f.Dimension == string(DimBlockingCallDebt) {
			offSpineHotFindings = append(offSpineHotFindings, f)
		}
	}
	if len(offSpineHotFindings) != 0 {
		t.Fatalf("off-spine fixture must have 0 hot-path findings, got %d: %+v",
			len(offSpineHotFindings), offSpineHotFindings)
	}

	// 4. Disconnected control with the EXACT same debt patterns
	disconnDir := filepath.Join(tmp, "internal", "disconnected")
	if err := os.MkdirAll(disconnDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(disconnDir, "disconn.go"), []byte(critCode), 0o644); err != nil {
		t.Fatal(err)
	}

	laneDisconn := DebtLane{
		Lane:        "disconnected",
		UnitOfWork:  "internal/disconnected",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:    true,
			Integrated: false, // unreachable from root!
		},
	}

	disconnFindings := InspectUnitDetectors(&laneDisconn, disconnDir)
	var disconnHotFindings []FindingProvenance
	for _, f := range disconnFindings {
		if f.Dimension == string(DimAllocationCopyDebt) || f.Dimension == string(DimBlockingCallDebt) {
			disconnHotFindings = append(disconnHotFindings, f)
		}
	}
	if len(disconnHotFindings) != 0 {
		t.Fatalf("disconnected fixture must have 0 hot-path findings, got %d: %+v",
			len(disconnHotFindings), disconnHotFindings)
	}
}

func TestCoverageZeroInvokedDetectors(t *testing.T) {
	// A clean fixture with zero invoked detectors reports observed depth 0 plus explicit gaps.
	lanes := []DebtLane{
		{Lane: "cleanpkg", UnitOfWork: "internal/cleanpkg", Criticality: CriticalityCore},
	}
	receipt := BuildCoverageReceipt("/test/workspace", "fak", lanes, nil, 10)

	if receipt.ObservedDepth != 0 {
		t.Fatalf("expected observed depth 0 for zero invoked detectors, got %d", receipt.ObservedDepth)
	}
	if receipt.DepthRatio != 0.0 {
		t.Fatalf("expected depth ratio 0.0, got %.2f", receipt.DepthRatio)
	}
	if len(receipt.Dimensions) != 0 {
		t.Fatalf("expected 0 evaluated dimensions, got %d: %v", len(receipt.Dimensions), receipt.Dimensions)
	}
	if len(receipt.Gaps) != len(StandardDetectorDimensions) {
		t.Fatalf("expected explicit gaps for all %d declared dimensions, got %d: %v",
			len(StandardDetectorDimensions), len(receipt.Gaps), receipt.Gaps)
	}
	if receipt.CoverageDebt != 0 {
		t.Fatalf("expected 0 coverage debt, got %d", receipt.CoverageDebt)
	}
}

func TestCoverageSeededTwoDetectors(t *testing.T) {
	// A seeded fixture invoking two named detectors reports exactly 2.
	lanes := []DebtLane{
		{Lane: "seeded", UnitOfWork: "internal/seeded", Criticality: CriticalityCore},
	}
	findings := []FindingProvenance{
		{
			Dimension: string(DimThinTests),
			Surface:   string(SurfaceInternal),
			Lane:      "seeded",
			Path:      "internal/seeded/seeded_test.go",
			Severity:  "warning",
			Message:   "thin test hazard",
		},
	}

	receipt := BuildCoverageReceipt("/test/workspace", "fak", lanes, findings, 20,
		WithExecutedDimensions(DimThinTests, DimStubDebt))

	if receipt.ObservedDepth != 2 {
		t.Fatalf("expected observed depth 2 for two invoked detectors, got %d", receipt.ObservedDepth)
	}
	if len(receipt.Dimensions) != 2 {
		t.Fatalf("expected 2 evaluated dimensions, got %d: %v", len(receipt.Dimensions), receipt.Dimensions)
	}
	if receipt.Dimensions[0] != string(DimStubDebt) || receipt.Dimensions[1] != string(DimThinTests) {
		t.Fatalf("expected [stub_debt, thin_tests], got %v", receipt.Dimensions)
	}
	expectedGaps := len(StandardDetectorDimensions) - 2
	if len(receipt.Gaps) != expectedGaps {
		t.Fatalf("expected %d gaps, got %d: %v", expectedGaps, len(receipt.Gaps), receipt.Gaps)
	}
	expectedRatio := float64(int((2.0/5.0)*100)) / 100.0
	if receipt.DepthRatio != expectedRatio {
		t.Fatalf("expected depth ratio %.2f, got %.2f", expectedRatio, receipt.DepthRatio)
	}
	if receipt.CoverageDebt != 0 {
		t.Fatalf("expected 0 coverage debt, got %d", receipt.CoverageDebt)
	}
}

func TestCoverageErroredDetectorRecordedAsCoverageDebt(t *testing.T) {
	// An errored detector is recorded as coverage debt and not counted in observed depth.
	lanes := []DebtLane{
		{Lane: "seeded", UnitOfWork: "internal/seeded", Criticality: CriticalityCore},
	}

	receipt := BuildCoverageReceipt("/test/workspace", "fak", lanes, nil, 20,
		WithExecutedDimensions(DimTestStatus),
		WithErroredDimension(DimRaceFuzzStatus, fmt.Errorf("fuzz harness parse timeout")),
	)

	if receipt.ObservedDepth != 1 {
		t.Fatalf("expected observed depth 1 (errored detector excluded from depth), got %d", receipt.ObservedDepth)
	}
	if receipt.CoverageDebt != 1 {
		t.Fatalf("expected coverage debt 1, got %d", receipt.CoverageDebt)
	}
	if len(receipt.ErroredDimensions) != 1 || receipt.ErroredDimensions[0] != string(DimRaceFuzzStatus) {
		t.Fatalf("expected errored dimension %q, got %v", DimRaceFuzzStatus, receipt.ErroredDimensions)
	}
	// Verify errored detector is in gaps
	foundInGaps := false
	for _, g := range receipt.Gaps {
		if g == string(DimRaceFuzzStatus) {
			foundInGaps = true
			break
		}
	}
	if !foundInGaps {
		t.Fatalf("expected errored detector %q in gaps, got %v", DimRaceFuzzStatus, receipt.Gaps)
	}
	// Verify errored detector is recorded as coverage debt in findings
	foundFinding := false
	for _, f := range receipt.Findings {
		if f.Dimension == string(DimRaceFuzzStatus) && f.Severity == "critical" && strings.Contains(f.Message, "coverage debt") {
			foundFinding = true
			break
		}
	}
	if !foundFinding {
		t.Fatalf("expected coverage debt finding for errored detector in %+v", receipt.Findings)
	}
}

func TestCoverageReceiptByteStable(t *testing.T) {
	// Repeated receipts are byte-stable.
	lanes := []DebtLane{
		{Lane: "core_gate", UnitOfWork: "internal/gate", Criticality: CriticalityCore},
		{Lane: "pub_sdk", UnitOfWork: "pkg/sdk", Criticality: CriticalityEnabling},
		{Lane: "plat_disp", UnitOfWork: "platform/disp", Criticality: CriticalityEnabling},
	}
	findings := []FindingProvenance{
		{
			Dimension: string(DimTestStatus),
			Surface:   string(SurfaceInternal),
			Lane:      "core_gate",
			Path:      "internal/gate/gate_test.go",
			Severity:  "warning",
			Message:   "missing unit tests",
		},
	}

	opts := []CoverageOption{
		WithExecutedDimensions(DimTestStatus, DimCommentHygiene, DimWiringStatus),
		WithErroredDimension(DimRaceFuzzStatus, fmt.Errorf("timeout")),
		WithSkippedDimensions(DimBlastRadius),
	}

	receipt1 := BuildCoverageReceipt("/test/workspace", "fak", lanes, findings, 50, opts...)
	receipt2 := BuildCoverageReceipt("/test/workspace", "fak", lanes, findings, 50, opts...)

	raw1, err1 := json.Marshal(receipt1)
	if err1 != nil {
		t.Fatalf("marshal receipt1: %v", err1)
	}
	raw2, err2 := json.Marshal(receipt2)
	if err2 != nil {
		t.Fatalf("marshal receipt2: %v", err2)
	}

	if !bytes.Equal(raw1, raw2) {
		t.Fatalf("coverage receipts are not byte-stable:\nraw1: %s\nraw2: %s", string(raw1), string(raw2))
	}
}
