package debtlane

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMockHazardDetection(t *testing.T) {
	tmp := t.TempDir()
	unitDir := filepath.Join(tmp, "internal", "mockservice")
	if err := os.MkdirAll(unitDir, 0o755); err != nil {
		t.Fatal(err)
	}

	code := `package mockservice

// MockTransport simulates HTTP network operations.
type MockTransport struct {
	Endpoint string
}

// FakeClient emulates client communication.
type FakeClient struct {
	Connected bool
}

// RealService is a genuine production service.
type RealService struct {
	ID string
}

// DoWork executes genuine work.
func (r *RealService) DoWork() int {
	return 42
}

// ExecuteStub executes unimplemented functionality.
func ExecuteStub() {
	panic("not implemented")
}

// ProcessTodo executes todo functionality.
func ProcessTodo() {
	panic("todo: wire real provider")
}
`
	filePath := filepath.Join(unitDir, "service.go")
	if err := os.WriteFile(filePath, []byte(code), 0o644); err != nil {
		t.Fatal(err)
	}

	fset := token.NewFileSet()
	fileNode, err := parser.ParseFile(fset, filePath, code, parser.ParseComments)
	if err != nil {
		t.Fatalf("parser.ParseFile failed: %v", err)
	}

	lane := &DebtLane{
		Lane:        "mockservice",
		UnitOfWork:  "internal/mockservice",
		Criticality: CriticalityCore,
	}

	findings := inspectMockHazards(fset, fileNode, "internal/mockservice/service.go", lane, SurfaceInternal)

	if len(findings) != 4 {
		t.Fatalf("expected 4 mock hazard findings, got %d: %+v", len(findings), findings)
	}

	expectedSymbols := []string{"MockTransport", "FakeClient", "ExecuteStub", "ProcessTodo"}
	for i, expected := range expectedSymbols {
		f := findings[i]
		if f.Dimension != string(DimMockHazard) {
			t.Errorf("finding %d: expected dimension %q, got %q", i, DimMockHazard, f.Dimension)
		}
		if f.Surface != string(SurfaceInternal) {
			t.Errorf("finding %d: expected surface %q, got %q", i, SurfaceInternal, f.Surface)
		}
		if f.Lane != "mockservice" {
			t.Errorf("finding %d: expected lane %q, got %q", i, "mockservice", f.Lane)
		}
		if f.Severity != "warning" {
			t.Errorf("finding %d: expected warning severity, got %q", i, f.Severity)
		}
		if !strings.Contains(f.Message, expected) {
			t.Errorf("finding %d: message %q does not contain symbol %q", i, f.Message, expected)
		}
	}

	// Verify via inspectGoPackageEvidence that lane.Findings and lane.FindingProvs are populated.
	testLane := &DebtLane{
		Lane:        "mockservice",
		UnitOfWork:  "internal/mockservice",
		Criticality: CriticalityCore,
	}
	packageFindings := inspectGoPackageEvidence(testLane, unitDir, SurfaceInternal)
	if len(packageFindings) < 4 {
		t.Fatalf("expected at least 4 package findings, got %d: %+v", len(packageFindings), packageFindings)
	}

	if len(testLane.FindingProvs) != len(packageFindings) {
		t.Errorf("expected testLane.FindingProvs to match package findings count %d, got %d",
			len(packageFindings), len(testLane.FindingProvs))
	}
	if len(testLane.Findings) != len(packageFindings) {
		t.Errorf("expected testLane.Findings to match package findings count %d, got %d",
			len(packageFindings), len(testLane.Findings))
	}

	foundMockTransport := false
	foundFakeClient := false
	for _, fp := range testLane.FindingProvs {
		if fp.Dimension == string(DimMockHazard) {
			if strings.Contains(fp.Message, "MockTransport") {
				foundMockTransport = true
			}
			if strings.Contains(fp.Message, "FakeClient") {
				foundFakeClient = true
			}
		}
	}
	if !foundMockTransport || !foundFakeClient {
		t.Errorf("expected MockTransport and FakeClient in lane.FindingProvs: %+v", testLane.FindingProvs)
	}
}

func TestMockHazardSuppressionViaNolint(t *testing.T) {
	tmp := t.TempDir()

	// 1. File-level suppression via //nolint:mock_hazard
	fileSuppressedCode1 := `//nolint:mock_hazard
package suppressed1

type MockDriver struct{}

func Run() {
	panic("not implemented")
}
`
	fset := token.NewFileSet()
	node1, err := parser.ParseFile(fset, "suppressed1.go", fileSuppressedCode1, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findings1 := inspectMockHazards(fset, node1, "suppressed1.go", nil, SurfaceInternal)
	if len(findings1) != 0 {
		t.Fatalf("expected 0 findings for file-level //nolint:mock_hazard, got %d: %+v", len(findings1), findings1)
	}

	// 2. File-level suppression via //nolint:mock
	fileSuppressedCode2 := `//nolint:mock
package suppressed2

type FakeDatabase struct{}

func Query() {
	panic("todo")
}
`
	node2, err := parser.ParseFile(fset, "suppressed2.go", fileSuppressedCode2, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findings2 := inspectMockHazards(fset, node2, "suppressed2.go", nil, SurfaceInternal)
	if len(findings2) != 0 {
		t.Fatalf("expected 0 findings for file-level //nolint:mock, got %d: %+v", len(findings2), findings2)
	}

	// 3. Declaration-level and line-level suppression
	declSuppressedCode := `package declsuppressed

//nolint:mock_hazard intentional mock for isolated test fixture
type MockGateway struct{}

//nolint:mock
type FakeStore struct{}

type MockInline struct{} //nolint:mock

// Genuine unsuppressed mock struct
type MockActive struct{}

//nolint:mock_hazard
func SuppressedMethod() {
	panic("not implemented")
}

func InlineSuppressed() {
	//nolint:mock
	panic("unimplemented")
}

func UnsuppressedPanic() {
	panic("not implemented")
}
`
	node3, err := parser.ParseFile(fset, "decl_suppressed.go", declSuppressedCode, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findings3 := inspectMockHazards(fset, node3, "decl_suppressed.go", nil, SurfaceInternal)
	if len(findings3) != 2 {
		t.Fatalf("expected exactly 2 findings for unsuppressed items, got %d: %+v", len(findings3), findings3)
	}

	if !strings.Contains(findings3[0].Message, "MockActive") {
		t.Errorf("expected first finding to be MockActive, got: %s", findings3[0].Message)
	}
	if !strings.Contains(findings3[1].Message, "UnsuppressedPanic") {
		t.Errorf("expected second finding to be UnsuppressedPanic, got: %s", findings3[1].Message)
	}

	// 4. Generated file is excluded
	genFileCode := `// Code generated by tool; DO NOT EDIT.
package generated

type MockGenClient struct{}

func Broken() {
	panic("not implemented")
}
`
	nodeGen, err := parser.ParseFile(fset, "gen.go", genFileCode, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findingsGen := inspectMockHazards(fset, nodeGen, "gen.go", nil, SurfaceInternal)
	if len(findingsGen) != 0 {
		t.Fatalf("expected 0 findings for generated file, got %d: %+v", len(findingsGen), findingsGen)
	}

	// Test line scanner helper countMockHazardMarkers
	testFilePath := filepath.Join(tmp, "scanner_test.go")
	if err := os.WriteFile(testFilePath, []byte(declSuppressedCode), 0o644); err != nil {
		t.Fatal(err)
	}
	count := countMockHazardMarkers(testFilePath)
	if count != 2 {
		t.Errorf("expected countMockHazardMarkers to return 2 for unsuppressed markers, got %d", count)
	}

	suppressedFilePath := filepath.Join(tmp, "scanner_suppressed.go")
	if err := os.WriteFile(suppressedFilePath, []byte(fileSuppressedCode1), 0o644); err != nil {
		t.Fatal(err)
	}
	suppressedCount := countMockHazardMarkers(suppressedFilePath)
	if suppressedCount != 0 {
		t.Errorf("expected countMockHazardMarkers to return 0 for file-level suppressed file, got %d", suppressedCount)
	}
}

func TestAllDetectorDimensions(t *testing.T) {
	dims := AllDetectorDimensions()

	// Verify DimMockHazard is included
	foundMockHazard := false
	for _, d := range dims {
		if d == DimMockHazard {
			foundMockHazard = true
			break
		}
	}
	if !foundMockHazard {
		t.Fatalf("expected DimMockHazard (%q) to be present in AllDetectorDimensions(), got: %v",
			DimMockHazard, dims)
	}

	// Verify all StandardDetectorDimensions are present
	for _, std := range StandardDetectorDimensions {
		found := false
		for _, d := range dims {
			if d == std {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected standard dimension %q to be present in AllDetectorDimensions()", std)
		}
	}

	// Verify DimCoverageDebt is present
	foundCoverageDebt := false
	for _, d := range dims {
		if d == DimCoverageDebt {
			foundCoverageDebt = true
			break
		}
	}
	if !foundCoverageDebt {
		t.Errorf("expected DimCoverageDebt (%q) to be present in AllDetectorDimensions()", DimCoverageDebt)
	}

	// Verify count matches expected total (15 standard + DimCoverageDebt + DimMockHazard = 17)
	expectedCount := len(StandardDetectorDimensions) + 2
	if len(dims) != expectedCount {
		t.Errorf("expected %d dimensions in AllDetectorDimensions(), got %d: %v",
			expectedCount, len(dims), dims)
	}
}

func TestMockHazardAdversarialASTShapes(t *testing.T) {
	fset := token.NewFileSet()

	// 1. Grouped type declarations and generic structs
	code1 := `package adversarial
type (
	MockGroupedA struct{}
	FakeGroupedB struct{}
	ValidReal struct{}
)

type MockPool[T any] struct {
	items []T
}

func ComplexPanic() {
	panic("unimplemented: custom prefix " + "and suffix")
}
`
	node1, err := parser.ParseFile(fset, "adv1.go", code1, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findings1 := inspectMockHazards(fset, node1, "adv1.go", nil, SurfaceInternal)
	if len(findings1) != 4 {
		t.Fatalf("expected 4 findings (MockGroupedA, FakeGroupedB, MockPool, ComplexPanic), got %d: %+v", len(findings1), findings1)
	}

	// 2. Multilint and block comment suppression
	code2 := `package adversarial
//nolint:unused,mock_hazard
type MockMultiLint struct{}

/* nolint:mock */
type FakeBlockComment struct{}

type MockActive struct{}
`
	node2, err := parser.ParseFile(fset, "adv2.go", code2, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findings2 := inspectMockHazards(fset, node2, "adv2.go", nil, SurfaceInternal)
	if len(findings2) != 1 {
		t.Fatalf("expected 1 finding (MockActive), got %d: %+v", len(findings2), findings2)
	}
	if !strings.Contains(findings2[0].Message, "MockActive") {
		t.Errorf("expected finding to be MockActive, got: %s", findings2[0].Message)
	}

	// 3. Valid production types should not produce mock_hazard findings
	code3 := `package adversarial
type Store struct{}
type MockeryParser struct{}
type FakerGenerator struct{}
type MockingbirdAudio struct{}
type MockupEngine struct{}
`
	node3, err := parser.ParseFile(fset, "adv3.go", code3, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	findings3 := inspectMockHazards(fset, node3, "adv3.go", nil, SurfaceInternal)
	if len(findings3) != 0 {
		t.Fatalf("expected 0 findings for valid production types, got %d: %+v", len(findings3), findings3)
	}

	// 4. Generated file variants
	codeGenProtoc := `// Code generated by protoc-gen-go. DO NOT EDIT.
package generated
type MockProtoc struct{}
`
	nodeProtoc, err := parser.ParseFile(fset, "protoc.go", codeGenProtoc, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	if findings := inspectMockHazards(fset, nodeProtoc, "protoc.go", nil, SurfaceInternal); len(findings) != 0 {
		t.Fatalf("expected 0 findings for protoc generated code, got %d", len(findings))
	}

	codeGenUpper := `// GENERATED BY MAKEFILE. DO NOT EDIT.
package generated
type MockUpper struct{}
`
	nodeUpper, err := parser.ParseFile(fset, "upper.go", codeGenUpper, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	if findings := inspectMockHazards(fset, nodeUpper, "upper.go", nil, SurfaceInternal); len(findings) != 0 {
		t.Fatalf("expected 0 findings for uppercase GENERATED code, got %d", len(findings))
	}
}
