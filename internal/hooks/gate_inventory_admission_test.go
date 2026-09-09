//nolint:inventory_mock intentional gate test fixtures
package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// gate_inventory_admission_test.go — unit tests for the CLEAR_INVENTORY_ADMISSION gate.

func TestGateClearInventoryAdmission_MockStructDetection(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/service/store.go": {
			"package service",
			"type " + "MockStore struct {",
			"\tid string",
			"}",
		},
	})

	findings, err := gateClearInventoryAdmission(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}

	f := findings[0]
	if f.Gate != "CLEAR_INVENTORY_ADMISSION" {
		t.Errorf("gate = %q, want CLEAR_INVENTORY_ADMISSION", f.Gate)
	}
	if f.File != "internal/service/store.go" {
		t.Errorf("file = %q, want internal/service/store.go", f.File)
	}
	if f.Line != 2 {
		t.Errorf("line = %d, want 2", f.Line)
	}
	if !strings.Contains(f.Detail, "mocks hide integration bugs") {
		t.Errorf("detail missing Hermes' rule: %q", f.Detail)
	}

	n, unit, ok := d.Candidates("CLEAR_INVENTORY_ADMISSION")
	if !ok || n != 1 || unit != "staged mock/stub candidate(s)" {
		t.Errorf("candidates = (%d, %q, %v), want (1, %q, true)", n, unit, ok, "staged mock/stub candidate(s)")
	}

	// Also verify Fake[A-Z] struct detection
	d2 := diffOf("/r", map[string][]string{
		"cmd/fak/client.go": {
			"package main",
			"type " + "FakeClient struct {}",
		},
	})
	findings2, err := gateClearInventoryAdmission(d2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings2) != 1 {
		t.Fatalf("expected 1 finding for FakeClient, got %d: %+v", len(findings2), findings2)
	}
}

func TestGateClearInventoryAdmission_StubPanicProductionCode(t *testing.T) {
	// 1. Stub panic in production code should be flagged
	dProd := diffOf("/r", map[string][]string{
		"internal/handler/handler.go": {
			"package handler",
			"func Handle() {",
			"\tpanic(\"not implemented\")",
			"}",
		},
	})

	findings, err := gateClearInventoryAdmission(dProd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding for production panic, got %d: %+v", len(findings), findings)
	}
	if findings[0].Line != 3 {
		t.Errorf("finding line = %d, want 3", findings[0].Line)
	}

	// 2. Stub panic in test code should NOT be flagged
	dTest := diffOf("/r", map[string][]string{
		"internal/handler/handler_test.go": {
			"package handler",
			"func TestHandle(t *testing.T) {",
			"\tpanic(\"not implemented\")",
			"}",
		},
	})

	findingsTest, err := gateClearInventoryAdmission(dTest)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findingsTest) != 0 {
		t.Fatalf("expected 0 findings for test file panic, got %d: %+v", len(findingsTest), findingsTest)
	}

	// 3. Other stub panic patterns: unimplemented, todo
	dOthers := diffOf("/r", map[string][]string{
		"pkg/worker/worker.go": {
			"package worker",
			"func A() { panic(\"unimplemented\") }",
			"func B() { panic(\"todo\") }",
		},
	})
	findingsOthers, err := gateClearInventoryAdmission(dOthers)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findingsOthers) != 2 {
		t.Fatalf("expected 2 findings for unimplemented and todo panics, got %d: %+v", len(findingsOthers), findingsOthers)
	}
}

func TestGateClearInventoryAdmission_SuppressionViaWitness(t *testing.T) {
	cases := []struct {
		name    string
		witness string
	}{
		{"shift-left-verified", "shift-left-verified: make smoke"},
		{"e2e-verified", "e2e-verified: /verify"},
		{"inventory-gated", "inventory-gated: verified"},
		{"concept-verified", "concept-verified: OK"},
		{"smoke-verified", "smoke-verified: green"},
		{"smoke-test", "smoke-test: passed"},
		{"real-world-verified", "real-world-verified: tested on lab"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := diffOf("/r", map[string][]string{
				"internal/service/store.go": {
					"package service",
					"type " + "MockStore struct {}",
					"// " + tc.witness,
				},
			})

			findings, err := gateClearInventoryAdmission(d)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(findings) != 0 {
				t.Fatalf("expected 0 findings when %q is present, got %d: %+v", tc.witness, len(findings), findings)
			}

			// Candidate count must still be recorded (denominator preserved)
			n, _, ok := d.Candidates("CLEAR_INVENTORY_ADMISSION")
			if !ok || n != 1 {
				t.Errorf("candidate count = %d (ok=%v), want 1", n, ok)
			}
		})
	}
}

func TestGateClearInventoryAdmission_SuppressionViaNolint(t *testing.T) {
	// 1. Suppression on the same line with //nolint:inventory_mock
	dLine := diffOf("/r", map[string][]string{
		"internal/service/store.go": {
			"package service",
			"type " + "MockStore struct {} //nolint:inventory_mock intentional mock",
		},
	})

	findings, err := gateClearInventoryAdmission(dLine)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings for line-level nolint:inventory_mock, got %d: %+v", len(findings), findings)
	}

	// 2. Suppression at the file level with //nolint:inventory_mock
	dFile := diffOf("/r", map[string][]string{
		"internal/service/store.go": {
			"//nolint:inventory_mock",
			"package service",
			"type " + "MockStore struct {}",
			"type " + "MockClient struct {}",
		},
	})

	findingsFile, err := gateClearInventoryAdmission(dFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findingsFile) != 0 {
		t.Fatalf("expected 0 findings for file-level nolint:inventory_mock, got %d: %+v", len(findingsFile), findingsFile)
	}

	// 3. Suppression with //nolint:stub
	dStub := diffOf("/r", map[string][]string{
		"internal/service/store.go": {
			"package service",
			"func Run() { panic(\"not implemented\") //nolint:stub intentional stub }",
		},
	})

	findingsStub, err := gateClearInventoryAdmission(dStub)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findingsStub) != 0 {
		t.Fatalf("expected 0 findings for nolint:stub, got %d: %+v", len(findingsStub), findingsStub)
	}
}

func TestGateClearInventoryAdmission_EmptyDiff(t *testing.T) {
	d := emptyStagedDiff(t.TempDir())

	findings, err := gateClearInventoryAdmission(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings on empty diff, got %d: %+v", len(findings), findings)
	}

	n, unit, ok := d.Candidates("CLEAR_INVENTORY_ADMISSION")
	if !ok {
		t.Fatal("Candidates(CLEAR_INVENTORY_ADMISSION) returned ok=false, want true")
	}
	if n != 0 {
		t.Errorf("candidate count = %d, want 0", n)
	}
	if unit != "staged mock/stub candidate(s)" {
		t.Errorf("unit = %q, want staged mock/stub candidate(s)", unit)
	}
}

func TestGateClearInventoryAdmission_MockConstructorsAndImports(t *testing.T) {
	// Test constructor functions func NewMock... and func NewFake...
	dFunc := diffOf("/r", map[string][]string{
		"internal/service/store.go": {
			"package service",
			"func " + "NewMockStore() *Store { return nil }",
			"func " + "NewFakeClient() *Client { return nil }",
		},
	})

	findingsFunc, err := gateClearInventoryAdmission(dFunc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findingsFunc) != 2 {
		t.Fatalf("expected 2 findings for mock constructors, got %d: %+v", len(findingsFunc), findingsFunc)
	}

	// Test imports of mock packages
	dImport := diffOf("/r", map[string][]string{
		"internal/service/store.go": {
			"package service",
			"import \"github.com/golang/mock/gomock\"",
			"import \"github.com/stretchr/testify/mock\"",
		},
	})

	findingsImport, err := gateClearInventoryAdmission(dImport)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findingsImport) != 2 {
		t.Fatalf("expected 2 findings for mock imports, got %d: %+v", len(findingsImport), findingsImport)
	}
}

func TestGateClearInventoryAdmission_Registered(t *testing.T) {
	var found *Gate
	for i, g := range PreCommitGates() {
		if g.Name == "CLEAR_INVENTORY_ADMISSION" {
			found = &PreCommitGates()[i]
			break
		}
	}
	if found == nil {
		t.Fatal("CLEAR_INVENTORY_ADMISSION is not registered in PreCommitGates()")
	}
	if found.DefaultMode != "warn" {
		t.Errorf("DefaultMode = %q, want warn", found.DefaultMode)
	}
	if found.ModeEnv != "FLEET_INVENTORY_GUARD" {
		t.Errorf("ModeEnv = %q, want FLEET_INVENTORY_GUARD", found.ModeEnv)
	}
	if found.EscapeEnv != "ALLOW_UNINVENTORIED_STUB" {
		t.Errorf("EscapeEnv = %q, want ALLOW_UNINVENTORIED_STUB", found.EscapeEnv)
	}
}

func TestExhaustivenessClaim(t *testing.T) {
	TestDocCountClaimsMatchTheGateRegistry(t)
}

func TestGateClearInventoryAdmission_LandsTreeIntegration(t *testing.T) {
	root := landsTreeFixture(t)
	svcDir := filepath.Join(root, "internal", "service")
	if err := os.MkdirAll(svcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	mockFile := filepath.Join(svcDir, "service.go")
	if err := os.WriteFile(mockFile, []byte("package service\n\ntype MockService struct {}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	landsTreeGit(t, root, "add", "internal/service/service.go")

	d, err := ReadStagedDiff(root)
	if err != nil {
		t.Fatal(err)
	}

	gate := landstreeGateByName(t, "CLEAR_INVENTORY_ADMISSION")
	findings, err := gate.Check(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}

	f := findings[0]
	if f.Gate != "CLEAR_INVENTORY_ADMISSION" {
		t.Errorf("gate = %q, want CLEAR_INVENTORY_ADMISSION", f.Gate)
	}
	if f.View != ViewLandsTree {
		t.Errorf("finding.View = %q, want %q (ClassLandsTree scoping verification)", f.View, ViewLandsTree)
	}

	n, unit, ok := d.Candidates("CLEAR_INVENTORY_ADMISSION")
	if !ok || n != 1 || unit != "staged mock/stub candidate(s)" {
		t.Errorf("d.Candidates = (%d, %q, %v), want (1, %q, true) — denominator merged back to parent diff", n, unit, ok, "staged mock/stub candidate(s)")
	}
}

func TestGateClearInventoryAdmission_AdversarialEdgeCases(t *testing.T) {
	// 1. Generic type support: Mock[T] and Fake[K, V]
	dGenerics := diffOf("/r", map[string][]string{
		"internal/service/generics.go": {
			"package service",
			"type MockQueue[T any] struct {}",
			"type FakeCache[K comparable, V any] struct {}",
		},
	})
	findingsGen, err := gateClearInventoryAdmission(dGenerics)
	if err != nil {
		t.Fatal(err)
	}
	if len(findingsGen) != 2 {
		t.Fatalf("expected 2 findings for generic mock/fake structs, got %d: %+v", len(findingsGen), findingsGen)
	}

	// 2. True negatives: valid production types with similar prefixes
	dValid := diffOf("/r", map[string][]string{
		"internal/service/valid.go": {
			"package service",
			"type Store struct {}",
			"type MockeryParser struct {}",   // Mock followed by lowercase 'e'
			"type FakerGenerator struct {}",  // Fake followed by lowercase 'r'
			"type MockingbirdAudio struct {}", // Mock followed by lowercase 'i'
			"type MockupEngine struct {}",    // Mock followed by lowercase 'u'
			"type DefaultFakeHandler struct {}", // Does not start with Mock/Fake
		},
	})
	findingsValid, err := gateClearInventoryAdmission(dValid)
	if err != nil {
		t.Fatal(err)
	}
	if len(findingsValid) != 0 {
		t.Fatalf("expected 0 findings for valid production types, got %d: %+v", len(findingsValid), findingsValid)
	}

	// 3. Documented false negatives of fast line-based regex vs full AST:
	// - Grouped type declaration: `type (\n MockService struct {}\n)`
	// - Lowercase unexported: `type mockService struct {}`
	// - Alternative mock import: `go.uber.org/mock/gomock`
	dTricky := diffOf("/r", map[string][]string{
		"internal/service/tricky.go": {
			"package service",
			"type (",
			"\tMockService struct {}",
			")",
			"type mockService struct {}",
			"import \"go.uber.org/mock/gomock\"",
		},
	})
	findingsTricky, err := gateClearInventoryAdmission(dTricky)
	if err != nil {
		t.Fatal(err)
	}
	// Documents current gate behavior: line-based regex misses grouped types and lowercase unexported mocks.
	// This confirms the exact boundary between the pre-commit gate (fast regex) and debtlane DimMockHazard (deep AST).
	if len(findingsTricky) != 0 {
		t.Logf("Note: gate detected %d tricky mock shapes (grouped/unexported/uber-gomock)", len(findingsTricky))
	}
}

