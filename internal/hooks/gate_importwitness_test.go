package hooks

import (
	"strings"
	"testing"
)

func TestGateImportWitness_UncommittedPackageFlagsFinding(t *testing.T) {
	d := &StagedDiff{
		Root:        t.TempDir(),
		StagedPaths: []string{"internal/alpha/alpha.go"},
		fileCache: map[string]fileEntry{
			"internal/alpha/alpha.go": {
				data:   []byte("package alpha\n\nimport \"github.com/anthony-chaudhary/fak/internal/missing\"\n"),
				exists: true,
			},
		},
		IndexPaths: []string{"internal/alpha/alpha.go"},
	}

	findings, err := gateImportWitness(d)
	if err != nil {
		t.Fatalf("gateImportWitness: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}
	if findings[0].Gate != "IMPORT_WITNESS" {
		t.Errorf("gate = %s, want IMPORT_WITNESS", findings[0].Gate)
	}
	if !strings.Contains(findings[0].Detail, "internal/missing") {
		t.Errorf("detail = %s, want mention of internal/missing", findings[0].Detail)
	}
}

func TestGateImportWitness_CleanWhenPackageCommitted(t *testing.T) {
	d := &StagedDiff{
		Root:        t.TempDir(),
		StagedPaths: []string{"internal/alpha/alpha.go"},
		fileCache: map[string]fileEntry{
			"internal/alpha/alpha.go": {
				data:   []byte("package alpha\n\nimport \"github.com/anthony-chaudhary/fak/internal/beta\"\n"),
				exists: true,
			},
		},
		IndexPaths: []string{"internal/alpha/alpha.go", "internal/beta/beta.go"},
	}

	findings, err := gateImportWitness(d)
	if err != nil {
		t.Fatalf("gateImportWitness: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings, got %d: %+v", len(findings), findings)
	}
}

func TestGateImportWitness_CleanWhenPackageStagedInSameCommit(t *testing.T) {
	d := &StagedDiff{
		Root:        t.TempDir(),
		StagedPaths: []string{"internal/alpha/alpha.go", "internal/beta/beta.go"},
		fileCache: map[string]fileEntry{
			"internal/alpha/alpha.go": {
				data:   []byte("package alpha\n\nimport \"github.com/anthony-chaudhary/fak/internal/beta\"\n"),
				exists: true,
			},
			"internal/beta/beta.go": {
				data:   []byte("package beta\n\nfunc Hello() {}\n"),
				exists: true,
			},
		},
	}

	findings, err := gateImportWitness(d)
	if err != nil {
		t.Fatalf("gateImportWitness: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings, got %d: %+v", len(findings), findings)
	}
}

func TestGateImportWitness_IgnoresTestFiles(t *testing.T) {
	d := &StagedDiff{
		Root:        t.TempDir(),
		StagedPaths: []string{"internal/alpha/alpha_test.go"},
		fileCache: map[string]fileEntry{
			"internal/alpha/alpha_test.go": {
				data:   []byte("package alpha\n\nimport \"github.com/anthony-chaudhary/fak/internal/missing\"\n"),
				exists: true,
			},
		},
	}

	findings, err := gateImportWitness(d)
	if err != nil {
		t.Fatalf("gateImportWitness: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("test files should be ignored, got %d findings: %+v", len(findings), findings)
	}
}

func TestGateImportWitness_RegisteredInPreCommitGates(t *testing.T) {
	found := false
	for _, g := range PreCommitGates() {
		if g.Name == "IMPORT_WITNESS" {
			found = true
			if g.Check == nil {
				t.Error("IMPORT_WITNESS Check function is nil")
			}
			break
		}
	}
	if !found {
		t.Fatal("IMPORT_WITNESS is not registered in PreCommitGates")
	}
}
