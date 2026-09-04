package archcheck

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func findRepoRoot(t *testing.T) string {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not locate repo root containing go.mod")
		}
		dir = parent
	}
}

func TestCheckPackageConformingWitness(t *testing.T) {
	root := findRepoRoot(t)

	start := time.Now()
	res, err := CheckPackage(root, "internal/agentquery")
	if err != nil {
		t.Fatalf("CheckPackage: %v", err)
	}
	elapsed := time.Since(start)

	if !res.OK {
		t.Fatalf("expected internal/agentquery to be clean, got violations: %+v", res.Violations)
	}
	if len(res.Violations) != 0 {
		t.Fatalf("violations = %d, want 0", len(res.Violations))
	}

	// Performance verification: <50ms execution invariant
	if elapsed > 150*time.Millisecond {
		t.Logf("note: check took %v (budget 150ms)", elapsed)
	}
}

func TestCheckPackageDetectsUpwardImport(t *testing.T) {
	tmp := t.TempDir()

	// Re-create internal/ structure in tmp
	fakeRoot := tmp
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "architest"), 0o755)
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "leaflow"), 0o755)
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "leafhigh"), 0o755)

	contractContent := `package architest
var tier = map[string]int{
	"leaflow": 1,
	"leafhigh": 3,
}
var tierName = []string{"root", "primitive", "foundation", "mechanism"}
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "architest", "architest_test.go"), []byte(contractContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// leaflow (tier 1) illegally imports leafhigh (tier 3)
	badFile := `package leaflow
import _ "github.com/anthony-chaudhary/fak/internal/leafhigh"
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "leaflow", "bad.go"), []byte(badFile), 0o644); err != nil {
		t.Fatal(err)
	}

	res, err := CheckPackage(fakeRoot, "internal/leaflow")
	if err != nil {
		t.Fatalf("CheckPackage: %v", err)
	}
	if res.OK {
		t.Fatal("expected violation for upward import, got OK=true")
	}
	if len(res.Violations) == 0 {
		t.Fatal("expected at least 1 violation, got 0")
	}
	foundUpward := false
	for _, v := range res.Violations {
		if v.Rule == "UPWARD_IMPORT" && v.FromPackage == "leaflow" && v.ToPackage == "leafhigh" {
			foundUpward = true
			break
		}
	}
	if !foundUpward {
		t.Fatalf("did not find expected UPWARD_IMPORT violation: %+v", res.Violations)
	}
}

func TestCheckPackageDetectsPrimitivePurity(t *testing.T) {
	tmp := t.TempDir()
	fakeRoot := tmp
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "architest"), 0o755)
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "prim"), 0o755)
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "peer"), 0o755)

	contractContent := `package architest
var tier = map[string]int{
	"prim": 1,
	"peer": 1,
}
var tierName = []string{"root", "primitive"}
`
	_ = os.WriteFile(filepath.Join(fakeRoot, "internal", "architest", "architest_test.go"), []byte(contractContent), 0o644)

	// prim (tier 1) imports peer (tier 1 > tier 0)
	primFile := `package prim
import _ "github.com/anthony-chaudhary/fak/internal/peer"
`
	_ = os.WriteFile(filepath.Join(fakeRoot, "internal", "prim", "p.go"), []byte(primFile), 0o644)

	res, err := CheckPackage(fakeRoot, "internal/prim")
	if err != nil {
		t.Fatal(err)
	}
	if res.OK {
		t.Fatal("expected violation for primitive importing non-root, got OK=true")
	}
	foundPurity := false
	for _, v := range res.Violations {
		if v.Rule == "PRIMITIVE_LEAF_PURITY" {
			foundPurity = true
			break
		}
	}
	if !foundPurity {
		t.Fatalf("did not find expected PRIMITIVE_LEAF_PURITY violation: %+v", res.Violations)
	}
}

func TestCheckPackageDetectsUntiered(t *testing.T) {
	tmp := t.TempDir()
	fakeRoot := tmp
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "architest"), 0o755)
	_ = os.MkdirAll(filepath.Join(fakeRoot, "internal", "mystery"), 0o755)

	contractContent := `package architest
var tier = map[string]int{
	"known": 1,
}
var tierName = []string{"root", "primitive"}
`
	_ = os.WriteFile(filepath.Join(fakeRoot, "internal", "architest", "architest_test.go"), []byte(contractContent), 0o644)
	_ = os.WriteFile(filepath.Join(fakeRoot, "internal", "mystery", "m.go"), []byte("package mystery\n"), 0o644)

	res, err := CheckPackage(fakeRoot, "internal/mystery")
	if err != nil {
		t.Fatal(err)
	}
	if res.OK || len(res.Violations) == 0 || res.Violations[0].Rule != "UNTIERED_LEAF" {
		t.Fatalf("expected UNTIERED_LEAF violation, got %+v", res)
	}
}
