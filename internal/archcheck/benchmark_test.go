package archcheck

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkLoadTiers(b *testing.B) {
	root := resolveRepoRoot("")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tiers, names, err := LoadTiers(root)
		if err != nil {
			b.Fatal(err)
		}
		if len(tiers) == 0 || len(names) == 0 {
			b.Fatal("empty tiers or names")
		}
	}
}

func BenchmarkCheckPackage(b *testing.B) {
	root := resolveRepoRoot("")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := CheckPackage(root, "internal/agentquery")
		if err != nil {
			b.Fatal(err)
		}
		if !res.OK || len(res.Violations) != 0 {
			b.Fatalf("unexpected violations: %+v", res.Violations)
		}
	}
}

func BenchmarkCheckSinglePackage(b *testing.B) {
	root := resolveRepoRoot("")
	tiers, names, err := LoadTiers(root)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		violations, err := checkSinglePackage(root, "internal/agentquery", tiers, names)
		if err != nil {
			b.Fatal(err)
		}
		if len(violations) != 0 {
			b.Fatalf("unexpected violations: %+v", violations)
		}
	}
}

func BenchmarkCheckPackageDetectsUpwardImport(b *testing.B) {
	fakeRoot := b.TempDir()
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "architest"), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "leaflow"), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "leafhigh"), 0o755); err != nil {
		b.Fatal(err)
	}

	contractContent := `package architest
var tier = map[string]int{
	"leaflow": 1,
	"leafhigh": 3,
}
var tierName = []string{"root", "primitive", "foundation", "mechanism"}
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "architest", "architest_test.go"), []byte(contractContent), 0o644); err != nil {
		b.Fatal(err)
	}

	badFile := `package leaflow
import _ "github.com/anthony-chaudhary/fak/internal/leafhigh"
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "leaflow", "bad.go"), []byte(badFile), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := CheckPackage(fakeRoot, "internal/leaflow")
		if err != nil {
			b.Fatal(err)
		}
		if res.OK || len(res.Violations) == 0 {
			b.Fatal("expected upward import violation")
		}
	}
}

func BenchmarkCheckPackageDetectsPrimitivePurity(b *testing.B) {
	fakeRoot := b.TempDir()
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "architest"), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "prim"), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "peer"), 0o755); err != nil {
		b.Fatal(err)
	}

	contractContent := `package architest
var tier = map[string]int{
	"prim": 1,
	"peer": 1,
}
var tierName = []string{"root", "primitive"}
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "architest", "architest_test.go"), []byte(contractContent), 0o644); err != nil {
		b.Fatal(err)
	}

	primFile := `package prim
import _ "github.com/anthony-chaudhary/fak/internal/peer"
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "prim", "p.go"), []byte(primFile), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := CheckPackage(fakeRoot, "internal/prim")
		if err != nil {
			b.Fatal(err)
		}
		if res.OK || len(res.Violations) == 0 {
			b.Fatal("expected primitive purity violation")
		}
	}
}

func BenchmarkCheckPackageDetectsUntiered(b *testing.B) {
	fakeRoot := b.TempDir()
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "architest"), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fakeRoot, "internal", "mystery"), 0o755); err != nil {
		b.Fatal(err)
	}

	contractContent := `package architest
var tier = map[string]int{
	"known": 1,
}
var tierName = []string{"root", "primitive"}
`
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "architest", "architest_test.go"), []byte(contractContent), 0o644); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakeRoot, "internal", "mystery", "m.go"), []byte("package mystery\n"), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := CheckPackage(fakeRoot, "internal/mystery")
		if err != nil {
			b.Fatal(err)
		}
		if res.OK || len(res.Violations) == 0 {
			b.Fatal("expected untiered leaf violation")
		}
	}
}
