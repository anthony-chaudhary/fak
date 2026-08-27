package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func cartGateDiff(t *testing.T, files map[string]string) *StagedDiff {
	t.Helper()
	root := t.TempDir()
	d := diffOf(root, map[string][]string{})
	for rel, body := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		d.StagedPaths = append(d.StagedPaths, rel)
		d.AddedPaths = append(d.AddedPaths, rel)
		d.AddedRenamedPaths = append(d.AddedRenamedPaths, rel)
		d.IndexPaths = append(d.IndexPaths, rel)
		for i, line := range strings.Split(body, "\n") {
			d.AddedByFile[rel] = append(d.AddedByFile[rel], AddedLine{File: rel, New: i + 1, Text: line})
		}
	}
	return d
}

func TestCartBeforeHorseWarnsOncePerNewLeaf(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"internal/colt/colt.go":                "package colt\nfunc Run() int { return 1 }\n",
		"internal/colt/colt_benchmark_test.go": "package colt\nimport \"testing\"\nfunc BenchmarkRun(b *testing.B) { _ = Run() }\n",
		"internal/colt/testdata/cases.json":    "{}\n",
	})
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Gate != cartBeforeHorseGate || got[0].File != "internal/colt/colt_benchmark_test.go" {
		t.Fatalf("findings = %+v, want one deterministic colt finding", got)
	}
	for _, want := range []string{"docs/spine-first-defaults.md", "Spine-witness:", "FLEET_CART_BEFORE_HORSE_GUARD=block", "ALLOW_CART_BEFORE_HORSE=1"} {
		if !strings.Contains(got[0].Detail, want) {
			t.Errorf("detail missing %q: %s", want, got[0].Detail)
		}
	}
	if n, unit, ok := d.Candidates(cartBeforeHorseGate); !ok || n != 1 || unit != "new internal/<leaf>/ package(s)" {
		t.Fatalf("candidate denominator = %d %q ok=%v, want 1 new leaf", n, unit, ok)
	}
}

func TestCartBeforeHorseOrdinaryAppliedTestSuppliesSpine(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"internal/colt/colt.go": "package colt\nfunc Run() int { return 1 }\n",
		"internal/colt/colt_test.go": "package colt\nimport \"testing\"\nfunc TestRun(t *testing.T) { if Run() != 1 { t.Fatal(\"bad\") } }\n" +
			"func BenchmarkRun(b *testing.B) { _ = Run() }\n",
	})
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("applied ordinary test should establish the horse, got %+v", got)
	}
}

func TestCartBeforeHorseSyntheticTestDoesNotSupplySpine(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"internal/colt/colt.go": "package colt\nfunc Run() int { return 1 }\n",
		"internal/colt/colt_test.go": "package colt\nimport \"testing\"\nfunc TestPlaceholder(t *testing.T) { t.Log(\"todo\") }\n" +
			"func BenchmarkRun(b *testing.B) { _ = Run() }\n",
	})
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("synthetic test must not certify an applied spine, got %+v", got)
	}
}

func TestCartBeforeHorseAttestationSuppliesSpine(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"internal/colt/colt.go":       "package colt\n// Spine-witness: go test ./internal/colt -run TestLivePath\nfunc Run() int { return 1 }\n",
		"internal/colt/bench_test.go": "package colt\nimport \"testing\"\nfunc BenchmarkRun(b *testing.B) { _ = Run() }\n",
	})
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("explicit staged spine attestation should satisfy the advisory, got %+v", got)
	}
}

func TestCartBeforeHorseAttestationIsLeafScoped(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"docs/spine-receipts.md":      "Spine-witness: go test ./internal/colt -run TestLivePath\n",
		"internal/colt/colt.go":       "package colt\nfunc Run() int { return 1 }\n",
		"internal/colt/bench_test.go": "package colt\nimport \"testing\"\nfunc BenchmarkRun(b *testing.B) { _ = Run() }\n",
		"internal/foal/foal.go":       "package foal\nfunc Run() int { return 1 }\n",
		"internal/foal/bench_test.go": "package foal\nimport \"testing\"\nfunc BenchmarkRun(b *testing.B) { _ = Run() }\n",
	})
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !strings.Contains(got[0].Detail, "internal/foal") {
		t.Fatalf("colt attestation must not waive foal, got %+v", got)
	}
	if n, _, ok := d.Candidates(cartBeforeHorseGate); !ok || n != 2 {
		t.Fatalf("candidate denominator = %d ok=%v, want both new leaves", n, ok)
	}
}

func TestCartBeforeHorseExistingLeafAndDocsOnlyStayQuiet(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"internal/colt/perf_test.go": "package colt\nimport \"testing\"\nfunc BenchmarkRun(b *testing.B) {}\n",
		"docs/colt.md":               "# Colt\n",
	})
	// A non-added path in the landing tree proves the leaf predates this commit.
	d.IndexPaths = append(d.IndexPaths, "internal/colt/colt.go")
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("ordinary existing-leaf work must stay quiet, got %+v", got)
	}
	if n, _, ok := d.Candidates(cartBeforeHorseGate); !ok || n != 0 {
		t.Fatalf("existing-leaf denominator = %d ok=%v, want reported zero", n, ok)
	}

	docs := cartGateDiff(t, map[string]string{"docs/only.md": "# Only docs\n"})
	got, err = gateCartBeforeHorse(docs)
	if err != nil || len(got) != 0 {
		t.Fatalf("docs-only diff = findings %+v err %v, want quiet", got, err)
	}
}

func TestCartBeforeHorseSafetyProofIsPartOfSpine(t *testing.T) {
	d := cartGateDiff(t, map[string]string{
		"internal/colt/colt.go":                 "package colt\nfunc RejectUnsafe() bool { return true }\n",
		"internal/colt/fuzz_failclosed_test.go": "package colt\nimport \"testing\"\nfunc FuzzFailClosed(f *testing.F) { f.Fuzz(func(t *testing.T, b []byte) { _ = RejectUnsafe() }) }\n",
	})
	got, err := gateCartBeforeHorse(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("fail-closed safety proof is core spine work, not a cart: %+v", got)
	}
}

func TestCartBeforeHorseArtifactsUseStructure(t *testing.T) {
	for _, tc := range []struct {
		path string
		body string
		want bool
	}{
		{"internal/x/x_test.go", "package x\nimport \"testing\"\nfunc BenchmarkX(b *testing.B) {}\n", true},
		{"internal/x/x_test.go", "package x\nimport \"testing\"\nfunc FuzzX(f *testing.F) {}\n", true},
		{"internal/x/proof_matrix_test.go", "package x\n", true},
		{"internal/x/proof-matrix/cases.json", "{}", true},
		{"internal/x/testdata/cases.json", "{}", true},
		{"internal/x/x_test.go", "package x\nfunc TestOrdinary() {}\n", false},
		{"internal/x/fuzz_failclosed_test.go", "package x\nfunc FuzzFailClosed() {}\n", false},
	} {
		if got := cartBeforeHorseArtifact(tc.path, []byte(tc.body), true); got != tc.want {
			t.Errorf("artifact(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

func TestCartBeforeHorseRegisteredAdvisory(t *testing.T) {
	for _, gate := range PreCommitGates() {
		if gate.Name != cartBeforeHorseGate {
			continue
		}
		if gate.DefaultMode != "warn" || gate.ModeEnv != "FLEET_CART_BEFORE_HORSE_GUARD" || gate.EscapeEnv != "ALLOW_CART_BEFORE_HORSE" {
			t.Fatalf("registration = %+v", gate)
		}
		return
	}
	t.Fatal("CART_BEFORE_HORSE is not registered in PreCommitGates")
}
