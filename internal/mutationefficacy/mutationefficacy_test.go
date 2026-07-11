package mutationefficacy

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestGenerateMutants_OperatorSwaps proves the pure mutant generator finds each operator
// family and byte-splices exactly one operator, preserving the rest of the file verbatim.
func TestGenerateMutants_OperatorSwaps(t *testing.T) {
	src := "package p\n\nfunc f(a, b int) bool { return a < b }\n"
	got := GenerateMutants("f.go", src, 0)
	if len(got) != 1 {
		t.Fatalf("want 1 mutant for one `<`, got %d: %+v", len(got), got)
	}
	m := got[0]
	if m.Op != "< -> <=" {
		t.Errorf("op label = %q, want %q", m.Op, "< -> <=")
	}
	want := "package p\n\nfunc f(a, b int) bool { return a <= b }\n"
	if m.Mutated != want {
		t.Errorf("mutated source =\n%q\nwant\n%q", m.Mutated, want)
	}
}

// TestGenerateMutants_CapAndCount checks the per-file cap bounds the mutant count and that a
// multi-operator file yields one mutant per mutable operator in source order.
func TestGenerateMutants_CapAndCount(t *testing.T) {
	src := "package p\n\nfunc g(a, b, c int) int {\n\tif a == b {\n\t\treturn a + c\n\t}\n\treturn a - c\n}\n"
	// Mutable ops: `==`, `+`, `-` -> 3 mutants uncapped.
	if all := GenerateMutants("g.go", src, 0); len(all) != 3 {
		t.Fatalf("uncapped want 3 mutants, got %d: %+v", len(all), all)
	}
	if capped := GenerateMutants("g.go", src, 2); len(capped) != 2 {
		t.Fatalf("cap=2 want 2 mutants, got %d", len(capped))
	}
}

// TestGenerateMutants_UnparsableIsNoOp proves a file the generator cannot parse yields no
// mutants and never panics.
func TestGenerateMutants_UnparsableIsNoOp(t *testing.T) {
	if got := GenerateMutants("bad.go", "this is not go", 0); got != nil {
		t.Fatalf("unparsable source should yield no mutants, got %+v", got)
	}
}

// TestFold_SurvivorsAreSoftNeverHard is the SOFT contract: survivors land on Soft, never on
// Defects, so the KPI adds zero HARD debt and can never gate the card (#3845).
func TestFold_SurvivorsAreSoftNeverHard(t *testing.T) {
	kpi := Fold([]PackageResult{
		{Pkg: "internal/foo", Applied: 4, Survived: 1, Survivors: []string{"foo.go:3 > -> >="}},
	})
	if kpi.Key != KPIKey {
		t.Errorf("key = %q, want %q", kpi.Key, KPIKey)
	}
	if len(kpi.Defects) != 0 {
		t.Errorf("survivors must NOT be HARD defects, got %d: %v", len(kpi.Defects), kpi.Defects)
	}
	if len(kpi.Soft) != 1 {
		t.Fatalf("want 1 SOFT survivor line, got %d: %v", len(kpi.Soft), kpi.Soft)
	}
	if kpi.Score != 75 { // 3 killed / 4 applied
		t.Errorf("score = %v, want 75 (kill rate)", kpi.Score)
	}
}

// TestFold_NoMutantsIsInsufficient100 proves an empty probe scores a clean 100 (INSUFFICIENT),
// never a hollow 0 -- absence of a survivor is not a survivor.
func TestFold_NoMutantsIsInsufficient100(t *testing.T) {
	kpi := Fold(nil)
	if kpi.Score != 100 || len(kpi.Defects) != 0 {
		t.Errorf("empty probe want score 100 / 0 defects, got score %v / %d defects", kpi.Score, len(kpi.Defects))
	}
}

// TestProbePackage_RestoresAlwaysWithFakeRunner exercises the mutate/restore orchestration with
// an injected runner (no toolchain): every mutant survives (runner always "passes"), and the
// source file is byte-identical to the original afterward -- restore-always holds.
func TestProbePackage_RestoresAlwaysWithFakeRunner(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "s.go")
	orig := []byte("package s\n\nfunc Pos(n int) bool { return n > 0 }\n")
	if err := os.WriteFile(srcPath, orig, 0o644); err != nil {
		t.Fatal(err)
	}
	// A runner that always reports PASS => every mutant survives.
	res := ProbePackage(dir, func(string) bool { return true }, 8)
	if res.Applied == 0 {
		t.Fatalf("expected at least one mutant applied, got %+v", res)
	}
	if res.Survived != res.Applied {
		t.Errorf("always-pass runner: want all %d survive, got %d", res.Applied, res.Survived)
	}
	after, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(orig) {
		t.Errorf("restore-always violated: file not restored to original\ngot:  %q\nwant: %q", after, orig)
	}
}

// TestProbePackage_EndToEnd is the issue's witness (#3845): a fixture package with a
// DELIBERATELY WEAK test shows a surviving mutant, while a fixture with a STRONG test kills
// every mutant -- proven by the REAL `go test` runner over the on-disk fixtures. It also
// re-asserts the tree is left clean.
func TestProbePackage_EndToEnd(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not on PATH; skipping the real-runner mutation witness")
	}
	run := GoTestRunner(120 * time.Second)

	weak := filepath.Join("testdata", "weakpkg")
	weakSrc := filepath.Join(weak, "weak.go")
	weakBefore := readFile(t, weakSrc)
	wr := ProbePackage(weak, run, 8)
	if wr.Err != "" {
		t.Fatalf("weakpkg probe errored: %s", wr.Err)
	}
	if wr.Applied == 0 {
		t.Fatalf("weakpkg: expected mutants applied, got %+v", wr)
	}
	if wr.Survived == 0 {
		t.Fatalf("weakpkg: a deliberately weak test MUST leave a surviving mutant, got 0 (%+v)", wr)
	}
	if got := readFile(t, weakSrc); got != weakBefore {
		t.Errorf("weakpkg source not restored after probe")
	}

	strong := filepath.Join("testdata", "strongpkg")
	strongSrc := filepath.Join(strong, "strong.go")
	strongBefore := readFile(t, strongSrc)
	sr := ProbePackage(strong, run, 8)
	if sr.Err != "" {
		t.Fatalf("strongpkg probe errored: %s", sr.Err)
	}
	if sr.Applied == 0 {
		t.Fatalf("strongpkg: expected mutants applied, got %+v", sr)
	}
	if sr.Survived != 0 {
		t.Errorf("strongpkg: a strong test should kill every mutant, got %d survivor(s): %v", sr.Survived, sr.Survivors)
	}
	if got := readFile(t, strongSrc); got != strongBefore {
		t.Errorf("strongpkg source not restored after probe")
	}
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
