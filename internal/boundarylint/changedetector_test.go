package boundarylint

import (
	"os"
	"path/filepath"
	"testing"
)

// TestChangeDetectorRule verifies the witness (verify the verifier): frozen-literal
// assertions — a magic enumeration count, a wholly-literal list equality, a pinned
// version string — are flagged; relational/invariant assertions and small structural
// fixture counts are not.
func TestChangeDetectorRule(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want int
	}{
		{"magic count flagged", `package p
func f(verbs []string) bool { return len(verbs) != 109 }`, 1},
		{"magic count reversed operands flagged", `package p
func f(verbs []string) bool { return 42 == len(verbs) }`, 1},
		{"small fixture count clean", `package p
func f(parts []string) bool { return len(parts) != 3 }`, 0},
		{"count against variable clean", `package p
func f(got []string, want int) bool { return len(got) != want }`, 0},
		{"len ordering relation clean", `package p
func f(a, b []string) bool { return len(a) < len(b) }`, 0},
		{"frozen list equality flagged", `package p
import "reflect"
func f(got []string) bool {
	return reflect.DeepEqual(got, []string{"a", "b", "c", "d", "e", "f"})
}`, 1},
		{"frozen list via slices.Equal flagged", `package p
import "slices"
func f(got []int) bool { return slices.Equal(got, []int{1, 2, 3, 4, 5}) }`, 1},
		{"frozen list via cmp.Diff flagged", `package p
import "cmp"
func f(got []string) bool { return cmp.Diff(got, []string{"a", "b", "c", "d", "e"}) == "" }`, 1},
		{"frozen map equality flagged", `package p
import "reflect"
func f(got map[string]int) bool { return reflect.DeepEqual(got, map[string]int{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}) }`, 1},
		{"short literal list clean", `package p
import "reflect"
func f(got []string) bool { return reflect.DeepEqual(got, []string{"a", "b"}) }`, 0},
		{"list with variables clean", `package p
import "reflect"
func f(got, a, b, c, d, e []string) bool {
	return reflect.DeepEqual(got, [][]string{a, b, c, d, e})
}`, 0},
		{"version literal flagged", `package p
func f(version string) bool { return version == "v1.42.7" }`, 1},
		{"version literal via selector flagged", `package p
type cfg struct{ Version string }
func f(c cfg) bool { return c.Version != "2.11.0" }`, 1},
		{"non-version string compare clean", `package p
func f(name string) bool { return name == "v-shape" }`, 0},
		{"version compared to variable clean", `package p
func f(version, want string) bool { return version == want }`, 0},
		{"version-y literal without version name clean", `package p
func f(id string) bool { return id == "1.2.3" }`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := checkSrc(t, ChangeDetectorTest{}, tc.src)
			if len(got) != tc.want {
				t.Fatalf("got %d findings %v, want %d", len(got), got, tc.want)
			}
		})
	}
}

// TestScanTestsWalksOnlyTestFiles guards the file filter from both sides: the same
// tell planted in a .go and a _test.go must be reported by exactly one scanner each —
// ScanTests sees only the test file, Scan only the non-test file.
func TestScanTestsWalksOnlyTestFiles(t *testing.T) {
	dir := t.TempDir()
	tell := `package p
func f(verbs []string) bool { return len(verbs) == 109 }
`
	if err := os.WriteFile(filepath.Join(dir, "prod.go"), []byte(tell), 0o644); err != nil {
		t.Fatalf("write prod.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "prod_test.go"), []byte(tell), 0o644); err != nil {
		t.Fatalf("write prod_test.go: %v", err)
	}

	rules := []Rule{ChangeDetectorTest{}}

	fromTests, err := ScanTests([]string{dir}, rules)
	if err != nil {
		t.Fatalf("ScanTests: %v", err)
	}
	if len(fromTests) != 1 || !filepath.IsAbs(fromTests[0].File) && fromTests[0].File == "" {
		t.Fatalf("ScanTests findings = %v, want exactly 1 (from prod_test.go)", fromTests)
	}
	if got := filepath.Base(filepath.FromSlash(fromTests[0].File)); got != "prod_test.go" {
		t.Fatalf("ScanTests hit %s, want prod_test.go", got)
	}

	fromScan, err := Scan([]string{dir}, rules)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	if len(fromScan) != 1 {
		t.Fatalf("Scan findings = %v, want exactly 1 (from prod.go)", fromScan)
	}
	if got := filepath.Base(filepath.FromSlash(fromScan[0].File)); got != "prod.go" {
		t.Fatalf("Scan hit %s, want prod.go", got)
	}
}

// TestScanTestsHonorsIgnoreComment keeps the family's suppression contract uniform:
// a trailing //boundarylint:ignore on the offending line silences the tell in a test
// file the same way it does in production source.
func TestScanTestsHonorsIgnoreComment(t *testing.T) {
	dir := t.TempDir()
	src := `package p
func f(hash string, verbs []string) bool {
	if len(hash) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST sha256 hex width is an algorithm invariant
		return false
	}
	return len(verbs) == 109
}
`
	if err := os.WriteFile(filepath.Join(dir, "sup_test.go"), []byte(src), 0o644); err != nil {
		t.Fatalf("write sup_test.go: %v", err)
	}
	got, err := ScanTests([]string{dir}, []Rule{ChangeDetectorTest{}})
	if err != nil {
		t.Fatalf("ScanTests: %v", err)
	}
	if len(got) != 1 || got[0].Line != 6 {
		t.Fatalf("findings = %v, want only the unsuppressed line-6 tell", got)
	}
}

// TestTestSuitePolicy is the live enforcement entrypoint for the test-suite family,
// the way TestBoundaryPolicy is for the production-source family — but as a RATCHET,
// not a hard failure over the whole tree. The tree already carries dozens of tells
// (some genuine change-detectors, some deliberate fixed-width invariants the heuristic
// can't yet distinguish); failing on all of them would red the trunk on day one. So a
// tell in a file listed in changeDetectorBaseline is grandfathered, and this test
// fails only when a NEW file (not in the baseline) introduces a change-detector — the
// same shrink-only ratchet as internal/pythongate. That stops the fast-growing suite
// from accreting new change-detectors while the backlog is burned down over time.
func TestCurrentChangeDetectorBaselineSnapshot(t *testing.T) {
	set := changeDetectorBaselineSet()
	for _, path := range []string{
		"cmd/disambiguationdemo/main_test.go",
		"internal/archreport/report_test.go",
		"internal/sessionrecovery/recovery_test.go",
		"internal/workaccount/registry_test.go",
	} {
		if !set[path] {
			t.Errorf("current grandfathered change-detector file %q missing from baseline", path)
		}
	}
	for _, stale := range []string{"internal/ggufload/gemma4_test.go", "internal/selfinstall/reap_test.go"} {
		if set[stale] {
			t.Errorf("stale change-detector baseline entry %q was not retired", stale)
		}
	}
}

func TestTestSuitePolicy(t *testing.T) {
	root := repoRoot(t)
	// The fail path is the shared ratchet entrypoint: only NEW (non-grandfathered)
	// change-detectors count. `fak boundary` reports the exact same set.
	newTells, err := ScanNewChangeDetectors(root)
	if err != nil {
		t.Fatalf("scan new change-detectors: %v", err)
	}
	for _, f := range newTells {
		t.Errorf("NEW change-detector test (not grandfathered in changeDetectorBaseline): %s\n"+
			"    assert the relation the value must hold, or //boundarylint:ignore CHANGE_DETECTOR_TEST a deliberate fixed-width invariant", f)
	}

	// Push the ratchet to tighten: a grandfathered file that no longer trips the rule
	// (its tells were converted to invariants) should be dropped from the baseline so
	// the list can only shrink. This is a soft signal, not a failure — a file can leave
	// the tracked set for unrelated reasons — but it keeps the baseline from rotting.
	all, err := ScanTests([]string{filepath.Join(root, "cmd"), filepath.Join(root, "internal")}, DefaultTestRules())
	if err != nil {
		t.Fatalf("scan tests: %v", err)
	}
	seen := map[string]bool{}
	for _, f := range all {
		if rel, rerr := filepath.Rel(root, filepath.FromSlash(f.File)); rerr == nil {
			seen[filepath.ToSlash(rel)] = true
		}
	}
	for p := range changeDetectorBaselineSet() {
		if !seen[p] {
			t.Logf("changeDetectorBaseline entry %q no longer trips the rule; drop it on the next refreeze", p)
		}
	}
}

// TestCatalogCoversTestSuiteRules keeps the documented policy honest for the
// test-suite family the way TestCatalogCoversEnforcedRules does for DefaultRules:
// every rule in DefaultTestRules must have a catalog entry.
func TestCatalogCoversTestSuiteRules(t *testing.T) {
	byCode := map[string]CatalogEntry{}
	for _, e := range Catalog {
		byCode[e.Code] = e
	}
	for _, r := range DefaultTestRules() {
		if _, ok := byCode[r.Code()]; !ok {
			t.Errorf("test-suite rule %s has no catalog entry", r.Code())
		}
	}
}
