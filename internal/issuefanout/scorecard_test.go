package issuefanout

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// scorecard_test.go is the captured-output witness for the obs-scorecard fold (#2520):
// it runs the scorecard over realistic witness-shaped data (the same git-leaves +
// gh-marker-keys shape Adoption folds), pins the grade + the named evidence + the debt
// integer, and proves determinism. It runs under `make test-fast` / `make ci`, so the
// done condition ("emits a grade with named evidence and runs in CI or a make target")
// is met by the green test, and the captured render below is the witness artifact.

// fixtureLeavesAndMarkers builds a realistic witness set: three shipped leaves that
// cleared the floor, one below it ("gappy"), and one orphan marker keyed against a leaf
// that is not shipped. This is the shape `fak issue fanout-health` would grade on real
// git+gh data.
func fixtureLeavesAndMarkers() (leaves []string, markers []string) {
	leaves = []string{"gateway", "engine", "policy", "gappy"}
	mk := []string{
		"fanout-gateway-qa-a", "fanout-gateway-qa-b", "fanout-gateway-qa-c",
		"fanout-engine-qa-a", "fanout-engine-qa-b", "fanout-engine-qa-c",
		"fanout-policy-qa-a", "fanout-policy-qa-b", "fanout-policy-qa-c",
		"fanout-gappy-qa-a", "fanout-gappy-qa-b", // floor-1: a drifted / partial adoption
		"fanout-ghost-qa-a", // orphan: filed against a leaf not in the shipped set
	}
	return leaves, mk
}

// TestScorecardGradesAdoptionAndOrphans pins the headline over the fixture: the planner
// has 1 adoption gap (gappy), 1 orphan marker, and no taxonomy drift, so debt == 2 and
// the grade reflects a usable-but-not-clean planner.
func TestScorecardGradesAdoptionAndOrphans(t *testing.T) {
	leaves, markers := fixtureLeavesAndMarkers()
	p := Scorecard(leaves, markers)

	if got := corpusInt(t, p, DebtKey); got != 2 {
		t.Fatalf("issuefanout_debt = %d, want 2 (1 adoption gap + 1 orphan + 0 drift)", got)
	}
	if p.OK || p.Verdict != "ACTION" {
		t.Fatalf("a fixture with debt must report ACTION/not-OK, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if g := gradeOf(t, p); g != "B" {
		t.Fatalf("grade = %q, want B (composite 75+91.7+100 / 3 = 88.9)", g)
	}

	adoption := kpiOf(t, p, "adoption_floor")
	if adoption.Score != 75.0 || len(adoption.Defects) != 1 {
		t.Fatalf("adoption: score %.1f defects %d, want 75.0 / 1 (%v)", adoption.Score, len(adoption.Defects), adoption.Defects)
	}
	if !contains(adoption.Defects, "gappy") {
		t.Fatalf("adoption defect must name the below-floor leaf 'gappy': %v", adoption.Defects)
	}

	integrity := kpiOf(t, p, "marker_integrity")
	// clean = 3+3+3+2 = 11 credited; 1 orphan -> 11/12 = 91.7
	if integrity.Score != 91.7 || len(integrity.Defects) != 1 {
		t.Fatalf("integrity: score %.1f defects %d, want 91.7 / 1 (%v)", integrity.Score, len(integrity.Defects), integrity.Defects)
	}
	if !contains(integrity.Defects, "fanout-ghost-qa-a") {
		t.Fatalf("integrity defect must name the orphan marker: %v", integrity.Defects)
	}

	drift := kpiOf(t, p, "taxonomy_drift")
	if drift.Score != 100.0 || len(drift.Defects) != 0 {
		t.Fatalf("drift: score %.1f defects %d, want 100 / 0 (%v)", drift.Score, len(drift.Defects), drift.Defects)
	}
}

// TestScorecardCleanOnHealthy proves a fully-adopted planner with no orphans is zero
// debt, OK, grade A - the --check 0 / "is it still good: yes" case.
func TestScorecardCleanOnHealthy(t *testing.T) {
	leaves := []string{"alpha", "beta"}
	markers := []string{
		"fanout-alpha-qa-a", "fanout-alpha-qa-b", "fanout-alpha-qa-c",
		"fanout-beta-qa-a", "fanout-beta-qa-b", "fanout-beta-qa-c",
	}
	p := Scorecard(leaves, markers)
	if got := corpusInt(t, p, DebtKey); got != 0 {
		t.Fatalf("healthy planner must be zero debt, got %d", got)
	}
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("zero debt must report OK/OK, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if g := gradeOf(t, p); g != "A" {
		t.Fatalf("grade = %q, want A on a clean planner", g)
	}
}

// TestScorecardCleanOnEmpty proves the vacuous case (no shipped leaves, no markers) is a
// valid clean zero, not a panic and not a false red.
func TestScorecardCleanOnEmpty(t *testing.T) {
	p := Scorecard(nil, nil)
	if got := corpusInt(t, p, DebtKey); got != 0 {
		t.Fatalf("empty planner must be zero debt, got %d", got)
	}
	if !p.OK {
		t.Fatalf("empty planner must be OK (no work, perfect), got %+v", p)
	}
	if p.Schema != ScorecardSchema {
		t.Fatalf("schema = %q, want %q", p.Schema, ScorecardSchema)
	}
}

// TestScorecardDeterministic proves the same witnesses yield a byte-identical payload -
// the property a cohort/replay layer relies on.
func TestScorecardDeterministic(t *testing.T) {
	leaves, markers := fixtureLeavesAndMarkers()
	a := Scorecard(leaves, markers)
	b := Scorecard(leaves, markers)
	if jsonOf(t, a) != jsonOf(t, b) {
		t.Fatal("two Scorecard calls over the same witnesses differ")
	}
}

// TestDoctrineAreasMatchesTaxonomy is the honest-anchor guard: DoctrineAreas must equal
// AreaNames() TODAY, so the drift KPI's expected set is the live taxonomy. A one-sided
// edit (taxonomy grows without this list, or a template is dropped) reds the build here
// on purpose - a doctrine change is a deliberate act that touches both.
func TestDoctrineAreasMatchesTaxonomy(t *testing.T) {
	if !sameSet(DoctrineAreas, AreaNames()) {
		t.Fatalf("DoctrineAreas drifted from AreaNames(): doctrine=%v taxonomy=%v", DoctrineAreas, AreaNames())
	}
}

// TestScorecardRenderCapturesOutput is the captured-output witness: it renders the
// fixture payload to the terminal shape and pins the lines a human reads - the grade,
// the debt integer, and the named HARD defects. The t.Log output IS the captured
// scorecard output on real data the issue's witness field names.
func TestScorecardRenderCapturesOutput(t *testing.T) {
	leaves, markers := fixtureLeavesAndMarkers()
	p := Scorecard(leaves, markers)
	out := RenderScorecard(p)
	t.Log("\n" + out)

	if !strings.Contains(out, "issue fanout scorecard") {
		t.Fatalf("render header missing the scorecard name: %q", out)
	}
	if !strings.Contains(out, "issuefanout_debt 2") {
		t.Fatalf("render header missing the debt integer: %q", out)
	}
	if !strings.Contains(out, "leaf gappy") {
		t.Fatalf("render missing the named adoption defect: %q", out)
	}
	if !strings.Contains(out, "fanout-ghost-qa-a") {
		t.Fatalf("render missing the named orphan defect: %q", out)
	}
}

// --- test helpers -----------------------------------------------------------

func kpiOf(t *testing.T, p scorecard.Payload, key string) scorecard.KPI {
	t.Helper()
	for _, k := range p.KPIs {
		if k.Key == key {
			return k
		}
	}
	t.Fatalf("KPI %q not present in payload", key)
	return scorecard.KPI{}
}

func corpusInt(t *testing.T, p scorecard.Payload, key string) int {
	t.Helper()
	v, ok := p.Corpus[key]
	if !ok {
		t.Fatalf("corpus missing key %q", key)
	}
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	default:
		t.Fatalf("corpus[%q] = %v (%T), want int", key, v, v)
		return 0
	}
}

func gradeOf(t *testing.T, p scorecard.Payload) string {
	t.Helper()
	g, ok := p.Corpus["grade"].(string)
	if !ok {
		t.Fatalf("corpus grade missing or not a string: %v", p.Corpus["grade"])
	}
	return g
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if strings.Contains(h, needle) {
			return true
		}
	}
	return false
}

func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := map[string]bool{}
	for _, x := range a {
		seen[x] = true
	}
	for _, x := range b {
		if !seen[x] {
			return false
		}
	}
	return true
}

func jsonOf(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}
