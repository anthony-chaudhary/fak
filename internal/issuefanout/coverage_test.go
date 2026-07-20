package issuefanout

import (
	"reflect"
	"strings"
	"testing"
)

// The fixture that pins the fold: a window with one fully-adopted leaf, one
// spine that shipped without its backlog, and one leaf with no spine at all.
// Each rate must isolate its own failure — the no-spine leaf must NOT also be
// counted as a fan-out gap, or a single missing spine would red both meters.
func TestCoverageTwoRatesFixture(t *testing.T) {
	witnesses := []LeafWitness{
		{Leaf: "adopted", HasTest: true, HasVerb: true},  // spine + backlog
		{Leaf: "silent", HasTest: true, HasVerb: true},   // spine, no backlog
		{Leaf: "nospine", HasTest: true, HasVerb: false}, // test only: not a spine
	}
	markers := []string{
		"fanout-adopted-qa-a", "fanout-adopted-qa-b", "fanout-adopted-docs-a", // == MinFanout
		"fanout-silent-qa-a", // below the floor
	}

	rep := Coverage(witnesses, markers, CoverageScan{Issues: 10, Cap: 100})

	if rep.Schema != CoverageSchema {
		t.Fatalf("Schema: got %q want %q", rep.Schema, CoverageSchema)
	}
	// Spine rate is over ALL new leaves: 2 of 3 carry a spine witness.
	if rep.NewLeaves != 3 || rep.SpineWitnessed != 2 {
		t.Fatalf("spine rate: got %d/%d want 2/3", rep.SpineWitnessed, rep.NewLeaves)
	}
	if got, want := rep.SpineCoverage, 2.0/3.0; got != want {
		t.Fatalf("SpineCoverage: got %v want %v", got, want)
	}
	// Fan-out rate is over SPINES only: 1 of the 2 spines cleared the floor.
	// The no-spine leaf is excluded from this denominator entirely.
	if rep.Spines != 2 || rep.FanoutCleared != 1 {
		t.Fatalf("fan-out rate: got %d/%d want 1/2", rep.FanoutCleared, rep.Spines)
	}
	if rep.FanoutCoverage != 0.5 {
		t.Fatalf("FanoutCoverage: got %v want 0.5", rep.FanoutCoverage)
	}
	if !reflect.DeepEqual(rep.SpineGaps, []string{"nospine"}) {
		t.Fatalf("SpineGaps: got %v want [nospine]", rep.SpineGaps)
	}
	if !reflect.DeepEqual(rep.FanoutGaps, []string{"silent"}) {
		t.Fatalf("FanoutGaps: got %v want [silent] (nospine owes no follow-ons yet)", rep.FanoutGaps)
	}
	if rep.OK {
		t.Fatalf("OK: got true, want false — both rates are short")
	}

	byLeaf := map[string]LeafCoverage{}
	for _, l := range rep.Leaves {
		byLeaf[l.Leaf] = l
	}
	if got := byLeaf["adopted"]; !got.HasSpine || !got.ClearsFloor || got.FanoutFiled != MinFanout || got.Gap != 0 {
		t.Fatalf("adopted row: got %+v want spine+cleared, filed=%d gap=0", got, MinFanout)
	}
	if got := byLeaf["silent"]; !got.HasSpine || got.ClearsFloor || got.Gap != MinFanout-1 {
		t.Fatalf("silent row: got %+v want spine, not cleared, gap=%d", got, MinFanout-1)
	}
	// The critical isolation assertion: no spine => no fan-out obligation.
	if got := byLeaf["nospine"]; got.HasSpine || got.ClearsFloor || got.Gap != 0 {
		t.Fatalf("nospine row: got %+v want no spine, no fan-out gap charged", got)
	}
}

// An empty window must not read as a false-green 100% — the honesty meter's
// whole purpose. Both rates score 0 with a named 0 denominator, and OK is true
// only because there are genuinely no offenders.
func TestCoverageEmptyWindowIsNotFalseGreen(t *testing.T) {
	rep := Coverage(nil, nil, CoverageScan{Issues: 0, Cap: 100})
	if rep.SpineCoverage != 0 || rep.FanoutCoverage != 0 {
		t.Fatalf("empty window rates: got %v/%v want 0/0 (never a vacuous 1.0)",
			rep.SpineCoverage, rep.FanoutCoverage)
	}
	if rep.NewLeaves != 0 || rep.Spines != 0 {
		t.Fatalf("empty window denominators: got %d/%d want 0/0", rep.NewLeaves, rep.Spines)
	}
	if !rep.OK {
		t.Fatalf("OK: got false, want true — no leaves means no offenders")
	}
	if out := RenderCoverage(rep); !strings.Contains(out, "n/a") {
		t.Fatalf("render must name the empty denominator, got:\n%s", out)
	}
}

// A leaf reported across several gathered rows keeps the strongest witness seen:
// the file-list walk yields one row per artifact, and a later test-only row must
// not erase an earlier row's verb evidence.
func TestCoverageMergesDuplicateLeafRows(t *testing.T) {
	rep := Coverage([]LeafWitness{
		{Leaf: "split", HasVerb: true},
		{Leaf: "split", HasTest: true},
	}, nil, CoverageScan{Issues: 1, Cap: 100})
	if rep.NewLeaves != 1 {
		t.Fatalf("NewLeaves: got %d want 1 (duplicate rows collapse)", rep.NewLeaves)
	}
	if rep.SpineWitnessed != 1 {
		t.Fatalf("merged witness: got %d want 1 — test and verb rows must union", rep.SpineWitnessed)
	}
}

// Regression (measured on the real tracker): the first wiring of this scorecard
// reused --live's 300-issue dedupe cap for the marker scan. Fan-out markers
// cluster in OLDER issues, so that window returned ZERO of the 54 markers that
// actually exist and the meter rendered a confident "fan-out coverage 0.0%".
//
// A scan that hits its cap therefore cannot certify anything: it is marked
// truncated, forced NOT OK, and rendered as NOT PROVEN. A false-red is as
// dishonest as a false-green, and this is the only bit that tells them apart.
func TestCoverageTruncatedScanIsNotProven(t *testing.T) {
	witnesses := []LeafWitness{{Leaf: "adopted", HasTest: true, HasVerb: true}}
	markers := []string{"fanout-adopted-qa-a", "fanout-adopted-qa-b", "fanout-adopted-docs-a"}

	// A complete scan over the same inputs is genuinely clean.
	full := Coverage(witnesses, markers, CoverageScan{Issues: 42, Cap: 5000})
	if !full.OK || full.ScanTruncated {
		t.Fatalf("untruncated scan: got ok=%v truncated=%v want true/false", full.OK, full.ScanTruncated)
	}

	// The same inputs under a scan that hit its cap prove nothing.
	cut := Coverage(witnesses, markers, CoverageScan{Issues: 300, Cap: 300})
	if !cut.ScanTruncated {
		t.Fatalf("scan at cap must be marked truncated, got %+v", cut)
	}
	if cut.OK {
		t.Fatalf("OK: got true, want false — a truncated scan cannot certify the fan-out rate")
	}
	if out := RenderCoverage(cut); !strings.Contains(out, "NOT PROVEN") {
		t.Fatalf("render must warn on a truncated scan, got:\n%s", out)
	}
	// The coverage cap must NOT be the anti-spam dedupe cap: that reuse is the bug.
	if DefaultCoverageScanCap <= DefaultDedupeCap {
		t.Fatalf("DefaultCoverageScanCap (%d) must exceed DefaultDedupeCap (%d) — the dedupe window misses older markers",
			DefaultCoverageScanCap, DefaultDedupeCap)
	}
}

// The gatherers are pure and pinned too: git's added-path list reduces to the
// leaves it introduces, and the tracked file list decides each spine artifact.
func TestNewLeavesFromPathsAndWitnessLeaves(t *testing.T) {
	added := []string{
		"internal/alpha/alpha.go",
		"internal/alpha/alpha_test.go", // same leaf twice -> one entry
		`internal\beta\beta.go`,        // windows separators normalize
		"cmd/fak/alpha.go",             // not an internal leaf
		"docs/notes/thing.md",          // ignored
	}
	got := NewLeavesFromPaths(added)
	if !reflect.DeepEqual(got, []string{"alpha", "beta"}) {
		t.Fatalf("NewLeavesFromPaths: got %v want [alpha beta]", got)
	}

	tracked := []string{
		"internal/alpha/alpha.go",
		"internal/alpha/alpha_test.go", // alpha: test yes
		"cmd/fak/alpha_report.go",      // alpha: verb yes -> spine
		"internal/beta/beta.go",        // beta: no test
		"cmd/fak/betamax.go",           // must NOT credit beta (different verb)
	}
	ws := WitnessLeaves(got, tracked)
	byLeaf := map[string]LeafWitness{}
	for _, w := range ws {
		byLeaf[w.Leaf] = w
	}
	if a := byLeaf["alpha"]; !a.HasTest || !a.HasVerb {
		t.Fatalf("alpha witness: got %+v want test+verb", a)
	}
	if b := byLeaf["beta"]; b.HasTest || b.HasVerb {
		t.Fatalf("beta witness: got %+v want neither (betamax.go is a different verb)", b)
	}
}

// Marker keys are read out of real issue bodies, across both spellings of the
// marker comment, so issues filed before the name settled still count.
func TestExtractMarkerKeys(t *testing.T) {
	got := ExtractMarkerKeys([]Issue{
		{Number: 1, Body: "<!-- fak-issuefanout-key: fanout-alpha-qa-a -->\n## Lane\nalpha"},
		{Number: 2, Body: "<!-- fak-fanout-key: fanout-alpha-docs-b -->"},
		{Number: 3, Body: "no marker here"},
		{Number: 4, Body: "dupe <!-- fak-issuefanout-key: fanout-alpha-qa-a -->"},
	})
	if !reflect.DeepEqual(got, []string{"fanout-alpha-docs-b", "fanout-alpha-qa-a"}) {
		t.Fatalf("ExtractMarkerKeys: got %v want both spellings, deduped and sorted", got)
	}
}

// A verb shell splits the leaf name into underscore-separated words, so an
// exact prefix match scored those leaves as having no runnable verb and thus no
// spine. Measured on the real tracker, that dropped issuefanout itself — 19
// filed follow-ons — out of the fan-out denominator, which is a false RED.
func TestWitnessLeavesMatchesUnderscoreSplitVerbShell(t *testing.T) {
	tracked := []string{
		"internal/issuefanout/coverage.go",
		"internal/issuefanout/coverage_test.go",
		"cmd/fak/issue_fanout.go",          // the shell: issue_fanout <-> issuefanout
		"cmd/fak/issue_fanout_coverage.go", // a suffixed shell for the same leaf
		"internal/bench/bench.go",
		"internal/bench/bench_test.go",
		"cmd/fak/benchloop.go", // must NOT credit bench: no word break after "bench"
		"internal/agent/agent.go",
		"internal/agent/agent_test.go",
		"cmd/fak/agentreadinessscore.go", // must NOT credit agent
	}
	byLeaf := map[string]LeafWitness{}
	for _, w := range WitnessLeaves([]string{"issuefanout", "bench", "agent"}, tracked) {
		byLeaf[w.Leaf] = w
	}
	if w := byLeaf["issuefanout"]; !w.HasVerb {
		t.Fatalf("issuefanout: got %+v want HasVerb (cmd/fak/issue_fanout.go is its shell)", w)
	}
	if w := byLeaf["bench"]; w.HasVerb {
		t.Fatalf("bench: got %+v want no verb (benchloop.go is a different leaf)", w)
	}
	if w := byLeaf["agent"]; w.HasVerb {
		t.Fatalf("agent: got %+v want no verb (agentreadinessscore.go is a different leaf)", w)
	}
}

// A leaf that merely GAINED a file in the window is not new. Counting it
// inflated the spine denominator 1.9x on a real 14-day window over this repo.
func TestNewLeavesInWindowSubtractsPreexistingLeaves(t *testing.T) {
	inWindow := []string{
		"internal/fresh/fresh.go", // genuinely new
		"internal/old/newfile.go", // old leaf, merely gained a file
	}
	before := []string{
		"internal/old/old.go", // proves old predates the window
	}
	got := NewLeavesInWindow(inWindow, before)
	if !reflect.DeepEqual(got, []string{"fresh"}) {
		t.Fatalf("NewLeavesInWindow: got %v want [fresh] (old predates the window)", got)
	}
	// With no pre-window witness the whole history is the window, so nothing is
	// subtracted — the reading that is correct for a repo younger than --since.
	all := NewLeavesInWindow(inWindow, nil)
	if !reflect.DeepEqual(all, []string{"fresh", "old"}) {
		t.Fatalf("NewLeavesInWindow(nil before): got %v want [fresh old]", all)
	}
}
