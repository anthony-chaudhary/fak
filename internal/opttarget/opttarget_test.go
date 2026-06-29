package opttarget

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// workingSet is the synthetic LRU working-set size the in-process measurement
// models: hit rate = min(size, workingSet)/workingSet. It is pure (no git, no
// worktree, no probe), so these tests run on every host, and — crucially — the
// SAME closure backs both the hand-wired and the compiled harness, so any journal
// difference is the COMPILER's doing, not the measurement's.
const workingSet = 7

// inProcessSeams returns the deterministic baseline/measure closures both
// harnesses share. The baseline is DefaultCacheSize's hit rate on a fixed ref;
// each candidate's metric is its own size's hit rate.
func inProcessSeams() (func() (float64, string, error), func(rsiloop.Candidate) (rsiloop.Measurement, error)) {
	hit := func(size int) float64 {
		if size >= workingSet {
			return 1.0
		}
		return float64(size) / workingSet
	}
	baseline := func() (float64, string, error) {
		return hit(rsiloop.DefaultCacheSize), "deadbeefcafe", nil
	}
	measure := func(c rsiloop.Candidate) (rsiloop.Measurement, error) {
		size, ok := c.Payload.(int)
		if !ok {
			return rsiloop.Measurement{}, fmt.Errorf("payload %T, want int", c.Payload)
		}
		return rsiloop.Measurement{Metric: hit(size), SuiteGreen: true, TruthClean: true}, nil
	}
	return baseline, measure
}

// runToJSONL drives a harness through rsiloop.Run into a temp journal and returns
// the file's bytes — the durable, replayable record the golden comparison reads.
func runToJSONL(t *testing.T, h rsiloop.Harness) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	j, err := rsiloop.NewJournal(path)
	if err != nil {
		t.Fatalf("new journal: %v", err)
	}
	if _, err := rsiloop.Run(h, j, 3, 0); err != nil {
		t.Fatalf("run: %v", err)
	}
	if err := j.Close(); err != nil {
		t.Fatalf("close journal: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	return string(b)
}

// TestCompiledJournalMatchesHandWired is the Phase 0 witness (epic #1279): a
// DECLARED OptTarget, compiled and driven by the SAME measurement seams the hand-
// wired harness uses, yields a BYTE-IDENTICAL journal. The only fields that could
// differ are compiler-lowered — the metric name (every row), the direction (which
// drives Improved/Decision/Kept), the candidate labels+payloads (from the
// grammar), and the baseline ref name — so a byte match proves the lowering is
// faithful: declaring a target is equivalent to hand-coding its harness.
func TestCompiledJournalMatchesHandWired(t *testing.T) {
	// 5,6 improve over the size-4 baseline (KEEP,KEEP); 4 regresses below the kept
	// 6 (REVERT); 8 saturates to 1.0 (KEEP) — a KEEP/REVERT mix with no escalate.
	values := []int{5, 6, 4, 8}
	baseline, measure := inProcessSeams()

	// (A) the hand-wired harness — exactly what a human writes today, no OptTarget.
	hand := rsiloop.Harness{
		MetricName:      "lru_hit_rate",
		LowerBetter:     false,
		BaselineRefName: "main",
		BaselineMetric:  baseline,
		Candidates: func() []rsiloop.Candidate {
			cs := make([]rsiloop.Candidate, 0, len(values))
			for _, n := range values {
				cs = append(cs, rsiloop.Candidate{
					Label:   fmt.Sprintf("%s=%d", rsiloop.TunableConstName, n),
					Payload: n,
				})
			}
			return cs
		},
		Measure: measure,
	}

	// (B) the DECLARED target, compiled and driven by the SAME seams via the real
	// HarnessMeasurer adapter (so the adapter is exercised too). Its metric name,
	// direction, and candidates come from the declaration — not from `hand`.
	compiled, err := Compile(CacheSizeTarget(values), HarnessMeasurer{H: hand})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	got, want := runToJSONL(t, compiled), runToJSONL(t, hand)
	if got != want {
		t.Fatalf("compiled journal != hand-wired journal\n hand-wired:\n%s\n compiled:\n%s", want, got)
	}
}

// TestCandidatesMatchHandWired is the direct (non-journal) form of the witness:
// the grammar lowers to exactly the hand-wired label/payload list. It localizes a
// label-format regression to the grammar rather than surfacing it only as a
// journal-diff failure.
func TestCandidatesMatchHandWired(t *testing.T) {
	values := []int{4, 5, 16, -1}
	tgt := CacheSizeTarget(values)
	got := tgt.candidates()
	if len(got) != len(values) {
		t.Fatalf("got %d candidates, want %d", len(got), len(values))
	}
	for i, n := range values {
		wantLabel := fmt.Sprintf("%s=%d", rsiloop.TunableConstName, n)
		if got[i].Label != wantLabel {
			t.Errorf("candidate %d label = %q, want %q", i, got[i].Label, wantLabel)
		}
		if got[i].Payload != n {
			t.Errorf("candidate %d payload = %v, want %d", i, got[i].Payload, n)
		}
	}
}

// TestCompileRejectsMalformed proves a malformed declaration is REFUSED, never
// lowered into a harness that measures nothing.
func TestCompileRejectsMalformed(t *testing.T) {
	_, measure := inProcessSeams()
	good := CacheSizeTarget([]int{5})
	m := HarnessMeasurer{H: rsiloop.Harness{Measure: measure}}

	cases := []struct {
		name string
		tgt  OptTarget
		m    Measurer
	}{
		{"empty-name", func() OptTarget { x := good; x.Name = ""; return x }(), m},
		{"empty-metric", func() OptTarget { x := good; x.Metric = ""; return x }(), m},
		{"empty-baseline-ref", func() OptTarget { x := good; x.BaselineRef = ""; return x }(), m},
		{"empty-measurer", func() OptTarget { x := good; x.Measurer = ""; return x }(), m},
		{"empty-sweep", func() OptTarget { x := good; x.Grammar = Grammar{Kind: GrammarIntSweep}; return x }(), m},
		{"no-site-const", func() OptTarget { x := good; x.Site.Const = ""; return x }(), m},
		{"unknown-grammar", func() OptTarget { x := good; x.Grammar.Kind = "free-form"; return x }(), m},
		{"nil-measurer", good, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Compile(c.tgt, c.m); err == nil {
				t.Fatalf("Compile(%s) = nil error, want refusal", c.name)
			}
		})
	}
}

// TestCacheSizeTargetCompiles proves the shipped demo declaration is well-formed:
// it validates and compiles into a harness whose declaration-derived fields are
// the cache-size demo's.
func TestCacheSizeTargetCompiles(t *testing.T) {
	_, measure := inProcessSeams()
	tgt := CacheSizeTarget([]int{5, 6})
	h, err := Compile(tgt, HarnessMeasurer{H: rsiloop.Harness{Measure: measure}})
	if err != nil {
		t.Fatalf("compile demo target: %v", err)
	}
	if h.MetricName != "lru_hit_rate" {
		t.Errorf("MetricName = %q, want lru_hit_rate", h.MetricName)
	}
	if h.LowerBetter {
		t.Error("LowerBetter = true, want false (cache hit rate is higher-better)")
	}
	if h.BaselineRefName != "main" {
		t.Errorf("BaselineRefName = %q, want main", h.BaselineRefName)
	}
	if n := len(h.Candidates()); n != 2 {
		t.Errorf("Candidates() len = %d, want 2", n)
	}
}
