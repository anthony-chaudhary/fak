package rsiloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// fakeStep is one scripted candidate measurement for the in-process harness — the
// engine tests drive Run WITHOUT a worktree, so the loop's keep/revert/escalate
// logic is exercised deterministically (no git, no go run, no wall-clock).
type fakeStep struct {
	label  string
	metric float64
	green  bool
	clean  bool
	err    error
}

func fakeHarness(name string, lowerBetter bool, baseline float64, baseRef string, steps []fakeStep) Harness {
	return Harness{
		MetricName:      name,
		LowerBetter:     lowerBetter,
		BaselineRefName: "test-ref",
		BaselineMetric: func() (float64, string, error) {
			return baseline, baseRef, nil
		},
		Candidates: func() []Candidate {
			cs := make([]Candidate, len(steps))
			for i, s := range steps {
				cs[i] = Candidate{Label: s.label, Payload: i}
			}
			return cs
		},
		Measure: func(c Candidate) (Measurement, error) {
			s := steps[c.Payload.(int)]
			if s.err != nil {
				return Measurement{}, s.err
			}
			return Measurement{Metric: s.metric, SuiteGreen: s.green, TruthClean: s.clean}, nil
		},
	}
}

// TestKPIMonotoneAndDeterministic locks the metric's two load-bearing properties:
// HitRate is non-decreasing in the cache size (a bigger LRU window can't create a
// miss), strictly rising over the demo range, and identical on every call (no RNG /
// wall-clock). The strict rises are what give the loop a real gain to find.
func TestKPIMonotoneAndDeterministic(t *testing.T) {
	prev := -1.0
	for n := 1; n <= workingSet; n++ {
		h := HitRate(n)
		if h < prev {
			t.Fatalf("HitRate not monotone: HitRate(%d)=%.6f < HitRate(%d)=%.6f", n, h, n-1, prev)
		}
		if h != HitRate(n) {
			t.Fatalf("HitRate(%d) not deterministic", n)
		}
		prev = h
	}
	// The exact baseline + candidate points the demo relies on (from `kpiprobe -dump`).
	if !(HitRate(4) < HitRate(6) && HitRate(6) < HitRate(8) && HitRate(8) < HitRate(10)) {
		t.Fatalf("demo points not strictly increasing: 4=%.6f 6=%.6f 8=%.6f 10=%.6f",
			HitRate(4), HitRate(6), HitRate(8), HitRate(10))
	}
	if HitRate(0) != 0 || HitRate(-5) != 0 {
		t.Fatalf("zero/negative cache must miss everything")
	}
}

// TestLoopKeepsRealGainsRevertsNoOp is the core trueness property: fed a sequence
// of measured metrics, the loop KEEPs each strict gain (advancing the running
// baseline — the recursion), and REVERTs the no-op. The keep-bit, not the test,
// decides; the test only supplies measurements a real worktree run would.
func TestLoopKeepsRealGainsRevertsNoOp(t *testing.T) {
	steps := []fakeStep{
		{"size=6", 0.157197, true, true, nil},  // gain over 0.068 -> KEEP
		{"size=8", 0.284091, true, true, nil},  // gain over 0.157 -> KEEP
		{"size=8", 0.284091, true, true, nil},  // no gain over 0.284 -> REVERT
		{"size=10", 0.467803, true, true, nil}, // gain over 0.284 -> KEEP
	}
	h := fakeHarness("lru_hit_rate", false, 0.068182, "deadbeef0000", steps)
	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Cycles != 4 {
		t.Fatalf("cycles=%d, want 4", res.Cycles)
	}
	if res.Kept != 3 {
		t.Fatalf("kept=%d, want 3", res.Kept)
	}
	wantKept := []bool{true, true, false, true}
	for i, r := range res.Rows {
		if r.Kept != wantKept[i] {
			t.Errorf("cycle %d kept=%v, want %v (decision=%s)", r.Cycle, r.Kept, wantKept[i], r.Decision)
		}
		if r.BaselineRef != "deadbeef0000" {
			t.Errorf("cycle %d baseline_ref=%q, want the main sha", r.Cycle, r.BaselineRef)
		}
		if !r.Measured {
			t.Errorf("cycle %d measured=false, want true (a real measurement)", r.Cycle)
		}
		if r.RefName != "test-ref" {
			t.Errorf("cycle %d ref_name=%q, want test-ref", r.Cycle, r.RefName)
		}
	}
	if res.FinalBaseline != 0.467803 {
		t.Fatalf("final baseline=%.6f, want 0.467803 (the last kept gain)", res.FinalBaseline)
	}
	// The no-op cycle's baseline must equal the prior kept gain (recursion advanced it).
	if res.Rows[2].Baseline != 0.284091 {
		t.Fatalf("revert cycle baseline=%.6f, want 0.284091", res.Rows[2].Baseline)
	}
}

// TestKeepBitNeedsAllThree proves the loop cannot keep a candidate on a metric gain
// alone: a huge gain with a RED suite, or with a DIRTY truth syscall, both REVERT.
// This is shipgate's keep-bit contract, exercised through the loop.
func TestKeepBitNeedsAllThree(t *testing.T) {
	cases := []struct {
		name string
		step fakeStep
		keep bool
	}{
		{"gain+green+clean -> KEEP", fakeStep{"c", 9.9, true, true, nil}, true},
		{"gain+RED suite -> REVERT", fakeStep{"c", 9.9, false, true, nil}, false},
		{"gain+DIRTY truth -> REVERT", fakeStep{"c", 9.9, true, false, nil}, false},
		{"measure ERROR -> REVERT", fakeStep{"c", 0, false, false, errBoom}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := fakeHarness("m", false, 1.0, "sha", []fakeStep{c.step})
			res, err := Run(h, nil, 3, 0)
			if err != nil {
				t.Fatal(err)
			}
			if res.Rows[0].Kept != c.keep {
				t.Errorf("kept=%v, want %v (decision=%s)", res.Rows[0].Kept, c.keep, res.Rows[0].Decision)
			}
		})
	}
}

// TestScorecardIsJournaledButNotAGateInput proves the structured score surface is
// telemetry, not authority: a candidate with a rich "lean" scorecard but no strict
// scalar gain still REVERTs, and the exact score travels to both the in-memory row
// and the durable JSONL journal for downstream RSI-like controls.
func TestScorecardIsJournaledButNotAGateInput(t *testing.T) {
	score := &Scorecard{
		Name:  "attention_sn",
		Value: 0.90,
		Grade: "lean",
		Components: []ScoreComponent{
			{Name: "mean_ratio", Value: 0.90, Unit: "ratio"},
			{Name: "mean_fault_ratio", Value: 0.0, Unit: "ratio"},
			{Name: "signal_tokens", Value: 9, Unit: "tokens"},
		},
	}
	h := Harness{
		MetricName:      "attention_sn",
		LowerBetter:     false,
		BaselineRefName: "test-ref",
		BaselineMetric: func() (float64, string, error) {
			return 0.90, "sha-score", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{{Label: "same-score"}}
		},
		Measure: func(Candidate) (Measurement, error) {
			return Measurement{Metric: 0.90, SuiteGreen: true, TruthClean: true, Score: score}, nil
		},
	}

	path := filepath.Join(t.TempDir(), "rsi.jsonl")
	j, err := NewJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Run(h, j, 3, 0)
	j.Close()
	if err != nil {
		t.Fatal(err)
	}
	if res.Rows[0].Kept || res.Rows[0].Decision != "REVERT" {
		t.Fatalf("scorecard must not move the keep-bit without scalar improvement: %+v", res.Rows[0])
	}
	if res.Rows[0].Score == nil || res.Rows[0].Score.Name != "attention_sn" || res.Rows[0].Score.Grade != "lean" {
		t.Fatalf("scorecard not copied to row: %+v", res.Rows[0].Score)
	}
	score.Grade = "mutated"
	score.Components[0].Value = 0.1
	if res.Rows[0].Score.Grade != "lean" || res.Rows[0].Score.Components[0].Value != 0.90 {
		t.Fatalf("row scorecard alias mutated by harness after the fact: %+v", res.Rows[0].Score)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var row Row
	if err := json.Unmarshal(b, &row); err != nil {
		t.Fatalf("journal row did not decode: %v\n%s", err, string(b))
	}
	if row.Score == nil || row.Score.Components[2].Name != "signal_tokens" || row.Score.Components[2].Value != 9 {
		t.Fatalf("journal lost structured scorecard: %+v", row.Score)
	}
}

// TestMeasureErrorMarksRowUnmeasured proves a candidate that fails to build/measure
// is journaled with Measured=false (so candidate_metric is NOT trusted as a real
// number) and carries a diagnostic Note — not a silent baseline-valued cell.
func TestMeasureErrorMarksRowUnmeasured(t *testing.T) {
	h := fakeHarness("m", false, 0.5, "sha", []fakeStep{{"broken", 0, false, false, errBoom}})
	res, _ := Run(h, nil, 3, 0)
	r := res.Rows[0]
	if r.Measured {
		t.Fatal("measure error must mark the row Measured=false")
	}
	if r.Kept || r.Decision != "REVERT" {
		t.Fatalf("measure error must REVERT, got kept=%v decision=%s", r.Kept, r.Decision)
	}
	if r.Note == "" {
		t.Fatal("measure error must leave a diagnostic Note")
	}
}

// TestBreakerEscalatesAndResets proves the escalation breaker: K consecutive
// non-keeps upgrade the decision to ESCALATE and stop the loop, and a KEEP in
// between RESETS the counter (so escalation needs K in a row, not K total).
func TestBreakerEscalatesAndResets(t *testing.T) {
	// k=2; pattern: revert, KEEP (reset), revert, revert -> escalate at cycle 4.
	steps := []fakeStep{
		{"a", 1.0, true, true, nil}, // no gain over baseline 1.0 -> REVERT (breaker 1)
		{"b", 2.0, true, true, nil}, // gain -> KEEP (breaker 0)
		{"c", 2.0, true, true, nil}, // no gain over 2.0 -> REVERT (breaker 1)
		{"d", 2.0, true, true, nil}, // no gain -> REVERT -> breaker 2 == k -> ESCALATE
	}
	h := fakeHarness("m", false, 1.0, "sha", steps)
	res, err := Run(h, nil, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Escalated {
		t.Fatalf("expected escalation, got final=%s", res.Final.String())
	}
	if res.Cycles != 4 {
		t.Fatalf("cycles=%d, want 4 (loop stops at escalate)", res.Cycles)
	}
	if res.Final != shipgate.ESCALATE {
		t.Fatalf("final=%s, want ESCALATE", res.Final.String())
	}
	if res.Kept != 1 {
		t.Fatalf("kept=%d, want 1", res.Kept)
	}
}

// TestLowerBetterDirection confirms the metric direction is honored: with
// LowerBetter, a SMALLER candidate metric is the gain.
func TestLowerBetterDirection(t *testing.T) {
	steps := []fakeStep{{"faster", 5.0, true, true, nil}} // 5 < baseline 10 -> KEEP
	h := fakeHarness("p50_latency", true, 10.0, "sha", steps)
	res, _ := Run(h, nil, 3, 0)
	if !res.Rows[0].Kept {
		t.Fatalf("lower-better gain should KEEP, got %s", res.Rows[0].Decision)
	}
	// And the wrong direction reverts.
	steps2 := []fakeStep{{"slower", 12.0, true, true, nil}}
	h2 := fakeHarness("p50_latency", true, 10.0, "sha", steps2)
	res2, _ := Run(h2, nil, 3, 0)
	if res2.Rows[0].Kept {
		t.Fatalf("lower-better regression should REVERT, got %s", res2.Rows[0].Decision)
	}
}

// TestJournalRoundTripAndTrack proves the journal is a replayable JSONL ledger and
// that Track/LastTrack form the ongoing benchmark-vs-main series.
func TestJournalRoundTripAndTrack(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rsi.jsonl")

	// One improve run (writes 1 row) then two track rows.
	j, err := NewJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	h := fakeHarness("lru_hit_rate", false, 0.10, "sha-aaaa", []fakeStep{{"size=6", 0.20, true, true, nil}})
	if _, err := Run(h, j, 3, 0); err != nil {
		t.Fatal(err)
	}
	// First track point.
	ht1 := fakeHarness("lru_hit_rate", false, 0.20, "sha-bbbb", nil)
	if _, err := Track(ht1, j); err != nil {
		t.Fatal(err)
	}
	// Second track point, lower (a regression on main).
	ht2 := fakeHarness("lru_hit_rate", false, 0.15, "sha-cccc", nil)
	if _, err := Track(ht2, j); err != nil {
		t.Fatal(err)
	}
	j.Close()

	// The clean file is well-formed JSONL: 3 non-empty lines.
	b, _ := os.ReadFile(path)
	lines := 0
	for _, ln := range splitNonEmpty(string(b)) {
		_ = ln
		lines++
	}
	if lines != 3 {
		t.Fatalf("journal has %d rows, want 3", lines)
	}

	// LastTrack returns the MOST RECENT track row (sha-cccc, 0.15), skipping the
	// improve row.
	last, ok, err := LastTrack(path)
	if err != nil || !ok {
		t.Fatalf("LastTrack ok=%v err=%v", ok, err)
	}
	if last.Mode != "track" || last.BaselineRef != "sha-cccc" || last.Baseline != 0.15 || last.RefName != "test-ref" {
		t.Fatalf("LastTrack = %+v, want the sha-cccc/0.15/test-ref track row", last)
	}
}

// TestLastTrackToleratesCorruption proves the regression guard does NOT fail open: a
// torn final line + a garbage line (a crash mid-Append) are SKIPPED, and the real
// prior track point is still returned. A single bad line must not blind the alert.
func TestLastTrackToleratesCorruption(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rsi.jsonl")
	j, err := NewJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Track(fakeHarness("m", false, 0.42, "sha-good", nil), j); err != nil {
		t.Fatal(err)
	}
	j.Close()

	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("{\"mode\":\"track\",\"baseline\":0.99\n") // torn: no closing brace
	f.WriteString("not json at all\n")
	f.Close()

	last, ok, err := LastTrack(path)
	if err != nil {
		t.Fatalf("LastTrack must not error on a corrupt tail: %v", err)
	}
	if !ok || last.BaselineRef != "sha-good" || last.Baseline != 0.42 {
		t.Fatalf("LastTrack = %+v (ok=%v), want the sha-good/0.42 row — corrupt tail must be skipped, not fail open", last, ok)
	}
}

// TestTunableRewriteContract guards the regex the worktree Proposer depends on: it
// must rewrite exactly the DefaultCacheSize literal and nothing else.
func TestTunableRewriteContract(t *testing.T) {
	src := []byte("const DefaultCacheSize = 4\nconst TunableConstName = \"DefaultCacheSize\"\n")
	if !tunableRewrite.Match(src) {
		t.Fatal("rewrite regex did not match the tunable literal")
	}
	out := tunableRewrite.ReplaceAll(src, []byte("${1}7"))
	want := "const DefaultCacheSize = 7\nconst TunableConstName = \"DefaultCacheSize\"\n"
	if string(out) != want {
		t.Fatalf("rewrite = %q, want %q", out, want)
	}
}

// TestParseKPI locks the probe-output contract the Measurer parses.
func TestParseKPI(t *testing.T) {
	v, err := parseKPI("some log line\nKPI=0.284091\n")
	if err != nil || v != 0.284091 {
		t.Fatalf("parseKPI = %v, %v; want 0.284091, nil", v, err)
	}
	if _, err := parseKPI("no kpi here"); err == nil {
		t.Fatal("expected error on missing KPI= line")
	}
}

func TestLRUHitRateScorecard(t *testing.T) {
	score := lruHitRateScorecard(8, HitRate(8))
	if score.Name != "lru_hit_rate" || score.Value != HitRate(8) {
		t.Fatalf("lru score header = %+v", score)
	}
	if got := scoreComponentValue(score, "cache_size"); got != 8 {
		t.Fatalf("cache_size component = %.0f, want 8 in %+v", got, score)
	}
	if got := scoreComponentValue(score, "trace_len"); got != float64(TraceLen()) {
		t.Fatalf("trace_len component = %.0f, want %d in %+v", got, TraceLen(), score)
	}
	if got := scoreComponentValue(score, "working_set"); got != workingSet {
		t.Fatalf("working_set component = %.0f, want %d in %+v", got, workingSet, score)
	}
	if score.Grade == "" {
		t.Fatalf("lru score should carry a grade: %+v", score)
	}
}

var errBoom = &boomErr{}

type boomErr struct{}

func (*boomErr) Error() string { return "boom" }

func splitNonEmpty(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == '\n' {
			if cur != "" {
				out = append(out, cur)
			}
			cur = ""
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

// TestTransientMeasureErrorRecoversWithoutTrippingBreaker reproduces issue #11608:
// two transient measurement failures at threshold 2 must not trip the regression
// breaker immediately and abandon later viable candidates. Transient attempts are
// retained as unmeasured evidence while candidates recover and keep their gains.
func TestTransientMeasureErrorRecoversWithoutTrippingBreaker(t *testing.T) {
	calls := make(map[string]int)

	h := Harness{
		MetricName:      "lru_hit_rate",
		LowerBetter:     false,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 1.0, "sha-base", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{Label: "c1"},
				{Label: "c2"},
				{Label: "c3"},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			calls[c.Label]++
			attempt := calls[c.Label]
			switch c.Label {
			case "c1":
				if attempt == 1 {
					return Measurement{}, NewTransientMeasureError(errors.New("git lock busy"))
				}
				return Measurement{Metric: 2.0, SuiteGreen: true, TruthClean: true}, nil
			case "c2":
				if attempt == 1 {
					return Measurement{}, NewTransientMeasureError(errors.New("network timeout"))
				}
				return Measurement{Metric: 3.0, SuiteGreen: true, TruthClean: true}, nil
			case "c3":
				return Measurement{Metric: 4.0, SuiteGreen: true, TruthClean: true}, nil
			default:
				return Measurement{}, errors.New("unknown candidate")
			}
		},
	}

	// Threshold k=2: in the unfixed code, 2 transient failures trip the breaker
	// immediately and abandon candidate c3. With bounded recovery, c1 and c2
	// recover, c3 is evaluated, and all three are kept.
	res, err := Run(h, nil, 2, 0)
	if err != nil {
		t.Fatal(err)
	}

	if res.Escalated {
		t.Fatalf("loop escalated prematurely; transient measurement errors must not trip the breaker immediately")
	}
	if res.Cycles != 3 {
		t.Fatalf("cycles=%d, want 3 (all candidates evaluated)", res.Cycles)
	}
	if res.Kept != 3 {
		t.Fatalf("kept=%d, want 3 (all candidates kept after recovery)", res.Kept)
	}
	if res.Final != shipgate.KEEP {
		t.Fatalf("final decision=%s, want KEEP", res.Final.String())
	}
	if res.FinalBaseline != 4.0 {
		t.Fatalf("final baseline=%.1f, want 4.0", res.FinalBaseline)
	}

	// 5 rows total:
	// row 0: c1 attempt 1 (transient error, unmeasured evidence, breaker 0)
	// row 1: c1 attempt 2 (measured gain, kept, breaker 0)
	// row 2: c2 attempt 1 (transient error, unmeasured evidence, breaker 0)
	// row 3: c2 attempt 2 (measured gain, kept, breaker 0)
	// row 4: c3 attempt 1 (measured gain, kept, breaker 0)
	if len(res.Rows) != 5 {
		t.Fatalf("len(res.Rows)=%d, want 5 (retaining all error attempts as unmeasured evidence)", len(res.Rows))
	}

	// Check unmeasured error evidence rows.
	for _, idx := range []int{0, 2} {
		r := res.Rows[idx]
		if r.Decision != "RETRY" {
			t.Errorf("row %d: decision=%s, want RETRY", idx, r.Decision)
		}
		if r.Measured {
			t.Errorf("row %d: measured=true, want false for transient error attempt", idx)
		}
		if r.Kept {
			t.Errorf("row %d: kept=true, want false for transient error attempt", idx)
		}
		if r.BreakerCount != 0 {
			t.Errorf("row %d: breaker count=%d, want 0 (transient recovery must not advance breaker)", idx, r.BreakerCount)
		}
		if !strings.Contains(r.Note, "transient") {
			t.Errorf("row %d: note %q should mention transient", idx, r.Note)
		}
	}

	// Check kept rows.
	for _, idx := range []int{1, 3, 4} {
		r := res.Rows[idx]
		if !r.Measured {
			t.Errorf("row %d: measured=false, want true", idx)
		}
		if !r.Kept {
			t.Errorf("row %d: kept=false, want true", idx)
		}
		if r.Decision != "KEEP" {
			t.Errorf("row %d: decision=%s, want KEEP", idx, r.Decision)
		}
		if r.BreakerCount != 0 {
			t.Errorf("row %d: breaker count=%d, want 0", idx, r.BreakerCount)
		}
	}
}

// TestTransientMeasureErrorExhaustionAdvancesBreaker proves that recovery is finite
// and bounded: when transient retries are exhausted, the failure advances the
// regression breaker and eventually escalates if failures persist.
func TestTransientMeasureErrorExhaustionAdvancesBreaker(t *testing.T) {
	calls := make(map[string]int)

	h := Harness{
		MetricName:      "p50",
		LowerBetter:     true,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 10.0, "sha-base", nil
		},
		TransientMeasurementRecoveryLimit: 1, // 1 retry allowed (total 2 attempts per candidate)
		Candidates: func() []Candidate {
			return []Candidate{
				{Label: "c1"},
				{Label: "c2"},
				{Label: "c3"},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			calls[c.Label]++
			return Measurement{}, NewTransientMeasureError(errors.New("service unavailable"))
		},
	}

	// Threshold k=2: c1 exhausts its retries (advances breaker to 1), then c2
	// exhausts its retries (advances breaker to 2 == k -> ESCALATE).
	res, err := Run(h, nil, 2, 0)
	if err != nil {
		t.Fatal(err)
	}

	if !res.Escalated {
		t.Fatalf("expected loop to escalate after exhausting retries on consecutive candidates")
	}
	if res.Final != shipgate.ESCALATE {
		t.Fatalf("final decision=%s, want ESCALATE", res.Final.String())
	}
	if res.Cycles != 2 {
		t.Fatalf("cycles=%d, want 2 (stopped on escalation at cycle 2)", res.Cycles)
	}

	// 4 rows total: c1 attempt 1 (recovering) + c1 attempt 2 (exhausted),
	// c2 attempt 1 (recovering) + c2 attempt 2 (exhausted -> escalate).
	if len(res.Rows) != 4 {
		t.Fatalf("len(res.Rows)=%d, want 4", len(res.Rows))
	}
	for i, r := range res.Rows {
		if r.Measured {
			t.Errorf("row %d measured=true, want false", i)
		}
	}
	if res.Rows[1].BreakerCount != 1 {
		t.Fatalf("c1 exhausted: breaker count=%d, want 1", res.Rows[1].BreakerCount)
	}
	if res.Rows[3].BreakerCount != 2 {
		t.Fatalf("c2 exhausted: breaker count=%d, want 2", res.Rows[3].BreakerCount)
	}
	if res.Rows[3].Decision != "ESCALATE" {
		t.Fatalf("c2 exhausted: decision=%s, want ESCALATE", res.Rows[3].Decision)
	}
}

// TestUntypedMeasureErrorPreservesImmediateRevert confirms untyped errors do not
// retry and immediately advance the breaker, preserving default legacy behavior.
func TestUntypedMeasureErrorPreservesImmediateRevert(t *testing.T) {
	calls := 0
	h := Harness{
		MetricName:      "m",
		LowerBetter:     false,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 1.0, "sha-base", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{{Label: "untyped-broken"}}
		},
		Measure: func(Candidate) (Measurement, error) {
			calls++
			return Measurement{}, errors.New("untyped compiler error")
		},
	}

	res, err := Run(h, nil, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("untyped error called Measure %d times, want exactly 1 (no retries)", calls)
	}
	if len(res.Rows) != 1 {
		t.Fatalf("len(res.Rows)=%d, want exactly 1 row", len(res.Rows))
	}
	r := res.Rows[0]
	if r.Measured {
		t.Fatal("untyped error must mark row Measured=false")
	}
	if r.BreakerCount != 1 {
		t.Fatalf("untyped error must advance breaker immediately, got %d", r.BreakerCount)
	}
	if r.Decision != "REVERT" {
		t.Fatalf("decision=%s, want REVERT", r.Decision)
	}
}

type customTransientErr struct {
	msg string
}

func (e *customTransientErr) Error() string          { return e.msg }
func (e *customTransientErr) TransientMeasure() bool { return true }

type customIsTransientErr struct {
	msg string
}

func (e *customIsTransientErr) Error() string     { return e.msg }
func (e *customIsTransientErr) IsTransient() bool { return true }

type customTransientBoolErr struct {
	msg string
}

func (e *customTransientBoolErr) Error() string   { return e.msg }
func (e *customTransientBoolErr) Transient() bool { return true }

// TestIsTransientMeasureErrorPredicate tests the recognition of transient errors.
func TestIsTransientMeasureErrorPredicate(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"untyped error", errors.New("something broke"), false},
		{"TransientMeasureError", NewTransientMeasureError(errors.New("disk full")), true},
		{"wrapped TransientMeasureError", fmt.Errorf("wrap: %w", NewTransientMeasureError(errors.New("inner"))), true},
		{"custom TransientMeasure()", &customTransientErr{"lock"}, true},
		{"custom IsTransient()", &customIsTransientErr{"timeout"}, true},
		{"custom Transient()", &customTransientBoolErr{"busy"}, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := IsTransientMeasureError(c.err)
			if got != c.want {
				t.Fatalf("IsTransientMeasureError(%v) = %v, want %v", c.err, got, c.want)
			}
		})
	}
}

// TestHarnessCustomIsTransientClassifier proves that Harness.IsTransientMeasureError
// allows callers to classify domain-specific errors without modifying error types.
func TestHarnessCustomIsTransientClassifier(t *testing.T) {
	attempts := 0
	h := Harness{
		MetricName:      "m",
		LowerBetter:     false,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 1.0, "sha-base", nil
		},
		IsTransientMeasureError: func(err error) bool {
			return strings.Contains(err.Error(), "transient-needle")
		},
		Candidates: func() []Candidate {
			return []Candidate{{Label: "c1"}}
		},
		Measure: func(Candidate) (Measurement, error) {
			attempts++
			if attempts == 1 {
				return Measurement{}, errors.New("error with transient-needle inside")
			}
			return Measurement{Metric: 2.0, SuiteGreen: true, TruthClean: true}, nil
		},
	}

	res, err := Run(h, nil, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Kept != 1 {
		t.Fatalf("kept=%d, want 1", res.Kept)
	}
	if attempts != 2 {
		t.Fatalf("attempts=%d, want 2 (recovered via custom classifier)", attempts)
	}
}

// TestTransientRecoveryAttemptRowsProduceRetryDecision verifies that transient
// error recovery attempt rows are stamped with Decision == "RETRY" rather than
// "REVERT", ensuring downstream meta-RSI consumers filter them out correctly.
func TestTransientRecoveryAttemptRowsProduceRetryDecision(t *testing.T) {
	calls := 0
	h := Harness{
		MetricName:      "p50",
		LowerBetter:     true,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 10.0, "sha-base", nil
		},
		TransientMeasurementRecoveryLimit: 2,
		Candidates: func() []Candidate {
			return []Candidate{{Label: "c1"}}
		},
		Measure: func(c Candidate) (Measurement, error) {
			calls++
			if calls == 1 {
				return Measurement{}, NewTransientMeasureError(errors.New("connection reset"))
			}
			return Measurement{Metric: 5.0, SuiteGreen: true, TruthClean: true}, nil
		},
	}

	res, err := Run(h, nil, 2, 0)
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Rows) != 2 {
		t.Fatalf("len(res.Rows)=%d, want 2 (1 transient retry attempt + 1 kept result)", len(res.Rows))
	}

	// Attempt 1: transient error recovery row must have Decision == "RETRY".
	attemptRow := res.Rows[0]
	if attemptRow.Decision != "RETRY" {
		t.Fatalf("attempt row decision=%q, want %q", attemptRow.Decision, "RETRY")
	}
	if attemptRow.Measured {
		t.Fatalf("attempt row measured=%v, want false", attemptRow.Measured)
	}
	if attemptRow.Kept {
		t.Fatalf("attempt row kept=%v, want false", attemptRow.Kept)
	}

	// Attempt 2: final kept row.
	finalRow := res.Rows[1]
	if finalRow.Decision != "KEEP" {
		t.Fatalf("final row decision=%q, want %q", finalRow.Decision, "KEEP")
	}
}

// TestTransientMeasureErrorExhaustionProducesRetryAndRevertRows proves that when
// transient measurement errors recur until retry attempts are exhausted:
//  1. Exactly N RETRY rows and 1 terminal REVERT/ESCALATE row are recorded in the
//     journal and persisted to disk.
//  2. Downstream metarsi integration (Fold and KeepRateTruthClean) correctly ignores
//     all intermediate RETRY rows and only evaluates terminal decision rows.
func TestTransientMeasureErrorExhaustionProducesRetryAndRevertRows(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "transient_exhaustion.jsonl")
	j, err := NewJournal(journalPath)
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}

	const retryLimit = 3
	calls := make(map[string]int)

	h := Harness{
		MetricName:      "p50",
		LowerBetter:     true,
		BaselineRefName: "main",
		BaselineMetric: func() (float64, string, error) {
			return 10.0, "sha-base", nil
		},
		TransientMeasurementRecoveryLimit: retryLimit,
		Candidates: func() []Candidate {
			return []Candidate{
				{Label: "c1"},
				{Label: "c2"},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			calls[c.Label]++
			return Measurement{}, NewTransientMeasureError(fmt.Errorf("transient error on %s attempt %d", c.Label, calls[c.Label]))
		},
	}

	// Threshold k=2:
	// c1 exhausts its retryLimit (3 retries) -> terminal row is REVERT (breaker count advances to 1).
	// c2 exhausts its retryLimit (3 retries) -> terminal row is ESCALATE (breaker count advances to 2 == k).
	res, err := Run(h, j, 2, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if err := j.Close(); err != nil {
		t.Fatalf("Journal.Close: %v", err)
	}

	// 1. Assert in-memory results: exactly N retry rows and 1 final row per candidate.
	const wantRowsPerCandidate = retryLimit + 1 // 3 RETRY + 1 terminal
	const wantTotalRows = wantRowsPerCandidate * 2
	if len(res.Rows) != wantTotalRows {
		t.Fatalf("len(res.Rows) = %d, want %d", len(res.Rows), wantTotalRows)
	}

	// 2. Assert journal persistence roundtrip via ReadJournal.
	persistedRows, err := ReadJournal(journalPath)
	if err != nil {
		t.Fatalf("ReadJournal: %v", err)
	}
	if len(persistedRows) != wantTotalRows {
		t.Fatalf("ReadJournal returned %d rows, want %d", len(persistedRows), wantTotalRows)
	}

	for i, r := range persistedRows {
		memRow := res.Rows[i]
		if r.Decision != memRow.Decision || r.Candidate != memRow.Candidate || r.BreakerCount != memRow.BreakerCount {
			t.Fatalf("row %d mismatch between memory and journal: memory=%+v, journal=%+v", i, memRow, r)
		}
	}

	// Check Candidate 1: exactly N RETRY rows + 1 final REVERT row.
	c1Rows := persistedRows[:wantRowsPerCandidate]
	for idx := 0; idx < retryLimit; idx++ {
		r := c1Rows[idx]
		if r.Decision != "RETRY" {
			t.Errorf("c1 attempt %d: decision = %q, want %q", idx+1, r.Decision, "RETRY")
		}
		if r.Mode != "improve" {
			t.Errorf("c1 attempt %d: mode = %q, want %q", idx+1, r.Mode, "improve")
		}
		if r.Measured {
			t.Errorf("c1 attempt %d: measured = true, want false", idx+1)
		}
		if r.Kept {
			t.Errorf("c1 attempt %d: kept = true, want false", idx+1)
		}
		if r.BreakerCount != 0 {
			t.Errorf("c1 attempt %d: breaker count = %d, want 0", idx+1, r.BreakerCount)
		}
		if !strings.Contains(r.Note, "transient, recovering") {
			t.Errorf("c1 attempt %d: note %q should mention transient, recovering", idx+1, r.Note)
		}
	}
	c1Final := c1Rows[retryLimit]
	if c1Final.Decision != "REVERT" {
		t.Fatalf("c1 final row decision = %q, want %q", c1Final.Decision, "REVERT")
	}
	if c1Final.Measured {
		t.Errorf("c1 final row: measured = true, want false")
	}
	if c1Final.Kept {
		t.Errorf("c1 final row: kept = true, want false")
	}
	if c1Final.BreakerCount != 1 {
		t.Fatalf("c1 final row breaker count = %d, want 1", c1Final.BreakerCount)
	}
	if !strings.Contains(c1Final.Note, "transient, exhausted") {
		t.Errorf("c1 final row note %q should mention transient, exhausted", c1Final.Note)
	}

	// Check Candidate 2: exactly N RETRY rows + 1 final ESCALATE row.
	c2Rows := persistedRows[wantRowsPerCandidate:]
	for idx := 0; idx < retryLimit; idx++ {
		r := c2Rows[idx]
		if r.Decision != "RETRY" {
			t.Errorf("c2 attempt %d: decision = %q, want %q", idx+1, r.Decision, "RETRY")
		}
		if r.Mode != "improve" {
			t.Errorf("c2 attempt %d: mode = %q, want %q", idx+1, r.Mode, "improve")
		}
		if r.Measured {
			t.Errorf("c2 attempt %d: measured = true, want false", idx+1)
		}
		if r.Kept {
			t.Errorf("c2 attempt %d: kept = true, want false", idx+1)
		}
		if r.BreakerCount != 1 {
			t.Errorf("c2 attempt %d: breaker count = %d, want 1", idx+1, r.BreakerCount)
		}
		if !strings.Contains(r.Note, "transient, recovering") {
			t.Errorf("c2 attempt %d: note %q should mention transient, recovering", idx+1, r.Note)
		}
	}
	c2Final := c2Rows[retryLimit]
	if c2Final.Decision != "ESCALATE" {
		t.Fatalf("c2 final row decision = %q, want %q", c2Final.Decision, "ESCALATE")
	}
	if c2Final.BreakerCount != 2 {
		t.Fatalf("c2 final row breaker count = %d, want 2", c2Final.BreakerCount)
	}
	if !strings.Contains(c2Final.Note, "transient, exhausted") {
		t.Errorf("c2 final row note %q should mention transient, exhausted", c2Final.Note)
	}

	// 3. Downstream metarsi integration: KeepRateTruthClean must ignore all
	// intermediate RETRY rows and only evaluate terminal decision rows.
	// c1 alone: exactly 1 evaluated cycle (terminal REVERT), rate 0.0.
	if rate := KeepRateTruthClean(c1Rows); rate != 0.0 {
		t.Fatalf("KeepRateTruthClean(c1Rows) = %v, want 0.0", rate)
	}

	// If we append one truth-clean keep row, the keep rate must be 1/2 = 0.50.
	// If RETRY rows were mistakenly counted in the denominator, rate would be 1/(3+1+1) = 0.20.
	c1WithKeep := append([]Row{}, c1Rows...)
	c1WithKeep = append(c1WithKeep, Row{Mode: "improve", Decision: "KEEP", Kept: true, TruthClean: true, SuiteGreen: true})
	if rate := KeepRateTruthClean(c1WithKeep); rate != 0.50 {
		t.Fatalf("KeepRateTruthClean(c1WithKeep) = %v, want 0.50 (1 keep out of 2 evaluated cycles)", rate)
	}

	// All persisted rows (c1 REVERT + c2 ESCALATE, plus intermediate RETRY rows):
	// Total cycles evaluated must be 2, clean = 0 -> rate = 0.0.
	if rate := KeepRateTruthClean(persistedRows); rate != 0.0 {
		t.Fatalf("KeepRateTruthClean(persistedRows) = %v, want 0.0", rate)
	}

	// With a keep appended to all persisted rows:
	// Total cycles evaluated must be 3 (c1 REVERT, c2 ESCALATE, and KEEP), clean = 1 -> rate = 1/3.
	allWithKeep := append([]Row{}, persistedRows...)
	allWithKeep = append(allWithKeep, Row{Mode: "improve", Decision: "KEEP", Kept: true, TruthClean: true, SuiteGreen: true})
	wantThird := 1.0 / 3.0
	if rate := KeepRateTruthClean(allWithKeep); rate != wantThird {
		t.Fatalf("KeepRateTruthClean(allWithKeep) = %v, want %v (1 keep out of 3 evaluated cycles)", rate, wantThird)
	}

	// 4. Downstream metarsi integration: metarsi.Fold must ignore all intermediate
	// RETRY rows and only count terminal decision rows within its window.
	cur := KeepPolicy{GainThreshold: 0.10, BreakerK: 2, Throttle: 4}
	cfg := MetaConfig{Window: 2, MinEscalations: 1, GainStep: 0.05, GainCeiling: 0.5}

	// In persistedRows:
	// Walking backwards:
	// - c2 terminal ESCALATE: seen = 1, esc = 1
	// - c2 RETRY rows (3): skipped! seen remains 1
	// - c1 terminal REVERT: seen = 2
	// Window of 2 is satisfied by the two terminal rows; all 6 intermediate RETRY rows skipped.
	p, ok := Fold(persistedRows, cur, cfg)
	if !ok {
		t.Fatalf("Fold(persistedRows) returned ok = false; RETRY rows should have been ignored")
	}
	if p.Escalations != 1 {
		t.Errorf("proposal escalations = %d, want 1", p.Escalations)
	}
	if p.Window != 2 {
		t.Errorf("proposal window = %d, want 2", p.Window)
	}

	// Furthermore, verify an older ESCALATE separated from the terminal REVERT by intermediate RETRY rows:
	// [ESCALATE, c1 RETRY 1, c1 RETRY 2, c1 RETRY 3, c1 REVERT]
	// If RETRY rows were not ignored, a window of 2 walking backward from REVERT would stop on RETRY
	// and never see the ESCALATE. With RETRY skipped, ESCALATE is reached at seen = 2.
	separatedRows := []Row{
		{Mode: "improve", Decision: "ESCALATE", Kept: false, TruthClean: true, SuiteGreen: false},
	}
	separatedRows = append(separatedRows, c1Rows...)
	p2, ok2 := Fold(separatedRows, cur, cfg)
	if !ok2 {
		t.Fatalf("Fold(separatedRows) returned ok = false; intermediate RETRY rows must not block reaching earlier cycles")
	}
	if p2.Escalations != 1 {
		t.Errorf("separated proposal escalations = %d, want 1", p2.Escalations)
	}
	if p2.Window != 2 {
		t.Errorf("separated proposal window = %d, want 2", p2.Window)
	}
}

// TestRunObserved_TransientRetryBackoffAndObserverSuppression verifies issue #11692:
//  1. Exponential backoff pacing is applied between transient retries (doubling from
//     base up to the cap).
//  2. Intermediate RETRY rows are suppressed from Observer dispatch; obs is invoked
//     only for terminal cycle decisions so external observers do not receive duplicate
//     cycle IDs or unfinalized decisions.
//  3. Backoff duration formula and context cancellation during backoff behave correctly.
func TestRunObserved_TransientRetryBackoffAndObserverSuppression(t *testing.T) {
	t.Run("HookedBackoffAndObserverSuppression", func(t *testing.T) {
		var (
			sleeps       []time.Duration
			observedRows []Row
			measureCalls = make(map[string]int)
		)

		h := Harness{
			MetricName:      "latency_p50",
			LowerBetter:     true,
			BaselineRefName: "main",
			BaselineMetric: func() (float64, string, error) {
				return 100.0, "sha-base", nil
			},
			TransientMeasurementRecoveryLimit: 2,
			TransientRetrySleep: func(d time.Duration) {
				sleeps = append(sleeps, d)
			},
			Candidates: func() []Candidate {
				return []Candidate{
					{Label: "c1"},
					{Label: "c2"},
				}
			},
			Measure: func(c Candidate) (Measurement, error) {
				measureCalls[c.Label]++
				call := measureCalls[c.Label]
				switch c.Label {
				case "c1":
					// attempt 0: transient fail
					// attempt 1: transient fail
					// attempt 2: succeed and keep
					if call == 1 {
						return Measurement{}, NewTransientMeasureError(errors.New("lock contention"))
					}
					if call == 2 {
						return Measurement{}, NewTransientMeasureError(errors.New("timeout transient"))
					}
					return Measurement{Metric: 50.0, SuiteGreen: true, TruthClean: true}, nil
				case "c2":
					// attempt 0: transient fail
					// attempt 1: transient fail
					// attempt 2: transient fail -> exhaustion -> REVERT
					return Measurement{}, NewTransientMeasureError(fmt.Errorf("exhaustion fail %d", call))
				default:
					return Measurement{}, errors.New("unknown")
				}
			},
		}

		obs := func(r Row) {
			observedRows = append(observedRows, r)
		}

		res, err := RunObserved(h, nil, 3, 0, obs)
		if err != nil {
			t.Fatalf("RunObserved: %v", err)
		}

		// (a) Verify exponential backoff pacing:
		// c1 fails attempt 0 (sleep 10ms), fails attempt 1 (sleep 20ms), succeeds on attempt 2 (no sleep).
		// c2 fails attempt 0 (sleep 10ms), fails attempt 1 (sleep 20ms), fails attempt 2 (exhausted -> no sleep).
		wantSleeps := []time.Duration{
			10 * time.Millisecond,
			20 * time.Millisecond,
			10 * time.Millisecond,
			20 * time.Millisecond,
		}
		if len(sleeps) != len(wantSleeps) {
			t.Fatalf("recorded sleeps count = %d, want %d: %v", len(sleeps), len(wantSleeps), sleeps)
		}
		for i, want := range wantSleeps {
			if sleeps[i] != want {
				t.Errorf("sleep[%d] = %v, want %v", i, sleeps[i], want)
			}
		}

		// (b) Verify observer suppression:
		// res.Rows has all attempt rows:
		// c1: 2 RETRY rows + 1 KEEP row = 3
		// c2: 2 RETRY rows + 1 REVERT row = 3
		// Total res.Rows = 6
		if len(res.Rows) != 6 {
			t.Fatalf("len(res.Rows) = %d, want 6", len(res.Rows))
		}

		// But observedRows must have ONLY terminal rows (exactly 2, one per cycle):
		if len(observedRows) != 2 {
			t.Fatalf("len(observedRows) = %d, want 2 (RETRY rows must be suppressed)", len(observedRows))
		}

		// Check row 0 (cycle 1 terminal decision):
		r1 := observedRows[0]
		if r1.Cycle != 1 {
			t.Errorf("observedRows[0].Cycle = %d, want 1", r1.Cycle)
		}
		if r1.Candidate != "c1" {
			t.Errorf("observedRows[0].Candidate = %q, want c1", r1.Candidate)
		}
		if r1.Decision != "KEEP" {
			t.Errorf("observedRows[0].Decision = %q, want KEEP", r1.Decision)
		}
		if !r1.Kept {
			t.Errorf("observedRows[0].Kept = false, want true")
		}

		// Check row 1 (cycle 2 terminal decision):
		r2 := observedRows[1]
		if r2.Cycle != 2 {
			t.Errorf("observedRows[1].Cycle = %d, want 2", r2.Cycle)
		}
		if r2.Candidate != "c2" {
			t.Errorf("observedRows[1].Candidate = %q, want c2", r2.Candidate)
		}
		if r2.Decision != "REVERT" {
			t.Errorf("observedRows[1].Decision = %q, want REVERT", r2.Decision)
		}
		if r2.Kept {
			t.Errorf("observedRows[1].Kept = true, want false")
		}

		// Confirm no duplicate cycle IDs or RETRY decisions were seen by the observer
		seenCycles := make(map[int]int)
		for _, r := range observedRows {
			if r.Decision == "RETRY" {
				t.Errorf("observer received intermediate RETRY row: %+v", r)
			}
			seenCycles[r.Cycle]++
			if seenCycles[r.Cycle] > 1 {
				t.Errorf("observer received duplicate cycle ID %d", r.Cycle)
			}
		}
	})

	t.Run("CustomBackoffDurationOverride", func(t *testing.T) {
		var sleeps []time.Duration
		h := Harness{
			MetricName:                        "p50",
			LowerBetter:                       true,
			BaselineRefName:                   "main",
			BaselineMetric:                    func() (float64, string, error) { return 10.0, "sha", nil },
			TransientMeasurementRecoveryLimit: 2,
			TransientRetryBackoff: func(attempt int) time.Duration {
				return time.Duration(100*(attempt+1)) * time.Millisecond
			},
			TransientRetrySleep: func(d time.Duration) {
				sleeps = append(sleeps, d)
			},
			Candidates: func() []Candidate { return []Candidate{{Label: "custom"}} },
			Measure: func(c Candidate) (Measurement, error) {
				if len(sleeps) < 2 {
					return Measurement{}, NewTransientMeasureError(errors.New("retry me"))
				}
				return Measurement{Metric: 5.0, SuiteGreen: true, TruthClean: true}, nil
			},
		}

		_, err := RunObserved(h, nil, 2, 0, nil)
		if err != nil {
			t.Fatalf("RunObserved: %v", err)
		}
		if len(sleeps) != 2 {
			t.Fatalf("len(sleeps) = %d, want 2", len(sleeps))
		}
		if sleeps[0] != 100*time.Millisecond || sleeps[1] != 200*time.Millisecond {
			t.Errorf("sleeps = %v, want [100ms, 200ms]", sleeps)
		}
	})

	t.Run("ContextCancellationAbortsDuringBackoff", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		h := Harness{
			MetricName:                        "p50",
			LowerBetter:                       true,
			BaselineRefName:                   "main",
			BaselineMetric:                    func() (float64, string, error) { return 10.0, "sha", nil },
			TransientMeasurementRecoveryLimit: 3,
			Context:                           ctx,
			TransientRetrySleep: func(d time.Duration) {
				cancel()
			},
			Candidates: func() []Candidate { return []Candidate{{Label: "cancel_cand"}} },
			Measure: func(c Candidate) (Measurement, error) {
				return Measurement{}, NewTransientMeasureError(errors.New("transient lock"))
			},
		}

		_, err := RunObserved(h, nil, 2, 0, nil)
		if err == nil {
			t.Fatal("expected error on cancelled context, got nil")
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("got error %v, want errors.Is(..., context.Canceled)", err)
		}
	})

	t.Run("DefaultTransientRetryBackoffCapAndDoubling", func(t *testing.T) {
		cases := []struct {
			attempt int
			want    time.Duration
		}{
			{-1, 10 * time.Millisecond},
			{0, 10 * time.Millisecond},
			{1, 20 * time.Millisecond},
			{2, 40 * time.Millisecond},
			{3, 80 * time.Millisecond},
			{4, 160 * time.Millisecond},
			{5, 320 * time.Millisecond},
			{6, 500 * time.Millisecond}, // capped
			{7, 500 * time.Millisecond},
			{15, 500 * time.Millisecond},
		}
		for _, tc := range cases {
			got := DefaultTransientRetryBackoff(tc.attempt)
			if got != tc.want {
				t.Errorf("DefaultTransientRetryBackoff(%d) = %v, want %v", tc.attempt, got, tc.want)
			}
		}
	})
}
