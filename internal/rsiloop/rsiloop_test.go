package rsiloop

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

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

// TestTransientMeasurementErrorRecoveredWithinBound proves issue #11608:
// Two transient infrastructure measurement failures at threshold k=2 do NOT
// prematurely trip the regression breaker or abandon later viable candidates
// when bounded transient recovery is configured. Failed attempts are retained
// in the journal as unmeasured evidence (preserving failure auditability).
func TestTransientMeasurementErrorRecoveredWithinBound(t *testing.T) {
	candidateAttempts := make(map[string]int)
	steps := []fakeStep{
		{label: "c1", metric: 2.0, green: true, clean: true},
		{label: "c2", metric: 3.0, green: true, clean: true},
		{label: "c3", metric: 4.0, green: true, clean: true},
	}

	h := Harness{
		MetricName:          "m",
		LowerBetter:         false,
		BaselineRefName:     "test-ref",
		MaxTransientRetries: 2,
		BaselineMetric: func() (float64, string, error) {
			return 1.0, "sha001", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{Label: "c1", Payload: 0},
				{Label: "c2", Payload: 1},
				{Label: "c3", Payload: 2},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			candidateAttempts[c.Label]++
			att := candidateAttempts[c.Label]
			switch c.Label {
			case "c1":
				// First attempt fails with ErrTransient; retry succeeds.
				if att == 1 {
					return Measurement{}, ErrTransient
				}
				return Measurement{Metric: steps[0].metric, SuiteGreen: steps[0].green, TruthClean: steps[0].clean}, nil
			case "c2":
				// First attempt fails with typed TransientError; retry succeeds.
				if att == 1 {
					return Measurement{}, NewTransientError(errors.New("timeout"))
				}
				return Measurement{Metric: steps[1].metric, SuiteGreen: steps[1].green, TruthClean: steps[1].clean}, nil
			case "c3":
				return Measurement{Metric: steps[2].metric, SuiteGreen: steps[2].green, TruthClean: steps[2].clean}, nil
			default:
				return Measurement{}, errors.New("unknown")
			}
		},
	}

	dir := t.TempDir()
	jPath := filepath.Join(dir, "journal.jsonl")
	j, err := NewJournal(jPath)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Run(h, j, 2, 0) // k = 2 breaker threshold
	_ = j.Close()
	if err != nil {
		t.Fatal(err)
	}

	if res.Escalated {
		t.Fatalf("loop escalated prematurely! final=%s", res.Final)
	}
	if res.Cycles != 3 {
		t.Fatalf("cycles=%d, want 3 (all candidates should be evaluated)", res.Cycles)
	}
	if res.Kept != 3 {
		t.Fatalf("kept=%d, want 3", res.Kept)
	}
	if res.FinalBaseline != 4.0 {
		t.Fatalf("final baseline=%.1f, want 4.0", res.FinalBaseline)
	}

	// Verify all 5 rows (2 failed attempts + 3 successful evaluations) in res.Rows
	// and in the persisted journal.
	if len(res.Rows) != 5 {
		t.Fatalf("total rows=%d, want 5 (2 transient RETRY + 3 KEEP)", len(res.Rows))
	}

	// Row 0: c1 transient retry (unmeasured, breaker count 0)
	r0 := res.Rows[0]
	if r0.Candidate != "c1" || r0.Measured || r0.Decision != "RETRY" || r0.BreakerCount != 0 {
		t.Fatalf("r0 mismatch: %+v", r0)
	}
	// Row 1: c1 kept (measured, breaker count 0)
	r1 := res.Rows[1]
	if r1.Candidate != "c1" || !r1.Measured || r1.Decision != "KEEP" || !r1.Kept {
		t.Fatalf("r1 mismatch: %+v", r1)
	}
	// Row 2: c2 transient retry (unmeasured, breaker count 0)
	r2 := res.Rows[2]
	if r2.Candidate != "c2" || r2.Measured || r2.Decision != "RETRY" || r2.BreakerCount != 0 {
		t.Fatalf("r2 mismatch: %+v", r2)
	}
	// Row 3: c2 kept (measured, breaker count 0)
	r3 := res.Rows[3]
	if r3.Candidate != "c2" || !r3.Measured || r3.Decision != "KEEP" || !r3.Kept {
		t.Fatalf("r3 mismatch: %+v", r3)
	}
	// Row 4: c3 kept (measured, breaker count 0)
	r4 := res.Rows[4]
	if r4.Candidate != "c3" || !r4.Measured || r4.Decision != "KEEP" || !r4.Kept {
		t.Fatalf("r4 mismatch: %+v", r4)
	}

	// Verify journal persistence
	b, err := os.ReadFile(jPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := splitNonEmpty(string(b))
	if len(lines) != 5 {
		t.Fatalf("journal line count=%d, want 5", len(lines))
	}
}

// TestPermanentBuildFailureTripsRegressionBreaker proves that untyped errors and
// build failures are NOT retried and advance the breaker, tripping it at threshold k.
func TestPermanentBuildFailureTripsRegressionBreaker(t *testing.T) {
	steps := []fakeStep{
		{label: "bad1", err: errBoom},
		{label: "bad2", err: errors.New("syntax error: unexpected token")},
		{label: "viable", metric: 10.0, green: true, clean: true},
	}
	h := fakeHarness("m", false, 1.0, "sha", steps)
	h.MaxTransientRetries = 2 // retries configured, but must NOT apply to permanent errors

	res, err := Run(h, nil, 2, 0) // k = 2
	if err != nil {
		t.Fatal(err)
	}
	if !res.Escalated {
		t.Fatalf("expected escalation from permanent failures, got final=%s", res.Final)
	}
	if res.Cycles != 2 {
		t.Fatalf("cycles=%d, want 2 (loop must stop after 2 permanent non-keeps at k=2)", res.Cycles)
	}
	if len(res.Rows) != 2 {
		t.Fatalf("rows=%d, want 2 (permanent errors must not retry)", len(res.Rows))
	}
	if res.Rows[0].Decision != "REVERT" || res.Rows[0].BreakerCount != 1 {
		t.Fatalf("row 0 mismatch: %+v", res.Rows[0])
	}
	if res.Rows[1].Decision != "ESCALATE" || res.Rows[1].BreakerCount != 2 {
		t.Fatalf("row 1 mismatch: %+v", res.Rows[1])
	}
}

// TestTransientMeasurementErrorExceedingBoundTripsBreaker proves that transient
// recovery is bounded: persistent transient failures that exhaust all retries
// are recorded as unrecovered non-keeps that advance and trip the breaker.
func TestTransientMeasurementErrorExceedingBoundTripsBreaker(t *testing.T) {
	h := Harness{
		MetricName:          "m",
		LowerBetter:         false,
		BaselineRefName:     "test-ref",
		MaxTransientRetries: 1, // only 1 retry allowed (2 attempts total)
		BaselineMetric: func() (float64, string, error) {
			return 1.0, "sha001", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{Label: "c1", Payload: 0},
				{Label: "c2", Payload: 1},
				{Label: "c3", Payload: 2},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			return Measurement{}, ErrTransient // fails transiently on every attempt
		},
	}

	res, err := Run(h, nil, 2, 0) // k = 2
	if err != nil {
		t.Fatal(err)
	}
	if !res.Escalated {
		t.Fatalf("expected escalation after exhausted retries, got final=%s", res.Final)
	}
	if res.Cycles != 2 {
		t.Fatalf("cycles=%d, want 2 (stopped after c1 and c2 exhausted retries)", res.Cycles)
	}
	// For each candidate: 1 RETRY row (attempt 1) + 1 REVERT/ESCALATE row (attempt 2 exhausted) = 4 rows total
	if len(res.Rows) != 4 {
		t.Fatalf("rows=%d, want 4", len(res.Rows))
	}
	if res.Rows[0].Decision != "RETRY" || res.Rows[0].BreakerCount != 0 {
		t.Fatalf("row 0: %+v", res.Rows[0])
	}
	if res.Rows[1].Decision != "REVERT" || res.Rows[1].BreakerCount != 1 {
		t.Fatalf("row 1: %+v", res.Rows[1])
	}
	if res.Rows[2].Decision != "RETRY" || res.Rows[2].BreakerCount != 1 {
		t.Fatalf("row 2: %+v", res.Rows[2])
	}
	if res.Rows[3].Decision != "ESCALATE" || res.Rows[3].BreakerCount != 2 {
		t.Fatalf("row 3: %+v", res.Rows[3])
	}
}

type customTransientErr struct {
	transient bool
}

func (c customTransientErr) Error() string   { return "custom error" }
func (c customTransientErr) Transient() bool { return c.transient }

// TestIsTransientClassification proves all transient markers (ErrTransient,
// TransientError, interface{ Transient() bool }) are recognized while untyped
// errors are not.
func TestIsTransientClassification(t *testing.T) {
	if IsTransient(nil) {
		t.Fatal("nil should not be transient")
	}
	if !IsTransient(ErrTransient) {
		t.Fatal("ErrTransient must be transient")
	}
	if !IsTransient(fmt.Errorf("wrapped: %w", ErrTransient)) {
		t.Fatal("wrapped ErrTransient must be transient")
	}
	if !IsTransient(&TransientError{Err: errors.New("timeout")}) {
		t.Fatal("*TransientError must be transient")
	}
	if !IsTransient(TransientError{Err: errors.New("timeout")}) {
		t.Fatal("TransientError value must be transient")
	}
	if !IsTransient(NewTransientError(errors.New("conn reset"))) {
		t.Fatal("NewTransientError must be transient")
	}
	if !IsTransient(customTransientErr{transient: true}) {
		t.Fatal("customTransientErr{transient: true} must be transient")
	}
	if IsTransient(customTransientErr{transient: false}) {
		t.Fatal("customTransientErr{transient: false} must not be transient")
	}
	if IsTransient(errors.New("syntax error")) {
		t.Fatal("untyped errors.New must not be transient")
	}
	if IsTransient(errBoom) {
		t.Fatal("errBoom must not be transient")
	}
}

// TestTransientBudgetBoundsTotalRetriesAcrossRun proves TransientBudget limits the total
// transient retry attempts across the entire run.
func TestTransientBudgetBoundsTotalRetriesAcrossRun(t *testing.T) {
	h := Harness{
		MetricName:          "m",
		LowerBetter:         false,
		BaselineRefName:     "test-ref",
		MaxTransientRetries: 2,
		TransientBudget:     1, // only 1 retry for the entire run
		BaselineMetric: func() (float64, string, error) {
			return 1.0, "sha001", nil
		},
		Candidates: func() []Candidate {
			return []Candidate{
				{Label: "c1", Payload: 0},
				{Label: "c2", Payload: 1},
			}
		},
		Measure: func(c Candidate) (Measurement, error) {
			return Measurement{}, ErrTransient
		},
	}

	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	// c1 uses the 1 budget retry: row 0 (RETRY), row 1 (REVERT, retries exhausted).
	// c2 cannot retry (budget 0 left): row 2 (REVERT, no retries allowed).
	if len(res.Rows) != 3 {
		t.Fatalf("rows=%d, want 3", len(res.Rows))
	}
	if res.Rows[0].Decision != "RETRY" {
		t.Fatalf("row 0 should be RETRY: %+v", res.Rows[0])
	}
	if res.Rows[1].Decision != "REVERT" {
		t.Fatalf("row 1 should be REVERT: %+v", res.Rows[1])
	}
	if res.Rows[2].Decision != "REVERT" {
		t.Fatalf("row 2 should be REVERT: %+v", res.Rows[2])
	}
}
