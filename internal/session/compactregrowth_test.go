package session

// compactregrowth_test.go — failure-class proof for the #4768 regrowth decomposition.
//
// Each fixture pins one class from the issue's Definition of Done: steady useful
// growth, duplicated setup reinjection, repeated tool output, one oversized result,
// timestamp ambiguity, a censored no-rebound window, and multiple compaction cycles.
// The aggregate test proves the corpus roll-up reproduces the rebound counts, the
// wall-clock exclusion of suspect windows, and the fast/slow cohort split.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func scanRegrowthFixture(t *testing.T, name string) CompactSessionReport {
	t.Helper()
	p := filepath.Join("testdata", "compactregrowth", name)
	f, err := os.Open(p)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		t.Fatalf("stat fixture: %v", err)
	}
	rep, err := ScanCompactRollout(f, p, fi.Size())
	if err != nil {
		t.Fatalf("scan %s: %v", name, err)
	}
	return rep
}

func regrowthOf(t *testing.T, rep CompactSessionReport, fire int) *CompactRegrowth {
	t.Helper()
	if len(rep.Fires) <= fire {
		t.Fatalf("fires = %d, want > %d", len(rep.Fires), fire)
	}
	r := rep.Fires[fire].Regrowth
	if r == nil {
		t.Fatalf("fire %d has no regrowth trajectory", fire)
	}
	return r
}

func crossingAt(r *CompactRegrowth, threshold int) *RegrowthCrossing {
	for i := range r.Crossings {
		if r.Crossings[i].Threshold == threshold {
			return &r.Crossings[i]
		}
	}
	return nil
}

// Steady useful growth: the window refills through distinct tool work. The trajectory
// must time every milestone, join the cache telemetry, attribute the bytes per tool,
// and raise NO anomalies — useful work is not waste.
func TestRegrowthSteadyGrowthTrajectory(t *testing.T) {
	r := regrowthOf(t, scanRegrowthFixture(t, "steady-growth.jsonl"), 0)

	if !r.Rebounded || r.Censored != "" {
		t.Fatalf("rebounded/censored = %v/%q, want true/empty", r.Rebounded, r.Censored)
	}
	if r.PostFloorTokens != 25000 {
		t.Errorf("post floor = %d, want 25000", r.PostFloorTokens)
	}
	if r.GrowthTokens != 205000-25000 {
		t.Errorf("growth = %d, want %d", r.GrowthTokens, 205000-25000)
	}
	if r.Samples != 9 {
		t.Errorf("samples = %d, want 9", r.Samples)
	}
	if len(r.Crossings) != len(RegrowthThresholds) {
		t.Fatalf("crossings = %d, want %d (50k/100k/150k/200k)", len(r.Crossings), len(RegrowthThresholds))
	}
	c := crossingAt(r, RegrowthReboundTokens)
	if c == nil || c.Seconds != 990 {
		t.Errorf("200k crossing = %+v, want 990 s", c)
	}
	if c != nil && (c.Samples != 9 || c.Turns != 8 || c.ToolCalls != 8) {
		t.Errorf("200k crossing samples/turns/tool-calls = %d/%d/%d, want 9/8/8", c.Samples, c.Turns, c.ToolCalls)
	}
	if c50 := crossingAt(r, 50000); c50 == nil || c50.Seconds >= c.Seconds {
		t.Errorf("50k crossing = %+v, want earlier than the 200k crossing", c50)
	}
	if r.TokensPerSecond <= 0 || r.TokensPerSample <= 0 {
		t.Errorf("velocities = %.1f tok/s, %.1f tok/sample, want both > 0", r.TokensPerSecond, r.TokensPerSample)
	}
	if r.CacheReadFraction < 0.5 || r.CacheReadFraction >= 1 {
		t.Errorf("cache-read fraction = %.4f, want in [0.5, 1) — the join must price regrowth net of reuse", r.CacheReadFraction)
	}
	for _, class := range []string{RegrowClassToolCallPrefix + "shell", RegrowClassToolResPrefix + "shell"} {
		st := r.Classes[class]
		if st == nil || st.Rows != 8 {
			t.Errorf("class %s = %+v, want 8 rows", class, st)
		}
	}
	if st := r.Classes[RegrowClassCompactSummary]; st == nil || st.Rows != 1 {
		t.Errorf("class %s = %+v, want the compacted row attributed once", RegrowClassCompactSummary, st)
	}
	if len(r.Anomalies) != 0 {
		t.Errorf("anomalies = %v, want none — steady useful growth must not be flagged", r.Anomalies)
	}
	if r.Confidence != CompactConfidenceHigh || r.Reason != CompactReasonOK {
		t.Errorf("confidence = %s/%s, want %s/%s", r.Confidence, r.Reason, CompactConfidenceHigh, CompactReasonOK)
	}
}

// A byte-identical instruction payload reinjected after the fire is cross-fire
// restatement: flagged, measured by length, attributed to the instructions class.
// The flagged row is the window's ONLY instructions row — the pre-fire copy was
// discarded by the cut — so the anomaly is a rollout-level restatement counter,
// never a license to drop the reinjection (#5255).
func TestRegrowthDuplicatedSetupFlagged(t *testing.T) {
	r := regrowthOf(t, scanRegrowthFixture(t, "duplicated-setup.jsonl"), 0)

	if !hasRegrowAnomaly(r, AnomalyDuplicateSetup) {
		t.Errorf("anomalies = %v, want %s", r.Anomalies, AnomalyDuplicateSetup)
	}
	st := r.Classes[RegrowClassInstructions]
	if st == nil || st.DupRows != 1 || st.DupBytes < RegrowthDupMinBytes {
		t.Errorf("instructions class = %+v, want 1 duplicate row of >= %d bytes", st, RegrowthDupMinBytes)
	}
	if st != nil && st.Rows != 1 {
		t.Errorf("instructions rows = %d, want 1 — the flagged duplicate must be the sole in-window copy (cross-fire restatement, not double residency)", st.Rows)
	}
	if r.Rebounded || r.Censored != RegrowthCensorRolloutEnd {
		t.Errorf("rebounded/censored = %v/%q, want false/%s", r.Rebounded, r.Censored, RegrowthCensorRolloutEnd)
	}
}

// The same tool output returned under DIFFERENT call ids must still read as a repeated
// span — dedup keys on content, not on the enveloping call id.
func TestRegrowthRepeatedToolOutputFlagged(t *testing.T) {
	r := regrowthOf(t, scanRegrowthFixture(t, "repeated-tool-output.jsonl"), 0)

	if !hasRegrowAnomaly(r, AnomalyRepeatedToolResult) {
		t.Errorf("anomalies = %v, want %s", r.Anomalies, AnomalyRepeatedToolResult)
	}
	st := r.Classes[RegrowClassToolResPrefix+"shell"]
	if st == nil || st.Rows != 3 || st.DupRows != 2 {
		t.Errorf("tool_result/shell = %+v, want 3 rows with 2 duplicates (distinct call ids, same content)", st)
	}
	if hasRegrowAnomaly(r, AnomalyDuplicateSetup) {
		t.Errorf("anomalies = %v — repeated tool output must not read as setup reinjection", r.Anomalies)
	}
}

// One ~400 KB result right after the fire: over the head bound AND the oversized bar.
// The row must be measured at full length, classified by tool via its call id, flagged
// oversized, and its body must never surface.
func TestRegrowthOversizedSingleEventFlagged(t *testing.T) {
	rep := scanRegrowthFixture(t, "oversized-result.jsonl")
	r := regrowthOf(t, rep, 0)

	if !hasRegrowAnomaly(r, AnomalyOversizedEvent) {
		t.Errorf("anomalies = %v, want %s", r.Anomalies, AnomalyOversizedEvent)
	}
	if !hasRegrowAnomaly(r, AnomalySuffixRecreation) {
		t.Errorf("anomalies = %v, want %s too — a giant single result IS an early burst", r.Anomalies, AnomalySuffixRecreation)
	}
	st := r.Classes[RegrowClassToolResPrefix+"shell"]
	if st == nil || st.Bytes < RegrowthOversizedRowBytes {
		t.Errorf("tool_result/shell = %+v, want >= %d bytes — full row length, not the truncated head", st, RegrowthOversizedRowBytes)
	}
	blob, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(blob), "MUST_NOT_LEAK_REGROWTH") {
		t.Error("oversized body leaked into the report — attribution must stay body-blind")
	}
}

// A rebound whose samples all share the fire's timestamp is impossible wall-clock
// velocity: typed TIMESTAMP_SUSPECT at low confidence, counted as a rebound, and
// excluded from every wall-clock statistic.
func TestRegrowthTimestampAmbiguityTypedAndExcluded(t *testing.T) {
	rep := scanRegrowthFixture(t, "timestamp-ambiguity.jsonl")
	r := regrowthOf(t, rep, 0)

	if !r.Rebounded {
		t.Fatal("fixture must rebound to >=200k")
	}
	if !hasRegrowAnomaly(r, AnomalyTimestampSuspect) {
		t.Errorf("anomalies = %v, want %s", r.Anomalies, AnomalyTimestampSuspect)
	}
	if r.Confidence != CompactConfidenceLow || r.Reason != AnomalyTimestampSuspect {
		t.Errorf("confidence = %s/%s, want %s/%s", r.Confidence, r.Reason, CompactConfidenceLow, AnomalyTimestampSuspect)
	}
	if r.TokensPerSecond != 0 {
		t.Errorf("tokens/s = %.1f, want 0 — no wall-clock velocity on a suspect clock", r.TokensPerSecond)
	}

	agg := AggregateCompactReports([]CompactSessionReport{rep}).Regrowth
	if agg == nil {
		t.Fatal("aggregate regrowth missing")
	}
	if agg.Rebounds != 1 || agg.TimestampSuspect != 1 {
		t.Errorf("rebounds/suspect = %d/%d, want 1/1", agg.Rebounds, agg.TimestampSuspect)
	}
	if agg.MedianSecondsToRebound != 0 || agg.ReboundsWithin30Min != 0 || agg.ReboundsWithin15Min != 0 {
		t.Errorf("wall-clock stats = median %.0f, 15m %d, 30m %d — suspect windows must be excluded",
			agg.MedianSecondsToRebound, agg.ReboundsWithin15Min, agg.ReboundsWithin30Min)
	}
	if agg.MedianSamplesToRebound != 4 {
		t.Errorf("median samples = %d, want 4 — sample counts stay valid on a broken clock", agg.MedianSamplesToRebound)
	}
}

// A session that ends below the bar is CENSORED — an observation limit, not a verdict.
func TestRegrowthCensoredNoRebound(t *testing.T) {
	r := regrowthOf(t, scanRegrowthFixture(t, "censored-no-rebound.jsonl"), 0)

	if r.Rebounded {
		t.Fatal("fixture must not rebound")
	}
	if r.Censored != RegrowthCensorRolloutEnd {
		t.Errorf("censored = %q, want %s", r.Censored, RegrowthCensorRolloutEnd)
	}
	if len(r.Crossings) != 1 || r.Crossings[0].Threshold != 50000 {
		t.Errorf("crossings = %+v, want just the 50k milestone", r.Crossings)
	}
	if r.TokensPerSecond <= 0 {
		t.Errorf("tokens/s = %.1f, want > 0 — the partial trajectory still has a clean clock", r.TokensPerSecond)
	}
	if len(r.Anomalies) != 0 {
		t.Errorf("anomalies = %v, want none", r.Anomalies)
	}
}

// Multiple compaction cycles: the first window is censored BY THE NEXT FIRE with the
// fire gap recorded; the second window rebounds on its own clock.
func TestRegrowthMultiCycleWindows(t *testing.T) {
	rep := scanRegrowthFixture(t, "multi-cycle.jsonl")
	if rep.FireCount != 2 {
		t.Fatalf("fires = %d, want 2", rep.FireCount)
	}
	w0 := regrowthOf(t, rep, 0)
	if w0.Rebounded || w0.Censored != RegrowthCensorNextFire {
		t.Errorf("window0 rebounded/censored = %v/%q, want false/%s", w0.Rebounded, w0.Censored, RegrowthCensorNextFire)
	}
	if w0.NextFireSeconds != 900 {
		t.Errorf("window0 next-fire = %.0f s, want 900", w0.NextFireSeconds)
	}
	w1 := regrowthOf(t, rep, 1)
	if !w1.Rebounded || w1.Censored != "" {
		t.Errorf("window1 rebounded/censored = %v/%q, want true/empty", w1.Rebounded, w1.Censored)
	}
	if c := crossingAt(w1, RegrowthReboundTokens); c == nil || c.Seconds != 630 {
		t.Errorf("window1 200k crossing = %+v, want 630 s (measured from ITS fire, not the first)", c)
	}
}

// The corpus roll-up: rebound counts, censor counts, the ranked attribution input, the
// anomaly tally, and the fast/slow cohort comparison, across all seven fixtures.
func TestRegrowthAggregateAcrossFixtures(t *testing.T) {
	res, err := AuditCompactCorpus(CompactAuditOptions{Root: filepath.Join("testdata", "compactregrowth")})
	if err != nil {
		t.Fatalf("audit corpus: %v", err)
	}
	agg := res.Aggregate.Regrowth
	if agg == nil {
		t.Fatal("aggregate regrowth missing")
	}
	if agg.FiresWithTelemetry != 8 {
		t.Errorf("fires with telemetry = %d, want 8 (7 fixtures, one with two fires)", agg.FiresWithTelemetry)
	}
	if agg.Rebounds != 3 || agg.Censored != 5 {
		t.Errorf("rebounds/censored = %d/%d, want 3/5", agg.Rebounds, agg.Censored)
	}
	if agg.TimestampSuspect != 1 {
		t.Errorf("timestamp suspect = %d, want 1", agg.TimestampSuspect)
	}
	// Clean-clock rebounds: 990 s (steady) and 630 s (multi-cycle window 2).
	if agg.MedianSecondsToRebound != 810 {
		t.Errorf("median seconds to rebound = %.0f, want 810 (suspect window excluded)", agg.MedianSecondsToRebound)
	}
	if agg.ReboundsWithin30Min != 2 || agg.ReboundsWithin15Min != 1 {
		t.Errorf("within 30m/15m = %d/%d, want 2/1", agg.ReboundsWithin30Min, agg.ReboundsWithin15Min)
	}
	if agg.MedianNextFireSeconds != 900 {
		t.Errorf("median next-fire = %.0f s, want 900", agg.MedianNextFireSeconds)
	}
	for _, a := range []string{
		AnomalyDuplicateSetup, AnomalyRepeatedToolResult,
		AnomalyOversizedEvent, AnomalySuffixRecreation, AnomalyTimestampSuspect,
	} {
		if agg.AnomalyCounts[a] == 0 {
			t.Errorf("anomaly counts = %v, want at least one %s", agg.AnomalyCounts, a)
		}
	}
	if st := agg.ClassTotals[RegrowClassToolResPrefix+"shell"]; st == nil || st.Bytes == 0 {
		t.Errorf("class totals missing tool_result/shell: %v", agg.ClassTotals)
	}
	if agg.Fast.Windows != 2 {
		t.Errorf("fast cohort windows = %d, want 2", agg.Fast.Windows)
	}
	if agg.Slow.Windows != 6 {
		t.Errorf("slow cohort windows = %d, want 6 (5 censored + 1 suspect rebound)", agg.Slow.Windows)
	}
	if agg.Fast.MedianToolCalls <= agg.Slow.MedianToolCalls {
		t.Errorf("fast/slow median tool calls = %d/%d — the steady fixture's useful work should dominate the fast cohort",
			agg.Fast.MedianToolCalls, agg.Slow.MedianToolCalls)
	}
}

// No fixture body may surface anywhere in any report — lengths and hashes only.
func TestRegrowthNeverEmitsBodies(t *testing.T) {
	for _, name := range []string{
		"steady-growth.jsonl", "duplicated-setup.jsonl", "repeated-tool-output.jsonl",
		"oversized-result.jsonl", "timestamp-ambiguity.jsonl",
		"censored-no-rebound.jsonl", "multi-cycle.jsonl",
	} {
		rep := scanRegrowthFixture(t, name)
		blob, err := json.Marshal(rep)
		if err != nil {
			t.Fatalf("marshal %s: %v", name, err)
		}
		for _, sentinel := range []string{"MUST_NOT_LEAK_REGROWTH", "invariant-setup-line", "repeated-bytes"} {
			if strings.Contains(string(blob), sentinel) {
				t.Errorf("%s: report leaked %q", name, sentinel)
			}
		}
	}
}

// The human report gains a regrowth section: the rebound headline, the ranked
// attribution table, and the fast/slow comparison.
func TestRenderCompactAuditIncludesRegrowthSection(t *testing.T) {
	res, err := AuditCompactCorpus(CompactAuditOptions{Root: filepath.Join("testdata", "compactregrowth")})
	if err != nil {
		t.Fatalf("audit corpus: %v", err)
	}
	var sb strings.Builder
	RenderCompactAudit(&sb, res, 3)
	out := sb.String()
	for _, want := range []string{
		"regrowth after compaction",
		"rebounded to >=200k resident",
		"regrowth attribution",
		"tool_result/shell",
		"timestamp-suspect excluded",
		"fast (<=30 min rebound)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q\n---\n%s", want, out)
		}
	}
	if strings.Contains(out, "MUST_NOT_LEAK_REGROWTH") {
		t.Error("render leaked a fixture body")
	}
}
