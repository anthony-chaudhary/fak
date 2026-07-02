package turntaxmeter

// hooklat_test.go — acceptance witnesses for the #1993 guard-hook latency rollup:
// the stream parses tolerantly, the percentiles are the nearest-rank of what was
// actually observed, and the budget alarm fires as the closed GATE_LATENCY_REGRESSION
// token — but abstains (Thin) on the issue's own n=13-is-small caveat.

import (
	"strings"
	"testing"
	"time"
)

// sampleStream is byte-shaped like the live .dos/metrics/observations.jsonl rows the
// #1993 evidence came from (guard-audit run, 2026-07-01), plus the tolerance cases:
// a foreign-family row, a garbage line, and a hook row with no measured latency.
const sampleStream = `{"exit": 0, "latency_ms": 61.48, "op": "OBSERVE", "outcome": "passthrough", "schema": {"family": "hook-observation", "version": 1}, "stream_state": "ADVANCING", "ts": "2026-07-01T15:07:34Z", "verb": "posttool"}
{"dialect": "claude-code", "exit": 0, "holder": "06ac6f07", "latency_ms": 78.872, "op": "OBSERVE", "outcome": "passthrough", "rung": "provenance", "schema": {"family": "hook-observation", "version": 1}, "ts": "2026-07-01T15:07:45Z", "verb": "pretool"}
{"schema": {"family": "lease-heartbeat", "version": 2}, "latency_ms": 9999.0, "ts": "2026-07-01T15:08:00Z"}
not json at all
{"exit": 0, "op": "OBSERVE", "outcome": "passthrough", "schema": {"family": "hook-observation", "version": 1}, "ts": "2026-07-01T15:08:02Z", "verb": "pretool"}
{"exit": 0, "latency_ms": 175.5, "op": "OBSERVE", "outcome": "passthrough", "schema": {"family": "hook-observation", "version": 1}, "ts": "2026-07-01T15:08:09Z", "verb": "posttool"}
`

func TestParseHookObservationsTolerant(t *testing.T) {
	obs, skipped, err := ParseHookObservations(strings.NewReader(sampleStream))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(obs) != 3 {
		t.Fatalf("parsed %d observations, want 3 (foreign family, garbage, and latency-less rows skipped)", len(obs))
	}
	if skipped != 3 {
		t.Fatalf("skipped = %d, want 3", skipped)
	}
	if obs[0].Verb != "posttool" || obs[0].LatencyMS != 61.48 || obs[0].Outcome != "passthrough" {
		t.Fatalf("row 0 = %+v, want posttool/61.48/passthrough", obs[0])
	}
	wantAt, _ := time.Parse(time.RFC3339, "2026-07-01T15:07:45Z")
	if !obs[1].At.Equal(wantAt) {
		t.Fatalf("row 1 At = %v, want %v", obs[1].At, wantAt)
	}
}

// TestFoldHookLatencyNearestRank pins the percentile method on a fully known
// sample: 1..100ms one each, so pN must read back exactly N — a real observed
// value, never an interpolation.
func TestFoldHookLatencyNearestRank(t *testing.T) {
	var obs []HookObservation
	for i := 1; i <= 100; i++ {
		verb := "pretool"
		if i%2 == 0 {
			verb = "posttool"
		}
		obs = append(obs, HookObservation{Verb: verb, LatencyMS: float64(i)})
	}
	r := FoldHookLatency(obs)
	tot := r.Total
	if tot.Count != 100 || tot.P50MS != 50 || tot.P90MS != 90 || tot.P99MS != 99 || tot.MaxMS != 100 {
		t.Fatalf("total = %+v, want count=100 p50=50 p90=90 p99=99 max=100", tot)
	}
	if tot.MeanMS != 50.5 {
		t.Fatalf("mean = %v, want 50.5", tot.MeanMS)
	}
	if len(r.ByVerb) != 2 || r.ByVerb[0].Verb != "posttool" || r.ByVerb[1].Verb != "pretool" {
		t.Fatalf("byVerb = %+v, want [posttool pretool] (sorted)", r.ByVerb)
	}
	if r.ByVerb[0].Count != 50 || r.ByVerb[1].Count != 50 {
		t.Fatalf("byVerb counts = %d/%d, want 50/50", r.ByVerb[0].Count, r.ByVerb[1].Count)
	}
}

// TestFoldHookLatencyThinTailIsMax proves the honest small-n reading: with n=13
// the nearest-rank p99 IS the max — exactly the #1993 evidence table shape
// (p99 == max == 175.5 at n=13).
func TestFoldHookLatencyThinTailIsMax(t *testing.T) {
	obs := make([]HookObservation, 13)
	for i := range obs {
		obs[i] = HookObservation{Verb: "pretool", LatencyMS: 70 + float64(i)}
	}
	obs[12].LatencyMS = 175.5
	r := FoldHookLatency(obs)
	if r.Total.P99MS != r.Total.MaxMS || r.Total.MaxMS != 175.5 {
		t.Fatalf("n=13 p99 = %v max = %v, want both 175.5", r.Total.P99MS, r.Total.MaxMS)
	}
}

func TestFoldHookLatencyEmpty(t *testing.T) {
	r := FoldHookLatency(nil)
	if r.Total.Count != 0 || len(r.ByVerb) != 0 {
		t.Fatalf("empty fold = %+v, want zero rollup", r)
	}
}

func TestFilterHookObservationsSince(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	obs := []HookObservation{
		{Verb: "pretool", LatencyMS: 1, At: t0.Add(-time.Hour)}, // before window
		{Verb: "pretool", LatencyMS: 2, At: t0},                 // at cutoff — kept
		{Verb: "pretool", LatencyMS: 3, At: t0.Add(time.Hour)},  // inside
		{Verb: "pretool", LatencyMS: 4},                         // no ts — cannot witness the window
	}
	kept := FilterHookObservationsSince(obs, t0)
	if len(kept) != 2 || kept[0].LatencyMS != 2 || kept[1].LatencyMS != 3 {
		t.Fatalf("since-filter kept %+v, want the t0 and t0+1h rows only", kept)
	}
	if all := FilterHookObservationsSince(obs, time.Time{}); len(all) != 4 {
		t.Fatalf("zero cutoff kept %d, want all 4", len(all))
	}
}

// TestJudgeHookLatency covers the closed verdict space: thin abstains even when
// over budget, an accumulated over-budget tail fires the GATE_LATENCY_REGRESSION
// token, a within-budget tail is OK, and no declared budget can never breach.
func TestJudgeHookLatency(t *testing.T) {
	over := HookLatencyStats{Count: MinHookAlarmSamples - 1, P99MS: 999}
	if v := JudgeHookLatency(over, DefaultHookP99BudgetMS); !v.OK || !v.Thin || v.Reason != "" {
		t.Fatalf("thin verdict = %+v, want OK+Thin abstention with no reason", v)
	}

	fired := HookLatencyStats{Count: 42, P99MS: 312}
	if v := JudgeHookLatency(fired, DefaultHookP99BudgetMS); v.OK || v.Thin || v.Reason != GateLatencyRegression {
		t.Fatalf("breach verdict = %+v, want !OK with reason %q", v, GateLatencyRegression)
	}

	fine := HookLatencyStats{Count: 42, P99MS: 175.5}
	if v := JudgeHookLatency(fine, DefaultHookP99BudgetMS); !v.OK || v.Thin || v.Reason != "" {
		t.Fatalf("ok verdict = %+v, want clean OK", v)
	}

	if v := JudgeHookLatency(fired, 0); !v.OK || v.Reason != "" {
		t.Fatalf("no-budget verdict = %+v, want fail-open OK (no declared budget cannot breach)", v)
	}
}

// TestGateLatencyRegressionTokenSpelling pins the alarm token to the dos.toml
// [reasons.GATE_LATENCY_REGRESSION] spelling — the fold must emit the exact member
// of the closed vocabulary, never a second spelling of the same reason.
func TestGateLatencyRegressionTokenSpelling(t *testing.T) {
	if GateLatencyRegression != "GATE_LATENCY_REGRESSION" {
		t.Fatalf("GateLatencyRegression = %q, want GATE_LATENCY_REGRESSION", GateLatencyRegression)
	}
}
