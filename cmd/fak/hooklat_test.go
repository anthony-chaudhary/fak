package main

// hooklat_test.go — the #1993 surface renders: the rollup table, the closed-token
// verdict line, and the one-line guard exit-summary row (including its zero-noise
// contract: no measured hooks, no line).

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

func hookRollupFixture() turntaxmeter.HookLatencyRollup {
	obs := make([]turntaxmeter.HookObservation, 0, 13)
	for i := 0; i < 6; i++ {
		obs = append(obs, turntaxmeter.HookObservation{Verb: "pretool", LatencyMS: 75 + float64(i)})
	}
	for i := 0; i < 7; i++ {
		obs = append(obs, turntaxmeter.HookObservation{Verb: "posttool", LatencyMS: 60 + 15*float64(i)})
	}
	return turntaxmeter.FoldHookLatency(obs)
}

func TestFormatHookLatencyTable(t *testing.T) {
	got := formatHookLatencyTable(hookRollupFixture())
	for _, want := range []string{"verb", "pretool", "posttool", "all", "p99"} {
		if !strings.Contains(got, want) {
			t.Fatalf("table missing %q:\n%s", want, got)
		}
	}
	if !strings.Contains(got, "13") {
		t.Fatalf("table missing the folded n=13 total:\n%s", got)
	}
}

func TestFormatHookLatencyTableEmpty(t *testing.T) {
	got := formatHookLatencyTable(turntaxmeter.FoldHookLatency(nil))
	if !strings.Contains(got, "no measured hook observations") {
		t.Fatalf("empty table must say so explicitly, got:\n%s", got)
	}
}

func TestFormatHookLatencyVerdictStates(t *testing.T) {
	breach := turntaxmeter.JudgeHookLatency(turntaxmeter.HookLatencyStats{Count: 42, P99MS: 312}, 250)
	if got := formatHookLatencyVerdict(breach); !strings.Contains(got, "GATE_LATENCY_REGRESSION") || !strings.Contains(got, "#2073") {
		t.Fatalf("breach verdict must carry the closed token and the reduce-it pointer, got: %s", got)
	}
	thin := turntaxmeter.JudgeHookLatency(turntaxmeter.HookLatencyStats{Count: 3, P99MS: 999}, 250)
	if got := formatHookLatencyVerdict(thin); !strings.Contains(got, "THIN") {
		t.Fatalf("thin verdict must announce the abstention, got: %s", got)
	}
	ok := turntaxmeter.JudgeHookLatency(turntaxmeter.HookLatencyStats{Count: 42, P99MS: 175}, 250)
	if got := formatHookLatencyVerdict(ok); !strings.Contains(got, "OK") {
		t.Fatalf("ok verdict, got: %s", got)
	}
	reportOnly := turntaxmeter.JudgeHookLatency(turntaxmeter.HookLatencyStats{Count: 42, P99MS: 175}, 0)
	if got := formatHookLatencyVerdict(reportOnly); !strings.Contains(got, "REPORT-ONLY") {
		t.Fatalf("no-budget verdict must say report-only, got: %s", got)
	}
}

func TestFormatGuardHookLatencyLine(t *testing.T) {
	r := hookRollupFixture()
	v := turntaxmeter.JudgeHookLatency(r.Total, turntaxmeter.DefaultHookP99BudgetMS)
	got := formatGuardHookLatencyLine(r, v, "session window")
	if !strings.HasPrefix(got, "fak guard: hook-latency — n=13") || !strings.Contains(got, "session window") {
		t.Fatalf("exit-summary line = %q", got)
	}
	if !strings.HasSuffix(got, "\n") {
		t.Fatalf("summary line must be newline-terminated, got %q", got)
	}
	// Zero-noise contract: no measured hooks, no line at all.
	empty := turntaxmeter.FoldHookLatency(nil)
	if got := formatGuardHookLatencyLine(empty, turntaxmeter.JudgeHookLatency(empty.Total, 250), "session window"); got != "" {
		t.Fatalf("no-observation line must be empty, got %q", got)
	}
}

func TestDiscoverHookObservationStreams(t *testing.T) {
	root := t.TempDir()
	mk := func(rel string) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("{}\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mk(filepath.Join(".dos", "metrics", "observations.jsonl"))
	mk(filepath.Join(".dispatch-runs", "run-b", ".dos", "metrics", "observations.jsonl"))
	mk(filepath.Join(".dispatch-runs", "run-a", ".dos", "metrics", "observations.jsonl"))

	got := discoverHookObservationStreams(root)
	if len(got) != 3 {
		t.Fatalf("discovered %d streams, want 3: %v", len(got), got)
	}
	if !strings.Contains(got[0], ".dos") || strings.Contains(got[0], ".dispatch-runs") {
		t.Fatalf("workspace's own stream must come first, got %v", got)
	}
	if !strings.Contains(got[1], "run-a") || !strings.Contains(got[2], "run-b") {
		t.Fatalf("dispatch-run streams must be sorted, got %v", got)
	}
	if empty := discoverHookObservationStreams(t.TempDir()); len(empty) != 0 {
		t.Fatalf("bare root discovered %v, want none", empty)
	}
}
