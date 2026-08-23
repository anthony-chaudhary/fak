package gateway

// session_metrics_test.go — the /metrics witness for the fak_sessions{state} gauge
// (#1204): the live session registry (listSessions, the C1 snapshot) folds at scrape
// time into a per-state count, the family is suppressed when no registry is wired
// (fail-closed), and a GC'd session decrements its bucket on the next scrape (no
// leaked counts). The witness regex this file exists to satisfy is documented in the
// issue: `go test ./internal/gateway -run 'Metrics.*Session|SessionGauge'`.

import (
	"context"
	"strings"
	"testing"
)

// TestMetricsSessionGauge folds the live session registry into fak_sessions{state}:
// every state in the closed vocabulary is emitted (0 when absent), the counts match
// the snapshot, and an unknown Run token is surfaced as a drift row instead of being
// silently dropped.
// TestMetricsSessionRefusalReasons exposes structured envelope refusals on /metrics.
func TestMetricsSessionRefusalReasons(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{
			{TraceID: "wall", Run: "stopped", Reason: "TIME_BUDGET_EXHAUSTED"},
			{TraceID: "spend", Run: "stopped", Reason: "BUDGET_SPEND_EXHAUSTED"},
			{TraceID: "throughput", Run: "stopped", Reason: "THROUGHPUT_BELOW_FLOOR"},
			{TraceID: "operator", Run: "paused", Reason: "maintenance window"},
		}
	}

	text := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_session_refusals gauge",
		`fak_session_refusals{reason="TIME_BUDGET_EXHAUSTED"} 1`,
		`fak_session_refusals{reason="BUDGET_SPEND_EXHAUSTED"} 1`,
		`fak_session_refusals{reason="THROUGHPUT_BELOW_FLOOR"} 1`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}
	if strings.Contains(text, "maintenance window") {
		t.Fatalf("free-form operator reason leaked into metric label:\n%s", text)
	}
}
func TestMetricsSessionGauge(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{
			{TraceID: "a", Run: "running"},
			{TraceID: "b", Run: "running"},
			{TraceID: "c", Run: "throttled"},
			{TraceID: "d", Run: "paused"},
			{TraceID: "e", Run: "weird-token"}, // drift: outside the closed vocabulary
		}
	}

	text := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_sessions gauge",
		`fak_sessions{state="running"} 2`,
		`fak_sessions{state="throttled"} 1`,
		`fak_sessions{state="paused"} 1`,
		// Absent closed-vocabulary states are emitted as 0 so a dashboard series is stable.
		`fak_sessions{state="draining"} 0`,
		`fak_sessions{state="stopped"} 0`,
		// Drift bucket: the unknown token is surfaced honestly, not dropped.
		`fak_sessions{state="weird-token"} 1`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}
	// The drift bucket is NOT one of the always-emitted closed-vocabulary rows: a
	// snapshot with no drift must carry exactly five state rows (the closed set).
	if c := strings.Count(text, "\nfak_sessions{state="); c != 6 { // 5 closed + 1 drift
		t.Fatalf("fak_sessions row count = %d, want 6 (5 closed-vocabulary + 1 drift)\n%s", c, text)
	}
}

// TestMetricsSessionGaugeFoldDecrementsOnGC proves the "no leaked counts" acceptance
// criterion: because the gauge is a READ-TIME fold off the live snapshot (not a
// per-turn increment), a session the registry GCs is absent from the next snapshot and
// its bucket decrements to 0 on the next scrape. A per-turn accumulator would have
// left the stale count behind.
func TestMetricsSessionGaugeFoldDecrementsOnGC(t *testing.T) {
	srv := newTestServer(t)
	snap := []SessionState{
		{TraceID: "a", Run: "running"},
		{TraceID: "b", Run: "paused"},
	}
	srv.listSessions = func(context.Context) []SessionState { return snap }

	if text := srv.renderMetrics(); !strings.Contains(text, `fak_sessions{state="running"} 1`) {
		t.Fatalf("pre-GC scrape missing running=1\n%s", text)
	}

	// The registry GCs session "a" (it transitioned to stopped and was reaped).
	snap = []SessionState{{TraceID: "b", Run: "paused"}}
	text := srv.renderMetrics()
	if !strings.Contains(text, `fak_sessions{state="running"} 0`) {
		t.Fatalf("post-GC scrape did not decrement running to 0\n%s", text)
	}
	if !strings.Contains(text, `fak_sessions{state="paused"} 1`) {
		t.Fatalf("post-GC scrape lost the paused survivor\n%s", text)
	}
}

// TestMetricsSessionGaugeDisabledIsFailClosed pins the fail-closed posture: a Server
// with no listSessions wired (the default serve path) omits the fak_sessions family
// entirely rather than emitting a phantom all-zero surface — mirroring GET
// /v1/fak/sessions, which 404s when the registry is unset.
func TestMetricsSessionGaugeDisabledIsFailClosed(t *testing.T) {
	srv := newTestServer(t)
	// No listSessions injected.
	text := srv.renderMetrics()
	if strings.Contains(text, "fak_sessions") {
		t.Fatalf("disabled registry emitted a fak_sessions surface:\n%s", text)
	}
}
