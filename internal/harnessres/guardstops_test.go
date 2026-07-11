package harnessres

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// The operator-question rider (GuardStopCounts) is carried on every harness row and is
// ALWAYS present, because a zero is a meaningful reading (no such stops) rather than an
// unread axis. These tests pin all three surfaces — the JSONL row, the human Report
// line, and the Prometheus gauges — so a missing gauge can never read as "no stops"
// (#4348).

func TestGuardStopCountsMarshalAlwaysPresent(t *testing.T) {
	// A bare snapshot that never observed a guard-stops ledger still emits an explicit
	// zero for both counts — the load-bearing "no gap" property.
	bare, err := (Snapshot{NumCPU: 1}).MarshalLedgerRow("guard", "anthropic", "claude", time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(bare, &m); err != nil {
		t.Fatalf("row is not valid JSON: %v\n%s", err, bare)
	}
	for _, k := range []string{"operator_directed_stops", "fail_open_stops"} {
		v, ok := m[k]
		if !ok {
			t.Errorf("%s missing — a zero must be explicit, not a gap: %s", k, bare)
			continue
		}
		if v.(float64) != 0 {
			t.Errorf("%s = %v, want explicit 0", k, v)
		}
	}

	// A snapshot carrying counts marshals them verbatim.
	snap := Snapshot{NumCPU: 1, GuardStops: GuardStopCounts{OperatorDirected: 3, FailOpen: 2}}
	b, err := snap.MarshalLedgerRow("guard", "anthropic", "claude", time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	m = nil
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m["operator_directed_stops"].(float64) != 3 {
		t.Errorf("operator_directed_stops = %v, want 3: %s", m["operator_directed_stops"], b)
	}
	if m["fail_open_stops"].(float64) != 2 {
		t.Errorf("fail_open_stops = %v, want 2: %s", m["fail_open_stops"], b)
	}
}

func TestGuardStopCountsReport(t *testing.T) {
	snap := Snapshot{
		Elapsed:    5 * time.Second,
		Samples:    3,
		NumCPU:     4,
		GuardStops: GuardStopCounts{OperatorDirected: 4, FailOpen: 1},
	}
	out := snap.Report()
	if !strings.Contains(out, "operator-questions 4 asked-human, 1 fail-open") {
		t.Errorf("Report missing operator-questions clause:\n%s", out)
	}
	// Zero still renders — proof in all cases, never a silent omission.
	zero := Snapshot{Elapsed: time.Second, Samples: 1, NumCPU: 1}.Report()
	if !strings.Contains(zero, "operator-questions 0 asked-human, 0 fail-open") {
		t.Errorf("zero Report must still state the counts:\n%s", zero)
	}
}

func TestGuardStopCountsPrometheus(t *testing.T) {
	snap := Snapshot{NumCPU: 1, GuardStops: GuardStopCounts{OperatorDirected: 7, FailOpen: 5}}
	prom := snap.PrometheusText()
	for _, want := range []string{
		"# TYPE fak_harness_operator_directed_stops gauge",
		"fak_harness_operator_directed_stops 7",
		"# TYPE fak_harness_fail_open_stops gauge",
		"fak_harness_fail_open_stops 5",
	} {
		if !strings.Contains(prom, want) {
			t.Errorf("Prometheus missing %q:\n%s", want, prom)
		}
	}
	// A snapshot with no counts still emits explicit zero gauges (a missing gauge in a
	// scrape must not read as "no stops").
	zero := (Snapshot{NumCPU: 1}).PrometheusText()
	if !strings.Contains(zero, "fak_harness_operator_directed_stops 0") ||
		!strings.Contains(zero, "fak_harness_fail_open_stops 0") {
		t.Errorf("zero snapshot must still emit explicit-zero gauges:\n%s", zero)
	}
}
