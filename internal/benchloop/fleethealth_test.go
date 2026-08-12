package benchloop

import (
	"testing"
	"time"
)

func TestSummarizeFleetRefusesHealthyWithoutAMeasurement(t *testing.T) {
	// The audited queue: eight durable requests, zero benchmark numbers, and a
	// scheduler that read result 0 anyway (#6503).
	cells := []FleetCell{
		{Machine: "gcp-g2-l4", State: "failed", Attempts: 3, Failures: 3, Seconds: 41},
		{Machine: "gcp-g2-l4-32", State: "failed", Attempts: 2, Failures: 2, Seconds: 33},
		{Machine: "gcp-a3-high-h100-1g", State: "failed", Attempts: 1, Failures: 1, Seconds: 12},
		{Machine: "a100", State: "failed", Attempts: 1, Failures: 1, Seconds: 9},
		{Machine: "cpu-server-a", State: "failed", Attempts: 1, Failures: 1, Seconds: 8},
		{Machine: "workstation-a", State: "waiting_session", Attempts: 1, Seconds: 3},
		{Machine: "node-macos-a", State: "waiting_session", Attempts: 1, Seconds: 3},
		{Machine: "gcp-g2-l4", Benchmark: "parity", State: "running", Attempts: 1, Seconds: 600},
	}
	got := SummarizeFleet(cells)
	if got.Result == FleetResultHealthy || got.Healthy {
		t.Fatalf("zero measurements reported healthy: %+v", got)
	}
	if got.Result != FleetResultFailed {
		t.Fatalf("result = %d, want %d with failed cells", got.Result, FleetResultFailed)
	}
	if got.Successful != 0 || got.Failed != 5 || got.Held != 2 || got.Running != 1 {
		t.Fatalf("counts = %+v", got)
	}
	if got.Attempted != 8 || got.RepeatedFailures != 2 {
		t.Fatalf("attempted=%d repeated=%d, want 8 and 2", got.Attempted, got.RepeatedFailures)
	}
	if got.ComputeSeconds != 709 {
		t.Fatalf("compute seconds = %v, want 709", got.ComputeSeconds)
	}
	if got.HeldReasons["session"] != 2 {
		t.Fatalf("held reasons = %v", got.HeldReasons)
	}
	if got.Schema != FleetUtilitySchema {
		t.Fatalf("schema = %q", got.Schema)
	}
}

func TestSummarizeFleetHeldOnlyQueueIsNotHealthy(t *testing.T) {
	// No failure this tick, and still not one benchmark number: the exact shape
	// that used to exit 0 because nothing was claimed.
	got := SummarizeFleet([]FleetCell{
		{Machine: "workstation-a", State: "waiting_session"},
		{Machine: "node-macos-a", State: "held", HeldReason: "session"},
		{Machine: "a100", State: "waiting_credentials"},
		{Machine: "gcp-g2-l4", State: "queued"},
	})
	if got.Result != FleetResultNoMeasurement || got.Healthy {
		t.Fatalf("held-only queue reported %+v", got)
	}
	if got.Held != 3 || got.Queued != 1 {
		t.Fatalf("counts = %+v", got)
	}
	if got.HeldReasons["credentials"] != 1 || got.HeldReasons["session"] != 2 {
		t.Fatalf("held reasons = %v", got.HeldReasons)
	}
}

func TestSummarizeFleetGreenOnlyWithASuccessfulMeasurement(t *testing.T) {
	got := SummarizeFleet([]FleetCell{
		{Machine: "gcp-g2-l4", State: "succeeded", Attempts: 1, Measured: true, Seconds: 22},
		{Machine: "node-macos-a", State: "waiting_session", Attempts: 1},
	})
	if got.Result != FleetResultHealthy || !got.Healthy {
		t.Fatalf("measured fleet reported %+v", got)
	}
	if got.Successful != 1 || got.Measured != 1 || got.Held != 1 {
		t.Fatalf("counts = %+v", got)
	}
	if idle := SummarizeFleet(nil); idle.Result != FleetResultHealthy || idle.Healthy {
		t.Fatalf("empty queue = %+v, want idle result 0 and healthy false", idle)
	}
}

func TestShouldDispatchFleetCellHoldsUnavailableNodes(t *testing.T) {
	now := time.Date(2026, 8, 12, 12, 0, 0, 0, time.UTC)
	if ok, why := ShouldDispatchFleetCell(FleetCell{State: "queued"}, now); !ok || why != "" {
		t.Fatalf("queued cell = (%v, %q)", ok, why)
	}
	held := FleetCell{State: "waiting_session", HeldSince: now.Add(-15 * time.Minute)}
	ok, why := ShouldDispatchFleetCell(held, now)
	if ok {
		t.Fatal("held node re-dispatched on the next 15-minute tick")
	}
	if why == "" {
		t.Fatal("held cell skipped without a reason")
	}
	held.HeldSince = now.Add(-FleetHeldRetry)
	if ok, _ := ShouldDispatchFleetCell(held, now); !ok {
		t.Fatal("held node never re-probed")
	}
	if ok, _ := ShouldDispatchFleetCell(FleetCell{State: "waiting_session"}, now); !ok {
		t.Fatal("a first hold with no timestamp must still be probed once")
	}
	if ok, _ := ShouldDispatchFleetCell(FleetCell{State: "running"}, now); ok {
		t.Fatal("running cell re-claimed")
	}
	if ok, _ := ShouldDispatchFleetCell(FleetCell{State: "succeeded"}, now); ok {
		t.Fatal("measured cell re-run")
	}
}

func TestShouldDispatchFleetCellBacksOffRepeatedFailures(t *testing.T) {
	now := time.Date(2026, 8, 12, 12, 0, 0, 0, time.UTC)
	cell := FleetCell{State: "failed", Failures: 1, LastAttempt: now.Add(-15 * time.Minute)}
	if ok, _ := ShouldDispatchFleetCell(cell, now); ok {
		t.Fatal("failed cell retried on the very next tick")
	}
	cell.LastAttempt = now.Add(-FleetFailureRetry)
	if ok, _ := ShouldDispatchFleetCell(cell, now); !ok {
		t.Fatal("failed cell never retried")
	}
	cell.Failures = 4
	cell.LastAttempt = now.Add(-2 * FleetFailureRetry)
	if ok, _ := ShouldDispatchFleetCell(cell, now); ok {
		t.Fatal("repeated failure did not back off")
	}
	if got, want := FleetFailureBackoff(4), 8*time.Hour; got != want {
		t.Fatalf("backoff(4) = %s, want %s", got, want)
	}
	if got := FleetFailureBackoff(9); got != FleetFailureRetryMax {
		t.Fatalf("backoff(9) = %s, want the %s cap", got, FleetFailureRetryMax)
	}
	if got := FleetFailureBackoff(0); got != FleetFailureRetry {
		t.Fatalf("backoff(0) = %s, want %s", got, FleetFailureRetry)
	}
}

func TestReconcileFleetRunningClearsStaleClaims(t *testing.T) {
	now := time.Date(2026, 8, 12, 12, 0, 0, 0, time.UTC)
	live := FleetClaim{Present: true, PID: 4242, Host: "control", Local: true, Alive: true, Started: now.Add(-time.Minute)}
	if state, why := ReconcileFleetRunning(live, now); state != FleetRunning || why != "" {
		t.Fatalf("live claim = (%q, %q)", state, why)
	}
	dead := live
	dead.Alive = false
	state, why := ReconcileFleetRunning(dead, now)
	if state != FleetQueued || why == "" {
		t.Fatalf("dead dispatcher = (%q, %q), want requeue with a reason", state, why)
	}
	orphan := live
	orphan.Present = false
	if state, _ := ReconcileFleetRunning(orphan, now); state != FleetQueued {
		t.Fatalf("orphaned lock = %q, want requeue", state)
	}
	remote := FleetClaim{Present: true, PID: 77, Host: "other", Started: now.Add(-time.Minute)}
	if state, _ := ReconcileFleetRunning(remote, now); state != FleetRunning {
		t.Fatalf("remote claim inside its lease = %q, want running", state)
	}
	remote.Started = now.Add(-FleetRunningLease)
	if state, why := ReconcileFleetRunning(remote, now); state != FleetQueued || why == "" {
		t.Fatalf("expired lease = (%q, %q), want requeue", state, why)
	}
}

func TestFleetMeasurementSeparatesIdentityFromNumbers(t *testing.T) {
	if _, _, ok := FleetMeasurement("FAK_BENCH_NODE=fak-cuda-build-l4\ngpu=NVIDIA L4\n"); ok {
		t.Fatal("identity marker alone counted as a measurement")
	}
	key, value, ok := FleetMeasurement("FAK_BENCH_NODE=fak-realmodel\nFAK_BENCH_HTTP=200 FAK_BENCH_SECONDS=1.25\n")
	if !ok || key != "FAK_BENCH_HTTP" || value != 200 {
		t.Fatalf("measurement = (%q, %v, %v)", key, value, ok)
	}
	if _, _, ok := FleetMeasurement("FAK_BENCH_MODEL=qwen/qwen3.6-27b\n"); ok {
		t.Fatal("non-numeric marker counted as a measurement")
	}
	if _, _, ok := FleetMeasurement(""); ok {
		t.Fatal("empty transcript counted as a measurement")
	}
}

func TestFleetReenableNeedsAWitnessedNumber(t *testing.T) {
	marker := SummarizeFleet([]FleetCell{{Machine: "gcp-g2-l4", State: "succeeded", Attempts: 1}})
	if ok, why := FleetReenableAllowed(marker); ok || why == "" {
		t.Fatalf("re-arm allowed on a marker-only witness: (%v, %q)", ok, why)
	}
	measured := SummarizeFleet([]FleetCell{{Machine: "gcp-g2-l4", State: "succeeded", Attempts: 1, Measured: true}})
	if ok, _ := FleetReenableAllowed(measured); !ok {
		t.Fatal("re-arm refused after a witnessed numeric benchmark")
	}
}
