package gateway

// observer_test.go — the witnesses for #2434 (stratified async observer rungs). The two
// named acceptance tests prove the two structural guarantees the type system is supposed to
// enforce: an observer CANNOT mutate the admitted bytes (it holds a read-only value copy),
// and a flaky observer AUTO-DISABLES after N failures in the window and journals
// HOOK_UNHEALTHY naming its id. A third test pins the /metrics surface (observer_lag_seconds
// + observer_disabled_total).

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// newObserverServer builds a real gateway whose result-admission chain is live (an
// allow-all adjudicator + the ctxmmu result screen), so admitInboundResults runs the true
// blocking chain before the observer stratum is handed its read-only copy.
func newObserverServer(t *testing.T) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	srv, err := New(Config{EngineID: "test", Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// captureObserver records every ObservedResult it is handed AND attempts to mutate its copy.
// Because ObservedResult is a value with no pointer/slice fields, the mutation touches only
// the observer's own copy — it structurally cannot reach the live transcript.
type captureObserver struct {
	id       string
	mu       sync.Mutex
	received []ObservedResult
}

func (o *captureObserver) ObserverID() string { return o.id }

func (o *captureObserver) Observe(_ context.Context, r ObservedResult) error {
	o.mu.Lock()
	o.received = append(o.received, r) // snapshot exactly as received (a value copy)
	o.mu.Unlock()
	// Try to tamper. This assigns to our local copy; the admitted bytes are unreachable.
	r.Content = "MUTATED-BY-OBSERVER"
	r.Verdict = "TAMPERED"
	_ = r
	return nil
}

func TestObserver_CannotMutateAdmittedResult(t *testing.T) {
	srv := newObserverServer(t)
	obs := &captureObserver{id: "capture"}
	srv.RegisterResultObserver(obs, time.Second, 1.0)

	ctx := WithPrincipal(context.Background(), "tenantA")
	const tool, args, result = "get_config", `{"k":"v"}`, `{"answer":42}`
	msgs := inboundTurn("c1", tool, args, result)

	if _, err := srv.admitInboundResults(ctx, msgs, nil, "trace-obs"); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	srv.observers.wait()

	// 1) The admitted transcript is byte-for-byte what the blocking chain settled on — the
	//    observer's attempt to mutate it was structurally impossible.
	if msgs[2].Content != result {
		t.Fatalf("observer changed the admitted result: got %q, want %q", msgs[2].Content, result)
	}

	// 2) The observer really ran, and on the admitted content (delivery works, off the turn).
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if len(obs.received) != 1 {
		t.Fatalf("observer saw %d results, want exactly 1", len(obs.received))
	}
	if obs.received[0].Content != result {
		t.Fatalf("observer was handed %q, want the admitted %q", obs.received[0].Content, result)
	}
	if obs.received[0].Tool != tool || obs.received[0].TraceID != "trace-obs" {
		t.Fatalf("observed metadata wrong: tool=%q trace=%q", obs.received[0].Tool, obs.received[0].TraceID)
	}
}

// failingObserver always errors, driving the health window toward auto-disable.
type failingObserver struct{ id string }

func (o failingObserver) ObserverID() string { return o.id }
func (o failingObserver) Observe(_ context.Context, _ ObservedResult) error {
	return fmt.Errorf("notifier down")
}

func TestObserver_AutoDisableAfterNFailures(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "observer-health.jsonl")
	st := newObserverStratum(journal, nil)
	const id = "flaky"
	st.register(failingObserver{id: id}, time.Second, 1.0)

	// Dispatch exactly failWindow admitted results; every observe fails, so the window fills
	// and the rung auto-disables.
	batch := make([]ObservedResult, st.failWindow)
	for i := range batch {
		batch[i] = ObservedResult{TraceID: "trace-flaky", Tool: "notify", Verdict: "ALLOW"}
	}
	st.dispatch(context.Background(), batch)
	st.wait()

	// The rung is disabled and the auto-disable counter names it exactly once.
	st.mu.Lock()
	rung := st.rungs[0]
	st.mu.Unlock()
	if !rung.isDisabled() {
		t.Fatalf("observer %q was not auto-disabled after %d failures", id, st.failWindow)
	}
	st.metricsMu.Lock()
	got := st.disabled[id]
	st.metricsMu.Unlock()
	if got != 1 {
		t.Fatalf("observer_disabled count for %q = %d, want 1", id, got)
	}

	// A further dispatch does not re-run the disabled observer.
	st.dispatch(context.Background(), []ObservedResult{{TraceID: "trace-flaky", Verdict: "ALLOW"}})
	st.wait()
	st.metricsMu.Lock()
	got = st.disabled[id]
	st.metricsMu.Unlock()
	if got != 1 {
		t.Fatalf("disabled observer re-fired: count for %q = %d, want still 1", id, got)
	}

	// The auto-disable wrote a HOOK_UNHEALTHY journal row naming the observer id.
	raw, err := os.ReadFile(journal)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	rows := strings.Count(string(raw), "HOOK_UNHEALTHY")
	if rows != 1 {
		t.Fatalf("journal has %d HOOK_UNHEALTHY rows, want 1:\n%s", rows, raw)
	}
	if !strings.Contains(string(raw), `"observer_id":"`+id+`"`) {
		t.Fatalf("HOOK_UNHEALTHY row does not name observer %q:\n%s", id, raw)
	}

	// /metrics exposes both families, with the disable counted for this observer.
	var b strings.Builder
	st.writeMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"observer_lag_seconds",
		"observer_disabled_total",
		`fak_gateway_observer_disabled_total{observer="` + id + `"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("writeMetrics output missing %q:\n%s", want, out)
		}
	}
}

// TestObserver_MetricsExposedAtZero pins the dogfood-at-zero posture: the two families are on
// /metrics from the first scrape, before any observer trips, so a panel always exists.
func TestObserver_MetricsExposedAtZero(t *testing.T) {
	srv := newObserverServer(t)
	scrape := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_gateway_observer_lag_seconds histogram",
		"# TYPE fak_gateway_observer_disabled_total counter",
	} {
		if !strings.Contains(scrape, want) {
			t.Fatalf("/metrics missing %q", want)
		}
	}
}
