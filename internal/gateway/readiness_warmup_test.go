package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestWarmupGate pins the #3051 timing policy as a pure state machine: a
// never-armed gate is silent, an armed-but-incomplete gate holds readiness
// (pending), completion releases it and records time_to_ready, and neither a
// re-arm after completion nor a second completion regresses a warm serve. No live
// model needed.
func TestWarmupGate(t *testing.T) {
	// never armed: silent, readiness unaffected.
	var g warmupGate
	if g.pending() {
		t.Fatal("zero-value warmupGate.pending() = true, want false (never armed is silent)")
	}
	if _, ok := g.ready(); ok {
		t.Fatal("zero-value warmupGate.ready() ok = true, want false")
	}

	// armed but not complete: readiness held.
	g.arm()
	if !g.pending() {
		t.Fatal("armed gate pending() = false, want true (readiness held until warmup)")
	}
	if _, ok := g.ready(); ok {
		t.Fatal("armed-incomplete gate ready() ok = true, want false")
	}

	// completion releases the hold and records the boot->first-token duration.
	g.markComplete(1500 * time.Millisecond)
	if g.pending() {
		t.Fatal("completed gate pending() = true, want false")
	}
	d, ok := g.ready()
	if !ok || d != 1500*time.Millisecond {
		t.Fatalf("completed gate ready() = (%v,%v), want (1.5s,true)", d, ok)
	}

	// re-arm after completion must NOT regress a warm serve back to pending.
	g.arm()
	if g.pending() {
		t.Fatal("re-armed completed gate pending() = true, want false (warm stays warm)")
	}

	// the first completion wins; a later completion does not overwrite time_to_ready.
	g.markComplete(9 * time.Second)
	if d, _ := g.ready(); d != 1500*time.Millisecond {
		t.Fatalf("second markComplete overwrote time_to_ready = %v, want 1.5s (first wins)", d)
	}
}

// TestWarmupGateMarkCompleteClampsNegative pins that a negative boot->first-token
// duration (a clock anomaly) is clamped to zero rather than surfaced as a negative
// time_to_ready_ms.
func TestWarmupGateMarkCompleteClampsNegative(t *testing.T) {
	var g warmupGate
	g.markComplete(-5 * time.Second)
	if d, ok := g.ready(); !ok || d != 0 {
		t.Fatalf("markComplete(-5s) ready() = (%v,%v), want (0,true)", d, ok)
	}
}

// TestWarmupGateMarkCompleteWithoutArm pins that a host running an unconditional
// warmup (markComplete without a prior arm) still exposes time_to_ready and is
// never pending.
func TestWarmupGateMarkCompleteWithoutArm(t *testing.T) {
	var g warmupGate
	g.markComplete(800 * time.Millisecond)
	if g.pending() {
		t.Fatal("markComplete without arm: pending() = true, want false")
	}
	if d, ok := g.ready(); !ok || d != 800*time.Millisecond {
		t.Fatalf("markComplete without arm: ready() = (%v,%v), want (800ms,true)", d, ok)
	}
}

// TestHealthzHoldsUntilWarmup captures the SERVED /healthz response (the issue's
// proof bar): an armed warmup gate flips ok:false with warmup_pending, and once
// the synthetic warmup completes /healthz reports ok:true and exposes
// time_to_ready_ms. A serve that never arms the gate is unaffected. This is the
// gateway-side witness; the live GLM-5.2 boot witness (host-blocked) is the
// remaining acceptance rung on #3051.
func TestHealthzHoldsUntilWarmup(t *testing.T) {
	// never armed: ready, no warmup fields.
	unarmed := &Server{}
	if body := warmupHealthzBody(t, unarmed); body["ok"] != true {
		t.Fatalf("unarmed serve: /healthz ok = %v, want true", body["ok"])
	}

	// armed but not warm: not ready, warmup_pending set.
	pending := &Server{}
	pending.ArmWarmupGate()
	body := warmupHealthzBody(t, pending)
	if body["ok"] != false {
		t.Fatalf("armed-pending serve: /healthz ok = %v, want false", body["ok"])
	}
	if body["warmup_pending"] != true {
		t.Fatalf("armed-pending serve: /healthz warmup_pending = %v, want true", body["warmup_pending"])
	}

	// warmup complete: ready again, with time_to_ready_ms exposed.
	pending.MarkWarmupComplete(1234 * time.Millisecond)
	warm := warmupHealthzBody(t, pending)
	if warm["ok"] != true {
		t.Fatalf("warm serve: /healthz ok = %v, want true", warm["ok"])
	}
	if _, present := warm["warmup_pending"]; present {
		t.Fatalf("warm serve: /healthz still carries warmup_pending, want it gone")
	}
	// JSON numbers decode as float64.
	if got, ok := warm["time_to_ready_ms"].(float64); !ok || got != 1234 {
		t.Fatalf("warm serve: /healthz time_to_ready_ms = %v (%T), want 1234", warm["time_to_ready_ms"], warm["time_to_ready_ms"])
	}
}

// warmupHealthzBody serves one /healthz request against s and returns the decoded
// JSON body. Named distinctly from the coherence gate's healthzBody helper so the
// two readiness tests never collide in the shared package.
func warmupHealthzBody(t *testing.T, s *Server) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	s.handleHealth(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/healthz status = %d, want %d", rec.Code, http.StatusOK)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /healthz body %q: %v", rec.Body.String(), err)
	}
	return body
}
