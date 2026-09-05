package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
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

// TestRunWarmupCompletesGate pins the warm-start execution half (#3051/#3083): an
// armed gate holds /healthz not-ready, and RunWarmup — issuing one synthetic
// completion through the planner — releases it and exposes time_to_ready_ms. Uses
// the offline MockPlanner, so no live model is needed; the ~500s real backend
// warmup is the host-blocked DGX residual, not this state-machine witness.
func TestRunWarmupCompletesGate(t *testing.T) {
	srv := &Server{planner: agent.NewMockPlanner("warmup-test")}
	srv.ArmWarmupGate()
	if !srv.warmup.pending() {
		t.Fatal("armed gate should be pending before RunWarmup")
	}
	if body := warmupHealthzBody(t, srv); body["ok"] != false {
		t.Fatalf("armed-pending serve: /healthz ok = %v, want false", body["ok"])
	}

	if _, err := srv.RunWarmup(context.Background()); err != nil {
		t.Fatalf("RunWarmup err = %v, want nil", err)
	}
	if srv.warmup.pending() {
		t.Fatal("gate still pending after RunWarmup")
	}
	body := warmupHealthzBody(t, srv)
	if body["ok"] != true {
		t.Fatalf("post-warmup serve: /healthz ok = %v, want true", body["ok"])
	}
	if _, present := body["time_to_ready_ms"]; !present {
		t.Fatalf("post-warmup serve: /healthz missing time_to_ready_ms, got %v", body)
	}
}

// TestRunWarmupNilPlannerReleasesGate pins that a serve with no planner (a backend
// that will never warm) does not get stuck pending: RunWarmup releases the gate
// rather than leaving readiness held forever.
func TestRunWarmupNilPlannerReleasesGate(t *testing.T) {
	srv := &Server{}
	srv.ArmWarmupGate()
	d, err := srv.RunWarmup(context.Background())
	if err != nil {
		t.Fatalf("nil-planner RunWarmup err = %v, want nil", err)
	}
	if d != 0 {
		t.Fatalf("nil-planner RunWarmup d = %v, want 0", d)
	}
	if srv.warmup.pending() {
		t.Fatal("nil-planner RunWarmup left the gate pending, want released")
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

// TestInferenceGatedDuringWarmup proves that incoming inference requests are gated
// while warmup is pending (HTTP 503 with Retry-After: 1 and "code":"warmup_pending"),
// and admitted (HTTP 200 OK) once warmup completes.
func TestInferenceGatedDuringWarmup(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = agent.NewMockPlanner("test-model")
	srv.ArmWarmupGate()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	chatPayload := []byte(`{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`)
	messagesPayload := []byte(`{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":100}`)

	// 1. Armed warmup gate: POST /v1/chat/completions returns 503, Retry-After: 1, "code":"warmup_pending"
	res, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatPayload))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	body, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("POST /v1/chat/completions status = %d, want 503", res.StatusCode)
	}
	if got := res.Header.Get("Retry-After"); got != "1" {
		t.Fatalf("POST /v1/chat/completions Retry-After = %q, want 1", got)
	}
	if !strings.Contains(string(body), `"code":"warmup_pending"`) {
		t.Fatalf("POST /v1/chat/completions body = %s, want to contain \"code\":\"warmup_pending\"", string(body))
	}

	// 2. Armed warmup gate: POST /v1/messages returns 503, Retry-After: 1, "code":"warmup_pending"
	res, err = http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(messagesPayload))
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	body, _ = io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("POST /v1/messages status = %d, want 503", res.StatusCode)
	}
	if got := res.Header.Get("Retry-After"); got != "1" {
		t.Fatalf("POST /v1/messages Retry-After = %q, want 1", got)
	}
	if !strings.Contains(string(body), `"code":"warmup_pending"`) {
		t.Fatalf("POST /v1/messages body = %s, want to contain \"code\":\"warmup_pending\"", string(body))
	}

	// 3. Mark warmup complete
	srv.MarkWarmupComplete(100 * time.Millisecond)

	// 4. Post-warmup: POST /v1/chat/completions returns 200 OK
	res, err = http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(chatPayload))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions post-warmup: %v", err)
	}
	body, _ = io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions post-warmup status = %d, want 200; body=%s", res.StatusCode, string(body))
	}

	// 5. Post-warmup: POST /v1/messages returns 200 OK
	res, err = http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(messagesPayload))
	if err != nil {
		t.Fatalf("POST /v1/messages post-warmup: %v", err)
	}
	body, _ = io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/messages post-warmup status = %d, want 200; body=%s", res.StatusCode, string(body))
	}
}
