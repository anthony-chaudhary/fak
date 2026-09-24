package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
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

	// armed but not warm: NOT READY. The #3051 contract is a 503 status (the exact
	// false-ready signal a proxy/opencode client must not route on), with
	// warmup_pending set and Retry-After so the client backs off instead of failing.
	pending := &Server{}
	pending.ArmWarmupGate()
	code, body := warmupHealthz(t, pending)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("armed-pending serve: /healthz status = %d, want 503 (not-ready)", code)
	}
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
	if code, body := warmupHealthz(t, srv); code != http.StatusServiceUnavailable || body["ok"] != false {
		t.Fatalf("armed-pending serve: /healthz = %d ok=%v, want 503 ok=false", code, body["ok"])
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

// TestReadyzGatesWhileWarming isolates the warmup seam on /readyz (distinct from
// the startup gate proven by TestReadyzRequiresStartupAndReusesHealthState): once
// startup is satisfied by MarkReady(), an armed-but-incomplete warmup gate must
// force /readyz to 503 with warmup_pending, and MarkWarmupComplete must release it
// to 200. The startup gate is satisfied first so this witnesses the warmup gate
// specifically, not the startup gate.
func TestReadyGatingWhileWarming(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = agent.NewMockPlanner("test-model")
	srv.MarkReady()

	readyzBody := func(t *testing.T) (int, map[string]any) {
		t.Helper()
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode /readyz body %q: %v", rec.Body.String(), err)
		}
		return rec.Code, body
	}

	srv.ArmWarmupGate()
	status, body := readyzBody(t)
	if status != http.StatusServiceUnavailable {
		t.Fatalf("warming /readyz status = %d, want 503; body=%v", status, body)
	}
	if body["ok"] != false {
		t.Fatalf("warming /readyz ok = %v, want false", body["ok"])
	}
	if body["warmup_pending"] != true {
		t.Fatalf("warming /readyz warmup_pending = %v, want true", body["warmup_pending"])
	}

	srv.MarkWarmupComplete(0)
	status, body = readyzBody(t)
	if status != http.StatusOK {
		t.Fatalf("warm /readyz status = %d, want 200; body=%v", status, body)
	}
	if body["ok"] != true {
		t.Fatalf("warm /readyz ok = %v, want true", body["ok"])
	}
	if body["warmup_pending"] == true {
		t.Fatalf("warm /readyz warmup_pending = %v, want absent/false", body["warmup_pending"])
	}
}

// warmupHealthzBody serves one /healthz request against s and returns the decoded
// JSON body, requiring the READY status (200). Named distinctly from the
// coherence gate's healthzBody helper so the two readiness tests never collide in
// the shared package. The not-ready counterpart is warmupHealthz, which pins the
// #3051 status contract: a warmup-pending body must answer 503, not 200.
func warmupHealthzBody(t *testing.T, s *Server) map[string]any {
	t.Helper()
	code, body := warmupHealthz(t, s)
	if code != http.StatusOK {
		t.Fatalf("/healthz status = %d, want %d (ready body)", code, http.StatusOK)
	}
	return body
}

// warmupHealthz serves one /healthz request against s and returns the status code
// and decoded body, asserting nothing about readiness — the shared seam for both
// the ready and not-ready halves.
func warmupHealthz(t *testing.T, s *Server) (int, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	s.handleHealth(rec, req)
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /healthz body %q: %v", rec.Body.String(), err)
	}
	return rec.Code, body
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

type warmupReporterPlanner struct {
	*agent.MockPlanner
	stats agent.KVMemoryStats
}

func (p *warmupReporterPlanner) KVMemoryStats() agent.KVMemoryStats {
	return p.stats
}

// TestRunWarmupDerivesAdmissionTokenBudget tests that RunWarmup probes the planner's KV memory
// capacity and dynamically derives the admission controller's TokenBudget (issue #5266).
func TestRunWarmupDerivesAdmissionTokenBudget(t *testing.T) {
	srv := newTestServer(t)
	mock := agent.NewMockPlanner("test-model")
	srv.planner = &warmupReporterPlanner{
		MockPlanner: mock,
		stats: agent.KVMemoryStats{
			FitBudgetBytes: 100_000,
			BytesPerToken:  10,
		},
	}
	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	srv.SetAdmissionController(ctl)

	if got, want := ctl.Policy().TokenBudget, 9000; got != want {
		// SetAdmissionController already queried planner and set 9000 if reporter present
		t.Logf("SetAdmissionController immediately derived TokenBudget = %d", got)
	}

	// Reset to default to verify RunWarmup specifically updates it
	ctl.SetTokenBudgetWithProvenance(8192, "default")
	srv.admissionMu.Lock()
	srv.warmupCapacity = nil
	srv.admissionMu.Unlock()

	_, err := srv.RunWarmup(context.Background())
	if err != nil {
		t.Fatalf("RunWarmup err: %v", err)
	}

	if got, want := ctl.Policy().TokenBudget, 9000; got != want {
		t.Fatalf("post-warmup TokenBudget = %d, want %d", got, want)
	}
	if got, want := ctl.TokenBudgetProvenance(), "measured"; got != want {
		t.Fatalf("post-warmup TokenBudgetProvenance = %q, want 'measured'", got)
	}
}

// failingWarmupPlanner is a MockPlanner that returns a scripted error for its
// first N completions, then behaves like the deterministic mock on success. It
// lets the warmup-failure tests distinguish a backend failure (an error reply)
// from a cancellation (a context error surfaced by the mock) from the eventual
// successful retry, all offline.
type failingWarmupPlanner struct {
	*agent.MockPlanner
	failures int
	err      error
	calls    int
}

func (p *failingWarmupPlanner) Complete(ctx context.Context, msgs []agent.Message, defs []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p.failures > 0 {
		p.failures--
		return nil, p.err
	}
	return p.MockPlanner.Complete(ctx, msgs, defs, opts...)
}

// TestWarmupFailureKeepsCacheGatePending pins #13353: a warmup that FAILS (the
// planner returns an error) must leave an armed gate PENDING and return the
// original error — a failed boot may not advertise warm readiness. A LATER
// successful call then completes the gate exactly once, so failure and recovery
// are distinguishable without a retry loop.
func TestWarmupFailureKeepsCacheGatePending(t *testing.T) {
	backendErr := errors.New("backend warmup exploded")
	srv := &Server{planner: &failingWarmupPlanner{
		MockPlanner: agent.NewMockPlanner("warmup-test"),
		failures:    1,
		err:         backendErr,
	}}
	srv.ArmWarmupGate()
	if !srv.warmup.pending() {
		t.Fatal("armed gate should be pending before RunWarmup")
	}

	if _, err := srv.RunWarmup(context.Background()); !errors.Is(err, backendErr) {
		t.Fatalf("failed RunWarmup err = %v, want %v", err, backendErr)
	}
	if !srv.warmup.pending() {
		t.Fatal("failed warmup released the gate, want still pending (a failed boot must not advertise readiness)")
	}
	if code, body := warmupHealthz(t, srv); code != http.StatusServiceUnavailable || body["ok"] != false {
		t.Fatalf("post-failure serve: /healthz = %d ok=%v, want 503 ok=false", code, body["ok"])
	}
	if _, ok := srv.warmup.ready(); ok {
		t.Fatal("failed warmup reported ready, want not-ready")
	}

	// A later successful call completes the gate exactly once.
	if d, err := srv.RunWarmup(context.Background()); err != nil {
		t.Fatalf("retry RunWarmup err = %v, want nil", err)
	} else if d < 0 {
		t.Fatalf("retry RunWarmup d = %v, want >= 0", d)
	}
	if srv.warmup.pending() {
		t.Fatal("gate still pending after successful retry")
	}
	body := warmupHealthzBody(t, srv)
	if body["ok"] != true {
		t.Fatalf("post-retry serve: /healthz ok = %v, want true", body["ok"])
	}
	if _, present := body["time_to_ready_ms"]; !present {
		t.Fatalf("post-retry serve: /healthz missing time_to_ready_ms, got %v", body)
	}
}

// TestWarmupCancellationKeepsCacheGatePending pins the cancellation arm of
// #13353: a cancelled context surfaces as an error, so the gate stays PENDING and
// readiness remains held — a cancelled warmup is not a warm backend.
func TestWarmupCancellationKeepsCacheGatePending(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	srv := &Server{planner: &failingWarmupPlanner{
		MockPlanner: agent.NewMockPlanner("warmup-test"),
	}}
	srv.ArmWarmupGate()

	if _, err := srv.RunWarmup(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled RunWarmup err = %v, want %v", err, context.Canceled)
	}
	if !srv.warmup.pending() {
		t.Fatal("cancelled warmup released the gate, want still pending")
	}
	if _, ok := srv.warmup.ready(); ok {
		t.Fatal("cancelled warmup reported ready, want not-ready")
	}
}

// ---- CW-07 (#13333): agent KV-cache warm readiness -------------------------------

// fakeAgentWarmer is an AgentWarmWarmer whose DeriveWarmPrefix returns a fixed,
// bounded descriptor and whose WarmPrefix returns a scripted receipt/error. It lets
// the readiness witness drive every receipt shape (ready+live claim, partial
// restore, identity mismatch, dead claim, unsupported) with no live model.
type fakeAgentWarmer struct {
	identity string
	receipt  agent.WarmReceipt
	err      error
	calls    int
}

func (f *fakeAgentWarmer) Model() string { return "fake-agent-warmer" }

func (f *fakeAgentWarmer) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (f *fakeAgentWarmer) DeriveWarmPrefix(tenant, agentName string, _ agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error) {
	if tenant == "" {
		return agent.WarmPrefixSpec{}, agent.ErrWarmPrefixUnavailable
	}
	return agent.WarmPrefixSpec{
		StableTokens:      16,
		StableTokenDigest: "sha256:stable",
		InstructionDigest: "sha256:instr",
		AdapterID:         "native-inkernel",
		Identity:          f.identity,
		Scope:             radixkv.CacheIdentity{Tenant: tenant, Agent: agentName},
	}, nil
}

func (f *fakeAgentWarmer) WarmPrefix(_ context.Context, _ agent.WarmPrefixSpec) (agent.WarmReceipt, error) {
	f.calls++
	return f.receipt, f.err
}

// readyWarmReceipt builds a receipt that satisfies the strongest admission rule:
// matching identity, ready status, full restore to the stable boundary, and a live
// claim. Tests mutate one axis at a time to prove the gate is not satisfied by the
// completion bit alone.
func readyWarmReceipt(identity string) agent.WarmReceipt {
	return agent.WarmReceipt{
		Ready:             true,
		Status:            agent.WarmStatusReady,
		Identity:          identity,
		StableTokenDigest: "sha256:stable",
		Scope:             radixkv.CacheIdentity{Tenant: "tenant-a", Agent: "agent-1"},
		RequestedTokens:   16,
		RestoredTokens:    16,
		Claim:             &agent.WarmClaimReceipt{Live: true, Tokens: 16, Bytes: 4096},
	}
}

// agentWarmHealthz serves one /healthz request against s and returns the decoded
// agent_warm block, or nil when the block is absent.
func agentWarmHealthz(t *testing.T, s *Server) (int, map[string]any) {
	t.Helper()
	code, body := warmupHealthz(t, s)
	aw, _ := body["agent_warm"].(map[string]any)
	return code, aw
}

// TestAgentCacheWarmReadiness is the CW-07 (#13333) witness: a CONFIGURED agent warm
// profile admits readiness only against a LIVE receipt � matching identity, restored
// prefix reaching the stable boundary, and (when required) a live residency claim.
// The #3051 backend-warmup completion bit alone cannot satisfy it; a cold, partial,
// mismatched, or claim-dead warm leaves readiness held with a closed reason; an
// unconfigured or unsupported cache never holds readiness at all.
func TestAgentCacheWarmReadiness(t *testing.T) {
	// (0) unconfigured serve: no agent_warm block, readiness unaffected even though
	// MarkWarmupComplete has been called (a loaded backend is NOT a warm prefix).
	unconfigured := &Server{}
	unconfigured.MarkWarmupComplete(time.Millisecond)
	if body := warmupHealthzBody(t, unconfigured); body["agent_warm"] != nil {
		t.Fatalf("unconfigured serve: /healthz agent_warm = %v, want absent", body["agent_warm"])
	}

	// (1) configured but never run: readiness HELD (pending), 503, typed block.
	warmer := &fakeAgentWarmer{identity: "sha256:desc-1"}
	pending := &Server{planner: warmer}
	if _, err := pending.SetAgentWarmProfile(AgentWarmProfile{
		Tenant: "tenant-a", Agent: "agent-1", RequireLiveClaim: true,
	}); err != nil {
		t.Fatalf("SetAgentWarmProfile: %v", err)
	}
	code, aw := agentWarmHealthz(t, pending)
	if code != http.StatusServiceUnavailable || aw == nil || aw["status"] != AgentWarmPending {
		t.Fatalf("configured-unwarmed: /healthz code=%d agent_warm=%v, want 503 status=pending", code, aw)
	}

	// (2) a live warm receipt admits readiness.
	warmer.receipt = readyWarmReceipt("sha256:desc-1")
	if _, err := pending.RunAgentWarmup(context.Background()); err != nil {
		t.Fatalf("RunAgentWarmup (ready): %v", err)
	}
	code, aw = agentWarmHealthz(t, pending)
	if code != http.StatusOK || aw == nil || aw["status"] != AgentWarmReady {
		t.Fatalf("live-warm: /healthz code=%d agent_warm=%v, want 200 status=ready", code, aw)
	}
	if got := aw["restored_tokens"]; got != float64(16) {
		t.Fatalf("live-warm: restored_tokens = %v, want 16", got)
	}

	// (3) a partial restore (restored < requested) is a DEGRADE, not a hit.
	warmer.receipt = readyWarmReceipt("sha256:desc-1")
	warmer.receipt.RestoredTokens = 8
	if _, err := pending.RunAgentWarmup(context.Background()); err != nil {
		t.Fatalf("RunAgentWarmup (partial): %v", err)
	}
	if code, aw = agentWarmHealthz(t, pending); code != http.StatusServiceUnavailable ||
		aw == nil || aw["status"] != AgentWarmDegraded || aw["reason"] != "partial_restore" {
		t.Fatalf("partial-restore: /healthz code=%d agent_warm=%v, want 503 degraded/partial_restore", code, aw)
	}

	// (4) an identity mismatch (a different descriptor warmed) is a DEGRADE.
	warmer.receipt = readyWarmReceipt("sha256:desc-OTHER")
	if _, err := pending.RunAgentWarmup(context.Background()); err != nil {
		t.Fatalf("RunAgentWarmup (mismatch): %v", err)
	}
	if code, aw = agentWarmHealthz(t, pending); aw == nil || aw["reason"] != "identity_mismatch" {
		t.Fatalf("identity-mismatch: /healthz agent_warm=%v, want reason=identity_mismatch", aw)
	}

	// (5) a ready receipt whose required claim is absent or dead is a DEGRADE � the
	// completion/ready bit alone cannot satisfy a configured live-claim contract.
	warmer.receipt = readyWarmReceipt("sha256:desc-1")
	warmer.receipt.Claim = nil
	if _, err := pending.RunAgentWarmup(context.Background()); err != nil {
		t.Fatalf("RunAgentWarmup (no claim): %v", err)
	}
	if code, aw = agentWarmHealthz(t, pending); aw == nil || aw["reason"] != "claim_absent" {
		t.Fatalf("claim-absent: /healthz agent_warm=%v, want reason=claim_absent", aw)
	}
	warmer.receipt = readyWarmReceipt("sha256:desc-1")
	warmer.receipt.Claim = &agent.WarmClaimReceipt{Live: false, Reason: "expired"}
	if _, err := pending.RunAgentWarmup(context.Background()); err != nil {
		t.Fatalf("RunAgentWarmup (dead claim): %v", err)
	}
	if code, aw = agentWarmHealthz(t, pending); aw == nil || aw["reason"] != "expired" {
		t.Fatalf("dead-claim: /healthz agent_warm=%v, want reason=expired", aw)
	}

	// (6) a backend ERROR leaves the gate DEGRADED with a bounded reason (no hang),
	// and served inference is held with the typed agent_warm_pending code.
	warmer.receipt = agent.WarmReceipt{Status: agent.WarmStatusCold, Reason: "prime_failed"}
	warmer.err = errors.New("boom")
	if _, err := pending.RunAgentWarmup(context.Background()); err == nil {
		t.Fatal("RunAgentWarmup (error) err = nil, want the backend error")
	}
	if code, aw = agentWarmHealthz(t, pending); aw == nil || aw["reason"] != "prime_failed" {
		t.Fatalf("warm-error: /healthz agent_warm=%v, want reason=prime_failed", aw)
	}
	rec := httptest.NewRecorder()
	pending.checkWarmupPending(rec)
	if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), `"code":"agent_warm_pending"`) {
		t.Fatalf("held inference: code=%d body=%s, want 503 agent_warm_pending", rec.Code, rec.Body.String())
	}

	// (7) an UNSUPPORTED planner does not hold readiness: there is no cache contract,
	// so the serve is unaffected. No live model is involved.
	unsupported := &Server{planner: &fakeAgentWarmer{identity: "sha256:desc-1", err: agent.ErrWarmPrefixUnsupported}}
	if _, err := unsupported.SetAgentWarmProfile(AgentWarmProfile{Tenant: "tenant-a"}); err != nil {
		t.Fatalf("SetAgentWarmProfile (unsupported): %v", err)
	}
	if _, err := unsupported.RunAgentWarmup(context.Background()); err == nil {
		t.Fatal("RunAgentWarmup (unsupported) err = nil, want ErrWarmPrefixUnsupported")
	}
	code, aw = agentWarmHealthz(t, unsupported)
	if code != http.StatusOK || aw == nil || aw["status"] != AgentWarmUnsupported {
		t.Fatalf("unsupported: /healthz code=%d agent_warm=%v, want 200 status=unsupported", code, aw)
	}
}

// hangingWarmupPlanner is a planner whose Complete never returns and which IGNORES
// its context — the exact strix1 hang shape (#13500): the backend forward spins in
// a user-space tight loop (a stale SPIR-V bundle makes the CPU-side q4k/q8 path
// spin before any device dispatch) and never observes cancellation. It blocks on a
// package-lifetime channel so the goroutine is parked, not CPU-spinning, in tests.
//
// The regression below is deliberately written WITHOUT naming any symbol the fix
// introduces (ErrWarmupTimeout, warmupCeiling): the mandatory red-then-green
// symptom witness overlays this whole file onto the PARENT source, so a reference
// to a fix-only symbol would fail to compile there and the witness would ABSTAIN
// instead of reproducing the hang. It asserts behavior instead.
type hangingWarmupPlanner struct{ block chan struct{} }

func (p *hangingWarmupPlanner) Model() string { return "hanging-warmup-planner" }

func (p *hangingWarmupPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	<-p.block // never closes: the forward never returns, and ctx is deliberately ignored
	return nil, nil
}

// TestRunWarmupBoundedTimeoutOnNonReturningBackend is the #13500 regression: a
// non-returning backend forward must NOT hold readiness warmup_pending forever.
// With a 1s ceiling, RunWarmup returns a timeout promptly, the gate stays PENDING
// (a hung backend is not warm), and /healthz still reports 503 warmup_pending —
// a typed, reported outcome instead of a silent hold. RED at the parent source
// (RunWarmup blocks forever, so the 5s guard trips); GREEN at the fix.
func TestRunWarmupBoundedTimeoutOnNonReturningBackend(t *testing.T) {
	t.Setenv("FAK_WARMUP_CEILING_S", "1")
	srv := &Server{planner: &hangingWarmupPlanner{block: make(chan struct{})}}
	srv.ArmWarmupGate()

	type result struct {
		d   time.Duration
		err error
	}
	done := make(chan result, 1)
	begin := time.Now()
	go func() {
		d, err := srv.RunWarmup(context.Background())
		done <- result{d: d, err: err}
	}()

	select {
	case r := <-done:
		if r.err == nil {
			t.Fatal("RunWarmup on a non-returning backend returned nil error, want a bounded timeout")
		}
		if !strings.Contains(r.err.Error(), "timed out") {
			t.Fatalf("RunWarmup timeout err = %v, want a message naming the timeout", r.err)
		}
		if r.d <= 0 {
			t.Fatalf("RunWarmup reported elapsed %v, want > 0", r.d)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("RunWarmup did not return within 5s on a non-returning backend — readiness is held unbounded (want a bounded ~1s timeout)")
	}
	if elapsed := time.Since(begin); elapsed > 5*time.Second {
		t.Fatalf("RunWarmup bounded wait took %v, want ~1s (the ceiling must bound the hold)", elapsed)
	}
	if !srv.warmup.pending() {
		t.Fatal("timed-out warmup released the gate, want still pending (a hung backend must not advertise readiness)")
	}
	if _, ok := srv.warmup.ready(); ok {
		t.Fatal("timed-out warmup reported ready, want not-ready")
	}
	if code, body := warmupHealthz(t, srv); code != http.StatusServiceUnavailable || body["ok"] != false {
		t.Fatalf("post-timeout serve: /healthz = %d ok=%v, want 503 ok=false", code, body["ok"])
	}
}

// TestRunWarmupCeilingNegativeDisablesBound pins the explicit opt-out: a negative
// FAK_WARMUP_CEILING_S restores the historical unbounded wait, so a backend that
// DOES return still completes the gate normally (the bound is skipped, not faked).
func TestRunWarmupCeilingNegativeDisablesBound(t *testing.T) {
	t.Setenv("FAK_WARMUP_CEILING_S", "-1")
	srv := &Server{planner: agent.NewMockPlanner("warmup-test")}
	srv.ArmWarmupGate()
	d, err := srv.RunWarmup(context.Background())
	if err != nil {
		t.Fatalf("RunWarmup (unbounded) err = %v, want nil", err)
	}
	if d < 0 {
		t.Fatalf("RunWarmup (unbounded) d = %v, want >= 0", d)
	}
	if srv.warmup.pending() {
		t.Fatal("unbounded warmup left the gate pending, want complete")
	}
}

// TestAgentWarmProfileUnconfiguredWhenPlannerNotWarmer pins that a non-native planner
// (no AgentWarmWarmer) leaves the profile unconfigured rather than arming a gate that
// can never be satisfied � the serve stays unaffected.
func TestAgentWarmProfileUnconfiguredWhenPlannerNotWarmer(t *testing.T) {
	srv := &Server{planner: agent.NewMockPlanner("no-warm")}
	if _, err := srv.SetAgentWarmProfile(AgentWarmProfile{Tenant: "tenant-a"}); !errors.Is(err, ErrAgentWarmUnconfigured) {
		t.Fatalf("SetAgentWarmProfile on non-warmer err = %v, want ErrAgentWarmUnconfigured", err)
	}
	if code, aw := agentWarmHealthz(t, srv); code != http.StatusOK || aw != nil {
		t.Fatalf("non-warmer: /healthz code=%d agent_warm=%v, want 200 absent", code, aw)
	}
	if _, err := srv.RunAgentWarmup(context.Background()); !errors.Is(err, ErrAgentWarmUnconfigured) {
		t.Fatalf("RunAgentWarmup on non-warmer err = %v, want ErrAgentWarmUnconfigured", err)
	}
}
