package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/harnessversion"
)

// TestHarnessVersion_DefaultRouterInitialization verifies that gateway.New initializes
// a default StickySessionRouter with v1 active when Config.HarnessRouter is nil, and
// that SetHarnessRouter and HarnessRouter properly inspect and update the router.
func TestHarnessVersion_DefaultRouterInitialization(t *testing.T) {
	srv := newTestServer(t)

	router := srv.HarnessRouter()
	if router == nil {
		t.Fatal("expected non-nil default HarnessRouter on Server")
	}

	if defaultVer := router.DefaultVersion(); defaultVer != string(harnessversion.VersionV1) {
		t.Fatalf("expected default version %q, got %q", harnessversion.VersionV1, defaultVer)
	}

	active := router.ActiveVersions()
	if len(active) != 1 || active[0].Version != string(harnessversion.VersionV1) {
		t.Fatalf("expected active version v1, got %+v", active)
	}

	customRouter := harnessversion.NewStickySessionRouter()
	if err := customRouter.Register(harnessversion.VersionDescriptor{
		Version: string(harnessversion.VersionV2),
		Weight:  100,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register custom v2 failed: %v", err)
	}

	srv.SetHarnessRouter(customRouter)
	if srv.HarnessRouter() != customRouter {
		t.Fatal("expected SetHarnessRouter to replace router")
	}
	if srv.HarnessRouter().DefaultVersion() != string(harnessversion.VersionV2) {
		t.Fatalf("expected custom router default version %q, got %q", harnessversion.VersionV2, srv.HarnessRouter().DefaultVersion())
	}
}

// TestHarnessVersion_ExplicitWireNegotiation verifies explicit wire negotiation via
// the X-Fak-Harness-Version header and path parameters.
func TestHarnessVersion_ExplicitWireNegotiation(t *testing.T) {
	srv := newTestServer(t)
	router := harnessversion.NewStickySessionRouter()
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v1",
		Weight:  100,
		Active:  true,
		Metadata: map[string]string{"default": "true"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v2",
		Weight:  50,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}
	srv.SetHarnessRouter(router)

	// 1. Explicit negotiation for v2 via header
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Session-Id", "sess-wire-v2")
	req.Header.Set(harnessversion.HeaderHarnessVersion, "v2")
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Errorf("expected response header %s = %q, got %q", harnessversion.HeaderHarnessVersion, "v2", got)
	}

	// 2. Explicit negotiation for v1 via header
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Session-Id", "sess-wire-v1")
	req.Header.Set(harnessversion.HeaderHarnessVersion, "v1")
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v1" {
		t.Errorf("expected response header %s = %q, got %q", harnessversion.HeaderHarnessVersion, "v1", got)
	}

	// 3. Explicit negotiation via URL path
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v2/chat/completions", nil)
	req.Header.Set("X-Fak-Session-Id", "sess-path-v2")
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Errorf("expected path negotiation response header %s = %q, got %q", harnessversion.HeaderHarnessVersion, "v2", got)
	}
}

// TestHarnessVersion_StickySessionAffinity verifies that once a session is mapped to
// a harness version, all subsequent turns maintain sticky affinity to that pinned version.
func TestHarnessVersion_StickySessionAffinity(t *testing.T) {
	srv := newTestServer(t)
	router := harnessversion.NewStickySessionRouter()
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v1",
		Weight:  100,
		Active:  true,
		Metadata: map[string]string{"default": "true"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v2",
		Weight:  50,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}
	srv.SetHarnessRouter(router)

	sessionID := "sticky-session-turn-series-101"

	// Turn 1: explicitly select v2 and pin
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Session-Id", sessionID)
	req.Header.Set(harnessversion.HeaderHarnessVersion, "v2")
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Fatalf("turn 1: expected response header v2, got %q", got)
	}

	// Turns 2-10: subsequent requests with NO version header must stay pinned to v2
	for turn := 2; turn <= 10; turn++ {
		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("X-Fak-Session-Id", sessionID)
		srv.beginServedRequest(rec, req)

		if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
			t.Fatalf("turn %d: expected sticky affinity to v2, got %q", turn, got)
		}
	}

	// Turn 11: even if an invalid or conflicting version header is sent, pinned version wins
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Session-Id", sessionID)
	req.Header.Set(harnessversion.HeaderHarnessVersion, "v1") // conflicting
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Fatalf("turn 11: expected pinned session to resist conflicting header and stay v2, got %q", got)
	}

	// Test affinity using X-Trace-Id when X-Fak-Session-Id is absent
	traceID := "trace-session-affinity-202"
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Trace-Id", traceID)
	req.Header.Set(harnessversion.HeaderHarnessVersion, "v2")
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Fatalf("trace turn 1: expected v2, got %q", got)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Trace-Id", traceID)
	srv.beginServedRequest(rec, req)

	if got := rec.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Fatalf("trace turn 2: expected sticky v2, got %q", got)
	}
}

// TestHarnessVersion_CanaryTrafficSplitting verifies weighted canary distribution
// across active versions for unpinned requests without wire signals.
func TestHarnessVersion_CanaryTrafficSplitting(t *testing.T) {
	srv := newTestServer(t)
	router := harnessversion.NewStickySessionRouter()
	// 70% traffic to v1, 30% traffic to v2
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v1",
		Weight:  70,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v2",
		Weight:  30,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}

	srv.SetHarnessRouter(router)

	// Inject deterministic RNG to verify bucket selection:
	// total weight = 100; roll < 70 -> v1; roll >= 70 -> v2
	rollValue := 10
	router.SetRandFunc(func(n int) int {
		return rollValue
	})

	// Session A: roll=10 should land in v1 bucket
	recA := httptest.NewRecorder()
	reqA := httptest.NewRequest(http.MethodPost, "/", nil)
	reqA.Header.Set("X-Fak-Session-Id", "sess-canary-alpha")
	srv.beginServedRequest(recA, reqA)

	if got := recA.Header().Get(harnessversion.HeaderHarnessVersion); got != "v1" {
		t.Errorf("expected canary roll=10 to select v1, got %q", got)
	}

	// Session B: roll=85 should land in v2 bucket
	rollValue = 85
	recB := httptest.NewRecorder()
	reqB := httptest.NewRequest(http.MethodPost, "/", nil)
	reqB.Header.Set("X-Fak-Session-Id", "sess-canary-beta")
	srv.beginServedRequest(recB, reqB)

	if got := recB.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Errorf("expected canary roll=85 to select v2, got %q", got)
	}

	// Verify both sessions are now stickily pinned regardless of subsequent RNG rolls
	rollValue = 99 // would otherwise select v2
	recA2 := httptest.NewRecorder()
	reqA2 := httptest.NewRequest(http.MethodPost, "/", nil)
	reqA2.Header.Set("X-Fak-Session-Id", "sess-canary-alpha")
	srv.beginServedRequest(recA2, reqA2)

	if got := recA2.Header().Get(harnessversion.HeaderHarnessVersion); got != "v1" {
		t.Errorf("expected sess-canary-alpha to remain pinned to v1, got %q", got)
	}

	rollValue = 0 // would otherwise select v1
	recB2 := httptest.NewRecorder()
	reqB2 := httptest.NewRequest(http.MethodPost, "/", nil)
	reqB2.Header.Set("X-Fak-Session-Id", "sess-canary-beta")
	srv.beginServedRequest(recB2, reqB2)

	if got := recB2.Header().Get(harnessversion.HeaderHarnessVersion); got != "v2" {
		t.Errorf("expected sess-canary-beta to remain pinned to v2, got %q", got)
	}
}

// TestHarnessVersion_FallbackToDefault verifies that passing an invalid, malformed,
// or unregistered version fails closed to the configured default stable version.
func TestHarnessVersion_FallbackToDefault(t *testing.T) {
	srv := newTestServer(t)
	router := harnessversion.NewStickySessionRouter()
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v1",
		Weight:  100,
		Active:  true,
		Metadata: map[string]string{"default": "true"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v2",
		Weight:  50,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}
	srv.SetHarnessRouter(router)

	invalidVersions := []string{
		"v999-unregistered",
		"invalid_version",
		"v3.0.0-rc1",
		"v0",
	}

	for _, badVer := range invalidVersions {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("X-Fak-Session-Id", "sess-bad-"+badVer)
		req.Header.Set(harnessversion.HeaderHarnessVersion, badVer)
		srv.beginServedRequest(rec, req)

		got := rec.Header().Get(harnessversion.HeaderHarnessVersion)
		if got != "v1" {
			t.Errorf("for invalid version %q: expected fallback to default v1, got %q", badVer, got)
		}
	}
}

// TestHarnessVersion_HTTPChatCompletionsIntegration tests end-to-end HTTP request processing
// through /v1/chat/completions verifying X-Fak-Harness-Version is set on the response.
func TestHarnessVersion_HTTPChatCompletionsIntegration(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "Hello from harness runtime"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}}

	router := harnessversion.NewStickySessionRouter()
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v1",
		Weight:  80,
		Active:  true,
		Metadata: map[string]string{"default": "true"},
	}); err != nil {
		t.Fatalf("Register v1 failed: %v", err)
	}
	if err := router.Register(harnessversion.VersionDescriptor{
		Version: "v2",
		Weight:  20,
		Active:  true,
	}); err != nil {
		t.Fatalf("Register v2 failed: %v", err)
	}
	srv.SetHarnessRouter(router)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. Send request with explicit v2
	chatBody, _ := json.Marshal(map[string]any{
		"model": "test-model",
		"messages": []map[string]string{
			{"role": "user", "content": "test prompt"},
		},
	})

	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(chatBody))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Session-Id", "http-session-1")
	req.Header.Set(harnessversion.HeaderHarnessVersion, "v2")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if ver := resp.Header.Get(harnessversion.HeaderHarnessVersion); ver != "v2" {
		t.Fatalf("expected response header %s = v2, got %q", harnessversion.HeaderHarnessVersion, ver)
	}

	// 2. Subsequent turn for the same session without version header stays sticky to v2
	req2, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(chatBody))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("X-Fak-Session-Id", "http-session-1")

	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatalf("Do turn 2: %v", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp2.StatusCode)
	}
	if ver := resp2.Header.Get(harnessversion.HeaderHarnessVersion); ver != "v2" {
		t.Fatalf("expected turn 2 response header %s = v2, got %q", harnessversion.HeaderHarnessVersion, ver)
	}
}
