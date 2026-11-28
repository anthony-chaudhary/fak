package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fak-test:runtime fast est=20ms
func TestRouterStatusWithoutRoster(t *testing.T) {
	srv := &Server{model: "boot-model"}
	rec := httptest.NewRecorder()
	srv.handleRouterStatus(rec, httptest.NewRequest(http.MethodGet, "/v1/router/status", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var got routerStatusResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Schema != "fak-router-status/v1" {
		t.Fatalf("schema=%q, want fak-router-status/v1", got.Schema)
	}
	if got.RoutingEnabled {
		t.Fatalf("routing_enabled=true, want false")
	}
	if got.DefaultModel != "boot-model" {
		t.Fatalf("default_model=%q, want boot-model", got.DefaultModel)
	}
	if got.Routes == nil || len(got.Routes) != 0 {
		t.Fatalf("routes=%v, want empty non-nil slice", got.Routes)
	}
}

// fak-test:runtime fast est=50ms
func TestRouterStatusWithRosterAndAfter503Failover(t *testing.T) {
	now := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	tracker := newRouteCircuitTracker(30*time.Second, func() time.Time { return now })

	primary := newOpenAIUpstreamMock(t, http.StatusServiceUnavailable, "", nil, nil)
	fallback := newOpenAIUpstreamMock(t, http.StatusOK, "ok-from-fb", nil, nil)
	t.Setenv("TEST_STATUS_FB_KEY", "sk-status")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1"},
			{ID: "cloud-fb", Kind: modelroute.KindOpenAI, BaseURL: fallback.URL + "/v1", CredEnv: "TEST_STATUS_FB_KEY"},
		},
		Bindings: []modelroute.Binding{
			{
				Model:         "default",
				Account:       "local-gpu",
				UpstreamModel: "qwen-local",
				Fallbacks:     []modelroute.FallbackTarget{{Account: "cloud-fb", UpstreamModel: "gpt-4o-mini"}},
			},
			{
				Model:          "private",
				Account:        "local-gpu",
				UpstreamModel:  "qwen-local",
				ResidencyFloor: "local_only",
			},
		},
	}

	srv := newTestFailoverGateway(t, roster, tracker)
	handler := srv.Handler()

	// Before any traffic: all targets healthy, totals zero.
	rec0 := httptest.NewRecorder()
	handler.ServeHTTP(rec0, httptest.NewRequest(http.MethodGet, "/v1/router/status", nil))
	if rec0.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec0.Code, rec0.Body.String())
	}
	var before routerStatusResponse
	if err := json.Unmarshal(rec0.Body.Bytes(), &before); err != nil {
		t.Fatalf("decode before: %v", err)
	}
	if !before.RoutingEnabled || len(before.Routes) != 2 {
		t.Fatalf("before=%+v, want routing_enabled=true and 2 routes", before)
	}
	if before.Routes[0].Model != "default" || before.Routes[0].Primary.Locality != "self_hosted" || before.Routes[0].Primary.State != "healthy" {
		t.Fatalf("unexpected primary route[0]: %+v", before.Routes[0])
	}
	if len(before.Routes[0].Fallbacks) != 1 || before.Routes[0].Fallbacks[0].Locality != "vendor" || before.Routes[0].Fallbacks[0].UpstreamModel != "gpt-4o-mini" {
		t.Fatalf("unexpected fallback route[0]: %+v", before.Routes[0].Fallbacks)
	}
	if before.Routes[1].Model != "private" || before.Routes[1].ResidencyFloor != "local_only" {
		t.Fatalf("unexpected route[1]: %+v", before.Routes[1])
	}

	// Trigger a 503 failover on "default".
	chatRec := postChatCompletion(t, handler, "default")
	if chatRec.Code != http.StatusOK {
		t.Fatalf("chat status=%d body=%s", chatRec.Code, chatRec.Body.String())
	}

	// Advance clock by 5s so cooldown_remaining_ms should be 25000ms.
	now = now.Add(5 * time.Second)
	rec1 := httptest.NewRecorder()
	handler.ServeHTTP(rec1, httptest.NewRequest(http.MethodGet, "/v1/router/status", nil))
	if rec1.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec1.Code, rec1.Body.String())
	}
	var after routerStatusResponse
	if err := json.Unmarshal(rec1.Body.Bytes(), &after); err != nil {
		t.Fatalf("decode after: %v", err)
	}
	if after.Routes[0].Primary.State != "cooldown" || after.Routes[0].Primary.CooldownRemainingMs != 25000 {
		t.Fatalf("primary after failover=%+v, want state=cooldown cooldown_remaining_ms=25000", after.Routes[0].Primary)
	}
	if after.Routes[0].Fallbacks[0].State != "healthy" {
		t.Fatalf("fallback after failover=%+v, want state=healthy", after.Routes[0].Fallbacks[0])
	}
	if after.Totals.Attempts != 2 || after.Totals.Successes != 1 || after.Totals.Failovers != 1 {
		t.Fatalf("totals=%+v, want attempts=2 successes=1 failovers=1", after.Totals)
	}
}

// fak-test:runtime fast est=20ms
func TestRouterStatusAuthGateMatchesModels(t *testing.T) {
	srv := newTestFailoverGateway(t, nil, nil)
	srv.requireKey = "secret-gateway-key"
	handler := srv.Handler()

	unauth := httptest.NewRecorder()
	handler.ServeHTTP(unauth, httptest.NewRequest(http.MethodGet, "/v1/router/status", nil))
	if unauth.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status=%d, want 401", unauth.Code)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/router/status", nil)
	req.Header.Set("Authorization", "Bearer secret-gateway-key")
	authed := httptest.NewRecorder()
	handler.ServeHTTP(authed, req)
	if authed.Code != http.StatusOK {
		t.Fatalf("authenticated status=%d, want 200; body=%s", authed.Code, authed.Body.String())
	}
}

// fak-test:runtime fast est=20ms
func TestMetricsRouterEmptySnapshot(t *testing.T) {
	srv := newTestFailoverGateway(t, nil, nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{
		"# TYPE fak_router_account_cooldown gauge",
		"# TYPE fak_router_failover_total counter",
		"fak_router_failover_total 0",
		"# TYPE fak_router_attempts_total counter",
		"fak_router_attempts_total 0",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("/metrics missing %q", want)
		}
	}
}

// fak-test:runtime fast est=50ms
func TestMetricsRouterAfterFailover(t *testing.T) {
	now := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	tracker := newRouteCircuitTracker(30*time.Second, func() time.Time { return now })

	primary := newOpenAIUpstreamMock(t, http.StatusServiceUnavailable, "", nil, nil)
	fallback := newOpenAIUpstreamMock(t, http.StatusOK, "ok-from-fb", nil, nil)
	t.Setenv("TEST_METRICS_FB_KEY", "sk-metrics")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: primary.URL + "/v1"},
			{ID: "cloud-fb", Kind: modelroute.KindOpenAI, BaseURL: fallback.URL + "/v1", CredEnv: "TEST_METRICS_FB_KEY"},
		},
		Bindings: []modelroute.Binding{
			{
				Model:         "default",
				Account:       "local-gpu",
				UpstreamModel: "qwen-local",
				Fallbacks:     []modelroute.FallbackTarget{{Account: "cloud-fb", UpstreamModel: "gpt-4o-mini"}},
			},
		},
	}

	srv := newTestFailoverGateway(t, roster, tracker)
	handler := srv.Handler()

	chatRec := postChatCompletion(t, handler, "default")
	if chatRec.Code != http.StatusOK {
		t.Fatalf("chat status=%d body=%s", chatRec.Code, chatRec.Body.String())
	}

	rec1 := httptest.NewRecorder()
	handler.ServeHTTP(rec1, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec1.Code != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", rec1.Code)
	}
	body1 := rec1.Body.String()
	for _, want := range []string{
		`fak_router_account_cooldown{account="cloud-fb"} 0`,
		`fak_router_account_cooldown{account="local-gpu"} 1`,
		"fak_router_failover_total 1",
		"fak_router_attempts_total 2",
	} {
		if !strings.Contains(body1, want) {
			t.Errorf("/metrics during cooldown missing %q", want)
		}
	}

	// Advance clock past the 30s cooldown; cooldown gauge returns to 0.
	now = now.Add(31 * time.Second)
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec2.Code != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", rec2.Code)
	}
	if !strings.Contains(rec2.Body.String(), `fak_router_account_cooldown{account="local-gpu"} 0`) {
		t.Errorf("/metrics after cooldown expiry did not clear local-gpu cooldown")
	}
}

// fak-test:runtime fast est=20ms
func TestRouterStatusDashboardPanelRendered(t *testing.T) {
	srv := newTestFailoverGateway(t, nil, nil)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.RemoteAddr = "127.0.0.1:4567"
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET / = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{
		`id="router-panel"`,
		`Local router`,
		`id="router-state"`,
		`id="router-routes"`,
		`id="router-attempts"`,
		`id="router-failovers"`,
		`id="router-list"`,
		`fetch("/v1/router/status"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("dashboard HTML missing %q", want)
		}
	}
}
