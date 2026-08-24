package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGatewayHomepageRendersDiscoverySurface(t *testing.T) {
	s, err := New(Config{
		EngineID:   "mock",
		Model:      `model<script>alert("x")</script>`,
		Provider:   "openai",
		APIKey:     "secret-that-must-not-render",
		RequireKey: "required",
		BaseURL:    "https://upstream.invalid/private",
		VDSO:       true,
	})
	if err != nil {
		t.Fatal(err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.RemoteAddr = "127.0.0.1:4567"
	s.Handler().ServeHTTP(rec, req)
	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("GET / = %d, want 200; body=%s", rec.Code, body)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/html; charset=utf-8" {
		t.Fatalf("Content-Type = %q", got)
	}
	for _, want := range []string{
		"fak gateway",
		"mock",
		"model&lt;script&gt;alert",
		"GET /healthz",
		"GET /v1/models",
		"GET /a2a/v1/agent-card",
		"docs/fak/openapi.yaml",
		"GET /metrics",
		"GET /debug/vars",
		"Rich dashboards",
		"on-demand · 9 dashboards",
		"Live gateway",
		`id="live-ready"`,
		`id="live-requests"`,
		`id="live-cache-hits"`,
		`id="live-inflight"`,
		`fetch("/healthz"`,
		`fetch("/metrics"`,
		"setInterval(refresh,5000)",
		"last good values are preserved",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("homepage missing %q", want)
		}
	}
	for _, secret := range []string{"secret-that-must-not-render", "upstream.invalid", `<script>alert`} {
		if strings.Contains(body, secret) {
			t.Errorf("homepage leaked unsafe value %q", secret)
		}
	}
}

func TestGatewayHomepageIsLoopbackDiscoverableButDoesNotCatchUnknownPaths(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "required"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		method string
		path   string
		want   int
	}{
		{http.MethodGet, "/", http.StatusOK},
		{http.MethodHead, "/", http.StatusOK},
		{http.MethodPost, "/", http.StatusMethodNotAllowed},
		{http.MethodGet, "/not-a-real-route", http.StatusNotFound},
	} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(tc.method, tc.path, nil)
		req.RemoteAddr = "127.0.0.1:4567"
		if tc.path != "/" {
			req.Header.Set("Authorization", "Bearer required")
		}
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != tc.want {
			t.Errorf("%s %s = %d, want %d; body=%s", tc.method, tc.path, rec.Code, tc.want, rec.Body.String())
		}
		if tc.method == http.MethodHead && rec.Body.Len() != 0 {
			t.Errorf("HEAD / body length = %d, want 0", rec.Body.Len())
		}
	}
}

func TestGatewayHomepageRequiresAuthOffBox(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai", RequireKey: "required"})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/", "/a2a/v1/agent-card"} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.RemoteAddr = "203.0.113.8:4567"
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("remote GET %s = %d, want 401", path, rec.Code)
		}
	}
}

func TestGatewayHomepageCapturedHTML(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "demo-model", Provider: "openai", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	raw, err := io.ReadAll(rec.Result().Body)
	if err != nil {
		t.Fatal(err)
	}
	// This captured response is the visual witness: it asserts the actual browser
	// surface has live state, a title, identity, and every API + rich-dashboard route.
	wantCards := 6 + len(richDashboardLinks)
	if strings.Count(string(raw), `class="card`) != wantCards {
		t.Fatalf("captured homepage card count = %d, want %d", strings.Count(string(raw), `class="card`), wantCards)
	}
	if !strings.Contains(string(raw), "demo-model") || !strings.Contains(string(raw), "local agent kernel") ||
		!strings.Contains(string(raw), `id="live-state"`) || !strings.Contains(string(raw), "refreshes every 5 seconds") {
		t.Fatalf("captured homepage lacks live identity: %s", raw)
	}
}

func TestGatewayHomepageCapturedRichDashboardRoutesDoNotAliasGateway(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "demo-model", Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	body := rec.Body.String()
	if err := auditRichDashboardLinks(richDashboardLinks); err != nil {
		t.Fatalf("rich dashboard catalog: %v", err)
	}
	if got := strings.Count(body, `class="card rich-dashboard"`); got != len(richDashboardLinks) {
		t.Fatalf("captured rich dashboard card count = %d, want %d\n%s", got, len(richDashboardLinks), body)
	}
	for _, dashboard := range richDashboardLinks {
		wantRoute := `data-dashboard-uid="` + dashboard.UID + `" href="/?dashboard=rich&amp;uid=` + dashboard.UID + `"`
		if strings.Count(body, wantRoute) != 1 {
			t.Errorf("captured homepage route for %q count = %d, want 1", dashboard.UID, strings.Count(body, wantRoute))
		}
		if !strings.Contains(body, "<h2>"+dashboard.Title+"</h2>") {
			t.Errorf("captured homepage missing title %q", dashboard.Title)
		}
	}
	for _, uid := range []string{"fak-fleet-session", "fak-fleet-overview", "fak-guard-adjudication", "fak-cache-health", "fak-startup-load"} {
		if !strings.Contains(body, `uid=`+uid+`"`) {
			t.Errorf("non-gateway selection %q still has no direct route", uid)
		}
	}
}
