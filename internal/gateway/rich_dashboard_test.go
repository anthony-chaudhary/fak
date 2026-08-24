package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRichDashboardDormantUntilFirstClickThenRedirects(t *testing.T) {
	m := newRichDashboardManager()
	defer m.close()
	m.baseURL = "http://grafana.test"
	var probes atomic.Int32
	m.probe = func(context.Context, string) error { probes.Add(1); return nil }
	s := testServerWithRichDashboards(t, m)

	home := httptest.NewRecorder()
	s.Handler().ServeHTTP(home, httptest.NewRequest(http.MethodGet, "/", nil))
	if got := probes.Load(); got != 0 {
		t.Fatalf("opening lightweight dashboard performed %d Grafana probes, want 0", got)
	}
	if !strings.Contains(home.Body.String(), "Rich dashboards") || !strings.Contains(home.Body.String(), "on-demand") {
		t.Fatalf("captured default dashboard lacks on-demand destination: %s", home.Body.String())
	}

	first := httptest.NewRecorder()
	s.Handler().ServeHTTP(first, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-cache-health", nil))
	if first.Code != http.StatusOK || !strings.Contains(first.Body.String(), "Starting the bundled Grafana stack on demand") {
		t.Fatalf("first click = %d %q, want progress render", first.Code, first.Body.String())
	}
	waitDashboardState(t, m, "ready")

	ready := httptest.NewRecorder()
	s.Handler().ServeHTTP(ready, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-cache-health", nil))
	if ready.Code != http.StatusSeeOther || ready.Header().Get("Location") != "http://grafana.test/d/fak-cache-health" {
		t.Fatalf("ready click = %d Location %q", ready.Code, ready.Header().Get("Location"))
	}
	if got := probes.Load(); got != 1 {
		t.Fatalf("ready reuse performed %d probes, want one activation probe", got)
	}
}

func TestRichDashboardConcurrentClicksDeduplicateStart(t *testing.T) {
	m := newRichDashboardManager()
	defer m.close()
	m.compose = "test-compose.yml"
	m.baseURL = ""
	m.start = func(context.Context, string) error { return nil }
	var probes atomic.Int32
	m.probe = func(context.Context, string) error {
		if probes.Add(1) == 1 {
			return errors.New("not ready before start")
		}
		return nil
	}
	// Bypass environment/host discovery so this test witnesses the manager's
	// single-flight transition independently of Docker availability.
	m.dockerAvailable = func() bool { return true }

	var starts atomic.Int32
	m.start = func(context.Context, string) error { starts.Add(1); time.Sleep(20 * time.Millisecond); return nil }

	const callers = 8
	done := make(chan struct{}, callers)
	for i := 0; i < callers; i++ {
		go func() { m.ensure(); done <- struct{}{} }()
	}
	for i := 0; i < callers; i++ {
		<-done
	}
	waitDashboardState(t, m, "ready")
	if got := starts.Load(); got != 1 {
		t.Fatalf("%d concurrent clicks started stack %d times, want 1", callers, got)
	}
}

func TestRichDashboardFailureAndDisabledRenders(t *testing.T) {
	for _, tc := range []struct {
		name, state, reason, want string
	}{
		{"disabled", "disabled", "Rich dashboards are disabled by FAK_DASHBOARDS. The lightweight live dashboard remains available.", "lightweight live dashboard remains available"},
		{"unavailable", "unavailable", "Docker is not available. Install/start Docker, or set FAK_GRAFANA_URL.", "Docker is not available"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newRichDashboardManager()
			defer m.close()
			m.state, m.reason = tc.state, tc.reason
			s := testServerWithRichDashboards(t, m)
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/?dashboard=rich", nil))
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), tc.want) || !strings.Contains(rec.Body.String(), "Return to the lightweight live dashboard") {
				t.Fatalf("captured %s render = %d %q", tc.name, rec.Code, rec.Body.String())
			}
		})
	}
}

func TestRichDashboardRejectsUnsafeOverrideAndUnknownDestination(t *testing.T) {
	for _, raw := range []string{"javascript:alert(1)", "http://user:secret@example.test", "//example.test"} {
		if _, err := safeDashboardBaseURL(raw); err == nil {
			t.Errorf("safeDashboardBaseURL(%q) accepted unsafe URL", raw)
		}
	}
	m := newRichDashboardManager()
	defer m.close()
	s := testServerWithRichDashboards(t, m)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=../../escape", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown destination = %d, want 400", rec.Code)
	}
}

func TestRichDashboardConfiguredURLOverrideSkipsDocker(t *testing.T) {
	t.Setenv("FAK_GRAFANA_URL", "https://grafana.example.test/team/")
	m := newRichDashboardManager()
	defer m.close()
	m.probe = func(context.Context, string) error { return nil }
	m.ensure()
	waitDashboardState(t, m, "ready")
	destination, err := richDashboardDestination(m.snapshot().URL, "fak-fleet-overview")
	if err != nil || destination != "https://grafana.example.test/team/d/fak-fleet-overview" {
		t.Fatalf("configured destination = %q, %v", destination, err)
	}
}
func TestRichDashboardCloseStopsOnlyOwnedStack(t *testing.T) {
	m := newRichDashboardManager()
	m.owned, m.compose = true, "compose.yml"
	var stops atomic.Int32
	m.stop = func(context.Context, string) error { stops.Add(1); return nil }
	m.close()
	m.close()
	if got := stops.Load(); got != 1 {
		t.Fatalf("close stopped owned stack %d times, want 1", got)
	}

	external := newRichDashboardManager()
	external.baseURL = "http://grafana.test"
	external.stop = func(context.Context, string) error { return errors.New("must not stop external Grafana") }
	external.close()
}

func testServerWithRichDashboards(t *testing.T, m *richDashboardManager) *Server {
	t.Helper()
	s, err := New(Config{EngineID: "mock", Model: "m", Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	s.richDashboards.close()
	s.richDashboards = m
	return s
}

func waitDashboardState(t *testing.T, m *richDashboardManager, want string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if m.snapshot().State == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("dashboard state = %q, want %q", m.snapshot().State, want)
}

func TestRichDashboardCatalogAuditRejectsDuplicateAndIncompleteRoutes(t *testing.T) {
	if err := auditRichDashboardLinks(richDashboardLinks); err != nil {
		t.Fatalf("shipped catalog: %v", err)
	}
	duplicate := append([]richDashboardLink(nil), richDashboardLinks...)
	duplicate = append(duplicate, duplicate[0])
	if err := auditRichDashboardLinks(duplicate); err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("duplicate audit = %v, want duplicated refusal", err)
	}
	incomplete := append([]richDashboardLink(nil), richDashboardLinks...)
	incomplete[0].Description = ""
	if err := auditRichDashboardLinks(incomplete); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("incomplete audit = %v, want incomplete refusal", err)
	}
}

func TestRichDashboardEveryCatalogClickPreservesSelectedDestination(t *testing.T) {
	for _, dashboard := range richDashboardLinks {
		t.Run(dashboard.UID, func(t *testing.T) {
			m := newRichDashboardManager()
			m.state, m.baseURL = "ready", "https://grafana.example.test/team"
			s := &Server{richDashboards: m}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid="+dashboard.UID, nil)
			if !s.handleRichDashboard(rec, req) {
				t.Fatal("rich dashboard request was not handled")
			}
			want := "https://grafana.example.test/team/d/" + dashboard.UID
			if rec.Code != http.StatusSeeOther || rec.Header().Get("Location") != want {
				t.Fatalf("click %q = (%d, %q), want (303, %q)", dashboard.UID, rec.Code, rec.Header().Get("Location"), want)
			}
		})
	}
}

func TestRichDashboardReusesReadyBundledStackWithoutOwnership(t *testing.T) {
	t.Setenv("FAK_DASHBOARDS", "")
	t.Setenv("FAK_GRAFANA_URL", "")
	t.Setenv("FAK_GRAFANA_COMPOSE", "compose.yml")
	m := newRichDashboardManager()
	startCalls, stopCalls := 0, 0
	m.probe = func(context.Context, string) error { return nil }
	m.dockerAvailable = func() bool { t.Fatal("ready stack should not require Docker discovery"); return false }
	m.start = func(context.Context, string) error { startCalls++; return nil }
	m.stop = func(context.Context, string) error { stopCalls++; return nil }

	if got := m.ensure(); got.State != "starting" {
		t.Fatalf("first state = %q, want starting", got.State)
	}
	deadline := time.Now().Add(time.Second)
	for m.snapshot().State != "ready" && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	got := m.snapshot()
	if got.URL != "http://localhost:3000" {
		t.Fatalf("URL = %q, want canonical bundled endpoint", got.URL)
	}
	if startCalls != 0 {
		t.Fatalf("start calls = %d, want 0", startCalls)
	}
	m.close()
	if stopCalls != 0 {
		t.Fatalf("stop calls = %d, want 0 for pre-existing stack", stopCalls)
	}
}
