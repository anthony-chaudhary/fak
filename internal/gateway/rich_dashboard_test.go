package gateway

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func TestRichDashboardServerSurvivesPriorABIReset(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	t.Cleanup(func() { abi.RegisterEngine("mock", engine.MockEngine) })

	m := newRichDashboardManager(RichDashboardConfig{})
	defer m.close()
	s := testServerWithRichDashboards(t, m)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("dashboard after ABI reset = %d, want 200", rec.Code)
	}
}

func TestRichDashboardProbeClientHasTimeout(t *testing.T) {
	if got := richDashboardProbeClient.Timeout; got != richDashboardProbeTimeout || got <= 0 {
		t.Fatalf("Grafana probe client timeout = %s, want %s", got, richDashboardProbeTimeout)
	}
}

func TestRichDashboardDormantUntilFirstClickThenRedirects(t *testing.T) {
	m := newRichDashboardManager(RichDashboardConfig{})
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
	body := first.Body.String()
	if first.Code != http.StatusOK ||
		!strings.Contains(body, "FAK does not start Docker Desktop") ||
		!strings.Contains(body, "stops with the gateway") ||
		!strings.Contains(body, "adopted and left running") ||
		!strings.Contains(body, "After a Docker or host restart") {
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
	m := newRichDashboardManager(RichDashboardConfig{})
	defer m.close()
	m.compose = "test-compose.yml"
	m.baseURL = ""
	m.listenerAddress = func() string { return "127.0.0.1:61666" }
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
	m.start = func(context.Context, string, string) (richDashboardStack, error) {
		starts.Add(1)
		time.Sleep(20 * time.Millisecond)
		return richDashboardStack{composePath: "test-compose.yml"}, nil
	}
	m.stop = func(context.Context, richDashboardStack) error { return nil }

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

func TestRichDashboardCloseDuringActivationStopsLateOwnedStack(t *testing.T) {
	m := newRichDashboardManager(RichDashboardConfig{})
	m.compose = "compose.yml"
	m.listenerAddress = func() string { return "127.0.0.1:61666" }
	m.dockerAvailable = func() bool { return true }
	var probes atomic.Int32
	m.probe = func(context.Context, string) error {
		if probes.Add(1) == 1 {
			return errors.New("not ready before start")
		}
		return nil
	}
	started := make(chan struct{})
	release := make(chan struct{})
	m.start = func(context.Context, string, string) (richDashboardStack, error) {
		close(started)
		<-release
		return richDashboardStack{composePath: "compose.yml"}, nil
	}
	stopped := make(chan struct{}, 1)
	m.stop = func(context.Context, richDashboardStack) error {
		stopped <- struct{}{}
		return nil
	}

	m.ensure()
	<-started
	m.close()
	close(release)
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("stack that finished starting after gateway close was not stopped")
	}
	if m.snapshot().State == "ready" {
		t.Fatal("closed manager accepted a late ready transition")
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
			m := newRichDashboardManager(RichDashboardConfig{})
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
	m := newRichDashboardManager(RichDashboardConfig{})
	defer m.close()
	s := testServerWithRichDashboards(t, m)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=../../escape", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown destination = %d, want 400", rec.Code)
	}
}

func TestRichDashboardConfiguredURLOverrideSkipsDocker(t *testing.T) {
	m := newRichDashboardManager(RichDashboardConfig{BaseURL: "https://grafana.example.test/team/"})
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
	m := newRichDashboardManager(RichDashboardConfig{})
	m.owned = true
	m.stack = richDashboardStack{composePath: "compose.yml"}
	var stops atomic.Int32
	m.stop = func(context.Context, richDashboardStack) error { stops.Add(1); return nil }
	m.close()
	m.close()
	if got := stops.Load(); got != 1 {
		t.Fatalf("close stopped owned stack %d times, want 1", got)
	}

	external := newRichDashboardManager(RichDashboardConfig{})
	external.baseURL = "http://grafana.test"
	external.stop = func(context.Context, richDashboardStack) error { return errors.New("must not stop external Grafana") }
	external.close()
}

func testServerWithRichDashboards(t *testing.T, m *richDashboardManager) *Server {
	t.Helper()
	// Tests in this package assemble isolated ABI registries with ResetForTest.
	// Re-register the dashboard fixture's dependency at its construction seam so
	// test-file and execution order cannot decide whether "mock" is available.
	abi.RegisterEngine("mock", engine.MockEngine)
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
			m := newRichDashboardManager(RichDashboardConfig{})
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
	m := newRichDashboardManager(RichDashboardConfig{})
	startCalls, stopCalls := 0, 0
	m.probe = func(context.Context, string) error { return nil }
	m.dockerAvailable = func() bool { t.Fatal("ready stack should not require Docker discovery"); return false }
	m.start = func(context.Context, string, string) (richDashboardStack, error) {
		startCalls++
		return richDashboardStack{}, nil
	}
	m.stop = func(context.Context, richDashboardStack) error { stopCalls++; return nil }

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

func TestRichDashboardOwnedStartUsesServerBoundGatewayAddress(t *testing.T) {
	s := &Server{}
	s.installRichDashboardManager(RichDashboardConfig{})
	m := s.richDashboards
	defer m.close()
	m.compose = "compose.yml"
	m.dockerAvailable = func() bool { return true }
	var probes atomic.Int32
	m.probe = func(context.Context, string) error {
		if probes.Add(1) == 1 {
			return errors.New("not ready before start")
		}
		return nil
	}
	var gotAddress string
	m.start = func(_ context.Context, compose, listenerAddress string) (richDashboardStack, error) {
		gotAddress = listenerAddress
		return richDashboardStack{composePath: compose}, nil
	}
	m.stop = func(context.Context, richDashboardStack) error { return nil }

	bound := "127.0.0.1:61666"
	s.boundAddr.Store(&bound)
	m.ensure()
	waitDashboardState(t, m, "ready")
	if gotAddress != bound {
		t.Fatalf("owned start gateway address = %q, want actual bound address %q", gotAddress, bound)
	}
}

func TestPrepareBundledGrafanaStackUsesLivePortWithoutMutatingTemplate(t *testing.T) {
	dir := t.TempDir()
	compose := filepath.Join(dir, "docker-compose.yml")
	composeText := "services:\n  prometheus:\n    volumes:\n      - \"" + bundledPrometheusConfigMount + ":/etc/prometheus/prometheus.yml:ro\"\n"
	if err := os.WriteFile(compose, []byte(composeText), 0o644); err != nil {
		t.Fatal(err)
	}
	templatePath := filepath.Join(dir, "prometheus.yml")
	templateText := "scrape_configs:\n  - job_name: fak_gateway\n    static_configs:\n      - targets: [\"host.docker.internal:8080\"]\n"
	if err := os.WriteFile(templatePath, []byte(templateText), 0o644); err != nil {
		t.Fatal(err)
	}

	stack, err := prepareBundledGrafanaStack(compose, "127.0.0.1:61666")
	if err != nil {
		t.Fatal(err)
	}
	generated, err := os.ReadFile(stack.prometheusConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(generated); !strings.Contains(got, `targets: ["host.docker.internal:61666"]`) || strings.Contains(got, `targets: ["host.docker.internal:8080"]`) {
		t.Fatalf("generated Prometheus config did not select live gateway port: %s", got)
	}
	tracked, err := os.ReadFile(templatePath)
	if err != nil {
		t.Fatal(err)
	}
	if string(tracked) != templateText {
		t.Fatalf("tracked Prometheus template mutated:\n%s", tracked)
	}
	if stack.prometheusConfigPath == templatePath || filepath.Dir(stack.prometheusConfigPath) == dir {
		t.Fatalf("generated config %q must be isolated from Compose source %q", stack.prometheusConfigPath, dir)
	}
	if err := cleanupBundledGrafanaStack(stack); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stack.prometheusConfigPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("generated config remains after cleanup: %v", err)
	}
	if _, err := os.Stat(stack.tempDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("temporary directory remains after cleanup: %v", err)
	}
}

func TestBundledGrafanaFilesWireGeneratedConfigMount(t *testing.T) {
	compose := findBundledGrafanaCompose()
	if compose == "" {
		t.Fatal("bundled Grafana Compose file not found")
	}
	composeBytes, err := os.ReadFile(compose)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(composeBytes), bundledPrometheusConfigMount) {
		t.Fatalf("bundled Compose does not mount generated Prometheus config through %s", bundledPrometheusConfigMount)
	}
	stack, err := prepareBundledGrafanaStack(compose, "127.0.0.1:61666")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cleanupBundledGrafanaStack(stack); err != nil {
			t.Error(err)
		}
	}()
	generated, err := os.ReadFile(stack.prometheusConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(generated), `targets: ["host.docker.internal:61666"]`) {
		t.Fatalf("bundled generated config did not use live gateway port:\n%s", generated)
	}
}

func TestBundledPrometheusTargetUsesOnlyValidatedListenerPort(t *testing.T) {
	for _, addr := range []string{"127.0.0.1:61666", "[::1]:61666", "0.0.0.0:61666"} {
		got, err := bundledPrometheusTargetForListener(addr)
		if err != nil {
			t.Fatalf("target(%q): %v", addr, err)
		}
		if got != "host.docker.internal:61666" {
			t.Fatalf("target(%q) = %q, want Docker host alias with live port", addr, got)
		}
	}
	for _, addr := range []string{"", "127.0.0.1", "127.0.0.1:not-a-port", "127.0.0.1:70000"} {
		if _, err := bundledPrometheusTargetForListener(addr); err == nil {
			t.Fatalf("target(%q) accepted unavailable/invalid listener", addr)
		}
	}
}

func TestDashboardComposeEnvOverridesStaleConfigPath(t *testing.T) {
	got := dashboardComposeEnv([]string{
		"A=1",
		"fak_prometheus_config=stale.yml",
		"B=2",
	}, `C:\Temp\fak-grafana\prometheus.yml`)
	var matches []string
	for _, entry := range got {
		if strings.HasPrefix(strings.ToUpper(entry), bundledPrometheusConfigEnv+"=") {
			matches = append(matches, entry)
		}
	}
	if len(matches) != 1 || matches[0] != `FAK_PROMETHEUS_CONFIG=C:\Temp\fak-grafana\prometheus.yml` {
		t.Fatalf("Compose env config entries = %#v, want one current path", matches)
	}
}

func TestRichDashboardKeyPreservation(t *testing.T) {
	t.Run("with key preserves query and displays ssh hint on unavailable", func(t *testing.T) {
		m := newRichDashboardManager(RichDashboardConfig{})
		defer m.close()
		m.state, m.reason = "unavailable", "Docker is not available."
		s := testServerWithRichDashboards(t, m)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-gateway-observability&key=valid-secret", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		body := rec.Body.String()
		if !strings.Contains(body, `href="/?key=valid-secret"`) {
			t.Fatalf("body missing key in return link: %s", body)
		}
		if !strings.Contains(body, "ssh -L 8080:localhost:8080 -L 3000:localhost:3000") {
			t.Fatalf("body missing ssh hint: %s", body)
		}
	})

	t.Run("without key renders clean root link", func(t *testing.T) {
		m := newRichDashboardManager(RichDashboardConfig{})
		defer m.close()
		m.state, m.reason = "unavailable", "Docker is not available."
		s := testServerWithRichDashboards(t, m)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/?dashboard=rich", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		body := rec.Body.String()
		if !strings.Contains(body, `href="/"`) {
			t.Fatalf("body missing clean root link: %s", body)
		}
		if strings.Contains(body, `href="/?`) {
			t.Fatalf("body contains unexpected query param in return link: %s", body)
		}
	})

	t.Run("starting state preserves key", func(t *testing.T) {
		m := newRichDashboardManager(RichDashboardConfig{})
		defer m.close()
		m.state = "starting"
		s := testServerWithRichDashboards(t, m)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&key=token-456", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		body := rec.Body.String()
		if !strings.Contains(body, `href="/?key=token-456"`) {
			t.Fatalf("body missing key in return link: %s", body)
		}
	})

	t.Run("with auth header preserves key", func(t *testing.T) {
		m := newRichDashboardManager(RichDashboardConfig{})
		defer m.close()
		m.state, m.reason = "unavailable", "Docker is not available."
		s := testServerWithRichDashboards(t, m)
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/?dashboard=rich", nil)
		req.Header.Set("Authorization", "Bearer bearer-secret")
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		body := rec.Body.String()
		if !strings.Contains(body, `href="/?key=bearer-secret"`) {
			t.Fatalf("body missing key in return link from bearer token: %s", body)
		}
	})
}

func TestRichDashboardClientBaseURL(t *testing.T) {
	const testUID = "fak-cache-health"

	t.Run("redirects with client dialed host and Grafana port", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			host        string
			tlsReq      bool
			forwarded   string
			grafanaBase string
			wantLoc     string
		}{
			{
				name:    "non-loopback IPv4 host",
				host:    "192.168.1.200:8080",
				wantLoc: "http://192.168.1.200:3000/d/" + testUID,
			},
			{
				name:    "non-loopback DNS host",
				host:    "strix-halo-fak.local:8080",
				wantLoc: "http://strix-halo-fak.local:3000/d/" + testUID,
			},
			{
				name:    "loopback localhost",
				host:    "localhost:8080",
				wantLoc: "http://localhost:3000/d/" + testUID,
			},
			{
				name:    "loopback IPv4 127.0.0.1",
				host:    "127.0.0.1:8080",
				wantLoc: "http://localhost:3000/d/" + testUID,
			},
			{
				name:    "loopback IPv6 [::1]",
				host:    "[::1]:8080",
				wantLoc: "http://localhost:3000/d/" + testUID,
			},
			{
				name:        "explicit FAK_GRAFANA_URL preserves external endpoint",
				host:        "192.168.1.200:8080",
				grafanaBase: "https://grafana.corp.internal",
				wantLoc:     "https://grafana.corp.internal/d/" + testUID,
			},
			{
				name:    "TLS request redirects to https",
				host:    "192.168.1.200:8080",
				tlsReq:  true,
				wantLoc: "https://192.168.1.200:3000/d/" + testUID,
			},
			{
				name:      "X-Forwarded-Proto https redirects to https",
				host:      "192.168.1.200:8080",
				forwarded: "https",
				wantLoc:   "https://192.168.1.200:3000/d/" + testUID,
			},
			{
				name:    "missing empty host falls back to loopback base",
				host:    "",
				wantLoc: "http://localhost:3000/d/" + testUID,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				m := newRichDashboardManager(RichDashboardConfig{BaseURL: tc.grafanaBase})
				defer m.close()
				m.state = "ready"
				if tc.grafanaBase != "" {
					m.baseURL = tc.grafanaBase
				} else {
					m.baseURL = bundledGrafanaURL
				}
				s := &Server{richDashboards: m}

				rec := httptest.NewRecorder()
				req := httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid="+testUID, nil)
				req.Host = tc.host
				if tc.tlsReq {
					req.TLS = &tls.ConnectionState{}
				}
				if tc.forwarded != "" {
					req.Header.Set("X-Forwarded-Proto", tc.forwarded)
				}

				if !s.handleRichDashboard(rec, req) {
					t.Fatal("handleRichDashboard returned false, want true")
				}
				if rec.Code != http.StatusSeeOther {
					t.Fatalf("status = %d, want 303 See Other", rec.Code)
				}
				if got := rec.Header().Get("Location"); got != tc.wantLoc {
					t.Fatalf("Location = %q, want %q", got, tc.wantLoc)
				}
			})
		}
	})

	t.Run("helper level unit verification", func(t *testing.T) {
		// Verify isLoopbackHost
		loopbacks := []string{
			"localhost", "localhost.", "LocalHost", "127.0.0.1", "127.0.0.2", "127.255.255.255",
			"::1", "[::1]", "[::1]:8080", "127.0.0.1:8080", "localhost:8080",
		}
		for _, h := range loopbacks {
			if !isLoopbackHost(h) {
				t.Errorf("isLoopbackHost(%q) = false, want true", h)
			}
		}
		nonLoopbacks := []string{
			"192.168.1.200", "192.168.1.200:8080", "strix-halo-fak.local", "strix-halo-fak.local:8080",
			"grafana.corp.internal", "2001:db8::1", "[2001:db8::1]", "[2001:db8::1]:3000", "",
		}
		for _, h := range nonLoopbacks {
			if isLoopbackHost(h) {
				t.Errorf("isLoopbackHost(%q) = true, want false", h)
			}
		}

		// Verify stripPort
		stripPortCases := []struct {
			in   string
			want string
		}{
			{"192.168.1.200:8080", "192.168.1.200"},
			{"192.168.1.200", "192.168.1.200"},
			{"[::1]:8080", "::1"},
			{"[::1]", "::1"},
			{"::1", "::1"},
			{"[2001:db8::1]:3000", "2001:db8::1"},
			{"[2001:db8::1]", "2001:db8::1"},
			{"strix-halo-fak.local:8080", "strix-halo-fak.local"},
			{"strix-halo-fak.local", "strix-halo-fak.local"},
			{"localhost:8080", "localhost"},
			{"localhost", "localhost"},
			{"", ""},
		}
		for _, tc := range stripPortCases {
			if got := stripPort(tc.in); got != tc.want {
				t.Errorf("stripPort(%q) = %q, want %q", tc.in, got, tc.want)
			}
		}

		// Verify clientDashboardBaseURL & clientBaseURL
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Host = "192.168.1.200:8080"
		if got := clientDashboardBaseURL("", req); got != "http://192.168.1.200:3000" {
			t.Fatalf("clientDashboardBaseURL(\"\", req) = %q, want http://192.168.1.200:3000", got)
		}
		if got := clientBaseURL("", req); got != "http://192.168.1.200:3000" {
			t.Fatalf("clientBaseURL(\"\", req) = %q, want http://192.168.1.200:3000", got)
		}

		// Non-loopback IPv6 with brackets
		reqIPv6 := httptest.NewRequest(http.MethodGet, "/", nil)
		reqIPv6.Host = "[2001:db8::1]:8080"
		if got := clientDashboardBaseURL("http://localhost:3000", reqIPv6); got != "http://[2001:db8::1]:3000" {
			t.Fatalf("clientDashboardBaseURL IPv6 = %q, want http://[2001:db8::1]:3000", got)
		}

		// Custom base port retention
		if got := clientDashboardBaseURL("http://localhost:3005", req); got != "http://192.168.1.200:3005" {
			t.Fatalf("clientDashboardBaseURL custom port = %q, want http://192.168.1.200:3005", got)
		}

		// Custom path retention
		if got := clientDashboardBaseURL("http://localhost:3000/grafana/", req); got != "http://192.168.1.200:3000/grafana" {
			t.Fatalf("clientDashboardBaseURL custom path = %q, want http://192.168.1.200:3000/grafana", got)
		}

		// Nil request
		if got := clientDashboardBaseURL("http://localhost:3000", nil); got != "http://localhost:3000" {
			t.Fatalf("clientDashboardBaseURL nil request = %q, want http://localhost:3000", got)
		}

		// Manager clientBaseURL method with nil manager
		var nilMgr *richDashboardManager
		if got := nilMgr.clientBaseURL(req); got != "http://192.168.1.200:3000" {
			t.Fatalf("nilMgr.clientBaseURL = %q, want http://192.168.1.200:3000", got)
		}

		// Manager clientBaseURL method with non-nil manager
		mgr := newRichDashboardManager(RichDashboardConfig{BaseURL: "http://localhost:3000"})
		defer mgr.close()
		if got := mgr.clientBaseURL(req); got != "http://192.168.1.200:3000" {
			t.Fatalf("mgr.clientBaseURL = %q, want http://192.168.1.200:3000", got)
		}
	})
}

func TestGrafanaReverseProxy(t *testing.T) {
	// 1. GET /grafana/api/health proxies to upstream mock Grafana and returns HTTP 200.
	// 3. Headers and query parameters are preserved through the proxy.
	// 4. Upstream receives X-Forwarded-Host, X-Forwarded-Proto, and X-Forwarded-Prefix.
	t.Run("proxies request, preserves headers and query, sets forwarded headers", func(t *testing.T) {
		var (
			recPath            string
			recQuery           string
			recCustomHeader    string
			recForwardedHost   string
			recForwardedProto  string
			recForwardedPrefix string
		)
		mockGrafana := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			recPath = r.URL.Path
			recQuery = r.URL.RawQuery
			recCustomHeader = r.Header.Get("X-Custom-Test-Header")
			recForwardedHost = r.Header.Get("X-Forwarded-Host")
			recForwardedProto = r.Header.Get("X-Forwarded-Proto")
			recForwardedPrefix = r.Header.Get("X-Forwarded-Prefix")

			if r.URL.Path == "/api/health" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"database": "ok"}`))
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		}))
		defer mockGrafana.Close()

		m := newRichDashboardManager(RichDashboardConfig{
			ProxyGrafana:       true,
			ProxyGrafanaPrefix: "/grafana",
			BaseURL:            mockGrafana.URL,
		})
		defer m.close()
		s := testServerWithRichDashboards(t, m)

		// Test 1: GET /grafana/api/health proxies to upstream mock Grafana and returns HTTP 200.
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/grafana/api/health", nil)
		req.Host = "fak.local:8080"
		s.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("GET /grafana/api/health code = %d, want 200", rec.Code)
		}
		if !strings.Contains(rec.Body.String(), `"database": "ok"`) {
			t.Fatalf("unexpected body: %s", rec.Body.String())
		}
		if recPath != "/api/health" {
			t.Fatalf("upstream path = %q, want /api/health", recPath)
		}
		if recForwardedHost != "fak.local:8080" {
			t.Fatalf("X-Forwarded-Host = %q, want fak.local:8080", recForwardedHost)
		}
		if recForwardedProto != "http" {
			t.Fatalf("X-Forwarded-Proto = %q, want http", recForwardedProto)
		}
		if recForwardedPrefix != "/grafana" {
			t.Fatalf("X-Forwarded-Prefix = %q, want /grafana", recForwardedPrefix)
		}

		// Test 3 & 4: Headers and query parameters preserved through the proxy
		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/grafana/api/datasources?query=active&limit=50", nil)
		req.Host = "fak.local:8080"
		req.Header.Set("X-Custom-Test-Header", "test-val-xyz")
		s.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("GET datasources code = %d, want 200", rec.Code)
		}
		if recPath != "/api/datasources" {
			t.Fatalf("upstream path = %q, want /api/datasources", recPath)
		}
		if recQuery != "query=active&limit=50" {
			t.Fatalf("upstream query = %q, want query=active&limit=50", recQuery)
		}
		if recCustomHeader != "test-val-xyz" {
			t.Fatalf("upstream custom header = %q, want test-val-xyz", recCustomHeader)
		}
	})

	// 2. GET /?dashboard=rich&uid=fak-gateway-observability redirects with HTTP 303 to relative path /grafana/d/fak-gateway-observability
	t.Run("redirects with relative path when proxy is enabled", func(t *testing.T) {
		m := newRichDashboardManager(RichDashboardConfig{
			ProxyGrafana:       true,
			ProxyGrafanaPrefix: "/grafana",
		})
		defer m.close()
		m.state = "ready"
		s := testServerWithRichDashboards(t, m)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-gateway-observability", nil)
		req.Host = "192.168.1.208:8080"

		if !s.handleRichDashboard(rec, req) {
			t.Fatal("handleRichDashboard returned false, want true")
		}
		if rec.Code != http.StatusSeeOther {
			t.Fatalf("status = %d, want 303 See Other", rec.Code)
		}
		wantLoc := "/grafana/d/fak-gateway-observability"
		if got := rec.Header().Get("Location"); got != wantLoc {
			t.Fatalf("Location = %q, want %q", got, wantLoc)
		}

		// Also verify through s.Handler()
		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-gateway-observability", nil)
		req.Host = "192.168.1.208:8080"
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusSeeOther {
			t.Fatalf("status via Handler = %d, want 303 See Other", rec.Code)
		}
		if got := rec.Header().Get("Location"); got != wantLoc {
			t.Fatalf("Location via Handler = %q, want %q", got, wantLoc)
		}
	})

	// 5. WebSocket upgrade compatibility test (HTTP 101 Switching Protocols via standard library http.Hijacker)
	t.Run("websocket upgrade compatibility", func(t *testing.T) {
		backendDone := make(chan struct{})
		mockBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
				http.Error(w, "expected websocket upgrade", http.StatusBadRequest)
				return
			}
			w.Header().Set("Connection", "Upgrade")
			w.Header().Set("Upgrade", "websocket")
			w.WriteHeader(http.StatusSwitchingProtocols)
			conn, bufrw, err := http.NewResponseController(w).Hijack()
			if err != nil {
				t.Errorf("backend Hijack: %v", err)
				return
			}
			defer conn.Close()

			line, err := bufrw.ReadString('\n')
			if err == nil {
				_, _ = bufrw.WriteString("echo: " + line)
				_ = bufrw.Flush()
			}
			close(backendDone)
		}))
		defer mockBackend.Close()

		m := newRichDashboardManager(RichDashboardConfig{
			ProxyGrafana:       true,
			ProxyGrafanaPrefix: "/grafana",
			BaseURL:            mockBackend.URL,
		})
		defer m.close()
		s := testServerWithRichDashboards(t, m)

		gwServer := httptest.NewServer(s.Handler())
		defer gwServer.Close()

		req, err := http.NewRequest(http.MethodGet, gwServer.URL+"/grafana/api/live/ws", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Connection", "Upgrade")
		req.Header.Set("Upgrade", "websocket")

		resp, err := gwServer.Client().Do(req)
		if err != nil {
			t.Fatalf("client Do failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusSwitchingProtocols {
			t.Fatalf("status = %d, want 101 Switching Protocols", resp.StatusCode)
		}
		if got := resp.Header.Get("Upgrade"); !strings.EqualFold(got, "websocket") {
			t.Fatalf("Upgrade header = %q, want websocket", got)
		}

		rwc, ok := resp.Body.(io.ReadWriteCloser)
		if !ok {
			t.Fatalf("resp.Body type %T does not implement io.ReadWriteCloser", resp.Body)
		}
		if _, err := io.WriteString(rwc, "hello ws\n"); err != nil {
			t.Fatalf("write to upgraded body failed: %v", err)
		}
		r := bufio.NewReader(rwc)
		line, err := r.ReadString('\n')
		if err != nil {
			t.Fatalf("read from upgraded body failed: %v", err)
		}
		if !strings.Contains(line, "echo: hello ws") {
			t.Fatalf("echo message = %q, want 'echo: hello ws'", line)
		}
		<-backendDone
	})

	// 6. Default/fallback verification: ProxyGrafana: false preserves standard non-proxied redirects.
	t.Run("fallback when proxy is disabled", func(t *testing.T) {
		m := newRichDashboardManager(RichDashboardConfig{
			ProxyGrafana: false,
		})
		defer m.close()
		m.state = "ready"
		s := testServerWithRichDashboards(t, m)

		// Non-proxied redirect points to client authority with Grafana port 3000
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-gateway-observability", nil)
		req.Host = "192.168.1.208:8080"
		s.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusSeeOther {
			t.Fatalf("redirect code = %d, want 303 See Other", rec.Code)
		}
		wantLoc := "http://192.168.1.208:3000/d/fak-gateway-observability"
		if got := rec.Header().Get("Location"); got != wantLoc {
			t.Fatalf("Location = %q, want %q", got, wantLoc)
		}

		// /grafana/ endpoint is not mounted when ProxyGrafana is false
		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/grafana/api/health", nil)
		s.Handler().ServeHTTP(rec, req)
		if rec.Code == http.StatusOK && strings.Contains(rec.Body.String(), "database") {
			t.Fatalf("unexpected proxy match when ProxyGrafana is disabled")
		}
	})

	// Loopback vs remote authExempt behavior with RequireKey
	t.Run("authExempt behavior with requireKey", func(t *testing.T) {
		mockGrafana := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		}))
		defer mockGrafana.Close()

		m := newRichDashboardManager(RichDashboardConfig{
			ProxyGrafana:       true,
			ProxyGrafanaPrefix: "/grafana",
			BaseURL:            mockGrafana.URL,
		})
		defer m.close()

		abi.RegisterEngine("mock", engine.MockEngine)
		s, err := New(Config{
			EngineID:   "mock",
			Model:      "m",
			Provider:   "openai",
			RequireKey: "secret-bearer-key",
		})
		if err != nil {
			t.Fatal(err)
		}
		s.richDashboards.close()
		s.richDashboards = m

		// Loopback request without auth header succeeds
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/grafana/api/health", nil)
		req.RemoteAddr = "127.0.0.1:54321"
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("loopback request code = %d, want 200 OK", rec.Code)
		}

		// Non-loopback request without auth header is rejected (401)
		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/grafana/api/health", nil)
		req.RemoteAddr = "192.168.1.100:54321"
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("remote unauthenticated request code = %d, want 401 Unauthorized", rec.Code)
		}

		// Non-loopback request with auth header succeeds
		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/grafana/api/health", nil)
		req.RemoteAddr = "192.168.1.100:54321"
		req.Header.Set("Authorization", "Bearer secret-bearer-key")
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("remote authenticated request code = %d, want 200 OK", rec.Code)
		}
	})
}

func TestRichDashboardApplianceCatalog(t *testing.T) {
	// 1. When initialized with appliance catalog profile (ApplianceProfile: true),
	// the manager catalog has length 6 and matches ApplianceDashboardCatalog().
	mApp := newRichDashboardManager(RichDashboardConfig{
		ApplianceProfile: true,
		BaseURL:          "http://grafana.test",
	})
	defer mApp.close()
	mApp.state = "ready"

	catalog := mApp.catalog()
	expected := ApplianceDashboardCatalog()
	if len(catalog) != 6 {
		t.Fatalf("catalog len = %d, want 6", len(catalog))
	}
	if len(mApp.links) != 6 {
		t.Fatalf("mApp.links len = %d, want 6", len(mApp.links))
	}
	if !reflect.DeepEqual(catalog, expected) {
		t.Fatalf("catalog mismatch:\ngot:  %+v\nwant: %+v", catalog, expected)
	}
	if got := mApp.getDefaultUID(); got != "fak-strix-index" {
		t.Fatalf("getDefaultUID = %q, want fak-strix-index", got)
	}

	sApp := testServerWithRichDashboards(t, mApp)

	// 2. GET /?dashboard=rich redirects (HTTP 303) to /d/fak-strix-index
	recDefault := httptest.NewRecorder()
	sApp.Handler().ServeHTTP(recDefault, httptest.NewRequest(http.MethodGet, "/?dashboard=rich", nil))
	if recDefault.Code != http.StatusSeeOther {
		t.Fatalf("GET /?dashboard=rich code = %d, want 303", recDefault.Code)
	}
	if loc := recDefault.Header().Get("Location"); loc != "http://grafana.test/d/fak-strix-index" {
		t.Fatalf("GET /?dashboard=rich Location = %q, want http://grafana.test/d/fak-strix-index", loc)
	}

	// 3. GET /?dashboard=rich&uid=fak-strix-serving redirects (HTTP 303) to /d/fak-strix-serving
	recServing := httptest.NewRecorder()
	sApp.Handler().ServeHTTP(recServing, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fak-strix-serving", nil))
	if recServing.Code != http.StatusSeeOther {
		t.Fatalf("GET /?dashboard=rich&uid=fak-strix-serving code = %d, want 303", recServing.Code)
	}
	if loc := recServing.Header().Get("Location"); loc != "http://grafana.test/d/fak-strix-serving" {
		t.Fatalf("GET /?dashboard=rich&uid=fak-strix-serving Location = %q, want http://grafana.test/d/fak-strix-serving", loc)
	}

	// 4. Request with dev-only UID like fleet-bottleneck returns HTTP 400 Bad Request
	recDevOnly := httptest.NewRecorder()
	sApp.Handler().ServeHTTP(recDevOnly, httptest.NewRequest(http.MethodGet, "/?dashboard=rich&uid=fleet-bottleneck", nil))
	if recDevOnly.Code != http.StatusBadRequest {
		t.Fatalf("GET /?dashboard=rich&uid=fleet-bottleneck code = %d, want 400", recDevOnly.Code)
	}

	// Verify homepage HTML rendering with appliance profile renders 6 dashboards
	recHome := httptest.NewRecorder()
	sApp.Handler().ServeHTTP(recHome, httptest.NewRequest(http.MethodGet, "/", nil))
	if recHome.Code != http.StatusOK {
		t.Fatalf("GET / code = %d, want 200", recHome.Code)
	}
	if !strings.Contains(recHome.Body.String(), "on-demand · 6 dashboards.") {
		t.Fatalf("homepage body missing 'on-demand · 6 dashboards.': %s", recHome.Body.String())
	}
	if !strings.Contains(recHome.Body.String(), "data-dashboard-uid=\"fak-strix-index\"") {
		t.Fatalf("homepage body missing fak-strix-index link")
	}

	// 5. Dev mode with default config preserves fak-gateway-observability as default redirect and includes the 9 generic dashboards without regression
	mDev := newRichDashboardManager(RichDashboardConfig{
		BaseURL: "http://grafana.test",
	})
	defer mDev.close()
	mDev.state = "ready"

	if len(mDev.catalog()) != 9 {
		t.Fatalf("dev mode catalog len = %d, want 9", len(mDev.catalog()))
	}
	if got := mDev.getDefaultUID(); got != "fak-gateway-observability" {
		t.Fatalf("dev mode getDefaultUID = %q, want fak-gateway-observability", got)
	}
	if !mDev.hasUID("fleet-bottleneck") {
		t.Fatalf("dev mode should have fleet-bottleneck")
	}
	if mDev.hasUID("fak-strix-index") {
		t.Fatalf("dev mode should not have fak-strix-index")
	}

	sDev := testServerWithRichDashboards(t, mDev)
	recDevDefault := httptest.NewRecorder()
	sDev.Handler().ServeHTTP(recDevDefault, httptest.NewRequest(http.MethodGet, "/?dashboard=rich", nil))
	if recDevDefault.Code != http.StatusSeeOther {
		t.Fatalf("dev GET /?dashboard=rich code = %d, want 303", recDevDefault.Code)
	}
	if loc := recDevDefault.Header().Get("Location"); loc != "http://grafana.test/d/fak-gateway-observability" {
		t.Fatalf("dev GET /?dashboard=rich Location = %q, want http://grafana.test/d/fak-gateway-observability", loc)
	}

	recDevHome := httptest.NewRecorder()
	sDev.Handler().ServeHTTP(recDevHome, httptest.NewRequest(http.MethodGet, "/", nil))
	if !strings.Contains(recDevHome.Body.String(), "on-demand · 9 dashboards.") {
		t.Fatalf("dev homepage missing 'on-demand · 9 dashboards.': %s", recDevHome.Body.String())
	}

	// 6. Custom catalog with custom default UID works as expected
	customCatalog := []RichDashboardLink{
		{UID: "custom-alpha", Title: "Custom Alpha", Description: "Alpha desc", Category: "custom"},
		{UID: "custom-beta", Title: "Custom Beta", Description: "Beta desc", Category: "custom"},
	}
	mCustom := newRichDashboardManager(RichDashboardConfig{
		Catalog:    customCatalog,
		DefaultUID: "custom-beta",
		BaseURL:    "http://grafana.test",
	})
	defer mCustom.close()
	mCustom.state = "ready"

	if len(mCustom.catalog()) != 2 {
		t.Fatalf("custom catalog len = %d, want 2", len(mCustom.catalog()))
	}
	if got := mCustom.getDefaultUID(); got != "custom-beta" {
		t.Fatalf("custom getDefaultUID = %q, want custom-beta", got)
	}
	if !mCustom.hasUID("custom-alpha") || !mCustom.hasUID("custom-beta") {
		t.Fatalf("custom manager missing custom UIDs")
	}
	if mCustom.hasUID("fak-strix-index") {
		t.Fatalf("custom manager should not have fak-strix-index")
	}

	sCustom := testServerWithRichDashboards(t, mCustom)
	recCustom := httptest.NewRecorder()
	sCustom.Handler().ServeHTTP(recCustom, httptest.NewRequest(http.MethodGet, "/?dashboard=rich", nil))
	if recCustom.Code != http.StatusSeeOther {
		t.Fatalf("custom GET /?dashboard=rich code = %d, want 303", recCustom.Code)
	}
	if loc := recCustom.Header().Get("Location"); loc != "http://grafana.test/d/custom-beta" {
		t.Fatalf("custom GET /?dashboard=rich Location = %q, want http://grafana.test/d/custom-beta", loc)
	}

	// Custom catalog with empty DefaultUID falls back to first entry
	mCustomFirst := newRichDashboardManager(RichDashboardConfig{
		Catalog: customCatalog,
	})
	defer mCustomFirst.close()
	if got := mCustomFirst.getDefaultUID(); got != "custom-alpha" {
		t.Fatalf("custom manager with empty DefaultUID = %q, want custom-alpha", got)
	}
}
