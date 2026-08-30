package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
