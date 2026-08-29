package gateway

import (
	"context"
	"errors"
	"fmt"
	"html/template"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dockerprocess"
)

const (
	richDashboardDefaultUID         = "fak-gateway-observability"
	richDashboardTimeout            = 90 * time.Second
	richDashboardProbeTimeout       = 5 * time.Second
	bundledGrafanaURL               = "http://localhost:3000"
	bundledPrometheusConfigEnv      = "FAK_PROMETHEUS_CONFIG"
	bundledPrometheusConfigMount    = "${FAK_PROMETHEUS_CONFIG:-./prometheus.yml}"
	bundledPrometheusTemplateTarget = "host.docker.internal:8080"
)

var richDashboardProbeClient = &http.Client{Timeout: richDashboardProbeTimeout}

var dashboardDockerAvailable = dockerprocess.Available

type richDashboardLink struct {
	UID         string
	Title       string
	Description string
	Category    string
}

var richDashboardLinks = []richDashboardLink{
	{UID: "fleet-bottleneck", Title: "Fleet bottlenecks", Description: "Rank the live fleet constraints that need operator attention.", Category: "debug"},
	{UID: "fak-gateway-observability", Title: "Gateway observability", Description: "Inspect gateway request rate, status mix, latency, and in-flight work.", Category: "debug"},
	{UID: "fak-dogfood-slow-requests", Title: "Slow requests", Description: "Debug slow Claude Code requests through the FAK gateway.", Category: "debug"},
	{UID: "fak-startup-load", Title: "Startup and model load", Description: "Follow readiness and model-loading phases from process start.", Category: "debug"},
	{UID: "fak-guard-adjudication", Title: "Guard adjudication", Description: "See verdicts, refusal classes, and guarded-session economics.", Category: "debug"},
	{UID: "fak-cache-health", Title: "Cache health", Description: "Inspect KV reuse, cache regimes, and provider-cache observations.", Category: "debug"},
	{UID: "fak-cache-value-rollup", Title: "Cache value", Description: "Compare witnessed reuse with observed and projected savings.", Category: "rollup"},
	{UID: "fak-fleet-overview", Title: "Fleet sessions", Description: "See which sessions are live and open their operational roll-ups.", Category: "rollup"},
	{UID: "fak-fleet-session", Title: "Session drill-down", Description: "Inspect one session's liveness, budget, cache posture, and verdicts.", Category: "debug"},
}

var richDashboardUIDs = func() map[string]struct{} {
	out := make(map[string]struct{}, len(richDashboardLinks))
	for _, dashboard := range richDashboardLinks {
		out[dashboard.UID] = struct{}{}
	}
	return out
}()

func auditRichDashboardLinks(links []richDashboardLink) error {
	seen := make(map[string]struct{}, len(links))
	for i, link := range links {
		link.UID = strings.TrimSpace(link.UID)
		if link.UID == "" || strings.TrimSpace(link.Title) == "" || strings.TrimSpace(link.Description) == "" {
			return fmt.Errorf("rich dashboard %d has incomplete route metadata", i)
		}
		if _, exists := seen[link.UID]; exists {
			return fmt.Errorf("rich dashboard UID %q is duplicated", link.UID)
		}
		if strings.ContainsAny(link.UID, "/?#&") {
			return fmt.Errorf("rich dashboard UID %q is not a safe route segment", link.UID)
		}
		seen[link.UID] = struct{}{}
	}
	return nil
}

type richDashboardManager struct {
	mu              sync.Mutex
	ctx             context.Context
	cancel          context.CancelFunc
	state           string
	reason          string
	baseURL         string
	compose         string
	stack           richDashboardStack
	dockerAvailable func() bool
	listenerAddress func() string
	owned           bool
	closed          bool
	start           func(context.Context, string, string) (richDashboardStack, error)
	stop            func(context.Context, richDashboardStack) error
	probe           func(context.Context, string) error
	startedAt       time.Time
}

type richDashboardSnapshot struct {
	State     string
	Reason    string
	URL       string
	StartedAt time.Time
}

type RichDashboardConfig struct {
	BaseURL     string
	ComposePath string
	Disabled    bool
}

type richDashboardStack struct {
	composePath          string
	prometheusConfigPath string
	tempDir              string
}

func newRichDashboardManager(cfg RichDashboardConfig) *richDashboardManager {
	ctx, cancel := context.WithCancel(context.Background())
	m := &richDashboardManager{
		ctx: ctx, cancel: cancel, state: "dormant",
		baseURL: strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/"),
		compose: strings.TrimSpace(cfg.ComposePath),
	}
	m.dockerAvailable = dashboardDockerAvailable
	m.listenerAddress = func() string { return "" }
	m.start = startBundledGrafana
	m.stop = stopBundledGrafana
	m.probe = probeGrafana
	if cfg.Disabled {
		m.state = "disabled"
		m.reason = "Rich dashboards are disabled by explicit gateway configuration. The lightweight live dashboard remains available."
	}
	if m.baseURL != "" {
		if _, err := safeDashboardBaseURL(m.baseURL); err != nil {
			m.state = "unavailable"
			m.reason = "FAK_GRAFANA_URL must be an http(s) URL without credentials."
		}
	}
	return m
}

func safeDashboardBaseURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil {
		return nil, errors.New("unsafe dashboard URL")
	}
	u.RawQuery, u.Fragment = "", ""
	return u, nil
}

func (m *richDashboardManager) ensure() richDashboardSnapshot {
	m.mu.Lock()
	if m.state != "dormant" || m.closed {
		s := m.snapshotLocked()
		m.mu.Unlock()
		return s
	}
	m.state = "starting"
	m.startedAt = time.Now()
	m.mu.Unlock()
	go m.activate()
	return m.snapshot()
}

func (m *richDashboardManager) activate() {
	base := m.baseURL
	compose := m.compose
	owned := false
	var stack richDashboardStack
	if base == "" {
		base = bundledGrafanaURL
		ctx, cancel := context.WithTimeout(m.ctx, 3*time.Second)
		ready := m.probe(ctx, base) == nil
		cancel()
		if ready {
			m.mu.Lock()
			if m.closed {
				m.mu.Unlock()
				return
			}
			m.baseURL, m.owned, m.state, m.reason = base, false, "ready", ""
			m.mu.Unlock()
			return
		}
		if compose == "" {
			compose = findBundledGrafanaCompose()
		}
		if compose == "" {
			m.fail("Bundled Grafana files were not found. Set FAK_GRAFANA_URL to an existing Grafana or run from a fak checkout.")
			return
		}
		if !m.dockerAvailable() {
			m.fail("Docker is not available. FAK does not start Docker Desktop; start Docker, or set FAK_GRAFANA_URL to an existing Grafana.")
			return
		}
		listenerAddress := ""
		if m.listenerAddress != nil {
			listenerAddress = m.listenerAddress()
		}
		ctx, cancel = context.WithTimeout(m.ctx, richDashboardTimeout)
		var err error
		stack, err = m.start(ctx, compose, listenerAddress)
		cancel()
		if err != nil {
			m.fail("Bundled Grafana could not start: " + boundedDashboardError(err))
			return
		}
		owned = true
	}

	deadline := time.Now().Add(richDashboardTimeout)
	for {
		ctx, cancel := context.WithTimeout(m.ctx, 3*time.Second)
		err := m.probe(ctx, base)
		cancel()
		if err == nil {
			m.mu.Lock()
			if m.closed {
				m.mu.Unlock()
				if owned {
					ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
					_ = m.stop(ctx, stack)
					cancel()
				}
				return
			}
			m.baseURL, m.owned, m.compose, m.stack, m.state, m.reason = base, owned, compose, stack, "ready", ""
			m.mu.Unlock()
			return
		}
		if time.Now().After(deadline) || m.ctx.Err() != nil {
			if owned {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				_ = m.stop(ctx, stack)
				cancel()
			}
			m.fail("Grafana did not become ready within 90 seconds. Check Docker and tools/grafana, or set FAK_GRAFANA_URL.")
			return
		}
		time.Sleep(time.Second)
	}
}

func (m *richDashboardManager) fail(reason string) {
	m.mu.Lock()
	m.state, m.reason = "unavailable", reason
	m.mu.Unlock()
}

func (m *richDashboardManager) snapshot() richDashboardSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.snapshotLocked()
}

func (m *richDashboardManager) snapshotLocked() richDashboardSnapshot {
	return richDashboardSnapshot{State: m.state, Reason: m.reason, URL: m.baseURL, StartedAt: m.startedAt}
}

func (m *richDashboardManager) close() {
	m.cancel()
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	m.closed = true
	owned, stack := m.owned, m.stack
	m.owned = false
	m.stack = richDashboardStack{}
	m.mu.Unlock()
	if owned && stack.composePath != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		_ = m.stop(ctx, stack)
		cancel()
	}
}

func findBundledGrafanaCompose() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		candidate := filepath.Join(wd, "tools", "grafana", "docker-compose.yml")
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(wd)
		if parent == wd {
			return ""
		}
		wd = parent
	}
}

func startBundledGrafana(ctx context.Context, compose, listenerAddress string) (richDashboardStack, error) {
	stack, err := prepareBundledGrafanaStack(compose, listenerAddress)
	if err != nil {
		return richDashboardStack{}, err
	}
	out, err := dockerprocess.ComposeCombinedOutput(ctx, filepath.Dir(stack.composePath),
		dashboardComposeEnv(os.Environ(), stack.prometheusConfigPath),
		"-f", stack.composePath, "--profile", "local-prometheus", "up", "-d")
	if err != nil {
		cleanupErr := cleanupBundledGrafanaStack(stack)
		return richDashboardStack{}, errors.Join(fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out))), cleanupErr)
	}
	return stack, nil
}

func stopBundledGrafana(ctx context.Context, stack richDashboardStack) error {
	err := dockerprocess.ComposeRun(ctx, filepath.Dir(stack.composePath),
		dashboardComposeEnv(os.Environ(), stack.prometheusConfigPath),
		"-f", stack.composePath, "down")
	return errors.Join(err, cleanupBundledGrafanaStack(stack))
}

func prepareBundledGrafanaStack(compose, listenerAddress string) (richDashboardStack, error) {
	compose, err := filepath.Abs(strings.TrimSpace(compose))
	if err != nil {
		return richDashboardStack{}, fmt.Errorf("resolve Compose path: %w", err)
	}
	composeBytes, err := os.ReadFile(compose)
	if err != nil {
		return richDashboardStack{}, fmt.Errorf("read bundled Compose config: %w", err)
	}
	if !strings.Contains(string(composeBytes), bundledPrometheusConfigMount) {
		return richDashboardStack{}, fmt.Errorf("bundled Compose config does not mount %s", bundledPrometheusConfigMount)
	}
	target, err := bundledPrometheusTargetForListener(listenerAddress)
	if err != nil {
		return richDashboardStack{}, err
	}
	templatePath := filepath.Join(filepath.Dir(compose), "prometheus.yml")
	templateBytes, err := os.ReadFile(templatePath)
	if err != nil {
		return richDashboardStack{}, fmt.Errorf("read bundled Prometheus config: %w", err)
	}
	rendered, err := rewriteBundledPrometheusTarget(string(templateBytes), target)
	if err != nil {
		return richDashboardStack{}, err
	}
	tempDir, err := os.MkdirTemp("", "fak-grafana-")
	if err != nil {
		return richDashboardStack{}, fmt.Errorf("create temporary Grafana config directory: %w", err)
	}
	stack := richDashboardStack{
		composePath:          compose,
		prometheusConfigPath: filepath.Join(tempDir, "prometheus.yml"),
		tempDir:              tempDir,
	}
	if err := os.WriteFile(stack.prometheusConfigPath, []byte(rendered), 0o644); err != nil {
		_ = cleanupBundledGrafanaStack(stack)
		return richDashboardStack{}, fmt.Errorf("write temporary Prometheus config: %w", err)
	}
	return stack, nil
}

func bundledPrometheusTargetForListener(listenerAddress string) (string, error) {
	_, port, err := net.SplitHostPort(strings.TrimSpace(listenerAddress))
	if err != nil {
		return "", fmt.Errorf("gateway listener address %q is unavailable: %w", listenerAddress, err)
	}
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return "", fmt.Errorf("gateway listener address %q has an invalid port", listenerAddress)
	}
	// The container reaches the selected listener port through Docker's host alias.
	// This never changes the gateway bind, so a loopback-only gateway stays off-host.
	return net.JoinHostPort("host.docker.internal", strconv.Itoa(n)), nil
}

func rewriteBundledPrometheusTarget(templateText, target string) (string, error) {
	needle := `targets: ["` + bundledPrometheusTemplateTarget + `"]`
	if strings.Count(templateText, needle) != 1 {
		return "", fmt.Errorf("bundled Prometheus config must contain exactly one default gateway target")
	}
	return strings.Replace(templateText, needle, `targets: ["`+target+`"]`, 1), nil
}

func dashboardComposeEnv(env []string, prometheusConfigPath string) []string {
	out := make([]string, 0, len(env)+1)
	prefix := bundledPrometheusConfigEnv + "="
	for _, entry := range env {
		if len(entry) >= len(prefix) && strings.EqualFold(entry[:len(prefix)], prefix) {
			continue
		}
		out = append(out, entry)
	}
	return append(out, prefix+prometheusConfigPath)
}

func cleanupBundledGrafanaStack(stack richDashboardStack) error {
	var errs []error
	if stack.prometheusConfigPath != "" {
		if err := os.Remove(stack.prometheusConfigPath); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, err)
		}
	}
	if stack.tempDir != "" {
		if err := os.Remove(stack.tempDir); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func probeGrafana(ctx context.Context, base string) error {
	u, err := safeDashboardBaseURL(base)
	if err != nil {
		return err
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/api/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return err
	}
	resp, err := richDashboardProbeClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return fmt.Errorf("Grafana health returned %s", resp.Status)
	}
	return nil
}

func boundedDashboardError(err error) string {
	text := strings.Join(strings.Fields(err.Error()), " ")
	if len(text) > 240 {
		text = text[:240] + "…"
	}
	return text
}

func richDashboardDestination(base, uid string) (string, error) {
	if _, ok := richDashboardUIDs[uid]; !ok {
		return "", errors.New("unknown dashboard")
	}
	u, err := safeDashboardBaseURL(base)
	if err != nil {
		return "", err
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/d/" + uid
	return u.String(), nil
}

var richDashboardPage = template.Must(template.New("rich-dashboard").Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fak rich dashboards</title>{{if eq .State "starting"}}<meta http-equiv="refresh" content="2">{{end}}
<style>body{margin:0;background:#0b1020;color:#edf2ff;font:16px/1.5 system-ui,sans-serif}main{width:min(680px,calc(100% - 32px));margin:12vh auto;padding:28px;border:1px solid #2a3652;border-radius:16px;background:#141b2d}a{color:#8eb5ff}.state{color:#76e6c5;font-weight:700;text-transform:uppercase;letter-spacing:.08em}.error{color:#ffb4a8}</style></head>
<body><main><div class="state {{if or (eq .State "unavailable") (eq .State "disabled")}}error{{end}}" role="status">{{.State}}</div><h1>Rich dashboards</h1>
{{if eq .State "starting"}}<p>Checking for a healthy Grafana, then starting the bundled Compose stack on demand if needed. FAK does not start Docker Desktop.</p><p>A stack started by this gateway stops with the gateway; an already-healthy Grafana is adopted and left running. After a Docker or host restart, start Docker and the gateway, then click a rich dashboard again.</p>
{{else}}<p>{{.Reason}}</p>{{end}}<p><a href="/">Return to the lightweight live dashboard</a></p></main></body></html>`))

func (s *Server) handleRichDashboard(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Query().Get("dashboard") != "rich" {
		return false
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return true
	}
	uid := strings.TrimSpace(r.URL.Query().Get("uid"))
	if uid == "" {
		uid = richDashboardDefaultUID
	}
	if _, ok := richDashboardUIDs[uid]; !ok {
		http.Error(w, "unknown rich dashboard", http.StatusBadRequest)
		return true
	}
	snap := s.richDashboards.ensure()
	if snap.State == "ready" {
		destination, err := richDashboardDestination(snap.URL, uid)
		if err == nil {
			recordDashboardAdoption("rich_ready")
			http.Redirect(w, r, destination, http.StatusSeeOther)
			return true
		}
	}
	if snap.State == "unavailable" || snap.State == "disabled" {
		recordDashboardAdoption("rich_unavailable")
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method == http.MethodHead {
		return true
	}
	_ = richDashboardPage.Execute(w, snap)
	return true
}
