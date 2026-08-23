package gateway

import (
	"context"
	"errors"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	richDashboardDefaultUID = "fak-gateway-observability"
	richDashboardTimeout    = 90 * time.Second
)

var dashboardDockerAvailable = func() bool {
	_, err := exec.LookPath("docker")
	return err == nil
}
var richDashboardUIDs = map[string]struct{}{
	"fleet-bottleneck": {}, "fak-gateway-observability": {}, "fak-dogfood-slow-requests": {},
	"fak-startup-load": {}, "fak-guard-adjudication": {}, "fak-cache-health": {},
	"fak-cache-value-rollup": {}, "fak-fleet-overview": {}, "fak-fleet-session": {},
}

type richDashboardManager struct {
	mu              sync.Mutex
	ctx             context.Context
	cancel          context.CancelFunc
	state           string
	reason          string
	baseURL         string
	compose         string
	dockerAvailable func() bool
	owned           bool
	start           func(context.Context, string) error
	stop            func(context.Context, string) error
	probe           func(context.Context, string) error
	startedAt       time.Time
}

type richDashboardSnapshot struct {
	State     string
	Reason    string
	URL       string
	StartedAt time.Time
}

func newRichDashboardManager() *richDashboardManager {
	ctx, cancel := context.WithCancel(context.Background())
	m := &richDashboardManager{
		ctx: ctx, cancel: cancel, state: "dormant",
		baseURL: strings.TrimRight(strings.TrimSpace(os.Getenv("FAK_GRAFANA_URL")), "/"),
		compose: strings.TrimSpace(os.Getenv("FAK_GRAFANA_COMPOSE")),
	}
	m.dockerAvailable = dashboardDockerAvailable
	m.start = startBundledGrafana
	m.stop = stopBundledGrafana
	m.probe = probeGrafana
	if disabledDashboardValue(os.Getenv("FAK_DASHBOARDS")) {
		m.state = "disabled"
		m.reason = "Rich dashboards are disabled by FAK_DASHBOARDS. The lightweight live dashboard remains available."
	}
	if m.baseURL != "" {
		if _, err := safeDashboardBaseURL(m.baseURL); err != nil {
			m.state = "unavailable"
			m.reason = "FAK_GRAFANA_URL must be an http(s) URL without credentials."
		}
	}
	return m
}

func disabledDashboardValue(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "0", "false", "off", "disabled":
		return true
	default:
		return false
	}
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
	if m.state != "dormant" {
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
	if base == "" {
		if compose == "" {
			compose = findBundledGrafanaCompose()
		}
		if compose == "" {
			m.fail("Bundled Grafana files were not found. Set FAK_GRAFANA_URL to an existing Grafana or run from a fak checkout.")
			return
		}
		if !m.dockerAvailable() {
			m.fail("Docker is not available. Install/start Docker, or set FAK_GRAFANA_URL to an existing Grafana.")
			return
		}
		ctx, cancel := context.WithTimeout(m.ctx, richDashboardTimeout)
		err := m.start(ctx, compose)
		cancel()
		if err != nil {
			m.fail("Bundled Grafana could not start: " + boundedDashboardError(err))
			return
		}
		base = "http://localhost:3000"
		owned = true
	}

	deadline := time.Now().Add(richDashboardTimeout)
	for {
		ctx, cancel := context.WithTimeout(m.ctx, 3*time.Second)
		err := m.probe(ctx, base)
		cancel()
		if err == nil {
			m.mu.Lock()
			m.baseURL, m.owned, m.compose, m.state, m.reason = base, owned, compose, "ready", ""
			m.mu.Unlock()
			return
		}
		if time.Now().After(deadline) || m.ctx.Err() != nil {
			if owned {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				_ = m.stop(ctx, compose)
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
	owned, compose := m.owned, m.compose
	m.owned = false
	m.mu.Unlock()
	if owned && compose != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		_ = m.stop(ctx, compose)
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

func startBundledGrafana(ctx context.Context, compose string) error {
	cmd := exec.CommandContext(ctx, "docker", "compose", "-f", compose, "--profile", "local-prometheus", "up", "-d")
	cmd.Dir = filepath.Dir(compose)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

func stopBundledGrafana(ctx context.Context, compose string) error {
	cmd := exec.CommandContext(ctx, "docker", "compose", "-f", compose, "down")
	cmd.Dir = filepath.Dir(compose)
	return cmd.Run()
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
	resp, err := http.DefaultClient.Do(req)
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
{{if eq .State "starting"}}<p>Starting the bundled Grafana stack on demand. Nothing heavy ran before this click. This page will continue automatically.</p>
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
			http.Redirect(w, r, destination, http.StatusSeeOther)
			return true
		}
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method == http.MethodHead {
		return true
	}
	_ = richDashboardPage.Execute(w, snap)
	return true
}
