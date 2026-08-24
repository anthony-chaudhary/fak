package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const schema = "fak.dashboarddemo-selfcheck/1"

type result struct {
	Schema             string `json:"schema"`
	Verdict            string `json:"verdict"`
	HomepageStatus     int    `json:"homepage_status"`
	HealthStatus       int    `json:"health_status"`
	MetricsStatus      int    `json:"metrics_status"`
	RefreshSeconds     int    `json:"refresh_seconds"`
	RichDashboardCount int    `json:"rich_dashboard_count"`
	RichDashboardsLazy bool   `json:"rich_dashboards_lazy"`
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the deterministic live-dashboard witness")
	flag.Parse()
	if !*selfcheck || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: go run ./cmd/dashboarddemo -selfcheck")
		os.Exit(2)
	}
	got, err := runSelfcheck()
	if err != nil {
		fmt.Fprintln(os.Stderr, "dashboarddemo selfcheck:", err)
		os.Exit(1)
	}
	_ = json.NewEncoder(os.Stdout).Encode(got)
}

func runSelfcheck() (result, error) {
	s, err := gateway.New(gateway.Config{EngineID: "mock", Model: "demo", Provider: "openai"})
	if err != nil {
		return result{}, err
	}
	defer s.Close()

	home := httptest.NewRecorder()
	s.Handler().ServeHTTP(home, httptest.NewRequest(http.MethodGet, "/", nil))
	health := httptest.NewRecorder()
	s.Handler().ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	metrics := httptest.NewRecorder()
	s.Handler().ServeHTTP(metrics, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := home.Body.String()
	count := strings.Count(body, `class="card rich-dashboard"`)
	got := result{
		Schema: schema, Verdict: "pass", HomepageStatus: home.Code, HealthStatus: health.Code,
		MetricsStatus: metrics.Code, RefreshSeconds: 5, RichDashboardCount: count,
		RichDashboardsLazy: strings.Contains(body, "on-demand") && strings.Contains(body, "setInterval(refresh,5000)"),
	}
	if home.Code != http.StatusOK || health.Code != http.StatusOK || metrics.Code != http.StatusOK || count != 9 || !got.RichDashboardsLazy {
		return result{}, fmt.Errorf("dashboard invariant failed: %+v", got)
	}
	return got, nil
}
