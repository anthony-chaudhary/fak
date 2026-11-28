package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type routeStatusPayload struct {
	Schema         string             `json:"schema"`
	RoutingEnabled bool               `json:"routing_enabled"`
	DefaultModel   string             `json:"default_model"`
	Routes         []routeStatusEntry `json:"routes"`
	Totals         routeStatusTotals  `json:"totals"`
}

type routeStatusEntry struct {
	Model          string              `json:"model"`
	ResidencyFloor string              `json:"residency_floor,omitempty"`
	Primary        routeStatusTarget   `json:"primary"`
	Fallbacks      []routeStatusTarget `json:"fallbacks"`
}

type routeStatusTarget struct {
	Account             string `json:"account"`
	Kind                string `json:"kind"`
	UpstreamModel       string `json:"upstream_model"`
	Locality            string `json:"locality"`
	State               string `json:"state"`
	CooldownRemainingMs int64  `json:"cooldown_remaining_ms"`
}

type routeStatusTotals struct {
	Attempts  int64 `json:"attempts"`
	Successes int64 `json:"successes"`
	Failovers int64 `json:"failovers"`
}

func routeStatusDefaultAddr() string {
	addr := strings.TrimSpace(os.Getenv("FAK_ADDR"))
	if addr == "" {
		return "http://127.0.0.1:8080"
	}
	return normalizeRouteStatusAddr(addr)
}

func normalizeRouteStatusAddr(raw string) string {
	addr := strings.TrimSpace(raw)
	if addr == "" {
		return ""
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	return strings.TrimRight(addr, "/")
}

func runRouteStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("route status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", routeStatusDefaultAddr(), "gateway base URL (default $FAK_ADDR or http://127.0.0.1:8080)")
	asJSON := fs.Bool("json", false, "emit raw /v1/router/status JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak route status: unexpected argument(s): %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	baseURL := normalizeRouteStatusAddr(*addr)
	if baseURL == "" {
		fmt.Fprintln(stderr, "fak route status: --addr must not be empty")
		return 2
	}

	raw, payload, err := fetchRouteStatus(baseURL)
	if err != nil {
		fmt.Fprintf(stderr, "fak route status: %v\n", err)
		return 1
	}

	degraded := hasLocalOnlyPrimaryCooldown(payload)
	if *asJSON {
		fmt.Fprintln(stdout, string(raw))
	} else {
		printRouteStatus(stdout, baseURL, payload, degraded)
	}
	if degraded {
		return 1
	}
	return 0
}

func fetchRouteStatus(baseURL string) (json.RawMessage, routeStatusPayload, error) {
	url := baseURL + "/v1/router/status"
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, routeStatusPayload{}, fmt.Errorf("gateway unreachable at %s (is `fak serve` running?): %w", baseURL, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, routeStatusPayload{}, fmt.Errorf("read %s: %w", url, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, routeStatusPayload{}, fmt.Errorf("gateway at %s returned HTTP %d for /v1/router/status", baseURL, resp.StatusCode)
	}
	trimmed := strings.TrimSpace(string(body))
	var payload routeStatusPayload
	if err := json.Unmarshal([]byte(trimmed), &payload); err != nil {
		return nil, routeStatusPayload{}, fmt.Errorf("gateway at %s returned invalid JSON for /v1/router/status: %w", baseURL, err)
	}
	return json.RawMessage(trimmed), payload, nil
}

func hasLocalOnlyPrimaryCooldown(p routeStatusPayload) bool {
	for _, r := range p.Routes {
		if r.ResidencyFloor == "local_only" && (r.Primary.State == "cooldown" || r.Primary.CooldownRemainingMs > 0) {
			return true
		}
	}
	return false
}

func printRouteStatus(w io.Writer, baseURL string, p routeStatusPayload, degraded bool) {
	fmt.Fprintln(w, "== fak route status ==")
	fmt.Fprintf(w, "gateway        : %s\n", baseURL)
	fmt.Fprintf(w, "schema         : %s\n", orDash(p.Schema))
	fmt.Fprintf(w, "routing_enabled: %t\n", p.RoutingEnabled)
	fmt.Fprintf(w, "default_model  : %s\n", orDash(p.DefaultModel))
	fmt.Fprintf(w, "totals         : attempts=%d successes=%d failovers=%d\n", p.Totals.Attempts, p.Totals.Successes, p.Totals.Failovers)
	if len(p.Routes) == 0 {
		fmt.Fprintln(w, "routes         : (none)")
		return
	}
	fmt.Fprintln(w, "routes:")
	for _, r := range p.Routes {
		fmt.Fprintf(w, "  %-16s residency_floor=%s\n", r.Model, orDash(r.ResidencyFloor))
		printRouteStatusTarget(w, "primary", r.Primary)
		for _, fb := range r.Fallbacks {
			printRouteStatusTarget(w, "fallback", fb)
		}
	}
	if degraded {
		fmt.Fprintln(w, "status         : DEGRADED (local_only primary in cooldown)")
	}
}

func printRouteStatusTarget(w io.Writer, role string, t routeStatusTarget) {
	cooldown := ""
	if t.CooldownRemainingMs > 0 || t.State == "cooldown" {
		cooldown = fmt.Sprintf(" cooldown=%dms", t.CooldownRemainingMs)
	}
	fmt.Fprintf(w, "    %-8s %-14s (%s) upstream=%-16s locality=%-11s state=%s%s\n",
		role+":", t.Account, orDash(t.Kind), orDash(t.UpstreamModel), orDash(t.Locality), orDash(t.State), cooldown)
}
