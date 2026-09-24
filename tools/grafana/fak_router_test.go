package grafana

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakRouterPath is the hand-authored unified-router dashboard (uid fak-router)
// rendered by the fak_router scrape job. No generator owns this artifact, so
// this test is its contract: it pins the board to the ROUTER scrape job and
// guards against the board silently drifting onto the generic fak_gateway job,
// which serves fak_gateway_* but does NOT emit fak_router_attempts_total.
const fakRouterPath = "dashboards/fak-router.json"

// fakRouterPanel is the subset of the Grafana panel schema this contract reads.
type fakRouterPanel struct {
	ID          int    `json:"id"`
	Type        string `json:"type"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Targets     []struct {
		Expr string `json:"expr"`
	} `json:"targets"`
}

type fakRouterDashboard struct {
	Title  string           `json:"title"`
	UID    string           `json:"uid"`
	Panels []fakRouterPanel `json:"panels"`
}

// loadFakRouter parses the dashboard relative to this test file so the contract
// holds regardless of the test's working directory.
func loadFakRouter(t *testing.T) fakRouterDashboard {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Join(filepath.Dir(file), filepath.FromSlash(fakRouterPath))
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dashboard artifact missing at %s: %v", path, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var d fakRouterDashboard
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("%s is not well-formed dashboard JSON: %v", path, err)
	}
	return d
}

// TestFakRouterDashboardContract pins the identity of the unified-router board
// and the scrape job every query must select. The board must read the router's
// own fak_router job (which emits fak_router_attempts_total), never the generic
// fak_gateway job.
func TestFakRouterDashboardContract(t *testing.T) {
	d := loadFakRouter(t)

	if d.UID != "fak-router" {
		t.Errorf("expected uid fak-router, got %q", d.UID)
	}
	if d.Title == "" {
		t.Error("dashboard must carry a title")
	}

	sawAttempts := false
	sawRouterJob := false
	for _, p := range d.Panels {
		for _, tgt := range p.Targets {
			expr := tgt.Expr
			if strings.TrimSpace(expr) == "" {
				continue
			}
			if strings.Contains(expr, "fak_router_attempts_total") {
				sawAttempts = true
			}
			if strings.Contains(expr, `job="fak_router"`) {
				sawRouterJob = true
			}
			// The board must never fall back to the generic gateway job: that
			// job does not emit the router's attempts counter, so such a panel
			// would render empty while looking healthy.
			if strings.Contains(expr, `job="fak_gateway"`) {
				t.Errorf("panel %d %q targets the generic fak_gateway job; the router board must use job=%q (expr: %s)",
					p.ID, p.Title, "fak_router", expr)
			}
		}
	}
	if !sawAttempts {
		t.Errorf("%s must query fak_router_attempts_total on at least one panel", fakRouterPath)
	}
	if !sawRouterJob {
		t.Errorf("%s must select the router scrape job on at least one panel", fakRouterPath)
	}
}

// TestFakRouterDashboardRoutingModeHonesty pins the operator-facing honesty
// contract on the panels whose counters are STRUCTURALLY zero in routing mode.
// The live router forwards every request through the routing path, so its own
// proxy request counter and the cache net-win gauges read 0 while
// fak_router_attempts_total climbs into the thousands. Those panels must say so
// in their descriptions; otherwise an operator reads a healthy routing-mode
// router as "no traffic" or "idle".
func TestFakRouterDashboardRoutingModeHonesty(t *testing.T) {
	d := loadFakRouter(t)

	descByTitle := map[string]string{}
	for _, p := range d.Panels {
		if p.Title != "" {
			descByTitle[p.Title] = p.Description
		}
	}

	// Routing attempts must be presented as the routing-mode demand signal.
	attempts, ok := descByTitle["Routing attempts (all backends)"]
	if !ok {
		t.Fatalf("%s must expose a 'Routing attempts (all backends)' panel as the routing-mode demand signal", fakRouterPath)
	}
	if !strings.Contains(attempts, "routing mode") && !strings.Contains(attempts, "routing-mode") {
		t.Errorf("routing attempts panel must explain that it is the routing-mode demand signal, got: %q", attempts)
	}

	// The proxy request counter and the net-win gauges must confess the caveat.
	for _, title := range []string{"Total requests", "Streaming requests", "Turn net-win ratio", "Prefill speedup (last turn)"} {
		desc, ok := descByTitle[title]
		if !ok {
			t.Errorf("%s is missing the %q panel", fakRouterPath, title)
			continue
		}
		if !strings.Contains(desc, "routing") {
			t.Errorf("panel %q must explain its routing-mode zero (description must mention routing), got: %q", title, desc)
		}
	}
}
