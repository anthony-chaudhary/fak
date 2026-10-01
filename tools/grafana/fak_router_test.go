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
	Options struct {
		Content string `json:"content"`
	} `json:"options"`
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
// contract for a routing-mode board. The live router forwards every request
// through the routing path, where the fak_gateway_* proxy counters stay
// STRUCTURALLY zero while the fak_router_* families climb into the thousands.
// The board is therefore built on the router families (guarded by
// TestFakRouterCommonStatsUseRouterFamilies), and this test pins the two honesty
// guarantees that keep that design from silently misleading an operator:
//
//  1. The routing-attempts panel names itself the routing-mode demand signal, so
//     one logical request fanning into several attempts is not mistaken for the
//     single-engine proxy path.
//  2. The board header explains the fak_gateway_* routing-mode zero and the
//     "Unavailable means missing evidence" rule, so an operator who remembers
//     the old board does not read an empty panel as idle traffic.
func TestFakRouterDashboardRoutingModeHonesty(t *testing.T) {
	d := loadFakRouter(t)

	descByTitle := map[string]string{}
	contentByTitle := map[string]string{}
	for _, p := range d.Panels {
		if p.Title != "" {
			descByTitle[p.Title] = p.Description
		}
		if p.Options.Content != "" {
			contentByTitle[p.Title] = p.Options.Content
		}
	}

	// (1) Routing attempts must be presented as the routing-mode demand signal.
	attempts, ok := descByTitle["Routing attempts (all backends)"]
	if !ok {
		t.Fatalf("%s must expose a 'Routing attempts (all backends)' panel as the routing-mode demand signal", fakRouterPath)
	}
	if !strings.Contains(attempts, "routing mode") && !strings.Contains(attempts, "routing-mode") {
		t.Errorf("routing attempts panel must explain that it is the routing-mode demand signal, got: %q", attempts)
	}

	// (2) The header must explain the hub/routing split honestly: the gateway
	// families are structurally zero on this job, and an absent series means
	// missing evidence rather than an idle router.
	header, ok := contentByTitle["Unified router: live routing traffic, backend health, and failover"]
	if !ok {
		t.Fatalf("%s must carry the unified-router header text panel", fakRouterPath)
	}
	for _, want := range []string{"structurally zero", "missing evidence"} {
		if !strings.Contains(header, want) {
			t.Errorf("router header must explain the routing-mode honesty rule %q, got: %q", want, header)
		}
	}
	if !strings.Contains(header, "fak_gateway_") || !strings.Contains(header, "fak_router_") {
		t.Errorf("router header must name both metric families so the routing-mode split is explicit, got: %q", header)
	}
}

// TestFakRouterStatusZeroIsNotBareNumeric pins the legibility contract on the
// "Attempts by status" panel. In fak_router_attempts_total the `status` label is
// the observed HTTP status, and the value `0` is NOT a count of zero: it means a
// transport error (dial failure, timeout, reset) that never produced an HTTP
// status. The live router writes a real, growing `status="0"` series on every
// transport failure, so a legend that renders the bare string `0` reads to an
// operator as "zero attempts" on a router that is actually failing upstream.
//
// The panel must relabel the `0` bucket to a self-describing legend and say what
// `0` means, so the number can never be misread as an idle bucket.
func TestFakRouterStatusZeroIsNotBareNumeric(t *testing.T) {
	d := loadFakRouter(t)

	var status *fakRouterPanel
	for i := range d.Panels {
		if d.Panels[i].Title == "Attempts by status" {
			status = &d.Panels[i]
			break
		}
	}
	if status == nil {
		t.Fatalf("%s must expose an 'Attempts by status' panel", fakRouterPath)
	}
	if len(status.Targets) == 0 {
		t.Fatalf("panel 'Attempts by status' has no query target")
	}
	expr := status.Targets[0].Expr

	// The query must relabel status="0" to a self-describing legend. A bare
	// `sum by (status)` renders a series called `0`, indistinguishable from a
	// zero count.
	if !strings.Contains(expr, "label_replace") || !strings.Contains(expr, "transport") {
		t.Errorf("panel 'Attempts by status' expr must relabel status=\"0\" to a named transport-error legend "+
			"(label_replace), otherwise the 0 bucket renders as a bare '0' that reads as a zero count; got: %s", expr)
	}
	// The description must teach the meaning too, so the panel stays legible when
	// the series is absent.
	if !strings.Contains(status.Description, "transport") {
		t.Errorf("panel 'Attempts by status' description must explain that status=0 is a transport error, not a zero count; got: %q", status.Description)
	}
}

// TestFakRouterCommonStatsUseRouterFamilies pins the "common stats" coverage
// contract for the unified-router board. When the live router serves
// third-party upstreams it runs in ROUTING mode: the fak_gateway_* proxy
// counters (requests, streaming requests, upstream token totals, net-win
// gauges, batch slots) are structurally ZERO there, while the router's own
// fak_router_* families climb into the thousands. A board built on the gateway
// families therefore renders a busy router as idle.
//
// Every "common stat" an operator reaches for first - RPS now, RPM, error
// rate, per-backend health, latency percentiles, outcome breakdown - must be
// derived from a fak_router_* family the routing-mode router actually emits.
// This test names the required stats and refuses a board that answers any of
// them from a structurally-zero gateway counter.
func TestFakRouterCommonStatsUseRouterFamilies(t *testing.T) {
	d := loadFakRouter(t)

	// Every expr on the board, keyed by panel title, for coverage lookups.
	exprByTitle := map[string][]string{}
	for _, p := range d.Panels {
		for _, tgt := range p.Targets {
			if strings.TrimSpace(tgt.Expr) != "" {
				exprByTitle[p.Title] = append(exprByTitle[p.Title], tgt.Expr)
			}
		}
	}

	// (a) No panel may derive a request/rate/error stat from a gateway counter:
	// those are structurally zero in routing mode.
	for title, exprs := range exprByTitle {
		for _, expr := range exprs {
			for _, dead := range []string{
				"fak_gateway_requests_total",
				"fak_gateway_stream_requests_total",
				"fak_gateway_upstream_fresh_tokens_total",
				"fak_gateway_upstream_cached_tokens_total",
				"fak_gateway_upstream_decode_tokens_total",
				"fak_gateway_batch_slots_assigned_total",
			} {
				if strings.Contains(expr, dead) {
					t.Errorf("panel %q derives a stat from %s, which is structurally zero on a routing-mode router; "+
						"use the fak_router_* family instead (expr: %s)", title, dead, expr)
				}
			}
		}
	}

	// (b) The board must expose each common stat, backed by the real router
	// family. required maps the operator stat to a substring the panel set must
	// contain somewhere.
	required := []struct {
		stat   string
		needle string
	}{
		{"request rate (RPS)", "rate(fak_router_attempts_total"},
		{"requests per minute", "fak_router_attempts_total"},
		{"success rate", "fak_router_requests_total"},
		{"inflight requests", "fak_router_inflight_requests"},
		{"requests by outcome", "fak_router_requests_total"},
		{"upstream error rate", "fak_router_attempt_failures_total"},
		{"attempt latency histogram", "fak_router_attempt_duration_seconds"},
		{"time-to-first-byte latency", "fak_router_attempt_ttfb_seconds"},
		{"per-backend health", "fak_router_backend_"},
	}
	for _, r := range required {
		found := false
		for _, exprs := range exprByTitle {
			for _, expr := range exprs {
				if strings.Contains(expr, r.needle) {
					found = true
				}
			}
		}
		if !found {
			t.Errorf("%s is missing from the board: no panel targets %s (the routing-mode router emits it)", r.stat, r.needle)
		}
	}
}
