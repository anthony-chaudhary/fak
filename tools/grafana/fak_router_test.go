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

// TestFakRouterGapKPIPanels pins the KPI-ranking gap panels on the unified-router
// board (fak-private ROUTER-PROMETHEUS-KPI-RANKING "Next steps", item 2): admission
// pressure, durable-store utilization + sample age, TTFT/TPOT p95, prefix-cache hit
// rate, balancer share vs weights, and the state-verdict gap note. Every expression
// must stay on the router job and the HEAD metric/label contract; the honesty tokens
// keep the absence-is-not-zero semantics from being silently dropped when a panel is
// edited.
func TestFakRouterGapKPIPanels(t *testing.T) {
	d := loadFakRouter(t)

	exprByTitle := map[string][]string{}
	descByTitle := map[string]string{}
	contentByTitle := map[string]string{}
	for _, p := range d.Panels {
		if p.Title != "" {
			descByTitle[p.Title] = p.Description
		}
		if p.Options.Content != "" {
			contentByTitle[p.Title] = p.Options.Content
		}
		for _, tgt := range p.Targets {
			if strings.TrimSpace(tgt.Expr) != "" {
				exprByTitle[p.Title] = append(exprByTitle[p.Title], tgt.Expr)
			}
		}
	}

	// (a) Each gap KPI must be backed by its real router family somewhere on the
	// board.
	required := []struct {
		kpi    string
		needle string
	}{
		{"admission outstanding", "fak_router_admission_outstanding"},
		{"admission bound", "fak_router_admission_bound"},
		{"admission high water", "fak_router_admission_high_water"},
		{"admission reject rate", "rate(fak_router_admission_rejected_total"},
		{"durable-store bytes", "fak_router_durable_store_bytes"},
		{"durable-store quota", "fak_router_durable_store_quota_bytes"},
		{"durable-store sample age", "fak_router_durable_store_sample_age_seconds"},
		{"TTFT p95 histogram", "fak_router_time_to_first_token_seconds_bucket"},
		{"TPOT p95 histogram", "fak_router_time_per_output_token_seconds_bucket"},
		{"prefix-cache numerator", "fak_router_cached_prompt_tokens_total"},
		{"prefix-cache denominator", "fak_router_cache_reported_prompt_tokens_total"},
		{"balancer share ratio", "fak_router_balance_share_ratio"},
		{"balancer snapshot age", "fak_router_balance_snapshot_age_seconds"},
		{"balancer fail-open rate", "fak_router_balance_decisions_total"},
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
			t.Errorf("%s is missing from the board: no panel targets %s", r.kpi, r.needle)
		}
	}

	// (b) The cache-hit panels must use the REPORTED denominator: any panel that
	// plots the cached-prompt numerator must also plot the reported denominator
	// in the SAME panel, otherwise the cached line has no reference to read
	// against. Checked per panel (not per expr) so a multi-series throughput
	// panel that shows prompt / reported / cached together still passes.
	for title, exprs := range exprByTitle {
		sawCached, sawReported := false, false
		for _, expr := range exprs {
			if strings.Contains(expr, "fak_router_cached_prompt_tokens_total") {
				sawCached = true
			}
			if strings.Contains(expr, "fak_router_cache_reported_prompt_tokens_total") {
				sawReported = true
			}
		}
		if sawCached && !sawReported {
			t.Errorf("panel %q plots the cached-prompt numerator without the reported denominator "+
				"(fak_router_cache_reported_prompt_tokens_total) in the same panel; got: %v", title, exprs)
		}
	}

	// (c) Admission fill must be the outstanding/bound ratio, not a raw gauge.
	const fillTitle = "Admission fill %"
	fill, ok := exprByTitle[fillTitle]
	if !ok {
		t.Fatalf("%s must expose an 'Admission fill %%' panel", fakRouterPath)
	}
	if len(fill) == 0 || !strings.Contains(fill[0], "fak_router_admission_outstanding") ||
		!strings.Contains(fill[0], "fak_router_admission_bound") {
		t.Errorf("panel 'Admission fill %%' must divide outstanding by bound; got: %v", fill)
	}

	// (d) Balancer share: pinned expression plus the absence-is-unmeasured honesty
	// token, so the panel cannot render an empty group as a healthy zero.
	share, ok := exprByTitle["Balancer share ratio by group / backend"]
	if !ok {
		t.Fatalf("%s must expose a 'Balancer share ratio by group / backend' panel", fakRouterPath)
	}
	if len(share) == 0 || !strings.Contains(share[0], "fak_router_balance_share_ratio") {
		t.Errorf("panel 'Balancer share ratio by group / backend' must target fak_router_balance_share_ratio; got: %v", share)
	}
	if !strings.Contains(descByTitle["Balancer share ratio by group / backend"], "unmeasured") {
		t.Errorf("balancer share panel description must say absence = unmeasured, not 0; got: %q",
			descByTitle["Balancer share ratio by group / backend"])
	}

	// (e) Fail-open rate must count only non-empty-reason decisions: crediting
	// fail-open rows to a member would fabricate choices.
	failOpen, ok := exprByTitle["Balancer fail-open rate / s"]
	if !ok {
		t.Fatalf("%s must expose a 'Balancer fail-open rate / s' panel", fakRouterPath)
	}
	if len(failOpen) == 0 || !strings.Contains(failOpen[0], "fak_router_balance_decisions_total") ||
		!strings.Contains(failOpen[0], `reason!=""`) {
		t.Errorf("panel 'Balancer fail-open rate / s' must filter fak_router_balance_decisions_total on reason!=\"\"; got: %v", failOpen)
	}

	// (f) Durable-store utilization must warn that the gauges are sampled, with
	// the sample-age series as the staleness witness.
	dsDesc := descByTitle["Durable store utilization %"]
	if dsDesc == "" {
		t.Fatalf("%s must expose a 'Durable store utilization %%' panel", fakRouterPath)
	}
	if !strings.Contains(dsDesc, "sample") {
		t.Errorf("durable-store panel description must say the gauges are sampled and to check sample age first; got: %q", dsDesc)
	}

	// (g) The state verdict has no prometheus metric yet: the board must carry a
	// panel that documents the gap and points at the live surface, not a proxy
	// metric masquerading as the verdict.
	state, ok := contentByTitle["Router state verdict: read live, no metric yet (gap)"]
	if !ok {
		t.Fatalf("%s must expose the router state-verdict gap panel", fakRouterPath)
	}
	for _, want := range []string{"/v1/router/state", "dead_ladder"} {
		if !strings.Contains(state, want) {
			t.Errorf("state-verdict gap panel must name %s; got: %q", want, state)
		}
	}
}

// TestFakRouterVolumePanelsAreWindowed pins the operator complaint that the
// lifetime-total panels did not respond to the selected time range: the volume
// tiles and the per-backend / per-status attempt panels must be windowed, not
// cumulative since process start. Totals use increase($__range) so they change
// with the range; live rates use rate($__rate_interval) so they track the
// window. A regression to a bare sum(counter) re-introduces the "lifetime
// total" misread this test exists to catch.
func TestFakRouterVolumePanelsAreWindowed(t *testing.T) {
	d := loadFakRouter(t)

	exprByTitle := map[string][]string{}
	for _, p := range d.Panels {
		for _, tgt := range p.Targets {
			if strings.TrimSpace(tgt.Expr) != "" {
				exprByTitle[p.Title] = append(exprByTitle[p.Title], tgt.Expr)
			}
		}
	}

	// (a) The two volume tiles must be range-windowed totals.
	for _, title := range []string{"Total requests", "Routing attempts (all backends)"} {
		exprs, ok := exprByTitle[title]
		if !ok {
			t.Fatalf("%s must expose a %q panel", fakRouterPath, title)
		}
		if len(exprs) == 0 || !strings.Contains(exprs[0], "increase(") || !strings.Contains(exprs[0], "$__range") {
			t.Errorf("panel %q must be a range-windowed total (increase over $__range), not a lifetime sum; got: %v", title, exprs)
		}
	}

	// (b) The per-backend / per-status attempt panels must be live rates.
	for _, title := range []string{"Attempts by backend", "Attempts by status"} {
		exprs, ok := exprByTitle[title]
		if !ok {
			t.Fatalf("%s must expose a %q panel", fakRouterPath, title)
		}
		if len(exprs) == 0 || !strings.Contains(exprs[0], "rate(") || !strings.Contains(exprs[0], "$__rate_interval") {
			t.Errorf("panel %q must be a live rate (rate over $__rate_interval), not a cumulative total; got: %v", title, exprs)
		}
	}

	// (c) The backend-health table's two attempt columns must be range-windowed.
	table, ok := exprByTitle["Backends: live health"]
	if !ok {
		t.Fatalf("%s must expose a 'Backends: live health' panel", fakRouterPath)
	}
	windowedAttempts := 0
	for _, expr := range table {
		if strings.Contains(expr, "fak_router_attempts_total") &&
			strings.Contains(expr, "increase(") && strings.Contains(expr, "$__range") {
			windowedAttempts++
		}
	}
	if windowedAttempts < 2 {
		t.Errorf("'Backends: live health' must window both attempt columns (attempts_total and fallback_attempts) over $__range; found %d, got: %v", windowedAttempts, table)
	}
}

// TestFakRouterCachingAndTokenPanels pins the caching and token-throughput
// section added for the operator complaint that the board was missing caching
// for the router's own surface and live token stats. Each new panel must be
// backed by the real router family it claims to plot, on the router job.
func TestFakRouterCachingAndTokenPanels(t *testing.T) {
	d := loadFakRouter(t)

	exprByTitle := map[string][]string{}
	for _, p := range d.Panels {
		for _, tgt := range p.Targets {
			if strings.TrimSpace(tgt.Expr) != "" {
				exprByTitle[p.Title] = append(exprByTitle[p.Title], tgt.Expr)
			}
		}
	}

	// (a) TTFB by cache hit/miss: the cache-split histogram, split on the cache label.
	// Pin the split clause itself (sum by (le, cache)), not the bare "cache"
	// substring: the metric name fak_router_attempt_ttfb_by_cache_seconds_bucket
	// already contains "cache", so a bare-substring check would pass even if the
	// panel dropped the hit/miss split and rendered a single unsplit line.
	ttfb, ok := exprByTitle["TTFB p95 by cache hit / miss"]
	if !ok {
		t.Fatalf("%s must expose a 'TTFB p95 by cache hit / miss' panel", fakRouterPath)
	}
	if len(ttfb) == 0 || !strings.Contains(ttfb[0], "fak_router_attempt_ttfb_by_cache_seconds_bucket") ||
		!strings.Contains(ttfb[0], "sum by (le, cache)") {
		t.Errorf("panel 'TTFB p95 by cache hit / miss' must target the cache-split TTFB histogram split on the cache label (sum by (le, cache)); got: %v", ttfb)
	}

	// (b) Cache-reported turns: the turn view of the hit-rate denominator.
	turns, ok := exprByTitle["Cache-reported turns / s"]
	if !ok {
		t.Fatalf("%s must expose a 'Cache-reported turns / s' panel", fakRouterPath)
	}
	if len(turns) == 0 || !strings.Contains(turns[0], "fak_router_cache_reported_turns_total") {
		t.Errorf("panel 'Cache-reported turns / s' must target fak_router_cache_reported_turns_total; got: %v", turns)
	}

	// (c) Prompt / cached / reported token throughput: all three prompt-token
	// families in one panel so the cached line has its reference lines.
	throughput, ok := exprByTitle["Prompt / cached / reported token throughput"]
	if !ok {
		t.Fatalf("%s must expose a 'Prompt / cached / reported token throughput' panel", fakRouterPath)
	}
	for _, want := range []string{
		"fak_router_prompt_tokens_total",
		"fak_router_cache_reported_prompt_tokens_total",
		"fak_router_cached_prompt_tokens_total",
	} {
		saw := false
		for _, expr := range throughput {
			if strings.Contains(expr, want) {
				saw = true
				break
			}
		}
		if !saw {
			t.Errorf("panel 'Prompt / cached / reported token throughput' must target %s; got: %v", want, throughput)
		}
	}

	// (d) Token usage by type: the GenAI-convention token histogram, split on token_type.
	token, ok := exprByTitle["Token usage by type (tokens / s)"]
	if !ok {
		t.Fatalf("%s must expose a 'Token usage by type (tokens / s)' panel", fakRouterPath)
	}
	if len(token) == 0 || !strings.Contains(token[0], "fak_router_token_usage_sum") ||
		!strings.Contains(token[0], "token_type") {
		t.Errorf("panel 'Token usage by type (tokens / s)' must target fak_router_token_usage_sum split on token_type; got: %v", token)
	}
}
