package gateway

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// A stalled upstream (the streaming idle-deadline trip) must surface as a DISTINCT 504 with the
// "upstream_stalled" code so a client/harness can tell a silent provider apart from a 4xx request
// error, a 5xx, or a parse failure — not the opaque code:null "upstream model error".
func TestUpstreamErrorStatus_StallIsDistinct504(t *testing.T) {
	status, code, msg := upstreamErrorStatus(&agent.UpstreamStalledError{Idle: 60 * time.Second})

	if status != http.StatusGatewayTimeout {
		t.Fatalf("a stall should be 504 Gateway Timeout, got %d", status)
	}
	if code != "upstream_stalled" {
		t.Fatalf("a stall should carry the distinct code, got %q", code)
	}
	if !strings.Contains(msg, "stalled") || !strings.Contains(msg, "silent") {
		t.Fatalf("stall message should name the condition: %q", msg)
	}
	// A stall must NOT be misclassified as a 4xx/5xx upstream status or an OOM.
	if code == "in_kernel_oom" || code == "upstream_unreachable" {
		t.Fatalf("stall misclassified as %q", code)
	}
}

// upstreamErrorKind is the single classifier the counter and the FAILED debug line share; its
// ladder must match upstreamErrorStatus so the metric and the client status never disagree.
func TestUpstreamErrorKind_ClassifiesEveryArm(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"stalled", &agent.UpstreamStalledError{Idle: time.Second}, "stalled"},
		{"unreachable", &agent.UpstreamUnreachableError{Err: http.ErrServerClosed}, "unreachable"},
		{"rate_limited", &agent.UpstreamStatusError{Status: 429}, "rate_limited"},
		{"auth", &agent.UpstreamStatusError{Status: 401}, "auth"},
		{"forbidden", &agent.UpstreamStatusError{Status: 403}, "forbidden"},
		{"status_4xx", &agent.UpstreamStatusError{Status: 404}, "status_4xx"},
		{"status_4xx_400", &agent.UpstreamStatusError{Status: 400}, "status_4xx"},
		{"status_5xx", &agent.UpstreamStatusError{Status: 503}, "status_5xx"},
		{"other", http.ErrHandlerTimeout, "other"},
	}
	for _, c := range cases {
		if got := upstreamErrorKind(c.err); got != c.want {
			t.Errorf("upstreamErrorKind(%s) = %q, want %q", c.name, got, c.want)
		}
	}
}

// observeUpstreamError must increment the per-kind counter exactly once per call and ignore a
// nil/unclassifiable-as-empty error, so the /metrics scrape reflects WHY turns failed.
func TestObserveUpstreamError_CountsByKind(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	m.observeUpstreamError(&agent.UpstreamStalledError{Idle: time.Second})
	m.observeUpstreamError(&agent.UpstreamStalledError{Idle: time.Second})
	m.observeUpstreamError(&agent.UpstreamStatusError{Status: 429}) // rate_limited, NOT status_4xx
	m.observeUpstreamError(&agent.UpstreamStatusError{Status: 404}) // generic 4xx
	m.observeUpstreamError(nil)                                     // no-op

	m.upstreamErrMu.Lock()
	defer m.upstreamErrMu.Unlock()
	if m.upstreamErrors["stalled"] != 2 {
		t.Fatalf("stalled count = %d, want 2", m.upstreamErrors["stalled"])
	}
	// A 429 must count as the distinct rate_limited kind — an operator scraping /metrics can
	// tell a rate-limit storm apart from a generic 4xx, which is the whole point of the split.
	if m.upstreamErrors["rate_limited"] != 1 {
		t.Fatalf("rate_limited count = %d, want 1", m.upstreamErrors["rate_limited"])
	}
	if m.upstreamErrors["status_4xx"] != 1 {
		t.Fatalf("status_4xx count = %d, want 1", m.upstreamErrors["status_4xx"])
	}
	if _, ok := m.upstreamErrors[""]; ok {
		t.Fatal("a nil error must not create an empty-kind counter")
	}
}

// The 401 token-rotation self-heal counter must (a) count by outcome, (b) ignore an unknown
// outcome so a caller typo cannot create a junk series, and (c) render BOTH outcome series on
// /metrics — even at 0 — so a dashboard panel exists from the first scrape and a missing
// "exhausted" series is never mistaken for "no login failures".
func TestObserveUpstreamAuthRefresh_CountsAndRenders(t *testing.T) {
	m := newGatewayMetrics(time.Now())

	m.observeUpstreamAuthRefresh("recovered")
	m.observeUpstreamAuthRefresh("recovered")
	m.observeUpstreamAuthRefresh("exhausted")
	m.observeUpstreamAuthRefresh("bogus") // unknown outcome: must be ignored.

	m.upstreamErrMu.Lock()
	if m.upstreamAuthRefreshes["recovered"] != 2 {
		t.Fatalf("recovered count = %d, want 2", m.upstreamAuthRefreshes["recovered"])
	}
	if m.upstreamAuthRefreshes["exhausted"] != 1 {
		t.Fatalf("exhausted count = %d, want 1", m.upstreamAuthRefreshes["exhausted"])
	}
	if _, ok := m.upstreamAuthRefreshes["bogus"]; ok {
		t.Fatal("an unknown outcome must not create a junk series")
	}
	m.upstreamErrMu.Unlock()

	var b strings.Builder
	m.writeUpstreamErrorMetrics(&b)
	out := b.String()
	for _, want := range []string{
		`fak_gateway_upstream_auth_refresh_total{outcome="recovered"} 2`,
		`fak_gateway_upstream_auth_refresh_total{outcome="exhausted"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("/metrics did not render %q:\n%s", want, out)
		}
	}
}

func TestDebugVarsExposeUpstreamIncidents(t *testing.T) {
	srv := newTestServer(t)
	srv.metrics.observeUpstreamError(&agent.UpstreamStatusError{Status: 401})
	srv.metrics.observeUpstreamError(&agent.UpstreamStatusError{Status: 429})
	srv.metrics.observeUpstreamAuthRefresh("exhausted")
	srv.metrics.observeUpstreamRetry(30 * time.Second)
	srv.metrics.observeUpstreamRetry(90 * time.Second)

	vars := srv.debugVars(time.Now())
	if vars.Upstream.ErrorsByKind["auth"] != 1 {
		t.Fatalf("/debug/vars upstream auth count = %d, want 1: %+v", vars.Upstream.ErrorsByKind["auth"], vars.Upstream)
	}
	if vars.Upstream.ErrorsByKind["rate_limited"] != 1 {
		t.Fatalf("/debug/vars upstream rate_limited count = %d, want 1: %+v", vars.Upstream.ErrorsByKind["rate_limited"], vars.Upstream)
	}
	if vars.Upstream.AuthRefreshByOutcome["exhausted"] != 1 {
		t.Fatalf("/debug/vars auth-refresh exhausted = %d, want 1: %+v", vars.Upstream.AuthRefreshByOutcome["exhausted"], vars.Upstream)
	}
	if _, ok := vars.Upstream.AuthRefreshByOutcome["recovered"]; !ok {
		t.Fatalf("/debug/vars must carry recovered even at zero: %+v", vars.Upstream.AuthRefreshByOutcome)
	}
	if vars.Upstream.Retries != 2 {
		t.Fatalf("/debug/vars upstream retries = %d, want 2: %+v", vars.Upstream.Retries, vars.Upstream)
	}
	if vars.Upstream.RetryWaitSeconds != 120 {
		t.Fatalf("/debug/vars retry wait = %v, want 120 (30s + 90s slept)", vars.Upstream.RetryWaitSeconds)
	}
}

// TestUpstreamRetryWaitSecondsRenders proves the retry family carries its TIME twin: the
// count says how often fak absorbed provider pushback, the wait counter says how much
// wall-clock that absorption cost — and it renders at 0 on a pushback-free session so the
// panel exists from the first scrape.
func TestUpstreamRetryWaitSecondsRenders(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeUpstreamRetry(30 * time.Second)
	m.observeUpstreamRetry(90 * time.Second)
	var b strings.Builder
	m.writeUpstreamErrorMetrics(&b)
	got := b.String()
	if !strings.Contains(got, "fak_gateway_upstream_retries_total 2") {
		t.Fatalf("retry count missing:\n%s", got)
	}
	if !strings.Contains(got, "fak_gateway_upstream_retry_wait_seconds_total 120") {
		t.Fatalf("retry wait seconds missing:\n%s", got)
	}

	var zero strings.Builder
	newGatewayMetrics(time.Now()).writeUpstreamErrorMetrics(&zero)
	if !strings.Contains(zero.String(), "fak_gateway_upstream_retry_wait_seconds_total 0") {
		t.Fatalf("zero-state retry wait row missing:\n%s", zero.String())
	}
}

// Both auth-refresh outcome series must render at 0 on a fresh metrics object — the panel must
// exist before any 401 happens, so an empty "exhausted" series can never read as a missing
// failure signal.
func TestUpstreamAuthRefreshRendersBothSeriesAtZero(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	var b strings.Builder
	m.writeUpstreamErrorMetrics(&b)
	out := b.String()
	for _, want := range []string{
		`fak_gateway_upstream_auth_refresh_total{outcome="recovered"} 0`,
		`fak_gateway_upstream_auth_refresh_total{outcome="exhausted"} 0`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("/metrics must render %q even at zero:\n%s", want, out)
		}
	}
}

// The upstream-error counter splits the operationally-distinct 4xx conditions into named kinds so
// a /metrics scrape can see a rate-limit storm vs an auth-failure storm vs a permission denial,
// not just a single status_4xx blob.
func TestUpstreamErrorsRenderDistinctKinds(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeUpstreamError(&agent.UpstreamStatusError{Status: 429})
	m.observeUpstreamError(&agent.UpstreamStatusError{Status: 401})
	m.observeUpstreamError(&agent.UpstreamStatusError{Status: 403})

	var b strings.Builder
	m.writeUpstreamErrorMetrics(&b)
	out := b.String()
	for _, want := range []string{
		`fak_gateway_upstream_errors_total{kind="rate_limited"} 1`,
		`fak_gateway_upstream_errors_total{kind="auth"} 1`,
		`fak_gateway_upstream_errors_total{kind="forbidden"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("/metrics did not render %q:\n%s", want, out)
		}
	}
}

// The counter family must render on the /metrics text scrape with the kind label, so a stall is
// scrapeable as fak_gateway_upstream_errors_total{kind="stalled"}.
func TestUpstreamErrorsRenderOnMetrics(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeUpstreamError(&agent.UpstreamStalledError{Idle: time.Second})

	var b strings.Builder
	m.writeUpstreamErrorMetrics(&b)
	out := b.String()
	if !strings.Contains(out, `fak_gateway_upstream_errors_total{kind="stalled"} 1`) {
		t.Fatalf("/metrics did not render the stalled upstream-error counter:\n%s", out)
	}
	if !strings.Contains(out, "# TYPE fak_gateway_upstream_errors_total counter") {
		t.Fatalf("/metrics missing the counter HELP/TYPE header:\n%s", out)
	}
}
