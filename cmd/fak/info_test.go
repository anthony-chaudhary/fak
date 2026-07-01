package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestGroupThousands(t *testing.T) {
	cases := map[int64]string{0: "0", 12: "12", 123: "123", 1234: "1,234", 12345: "12,345", 1234567: "1,234,567"}
	for in, want := range cases {
		if got := groupThousands(in); got != want {
			t.Fatalf("groupThousands(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestSignedTokens(t *testing.T) {
	if got := signedTokens(12345); got != "+12,345" {
		t.Fatalf("positive saving = %q, want +12,345", got)
	}
	// Negative is the load-bearing case: net saving is below zero until reads repay writes.
	if got := signedTokens(-1234); got != "-1,234" {
		t.Fatalf("negative saving = %q, want -1,234", got)
	}
}

func TestRenderGuardInfoLineNoCacheCleanFloor(t *testing.T) {
	var v guardInfoVars
	v.Inference.Turns = 4
	v.Gateway.InflightRequests = 0
	line := renderGuardInfoLine(v)
	if !strings.Contains(line, "cache: nothing yet") {
		t.Fatalf("no-cache should read 'cache: nothing yet', got: %s", line)
	}
	if !strings.Contains(line, "safety: nothing blocked") {
		t.Fatalf("zero refusals should read 'safety: nothing blocked', got: %s", line)
	}
	if !strings.Contains(line, "replies 4") {
		t.Fatalf("missing replies: %s", line)
	}
}

func TestRenderGuardInfoLineProvenCacheAndSafety(t *testing.T) {
	var v guardInfoVars
	v.Kernel.Denies = 1
	v.Kernel.Transforms = 2
	v.Kernel.Quarantines = 1
	v.Kernel.ResultDenies = 1 // folds into quarantined => 2
	v.VCache = &struct {
		CacheReadTokens int64   `json:"cache_read_tokens"`
		SavedTokenEquiv float64 `json:"saved_token_equiv"`
		HitRate         float64 `json:"hit_rate"`
		Multiplier      float64 `json:"multiplier"`
		Status          string  `json:"status"`
	}{CacheReadTokens: 1000, SavedTokenEquiv: 12345, HitRate: 0.88, Multiplier: 2.1, Status: "PROVEN"}

	line := renderGuardInfoLine(v)
	for _, want := range []string{"saving money", "×2.10 cheaper", "+12,345 tokens", "reused 88%", "blocked 1", "fixed 2", "set aside 2"} {
		if !strings.Contains(line, want) {
			t.Fatalf("line missing %q:\n%s", want, line)
		}
	}
}

func TestRenderGuardInfoLineCacheAttributionSplit(t *testing.T) {
	v := provenVisualVars()
	v.CacheAttribution = cacheAttributionFixture(80, 20, 100)

	line := renderGuardInfoLine(v)
	for _, want := range []string{
		"split default cache 80%",
		"fak 20%",
		"~80 tok",
		"~20 tok",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("cache attribution line missing %q:\n%s", want, line)
		}
	}
}

// debugVarsStub returns a gateway whose /debug/vars matches the guardInfoVars shape.
func debugVarsStub(t *testing.T) *httptest.Server {
	t.Helper()
	const body = `{
		"gateway":{"uptime_seconds":42,"inflight_requests":1,"vdso":true},
		"kernel":{"submits":3,"admitted":2,"denies":1,"transforms":0,"quarantines":1,"result_denies":0},
		"inference":{"turns":5},
		"vcache":{"cache_read_tokens":1000,"saved_token_equiv":12345,"hit_rate":0.88,"multiplier":2.1,"status":"PROVEN"}
	}`
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
}

func TestRunInfoOnceRendersLine(t *testing.T) {
	srv := debugVarsStub(t)
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL, "--once"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"saving money", "×2.10 cheaper", "blocked 1", "set aside 1", "replies 5", "busy with 1", "running 42s"} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %q:\n%s", want, out)
		}
	}
	// --once is a quiet one-shot probe: ONE status line, no standing header and no guide
	// (those belong to the watch loop). A probe that prints 5 lines of guide to then report
	// one number is the pane-spam this command exists to avoid.
	if strings.Contains(out, "fak info · ") || strings.Contains(out, "what this means:") {
		t.Fatalf("--once must not print the header/guide:\n%s", out)
	}
	if got := strings.Count(strings.TrimRight(out, "\n"), "\n"); got != 0 {
		t.Fatalf("--once must print exactly one line, got %d extra newlines:\n%s", got, out)
	}
}

// healthyThenGoneClient returns a debug client whose first `serveHealthy` gets succeed and
// every get after that fails — modeling a guarded session that ends mid-watch (the gateway is
// torn down). It lets the overlay loop run a few real ticks then hit the close path, with no
// sleeping: the stub server is closed after the healthy gets are drained.
func healthyThenGoneClient(t *testing.T, serveHealthy int) *claudeMacDebugClient {
	t.Helper()
	srv := debugVarsStub(t)
	hits := 0
	// Wrap the stub: count healthy responses; once we've served enough, close the server so
	// subsequent dials are refused (the "session ended" signal the overlay watches for).
	mux := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits > serveHealthy {
			http.Error(w, "gone", http.StatusServiceUnavailable)
			return
		}
		resp, err := http.Get(srv.URL + r.URL.Path)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer func() { _ = resp.Body.Close() }()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.Copy(w, resp.Body)
	}))
	t.Cleanup(mux.Close)
	base, err := normalizeTUIAgentGatewayURL(mux.URL)
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	return &claudeMacDebugClient{base: base, hc: &http.Client{Timeout: 2 * time.Second}}
}

// TestRunInfoOverlayNonTTYAppends proves the off-TTY (piped/redirected) path appends one
// whole, newline-terminated status line per tick — the log-friendly mode — and ends on the
// gateway-closed line rather than spinning. It must NOT emit any in-place redraw escape.
func TestRunInfoOverlayNonTTYAppends(t *testing.T) {
	c := healthyThenGoneClient(t, 1)
	var stdout, stderr bytes.Buffer
	code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false /*once*/, false /*tty*/, 0 /*width*/, 0 /*height*/, "line")
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if strings.Contains(out, "\r") || strings.Contains(out, "\033[K") {
		t.Fatalf("non-TTY output must not use in-place redraw escapes:\n%q", out)
	}
	if !strings.Contains(out, "what this means:") {
		t.Fatalf("watch loop must print the guide once:\n%s", out)
	}
	if !strings.Contains(out, "fak info: gateway closed") {
		t.Fatalf("must end on the gateway-closed line:\n%s", out)
	}
}

// TestRunInfoOverlayTTYRedrawsInPlace proves the TTY path overwrites a single status row each
// tick (\r + clear-to-EOL) instead of scrolling — the signal/noise fix. The closing note still
// breaks to its own clean row so the parked line is not clobbered.
func TestRunInfoOverlayTTYRedrawsInPlace(t *testing.T) {
	c := healthyThenGoneClient(t, 2)
	var stdout, stderr bytes.Buffer
	code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false /*once*/, true /*tty*/, 0 /*width*/, 0 /*height*/, "line")
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "\r\033[K") {
		t.Fatalf("TTY output must redraw in place (\\r + clear-to-EOL):\n%q", out)
	}
	if !strings.Contains(out, "fak info: gateway closed") {
		t.Fatalf("must end on the gateway-closed line:\n%s", out)
	}
}

func TestRunInfoJSONSnapshot(t *testing.T) {
	srv := debugVarsStub(t)
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	var v guardInfoVars
	if err := json.Unmarshal(stdout.Bytes(), &v); err != nil {
		t.Fatalf("json output did not round-trip: %v\n%s", err, stdout.String())
	}
	if v.VCache == nil || v.VCache.Status != "PROVEN" {
		t.Fatalf("vcache not decoded: %+v", v.VCache)
	}
	if v.Kernel.Denies != 1 || v.Inference.Turns != 5 {
		t.Fatalf("kernel/inference not decoded: %+v / %+v", v.Kernel, v.Inference)
	}
}

func TestGuardInfoVarsDecodesUpstreamIncidents(t *testing.T) {
	raw := []byte(`{
		"gateway":{"uptime_seconds":42,"inflight_requests":1,"vdso":true},
		"kernel":{"submits":3,"admitted":2,"denies":1,"transforms":0,"quarantines":1,"result_denies":0},
		"inference":{"turns":5},
		"upstream":{
			"errors_by_kind":{"auth":1,"rate_limited":2},
			"retries":3,
			"auth_refresh_by_outcome":{"recovered":1,"exhausted":1}
		}
	}`)
	var v guardInfoVars
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("decode guardInfoVars upstream block: %v", err)
	}
	if v.Upstream.ErrorsByKind["auth"] != 1 || v.Upstream.ErrorsByKind["rate_limited"] != 2 {
		t.Fatalf("upstream errors not decoded: %+v", v.Upstream.ErrorsByKind)
	}
	if v.Upstream.AuthRefreshByOutcome["exhausted"] != 1 || v.Upstream.AuthRefreshByOutcome["recovered"] != 1 {
		t.Fatalf("auth-refresh outcomes not decoded: %+v", v.Upstream.AuthRefreshByOutcome)
	}
	if v.Upstream.Retries != 3 {
		t.Fatalf("upstream retries = %d, want 3", v.Upstream.Retries)
	}
}

func TestGuardInfoVarsDecodesCacheAttribution(t *testing.T) {
	raw := []byte(`{
		"cache_attribution":{
			"provider_token_equiv":80,
			"fak_token_equiv":20,
			"total_token_equiv":100,
			"provider_prompt_cache_read_token_equiv":120,
			"provider_prompt_cache_write_premium_token_equiv":-40,
			"fak_compaction_shed_tokens":15,
			"fak_kv_prefix_reused_tokens":5,
			"fak_vdso_avoided_calls":2
		}
	}`)
	var v guardInfoVars
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("decode guardInfoVars cache_attribution block: %v", err)
	}
	if v.CacheAttribution == nil {
		t.Fatal("cache_attribution did not decode")
	}
	if v.CacheAttribution.ProviderTokenEquiv != 80 || v.CacheAttribution.FakTokenEquiv != 20 || v.CacheAttribution.TotalTokenEquiv != 100 {
		t.Fatalf("cache_attribution owner totals not decoded: %+v", v.CacheAttribution)
	}
	if v.CacheAttribution.FakCompactionShedTokens != 15 || v.CacheAttribution.FakKVPrefixReusedTokens != 5 || v.CacheAttribution.FakVDSOAvoidedCalls != 2 {
		t.Fatalf("cache_attribution fak mechanisms not decoded: %+v", v.CacheAttribution)
	}
}

func cacheAttributionFixture(provider, fak, total float64) *guardInfoCacheAttribution {
	return &guardInfoCacheAttribution{ProviderTokenEquiv: provider, FakTokenEquiv: fak, TotalTokenEquiv: total}
}

func TestRunInfoRejectsBadInterval(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runInfo(&stdout, &stderr, []string{"--gateway-url", "http://127.0.0.1:1", "--interval", "0s"}); code != 2 {
		t.Fatalf("bad interval exit = %d, want 2", code)
	}
}

// TestGuardInfoFetchErrorLineFriendlyAndPassthrough pins the first-run UX: a "nothing is
// listening" error (the common case of running `fak info` before a `fak guard` is up) becomes a
// plain-words, actionable hint naming the URL and how to start a gateway — never the raw Go dial
// phrase — while a real fault from a gateway that IS answering (an HTTP status, an auth refusal)
// is passed through verbatim so it stays visible.
func TestGuardInfoFetchErrorLineFriendlyAndPassthrough(t *testing.T) {
	const base = "http://127.0.0.1:8080"

	for _, e := range []error{
		errors.New(`Get "http://127.0.0.1:8080/debug/vars": dial tcp 127.0.0.1:8080: connectex: No connection could be made because the target machine actively refused it.`),
		errors.New("dial tcp 127.0.0.1:8080: connect: connection refused"),
		errors.New(`Get "http://nope/debug/vars": dial tcp: lookup nope: no such host`),
		errors.New("context deadline exceeded (Client.Timeout exceeded while awaiting headers)"),
	} {
		if !guardInfoUnreachable(e) {
			t.Fatalf("should be classed unreachable: %v", e)
		}
		line := guardInfoFetchErrorLine(base, e)
		for _, want := range []string{"no fak gateway answering at " + base, "fak guard", "--gateway-url"} {
			if !strings.Contains(line, want) {
				t.Fatalf("friendly line missing %q: %s", want, line)
			}
		}
		// The raw Go dial phrase must NOT leak into the friendly hint.
		if strings.Contains(line, "dial tcp") || strings.Contains(line, "connectex") {
			t.Fatalf("friendly hint must not echo the raw net error: %s", line)
		}
	}

	// A gateway that IS answering (an HTTP status) is a real fault: pass it through verbatim and
	// do not class it as unreachable.
	httpErr := errors.New("GET /debug/vars: status 401")
	if guardInfoUnreachable(httpErr) {
		t.Fatalf("an HTTP-status error must not be classed unreachable: %v", httpErr)
	}
	if got, want := guardInfoFetchErrorLine(base, httpErr), "fak info: GET /debug/vars: status 401"; got != want {
		t.Fatalf("status error must pass through verbatim: got %q want %q", got, want)
	}
}
