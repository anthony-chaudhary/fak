package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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

func TestRenderGuardInfoLineAssumptionsLedgerGolden(t *testing.T) {
	var v guardInfoVars
	v.Assumptions = []gateway.SessionAssumption{
		{TraceID: "run-1", Key: "gpu-route", Statement: "lab GPU pool", Source: "inferred", Confidence: 0.60, Expiry: "next reset"},
		{TraceID: "run-1", Key: "deployment-target", Statement: "staging", Source: "user_stated", Confidence: 0.95, Expiry: "session", SourceRef: "user:turn-3"},
		{TraceID: "run-1", Key: "budget", Statement: "2h wall clock", Source: "queried", Confidence: 1, Expiry: "2026-07-01T12:00:00Z", SourceRef: "answer:turn-4"},
	}

	got := renderGuardInfoLine(v)
	want := "assumptions: 3 active — queried budget=2h wall clock (100%, expires 2026-07-01T12:00:00Z, from answer:turn-4); user-stated deployment-target=staging (95%, expires session, from user:turn-3); inferred gpu-route=lab GPU pool (60%, expires next reset)"
	if !strings.Contains(got, want) {
		t.Fatalf("assumption ledger render mismatch:\nwant suffix: %s\ngot: %s", want, got)
	}
}

func TestGuardInfoVarsDecodesAssumptionsLedger(t *testing.T) {
	raw := []byte(`{
		"assumptions":[
			{"trace_id":"run-1","key":"deployment-target","statement":"staging","source":"user_stated","confidence":0.95,"expiry":"session","source_ref":"user:turn-3"},
			{"trace_id":"run-1","key":"gpu-route","statement":"lab GPU pool","source":"inferred","confidence":0.60,"expiry":"next reset"},
			{"trace_id":"run-1","key":"budget","statement":"2h wall clock","source":"queried","confidence":1,"expiry":"2026-07-01T12:00:00Z","source_ref":"answer:turn-4"}
		]
	}`)
	var v guardInfoVars
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("decode guardInfoVars assumptions block: %v", err)
	}
	if len(v.Assumptions) != 3 {
		t.Fatalf("assumptions decoded %d rows, want 3: %+v", len(v.Assumptions), v.Assumptions)
	}
	if v.Assumptions[0].Source != "user_stated" || v.Assumptions[2].Expiry == "" {
		t.Fatalf("assumptions fields not decoded: %+v", v.Assumptions)
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

// --- issue #1602: managed-context prefix-stability score ---

func TestGuardInfoPrefixStabilityTextNilIsSilent(t *testing.T) {
	// A gateway build that has not wired PrefixStability yet must render NOTHING extra —
	// distinct from an explicit "unknown" verdict, which DOES render (see below).
	if got := guardInfoPrefixStabilityText(nil); got != "" {
		t.Fatalf("nil prefix-stability block should render empty, got %q", got)
	}
}

func TestGuardInfoPrefixStabilityTextStates(t *testing.T) {
	cases := []struct {
		name string
		p    *guardInfoPrefixStability
		want []string
	}{
		{
			name: "stable",
			p:    &guardInfoPrefixStability{State: string(cachemeta.PrefixStable)},
			want: []string{"prefix: stable"},
		},
		{
			name: "unknown",
			p:    &guardInfoPrefixStability{State: string(cachemeta.PrefixUnknown)},
			want: []string{"prefix: unknown", "no baseline yet"},
		},
		{
			name: "mutated",
			p: &guardInfoPrefixStability{
				State:                     string(cachemeta.PrefixMutated),
				FirstDivergentSegment:     1,
				FirstDivergentTokenOffset: 100,
				FirstDivergentKind:        string(cachemeta.SegToolSchema),
			},
			want: []string{"prefix: mutated", "segment 1", "offset 100 tokens", "tool_schema"},
		},
		{
			name: "mutated-sealed",
			p: &guardInfoPrefixStability{
				State:                 string(cachemeta.PrefixMutated),
				FirstDivergentSegment: 1,
				ProtectedSpanBroken:   true,
			},
			want: []string{"prefix: mutated", "[sealed]"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardInfoPrefixStabilityText(tc.p)
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Fatalf("%s: text %q missing %q", tc.name, got, want)
				}
			}
		})
	}
}

// TestRenderGuardInfoLineAppendsPrefixStability proves the status line grows a
// "· prefix: ..." suffix once the gateway reports the block, and stays unchanged (no
// trailing " · " or stray text) when the block is absent.
func TestRenderGuardInfoLineAppendsPrefixStability(t *testing.T) {
	var v guardInfoVars
	v.Inference.Turns = 1
	base := renderGuardInfoLine(v)
	if strings.Contains(base, "prefix:") {
		t.Fatalf("no PrefixStability block should not mention prefix: %s", base)
	}

	v.PrefixStability = &guardInfoPrefixStability{State: string(cachemeta.PrefixStable)}
	withPrefix := renderGuardInfoLine(v)
	if !strings.Contains(withPrefix, "prefix: stable") {
		t.Fatalf("line missing prefix-stability suffix: %s", withPrefix)
	}
	if !strings.HasPrefix(withPrefix, base) {
		t.Fatalf("prefix-stability text should be appended, not alter the existing line: base=%q got=%q", base, withPrefix)
	}
}

// TestGuardInfoVarsDecodesPrefixStability mirrors TestGuardInfoVarsDecodesCacheAttribution:
// the JSON wire shape must round-trip field-for-field, and the block must stay a nil
// pointer (omitted) when absent so "no field" and "explicitly unknown" are distinguishable.
func TestGuardInfoVarsDecodesPrefixStability(t *testing.T) {
	raw := []byte(`{
		"prefix_stability":{
			"state":"prefix-mutated",
			"first_divergent_segment":2,
			"first_divergent_token_offset":150,
			"first_divergent_kind":"tool_schema",
			"protected_span_broken":false,
			"reason":"protected span diverged at segment 2 (token offset 150)"
		}
	}`)
	var v guardInfoVars
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("decode guardInfoVars prefix_stability block: %v", err)
	}
	if v.PrefixStability == nil {
		t.Fatal("prefix_stability did not decode")
	}
	if v.PrefixStability.State != "prefix-mutated" || v.PrefixStability.FirstDivergentSegment != 2 || v.PrefixStability.FirstDivergentTokenOffset != 150 {
		t.Fatalf("prefix_stability fields not decoded: %+v", v.PrefixStability)
	}

	var absent guardInfoVars
	if err := json.Unmarshal([]byte(`{}`), &absent); err != nil {
		t.Fatalf("decode empty object: %v", err)
	}
	if absent.PrefixStability != nil {
		t.Fatalf("prefix_stability must stay nil when the gateway omits the block, got %+v", absent.PrefixStability)
	}
}

// TestGuardInfoPrefixStabilityFromScoreLowersAllFields proves the live-score-to-wire-shape
// lowering used by a future gateway wire-up (and by runInfoPrefixTranscript) carries every
// field the renderer reads.
func TestGuardInfoPrefixStabilityFromScoreLowersAllFields(t *testing.T) {
	score := cachemeta.PrefixStabilityScore{
		State:                     cachemeta.PrefixMutated,
		FirstDivergentSegment:     3,
		FirstDivergentTokenOffset: 42,
		FirstDivergentKind:        cachemeta.SegSealed,
		ProtectedSpanBroken:       true,
		Reason:                    "protected span sealed",
	}
	got := guardInfoPrefixStabilityFromScore(score)
	if got.State != string(cachemeta.PrefixMutated) || got.FirstDivergentSegment != 3 || got.FirstDivergentTokenOffset != 42 {
		t.Fatalf("lowered fields mismatch: %+v", got)
	}
	if got.FirstDivergentKind != string(cachemeta.SegSealed) || !got.ProtectedSpanBroken || got.Reason != score.Reason {
		t.Fatalf("lowered fields mismatch: %+v", got)
	}
}

// TestGuardInfoManagedContextTextNilIsSilent mirrors
// TestGuardInfoPrefixStabilityTextNilIsSilent: a gateway build that has not wired
// issue #1577's managed-context tracker yet must render NOTHING extra, not a
// fabricated all-zero status line.
func TestGuardInfoManagedContextTextNilIsSilent(t *testing.T) {
	if got := guardInfoManagedContextText(nil); got != "" {
		t.Fatalf("nil managed-context block should render empty, got %q", got)
	}
}

// TestGuardInfoManagedContextTextRendersEachSeverity proves the `fak info` line
// reflects the closed #1579 severity for a live session, and that the six named
// signals (issue #1577's In-scope list) all appear in the rendered text — the same
// contract internal/scorecardpane.RenderContextStatusLine's own tests prove directly,
// checked here at the guardInfoVars wire-shape seam.
func TestGuardInfoManagedContextTextRendersEachSeverity(t *testing.T) {
	cases := []struct {
		name     string
		severity string
	}{
		{"fresh", "fresh"},
		{"constrained", "constrained"},
		{"query_needed", "query_needed"},
		{"stale_risk", "stale_risk"},
		{"budget_draining", "budget_draining"},
		{"reset_imminent", "reset_imminent"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &guardInfoManagedContext{
				Severity: tc.severity, ResidentTokens: 4000, BudgetTokens: 10000,
				CacheState: "stable", ResetCount: 1, QueryNeededCount: 2, StaleAssumptionCount: 3,
			}
			got := guardInfoManagedContextText(m)
			for _, want := range []string{
				"context: " + tc.severity, "resident 4,000 tok", "budget left 6,000 tok",
				"cache stable", "resets 1", "query-needed 2", "stale 3",
			} {
				if !strings.Contains(got, want) {
					t.Errorf("%s: text %q missing %q", tc.name, got, want)
				}
			}
		})
	}
}

// TestRenderGuardInfoLineAppendsManagedContext proves the status line grows a
// "· context: ..." suffix once the gateway reports the managed-context block, and
// stays unchanged when the block is absent — the same append-don't-alter contract
// TestRenderGuardInfoLineAppendsPrefixStability proves for the prefix-stability
// suffix, and the concrete evidence for the issue's done condition ("fak info ...
// show a one-line managed-context status for live sessions").
func TestRenderGuardInfoLineAppendsManagedContext(t *testing.T) {
	var v guardInfoVars
	v.Inference.Turns = 1
	base := renderGuardInfoLine(v)
	if strings.Contains(base, "context:") {
		t.Fatalf("no ManagedContext block should not mention context:: %s", base)
	}

	v.ManagedContext = &guardInfoManagedContext{Severity: "fresh", ResidentTokens: 100, BudgetTokens: 1000}
	withCtx := renderGuardInfoLine(v)
	if !strings.Contains(withCtx, "context: fresh") {
		t.Fatalf("line missing managed-context suffix: %s", withCtx)
	}
	if !strings.HasPrefix(withCtx, base) {
		t.Fatalf("managed-context text should be appended, not alter the existing line: base=%q got=%q", base, withCtx)
	}
}

// TestGuardInfoVarsDecodesManagedContext mirrors TestGuardInfoVarsDecodesPrefixStability:
// the JSON wire shape must round-trip field-for-field, and the block must stay a nil
// pointer (omitted) when absent so "no field yet" and "explicitly zero" stay
// distinguishable.
func TestGuardInfoVarsDecodesManagedContext(t *testing.T) {
	raw := []byte(`{
		"managed_context":{
			"severity":"budget_draining",
			"resident_tokens":8000,
			"budget_tokens":10000,
			"cache_state":"mutated",
			"reset_count":2,
			"query_needed_count":5,
			"stale_assumption_count":1
		}
	}`)
	var v guardInfoVars
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("decode guardInfoVars managed_context block: %v", err)
	}
	if v.ManagedContext == nil {
		t.Fatal("managed_context did not decode")
	}
	if v.ManagedContext.Severity != "budget_draining" || v.ManagedContext.ResidentTokens != 8000 ||
		v.ManagedContext.BudgetTokens != 10000 || v.ManagedContext.CacheState != "mutated" ||
		v.ManagedContext.ResetCount != 2 || v.ManagedContext.QueryNeededCount != 5 ||
		v.ManagedContext.StaleAssumptionCount != 1 {
		t.Fatalf("managed_context fields not decoded: %+v", v.ManagedContext)
	}

	var absent guardInfoVars
	if err := json.Unmarshal([]byte(`{}`), &absent); err != nil {
		t.Fatalf("decode empty object: %v", err)
	}
	if absent.ManagedContext != nil {
		t.Fatalf("managed_context must stay nil when the gateway omits the block, got %+v", absent.ManagedContext)
	}
}

// writePrefixTranscriptFixture writes a minimal Claude-Code-style JSONL transcript with
// THREE assistant turns: turn 1 seeds the baseline (system + tool schema), turn 2 repeats
// the same system+tool schema byte-for-byte (stable), and turn 3 changes the tool schema
// (mutated) — the same three-state arc prefix_score_test.go exercises directly on the
// tracker, but round-tripped through the JSONL parsing path this file adds.
func writePrefixTranscriptFixture(t *testing.T) string {
	t.Helper()
	lines := []string{
		`{"message":{"role":"system","content":"You are a coding agent. Follow the rules."}}`,
		`{"message":{"role":"assistant","content":[{"type":"tool_use","content":"{\"tools\":[\"read\",\"write\"]}"}]}}`,
		`{"message":{"role":"user","content":"do the first thing"}}`,
		`{"message":{"role":"assistant","content":[{"type":"text","text":"ok, done"}]}}`,
		`{"message":{"role":"user","content":"do the second thing"}}`,
		`{"message":{"role":"assistant","content":[{"type":"text","text":"ok, also done"}]}}`,
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "transcript.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// TestRunInfoPrefixTranscriptReportsStates proves the `--prefix-transcript` offline path
// (issue #1602's done condition: "a live session reports prefix-stable, prefix-mutated,
// or unknown with the first divergent span") actually computes and prints a verdict for a
// recorded transcript with no gateway involved.
func TestRunInfoPrefixTranscriptReportsStates(t *testing.T) {
	path := writePrefixTranscriptFixture(t)
	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--prefix-transcript", path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "turn 1") || !strings.Contains(out, string(cachemeta.PrefixUnknown)) {
		t.Fatalf("turn 1 should report unknown (seeds the baseline):\n%s", out)
	}
	if !strings.Contains(out, "summary:") {
		t.Fatalf("missing summary line:\n%s", out)
	}
}

// TestRunInfoPrefixTranscriptJSON proves --json emits a decodable prefixTranscriptReport
// whose turns carry the real cachemeta.PrefixStabilityScore fields (not just prose).
func TestRunInfoPrefixTranscriptJSON(t *testing.T) {
	path := writePrefixTranscriptFixture(t)
	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--prefix-transcript", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	var report prefixTranscriptReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("json output did not round-trip: %v\n%s", err, stdout.String())
	}
	if len(report.Turns) == 0 {
		t.Fatalf("no turns decoded from JSON report:\n%s", stdout.String())
	}
	if report.Turns[0].Score.State != cachemeta.PrefixUnknown {
		t.Fatalf("turn 1 state = %q, want %q", report.Turns[0].Score.State, cachemeta.PrefixUnknown)
	}
	if report.Summary == nil {
		t.Fatal("summary not populated")
	}
}

// TestRunInfoPrefixTranscriptMissingFileFails proves a bad path is a clean, reported
// failure (exit 1 with a stderr message) rather than a panic or a silent empty report.
func TestRunInfoPrefixTranscriptMissingFileFails(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--prefix-transcript", filepath.Join(t.TempDir(), "does-not-exist.jsonl")})
	if code == 0 {
		t.Fatalf("missing transcript should fail, got exit 0: stdout=%s", stdout.String())
	}
	if stderr.String() == "" {
		t.Fatal("missing transcript should report an error on stderr")
	}
}

// TestProtectedSpanOfCapsAtFirstMessage proves protectedSpanOf keeps only the leading
// stable/tool-schema/volatile run (and a capping sealed span), never spilling into
// ordinary message/tool-result content — the span PrefixStabilityTracker is meant to
// score turn over turn.
func TestProtectedSpanOfCapsAtFirstMessage(t *testing.T) {
	turn := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100},
		{Kind: cachemeta.SegToolSchema, Tokens: 200},
		{Kind: cachemeta.SegMessage, Tokens: 10},
		{Kind: cachemeta.SegToolResult, Tokens: 20},
	}
	got := protectedSpanOf(turn)
	if len(got) != 2 {
		t.Fatalf("protectedSpanOf returned %d segments, want 2 (stable+tool-schema only): %+v", len(got), got)
	}

	sealedTurn := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100},
		{Kind: cachemeta.SegSealed, Tokens: 50},
		{Kind: cachemeta.SegMessage, Tokens: 10},
	}
	gotSealed := protectedSpanOf(sealedTurn)
	if len(gotSealed) != 2 {
		t.Fatalf("protectedSpanOf with a sealed cap returned %d segments, want 2 (stable+sealed): %+v", len(gotSealed), gotSealed)
	}
	if gotSealed[1].Kind != cachemeta.SegSealed {
		t.Fatalf("protectedSpanOf must include the capping sealed segment itself, got %+v", gotSealed[1])
	}
}
