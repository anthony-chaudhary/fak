package main

// dispatch_route_health_test.go — the #3035 witness matrix: every typed failure class the
// smoke gate must produce (timeout, 401/403 auth, 404 model, 429 with a provider reset
// hint, provider 5xx) plus the load-bearing suppression property — a failing route/model/
// account is suppressed for its cooldown while a healthy SIBLING route on the same
// provider/account keeps passing the gate.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestClassifyRouteProbeMatrix(t *testing.T) {
	now := int64(1_700_000_000)
	cases := []struct {
		name     string
		status   int
		header   http.Header
		body     string
		err      error
		class    routeHealthClass
		cooldown int64
		hint     bool
	}{
		{name: "healthy", status: 200, class: routeClassHealthy, cooldown: 0},
		{name: "auth-401", status: 401, class: routeClassAuth, cooldown: 3600},
		{name: "auth-403", status: 403, class: routeClassAuth, cooldown: 3600},
		{name: "model-404", status: 404, class: routeClassModelUnavailable, cooldown: 3600},
		{name: "retired-410", status: 410, class: routeClassUnsupported, cooldown: 21600},
		{
			name:   "rate-429-retry-after-seconds",
			status: 429,
			header: http.Header{"Retry-After": []string{"120"}},
			class:  routeClassRateLimited, cooldown: 120, hint: true,
		},
		{
			name:   "rate-429-reset-epoch",
			status: 429,
			header: http.Header{"X-Ratelimit-Reset": []string{fmt.Sprint(now + 300)}},
			class:  routeClassRateLimited, cooldown: 300, hint: true,
		},
		{
			name:   "rate-429-no-hint-default",
			status: 429,
			class:  routeClassRateLimited, cooldown: 900,
		},
		{name: "provider-500", status: 500, class: routeClassProvider5xx, cooldown: 300},
		{
			name:   "provider-503-retry-after",
			status: 503,
			header: http.Header{"Retry-After": []string{"45"}},
			class:  routeClassProvider5xx, cooldown: 45, hint: true,
		},
		{
			name:   "expired-400-retired-model",
			status: 400,
			body:   `{"error":{"message":"the model deepseek-old has been retired"}}`,
			class:  routeClassUnsupported, cooldown: 21600,
		},
		{
			name:   "missing-400-model-not-found",
			status: 400,
			body:   `{"error":{"message":"model kimi-x not found"}}`,
			class:  routeClassModelUnavailable, cooldown: 3600,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := classifyRouteProbe(tc.status, tc.header, tc.body, tc.err, now)
			if out.Class != tc.class {
				t.Fatalf("class = %s, want %s", out.Class, tc.class)
			}
			if out.CooldownSecs != tc.cooldown {
				t.Fatalf("cooldown = %d, want %d", out.CooldownSecs, tc.cooldown)
			}
			if out.ProviderHint != tc.hint {
				t.Fatalf("provider hint = %v, want %v", out.ProviderHint, tc.hint)
			}
		})
	}
}

func TestRouteRetryAfterSecs(t *testing.T) {
	now := int64(1_700_000_000)
	httpDate := time.Unix(now+90, 0).UTC().Format(http.TimeFormat)
	cases := []struct {
		name string
		h    http.Header
		want int64
		ok   bool
	}{
		{name: "none", h: http.Header{}, want: 0, ok: false},
		{name: "retry-after-delta", h: http.Header{"Retry-After": []string{"60"}}, want: 60, ok: true},
		{name: "retry-after-http-date", h: http.Header{"Retry-After": []string{httpDate}}, want: 90, ok: true},
		{name: "reset-epoch", h: http.Header{"X-Ratelimit-Reset": []string{fmt.Sprint(now + 240)}}, want: 240, ok: true},
		{name: "reset-delta", h: http.Header{"X-Ratelimit-Reset": []string{"30"}}, want: 30, ok: true},
		{name: "reset-duration", h: http.Header{"X-Ratelimit-Reset-Requests": []string{"6m30s"}}, want: 390, ok: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := routeRetryAfterSecs(tc.h, now)
			if got != tc.want || ok != tc.ok {
				t.Fatalf("routeRetryAfterSecs = (%d, %v), want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestProbeProviderRouteHTTPMatrix drives the real probe against hermetic HTTP servers:
// the classes flow end-to-end from wire response to ledger record, including the transport
// timeout the trigger incident hit.
func TestProbeProviderRouteHTTPMatrix(t *testing.T) {
	serve := func(status int, header map[string]string, body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/chat/completions" {
				t.Errorf("probe hit %s, want /chat/completions", r.URL.Path)
			}
			for k, v := range header {
				w.Header().Set(k, v)
			}
			w.WriteHeader(status)
			fmt.Fprint(w, body)
		}))
	}

	t.Run("healthy", func(t *testing.T) {
		srv := serve(200, nil, `{"choices":[{"message":{"role":"assistant","content":"OK"}}]}`)
		defer srv.Close()
		rec, err := probeProviderRoute(routeProbeSpec{
			Provider: "nim", Account: "seat1", Model: "kimi-k2",
			BaseURL: srv.URL, Timeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatalf("probe: %v", err)
		}
		if rec.Class != string(routeClassHealthy) || rec.Status != 200 {
			t.Fatalf("class=%s status=%d, want healthy/200", rec.Class, rec.Status)
		}
		if rec.CooldownUntilUnix != 0 {
			t.Fatalf("healthy probe must not set a cooldown, got %d", rec.CooldownUntilUnix)
		}
		if rec.Route != "nim/seat1/kimi-k2" {
			t.Fatalf("route key = %s", rec.Route)
		}
		if !strings.Contains(rec.Recheck, "fak dispatch route-health probe") ||
			!strings.Contains(rec.Recheck, "--model kimi-k2") {
			t.Fatalf("recheck command incomplete: %s", rec.Recheck)
		}
	})

	t.Run("auth-401", func(t *testing.T) {
		srv := serve(401, nil, `{"error":"invalid api key"}`)
		defer srv.Close()
		rec, err := probeProviderRoute(routeProbeSpec{Provider: "nim", Model: "m", BaseURL: srv.URL, Timeout: 5 * time.Second})
		if err != nil {
			t.Fatalf("probe: %v", err)
		}
		if rec.Class != string(routeClassAuth) {
			t.Fatalf("class = %s, want auth", rec.Class)
		}
	})

	t.Run("model-404", func(t *testing.T) {
		srv := serve(404, nil, `{"error":"model not found"}`)
		defer srv.Close()
		rec, err := probeProviderRoute(routeProbeSpec{Provider: "nim", Model: "gone", BaseURL: srv.URL, Timeout: 5 * time.Second})
		if err != nil {
			t.Fatalf("probe: %v", err)
		}
		if rec.Class != string(routeClassModelUnavailable) {
			t.Fatalf("class = %s, want model_unavailable", rec.Class)
		}
	})

	t.Run("rate-429-with-reset", func(t *testing.T) {
		srv := serve(429, map[string]string{"Retry-After": "120"}, `{"error":"rate limited"}`)
		defer srv.Close()
		rec, err := probeProviderRoute(routeProbeSpec{Provider: "nim", Model: "m", BaseURL: srv.URL, Timeout: 5 * time.Second})
		if err != nil {
			t.Fatalf("probe: %v", err)
		}
		if rec.Class != string(routeClassRateLimited) {
			t.Fatalf("class = %s, want rate_limited", rec.Class)
		}
		if !rec.ProviderHint {
			t.Fatal("Retry-After hint not honored")
		}
		if got := rec.CooldownUntilUnix - rec.ProbedAtUnix; got != 120 {
			t.Fatalf("cooldown window = %ds, want 120s from Retry-After", got)
		}
	})

	t.Run("provider-5xx", func(t *testing.T) {
		srv := serve(500, nil, `{"error":"internal"}`)
		defer srv.Close()
		rec, err := probeProviderRoute(routeProbeSpec{Provider: "nim", Model: "m", BaseURL: srv.URL, Timeout: 5 * time.Second})
		if err != nil {
			t.Fatalf("probe: %v", err)
		}
		if rec.Class != string(routeClassProvider5xx) {
			t.Fatalf("class = %s, want provider_5xx", rec.Class)
		}
	})

	t.Run("timeout", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(2 * time.Second)
		}))
		defer srv.Close()
		rec, err := probeProviderRoute(routeProbeSpec{Provider: "nim", Model: "slow", BaseURL: srv.URL, Timeout: 100 * time.Millisecond})
		if err != nil {
			t.Fatalf("probe: %v", err)
		}
		if rec.Class != string(routeClassTimeout) {
			t.Fatalf("class = %s, want timeout", rec.Class)
		}
		if rec.CooldownUntilUnix <= rec.ProbedAtUnix {
			t.Fatal("timeout must set a cooldown")
		}
	})
}

// plantRouteHealthLedger writes ledger rows for the gate/status folds.
func plantRouteHealthLedger(t *testing.T, workspace string, recs ...routeHealthRecord) string {
	t.Helper()
	path := routeHealthLedgerPath(workspace)
	for _, rec := range recs {
		if err := appendRouteHealthRecord(path, rec); err != nil {
			t.Fatalf("append ledger: %v", err)
		}
	}
	return path
}

// TestRouteHealthGateSuppressesOnlyFailingRoute is the issue's load-bearing property: the
// 429-suppressed deepseek route is held for its cooldown while the healthy kimi SIBLING on
// the same provider/account passes, an unprobed route passes, and the suppressed route is
// re-admitted once its cooldown elapses.
func TestRouteHealthGateSuppressesOnlyFailingRoute(t *testing.T) {
	ws := t.TempDir()
	now := int64(2_000_000_000)
	plantRouteHealthLedger(t, ws,
		routeHealthRecord{
			Schema: routeHealthSchema, Route: "nim/seat1/deepseek-chat",
			Provider: "nim", Account: "seat1", Model: "deepseek-chat",
			Class: string(routeClassRateLimited), Status: 429,
			ProbedAtUnix: now - 60, CooldownUntilUnix: now + 840, ProviderHint: true,
			Recheck: "fak dispatch route-health probe --base-url https://nim.example/v1 --model deepseek-chat --provider nim --account seat1",
		},
		routeHealthRecord{
			Schema: routeHealthSchema, Route: "nim/seat1/kimi-k2",
			Provider: "nim", Account: "seat1", Model: "kimi-k2",
			Class: string(routeClassHealthy), Status: 200,
			ProbedAtUnix: now - 30,
			Recheck:      "fak dispatch route-health probe --base-url https://nim.example/v1 --model kimi-k2 --provider nim --account seat1",
		},
	)

	gate := func(args ...string) (int, string) {
		var stdout, stderr bytes.Buffer
		argv := append([]string{"gate", "--workspace", ws, "--now", fmt.Sprint(now)}, args...)
		code := runDispatchRouteHealth(&stdout, &stderr, argv)
		return code, stdout.String() + stderr.String()
	}

	if code, out := gate("--provider", "nim", "--account", "seat1", "--model", "deepseek-chat"); code != 3 {
		t.Fatalf("failing route: exit %d, want 3 (suppressed); out: %s", code, out)
	} else if !strings.Contains(out, "rate_limited") || !strings.Contains(out, "recheck") {
		t.Fatalf("suppressed verdict must name the class and the recheck command; out: %s", out)
	}
	if code, out := gate("--provider", "nim", "--account", "seat1", "--model", "kimi-k2"); code != 0 {
		t.Fatalf("healthy sibling route: exit %d, want 0; out: %s", code, out)
	}
	if code, out := gate("--provider", "nim", "--account", "seat1", "--model", "unprobed-model"); code != 0 {
		t.Fatalf("unprobed route: exit %d, want 0 (fail open); out: %s", code, out)
	}

	// After the cooldown elapses the suppressed route is re-admitted (recheck advised).
	var stdout, stderr bytes.Buffer
	code := runDispatchRouteHealth(&stdout, &stderr, []string{
		"gate", "--workspace", ws, "--now", fmt.Sprint(now + 900),
		"--provider", "nim", "--account", "seat1", "--model", "deepseek-chat",
	})
	if code != 0 {
		t.Fatalf("cooldown-elapsed route: exit %d, want 0; out: %s%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "cooldown elapsed") {
		t.Fatalf("re-admitted verdict should say the cooldown elapsed; out: %s", stdout.String())
	}
}

// TestRouteHealthStatusCard checks the operator surface: last probe age, failure class,
// cooldown/reset time, and the exact recheck command are all rendered, and the --json
// snapshot round-trips the same facts.
func TestRouteHealthStatusCard(t *testing.T) {
	ws := t.TempDir()
	now := int64(2_000_000_000)
	plantRouteHealthLedger(t, ws,
		routeHealthRecord{
			Schema: routeHealthSchema, Route: "nim/seat1/deepseek-chat",
			Provider: "nim", Account: "seat1", Model: "deepseek-chat",
			Class: string(routeClassTimeout),
			ProbedAtUnix: now - 120, CooldownUntilUnix: now + 480,
			Recheck: "fak dispatch route-health probe --base-url https://nim.example/v1 --model deepseek-chat --provider nim",
		},
	)

	var stdout, stderr bytes.Buffer
	if code := runDispatchRouteHealth(&stdout, &stderr, []string{"status", "--workspace", ws, "--now", fmt.Sprint(now)}); code != 0 {
		t.Fatalf("status exit %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"probe_age=120s",
		"class=timeout",
		"cooldown=480s left",
		"recheck: fak dispatch route-health probe",
		"1 suppressed",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("status card missing %q; out:\n%s", want, out)
		}
	}

	stdout.Reset()
	if code := runDispatchRouteHealth(&stdout, &stderr, []string{"status", "--workspace", ws, "--now", fmt.Sprint(now), "--json"}); code != 0 {
		t.Fatalf("status --json exit %d; stderr: %s", code, stderr.String())
	}
	var snap routeHealthStatusSnapshot
	if err := json.Unmarshal(stdout.Bytes(), &snap); err != nil {
		t.Fatalf("parse snapshot: %v", err)
	}
	if snap.Schema != routeHealthStatusSchema || snap.RouteCount != 1 || snap.SuppressedCount != 1 {
		t.Fatalf("snapshot = %+v", snap)
	}
	row := snap.Routes[0]
	if row.ProbeAgeSecs != 120 || !row.Suppressed || row.CooldownRemainingSecs != 480 || row.Recheck == "" {
		t.Fatalf("row = %+v", row)
	}
}

// TestRouteHealthProbeAppendsLedger runs the probe verb end-to-end: the record lands in
// the workspace ledger and the latest-per-route fold picks the newest row.
func TestRouteHealthProbeAppendsLedger(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		fmt.Fprint(w, `{"choices":[{"message":{"content":"OK"}}]}`)
	}))
	defer srv.Close()

	ws := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runDispatchRouteHealth(&stdout, &stderr, []string{
		"probe", "--base-url", srv.URL, "--model", "kimi-k2",
		"--provider", "nim", "--account", "seat1",
		"--workspace", ws, "--json",
	})
	if code != 0 {
		t.Fatalf("probe exit %d; stderr: %s", code, stderr.String())
	}
	var rec routeHealthRecord
	if err := json.Unmarshal(stdout.Bytes(), &rec); err != nil {
		t.Fatalf("parse record: %v", err)
	}
	if rec.Schema != routeHealthSchema || rec.Class != string(routeClassHealthy) {
		t.Fatalf("record = %+v", rec)
	}

	path := routeHealthLedgerPath(ws)
	if _, err := os.Stat(filepath.Clean(path)); err != nil {
		t.Fatalf("ledger not written: %v", err)
	}
	records, err := loadRouteHealthLatest(path)
	if err != nil {
		t.Fatalf("fold ledger: %v", err)
	}
	if len(records) != 1 || records[0].Route != "nim/seat1/kimi-k2" {
		t.Fatalf("fold = %+v", records)
	}

	// A newer failing row for the same route wins the fold; the healthy older row does not
	// mask it.
	newer := records[0]
	newer.Class = string(routeClassRateLimited)
	newer.ProbedAtUnix++
	newer.CooldownUntilUnix = newer.ProbedAtUnix + 60
	if err := appendRouteHealthRecord(path, newer); err != nil {
		t.Fatalf("append newer: %v", err)
	}
	records, err = loadRouteHealthLatest(path)
	if err != nil {
		t.Fatalf("re-fold ledger: %v", err)
	}
	if len(records) != 1 || records[0].Class != string(routeClassRateLimited) {
		t.Fatalf("latest fold = %+v", records)
	}
}
