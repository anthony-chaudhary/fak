package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// stubAgentRouterProbe installs a canned router probe for the duration of one
// test so the native-router resolution is hermetic: no test binds
// 127.0.0.1:8080, and a live dev-box mock serve on that port can never flip a
// test verdict.
func stubAgentRouterProbe(t *testing.T, model string, ok bool, hit *bool) {
	t.Helper()
	prev := agentRouterProbe
	agentRouterProbe = func() (string, string, bool) {
		if hit != nil {
			*hit = true
		}
		return agentRouterOrigin, model, ok
	}
	t.Cleanup(func() { agentRouterProbe = prev })
}

// stubAgentRouterProbeOrigin installs a canned router probe that reports a
// specific origin (e.g. the localhost fallback) so a test can pin that the
// resolver adopts the origin that actually answered.
func stubAgentRouterProbeOrigin(t *testing.T, origin, model string, ok bool) {
	t.Helper()
	prev := agentRouterProbe
	agentRouterProbe = func() (string, string, bool) {
		return origin, model, ok
	}
	t.Cleanup(func() { agentRouterProbe = prev })
}

func TestAgentEndpointResolverRouterDefaultLive(t *testing.T) {
	stubAgentRouterProbe(t, "Qwen3.8-27B-UD-Q2_K_XL", true, nil)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if ep.baseURL != agentRouterOrigin+"/v1" {
		t.Fatalf("baseURL = %q, want %q", ep.baseURL, agentRouterOrigin+"/v1")
	}
	if !ep.routerProbed {
		t.Fatal("routerProbed = false, want true on the native-router path")
	}
	if *af.model != "Qwen3.8-27B-UD-Q2_K_XL" {
		t.Fatalf("model = %q, want the probed served model id", *af.model)
	}
}

// TestAgentEndpointResolverRouterDownFailsLoud pins the COMPOSED behavior: on
// trunk the native agent fails loud when no endpoint resolves, so a router-down
// bare-resolve must yield an empty base URL (no offline-mock fallback) while
// still recording that the probe ran.
func TestAgentEndpointResolverRouterDownFailsLoud(t *testing.T) {
	stubAgentRouterProbe(t, "", false, nil)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if ep.baseURL != "" {
		t.Fatalf("baseURL = %q, want empty (the fail-loud case when the router is not live)", ep.baseURL)
	}
	if !ep.routerProbed {
		t.Fatal("routerProbed = false, want true (the probe ran and failed)")
	}
	if *af.model != "gemini-3.8-flash" {
		t.Fatalf("model = %q, want the unchanged default", *af.model)
	}
}

func TestAgentEndpointResolverMockIdNotAdopted(t *testing.T) {
	stubAgentRouterProbe(t, "mock", true, nil)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if ep.baseURL != agentRouterOrigin+"/v1" {
		t.Fatalf("baseURL = %q, want %q", ep.baseURL, agentRouterOrigin+"/v1")
	}
	if *af.model != "gemini-3.8-flash" {
		t.Fatalf("model = %q, want the default (a mock id must never be adopted)", *af.model)
	}
}

func TestAgentEndpointResolverExplicitBaseURLWinsNoProbe(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "model": "served-explicit"})
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()
	hit := false
	stubAgentRouterProbe(t, "should-not-matter", true, &hit)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{"--base-url", ts.URL + "/v1"}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, true, false, false, io.Discard)
	if hit {
		t.Fatal("the router probe ran despite an explicit --base-url")
	}
	if ep.baseURL != ts.URL+"/v1" {
		t.Fatalf("baseURL = %q, want %q", ep.baseURL, ts.URL+"/v1")
	}
	if *af.model != "served-explicit" {
		t.Fatalf("model = %q, want the served id auto-detected from the explicit endpoint", *af.model)
	}
}

func TestAgentEndpointResolverOfflineSkipsProbe(t *testing.T) {
	hit := false
	stubAgentRouterProbe(t, "anything", true, &hit)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{"--offline"}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if hit {
		t.Fatal("the router probe ran despite --offline")
	}
	if ep.baseURL != "" {
		t.Fatalf("baseURL = %q, want empty", ep.baseURL)
	}
}

func TestAgentEndpointResolverEnvVarWins(t *testing.T) {
	hit := false
	stubAgentRouterProbe(t, "anything", true, &hit)
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:65530/v1")
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if hit {
		t.Fatal("the router probe ran despite the provider env var")
	}
	if ep.baseURL != "http://127.0.0.1:65530/v1" {
		t.Fatalf("baseURL = %q, want http://127.0.0.1:65530/v1", ep.baseURL)
	}
	if *af.model != "gemini-3.8-flash" {
		t.Fatalf("model = %q, want the unchanged default (a closed port detects nothing)", *af.model)
	}
}

func TestAgentEndpointResolverModelExplicitNotOverridden(t *testing.T) {
	stubAgentRouterProbe(t, "Qwen3.8-27B-UD-Q2_K_XL", true, nil)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{"--model", "my-model"}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, true, io.Discard)
	if ep.baseURL != agentRouterOrigin+"/v1" {
		t.Fatalf("baseURL = %q, want %q", ep.baseURL, agentRouterOrigin+"/v1")
	}
	if *af.model != "my-model" {
		t.Fatalf("model = %q, want the explicit flag preserved", *af.model)
	}
}

// TestAgentNativeRouterFallbackBareRun pins the COMPOSED bare-run behavior: a
// bare `fak agent` on trunk fails loud when no endpoint resolves, and when the
// native-router probe ran and the router was not live the guidance must name
// the probed router rather than claim no base URL was given. The fail-loud path
// calls os.Exit(2), which cannot be captured in-process and whose subprocess
// form hangs under this box's Defender (the pre-existing TestAgentNoEndpoint-
// FailsLoud hangs identically), so this exercises the two pure pieces the exit
// path composes: the resolver's router-down verdict, then the guidance text.
func TestAgentNativeRouterFallbackBareRun(t *testing.T) {
	stubAgentRouterProbe(t, "", false, nil)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if ep.baseURL != "" {
		t.Fatalf("baseURL = %q, want empty (the fail-loud case when the router is not live)", ep.baseURL)
	}
	if !ep.routerProbed {
		t.Fatal("routerProbed = false, want true (the probe ran and failed)")
	}

	var stderr strings.Builder
	agentNoEndpointGuidance(&stderr, ep.routerProbed)
	out := stderr.String()
	if !strings.Contains(out, "not responding") || !strings.Contains(out, agentRouterOrigin) {
		t.Fatalf("fail-loud guidance did not name the probed router (%s):\n%s", agentRouterOrigin, out)
	}
	if !strings.Contains(out, "fak agentdemo") || !strings.Contains(out, "--offline") {
		t.Fatalf("fail-loud guidance did not name the explicit demo opt-ins:\n%s", out)
	}
}

// TestAgentEndpointResolverUsesProbedOrigin pins the loopback family fix: when
// the probe answers via a fallback origin (e.g. localhost for a host where the
// 127.0.0.1 literal is refused), the resolver must build the effective baseURL
// from THAT origin, not the hardcoded literal, or the later inference calls
// would fail on the same family the probe just fell back from.
func TestAgentEndpointResolverUsesProbedOrigin(t *testing.T) {
	const fallbackOrigin = "http://localhost:8080"
	stubAgentRouterProbeOrigin(t, fallbackOrigin, "qwen38:27b-q4", true)
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if ep.baseURL != fallbackOrigin+"/v1" {
		t.Fatalf("baseURL = %q, want %q (the origin that actually answered)", ep.baseURL, fallbackOrigin+"/v1")
	}
	if !ep.routerProbed {
		t.Fatal("routerProbed = false, want true")
	}
	if *af.model != "qwen38:27b-q4" {
		t.Fatalf("model = %q, want the probed served model id", *af.model)
	}
}
