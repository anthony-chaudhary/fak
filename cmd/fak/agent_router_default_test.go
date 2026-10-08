package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dropin"
)

// isolateAgentEndpointEnv unsets every ambient input resolveAgentEndpoint reads
// ahead of (or alongside) the stubbed probe, restoring them at cleanup. A host
// that exports a provider base URL (a WSL distro with OPENAI_BASE_URL pointed at
// its local router) otherwise resolves before the probe and a live endpoint
// answers the model-detect call, so the same commit passed on Windows and failed
// under WSL. FAK_GATEWAY_KEY would trigger a live credential challenge.
func isolateAgentEndpointEnv(t *testing.T) {
	t.Helper()
	nodeTestRedirectConfig(t) // ambient node.json must not change loopback tests
	for _, k := range []string{
		dropin.EnvVar("openai", ""),
		dropin.EnvVar("anthropic", ""),
		dropin.EnvVar("gemini", ""),
		"FAK_AGENT_ROUTER_ORIGIN",
		"FAK_GATEWAY_KEY",
	} {
		t.Setenv(k, "") // records the original value for cleanup
		_ = os.Unsetenv(k)
	}
}

// stubAgentRouterProbe installs a canned router probe for the duration of one
// test so the native-router resolution is hermetic: no test binds
// 127.0.0.1:8080, and a live dev-box mock serve on that port can never flip a
// test verdict.
func stubAgentRouterProbe(t *testing.T, model string, ok bool, hit *bool) {
	t.Helper()
	isolateAgentEndpointEnv(t)
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
	isolateAgentEndpointEnv(t)
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

// A paired gateway is the bare agent's default endpoint, and its saved key is
// released only after that endpoint answers a fresh keyed health challenge.
func TestAgentEndpointResolverUsesPairedNode(t *testing.T) {
	isolateAgentEndpointEnv(t)
	const key = "paired-gateway-key"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		if challenge := r.Header.Get(routerAuthChallengeHeader); challenge != "" {
			nonce, err := base64.StdEncoding.DecodeString(challenge)
			if err != nil || len(nonce) != 32 { //boundarylint:ignore CHANGE_DETECTOR_TEST router auth challenge nonce is a fixed 32-byte protocol width
				http.Error(w, "invalid challenge", http.StatusBadRequest)
				return
			}
			mac := hmac.New(sha256.New, []byte(key))
			_, _ = mac.Write([]byte(routerAuthProofDomain))
			_, _ = mac.Write(nonce)
			w.Header().Set(routerAuthProofHeader, base64.StdEncoding.EncodeToString(mac.Sum(nil)))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"model":"served-paired"}`)
	}))
	defer srv.Close()
	if err := nodeWriteCfg(nodeCfg{URL: srv.URL, Key: key}); err != nil {
		t.Fatalf("nodeWriteCfg: %v", err)
	}
	if got := routerOrigin(); got != srv.URL {
		t.Fatalf("routerOrigin = %q, want paired node %q", got, srv.URL)
	}

	fs, af := newAgentFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ep := agentEndpointResolver(af, false, false, false, io.Discard)
	if ep.baseURL != srv.URL+"/v1" || ep.routerKey != key || *af.model != "served-paired" {
		t.Fatalf("paired baseURL = %q, key verified = %t, model = %q; want %s/v1, true, served-paired", ep.baseURL, ep.routerKey == key, *af.model, srv.URL)
	}
	if got := agentRouterProbeTimeout("http://192.0.2.9:8080"); got != 2*time.Second {
		t.Fatalf("remote probe timeout = %s, want 2s", got)
	}
	if got := agentRouterProbeTimeout(agentRouterOrigin); got != 150*time.Millisecond {
		t.Fatalf("loopback probe timeout = %s, want 150ms", got)
	}

	// A bad saved key must not fall through to a workspace key, even if that
	// workspace happens to name the same endpoint.
	writeRouterConfig(t, srv.URL+"/v1", "unproved-workspace-key")
	if err := nodeWriteCfg(nodeCfg{URL: srv.URL, Key: "wrong-paired-key"}); err != nil {
		t.Fatalf("nodeWriteCfg wrong key: %v", err)
	}
	if got := resolveRouterAPIKey("", false, true, srv.URL+"/v1"); got != "" {
		t.Fatal("unproved paired key fell through to a workspace credential")
	}
	if err := nodeWriteCfg(nodeCfg{URL: srv.URL}); err != nil {
		t.Fatalf("nodeWriteCfg missing key: %v", err)
	}
	if got := resolveRouterAPIKey("", false, true, srv.URL+"/v1"); got != "" {
		t.Fatal("missing paired key fell through to a workspace credential")
	}
	if _, paired := pairedNodeRouterAPIKey(srv.URL + "-other"); paired {
		t.Fatal("a different origin matched the saved paired node")
	}
}

func TestAgentPairedNodePrecedenceAndDown(t *testing.T) {
	isolateAgentEndpointEnv(t)
	const paired = "http://127.0.0.1:0"
	if err := nodeWriteCfg(nodeCfg{URL: paired, Key: "paired-key"}); err != nil {
		t.Fatalf("nodeWriteCfg: %v", err)
	}
	if got := routerOrigin(); got != paired {
		t.Fatalf("paired router origin = %q, want %q", got, paired)
	}
	if origin, _, ok := probeLocalRouterOrigin(); ok {
		t.Fatalf("down paired node silently fell back to %q", origin)
	}
	var guidance strings.Builder
	agentNoEndpointGuidance(&guidance, true)
	if !strings.Contains(guidance.String(), paired) || strings.Contains(guidance.String(), agentRouterOrigin) {
		t.Fatalf("down-node guidance named the wrong router: %s", guidance.String())
	}
	t.Setenv("FAK_AGENT_ROUTER_ORIGIN", "http://127.0.0.1:2")
	if got := routerOrigin(); got != "http://127.0.0.1:2" {
		t.Fatalf("explicit router origin = %q, want env override", got)
	}
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:0/v1")
	fs, af := newAgentFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if ep := agentEndpointResolver(af, false, false, false, io.Discard); ep.baseURL != "http://127.0.0.1:0/v1" || ep.routerProbed {
		t.Fatalf("provider base URL did not override paired node and router env: baseURL=%q routerProbed=%t", ep.baseURL, ep.routerProbed)
	}
}

// A saved IPv4 loopback node may answer only on IPv6. The actual localhost
// fallback must receive the paired key after proving possession, while an
// unrelated origin must never inherit that pairing.
func TestAgentPairedNodeKeyOnLoopbackFamilyFallback(t *testing.T) {
	isolateAgentEndpointEnv(t)
	ln, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Skipf("IPv6 loopback is unavailable: %v", err)
	}
	const key = "family-fallback-key"
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		if challenge := r.Header.Get(routerAuthChallengeHeader); challenge != "" {
			nonce, err := base64.StdEncoding.DecodeString(challenge)
			if err != nil || len(nonce) != 32 { //boundarylint:ignore CHANGE_DETECTOR_TEST router auth challenge nonce is a fixed 32-byte protocol width
				http.Error(w, "invalid challenge", http.StatusBadRequest)
				return
			}
			mac := hmac.New(sha256.New, []byte(key))
			_, _ = mac.Write([]byte(routerAuthProofDomain))
			_, _ = mac.Write(nonce)
			w.Header().Set(routerAuthProofHeader, base64.StdEncoding.EncodeToString(mac.Sum(nil)))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"model":"ipv6-model"}`)
	}))
	srv.Listener = ln
	srv.Start()
	defer srv.Close()
	_, port, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		t.Fatalf("listener address: %v", err)
	}
	pairedOrigin := "http://127.0.0.1:" + port
	fallbackOrigin := "http://localhost:" + port
	if _, ok := probeGatewayOnce(agentRouterProbeClient(fallbackOrigin), fallbackOrigin); !ok {
		t.Skip("localhost does not resolve to the available IPv6 loopback listener")
	}
	if err := nodeWriteCfg(nodeCfg{URL: pairedOrigin, Key: key}); err != nil {
		t.Fatalf("nodeWriteCfg: %v", err)
	}
	if origin, model, ok := probeLocalRouterOrigin(); !ok || origin != fallbackOrigin || model != "ipv6-model" {
		t.Fatalf("fallback probe = (%q, %q, %t), want (%q, ipv6-model, true)", origin, model, ok, fallbackOrigin)
	}
	if got, paired := pairedNodeRouterAPIKey(fallbackOrigin); !paired || got != key {
		t.Fatalf("fallback credential = (paired=%t, verified=%t), want (true, true)", paired, got == key)
	}
	if _, paired := pairedNodeRouterAPIKey("http://example.invalid:" + port); paired {
		t.Fatal("an unrelated host inherited the loopback pairing")
	}

	// If a different server later answers the selected IPv4 origin, that
	// server must not borrow the still-running IPv6 gateway's valid proof.
	wrong, err := net.Listen("tcp4", "127.0.0.1:"+port)
	if err != nil {
		t.Logf("cannot bind the paired IPv4 address beside IPv6: %v", err)
		return
	}
	wrongServer := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"model":"wrong-model"}`)
	}))
	wrongServer.Listener = wrong
	wrongServer.Start()
	defer wrongServer.Close()
	if origin, model, ok := probeLocalRouterOrigin(); !ok || origin != pairedOrigin || model != "wrong-model" {
		t.Fatalf("selected origin = (%q, %q, %t), want unproved IPv4 server", origin, model, ok)
	}
	if got, paired := pairedNodeRouterAPIKey(pairedOrigin); !paired || got != "" {
		t.Fatalf("unproved selected origin borrowed another loopback gateway's key: paired=%t released=%t", paired, got != "")
	}
}

// A redirecting selected origin must not borrow another gateway's health proof
// and then receive the paired bearer on its own inference path.
func TestAgentPairedNodeRejectsRedirectedProof(t *testing.T) {
	isolateAgentEndpointEnv(t)
	const key = "real-gateway-key"
	real := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		if challenge := r.Header.Get(routerAuthChallengeHeader); challenge != "" {
			nonce, err := base64.StdEncoding.DecodeString(challenge)
			if err != nil || len(nonce) != 32 { //boundarylint:ignore CHANGE_DETECTOR_TEST router auth challenge nonce is a fixed 32-byte protocol width
				http.Error(w, "invalid challenge", http.StatusBadRequest)
				return
			}
			mac := hmac.New(sha256.New, []byte(key))
			_, _ = mac.Write([]byte(routerAuthProofDomain))
			_, _ = mac.Write(nonce)
			w.Header().Set(routerAuthProofHeader, base64.StdEncoding.EncodeToString(mac.Sum(nil)))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"model":"real-model"}`)
	}))
	defer real.Close()
	redirecting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, real.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer redirecting.Close()
	if err := nodeWriteCfg(nodeCfg{URL: redirecting.URL, Key: key}); err != nil {
		t.Fatalf("nodeWriteCfg: %v", err)
	}
	if got, paired := pairedNodeRouterAPIKey(redirecting.URL); !paired || got != "" {
		t.Fatalf("redirected origin released paired key: paired=%t keyReleased=%t", paired, got != "")
	}
	if origin, _, ok := probeLocalRouterOrigin(); ok {
		t.Fatalf("redirected origin was accepted as live: %q", origin)
	}
}
