package main

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"time"
)

func TestDispatchTickRefusesDeadUpstreamBeforeSpawner(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, false)
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ok":                    false,
			"provider_reachability": map[string]any{"evaluated": true, "ok": false, "status": 502, "error": "upstream unreachable"},
		})
	}))
	defer gateway.Close()
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", gateway.URL+"/v1")

	oldBroker := launchSpawnBroker
	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant { return allowLaunchBrokerGrant(a, "unit-test-allow") }
	dispatchIssueWorkerSpawner = func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{}, nil
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		dispatchIssueWorkerSpawner = oldSpawner
	})

	out, _, code := runDispatchAt("tick", "--workspace", root, "--backend", "codex", "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--live", "--json")
	if code != 1 {
		t.Fatalf("dispatch exit=%d output=%s", code, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatal(err)
	}
	if spawned || got["action"] != "provider_unreachable" || got["verdict"] != "PROVIDER_REACHABILITY_REFUSED" {
		t.Fatalf("spawned=%v receipt=%#v", spawned, got)
	}
}

func TestDispatchProviderReachabilityRefusesDeadGatewayBeforeSpawn(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	base := "http://" + ln.Addr().String() + "/v1"
	_ = ln.Close()
	got := dispatchProviderReachabilityCheck([]string{"fak", "guard", "--base-url", base, "--", "codex"})
	if got["evaluated"] != true || got["ok"] != false || dispatchMapString(got, "id") != "provider_reachability" {
		t.Fatalf("dead gateway check = %#v", got)
	}
}

func TestDispatchProviderReachabilityRefusesDeadUpstreamBehindLiveGateway(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" || r.URL.Query().Get("deep") != "1" {
			t.Fatalf("probe = %s?%s", r.URL.Path, r.URL.RawQuery)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ok":                    false,
			"provider_reachability": map[string]any{"evaluated": true, "ok": false, "status": 502, "error": "upstream unreachable"},
		})
	}))
	defer gateway.Close()
	got := dispatchProviderReachabilityCheck([]string{"fak", "guard", "--base-url", gateway.URL + "/v1", "--", "codex"})
	if got["ok"] != false || dispatchMapString(got, "reason") != "upstream unreachable" || got["upstream_status"] != 502 {
		t.Fatalf("dead upstream check = %#v", got)
	}
}

func TestDispatchProviderReachabilityAdmitsHealthyRouteWithoutModelTurn(t *testing.T) {
	requests := 0
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Method != http.MethodGet || r.URL.Path != "/healthz" || r.URL.Query().Get("deep") != "1" {
			t.Fatalf("probe = %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ok":                    true,
			"provider_reachability": map[string]any{"evaluated": true, "ok": true, "status": 405},
		})
	}))
	defer gateway.Close()
	got := dispatchProviderReachabilityCheck([]string{"fak", "guard", "--base-url", gateway.URL + "/v1", "--", "codex"})
	if got["ok"] != true || requests != 1 {
		t.Fatalf("healthy route check = %#v requests=%d", got, requests)
	}
}

func TestDispatchProviderReachabilityTimeoutIsTyped(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer gateway.Close()
	old := dispatchProviderProbeClient
	dispatchProviderProbeClient = &http.Client{Timeout: 25 * time.Millisecond}
	t.Cleanup(func() { dispatchProviderProbeClient = old })
	got := dispatchProviderReachabilityCheck([]string{"fak", "guard", "--base-url", gateway.URL + "/v1", "--", "codex"})
	if got["evaluated"] != true || got["ok"] != false || dispatchMapString(got, "reason") == "" {
		t.Fatalf("timeout check = %#v", got)
	}
}

func TestDispatchDeepHealthURLDropsCredentials(t *testing.T) {
	got, err := dispatchDeepHealthURL("http://user:secret@example.test:8080/v1")
	if err != nil {
		t.Fatal(err)
	}
	if public := dispatchPublicEndpoint(got); public != "http://example.test:8080/healthz" {
		t.Fatalf("public endpoint = %q", public)
	}
}
