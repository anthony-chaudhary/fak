package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// A paired client can opt into guard --router without copying the gateway key
// into an environment variable; explicit key settings still take precedence.
func TestGuardRouterUsesPairedNodeKey(t *testing.T) {
	nodeTestRedirectConfig(t)
	t.Setenv("FAK_AGENT_ROUTER_ORIGIN", "")
	t.Setenv(guardRouterKeyEnv, "")
	const pairedKey = "paired-router-key"
	var replyKey atomic.Value
	replyKey.Store(pairedKey)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		nonce, err := base64.StdEncoding.DecodeString(r.Header.Get(routerAuthChallengeHeader))
		if err != nil || len(nonce) != 32 {
			http.Error(w, "missing challenge", http.StatusBadRequest)
			return
		}
		mac := hmac.New(sha256.New, []byte(replyKey.Load().(string)))
		_, _ = mac.Write([]byte(routerAuthProofDomain))
		_, _ = mac.Write(nonce)
		w.Header().Set(routerAuthProofHeader, base64.StdEncoding.EncodeToString(mac.Sum(nil)))
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	if err := nodeWriteCfg(nodeCfg{URL: srv.URL, Key: pairedKey}); err != nil {
		t.Fatalf("nodeWriteCfg: %v", err)
	}

	getenv := func(name string) string { return "" }
	inputs := guardRouterInputs{provider: "anthropic"}
	up, err := resolveGuardRouterUpstream(inputs, getenv)
	if err != nil {
		t.Fatalf("resolve paired router: %v", err)
	}
	if up.origin != srv.URL || up.key != pairedKey {
		t.Fatalf("paired router = %+v; want origin %q and saved key", up, srv.URL)
	}
	if got := up.headers()["Authorization"]; got != "Bearer "+pairedKey {
		t.Fatalf("paired Authorization = %q, want saved key bearer", got)
	}
	if strings.Contains(guardRouterBanner(up), "$node.json") {
		t.Fatalf("paired key is not an environment variable: %s", guardRouterBanner(up))
	}
	replyKey.Store("wrong-key")
	up, err = resolveGuardRouterUpstream(inputs, getenv)
	if err == nil || !strings.Contains(err.Error(), "paired node key") || up.key != "" || up.headers() != nil {
		t.Fatalf("unproved saved key must fail before launch: upstream = %+v, err = %v", up, err)
	}
	if err := nodeWriteCfg(nodeCfg{URL: srv.URL}); err != nil {
		t.Fatalf("nodeWriteCfg keyless: %v", err)
	}
	up, err = resolveGuardRouterUpstream(inputs, getenv)
	if err != nil || up.key != "" || up.headers() != nil {
		t.Fatalf("genuinely keyless pairing should remain usable: upstream = %+v, err = %v", up, err)
	}
	if err := nodeWriteCfg(nodeCfg{URL: srv.URL, Key: pairedKey}); err != nil {
		t.Fatalf("nodeWriteCfg stale key: %v", err)
	}

	getenv = func(name string) string {
		switch name {
		case guardRouterKeyEnv:
			return "ambient-key"
		case "EXPLICIT_ROUTER_KEY":
			return "explicit-key"
		}
		return ""
	}
	up, err = resolveGuardRouterUpstream(inputs, getenv)
	if err != nil || up.key != "ambient-key" {
		t.Fatalf("ambient key should precede pairing: upstream = %+v, err = %v", up, err)
	}
	inputs.apiKeyEnv = "EXPLICIT_ROUTER_KEY"
	up, err = resolveGuardRouterUpstream(inputs, getenv)
	if err != nil || up.key != "explicit-key" {
		t.Fatalf("named key should precede ambient key and pairing: upstream = %+v, err = %v", up, err)
	}
}
