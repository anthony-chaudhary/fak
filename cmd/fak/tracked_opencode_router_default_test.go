package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// trackedOpenCodeConfig is the subset of the tracked repo-root opencode.json
// this contract test reads.
type trackedOpenCodeConfig struct {
	Model      string `json:"model"`
	SmallModel string `json:"small_model"`
	Provider   map[string]struct {
		Options struct {
			BaseURL string `json:"baseURL"`
		} `json:"options"`
	} `json:"provider"`
}

// TestTrackedOpenCodeDefaultIsKeyringPreservable is the contract witness for the
// worker default the Ops queue spawns against.
//
// THE LOAD-BEARING INVARIANT is not merely "the default points at the router" —
// it is "the default's PROVIDER is the one the keyring Flash rotation can
// PRESERVE as the chain head". resbroker's Flash rotation keeps the resolved head
// at index 0 only when that head's provider is a READY Flash provider; otherwise
// it PREPENDS one candidate per usable keyring record, and each such candidate's
// BaseURL is the CREDENTIAL RECORD's own direct-Hive upstream. A default on a
// provider absent from the rotation set (e.g. `fak-router`) therefore yields a
// chain HEAD of `hive-ai=https://api-cdn.thehive.ai/api/v3/` — a direct call with
// no failover ladder — even though `fak-router` itself is declared. A paused
// Hive organization then answers HTTP 405, every spawned worker dies "Invalid or
// expired token", opencode's terminal completion rate falls below the 0.80
// harness-health floor, and the fleet refuses every queued opencode spawn
// (`harness is degraded`) — a total dispatch outage from one config route.
//
// The verified-working shape is: keep the `hive-ai/<model>` HANDLE (so it is a
// ready Flash provider the rotation preserves) and move only its ROUTE by
// declaring `provider.hive-ai.options.baseURL` at the loopback router. Verified
// live 2026-10-03: model `hive-ai/...` with hive-ai routed produced chain head
// `hive-ai=http://127.0.0.1:18101/v1`; model `fak-router/...` produced
// `hive-ai=https://api-cdn.thehive.ai/api/v3/`.
//
// This test fails on the parent commit (default on `fak-router`) and passes on
// the fix (default on the routed `hive-ai`).
func TestTrackedOpenCodeDefaultIsKeyringPreservable(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "opencode.json"))
	if err != nil {
		t.Fatalf("read tracked opencode.json: %v", err)
	}
	var cfg trackedOpenCodeConfig
	if err := json.Unmarshal(stripUTF8BOMBytes(raw), &cfg); err != nil {
		t.Fatalf("parse tracked opencode.json: %v", err)
	}

	if strings.TrimSpace(cfg.Model) == "" {
		t.Fatal("tracked opencode.json declares no default model")
	}
	provider := cfg.Model
	if i := strings.Index(cfg.Model, "/"); i > 0 {
		provider = cfg.Model[:i]
	}
	if provider == "" {
		t.Fatalf("default model %q carries no provider prefix", cfg.Model)
	}

	// The default provider must be a keyring Flash provider the rotation
	// preserves. hive-ai is that provider; a router-only default is NOT
	// preserved and re-opens the direct-Hive head.
	if !isDirectHiveProviderName(provider) {
		t.Fatalf("default model provider %q is not the keyring Flash provider hive-ai: the Flash rotation does not preserve it, so it prepends the keyring's own direct-Hive candidates and the launch dials a paused org (verified 2026-10-03)", provider)
	}

	// ...and that provider's declared endpoint must be the loopback router, so
	// the route (not just the handle) is router-bound.
	p, ok := cfg.Provider[provider]
	if !ok {
		t.Fatalf("default model provider %q is not declared in the tracked config", provider)
	}
	if isDirectHiveURL(p.Options.BaseURL) {
		t.Fatalf("default model provider %q resolves to a DIRECT Hive upstream (%s); a paused org would kill every spawned worker", provider, p.Options.BaseURL)
	}
	if !strings.HasPrefix(p.Options.BaseURL, "http://127.0.0.1:") {
		t.Fatalf("default model provider %q baseURL = %q, want the loopback router", provider, p.Options.BaseURL)
	}

	// The router provider must be DECLARED too, so the shared-router spelling
	// resolves for any other code path.
	router, ok := cfg.Provider["fak-router"]
	if !ok {
		t.Fatal("tracked opencode.json declares no provider.fak-router")
	}
	if !strings.HasPrefix(router.Options.BaseURL, "http://127.0.0.1:") {
		t.Fatalf("provider.fak-router baseURL = %q, want the loopback router", router.Options.BaseURL)
	}

	// Every DECLARED hive-ai provider must traverse the router: a direct-hive
	// base URL here would re-open the outage wherever it is resolved.
	if hive, ok := cfg.Provider["hive-ai"]; ok && isDirectHiveURL(hive.Options.BaseURL) {
		t.Fatalf("provider.hive-ai baseURL = %q is a DIRECT Hive upstream; it must route through the local router", hive.Options.BaseURL)
	}
}

// isDirectHiveProviderName reports whether a provider id names the hosted Hive
// API under its documented spellings.
func isDirectHiveProviderName(provider string) bool {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "hive-ai", "hive", "hive_ai":
		return true
	default:
		return false
	}
}

// isDirectHiveURL reports whether raw names a Hive AI upstream host. It parses
// the host so a path or query cannot forge a match.
func isDirectHiveURL(raw string) bool {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return false
	}
	host := raw
	if i := strings.Index(host, "://"); i >= 0 {
		host = host[i+3:]
	}
	if i := strings.IndexAny(host, "/?#"); i >= 0 {
		host = host[:i]
	}
	if i := strings.Index(host, "@"); i >= 0 {
		host = host[i+1:]
	}
	if i := strings.Index(host, ":"); i >= 0 {
		host = host[:i]
	}
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	switch host {
	case "api.thehive.ai", "api-cdn.thehive.ai", "thehive.ai":
		return true
	default:
		return false
	}
}

func stripUTF8BOMBytes(raw []byte) []byte {
	return []byte(strings.TrimPrefix(string(raw), "\ufeff"))
}
