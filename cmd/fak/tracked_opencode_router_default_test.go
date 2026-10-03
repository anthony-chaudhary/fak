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

// TestTrackedOpenCodeDefaultNeverDialsDirectHive is the contract witness for
// the worker default the Ops queue spawns against. A DIRECT Hive upstream has no
// failover ladder, so when the Hive organization is paused (HTTP 405) every
// spawned worker dies "Invalid or expired token"; opencode's terminal completion
// rate then falls below the 0.80 harness-health floor and the fleet refuses
// every queued opencode spawn (`harness is degraded`) — a total dispatch outage
// from one config route (witnessed 2026-10-02: completed/total 0.11 over 502
// terminal runs, and the P0 oss-port-p0-* queue never admitted).
//
// The fix keeps the model HANDLE while moving the ROUTE: the default model is
// resolved through the local fak router, which fronts the same upstream and owns
// the paused-org verdict + failover ladder. This test fails on the pre-fix
// config (model "hive-ai/..." with only a direct-hive provider) and passes once
// the default is routed.
func TestTrackedOpenCodeDefaultNeverDialsDirectHive(t *testing.T) {
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

	// The DEFAULT model's provider must resolve to a NON-direct-Hive endpoint:
	// either the local router provider, or a provider whose declared baseURL is
	// the loopback router (the keyring Flash rotation preserves the resolved head
	// when that provider is a ready Flash provider, so routing hive-ai itself is a
	// valid shape that keeps the model handle unchanged).
	p, ok := cfg.Provider[provider]
	if !ok {
		t.Fatalf("default model provider %q is not declared in the tracked config", provider)
	}
	base := p.Options.BaseURL
	if isDirectHiveURL(base) {
		t.Fatalf("default model provider %q resolves to a DIRECT Hive upstream (%s); a paused org would kill every spawned worker", provider, base)
	}
	if !strings.HasPrefix(base, "http://127.0.0.1:") {
		t.Fatalf("default model provider %q baseURL = %q, want the loopback router", provider, base)
	}

	// The router provider must be DECLARED and point at the loopback router, or
	// opencode cannot instantiate a routed default model.
	router, ok := cfg.Provider["fak-router"]
	if !ok {
		t.Fatal("tracked opencode.json declares no provider.fak-router for the default model")
	}
	if !strings.HasPrefix(router.Options.BaseURL, "http://127.0.0.1:") {
		t.Fatalf("provider.fak-router baseURL = %q, want the loopback router", router.Options.BaseURL)
	}

	// Every DECLARED hive-ai provider must traverse the router: a direct-hive
	// base URL here would re-open the outage for any code path that resolves it.
	if hive, ok := cfg.Provider["hive-ai"]; ok && isDirectHiveURL(hive.Options.BaseURL) {
		t.Fatalf("provider.hive-ai baseURL = %q is a DIRECT Hive upstream; it must route through the local router", hive.Options.BaseURL)
	}
}

// isDirectHiveProviderName reports whether a provider id names the hosted Hive
// API under its documented spellings. Retained for the closed vocabulary even
// where the endpoint check is the load-bearing assertion.
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
