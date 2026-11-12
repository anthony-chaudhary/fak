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
// declaring `provider.hive-ai.options.baseURL` at the loopback router, either
// literally or via the per-host router file token that each host resolves to
// its own loopback router origin. Verified
// live 2026-10-03: model `hive-ai/...` with hive-ai routed produced chain head
// `hive-ai=http://127.0.0.1:18101/v1`; model `fak-router/...` produced
// `hive-ai=https://api-cdn.thehive.ai/api/v3/`.
//
// This test fails on the parent commit (default on `fak-router`) and passes on
// the fix (default on the routed `hive-ai`).
//
// TOMBSTONE 2026-11-08 (operator decision): the hive-ai provider is retired
// everywhere, so the invariant above is INVERTED. The keyring Flash rotation
// that preserved the hive-ai handle is gone with the provider, and the
// default must NOT ride the hive-ai handle any more — it rides the declared
// `fak-router` provider directly. What survives of the old contract is the
// tombstone guard: any provider still DECLARED under a hive-ai spelling must
// never resolve to a direct thehive.ai upstream (the paused-org outage shape
// is now a dead route to refuse, not a live one to preserve). The default
// model was repointed to `fak-router/deepseek-ai/DeepSeek-V4.1-Flash` (main)
// and `fak-router/qwen/qwen3.8-27b` (small) — the ids exactly as declared in
// the tracked config's fak-router block.
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

	// TOMBSTONE (2026-11-08): the default provider must NOT be the retired
	// hive-ai provider. The keyring Flash rotation that once preserved that
	// handle as chain head is gone with the provider, so the default rides a
	// live provider (fak-router) directly.
	if isTombstonedHiveProviderName(provider) {
		t.Fatalf("default model provider %q is the RETIRED hive-ai provider (tombstoned 2026-11-08): the keyring Flash rotation that preserved it is gone, so the tracked default must ride a live declared provider (fak-router)", provider)
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
	if !isRouterBoundBaseURL(p.Options.BaseURL) {
		t.Fatalf("default model provider %q baseURL = %q, want the loopback router", provider, p.Options.BaseURL)
	}

	// The router provider must be DECLARED too, so the shared-router spelling
	// resolves for any other code path.
	router, ok := cfg.Provider["fak-router"]
	if !ok {
		t.Fatal("tracked opencode.json declares no provider.fak-router")
	}
	if !isRouterBoundBaseURL(router.Options.BaseURL) {
		t.Fatalf("provider.fak-router baseURL = %q, want the loopback router", router.Options.BaseURL)
	}

	// TOMBSTONE GUARD: if a provider is still declared under any hive-ai
	// spelling, it must never resolve to a DIRECT Hive upstream — the retired
	// provider may exist only as a routed stub, never as a live direct route
	// (the same outage shape the 2026-10-03 fix guarded, now refused outright).
	for name, p := range cfg.Provider {
		if !isTombstonedHiveProviderName(name) {
			continue
		}
		if isDirectHiveURL(p.Options.BaseURL) {
			t.Fatalf("provider %q (retired hive-ai, tombstoned 2026-11-08) baseURL = %q is a DIRECT Hive upstream; a tombstoned provider must never carry a live direct route", name, p.Options.BaseURL)
		}
	}
}

// routerFileToken is the per-host indirection the tracked config uses instead
// of pinning a literal port: each host resolves ~/.fak/router/base_url to its
// own loopback router origin.
const routerFileToken = "{file:~/.fak/router/base_url}"

// isRouterBoundBaseURL reports whether a tracked baseURL binds the provider
// route to the local fak router: a literal loopback origin or the per-host
// router file token. Anything else (including a direct-Hive upstream) fails.
func isRouterBoundBaseURL(raw string) bool {
	raw = strings.TrimSpace(raw)
	return strings.HasPrefix(raw, "http://127.0.0.1:") || raw == routerFileToken
}

// isTombstonedHiveProviderName reports whether a provider id names the RETIRED
// hosted Hive API provider under its documented spellings — the tombstone
// path: such a provider may not be the tracked default and may not carry a
// direct thehive.ai upstream.
func isTombstonedHiveProviderName(provider string) bool {
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
