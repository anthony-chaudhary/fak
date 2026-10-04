package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms
func TestLaunchSessionHeaderPerHarness(t *testing.T) {
	id := newLaunchSessionID("claude")
	if !strings.HasPrefix(id, "claude-") || len(id) != len("claude-")+16 {
		t.Fatalf("launch session id = %q, want claude-<16 hex>", id)
	}
	if other := newLaunchSessionID("claude"); other == id {
		t.Fatalf("two launches minted the same id %q", id)
	}

	// Claude Code: ANTHROPIC_CUSTOM_HEADERS keeps user headers and a user-set session id.
	if got := claudeCustomHeadersWithSession("", "s1"); got != fakSessionHeader+": s1" {
		t.Fatalf("empty custom headers -> %q", got)
	}
	if got := claudeCustomHeadersWithSession("X-A: 1\n", "s1"); got != "X-A: 1\n"+fakSessionHeader+": s1" {
		t.Fatalf("appended custom headers -> %q", got)
	}
	if got := claudeCustomHeadersWithSession("x-fak-session-id: mine", "s1"); got != "x-fak-session-id: mine" {
		t.Fatalf("user session header overwritten -> %q", got)
	}

	// Codex: one -c override on the provider's http_headers table.
	args := codexSessionHeaderArg("fak", "s1")
	if len(args) != 2 || args[0] != "-c" || args[1] != `model_providers.fak.http_headers={ "X-Fak-Session-Id" = "s1" }` {
		t.Fatalf("codex args = %q", args)
	}
	if codexSessionHeaderArg("fak", "") != nil {
		t.Fatal("empty session id must add no codex override")
	}
	argv, _ := buildCodexRawArgv(codexLaunchOptions{baseURL: "http://127.0.0.1:8080/v1", model: "m", sessionID: "s1"})
	if !strings.Contains(strings.Join(argv, "\x00"), "\x00"+args[1]) {
		t.Fatalf("raw codex argv missing session header override: %q", argv)
	}

	// OpenCode: provider.fak.options.headers.
	cfg, err := openCodeConfigWithSessionHeader([]byte(`{"provider":{"fak":{"options":{"baseURL":"http://x/v1","headers":{"X-A":"1"}}}}}`), "fak", "s1")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Provider map[string]struct {
			Options struct {
				Headers map[string]string `json:"headers"`
			} `json:"options"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(cfg, &doc); err != nil {
		t.Fatal(err)
	}
	if h := doc.Provider["fak"].Options.Headers; h[fakSessionHeader] != "s1" || h["X-A"] != "1" {
		t.Fatalf("opencode headers = %v", h)
	}

	// Pi: the launch extension's provider literal carries the header.
	src, err := piLaunchProviderExtensionSource("http://127.0.0.1:8080/v1", "m", 80000, "", "s1")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(src, `headers: { "X-Fak-Session-Id": "s1" }`) {
		t.Fatalf("pi extension source lacks session header:\n%s", src)
	}
}
