package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// launch_session.go — a launch-scoped session id stamped onto every model request a
// third-party harness (Claude Code, Codex, OpenCode, Pi) sends to fak serve, through each
// harness's own supported custom-header mechanism. The gateway reads X-Fak-Session-Id to
// attribute a subagent's prefix-KV reuse to the coordinator launch that paid for it
// (internal/gateway/prefix_reuse_attribution.go). The id is random, carries no host or
// user data, and only exists for the life of one launch.

const fakSessionHeader = "X-Fak-Session-Id"

// newLaunchSessionID mints one launch-scoped id, e.g. "claude-5f0c1a2b3d4e5f60".
func newLaunchSessionID(harness string) string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return ""
	}
	return harness + "-" + hex.EncodeToString(b[:])
}

// claudeCustomHeadersWithSession returns the ANTHROPIC_CUSTOM_HEADERS value (newline-
// separated "Name: Value" lines) with the session header appended. A caller-supplied
// X-Fak-Session-Id is kept, never overwritten.
func claudeCustomHeadersWithSession(existing, sessionID string) string {
	if sessionID == "" {
		return existing
	}
	for _, line := range strings.Split(existing, "\n") {
		name, _, ok := strings.Cut(line, ":")
		if ok && strings.EqualFold(strings.TrimSpace(name), fakSessionHeader) {
			return existing
		}
	}
	line := fakSessionHeader + ": " + sessionID
	if strings.TrimSpace(existing) == "" {
		return line
	}
	return strings.TrimRight(existing, "\n") + "\n" + line
}

// codexSessionHeaderArg is the Codex -c override that adds the session header to the fak
// model provider's static http_headers table.
func codexSessionHeaderArg(providerID, sessionID string) []string {
	if sessionID == "" {
		return nil
	}
	return []string{"-c", "model_providers." + providerID + ".http_headers={ " + fmt.Sprintf("%q", fakSessionHeader) + " = " + fmt.Sprintf("%q", sessionID) + " }"}
}

// withProviderSessionHeader sets headers[X-Fak-Session-Id] on a provider object (Pi
// registerProvider config or an OpenCode provider options block), preserving any other
// headers and a caller-supplied session id.
func withProviderSessionHeader(target map[string]any, sessionID string) {
	if target == nil || sessionID == "" {
		return
	}
	headers, _ := target["headers"].(map[string]any)
	if headers == nil {
		headers = map[string]any{}
	}
	for k := range headers {
		if strings.EqualFold(k, fakSessionHeader) {
			return
		}
	}
	headers[fakSessionHeader] = sessionID
	target["headers"] = headers
}

// openCodeConfigWithSessionHeader adds the session header to provider.<id>.options.headers
// of an OpenCode config document (@ai-sdk/openai-compatible forwards options.headers).
func openCodeConfigWithSessionHeader(config []byte, providerID, sessionID string) ([]byte, error) {
	if sessionID == "" {
		return config, nil
	}
	var cfg map[string]any
	if err := json.Unmarshal(config, &cfg); err != nil {
		return nil, fmt.Errorf("decode opencode config: %w", err)
	}
	providers, _ := cfg["provider"].(map[string]any)
	provider, _ := providers[providerID].(map[string]any)
	if provider == nil {
		return config, nil
	}
	options, _ := provider["options"].(map[string]any)
	if options == nil {
		options = map[string]any{}
		provider["options"] = options
	}
	withProviderSessionHeader(options, sessionID)
	return json.MarshalIndent(cfg, "", "  ")
}
