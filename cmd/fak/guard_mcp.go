package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// guard_mcp.go wires fak's own MCP self-query surface (fak_index_*, fak_memory_*,
// fak_tools_search — internal/gateway/mcp.go:597-668) into the wrapped Claude Code
// child BY DEFAULT (#1499, the C1 child of the #1494 "fak can answer 'what can I do?'"
// epic). Without this, the gateway already speaks MCP over JSON-RPC at POST /mcp, but
// a default `fak guard -- claude` session has no way to discover it: the child never
// learns a "fak" server exists unless the operator hand-writes a project .mcp.json.
//
// The install writes a session-scoped MCP client config naming exactly one remote HTTP
// server ("fak" -> gwURL+"/mcp") and passes it via Claude Code's --mcp-config flag. That
// flag ADDS servers to whatever project/user config Claude Code already loads (this
// never sets --strict-mcp-config), so an operator's own .mcp.json — this repo's own
// carries a "dos" entry — is untouched. Every fak_* tool call the child makes is still
// re-adjudicated by the same guard floor as any other tool call (see guard.go's
// re-adjudication note near cmdGuard); this widens discovery only, never the danger
// floor. Mirrors the install shape of guard_precompact.go / guard_codex.go.

// guardMCPServerName is the key this repo's own .mcp.json and examples/mcp/.mcp.json
// already use for fak's MCP surface, kept consistent here.
const guardMCPServerName = "fak"

// guardMCPInstall records what the MCP config injection did, for the banner and tests.
type guardMCPInstall struct {
	Applied    bool
	ConfigPath string
	URL        string
	Reason     string
}

// guardMCPClientConfig is the subset of the Claude Code / Cursor .mcp.json schema this
// install needs: one named server, described by its remote HTTP endpoint.
type guardMCPClientConfig struct {
	MCPServers map[string]guardMCPClientServer `json:"mcpServers"`
}

type guardMCPClientServer struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// installGuardMCPRegistration wires fak's MCP surface into a Claude Code child by
// writing a session-scoped --mcp-config file that points "fak" at the gateway's /mcp
// endpoint. A non-Claude agent, or enabled=false (the operator's --mcp-register=false
// opt-out for an operator who supplies their own MCP config), returns command
// unchanged with no install performed. An empty command is a no-op.
func installGuardMCPRegistration(command []string, enabled bool, gwURL string) ([]string, guardMCPInstall, error) {
	if !enabled {
		return command, guardMCPInstall{Reason: "disabled"}, nil
	}
	if len(command) == 0 || !guardPreCompactIsClaudeCommand(command) {
		return command, guardMCPInstall{Reason: "non-claude-child"}, nil
	}
	dir, err := os.MkdirTemp("", "fak-guard-mcp-*")
	if err != nil {
		return command, guardMCPInstall{}, err
	}
	return installGuardMCPRegistrationAt(command, gwURL, dir)
}

// installGuardMCPRegistrationAt is installGuardMCPRegistration with the session
// directory injected, so tests can assert on the written file without touching the OS
// temp dir. It performs the same Claude-only gate as installGuardMCPRegistration.
func installGuardMCPRegistrationAt(command []string, gwURL, dir string) ([]string, guardMCPInstall, error) {
	if len(command) == 0 || !guardPreCompactIsClaudeCommand(command) {
		return command, guardMCPInstall{Reason: "non-claude-child"}, nil
	}
	if strings.TrimSpace(dir) == "" {
		return command, guardMCPInstall{}, fmt.Errorf("empty MCP config directory")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return command, guardMCPInstall{}, err
	}
	configPath := filepath.Join(dir, "fak-mcp-config.json")
	mcpURL := guardMCPURLFromGatewayBase(gwURL)
	if err := writeGuardMCPConfig(configPath, mcpURL); err != nil {
		return command, guardMCPInstall{}, err
	}
	install := guardMCPInstall{Applied: true, ConfigPath: configPath, URL: mcpURL}
	return appendClaudeMCPConfigArg(command, configPath), install, nil
}

// writeGuardMCPConfig writes the one-server MCP client config naming "fak" as a remote
// HTTP MCP server at mcpURL.
func writeGuardMCPConfig(path, mcpURL string) error {
	cfg := guardMCPClientConfig{MCPServers: map[string]guardMCPClientServer{
		guardMCPServerName: {Type: "http", URL: mcpURL},
	}}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

// guardMCPURLFromGatewayBase is the gateway's MCP endpoint: the bare origin (gwURL
// carries no /v1 suffix — see cmdGuard's `gwURL := "http://" + ln.Addr().String()`)
// plus /mcp, the route internal/gateway/http.go registers directly on the mux.
func guardMCPURLFromGatewayBase(gwURL string) string {
	return strings.TrimRight(strings.TrimSpace(gwURL), "/") + "/mcp"
}

// appendClaudeMCPConfigArg inserts --mcp-config <path> immediately after the Claude
// executable, mirroring appendClaudeSettingsArg (guard_precompact.go) — before any
// subcommand or user args, since Claude Code's global flags precede them. --mcp-config
// ADDS the named server(s) to whatever project/user config Claude Code already loads;
// this never sets --strict-mcp-config, so it can only add servers, never replace them.
func appendClaudeMCPConfigArg(command []string, configPath string) []string {
	if len(command) == 0 {
		return command
	}
	out := make([]string, 0, len(command)+2)
	out = append(out, command[0], "--mcp-config", configPath)
	return append(out, command[1:]...)
}

// printGuardMCPNote explains the MCP registration on the banner, mirroring
// printGuardCodexNote (guard_codex.go).
func printGuardMCPNote(w io.Writer, in guardMCPInstall) {
	if !in.Applied {
		return
	}
	fmt.Fprintf(w, "fak guard: Claude MCP self-query surface registered — fak_index_*/fak_memory_*/fak_tools_search reachable at %s (config %s; every call is still re-adjudicated by the guard floor)\n", in.URL, in.ConfigPath)
}
