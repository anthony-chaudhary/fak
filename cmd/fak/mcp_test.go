package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMCPConfigJSONAndCodex(t *testing.T) {
	t.Chdir(t.TempDir())

	var out, errb bytes.Buffer
	if code := runMCP(&out, &errb, []string{"config", "--client", "project", "--portable", "--expose", "fak_capabilities", "--json"}); code != 0 {
		t.Fatalf("mcp config project = %d, stderr=%s", code, errb.String())
	}
	var proj map[string]any
	if err := json.Unmarshal(out.Bytes(), &proj); err != nil {
		t.Fatalf("unmarshal project config: %v\n%s", err, out.String())
	}
	if proj["schema"] != mcpConfigSchema || proj["command"] != "fak" {
		t.Fatalf("unexpected project config payload: %+v", proj)
	}
	wm, _ := proj["workspace_mode"].(map[string]any)
	if wm["mode"] != "standalone" {
		t.Fatalf("expected standalone workspace_mode in temp dir, got %+v", wm)
	}

	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"config", "--client", "codex", "--portable"}); code != 0 {
		t.Fatalf("mcp config codex = %d, stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "[mcp_servers.fak]") || !strings.Contains(out.String(), `"serve", "--stdio"`) {
		t.Fatalf("unexpected codex TOML snippet:\n%s", out.String())
	}
}

func TestMCPInstallMergeDryRunForceAndUninstall(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	// Seed an existing .mcp.json with a sibling server to prove preservation.
	mcpPath := filepath.Join(dir, ".mcp.json")
	seed := `{"mcpServers":{"other":{"command":"other-bin","args":["--stdio"]}}}` + "\n"
	if err := os.WriteFile(mcpPath, []byte(seed), 0o644); err != nil {
		t.Fatalf("seed .mcp.json: %v", err)
	}

	// Dry-run must not modify .mcp.json.
	var out, errb bytes.Buffer
	if code := runMCP(&out, &errb, []string{"install", "--client", "project", "--portable", "--dry-run", "--json"}); code != 0 {
		t.Fatalf("mcp install --dry-run = %d, stderr=%s", code, errb.String())
	}
	gotSeed, _ := os.ReadFile(mcpPath)
	if string(gotSeed) != seed {
		t.Fatalf("dry-run modified .mcp.json:\n%s", string(gotSeed))
	}

	// Real install (--skip-verify for hermetic unit test; handshake is tested separately).
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"install", "--client", "project", "--portable", "--skip-verify", "--json"}); code != 0 {
		t.Fatalf("mcp install = %d, stderr=%s", code, errb.String())
	}

	// Verify sibling server is preserved and fak is added without --policy (standalone mode).
	raw, err := os.ReadFile(mcpPath)
	if err != nil {
		t.Fatalf("read .mcp.json: %v", err)
	}
	var parsed struct {
		MCPServers map[string]mcpJSONServer `json:"mcpServers"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("parse .mcp.json: %v\n%s", err, string(raw))
	}
	if parsed.MCPServers["other"].Command != "other-bin" {
		t.Fatalf("sibling server lost: %+v", parsed.MCPServers)
	}
	if parsed.MCPServers["fak"].Command != "fak" {
		t.Fatalf("fak server missing or wrong: %+v", parsed.MCPServers["fak"])
	}

	// Second install without --force must refuse.
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"install", "--client", "project", "--portable", "--skip-verify", "--json"}); code == 0 {
		t.Fatalf("expected non-zero exit when fak already configured without --force")
	}

	// Status reports configured.
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"status", "--client", "project", "--json"}); code != 0 {
		t.Fatalf("mcp status = %d, stderr=%s", code, errb.String())
	}
	var st map[string]any
	if err := json.Unmarshal(out.Bytes(), &st); err != nil {
		t.Fatalf("unmarshal status: %v", err)
	}
	if st["configured"] != true {
		t.Fatalf("expected configured=true, got %+v", st)
	}

	// Uninstall removes only fak and keeps sibling server.
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"uninstall", "--client", "project", "--json"}); code != 0 {
		t.Fatalf("mcp uninstall = %d, stderr=%s", code, errb.String())
	}
	rawAfter, _ := os.ReadFile(mcpPath)
	var parsedAfter struct {
		MCPServers map[string]mcpJSONServer `json:"mcpServers"`
	}
	if err := json.Unmarshal(rawAfter, &parsedAfter); err != nil {
		t.Fatalf("parse .mcp.json after uninstall: %v", err)
	}
	if _, ok := parsedAfter.MCPServers["fak"]; ok {
		t.Fatalf("fak should be removed after uninstall: %+v", parsedAfter.MCPServers)
	}
	if parsedAfter.MCPServers["other"].Command != "other-bin" {
		t.Fatalf("sibling server lost on uninstall: %+v", parsedAfter.MCPServers)
	}
}

func TestMCPOpenCodeAutoAndHelp(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	// Help flags must exit 0 and write usage to stdout.
	var out, errb bytes.Buffer
	if code := runMCP(&out, &errb, []string{"install", "-h"}); code != 0 {
		t.Fatalf("mcp install -h = %d, stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "--client") {
		t.Fatalf("expected --client in -h stdout, got:\n%s", out.String())
	}

	// Seed opencode.json with sibling MCP entry and non-MCP keys.
	ocPath := filepath.Join(dir, "opencode.json")
	seed := `{"$schema":"https://opencode.ai/config.json","mcp":{"dos":{"command":["dos-mcp"],"type":"local","enabled":true}}}` + "\n"
	if err := os.WriteFile(ocPath, []byte(seed), 0o644); err != nil {
		t.Fatal(err)
	}

	// --client auto must detect opencode.json cleanly.
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"install", "--client", "auto", "--portable", "--skip-verify", "--json"}); code != 0 {
		t.Fatalf("mcp install auto (opencode) = %d, stderr=%s", code, errb.String())
	}
	var inst map[string]any
	if err := json.Unmarshal(out.Bytes(), &inst); err != nil {
		t.Fatalf("unmarshal install: %v\n%s", err, out.String())
	}
	if inst["client"] != "opencode" {
		t.Fatalf("expected auto client=opencode, got %+v", inst)
	}

	raw, err := os.ReadFile(ocPath)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if doc["$schema"] != "https://opencode.ai/config.json" {
		t.Fatalf("lost $schema key: %+v", doc)
	}
	mcpMap, _ := doc["mcp"].(map[string]any)
	if mcpMap["dos"] == nil || mcpMap["fak"] == nil {
		t.Fatalf("expected both dos and fak in opencode.json mcp: %+v", mcpMap)
	}

	// Status auto picks up opencode entry.
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"status", "--client", "auto", "--json"}); code != 0 {
		t.Fatalf("mcp status auto = %d, stderr=%s", code, errb.String())
	}
	var st map[string]any
	if err := json.Unmarshal(out.Bytes(), &st); err != nil {
		t.Fatal(err)
	}
	if st["configured"] != true || st["client"] != "opencode" {
		t.Fatalf("expected configured opencode status, got %+v", st)
	}

	// Uninstall preserves sibling dos entry and removes fak.
	out.Reset()
	errb.Reset()
	if code := runMCP(&out, &errb, []string{"uninstall", "--client", "auto", "--json"}); code != 0 {
		t.Fatalf("mcp uninstall auto = %d, stderr=%s", code, errb.String())
	}
	rawAfter, _ := os.ReadFile(ocPath)
	var docAfter map[string]any
	if err := json.Unmarshal(rawAfter, &docAfter); err != nil {
		t.Fatal(err)
	}
	mcpAfter, _ := docAfter["mcp"].(map[string]any)
	if _, exists := mcpAfter["fak"]; exists {
		t.Fatalf("fak should be removed from opencode.json: %+v", mcpAfter)
	}
	if mcpAfter["dos"] == nil {
		t.Fatalf("sibling dos entry lost from opencode.json: %+v", mcpAfter)
	}
}

func TestDetectWorkspaceMode(t *testing.T) {
	// Standalone: temp dir without dos.toml.
	t.Chdir(t.TempDir())
	if got := detectWorkspaceMode(); got.Mode != "standalone" || got.InFakRepo {
		t.Fatalf("expected standalone mode, got %+v", got)
	}

	// Public repo: temp dir with dos.toml and no sibling fak-private.
	pub := t.TempDir()
	if err := os.WriteFile(filepath.Join(pub, "dos.toml"), []byte("[branch_roles]\n"), 0o644); err != nil {
		t.Fatalf("seed dos.toml: %v", err)
	}
	t.Chdir(pub)
	if got := detectWorkspaceMode(); got.Mode != "public_repo" || !got.InFakRepo || got.CompanionPresent {
		t.Fatalf("expected public_repo mode, got %+v", got)
	}
}
