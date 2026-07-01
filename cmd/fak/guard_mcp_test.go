package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestGuardMCPRegistrationInstallsClaudeConfig(t *testing.T) {
	dir := t.TempDir()
	command, install, err := installGuardMCPRegistrationAt(
		[]string{"claude", "-p", "hello"},
		"http://127.0.0.1:4567",
		dir,
	)
	if err != nil {
		t.Fatalf("install mcp registration: %v", err)
	}
	if !install.Applied {
		t.Fatalf("mcp registration not applied: %+v", install)
	}
	if got, want := install.URL, "http://127.0.0.1:4567/mcp"; got != want {
		t.Fatalf("install.URL = %q, want %q", got, want)
	}
	if got, want := command[1], "--mcp-config"; got != want {
		t.Fatalf("command missing --mcp-config flag: %v", command)
	}
	if got, want := command[2], install.ConfigPath; got != want {
		t.Fatalf("mcp config path = %q, want %q", got, want)
	}
	if got, want := strings.Join(command[3:], "\x00"), strings.Join([]string{"-p", "hello"}, "\x00"); got != want {
		t.Fatalf("user args changed or --mcp-config was appended after prompt args: %v", command)
	}

	data, err := os.ReadFile(install.ConfigPath)
	if err != nil {
		t.Fatalf("read mcp config: %v", err)
	}
	var cfg guardMCPClientConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("unmarshal mcp config: %v\n%s", err, data)
	}
	fak, ok := cfg.MCPServers["fak"]
	if !ok {
		t.Fatalf("mcp config missing fak server: %+v", cfg)
	}
	if fak.Type != "http" || fak.URL != "http://127.0.0.1:4567/mcp" {
		t.Fatalf("fak server = %+v, want http server at .../mcp", fak)
	}
}

func TestGuardMCPRegistrationSkipsOffAndNonClaude(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
		command []string
	}{
		{name: "off", enabled: false, command: []string{"claude"}},
		{name: "non-claude", enabled: true, command: []string{"codex"}},
		{name: "empty", enabled: true, command: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "unused")
			command, install, err := func() ([]string, guardMCPInstall, error) {
				if !tc.enabled {
					return installGuardMCPRegistration(tc.command, false, "http://127.0.0.1:4567")
				}
				return installGuardMCPRegistrationAt(tc.command, "http://127.0.0.1:4567", dir)
			}()
			if err != nil {
				t.Fatalf("install mcp registration: %v", err)
			}
			if install.Applied {
				t.Fatalf("mcp registration applied unexpectedly: %+v", install)
			}
			if strings.Join(command, "\x00") != strings.Join(tc.command, "\x00") {
				t.Fatalf("command changed: %v -> %v", tc.command, command)
			}
		})
	}
}

func TestGuardMCPURLFromGatewayBase(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"http://127.0.0.1:4567", "http://127.0.0.1:4567/mcp"},
		{"http://127.0.0.1:4567/", "http://127.0.0.1:4567/mcp"},
	} {
		if got := guardMCPURLFromGatewayBase(tc.in); got != tc.want {
			t.Fatalf("guardMCPURLFromGatewayBase(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestGuardMCPRegistrationReachesLiveGatewayMCPEndpoint is the #1499 render witness:
// a real *gateway.Server, served exactly as cmdGuard serves it (Handler() behind an
// httptest listener standing in for the loopback listener guard.go binds), answers
// tools/list over the SAME /mcp endpoint the written --mcp-config file points at —
// proving the wired-up config reaches a live fak_index_*/fak_memory_* surface, not
// just a config file with the right shape.
func TestGuardMCPRegistrationReachesLiveGatewayMCPEndpoint(t *testing.T) {
	srv, err := gateway.New(gateway.Config{
		EngineID:     "inkernel",
		Model:        "guard-mcp-test",
		Invalidation: "global",
		Logf:         func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	httpSrv := httptest.NewServer(srv.Handler())
	defer httpSrv.Close()

	dir := t.TempDir()
	_, install, err := installGuardMCPRegistrationAt([]string{"claude"}, httpSrv.URL, dir)
	if err != nil {
		t.Fatalf("install mcp registration: %v", err)
	}
	if !install.Applied {
		t.Fatalf("mcp registration not applied: %+v", install)
	}

	body := []byte(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}`)
	req, err := http.NewRequest(http.MethodPost, install.URL, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", install.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var rpc struct {
		Result struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&rpc); err != nil {
		t.Fatalf("decode tools/list response: %v", err)
	}
	names := map[string]bool{}
	for _, tool := range rpc.Result.Tools {
		names[tool.Name] = true
	}
	for _, want := range []string{"fak_memory_run", "fak_index_verbs", "fak_tools_search"} {
		if !names[want] {
			t.Fatalf("tools/list at %s missing %s: got %+v", install.URL, want, names)
		}
	}
}
