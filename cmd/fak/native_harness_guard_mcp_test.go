package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dropin"
)

func TestGuardNativeHarnessEndToEnd(t *testing.T) {
	// 1. Test that guardIsFakCommand([]string{"fak", "agent"}) is true.
	fakCmdCases := []struct {
		cmd  []string
		want bool
	}{
		{[]string{"fak", "agent"}, true},
		{[]string{"fak", "agent", "--task", "hello"}, true},
		{[]string{"fak-agent", "--task", "hello"}, true},
		{[]string{"./fak", "agent"}, true},
		{[]string{"claude"}, false},
		{[]string{"bash"}, false},
		{nil, false},
	}
	for _, tc := range fakCmdCases {
		if got := guardIsFakCommand(tc.cmd); got != tc.want {
			t.Errorf("guardIsFakCommand(%v) = %v, want %v", tc.cmd, got, tc.want)
		}
	}

	// 2. Test installGuardMCPRegistrationAt on []string{"fak", "agent", "--task", "hello"}:
	// correctly writes the fak-mcp-config.json file and transforms the command into
	// ["fak", "agent", "--mcp-config", <path>, "--task", "hello"].
	sessionDir := t.TempDir()
	gwURL := "http://127.0.0.1:8765"
	origCmd := []string{"fak", "agent", "--task", "hello"}
	transformedCmd, install, err := installGuardMCPRegistrationAt(origCmd, gwURL, sessionDir)
	if err != nil {
		t.Fatalf("installGuardMCPRegistrationAt error: %v", err)
	}
	if !install.Applied {
		t.Fatal("install.Applied is false, want true")
	}
	if !install.IsFak {
		t.Fatal("install.IsFak is false, want true")
	}
	expectedConfigPath := filepath.Join(sessionDir, "fak-mcp-config.json")
	if install.ConfigPath != expectedConfigPath {
		t.Fatalf("install.ConfigPath = %q, want %q", install.ConfigPath, expectedConfigPath)
	}
	expectedURL := "http://127.0.0.1:8765/mcp"
	if install.URL != expectedURL {
		t.Fatalf("install.URL = %q, want %q", install.URL, expectedURL)
	}

	// Verify the written file content
	configFileBytes, err := os.ReadFile(expectedConfigPath)
	if err != nil {
		t.Fatalf("ReadFile(%q): %v", expectedConfigPath, err)
	}
	var parsedConfig guardMCPClientConfig
	if err := json.Unmarshal(configFileBytes, &parsedConfig); err != nil {
		t.Fatalf("Unmarshal config file error: %v, raw: %s", err, string(configFileBytes))
	}
	serverEntry, ok := parsedConfig.MCPServers[guardMCPServerName]
	if !ok {
		t.Fatalf("config file missing server %q: %+v", guardMCPServerName, parsedConfig.MCPServers)
	}
	if serverEntry.Type != "http" || serverEntry.URL != expectedURL {
		t.Fatalf("server entry = %+v, want type=http and url=%q", serverEntry, expectedURL)
	}

	// Verify command transformation
	expectedCmd := []string{"fak", "agent", "--mcp-config", expectedConfigPath, "--task", "hello"}
	if !reflect.DeepEqual(transformedCmd, expectedCmd) {
		t.Fatalf("transformed command = %v, want %v", transformedCmd, expectedCmd)
	}

	// 3. Test that FAK_MCP_CONFIG environment variable is loaded by newAgentFlagSet.
	t.Run("FAK_MCP_CONFIG_Env", func(t *testing.T) {
		mockPath := filepath.Join(t.TempDir(), "env-mcp-config.json")
		t.Setenv("FAK_MCP_CONFIG", mockPath)

		fs, af := newAgentFlagSet()
		if err := fs.Parse([]string{}); err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if *af.mcpConfig != mockPath {
			t.Fatalf("newAgentFlagSet mcpConfig from env = %q, want %q", *af.mcpConfig, mockPath)
		}

		// CLI flag overrides env var
		cliOverride := filepath.Join(t.TempDir(), "cli-override.json")
		fs2, af2 := newAgentFlagSet()
		if err := fs2.Parse([]string{"--mcp-config", cliOverride}); err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if *af2.mcpConfig != cliOverride {
			t.Fatalf("newAgentFlagSet mcpConfig override = %q, want %q", *af2.mcpConfig, cliOverride)
		}
	})

	// 4. Test that dropin.EnvVar("openai", "") (OPENAI_BASE_URL) is picked up by fak agent when --base-url is omitted.
	t.Run("OPENAI_BASE_URL_PickUp", func(t *testing.T) {
		envKey := dropin.EnvVar("openai", "")
		if envKey != "OPENAI_BASE_URL" {
			t.Fatalf("dropin.EnvVar(\"openai\", \"\") = %q, want \"OPENAI_BASE_URL\"", envKey)
		}

		mockBaseURL := "http://127.0.0.1:9099/v1"
		t.Setenv("OPENAI_BASE_URL", mockBaseURL)

		fs, af := newAgentFlagSet()
		if err := fs.Parse([]string{"--provider", "openai"}); err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if *af.baseURL != "" {
			t.Fatalf("expected *af.baseURL to be empty when flag omitted, got %q", *af.baseURL)
		}
		effectiveBaseURL := *af.baseURL
		if effectiveBaseURL == "" {
			if env := os.Getenv(dropin.EnvVar(*af.provider, "")); env != "" {
				effectiveBaseURL = env
			}
		}
		if effectiveBaseURL != mockBaseURL {
			t.Fatalf("effectiveBaseURL = %q, want %q", effectiveBaseURL, mockBaseURL)
		}

		// End-to-end execution of cmdAgent verifying the mock OpenAI server receives the request
		var serverHit atomic.Bool
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			serverHit.Store(true)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"choices": [
					{
						"message": {
							"role": "assistant",
							"content": "e2e response from mock openai endpoint"
						},
						"finish_reason": "stop"
					}
				],
				"usage": {
					"prompt_tokens": 5,
					"completion_tokens": 5,
					"total_tokens": 10
				}
			}`))
		}))
		defer ts.Close()

		t.Setenv("OPENAI_BASE_URL", ts.URL+"/v1")
		outPath := filepath.Join(t.TempDir(), "agent_run_receipt.json")
		captureAgentStdio(t, func() {
			cmdAgent([]string{
				"--native",
				"--provider", "openai",
				"--task", "verify env base url",
				"--out", outPath,
				"--max-turns", "1",
			})
		})

		if !serverHit.Load() {
			t.Fatal("expected mock OpenAI server to receive request via OPENAI_BASE_URL when --base-url was omitted")
		}

		// Verify receipt was generated
		if _, err := os.Stat(outPath); err != nil {
			t.Fatalf("expected receipt file at %q, err: %v", outPath, err)
		}
	})
}
