package projectassets

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateCodexConfig(t *testing.T) {
	out := GenerateCodexConfig("http://127.0.0.1:8080/v1", "qwen38:27b-q4", "responses", "OPENAI_API_KEY")
	if !strings.Contains(out, `model_provider = "fak"`) {
		t.Errorf("missing model_provider = \"fak\": %s", out)
	}
	if !strings.Contains(out, `model = "qwen38:27b-q4"`) {
		t.Errorf("missing model = \"qwen38:27b-q4\": %s", out)
	}
	if !strings.Contains(out, `[model_providers.fak]`) {
		t.Errorf("missing [model_providers.fak]: %s", out)
	}
	if !strings.Contains(out, `base_url = "http://127.0.0.1:8080/v1"`) {
		t.Errorf("missing base_url: %s", out)
	}
	if !strings.Contains(out, `wire_api = "responses"`) {
		t.Errorf("missing wire_api: %s", out)
	}
}

func TestEnsureCodexProviderConfigFresh(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "config.toml")

	modified, err := EnsureCodexProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4", "responses", "OPENAI_API_KEY")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true on fresh file")
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("failed to read created config: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, `model_provider = "fak"`) {
		t.Errorf("missing model_provider in config: %s", content)
	}
	if !strings.Contains(content, `[model_providers.fak]`) {
		t.Errorf("missing [model_providers.fak] in config: %s", content)
	}

	// Idempotent second call
	modifiedAgain, err := EnsureCodexProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4", "responses", "OPENAI_API_KEY")
	if err != nil {
		t.Fatalf("unexpected error on second call: %v", err)
	}
	if modifiedAgain {
		t.Errorf("expected modified=false on idempotent second call")
	}
}

func TestEnsureCodexProviderConfigPreservesExisting(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "config.toml")

	initial := `# Top level comment
approval_policy = "never"

[mcp_servers.my_custom]
command = "npx"
args = ["custom-server"]

[model_providers.other]
name = "Other Provider"
base_url = "https://api.example.com/v1"
`
	if err := os.WriteFile(targetPath, []byte(initial), 0644); err != nil {
		t.Fatalf("failed to write initial: %v", err)
	}

	modified, err := EnsureCodexProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4", "responses", "OPENAI_API_KEY")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true when adding fak provider")
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}
	content := string(data)

	// Check preserved content
	if !strings.Contains(content, `approval_policy = "never"`) {
		t.Errorf("lost approval_policy: %s", content)
	}
	if !strings.Contains(content, `[mcp_servers.my_custom]`) {
		t.Errorf("lost mcp_servers.my_custom: %s", content)
	}
	if !strings.Contains(content, `[model_providers.other]`) {
		t.Errorf("lost model_providers.other: %s", content)
	}

	// Check new content
	if !strings.Contains(content, `model_provider = "fak"`) {
		t.Errorf("missing model_provider = \"fak\": %s", content)
	}
	if !strings.Contains(content, `[model_providers.fak]`) {
		t.Errorf("missing [model_providers.fak]: %s", content)
	}
}

func TestResolveCodexConfigFile(t *testing.T) {
	tmp := t.TempDir()
	// Test explicit dir with .codex/config.toml
	codexDir := filepath.Join(tmp, ".codex")
	if err := os.MkdirAll(codexDir, 0755); err != nil {
		t.Fatal(err)
	}
	cfgFile := filepath.Join(codexDir, "config.toml")
	if err := os.WriteFile(cfgFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	resolved := ResolveCodexConfigFile("", tmp)
	if resolved != cfgFile {
		t.Errorf("ResolveCodexConfigFile = %q, want %q", resolved, cfgFile)
	}
}
