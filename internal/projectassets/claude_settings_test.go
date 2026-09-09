package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGenerateClaudeSettings(t *testing.T) {
	out, err := GenerateClaudeSettings("http://127.0.0.1:8080", "qwen38:27b-q4", "fak-local-dogfood")
	if err != nil {
		t.Fatalf("GenerateClaudeSettings failed: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal generated settings: %v", err)
	}

	envRaw, ok := parsed["env"]
	if !ok {
		t.Fatalf("missing env key in generated settings")
	}
	env, ok := envRaw.(map[string]interface{})
	if !ok {
		t.Fatalf("env is not a map: %T", envRaw)
	}

	if env["ANTHROPIC_BASE_URL"] != "http://127.0.0.1:8080" {
		t.Errorf("ANTHROPIC_BASE_URL = %v, want http://127.0.0.1:8080", env["ANTHROPIC_BASE_URL"])
	}
	if env["ANTHROPIC_API_KEY"] != "fak-local-dogfood" {
		t.Errorf("ANTHROPIC_API_KEY = %v, want fak-local-dogfood", env["ANTHROPIC_API_KEY"])
	}
	if env["ANTHROPIC_MODEL"] != "qwen38:27b-q4" {
		t.Errorf("ANTHROPIC_MODEL = %v, want qwen38:27b-q4", env["ANTHROPIC_MODEL"])
	}
	if env["ANTHROPIC_DEFAULT_OPUS_MODEL"] != "qwen38:27b-q4" {
		t.Errorf("ANTHROPIC_DEFAULT_OPUS_MODEL = %v, want qwen38:27b-q4", env["ANTHROPIC_DEFAULT_OPUS_MODEL"])
	}
	if env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] != "1" {
		t.Errorf("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = %v, want 1", env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"])
	}
	if env["API_TIMEOUT_MS"] != "1800000" {
		t.Errorf("API_TIMEOUT_MS = %v, want 1800000", env["API_TIMEOUT_MS"])
	}
}

func TestEnsureClaudeSettingsFresh(t *testing.T) {
	tmp := t.TempDir()

	modified, err := EnsureClaudeSettingsConfig(tmp, "127.0.0.1:8080/v1", "qwen38:27b-q4", "")
	if err != nil {
		t.Fatalf("EnsureClaudeSettingsConfig failed: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true on fresh create")
	}

	settingsFile := filepath.Join(tmp, ".claude", "settings.json")
	data, err := os.ReadFile(settingsFile)
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal written settings: %v", err)
	}

	env := parsed["env"].(map[string]interface{})
	// Normalization strips trailing /v1 and adds http://
	if env["ANTHROPIC_BASE_URL"] != "http://127.0.0.1:8080" {
		t.Errorf("expected normalized ANTHROPIC_BASE_URL=http://127.0.0.1:8080, got %v", env["ANTHROPIC_BASE_URL"])
	}
	if env["ANTHROPIC_API_KEY"] != DefaultClaudeAPIKey {
		t.Errorf("expected default API key %v, got %v", DefaultClaudeAPIKey, env["ANTHROPIC_API_KEY"])
	}

	// Second run should be no-op
	modifiedAgain, err := EnsureClaudeSettingsConfig(tmp, "http://127.0.0.1:8080", "qwen38:27b-q4", "")
	if err != nil {
		t.Fatalf("second EnsureClaudeSettingsConfig failed: %v", err)
	}
	if modifiedAgain {
		t.Fatalf("expected modified=false on idempotent rerun")
	}
}

func TestEnsureClaudeSettingsPreservesExisting(t *testing.T) {
	tmp := t.TempDir()
	settingsDir := filepath.Join(tmp, ".claude")
	if err := os.MkdirAll(settingsDir, 0o755); err != nil {
		t.Fatalf("mkdir .claude: %v", err)
	}

	existingContent := `{
  "outputStyle": "Concise",
  "theme": "dark",
  "hooks": {
    "PreToolUse": [
      {
        "command": "fak hooks agent pretool"
      }
    ]
  }
}`
	if err := os.WriteFile(filepath.Join(settingsDir, "settings.json"), []byte(existingContent), 0o644); err != nil {
		t.Fatalf("write existing settings.json: %v", err)
	}

	modified, err := EnsureClaudeSettingsConfig(tmp, "http://127.0.0.1:8080", "qwen38:27b-q4", "custom-key")
	if err != nil {
		t.Fatalf("EnsureClaudeSettingsConfig failed: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true")
	}

	data, err := os.ReadFile(filepath.Join(settingsDir, "settings.json"))
	if err != nil {
		t.Fatalf("read updated settings.json: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Assert existing keys preserved
	if parsed["outputStyle"] != "Concise" {
		t.Errorf("outputStyle overwritten: %v", parsed["outputStyle"])
	}
	if parsed["theme"] != "dark" {
		t.Errorf("theme overwritten: %v", parsed["theme"])
	}
	if parsed["hooks"] == nil {
		t.Errorf("hooks removed")
	}

	// Assert env injected
	env := parsed["env"].(map[string]interface{})
	if env["ANTHROPIC_BASE_URL"] != "http://127.0.0.1:8080" {
		t.Errorf("ANTHROPIC_BASE_URL = %v", env["ANTHROPIC_BASE_URL"])
	}
	if env["ANTHROPIC_API_KEY"] != "custom-key" {
		t.Errorf("ANTHROPIC_API_KEY = %v", env["ANTHROPIC_API_KEY"])
	}
}

func TestNormalizeClaudeBaseURL(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"", DefaultClaudeBaseURL},
		{"127.0.0.1:8080", "http://127.0.0.1:8080"},
		{"127.0.0.1:8080/", "http://127.0.0.1:8080"},
		{"http://localhost:8080/v1", "http://localhost:8080"},
		{"http://localhost:8080/v1/", "http://localhost:8080"},
		{"https://remote-mac.local:8080", "https://remote-mac.local:8080"},
	}
	for _, tc := range cases {
		got := NormalizeClaudeBaseURL(tc.input)
		if got != tc.want {
			t.Errorf("NormalizeClaudeBaseURL(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
