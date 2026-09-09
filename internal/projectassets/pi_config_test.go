package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGeneratePiConfig(t *testing.T) {
	out, err := GeneratePiConfig("http://127.0.0.1:8080/v1", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("failed to parse generated config: %v", err)
	}

	provs, ok := parsed["providers"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing providers map in %s", string(out))
	}
	fak, ok := provs["fak"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing fak provider in %s", string(out))
	}
	if fak["baseUrl"] != "http://127.0.0.1:8080/v1" {
		t.Errorf("baseUrl = %v, want http://127.0.0.1:8080/v1", fak["baseUrl"])
	}
	if fak["api"] != "openai-completions" {
		t.Errorf("api = %v, want openai-completions", fak["api"])
	}
	if fak["apiKey"] != "fak" {
		t.Errorf("apiKey = %v, want fak", fak["apiKey"])
	}
	models, ok := fak["models"].([]interface{})
	if !ok || len(models) == 0 {
		t.Fatalf("missing models array: %v", fak["models"])
	}
	m0, ok := models[0].(map[string]interface{})
	if !ok {
		t.Fatalf("models[0] not an object: %v", models[0])
	}
	if m0["id"] != "qwen38:27b-q4" {
		t.Errorf("model id = %v, want qwen38:27b-q4", m0["id"])
	}
	compat, ok := m0["compat"].(map[string]interface{})
	if !ok || compat[piDeveloperRoleKey] != false {
		t.Errorf("compat = %v, want %s: false", compat, piDeveloperRoleKey)
	}
}

func TestEnsurePiProviderConfigFresh(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "models.json")
	resolvedPath, modified, err := EnsurePiProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true on fresh directory")
	}
	if resolvedPath != targetPath {
		t.Errorf("resolvedPath = %q, want %q", resolvedPath, targetPath)
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("failed to read created config: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to parse created config: %v", err)
	}

	provs := parsed["providers"].(map[string]interface{})
	fak := provs["fak"].(map[string]interface{})
	if fak["baseUrl"] != "http://127.0.0.1:8080/v1" {
		t.Errorf("unexpected baseUrl: %v", fak["baseUrl"])
	}

	// Calling again should be idempotent (modified=false)
	_, modifiedAgain, err := EnsurePiProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error on second call: %v", err)
	}
	if modifiedAgain {
		t.Errorf("expected modified=false on idempotent call")
	}
}

func TestEnsurePiProviderConfigPreservesExisting(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "models.json")
	initialConfig := `{
  "providers": {
    "anthropic": {
      "baseUrl": "https://api.anthropic.com",
      "apiKey": "$ANTHROPIC_API_KEY"
    },
    "ollama": {
      "baseUrl": "http://localhost:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "models": [
        { "id": "llama3.1:8b" }
      ]
    }
  }
}`
	if err := os.WriteFile(targetPath, []byte(initialConfig), 0644); err != nil {
		t.Fatal(err)
	}

	resolvedPath, modified, err := EnsurePiProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true when adding fak provider")
	}
	if resolvedPath != targetPath {
		t.Errorf("resolvedPath = %q, want %q", resolvedPath, targetPath)
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to parse updated config: %v", err)
	}

	provs := parsed["providers"].(map[string]interface{})

	// Check preserved providers
	if provs["anthropic"] == nil {
		t.Errorf("lost anthropic provider")
	}
	if provs["ollama"] == nil {
		t.Errorf("lost ollama provider")
	}

	// Check added fak provider
	fak, ok := provs["fak"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing fak provider")
	}
	if fak["baseUrl"] != "http://127.0.0.1:8080/v1" {
		t.Errorf("unexpected baseUrl: %v", fak["baseUrl"])
	}
}

func TestEnsurePiProviderConfigUpdatesExistingFak(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "models.json")
	initialConfig := `{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [
        { "id": "existing-model", "name": "Existing Model" }
      ]
    }
  }
}`
	if err := os.WriteFile(targetPath, []byte(initialConfig), 0644); err != nil {
		t.Fatal(err)
	}

	_, modified, err := EnsurePiProviderConfig(targetPath, "http://127.0.0.1:8080/v1", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true when updating baseUrl and adding model")
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	provs := parsed["providers"].(map[string]interface{})
	fak := provs["fak"].(map[string]interface{})
	if fak["baseUrl"] != "http://127.0.0.1:8080/v1" {
		t.Errorf("baseUrl not updated: %v", fak["baseUrl"])
	}

	models := fak["models"].([]interface{})
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	m0 := models[0].(map[string]interface{})
	m1 := models[1].(map[string]interface{})
	if m0["id"] != "existing-model" || m1["id"] != "qwen38:27b-q4" {
		t.Errorf("unexpected models: %v, %v", m0, m1)
	}
}

func TestResolvePiConfigPath(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("PI_CODING_AGENT_DIR", filepath.Join(tmp, "agent"))

	// 1. Empty target resolves to PI_CODING_AGENT_DIR/models.json
	gotDefault := ResolvePiConfigPath("")
	wantDefault := filepath.Join(tmp, "agent", "models.json")
	if gotDefault != wantDefault {
		t.Errorf("ResolvePiConfigPath(\"\") = %q, want %q", gotDefault, wantDefault)
	}

	// 2. Direct JSON file
	jsonFile := filepath.Join(tmp, "custom_models.json")
	if got := ResolvePiConfigPath(jsonFile); got != jsonFile {
		t.Errorf("ResolvePiConfigPath(%q) = %q, want %q", jsonFile, got, jsonFile)
	}

	// 3. Directory target
	dirTarget := filepath.Join(tmp, "custom_dir")
	wantDir := filepath.Join(dirTarget, "models.json")
	if got := ResolvePiConfigPath(dirTarget); got != wantDir {
		t.Errorf("ResolvePiConfigPath(%q) = %q, want %q", dirTarget, got, wantDir)
	}
}

func TestNormalizePiBaseURL(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"", DefaultPiBaseURL},
		{"127.0.0.1:8080", "http://127.0.0.1:8080/v1"},
		{"http://127.0.0.1:8080", "http://127.0.0.1:8080/v1"},
		{"http://127.0.0.1:8080/", "http://127.0.0.1:8080/v1"},
		{"http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1"},
		{"https://remote-mac.local:8080/v1", "https://remote-mac.local:8080/v1"},
		{"https://remote-mac.local:8080", "https://remote-mac.local:8080/v1"},
	}

	for _, tc := range tests {
		got := NormalizePiBaseURL(tc.input)
		if got != tc.want {
			t.Errorf("NormalizePiBaseURL(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestNormalizePiModelID(t *testing.T) {
	if got := NormalizePiModelID(""); got != DefaultPiModelID {
		t.Errorf("NormalizePiModelID(\"\") = %q, want %q", got, DefaultPiModelID)
	}
	if got := NormalizePiModelID("mock"); got != DefaultPiModelID {
		t.Errorf("NormalizePiModelID(\"mock\") = %q, want %q", got, DefaultPiModelID)
	}
	if got := NormalizePiModelID("custom-model"); got != "custom-model" {
		t.Errorf("NormalizePiModelID(\"custom-model\") = %q, want %q", got, "custom-model")
	}
}
