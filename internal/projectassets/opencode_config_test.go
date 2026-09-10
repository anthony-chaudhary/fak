package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGenerateOpenCodeConfig(t *testing.T) {
	out, err := GenerateOpenCodeConfig("http://127.0.0.1:9090/v1", "my-test-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("failed to parse generated config: %v", err)
	}

	if parsed["snapshot"] != false {
		t.Errorf("expected snapshot to be false, got %v", parsed["snapshot"])
	}

	prov, ok := parsed["provider"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing provider map")
	}
	fak, ok := prov["fak"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing fak provider")
	}
	opts, ok := fak["options"].(map[string]interface{})
	if !ok || opts["baseURL"] != "http://127.0.0.1:9090/v1" {
		t.Errorf("unexpected options baseURL: %v", opts)
	}
	models, ok := fak["models"].(map[string]interface{})
	if !ok || models["my-test-model"] == nil {
		t.Errorf("missing model entry in fak provider: %v", models)
	}
}

func TestEnsureOpenCodeProviderConfigFresh(t *testing.T) {
	tmp := t.TempDir()
	modified, err := EnsureOpenCodeProviderConfig(tmp, "http://127.0.0.1:8080/v1", "qwen38:27b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true on fresh directory")
	}

	configPath := filepath.Join(tmp, "opencode.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read created config: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to parse created config: %v", err)
	}
	if parsed["snapshot"] != false {
		t.Errorf("expected snapshot: false")
	}

	// Calling again should be idempotent (modified=false)
	modifiedAgain, err := EnsureOpenCodeProviderConfig(tmp, "http://127.0.0.1:8080/v1", "qwen38:27b")
	if err != nil {
		t.Fatalf("unexpected error on second call: %v", err)
	}
	if modifiedAgain {
		t.Errorf("expected modified=false on idempotent call")
	}
}

func TestEnsureOpenCodeProviderConfigPreservesExisting(t *testing.T) {
	tmp := t.TempDir()
	initialConfig := `{
  "$schema": "https://opencode.ai/config.json",
  "instructions": ["AGENTS.md"],
  "skills": {
    "paths": [".agents/skills"]
  },
  "snapshot": true
}`
	if err := os.WriteFile(filepath.Join(tmp, "opencode.json"), []byte(initialConfig), 0644); err != nil {
		t.Fatal(err)
	}

	modified, err := EnsureOpenCodeProviderConfig(tmp, "http://127.0.0.1:8181/v1", "custom-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true when updating config")
	}

	data, err := os.ReadFile(filepath.Join(tmp, "opencode.json"))
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	// Verify snapshot was corrected to false
	if parsed["snapshot"] != false {
		t.Errorf("snapshot was not updated to false")
	}
	// Verify existing instructions preserved
	inst, ok := parsed["instructions"].([]interface{})
	if !ok || len(inst) != 1 || inst[0] != "AGENTS.md" {
		t.Errorf("instructions was not preserved: %v", parsed["instructions"])
	}
	// Verify skills preserved
	if parsed["skills"] == nil {
		t.Errorf("skills was not preserved")
	}
	// Verify fak provider added
	prov := parsed["provider"].(map[string]interface{})
	fak := prov["fak"].(map[string]interface{})
	opts := fak["options"].(map[string]interface{})
	if opts["baseURL"] != "http://127.0.0.1:8181/v1" {
		t.Errorf("expected updated baseURL, got %v", opts["baseURL"])
	}
}

func TestEnsureOpenCodeProviderConfigWiresHaloFastTier(t *testing.T) {
	tmp := t.TempDir()
	initialConfig := `{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "agent.tier.fast": {
      "model": "qwen-2.5-coder-32b-instruct"
    }
  }
}`
	if err := os.WriteFile(filepath.Join(tmp, "opencode.json"), []byte(initialConfig), 0644); err != nil {
		t.Fatal(err)
	}

	modified, err := EnsureOpenCodeProviderConfig(tmp, "http://127.0.0.1:8080/v1", DefaultOpenCodeHaloModelID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !modified {
		t.Fatalf("expected modified=true")
	}

	data, err := os.ReadFile(filepath.Join(tmp, "opencode.json"))
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	agents := parsed["agent"].(map[string]interface{})
	fast := agents["agent.tier.fast"].(map[string]interface{})
	if fast["model"] != "fak/qwen-2.5-coder-32b-instruct" {
		t.Errorf("expected fast tier model to be wired to fak/qwen-2.5-coder-32b-instruct, got %v", fast["model"])
	}
}

