package projectassets

import (
	"os"
	"path/filepath"
	"testing"
)

// TestEnsurePiProviderConfigRaisesStaleMaxTokens is the regression witness for the
// Halo Qwen entry that kept a hand-set maxTokens 4096 while the derivation (and
// the compaction reserve built from it) assumed 16384. With no recorded pin,
// maxTokens follows the derived output budget in both directions.
func TestEnsurePiProviderConfigRaisesStaleMaxTokens(t *testing.T) {
	target := filepath.Join(t.TempDir(), "models.json")
	raw := `{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [
        {"id": "qwen38:27b-q4", "name": "old", "contextWindow": 131072, "maxTokens": 4096,
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
         "compat": {"` + piDeveloperRoleKey + `": false}}
      ]
    }
  }
}`
	if err := os.WriteFile(target, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := EnsurePiProviderConfigForWindow(target, "http://127.0.0.1:8080/v1", "qwen38:27b-q4", 131072); err != nil {
		t.Fatalf("EnsurePiProviderConfigForWindow: %v", err)
	}
	model := firstFakModel(t, readJSONFile(t, target))
	want := PiSafeContextBudget(131072)
	if mt, _ := numericField(model["maxTokens"]); mt != want.MaxOutputTokens {
		t.Fatalf("maxTokens = %d, want the derived output budget %d", mt, want.MaxOutputTokens)
	}
}
