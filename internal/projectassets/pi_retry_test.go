package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultPiRouterRetryPolicyOutlastsSaturatedRouter(t *testing.T) {
	if got := DefaultPiRouterRetryPolicy.TotalDelayMs(); got != 300000 {
		t.Fatalf("TotalDelayMs = %d, want 300000 (4+8+16+32+60*4 s)", got)
	}
}

func TestEnsurePiRetryPolicyOwnsAgentKeysOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(path, []byte(`{"theme":"dark","retry":{"enabled":false,"maxRetries":3,"provider":{"timeoutMs":5}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, modified, err := EnsurePiRetryPolicy(path, DefaultPiRouterRetryPolicy); err != nil || !modified {
		t.Fatalf("first ensure modified=%v err=%v, want a write", modified, err)
	}
	data, _ := os.ReadFile(path)
	raw := map[string]interface{}{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	if !PiRetryPolicyMatches(raw, DefaultPiRouterRetryPolicy) {
		t.Fatalf("settings retry = %v, want the default router policy enabled", raw["retry"])
	}
	provider, _ := raw["retry"].(map[string]interface{})["provider"].(map[string]interface{})
	if raw["theme"] != "dark" || provider["timeoutMs"] != float64(5) {
		t.Fatalf("settings = %v, want unrelated keys preserved", raw)
	}
	if _, modified, err := EnsurePiRetryPolicy(path, DefaultPiRouterRetryPolicy); err != nil || modified {
		t.Fatalf("second ensure modified=%v err=%v, want a no-op", modified, err)
	}
}
