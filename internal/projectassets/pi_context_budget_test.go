package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestPiSafeContextBudget pins the core doctrine: the resident target is at most HALF the
// served window, never the raw cap, and the derived reserve/keep are ordered sanely.
func TestPiSafeContextBudget(t *testing.T) {
	cases := []struct {
		name       string
		window     int
		wantTarget int
	}{
		{"128k window halves to 65536", 131072, 65536},
		{"64k window halves to 32768", 65536, 32768},
		{"16k window halves to 8192", 16384, 8192},
		{"8k window halved stays legal", 8192, 4096},
		{"unset window falls back and halves", 0, DefaultPiServedWindow / 2},
		{"negative window falls back", -5, DefaultPiServedWindow / 2},
		{"absurdly small window floors at the default prior", 1024, DefaultPiServedWindow / 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := PiSafeContextBudget(tc.window)
			if b.ResidentTarget != tc.wantTarget {
				t.Fatalf("ResidentTarget = %d, want %d", b.ResidentTarget, tc.wantTarget)
			}
			// The core invariant, stated against the EFFECTIVE window (after fallback).
			if b.ResidentTarget > b.ServedWindow/2 {
				t.Fatalf("resident target %d exceeds 50%% of served window %d", b.ResidentTarget, b.ServedWindow)
			}
			if b.ResidentTarget > b.ServedWindow {
				t.Fatalf("resident target %d exceeds served window %d", b.ResidentTarget, b.ServedWindow)
			}
			if b.OutputReserve <= 0 || b.OutputReserve > b.ResidentTarget {
				t.Fatalf("OutputReserve = %d, want in (0, %d]", b.OutputReserve, b.ResidentTarget)
			}
			if b.KeepRecentTokens <= 0 || b.KeepRecentTokens > b.ResidentTarget {
				t.Fatalf("KeepRecentTokens = %d, want in (0, %d]", b.KeepRecentTokens, b.ResidentTarget)
			}
			if b.Provenance != PiBudgetProvenance {
				t.Fatalf("Provenance = %q, want %q", b.Provenance, PiBudgetProvenance)
			}
		})
	}
}

// TestPiSafeContextBudgetMonotone witnesses that a larger served window never yields a smaller
// resident target — the derivation is monotone, so an operator widening the window cannot
// accidentally shrink the resident budget.
func TestPiSafeContextBudgetMonotone(t *testing.T) {
	prev := 0
	for _, w := range []int{16384, 32768, 65536, 131072, 262144, 1048576} {
		got := PiSafeContextBudget(w).ResidentTarget
		if got < prev {
			t.Fatalf("window %d gave target %d, below prior %d (not monotone)", w, got, prev)
		}
		prev = got
	}
}

// TestGeneratePiConfigNeverAdvertisesRawWindow is the regression witness for the "cap is not
// target" violation: the generated models.json must NOT advertise the raw served window.
func TestGeneratePiConfigNeverAdvertisesRawWindow(t *testing.T) {
	const window = 131072
	out, err := GeneratePiConfigForWindow("http://127.0.0.1:8080/v1", "qwen38:27b-q4", window)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("parse generated config: %v", err)
	}
	model := firstFakModel(t, parsed)
	cw, ok := numericField(model["contextWindow"])
	if !ok {
		t.Fatalf("contextWindow missing or not numeric: %v", model["contextWindow"])
	}
	if cw == window {
		t.Fatalf("contextWindow advertises the raw served window %d (cap-is-not-target violation)", window)
	}
	if cw > window/2 {
		t.Fatalf("contextWindow = %d exceeds 50%% of served window %d", cw, window)
	}
	mt, ok := numericField(model["maxTokens"])
	if !ok || mt <= 0 {
		t.Fatalf("maxTokens missing or non-positive: %v", model["maxTokens"])
	}
	// The model must still be usable: id and provider wiring preserved.
	if model["id"] != "qwen38:27b-q4" {
		t.Fatalf("model id = %v, want qwen38:27b-q4", model["id"])
	}
}

// TestEnsurePiProviderConfigRepairsRawWindow witnesses the upgrade path: a models.json fak
// itself wrote with the pre-doctrine raw contextWindow is corrected to the safe target.
func TestEnsurePiProviderConfigRepairsRawWindow(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "models.json")
	raw := `{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [
        {"id": "qwen38:27b-q4", "name": "old", "contextWindow": 131072, "maxTokens": 16384,
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
         "compat": {"` + piDeveloperRoleKey + `": false}}
      ]
    },
    "other": {"baseUrl": "https://example.invalid/v1", "models": []}
  }
}`
	if err := os.WriteFile(target, []byte(raw), 0644); err != nil {
		t.Fatalf("seed models.json: %v", err)
	}

	_, modified, err := EnsurePiProviderConfig(target, "http://127.0.0.1:8080/v1", "qwen38:27b-q4")
	if err != nil {
		t.Fatalf("EnsurePiProviderConfig: %v", err)
	}
	if !modified {
		t.Fatal("expected the raw contextWindow to be repaired, modified=false")
	}

	after := readJSONFile(t, target)
	model := firstFakModel(t, after)
	cw, _ := numericField(model["contextWindow"])
	if cw != DefaultPiServedWindow/2 {
		t.Fatalf("contextWindow = %d, want %d (half the window)", cw, DefaultPiServedWindow/2)
	}
	// Non-clobber invariant: the unrelated provider survives.
	provs := after["providers"].(map[string]interface{})
	if _, ok := provs["other"]; !ok {
		t.Fatal("repair dropped an unrelated provider")
	}
}

// TestEnsurePiSafeCompactionFresh witnesses the settings writer creating a safe compaction
// block from nothing.
func TestEnsurePiSafeCompactionFresh(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	budget := PiSafeContextBudget(131072)

	path, modified, err := EnsurePiSafeCompaction(target, budget)
	if err != nil {
		t.Fatalf("EnsurePiSafeCompaction: %v", err)
	}
	if !modified {
		t.Fatal("expected a fresh settings.json to be created, modified=false")
	}
	if path != target {
		t.Fatalf("path = %q, want %q", path, target)
	}
	doc := readJSONFile(t, target)
	block, ok := doc["compaction"].(map[string]interface{})
	if !ok {
		t.Fatalf("compaction block missing: %v", doc["compaction"])
	}
	if enabled, _ := block["enabled"].(bool); !enabled {
		t.Fatal("compaction.enabled should be true")
	}
	if got, _ := numericField(block["reserveTokens"]); got != budget.OutputReserve {
		t.Fatalf("reserveTokens = %d, want %d", got, budget.OutputReserve)
	}
	if got, _ := numericField(block["keepRecentTokens"]); got != budget.KeepRecentTokens {
		t.Fatalf("keepRecentTokens = %d, want %d", got, budget.KeepRecentTokens)
	}
}

// TestEnsurePiSafeCompactionPreservesUserKeys witnesses the non-clobber invariant: unrelated
// settings and a stricter operator reserve survive the write.
func TestEnsurePiSafeCompactionPreservesUserKeys(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	// The operator already set a STRICTER reserve (larger than the derived one) and a theme.
	seed := `{
  "theme": "light",
  "defaultProvider": "hive-ai",
  "compaction": {"enabled": true, "reserveTokens": 100000, "keepRecentTokens": 50000}
}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatalf("seed settings.json: %v", err)
	}
	budget := PiSafeContextBudget(131072)

	_, _, err := EnsurePiSafeCompaction(target, budget)
	if err != nil {
		t.Fatalf("EnsurePiSafeCompaction: %v", err)
	}
	doc := readJSONFile(t, target)
	if doc["theme"] != "light" {
		t.Fatalf("theme clobbered: %v", doc["theme"])
	}
	if doc["defaultProvider"] != "hive-ai" {
		t.Fatalf("defaultProvider clobbered: %v", doc["defaultProvider"])
	}
	block := doc["compaction"].(map[string]interface{})
	// A stricter (larger) reserve must not be LOWERED to the derived value.
	if got, _ := numericField(block["reserveTokens"]); got != 100000 {
		t.Fatalf("reserveTokens = %d, want the operator's stricter 100000 preserved", got)
	}
	if got, _ := numericField(block["keepRecentTokens"]); got != 50000 {
		t.Fatalf("keepRecentTokens = %d, want the operator's 50000 preserved", got)
	}
}

// TestEnsurePiSafeCompactionRaisesWeakReserve witnesses that a reserve too small to protect the
// session is raised to the derived one.
func TestEnsurePiSafeCompactionRaisesWeakReserve(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	seed := `{"compaction": {"reserveTokens": 128}}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatalf("seed settings.json: %v", err)
	}
	budget := PiSafeContextBudget(131072)

	_, modified, err := EnsurePiSafeCompaction(target, budget)
	if err != nil {
		t.Fatalf("EnsurePiSafeCompaction: %v", err)
	}
	if !modified {
		t.Fatal("expected a weak reserve to be raised, modified=false")
	}
	block := readJSONFile(t, target)["compaction"].(map[string]interface{})
	if got, _ := numericField(block["reserveTokens"]); got != budget.OutputReserve {
		t.Fatalf("reserveTokens = %d, want raised to %d", got, budget.OutputReserve)
	}
}

func firstFakModel(t *testing.T, doc map[string]interface{}) map[string]interface{} {
	t.Helper()
	provs, ok := doc["providers"].(map[string]interface{})
	if !ok {
		t.Fatalf("providers missing: %v", doc)
	}
	fak, ok := provs["fak"].(map[string]interface{})
	if !ok {
		t.Fatalf("fak provider missing: %v", provs)
	}
	models, ok := fak["models"].([]interface{})
	if !ok || len(models) == 0 {
		t.Fatalf("fak models missing: %v", fak["models"])
	}
	m, ok := models[0].(map[string]interface{})
	if !ok {
		t.Fatalf("models[0] not an object: %v", models[0])
	}
	return m
}

func readJSONFile(t *testing.T, path string) map[string]interface{} {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var doc map[string]interface{}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return doc
}
