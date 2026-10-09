package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// TestPiSafeContextBudget pins the Pi mapping of the harnesskit envelope: the served window
// is used in full up to the quality cap, never halved, and the compaction numbers come from
// the envelope.
func TestPiSafeContextBudget(t *testing.T) {
	cases := []struct {
		name       string
		window     int
		wantTarget int
		wantViable bool
	}{
		{"128k slot is used in full", 131072, 131072, true},
		{"256k window is quality-capped", 262144, harnesskit.QualityCapTokens, true},
		{"1M window is quality-capped", 1048576, harnesskit.QualityCapTokens, true},
		{"32k window is not viable for the harness", 32768, 32768, false},
		{"unset window falls back to the default prior", 0, DefaultPiServedWindow, true},
		{"negative window falls back", -5, DefaultPiServedWindow, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := PiSafeContextBudget(tc.window)
			if b.ResidentTarget != tc.wantTarget {
				t.Fatalf("ResidentTarget = %d, want %d", b.ResidentTarget, tc.wantTarget)
			}
			if b.Viable != tc.wantViable {
				t.Fatalf("Viable = %t, want %t (reason %q)", b.Viable, tc.wantViable, b.Reason)
			}
			if !b.Viable && b.Reason != harnesskit.ReasonWindowTooSmallForHarness {
				t.Fatalf("Reason = %q, want %q", b.Reason, harnesskit.ReasonWindowTooSmallForHarness)
			}
			env := b.Envelope
			if b.ReserveTokens != env.ReserveTokens || b.KeepRecentTokens != env.KeepRecentTokens || b.MaxOutputTokens != env.OutputTokens {
				t.Fatalf("budget %+v does not mirror envelope %+v", b, env)
			}
			if b.ResidentTarget-b.ReserveTokens != env.CompactTrigger {
				t.Fatalf("Pi trigger %d != envelope trigger %d", b.ResidentTarget-b.ReserveTokens, env.CompactTrigger)
			}
			if b.Provenance != PiBudgetProvenance {
				t.Fatalf("Provenance = %q, want %q", b.Provenance, PiBudgetProvenance)
			}
		})
	}
}

// TestPiSafeContextBudgetNoStacking is the regression witness for the 131072 -> 65536 bug:
// re-deriving from a written contextWindow never shrinks it again.
func TestPiSafeContextBudgetNoStacking(t *testing.T) {
	for _, w := range []int{131072, 1048576} {
		first := PiSafeContextBudget(w)
		if again := PiSafeContextBudget(first.ResidentTarget); again.ResidentTarget != first.ResidentTarget {
			t.Fatalf("window %d: %d re-derived to %d (stacked)", w, first.ResidentTarget, again.ResidentTarget)
		}
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

// TestGeneratePiConfigAdvertisesEnvelopeWindow witnesses that the generated models.json carries
// the envelope window: a 1M served window is quality-capped, never advertised raw or halved.
func TestGeneratePiConfigAdvertisesEnvelopeWindow(t *testing.T) {
	const window = 1048576
	out, err := GeneratePiConfigForWindow("http://127.0.0.1:8080/v1", "local-1m-model", window)
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
	if cw != harnesskit.QualityCapTokens {
		t.Fatalf("contextWindow = %d, want the quality cap %d", cw, harnesskit.QualityCapTokens)
	}
	mt, ok := numericField(model["maxTokens"])
	if !ok || mt != PiSafeContextBudget(window).MaxOutputTokens {
		t.Fatalf("maxTokens = %v, want the envelope output budget", model["maxTokens"])
	}
	// The model must still be usable: id and provider wiring preserved.
	if model["id"] != "local-1m-model" {
		t.Fatalf("model id = %v, want local-1m-model", model["id"])
	}
}

// TestEnsurePiProviderConfigRepairsHalvedWindow witnesses the upgrade path: a models.json fak
// itself wrote with the retired halved contextWindow is corrected to the envelope window.
func TestEnsurePiProviderConfigRepairsHalvedWindow(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "models.json")
	raw := `{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [
        {"id": "qwen38:27b-q4", "name": "old", "contextWindow": 65536, "maxTokens": 16384,
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
		t.Fatal("expected the halved contextWindow to be repaired, modified=false")
	}

	after := readJSONFile(t, target)
	model := firstFakModel(t, after)
	cw, _ := numericField(model["contextWindow"])
	if cw != DefaultPiServedWindow {
		t.Fatalf("contextWindow = %d, want %d (the served slot, not half of it)", cw, DefaultPiServedWindow)
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
	if got, _ := numericField(block["reserveTokens"]); got != budget.ReserveTokens {
		t.Fatalf("reserveTokens = %d, want %d", got, budget.ReserveTokens)
	}
	if got, _ := numericField(block["keepRecentTokens"]); got != budget.KeepRecentTokens {
		t.Fatalf("keepRecentTokens = %d, want %d", got, budget.KeepRecentTokens)
	}
}

// TestEnsurePiSafeCompactionPreservesUserKeys witnesses the non-clobber invariant: unrelated
// settings survive, while the fak-owned reserve/keep are set to the envelope exactly.
func TestEnsurePiSafeCompactionPreservesUserKeys(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "settings.json")
	// A stale reserve/keep (e.g. from the retired halving) plus unrelated keys.
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
	if got, _ := numericField(block["reserveTokens"]); got != budget.ReserveTokens {
		t.Fatalf("reserveTokens = %d, want the envelope %d", got, budget.ReserveTokens)
	}
	if got, _ := numericField(block["keepRecentTokens"]); got != budget.KeepRecentTokens {
		t.Fatalf("keepRecentTokens = %d, want the envelope %d", got, budget.KeepRecentTokens)
	}
	if _, modified, err := EnsurePiSafeCompaction(target, budget); err != nil || modified {
		t.Fatalf("second write: modified=%t err=%v, want idempotent", modified, err)
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
	if got, _ := numericField(block["reserveTokens"]); got != budget.ReserveTokens {
		t.Fatalf("reserveTokens = %d, want raised to %d", got, budget.ReserveTokens)
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
