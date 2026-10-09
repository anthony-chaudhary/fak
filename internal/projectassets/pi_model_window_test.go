package projectassets

// pi_model_windows_test.go — the witness for PER-MODEL served windows.
//
// The pre-fix bug: the Pi `fak` provider catalog carried ONE served window (whatever the
// backend's /v1/models last advertised, on the operator host the Qwen 131072) and the safe
// budget derived from it (65536) was applied to EVERY model entry — so
// `deepseek-ai/DeepSeek-V4.1-Flash`, whose real window is ~1M, advertised contextWindow
// 65536. These tests prove the resolution is per-model, not flat, and that DeepSeek's ~1M
// served window is quality-capped once by the harnesskit envelope.

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// TestPiModelServedWindowDeepSeekPinned is the objective witness: DeepSeek V4.1 Flash
// resolves to its own ~1M served window, and its context window is the quality cap.
func TestPiModelServedWindowDeepSeekPinned(t *testing.T) {
	for _, id := range []string{
		"deepseek-ai/DeepSeek-V4.1-Flash",
		"DeepSeek-V4.1-Flash",
		"deepseek-v41-flash",
	} {
		got := PiModelServedWindow(id, 0)
		if got != PiDeepSeekFlashServedWindow {
			t.Errorf("PiModelServedWindow(%q) = %d, want %d", id, got, PiDeepSeekFlashServedWindow)
		}
		budget := PiModelContextBudget(id, 0)
		if budget.ResidentTarget != harnesskit.QualityCapTokens || !budget.Envelope.QualityCapped {
			t.Errorf("PiModelContextBudget(%q).ResidentTarget = %d, want quality cap %d",
				id, budget.ResidentTarget, harnesskit.QualityCapTokens)
		}
	}
}

// TestPiModelServedWindowPerModel proves the resolution is NOT flat: a DeepSeek id and a
// Qwen id in the same catalog get different windows, and neither inherits the other's.
func TestPiModelServedWindowPerModel(t *testing.T) {
	deepseek := PiModelServedWindow("deepseek-ai/DeepSeek-V4.1-Flash", 0)
	qwen := PiModelServedWindow("Qwen3.8-27B-UD-Q2_K_XL", 0)
	if deepseek == qwen {
		t.Fatalf("DeepSeek window %d equals Qwen window %d; resolution is still flat", deepseek, qwen)
	}
	if qwen != DefaultPiServedWindow {
		t.Errorf("Qwen window = %d, want %d", qwen, DefaultPiServedWindow)
	}
	// The pre-fix value the operator observed on DeepSeek: half the Qwen window.
	if deepseek == DefaultPiServedWindow/2 {
		t.Fatalf("DeepSeek window is still the Qwen-derived %d", DefaultPiServedWindow/2)
	}
}

// TestPiModelServedWindowUnknownFallsThrough keeps the historical behaviour for an id fak
// has never heard of: the caller's explicit window, else the doctrine default.
func TestPiModelServedWindowUnknownFallsThrough(t *testing.T) {
	if got := PiModelServedWindow("some-unknown-model-xyz", 0); got != DefaultPiServedWindow {
		t.Errorf("unknown model with no override = %d, want %d", got, DefaultPiServedWindow)
	}
	if got := PiModelServedWindow("some-unknown-model-xyz", 65536); got != 65536 {
		t.Errorf("unknown model with override = %d, want 65536", got)
	}
}

// TestPiModelServedWindowExplicitSmallerWins: an operator who passes a --window SMALLER than
// a known model's registry window is making a deliberate conservative pin and is honored.
func TestPiModelServedWindowExplicitSmallerWins(t *testing.T) {
	if got := PiModelServedWindow("deepseek-ai/DeepSeek-V4.1-Flash", 65536); got != 65536 {
		t.Errorf("explicit smaller window = %d, want 65536 (operator pin honored)", got)
	}
}

// TestGeneratePiConfigForWindowDeepSeekIsQualityCapped is the config-level witness: the
// models.json generator writes the quality cap for DeepSeek V4.1 Flash, not 65536.
func TestGeneratePiConfigForWindowDeepSeekIsQualityCapped(t *testing.T) {
	out, err := GeneratePiConfigForWindow("http://127.0.0.1:8080/v1", "deepseek-ai/DeepSeek-V4.1-Flash", 0)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	var cfg map[string]interface{}
	if err := json.Unmarshal(out, &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	providers := cfg["providers"].(map[string]interface{})
	models := providers["fak"].(map[string]interface{})["models"].([]interface{})
	if len(models) != 1 {
		t.Fatalf("models = %d, want 1", len(models))
	}
	entry := models[0].(map[string]interface{})
	want := PiModelContextBudget("deepseek-ai/DeepSeek-V4.1-Flash", 0)
	if cw, _ := numericField(entry["contextWindow"]); cw != harnesskit.QualityCapTokens {
		t.Fatalf("contextWindow = %d, want %d", cw, harnesskit.QualityCapTokens)
	}
	if mt, _ := numericField(entry["maxTokens"]); mt != want.MaxOutputTokens {
		t.Fatalf("maxTokens = %d, want %d (the envelope output budget)", mt, want.MaxOutputTokens)
	}
}

// TestEnsurePiProviderConfigRepairsQwenLeakDeepSeek is the repair witness: a catalog already
// written by the buggy flat-window path (DeepSeek stuck at the Qwen-derived 65536) is raised
// to the DeepSeek target on the next write, while the Qwen entry keeps its own value.
func TestEnsurePiProviderConfigRepairsQwenLeakDeepSeek(t *testing.T) {
	tmp := t.TempDir()
	target := tmp + "/models.json"
	// Seed exactly the shape the live operator host carries: one flat budget applied to
	// every model.
	seed := `{
  "providers": {
    "fak": {
      "api": "openai-completions",
      "apiKey": "fak",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "models": [
        {"id": "Qwen3.8-27B-UD-Q2_K_XL", "contextWindow": 65536, "maxTokens": 8192},
        {"id": "deepseek-ai/DeepSeek-V4.1-Flash", "contextWindow": 65536, "maxTokens": 8192}
      ]
    }
  }
}`
	if err := os.WriteFile(target, []byte(seed), 0644); err != nil {
		t.Fatal(err)
	}

	path, modified, err := EnsurePiProviderConfigForWindow(target, "http://127.0.0.1:8080/v1", "deepseek-ai/DeepSeek-V4.1-Flash", 0)
	if err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if !modified {
		t.Fatal("modified = false, want true (the DeepSeek entry must be repaired)")
	}
	raw := readJSONFile(t, path)
	models := raw["providers"].(map[string]interface{})["fak"].(map[string]interface{})["models"].([]interface{})
	got := map[string]int{}
	for _, m := range models {
		e := m.(map[string]interface{})
		id, _ := e["id"].(string)
		cw, _ := numericField(e["contextWindow"])
		got[id] = cw
	}
	if got["deepseek-ai/DeepSeek-V4.1-Flash"] != harnesskit.QualityCapTokens {
		t.Errorf("DeepSeek contextWindow = %d, want %d (repaired)", got["deepseek-ai/DeepSeek-V4.1-Flash"], harnesskit.QualityCapTokens)
	}
	if got["Qwen3.8-27B-UD-Q2_K_XL"] != 65536 {
		t.Errorf("Qwen contextWindow = %d, want the seeded 65536 (untouched by DeepSeek's repair)",
			got["Qwen3.8-27B-UD-Q2_K_XL"])
	}
}

// TestEnsurePiProviderConfigIdempotentPerModel: a second write with no change reports
// modified=false, so fak pi does not churn the file every launch.
func TestEnsurePiProviderConfigIdempotentPerModel(t *testing.T) {
	tmp := t.TempDir()
	target := tmp + "/models.json"
	if _, _, err := EnsurePiProviderConfigForWindow(target, "http://127.0.0.1:8080/v1", "deepseek-ai/DeepSeek-V4.1-Flash", 0); err != nil {
		t.Fatal(err)
	}
	_, modified, err := EnsurePiProviderConfigForWindow(target, "http://127.0.0.1:8080/v1", "deepseek-ai/DeepSeek-V4.1-Flash", 0)
	if err != nil {
		t.Fatal(err)
	}
	if modified {
		t.Fatal("modified = true on an unchanged second write, want false (idempotent)")
	}
}
