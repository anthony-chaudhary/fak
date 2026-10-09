package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

func piSamplingOf(t *testing.T, rendered []byte) map[string]map[string]interface{} {
	t.Helper()
	var doc struct {
		Providers map[string]struct {
			Models []map[string]interface{} `json:"models"`
		} `json:"providers"`
	}
	if err := json.Unmarshal(rendered, &doc); err != nil {
		t.Fatalf("rendered config is not JSON: %v", err)
	}
	out := map[string]map[string]interface{}{}
	for _, m := range doc.Providers["fak"].Models {
		sp, _ := m["samplingParams"].(map[string]interface{})
		out[m["id"].(string)] = sp
	}
	return out
}

func canonicalPiSampling(t *testing.T, id string) map[string]interface{} {
	t.Helper()
	raw, err := json.Marshal(projectassets.PiSamplingParams(id))
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// fak-test:runtime fast est=50ms
func TestPiConfigFromRouterOwnsQwenSampling(t *testing.T) {
	const qwen = "Qwen3.8-27B-UD-Q2_K_XL"
	const other = "deepseek-ai/DeepSeek-V4.1-Flash"
	rows := []piRouterRow{
		{ID: qwen, OwnedBy: "halo", Window: 131072},
		{ID: other, OwnedBy: "halo", Window: 131072},
		{ID: "Qwen3.8-27B-Q4_K_M", OwnedBy: "halo", Window: 131072},
	}
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")
	hand := `{
  "providers": {
    "fak": {
      "baseUrl": "http://127.0.0.1:18101/v1",
      "models": [
        {"id": "Qwen3.8-27B-UD-Q2_K_XL", "contextWindow": 65536, "maxTokens": 4096, "samplingParams": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 1}},
        {"id": "deepseek-ai/DeepSeek-V4.1-Flash", "contextWindow": 65536, "maxTokens": 8192}
      ]
    }
  }
}
`
	if err := os.WriteFile(modelsPath, []byte(hand), 0o644); err != nil {
		t.Fatal(err)
	}
	plan, err := buildPiRouterPlan(rows, "http://127.0.0.1:18101/v1", modelsPath, filepath.Join(dir, "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	got := piSamplingOf(t, plan.Rendered)
	if want := canonicalPiSampling(t, qwen); !reflect.DeepEqual(got[qwen], want) {
		t.Fatalf("existing %s samplingParams = %v, want the fak-owned profile %v", qwen, got[qwen], want)
	}
	if want := canonicalPiSampling(t, "Qwen3.8-27B-Q4_K_M"); !reflect.DeepEqual(got["Qwen3.8-27B-Q4_K_M"], want) {
		t.Fatalf("added Qwen entry samplingParams = %v, want %v", got["Qwen3.8-27B-Q4_K_M"], want)
	}
	if got[other] != nil {
		t.Fatalf("%s gained samplingParams %v; only profiled models are owned", other, got[other])
	}
	var sampled bool
	for _, c := range plan.Changed {
		if c.ID == qwen && c.Sampling {
			sampled = true
		}
	}
	if !sampled {
		t.Fatalf("Changed = %+v, want %s reported as a sampling change", plan.Changed, qwen)
	}

	if err := os.WriteFile(modelsPath, plan.Rendered, 0o644); err != nil {
		t.Fatal(err)
	}
	again, err := buildPiRouterPlan(rows, "http://127.0.0.1:18101/v1", modelsPath, filepath.Join(dir, "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	if again.configChanged() || len(again.Changed) != 0 {
		t.Fatalf("second render changed the file: %+v", again.Changed)
	}
}

// fak-test:runtime fast est=5ms
func TestPiSamplingParamsProfile(t *testing.T) {
	for id, want := range map[string]bool{
		"Qwen3.8-27B-UD-Q2_K_XL": true,
		"halo/qwen3.8-27b":       true,
		"qwen38:27b":             true,
		"deepseek-v4.1-flash":    false,
		"Qwen2.5-Coder-7B":       false,
	} {
		sp := projectassets.PiSamplingParams(id)
		if (sp != nil) != want {
			t.Errorf("PiSamplingParams(%q) = %v, want profile=%v", id, sp, want)
			continue
		}
		if want && (sp["temperature"] != 0.7 || sp["presence_penalty"] != 1.5) {
			t.Errorf("PiSamplingParams(%q) = %v, want temperature 0.7 and presence_penalty 1.5", id, sp)
		}
	}
}
