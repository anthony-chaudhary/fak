package harnesskit

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=5ms
func TestSamplingParamsQwen38WireProfile(t *testing.T) {
	const want = `{"min_p":0,"presence_penalty":1.5,"temperature":0.7,"top_k":20,"top_p":0.8}`
	for id, profiled := range map[string]bool{
		"Qwen3.8-27B-UD-Q2_K_XL":            true,
		"fak-router/Qwen3.8-27B-UD-Q2_K_XL": true,
		"qwen38:27b":                        true,
		"deepseek-ai/DeepSeek-V4.1-Flash":   false,
		"Qwen2.5-Coder-7B":                  false,
		"":                                  false,
	} {
		sp := SamplingParams(id)
		if !profiled {
			if sp != nil {
				t.Errorf("SamplingParams(%q) = %v, want nil", id, sp)
			}
			continue
		}
		raw, err := json.Marshal(sp)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != want {
			t.Errorf("SamplingParams(%q) wire = %s, want %s", id, raw, want)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestSamplingParamsReturnsFreshMap(t *testing.T) {
	first := SamplingParams("Qwen3.8-27B-UD-Q2_K_XL")
	first["temperature"] = 0.0
	if got := SamplingParams("Qwen3.8-27B-UD-Q2_K_XL")["temperature"]; got != 0.7 {
		t.Fatalf("temperature after caller mutation = %v, want 0.7", got)
	}
}
