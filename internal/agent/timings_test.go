package agent

// fak-test:runtime fast est=1s

import (
	"encoding/json"
	"testing"
)

func TestNewTimingsMatchesLlamaCppShape(t *testing.T) {
	tm := NewTimings(100, 40, 10, 0.030, 0.020)
	want := Timings{CacheN: 40, PromptN: 60, PromptMS: 30, PromptPerTokenMS: 0.5, PromptPerSecond: 2000,
		PredictedN: 10, PredictedMS: 20, PredictedPerTokenMS: 2, PredictedPerSecond: 500}
	const eps = 1e-9
	near := func(a, b float64) bool { return a-b < eps && b-a < eps }
	if tm.CacheN != want.CacheN || tm.PromptN != want.PromptN || tm.PredictedN != want.PredictedN ||
		!near(tm.PromptMS, want.PromptMS) || !near(tm.PromptPerTokenMS, want.PromptPerTokenMS) ||
		!near(tm.PromptPerSecond, want.PromptPerSecond) || !near(tm.PredictedMS, want.PredictedMS) ||
		!near(tm.PredictedPerTokenMS, want.PredictedPerTokenMS) || !near(tm.PredictedPerSecond, want.PredictedPerSecond) {
		t.Fatalf("NewTimings = %+v, want %+v", *tm, want)
	}
	b, err := json.Marshal(tm)
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]any
	if err := json.Unmarshal(b, &keys); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
		"predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"} {
		if _, ok := keys[k]; !ok {
			t.Fatalf("timings JSON missing llama.cpp key %q: %s", k, b)
		}
	}
}

func TestNewTimingsClampsAndAvoidsDivideByZero(t *testing.T) {
	tm := NewTimings(10, 50, 0, 0, 0)
	if tm.CacheN != 10 || tm.PromptN != 0 || tm.PromptPerSecond != 0 || tm.PredictedPerSecond != 0 {
		t.Fatalf("NewTimings clamp = %+v", *tm)
	}
}
