package canon

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

func TestAdaptTokenUsageFixturesReconcileAndPreserveNative(t *testing.T) {
	tests := []struct {
		provider      string
		want          map[TokenClass]int64
		input, output int64
		unknown       string
	}{
		{"openai", map[TokenClass]int64{inputFreshClass: 45, inputReadClass: 40, inputWriteClass: 15, outputVisibleClass: 20, outputReasoningClass: 10}, 100, 30, "future_modality_tokens"},
		{"anthropic", map[TokenClass]int64{inputFreshClass: 20, inputReadClass: 50, inputWriteClass: 5, outputVisibleClass: 11}, 75, 11, "future_cache_tier"},
		{"gemini", map[TokenClass]int64{inputFreshClass: 45, inputReadClass: 60, outputVisibleClass: 20, outputReasoningClass: 10}, 105, 30, "future_cache_region"},
		{"local", map[TokenClass]int64{inputFreshClass: 17, outputVisibleClass: 9}, 17, 9, "native_extra"},
	}
	for _, tt := range tests {
		t.Run(tt.provider, func(t *testing.T) {
			raw, err := os.ReadFile("testdata/token_usage/" + tt.provider + ".json")
			if err != nil {
				t.Fatal(err)
			}
			got, err := AdaptTokenUsage(tt.provider, raw)
			if err != nil {
				t.Fatal(err)
			}
			if got.Schema != TokenUsageSchema || got.Input != tt.input || got.Output != tt.output || got.Total != tt.input+tt.output {
				t.Fatalf("bad totals/schema: %+v", got)
			}
			if len(got.Classes) != len(tt.want) {
				t.Fatalf("classes=%v want %v", got.Classes, tt.want)
			}
			var sum int64
			for class, want := range tt.want {
				if got.Classes[class] != want {
					t.Errorf("%s=%d want %d", class, got.Classes[class], want)
				}
				sum += got.Classes[class]
			}
			if sum != got.Total {
				t.Fatalf("double-counted classes: sum=%d total=%d", sum, got.Total)
			}
			if !bytes.Contains(got.Raw, []byte(tt.unknown)) {
				t.Fatalf("raw provenance lost unknown field %q: %s", tt.unknown, got.Raw)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var round TokenUsage
			if err := json.Unmarshal(encoded, &round); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(round.Raw, got.Raw) {
				t.Fatalf("raw provenance changed on round trip: %s", round.Raw)
			}
		})
	}
}

func TestAdaptTokenUsageRejectsOverlappingOpenAIDetail(t *testing.T) {
	_, err := AdaptTokenUsage("openai", json.RawMessage(`{"input_tokens":2,"output_tokens":1,"input_tokens_details":{"cached_tokens":3}}`))
	if err == nil {
		t.Fatal("expected inclusive detail overflow refusal")
	}
}

func TestAdaptTokenUsageRejectsOverlappingOpenAICacheDetails(t *testing.T) {
	_, err := AdaptTokenUsage("openai", json.RawMessage(`{"input_tokens":2,"output_tokens":1,"input_tokens_details":{"cache_write_tokens":2,"cached_tokens":1}}`))
	if err == nil {
		t.Fatal("expected inclusive cache detail overflow refusal")
	}
}

func TestAdaptTokenUsageRejectsGeminiCachedContentAbovePrompt(t *testing.T) {
	_, err := AdaptTokenUsage("gemini", json.RawMessage(`{"promptTokenCount":2,"cachedContentTokenCount":3,"totalTokenCount":2}`))
	if err == nil {
		t.Fatal("expected inclusive cached-content overflow refusal")
	}
}

func TestAdaptTokenUsageRejectsInconsistentGeminiTotal(t *testing.T) {
	_, err := AdaptTokenUsage("gemini", json.RawMessage(`{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":4}`))
	if err == nil {
		t.Fatal("expected generated component total mismatch refusal")
	}
}

func TestAdaptTokenUsageRejectsUnknownProviderWithoutLosingContract(t *testing.T) {
	_, err := AdaptTokenUsage("mystery", json.RawMessage(`{"input_tokens":1}`))
	if err == nil {
		t.Fatal("expected unsupported provider refusal")
	}
}

func TestAdaptGeminiUsageDeclaresUnavailableCacheWriteClass(t *testing.T) {
	classes, _, _, err := adaptGeminiUsage([]byte(`{"promptTokenCount":2,"cachedContentTokenCount":1,"totalTokenCount":2}`))
	if err != nil {
		t.Fatal(err)
	}
	if value, present := classes[inputWriteClass]; !present || value != 0 {
		t.Fatalf("cache-write class=(%d,%v), want explicit unavailable zero", value, present)
	}
}
