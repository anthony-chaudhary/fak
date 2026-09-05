package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

func intent(h CacheHorizon, privacy CacheResidency) *CacheIntent {
	return &CacheIntent{Version: CacheIntentVersion, Enabled: true, Horizon: h, AffinityID: "tenant/session", PrivacyCeiling: privacy, Preference: CachePreferenceAutomatic}
}

func TestCacheHintNegotiationCompatibility(t *testing.T) {
	tests := []struct {
		name     string
		provider Provider
		model    string
		in       *CacheIntent
		status   CacheHintStatus
		wantErr  bool
		reason   string
	}{
		{"openai supported", ProviderOpenAIResponses, "gpt-5.6", intent(CacheHorizonTwentyFourHours, CacheResidencyExtended), CacheHintSupported, false, ""},
		{"openai gpt-6-astra supported", ProviderOpenAIResponses, "gpt-6-astra", intent(CacheHorizonTwentyFourHours, CacheResidencyExtended), CacheHintSupported, false, ""},
		{"openai gpt-7 supported", ProviderOpenAIResponses, "gpt-7", intent(CacheHorizonTwentyFourHours, CacheResidencyExtended), CacheHintSupported, false, ""},
		{"openai o5 supported", ProviderOpenAIResponses, "o5", intent(CacheHorizonTwentyFourHours, CacheResidencyExtended), CacheHintSupported, false, ""},
		{"anthropic claude-4-sonnet", ProviderAnthropic, "claude-4-sonnet", func() *CacheIntent {
			x := intent(CacheHorizonOneHour, CacheResidencyMemory)
			x.Preference = CachePreferenceExplicit
			return x
		}(), CacheHintSupported, false, ""},
		{"anthropic claude-4.5", ProviderAnthropic, "claude-4.5", func() *CacheIntent {
			x := intent(CacheHorizonOneHour, CacheResidencyMemory)
			x.Preference = CachePreferenceExplicit
			return x
		}(), CacheHintSupported, false, ""},
		{"anthropic claude-5", ProviderAnthropic, "claude-5", func() *CacheIntent {
			x := intent(CacheHorizonOneHour, CacheResidencyMemory)
			x.Preference = CachePreferenceExplicit
			return x
		}(), CacheHintSupported, false, ""},
		{"openai privacy fail closed", ProviderOpenAIResponses, "gpt-5.6", intent(CacheHorizonTwentyFourHours, CacheResidencyMemory), CacheHintRejected, true, "privacy"},
		{"openai unsupported fail closed", ProviderOpenAIResponses, "gpt-4.1", intent(CacheHorizonTwentyFourHours, CacheResidencyExtended), CacheHintRejected, true, "does not support"},
		{"openai advisory downgrade", ProviderOpenAIResponses, "gpt-4.1", func() *CacheIntent {
			x := intent(CacheHorizonTwentyFourHours, CacheResidencyExtended)
			x.Advisory = true
			return x
		}(), CacheHintDowngraded, false, "does not support"},
		{"openai provider default", ProviderOpenAIResponses, "gpt-5.6", intent(CacheHorizonProviderDefault, CacheResidencyMemory), CacheHintProviderDefaulted, false, ""},
		{"anthropic one hour", ProviderAnthropic, "claude-sonnet-4-5", func() *CacheIntent {
			x := intent(CacheHorizonOneHour, CacheResidencyMemory)
			x.Preference = CachePreferenceExplicit
			return x
		}(), CacheHintSupported, false, ""},
		{"anthropic old model rejected", ProviderAnthropic, "claude-2.1", func() *CacheIntent {
			x := intent(CacheHorizonOneHour, CacheResidencyMemory)
			x.Preference = CachePreferenceExplicit
			return x
		}(), CacheHintRejected, true, "does not support"},
		{"gemini advisory", ProviderGemini, "gemini-2.5-pro", func() *CacheIntent {
			x := intent(CacheHorizonFiveMinutes, CacheResidencyMemory)
			x.Advisory = true
			return x
		}(), CacheHintDowngraded, false, "no explicit"},
		{"version rejected", ProviderOpenAI, "gpt-5.6", &CacheIntent{Version: "future", Enabled: true, PrivacyCeiling: CacheResidencyMemory}, CacheHintRejected, true, "version"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := negotiateCacheIntent(tt.provider, tt.model, tt.in)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err=%v", err)
			}
			if got.Status != tt.status {
				t.Fatalf("status=%s want %s", got.Status, tt.status)
			}
			if tt.reason != "" && !strings.Contains(got.Reason, tt.reason) {
				t.Fatalf("reason=%q", got.Reason)
			}
		})
	}
}

func TestOpenAIExactOutgoingCacheHintAndUnknownField(t *testing.T) {
	in := intent(CacheHorizonTwentyFourHours, CacheResidencyExtended)
	got, err := negotiateCacheIntent(ProviderOpenAIResponses, "gpt-5.6", in)
	if err != nil {
		t.Fatal(err)
	}
	body, err := applyCacheHintToJSON([]byte(`{"model":"gpt-5.6","future_field":{"x":1}}`), got)
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	if err = json.Unmarshal(body, &obj); err != nil {
		t.Fatal(err)
	}
	if obj["prompt_cache_key"] != "tenant/session" || obj["prompt_cache_retention"] != "24h" {
		t.Fatalf("wire=%s", body)
	}
	if obj["future_field"].(map[string]any)["x"] != float64(1) {
		t.Fatalf("unknown field lost: %s", body)
	}
}

func TestAnthropicExactOutgoingTTLAndLegacyBreakpoint(t *testing.T) {
	got, err := negotiateCacheIntent(ProviderAnthropic, "claude-sonnet-4-5", func() *CacheIntent {
		x := intent(CacheHorizonOneHour, CacheResidencyMemory)
		x.Preference = CachePreferenceExplicit
		return x
	}())
	if err != nil {
		t.Fatal(err)
	}
	body, err := applyCacheHintToJSON([]byte(`{"system":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral"},"future":"kept"}],"messages":[]}`), got)
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	json.Unmarshal(body, &obj)
	block := obj["system"].([]any)[0].(map[string]any)
	cc := block["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" || cc["ttl"] != "1h" || block["future"] != "kept" {
		t.Fatalf("wire=%s", body)
	}
}

func TestAnthropicBreakpointLimitAndMixedTTLOrder(t *testing.T) {
	five := map[string]any{"cache_control": map[string]any{"type": "ephemeral", "ttl": "5m"}}
	hour := map[string]any{"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"}}
	if err := validateAnthropicTTLOrder(map[string]any{"system": []any{hour, five}}); err != nil {
		t.Fatalf("valid order: %v", err)
	}
	if err := validateAnthropicTTLOrder(map[string]any{"system": []any{five, hour}}); err == nil || !strings.Contains(err.Error(), "1h before 5m") {
		t.Fatalf("mixed order err=%v", err)
	}
	blocks := []any{hour, hour, hour, hour, hour}
	if err := validateAnthropicTTLOrder(map[string]any{"system": blocks}); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("limit err=%v", err)
	}
}

func TestNilCacheIntentPreservesWire(t *testing.T) {
	body := []byte(`{"unknown":true}`)
	got, err := applyCacheHintToJSON(body, CacheHintResult{})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Fatalf("got %s", got)
	}
}

func TestUnsupportedProviderCacheHintsRemainWireNeutral(t *testing.T) {
	xai := CacheHintResult{Provider: ProviderXAI, Requested: intent(CacheHorizonFiveMinutes, CacheResidencyMemory), Status: CacheHintSupported, Emitted: map[string]any{"affinity": "tenant/session"}}
	body, err := applyCacheHintToJSON([]byte(`{"model":"grok"}`), xai)
	if err != nil {
		t.Fatal(err)
	}
	var xaiBody map[string]any
	if err := json.Unmarshal(body, &xaiBody); err != nil {
		t.Fatal(err)
	}
	if _, leaked := xaiBody["prompt_cache_key"]; leaked {
		t.Fatalf("xAI request gained an unsupported cache field: %s", body)
	}

	gemini := CacheHintResult{Provider: ProviderGemini, Requested: intent(CacheHorizonFiveMinutes, CacheResidencyMemory), Status: CacheHintDowngraded, Emitted: map[string]any{"affinity": "must-not-leak"}}
	body, err = applyCacheHintToJSON([]byte(`{"model":"gemini","future":true}`), gemini)
	if err != nil {
		t.Fatal(err)
	}
	var geminiBody map[string]any
	if err := json.Unmarshal(body, &geminiBody); err != nil {
		t.Fatal(err)
	}
	if _, leaked := geminiBody["prompt_cache_key"]; leaked || geminiBody["future"] != true {
		t.Fatalf("Gemini advisory mutated unsupported cache fields: %s", body)
	}
}

func TestFutureModelsCacheRetentionAndTTL(t *testing.T) {
	openAITests := []struct {
		model string
		want  bool
	}{
		{"gpt-5", true},
		{"gpt-5.6", true},
		{"gpt-6", true},
		{"gpt-6-astra", true},
		{"gpt-7", true},
		{"gpt-8", true},
		{"gpt-9", true},
		{"o1", true},
		{"o1-mini", true},
		{"o3", true},
		{"o3-mini", true},
		{"o4", true},
		{"o5", true},
		{"o5-mini", true},
		{"o6", true},
		{"astra", true},
		{"astra-2", true},
		{"openai/gpt-7", true},
		{"openai/o5", true},
		{"gpt-4.1", false},
		{"gpt-4o", false},
		{"gpt-3.5-turbo", false},
	}
	for _, tc := range openAITests {
		if got := openAISupportsExtendedRetention(tc.model); got != tc.want {
			t.Errorf("openAISupportsExtendedRetention(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}

	anthropicTests := []struct {
		model string
		want  bool
	}{
		{"claude-3-sonnet", true},
		{"claude-3.5-sonnet", true},
		{"claude-3.7-sonnet", true},
		{"claude-4-sonnet", true},
		{"claude-4.5", true},
		{"claude-5", true},
		{"claude-sonnet-4-5", true},
		{"claude-opus-4", true},
		{"claude-opus-5", true},
		{"claude-haiku-4", true},
		{"claude-haiku-5", true},
		{"anthropic/claude-4-sonnet", true},
		{"claude-2.1", false},
		{"claude-1.3", false},
	}
	for _, tc := range anthropicTests {
		if got := anthropicSupportsTTL(tc.model); got != tc.want {
			t.Errorf("anthropicSupportsTTL(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}
