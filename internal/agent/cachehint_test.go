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
