package capabilitymatrix

import (
	"testing"
)

func TestCapabilityMatrix_LookupKnownAndFutureModels(t *testing.T) {
	ResetRegistry()

	tests := []struct {
		model                 string
		wantThinking          bool
		wantExtendedRetention bool
		wantAnthropicTTL      bool
		wantResponsesAPI      bool
		wantStrictSchema      bool
	}{
		// 1. Required known and future models from Issue #11507
		{
			model:                 "gpt-6.5",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "gpt-7",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:            "claude-4.5",
			wantThinking:     true,
			wantAnthropicTTL: true,
			wantStrictSchema: true,
		},
		{
			model:                 "o3",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "astra",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		// 2. Additional future and known OpenAI models
		{
			model:                 "gpt-8",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "gpt-6-astra",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "gpt-5.6",
			wantThinking:          false,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "gpt-4o",
			wantThinking:          false,
			wantExtendedRetention: false,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "gpt-4.1",
			wantThinking:          false,
			wantExtendedRetention: false,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "o1",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "o5-mini",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		{
			model:                 "openai/gpt-7",
			wantThinking:          true,
			wantExtendedRetention: true,
			wantResponsesAPI:      true,
			wantStrictSchema:      true,
		},
		// 3. Anthropic models
		{
			model:            "claude-3-7-sonnet",
			wantThinking:     true,
			wantAnthropicTTL: true,
			wantStrictSchema: true,
		},
		{
			model:            "claude-3.7-sonnet-20250219",
			wantThinking:     true,
			wantAnthropicTTL: true,
			wantStrictSchema: true,
		},
		{
			model:            "claude-4-sonnet",
			wantThinking:     true,
			wantAnthropicTTL: true,
			wantStrictSchema: true,
		},
		{
			model:            "claude-5",
			wantThinking:     true,
			wantAnthropicTTL: true,
			wantStrictSchema: true,
		},
		{
			model:            "claude-3-5-sonnet-20241022",
			wantThinking:     false,
			wantAnthropicTTL: true,
			wantStrictSchema: true,
		},
		{
			model:            "claude-2.1",
			wantThinking:     false,
			wantAnthropicTTL: false,
			wantStrictSchema: false,
		},
		// 4. Gemini models
		{
			model:            "gemini-2.0-flash-thinking",
			wantThinking:     true,
			wantStrictSchema: true,
		},
		{
			model:            "gemini-2.5-pro",
			wantThinking:     true,
			wantStrictSchema: true,
		},
		{
			model:            "gemini-3.8-flash",
			wantThinking:     true,
			wantStrictSchema: true,
		},
		{
			model:            "gemini-4.0",
			wantThinking:     true,
			wantStrictSchema: true,
		},
		// 5. Generic models without thinking
		{
			model:                 "llama-3.3-70b",
			wantThinking:          false,
			wantExtendedRetention: false,
		},
		{
			model:                 "mistral-large",
			wantThinking:          false,
			wantExtendedRetention: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			caps := Lookup(tc.model)
			if caps.Thinking != tc.wantThinking {
				t.Errorf("Lookup(%q).Thinking = %v, want %v", tc.model, caps.Thinking, tc.wantThinking)
			}
			if caps.ExtendedRetention != tc.wantExtendedRetention {
				t.Errorf("Lookup(%q).ExtendedRetention = %v, want %v", tc.model, caps.ExtendedRetention, tc.wantExtendedRetention)
			}
			if caps.AnthropicTTL != tc.wantAnthropicTTL {
				t.Errorf("Lookup(%q).AnthropicTTL = %v, want %v", tc.model, caps.AnthropicTTL, tc.wantAnthropicTTL)
			}
			if tc.wantResponsesAPI && !caps.ResponsesAPI {
				t.Errorf("Lookup(%q).ResponsesAPI = %v, want true", tc.model, caps.ResponsesAPI)
			}
			if tc.wantStrictSchema && !caps.StrictSchema {
				t.Errorf("Lookup(%q).StrictSchema = %v, want true", tc.model, caps.StrictSchema)
			}
		})
	}
}

func TestCapabilityMatrix_EnvironmentOverrides(t *testing.T) {
	ResetRegistry()
	defer ResetRegistry()

	// Initial check without override
	if Lookup("gpt-4o").Thinking {
		t.Fatal("gpt-4o should not support thinking by default")
	}
	if Lookup("custom-super-model").Thinking {
		t.Fatal("custom-super-model should not support thinking by default")
	}

	// Apply JSON overrides via FAK_MODEL_CAPABILITIES
	overrideJSON := `{
		"gpt-4o": {
			"supports_thinking": true
		},
		"custom-super-model": {
			"supports_thinking": true,
			"supports_extended_retention": true,
			"context_window": 500000
		},
		"gpt-7": {
			"supports_thinking": false
		}
	}`
	t.Setenv("FAK_MODEL_CAPABILITIES", overrideJSON)

	// Verify gpt-4o now supports thinking
	caps4o := Lookup("gpt-4o")
	if !caps4o.Thinking {
		t.Errorf("expected gpt-4o thinking to be overridden to true")
	}
	// gpt-4o should retain its base properties like StrictSchema
	if !caps4o.StrictSchema {
		t.Errorf("gpt-4o lost StrictSchema after override")
	}

	// Verify custom model capabilities
	capsCustom := Lookup("custom-super-model")
	if !capsCustom.Thinking || !capsCustom.ExtendedRetention {
		t.Errorf("custom-super-model capabilities not reflected: %+v", capsCustom)
	}
	if capsCustom.ContextWindow != 500000 {
		t.Errorf("custom-super-model context window got %d, want 500000", capsCustom.ContextWindow)
	}

	// Verify gpt-7 thinking disabled via override
	caps7 := Lookup("gpt-7")
	if caps7.Thinking {
		t.Errorf("expected gpt-7 thinking to be overridden to false")
	}
	if !caps7.ExtendedRetention {
		t.Errorf("gpt-7 should still support extended retention")
	}
}

func TestCapabilityMatrix_NormalizeCodexModelSlug(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"gpt-6-astra", GPT6AstraModel},
		{"gpt 6 astra", GPT6AstraModel},
		{"GPT 6 ASTRA", GPT6AstraModel},
		{"gpt6astra", GPT6AstraModel},
		{"gpt-6", GPT6AstraModel},
		{"gpt6", GPT6AstraModel},
		{"astra", GPT6AstraModel},
		{"ASTRA", GPT6AstraModel},
		{"gpt-6 astra", GPT6AstraModel},
		{"gpt 6-astra", GPT6AstraModel},
		{"gpt6-astra", GPT6AstraModel},
		{"astra-gpt-6", GPT6AstraModel},
		{"astra gpt 6", GPT6AstraModel},
		{"astra gpt-6", GPT6AstraModel},
		{"astra-gpt6", GPT6AstraModel},
		{"astragpt6", GPT6AstraModel},
		{"openai/gpt-6-astra", GPT6AstraModel},
		{"openai/gpt-6", GPT6AstraModel},
		{"openai/astra", GPT6AstraModel},
		{"openai/gpt-6 astra", GPT6AstraModel},
		{"openai/astra-gpt-6", GPT6AstraModel},
		{"openai/astra gpt 6", GPT6AstraModel},
		{"openai/astra-gpt6", GPT6AstraModel},
		{"  gpt 6 astra  ", GPT6AstraModel},
		{"astra-2", "astra-2"},
		{"gpt-5.6-sol", "gpt-5.6-sol"},
		{"custom-model", "custom-model"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := NormalizeCodexModelSlug(tc.in); got != tc.want {
			t.Errorf("NormalizeCodexModelSlug(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestCapabilityMatrix_LookupProfile(t *testing.T) {
	prof := LookupProfile("gpt-6")
	if prof.Canonical != GPT6AstraModel {
		t.Errorf("LookupProfile('gpt-6').Canonical = %q, want %q", prof.Canonical, GPT6AstraModel)
	}
	if !prof.Capabilities.Thinking {
		t.Errorf("LookupProfile('gpt-6').Capabilities.Thinking = false, want true")
	}
}

func TestCapabilityMatrix_LookupProfile_Astra(t *testing.T) {
	astraModels := []string{"gpt-6-astra", "astra", "gpt-6", "astra-gpt-6", "astra gpt 6"}
	for _, m := range astraModels {
		t.Run(m, func(t *testing.T) {
			prof := LookupProfile(m)
			if prof.Canonical != GPT6AstraModel {
				t.Errorf("LookupProfile(%q).Canonical = %q, want %q", m, prof.Canonical, GPT6AstraModel)
			}
			if prof.Capabilities.ContextWindow != 1000000 {
				t.Errorf("LookupProfile(%q).Capabilities.ContextWindow = %d, want 1000000", m, prof.Capabilities.ContextWindow)
			}
			if prof.Capabilities.MaxOutputTokens != 131072 {
				t.Errorf("LookupProfile(%q).Capabilities.MaxOutputTokens = %d, want 131072", m, prof.Capabilities.MaxOutputTokens)
			}
			if !prof.Capabilities.Thinking {
				t.Errorf("LookupProfile(%q).Capabilities.Thinking = false, want true", m)
			}
		})
	}

	t.Run("astra-2", func(t *testing.T) {
		prof := LookupProfile("astra-2")
		if prof.Canonical == GPT6AstraModel {
			t.Errorf("LookupProfile('astra-2').Canonical = %q, should not be %q", prof.Canonical, GPT6AstraModel)
		}
		if got := NormalizeCodexModelSlug("astra-2"); got == GPT6AstraModel {
			t.Errorf("NormalizeCodexModelSlug('astra-2') = %q, should not be %q", got, GPT6AstraModel)
		}
	})
}
