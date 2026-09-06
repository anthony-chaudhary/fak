package capabilitymatrix

import (
	"encoding/json"
	"testing"
)

var (
	benchCapsSink    Capabilities
	benchProfileSink ModelProfile
	benchSlugSink    string
	benchBytesSink   []byte
)

func BenchmarkLookup(b *testing.B) {
	cases := []struct {
		model            string
		wantStrictSchema bool
	}{
		{model: "gpt-4o", wantStrictSchema: true},
		{model: "gpt-6-astra", wantStrictSchema: true},
		{model: "claude-3-7-sonnet", wantStrictSchema: true},
		{model: "gemini-3.8-flash", wantStrictSchema: true},
		{model: "o3", wantStrictSchema: true},
		{model: "gpt-7", wantStrictSchema: true},
		{model: "claude-4.5", wantStrictSchema: true},
		{model: "openai/gpt-7", wantStrictSchema: true},
		{model: "llama-3.3-70b", wantStrictSchema: false},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		caps := Lookup(tc.model)
		if caps.StrictSchema != tc.wantStrictSchema {
			b.Fatalf("Lookup(%q).StrictSchema = %v, want %v", tc.model, caps.StrictSchema, tc.wantStrictSchema)
		}
		benchCapsSink = caps
	}
}

func BenchmarkLookup_KnownProfiles(b *testing.B) {
	cases := []struct {
		model        string
		wantThinking bool
	}{
		{"gpt-4o", false},
		{"gpt-4o-mini", false},
		{"gpt-4.1", false},
		{"gpt-4", false},
		{"gpt-3.5-turbo", false},
		{"gpt-5", false},
		{"gpt-5.6", false},
		{"gpt-6-astra", true},
		{"o1", true},
		{"o1-mini", true},
		{"o3", true},
		{"claude-3-5-sonnet-20241022", false},
		{"claude-3-7-sonnet", true},
		{"gemini-2.0-flash", true},
		{"gemini-2.5-pro", true},
		{"gemini-3.8-flash", true},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		caps := Lookup(tc.model)
		if caps.Thinking != tc.wantThinking {
			b.Fatalf("Lookup(%q).Thinking = %v, want %v", tc.model, caps.Thinking, tc.wantThinking)
		}
		benchCapsSink = caps
	}
}

func BenchmarkLookup_InferredModels(b *testing.B) {
	cases := []struct {
		model        string
		wantThinking bool
	}{
		{"gpt-6.5", true},
		{"gpt-7", true},
		{"gpt-8", true},
		{"claude-4", true},
		{"claude-4.5", true},
		{"claude-5", true},
		{"o4-preview", true},
		{"o5-mini", true},
		{"o10", true},
		{"gemini-4.0", true},
		{"gemini-5.0-pro", true},
		{"unmapped-custom-model", false},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		caps := Lookup(tc.model)
		if caps.Thinking != tc.wantThinking {
			b.Fatalf("Lookup(%q).Thinking = %v, want %v", tc.model, caps.Thinking, tc.wantThinking)
		}
		benchCapsSink = caps
	}
}

func BenchmarkLookup_ProviderPrefixed(b *testing.B) {
	cases := []struct {
		model            string
		wantStrictSchema bool
	}{
		{"openai/gpt-4o", true},
		{"openai/gpt-7", true},
		{"anthropic/claude-3-7-sonnet", true},
		{"anthropic/claude-4.5", true},
		{"google/gemini-2.5-pro", true},
		{"google/gemini-4.0", true},
		{"custom-provider/llama-3.3-70b", false},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		caps := Lookup(tc.model)
		if caps.StrictSchema != tc.wantStrictSchema {
			b.Fatalf("Lookup(%q).StrictSchema = %v, want %v", tc.model, caps.StrictSchema, tc.wantStrictSchema)
		}
		benchCapsSink = caps
	}
}

func BenchmarkNormalizeModelSlug(b *testing.B) {
	cases := []struct {
		input string
		want  string
	}{
		{"gpt-6-astra", GPT6AstraModel},
		{"gpt 6 astra", GPT6AstraModel},
		{"GPT 6 ASTRA", GPT6AstraModel},
		{"gpt6astra", GPT6AstraModel},
		{"gpt-6", GPT6AstraModel},
		{"astra", GPT6AstraModel},
		{"astra-gpt-6", GPT6AstraModel},
		{"openai/gpt-6-astra", GPT6AstraModel},
		{"openai/astra", GPT6AstraModel},
		{"gpt-4o", "gpt-4o"},
		{"claude-3-7-sonnet", "claude-3-7-sonnet"},
		{"gemini-2.5-pro", "gemini-2.5-pro"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		got := NormalizeModelSlug(tc.input)
		if got != tc.want {
			b.Fatalf("NormalizeModelSlug(%q) = %q, want %q", tc.input, got, tc.want)
		}
		benchSlugSink = got
	}
}

func BenchmarkLookupProfile(b *testing.B) {
	cases := []struct {
		model         string
		wantCanonical string
	}{
		{"gpt-4o", "gpt-4o"},
		{"gpt-6", GPT6AstraModel},
		{"astra", GPT6AstraModel},
		{"claude-3-7-sonnet", "claude-3-7-sonnet"},
		{"gemini-3.8-flash", "gemini-3.8-flash"},
		{"gpt-7", "gpt-7"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		prof := LookupProfile(tc.model)
		if prof.Canonical != tc.wantCanonical {
			b.Fatalf("LookupProfile(%q).Canonical = %q, want %q", tc.model, prof.Canonical, tc.wantCanonical)
		}
		benchProfileSink = prof
	}
}

func BenchmarkMatrixEvaluation(b *testing.B) {
	matrix := []string{
		"gpt-4o",
		"gpt-4.1",
		"gpt-5",
		"gpt-5.6",
		"gpt-6-astra",
		"gpt-6.5",
		"gpt-7",
		"gpt-8",
		"o1",
		"o1-mini",
		"o3",
		"o5-mini",
		"astra",
		"astra-gpt-6",
		"openai/gpt-7",
		"claude-3-5-sonnet-20241022",
		"claude-3-7-sonnet",
		"claude-4-sonnet",
		"claude-4.5",
		"claude-5",
		"gemini-2.0-flash-thinking",
		"gemini-2.5-pro",
		"gemini-3.8-flash",
		"gemini-4.0",
		"llama-3.3-70b",
		"mistral-large",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matched := 0
		for _, m := range matrix {
			c := Lookup(m)
			if c.Thinking && c.ExtendedRetention && c.StrictSchema {
				matched++
				benchCapsSink = c
			}
		}
		if matched == 0 {
			b.Fatal("expected at least one model to match thinking+extended_retention+strict_schema")
		}
	}
}

func BenchmarkMatrixEvaluation_Parallel(b *testing.B) {
	matrix := []string{
		"gpt-4o",
		"gpt-6-astra",
		"gpt-7",
		"o3",
		"claude-3-7-sonnet",
		"claude-4.5",
		"gemini-2.5-pro",
		"gemini-4.0",
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		idx := 0
		var localCaps Capabilities
		for pb.Next() {
			m := matrix[idx%len(matrix)]
			c := Lookup(m)
			if m == "gpt-6-astra" && !c.Thinking {
				b.Fatalf("expected gpt-6-astra to support thinking")
			}
			localCaps = c
			idx++
		}
		_ = localCaps
	})
}

func BenchmarkLookup_WithEnvOverrides(b *testing.B) {
	ResetRegistry()
	defer ResetRegistry()

	b.Setenv("FAK_MODEL_CAPABILITIES", `{
		"gpt-4o": {"supports_thinking": true},
		"custom-fleet-model": {"supports_thinking": true, "supports_extended_retention": true, "context_window": 500000},
		"gpt-7": {"supports_thinking": false}
	}`)

	cases := []struct {
		model        string
		wantThinking bool
	}{
		{"gpt-4o", true},
		{"custom-fleet-model", true},
		{"gpt-7", false},
		{"claude-3-7-sonnet", true},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := cases[i%len(cases)]
		caps := Lookup(tc.model)
		if caps.Thinking != tc.wantThinking {
			b.Fatalf("Lookup(%q).Thinking = %v, want %v", tc.model, caps.Thinking, tc.wantThinking)
		}
		benchCapsSink = caps
	}
}

func BenchmarkCapabilities_MarshalJSON(b *testing.B) {
	caps := Capabilities{
		Thinking:          true,
		ExtendedRetention: true,
		StrictSchema:      true,
		ResponsesAPI:      true,
		AnthropicTTL:      true,
		ContextWindow:     200000,
		MaxOutputTokens:   64000,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(caps)
		if err != nil {
			b.Fatalf("json.Marshal failed: %v", err)
		}
		if len(data) == 0 {
			b.Fatal("unexpected empty JSON payload")
		}
		benchBytesSink = data
	}
}

func BenchmarkCapabilities_UnmarshalJSON(b *testing.B) {
	payload := []byte(`{
		"thinking": true,
		"supports_thinking": true,
		"extended_retention": true,
		"supports_extended_retention": true,
		"strict_schema": true,
		"supports_strict_schema": true,
		"responses_api": true,
		"supports_responses_api": true,
		"anthropic_ttl": true,
		"supports_anthropic_ttl": true,
		"context_window": 200000,
		"max_output_tokens": 64000
	}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var caps Capabilities
		if err := json.Unmarshal(payload, &caps); err != nil {
			b.Fatalf("json.Unmarshal failed: %v", err)
		}
		if !caps.Thinking || !caps.ExtendedRetention || !caps.StrictSchema {
			b.Fatal("unexpected unmarshaled capabilities")
		}
		benchCapsSink = caps
	}
}
