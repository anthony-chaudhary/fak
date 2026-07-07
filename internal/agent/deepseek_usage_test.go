package agent

import "testing"

// TestDeepSeekUsageCacheCounters proves the DeepSeek wire shape — TOP-LEVEL
// prompt_cache_hit_tokens / prompt_cache_miss_tokens, with prompt_tokens == hit + miss —
// normalizes through the same provider-neutral accessors every cache-value consumer
// reads: CachedPromptTokens() returns the hit, UncachedPromptTokens() returns the miss,
// and ContextWindowTokens() reflects the full resident context WITHOUT double-counting
// the cache counters back on top of prompt_tokens. DeepSeek rides the OpenAI
// chat-completions adapter, so the fixture goes through ParseResponse like a live turn.
func TestDeepSeekUsageCacheCounters(t *testing.T) {
	cases := []struct {
		name         string
		raw          string
		wantCached   int
		wantUncached int
		wantResident int
	}{
		// Warm turn: 1000-token prompt, 800 served from the provider's KV cache.
		{"warm_hit_and_miss",
			`{"model":"deepseek-v4-pro","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":20,"total_tokens":1020,"prompt_cache_hit_tokens":800,"prompt_cache_miss_tokens":200}}`,
			800, 200, 1000},
		// Cold turn: everything missed, nothing cached.
		{"cold_all_miss",
			`{"model":"deepseek-v4-flash","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":500,"completion_tokens":9,"total_tokens":509,"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":500}}`,
			0, 500, 500},
		// Fully cached edge: miss == 0, the whole prompt was a hit.
		{"fully_cached",
			`{"model":"deepseek-v4-pro","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":700,"completion_tokens":3,"total_tokens":703,"prompt_cache_hit_tokens":700,"prompt_cache_miss_tokens":0}}`,
			700, 0, 700},
	}
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			comp, err := adapter.ParseResponse([]byte(c.raw))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			u := comp.Usage
			if got := u.CachedPromptTokens(); got != c.wantCached {
				t.Errorf("CachedPromptTokens() = %d, want %d (prompt_cache_hit_tokens)", got, c.wantCached)
			}
			if got := u.UncachedPromptTokens(); got != c.wantUncached {
				t.Errorf("UncachedPromptTokens() = %d, want %d (prompt_cache_miss_tokens)", got, c.wantUncached)
			}
			if got := u.UncachedPromptTokens() + u.CachedPromptTokens(); got != c.wantResident {
				t.Errorf("uncached+cached = %d, want full resident prompt %d", got, c.wantResident)
			}
			if got := u.ContextWindowTokens(); got != c.wantResident {
				t.Errorf("ContextWindowTokens() = %d, want %d (no double-counting of hit/miss atop prompt_tokens)", got, c.wantResident)
			}
		})
	}
}

// TestDeepSeekUsageReasoningTokens proves the completion_tokens_details.reasoning_tokens
// subcounter is surfaced SEPARATELY: ReasoningTokens() reports it, while CompletionTokens
// keeps the provider's own total (which INCLUDES reasoning on the DeepSeek wire) — the
// reasoning slice is never mixed into or subtracted from final-answer token accounting.
func TestDeepSeekUsageReasoningTokens(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	raw := `{"model":"deepseek-v4-pro","choices":[{"message":{"role":"assistant","reasoning_content":"think","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,` +
		`"prompt_cache_hit_tokens":60,"prompt_cache_miss_tokens":40,` +
		`"completion_tokens_details":{"reasoning_tokens":30}}}`
	comp, err := adapter.ParseResponse([]byte(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got := comp.Usage.ReasoningTokens(); got != 30 {
		t.Errorf("ReasoningTokens() = %d, want 30", got)
	}
	if comp.Usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want the provider's untouched 50 (reasoning is a breakdown, not a deduction)", comp.Usage.CompletionTokens)
	}
	// A wire with no details block reads 0, never panics.
	if got := (Usage{CompletionTokens: 7}).ReasoningTokens(); got != 0 {
		t.Errorf("ReasoningTokens() with no details = %d, want 0", got)
	}
}

// TestDeepSeekUsageDoesNotDisturbOtherProviders pins that adding the DeepSeek top-level
// counters left the existing provider shapes byte-identical in meaning: a usage block
// with ONLY the OpenAI nested details, the Anthropic separate cache_read field, or no
// cache fields at all reads exactly as before.
func TestDeepSeekUsageDoesNotDisturbOtherProviders(t *testing.T) {
	openai := Usage{PromptTokens: 13, PromptTokensDetails: &UsageTokenDetails{CachedTokens: 5}}
	if openai.CachedPromptTokens() != 5 || openai.UncachedPromptTokens() != 8 || openai.ContextWindowTokens() != 13 {
		t.Errorf("openai shape drifted: cached=%d uncached=%d window=%d, want 5/8/13",
			openai.CachedPromptTokens(), openai.UncachedPromptTokens(), openai.ContextWindowTokens())
	}
	anthropic := Usage{PromptTokens: 13, CacheReadInputTokens: 7, CacheCreationInputTokens: 90}
	if anthropic.CachedPromptTokens() != 7 || anthropic.UncachedPromptTokens() != 13 || anthropic.ContextWindowTokens() != 110 {
		t.Errorf("anthropic shape drifted: cached=%d uncached=%d window=%d, want 7/13/110",
			anthropic.CachedPromptTokens(), anthropic.UncachedPromptTokens(), anthropic.ContextWindowTokens())
	}
	bare := Usage{PromptTokens: 42}
	if bare.CachedPromptTokens() != 0 || bare.UncachedPromptTokens() != 42 || bare.ContextWindowTokens() != 42 {
		t.Errorf("bare shape drifted: cached=%d uncached=%d window=%d, want 0/42/42",
			bare.CachedPromptTokens(), bare.UncachedPromptTokens(), bare.ContextWindowTokens())
	}
}
