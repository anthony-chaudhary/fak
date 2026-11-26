package agent

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/timingspan"
)

// Usage is the token accounting a completion reports.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	// CostUSD is an upstream-reported completion cost. It stays nil when the
	// provider omits dollars; fak never derives it from token counts here.
	CostUSD                  *float64           `json:"cost_usd,omitempty"`
	CostStatus               string             `json:"cost_status,omitempty"`
	CostProvenance           string             `json:"cost_provenance,omitempty"`
	TotalTokens              int                `json:"total_tokens"`
	PromptTokensDetails      *UsageTokenDetails `json:"prompt_tokens_details,omitempty"`
	InputTokensDetails       *UsageTokenDetails `json:"input_tokens_details,omitempty"`
	CacheReadInputTokens     int                `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int                `json:"cache_creation_input_tokens,omitempty"`
	// PromptCacheHitTokens / PromptCacheMissTokens are DeepSeek's TOP-LEVEL prompt-cache
	// counters (context caching is on by default there): hit is the prefix served from the
	// provider's KV cache, miss is the remainder it re-ingested, and prompt_tokens == hit
	// + miss. Both are OBSERVED provider-relayed counters — a DeepSeek cache hit is the
	// provider's own doing unless a separate fak-authored mechanism shaped the request.
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens,omitempty"`
	// CompletionTokensDetails carries the reasoning subcounter DeepSeek-style reasoning
	// models report; completion_tokens still INCLUDES it, so it is a breakdown, not an
	// additional axis.
	CompletionTokensDetails *UsageCompletionTokenDetails `json:"completion_tokens_details,omitempty"`
}

// Timings is llama.cpp server's per-request `timings` object, ported field for
// field so llama.cpp-aware clients and agents read fak without an adapter.
// CacheN is the prompt prefix served from the KV cache (not re-prefilled).
type Timings struct {
	CacheN              int     `json:"cache_n"`
	PromptN             int     `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          int     `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
	// FirstEmitMS is request receipt to the first byte sent to the client; it
	// includes the whole prefill and any reasoning decode that ran before it.
	FirstEmitMS float64 `json:"first_emit_ms,omitempty"`
	// DraftN / DraftNAccepted are llama.cpp's speculative-decoding pair, present
	// only when the turn drafted tokens.
	DraftN         int `json:"draft_n,omitempty"`
	DraftNAccepted int `json:"draft_n_accepted,omitempty"`
	// Inkling2Width is the cohort width an Inkling-2 turn decoded in (1 alone) and
	// Inkling2Mode names that path, so a client can witness server-side coalescing.
	Inkling2Width int    `json:"inkling2_width,omitempty"`
	Inkling2Mode  string `json:"inkling2_mode,omitempty"`
	// MetalPipelines is present when the model keeps a Metal pipeline archive,
	// so a cold start that compiled reads apart from an archive hit.
	MetalPipelines *MetalPipelineTimings `json:"metal_pipelines,omitempty"`
}

// MetalPipelineTimings counts the model's Metal pipelines since it loaded:
// those Metal compiled, those created from the archive, the archive's listed
// count at load, and the load time spent creating pipelines before any prompt.
type MetalPipelineTimings struct {
	Compiled    int     `json:"compiled"`
	ArchiveHits int     `json:"archive_hits"`
	Archived    int     `json:"archived"`
	LoadMS      float64 `json:"load_ms"`
}

// NewTimings builds Timings from token counts and phase seconds. Like llama.cpp,
// PromptN counts only the tokens actually prefilled (promptTokens - cachedTokens).
func NewTimings(promptTokens, cachedTokens, predicted int, prefillS, decodeS float64) *Timings {
	if cachedTokens < 0 {
		cachedTokens = 0
	}
	if cachedTokens > promptTokens {
		cachedTokens = promptTokens
	}
	t := &Timings{CacheN: cachedTokens, PromptN: promptTokens - cachedTokens, PredictedN: predicted}
	if prefillS > 0 {
		t.PromptMS = prefillS * 1e3
		if t.PromptN > 0 {
			t.PromptPerTokenMS = t.PromptMS / float64(t.PromptN)
			t.PromptPerSecond = float64(t.PromptN) / prefillS
		}
	}
	if decodeS > 0 {
		t.PredictedMS = decodeS * 1e3
		if predicted > 0 {
			t.PredictedPerTokenMS = t.PredictedMS / float64(predicted)
			t.PredictedPerSecond = float64(predicted) / decodeS
		}
	}
	return t
}

// MarshalJSON adds "spans", naming the span each measured clock field timed.
// It is derived here rather than stored so Timings stays comparable.
func (t Timings) MarshalJSON() ([]byte, error) {
	type plain Timings
	var spans map[string]string
	declare := func(field string, s timingspan.Span) {
		if spans == nil {
			spans = map[string]string{}
		}
		spans[field] = s.String()
	}
	if t.PromptMS > 0 {
		declare("prompt_ms", timingspan.PromptSpan)
	}
	if t.PredictedMS > 0 {
		declare("predicted_ms", timingspan.DecodeSpan)
	}
	if t.FirstEmitMS > 0 {
		declare("first_emit_ms", timingspan.FirstEmitSpan)
	}
	return json.Marshal(struct {
		plain
		Spans map[string]string `json:"spans,omitempty"`
	}{plain(t), spans})
}

// UsageTokenDetails carries provider-specific prompt/input token subcounters.
type UsageTokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// UsageCompletionTokenDetails carries provider-specific completion token subcounters
// (the DeepSeek/OpenAI-compatible completion_tokens_details block).
type UsageCompletionTokenDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// CachedPromptTokens is the provider-reported prompt-cache hit count, normalized
// across OpenAI chat-completions, OpenAI Responses, Anthropic-style, and DeepSeek
// top-level counters.
func (u Usage) CachedPromptTokens() int {
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 {
		return u.PromptTokensDetails.CachedTokens
	}
	if u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0 {
		return u.InputTokensDetails.CachedTokens
	}
	if u.CacheReadInputTokens > 0 {
		return u.CacheReadInputTokens
	}
	// DeepSeek reports its prompt-cache hit as a TOP-LEVEL field, not nested details.
	if u.PromptCacheHitTokens > 0 {
		return u.PromptCacheHitTokens
	}
	return 0
}

// UncachedPromptTokens is the prompt the model actually re-ingested this turn — the
// full prompt minus the provider's cache-read hit — normalized so the count means the
// same thing across providers. Anthropic already reports prompt/input_tokens as the
// UNCACHED remainder (cache_read_input_tokens is a separate field), so it is returned
// as-is. OpenAI (chat + Responses) and Gemini fold the cached hit INTO prompt_tokens,
// so the cached portion is peeled back off to leave the uncached remainder. The result
// is never negative, and UncachedPromptTokens() + CachedPromptTokens() == the full
// resident prompt on every provider. This is the companion of CachedPromptTokens(): a
// consumer that splits a turn into (uncached, cached) — e.g. the vCache observe plane's
// baseline-token-equiv — gets a provider-consistent split from the pair.
func (u Usage) UncachedPromptTokens() int {
	// DeepSeek reports the uncached remainder DIRECTLY as prompt_cache_miss_tokens
	// (prompt_tokens == hit + miss), so the provider's own miss counter is returned
	// as-is — it already satisfies uncached + cached == the full resident prompt.
	if u.PromptCacheMissTokens > 0 {
		return u.PromptCacheMissTokens
	}
	n := u.PromptTokens
	// OpenAI/Gemini shape: prompt_tokens INCLUDES the cache-read hit (reported in
	// prompt_tokens_details/input_tokens_details), so subtract it to match Anthropic's
	// already-uncached input_tokens. The Anthropic shape carries its cache read in the
	// separate CacheReadInputTokens field and is left untouched. DeepSeek's fully-cached
	// edge (hit > 0, miss == 0) also folds its hit INTO prompt_tokens, so it peels off
	// the same way.
	if (u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0) ||
		(u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0) ||
		u.PromptCacheHitTokens > 0 {
		n -= u.CachedPromptTokens()
	}
	if n < 0 {
		n = 0
	}
	return n
}

// ContextWindowTokens is the prompt/context size that should count against a
// long-session context budget. OpenAI-style prompt_tokens already include cached
// prompt tokens, so their details are NOT added again. Anthropic reports
// input_tokens as the uncached remainder and cache_read/cache_creation separately;
// those counters are added back so the budget reflects the full context the model
// attended to.
func (u Usage) ContextWindowTokens() int {
	// Provider counts remain untouched. A positive sum beyond int capacity
	// must exhaust a finite context budget, not wrap negative and skip its debit.
	add := func(a, b int) int {
		maxInt := int(^uint(0) >> 1)
		if b > 0 && a > maxInt-b {
			return maxInt
		}
		return a + b
	}
	n := u.PromptTokens
	// DeepSeek folds hit + miss INTO prompt_tokens (like OpenAI), so its cache counters
	// are NOT added again — that would double-count the resident context. Only when a
	// wire omits prompt_tokens entirely do the two counters reconstruct it.
	if n == 0 && (u.PromptCacheHitTokens > 0 || u.PromptCacheMissTokens > 0) {
		n = add(u.PromptCacheHitTokens, u.PromptCacheMissTokens)
	}
	if u.CacheReadInputTokens > 0 {
		n = add(n, u.CacheReadInputTokens)
	}
	if u.CacheCreationInputTokens > 0 {
		n = add(n, u.CacheCreationInputTokens)
	}
	return n
}

// ReasoningTokens is the provider-reported reasoning/thinking slice of the completion
// (DeepSeek-style completion_tokens_details.reasoning_tokens), or 0 when the wire does
// not report one. It is surfaced as a SEPARATE subcounter: CompletionTokens keeps the
// provider's own meaning (DeepSeek's completion_tokens INCLUDES reasoning), so reasoning
// is never mixed into — or silently subtracted from — final-answer token accounting.
func (u Usage) ReasoningTokens() int {
	if u.CompletionTokensDetails != nil {
		return u.CompletionTokensDetails.ReasoningTokens
	}
	return 0
}
