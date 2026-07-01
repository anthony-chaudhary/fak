package gateway

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestCodexCacheHitIsVisibleInObservePlane is the witness that provider prompt-cache
// value is really recorded for the OpenAI Responses (codex) and chat wires, not only for
// Anthropic. The live per-family observe plane is fed from logInferenceTurn ->
// observeVCacheTurn; before the fix that call read the Anthropic-only
// cache_read_input_tokens field, which is 0 for OpenAI/Gemini (their hit lands in
// prompt_tokens_details.cached_tokens). So a codex session that the upstream served
// entirely from cache registered read=0 and looked permanently COLD.
//
// The fix normalizes the read axis (CachedPromptTokens) and the uncached remainder
// (UncachedPromptTokens — OpenAI folds the hit INTO prompt_tokens, so it is peeled back
// off). This test drives a codex-shaped turn and asserts the window now records the hit,
// with input+read == the full resident prompt.
func TestCodexCacheHitIsVisibleInObservePlane(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now())}

	// A codex (OpenAI Responses) turn: prompt_tokens=1000 INCLUDES the 800 cache hit,
	// reported in prompt_tokens_details — exactly what the Responses adapter produces.
	codex := agent.Usage{
		PromptTokens:        1000,
		CompletionTokens:    20,
		PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 800},
	}
	s.logInferenceTurn("codex-sess", "openai_responses", false, codex, "stop", time.Millisecond, false)

	turns, _ := s.metrics.vcacheTurnsSnapshot()
	if len(turns) != 1 {
		t.Fatalf("want 1 observed turn, got %d", len(turns))
	}
	got := turns[0]
	// The cache hit is now VISIBLE (read>0), not silently dropped to 0.
	if got.CacheRead != 800 {
		t.Fatalf("codex cache hit not recorded in observe plane: CacheRead=%d, want 800", got.CacheRead)
	}
	// The uncached remainder is peeled back off prompt_tokens so input+read == the full
	// prompt (1000), matching how Anthropic's already-uncached input_tokens reads.
	if got.InputTokens != 200 {
		t.Fatalf("codex uncached input not normalized: InputTokens=%d, want 200 (1000-800)", got.InputTokens)
	}
	if got.InputTokens+got.CacheRead != 1000 {
		t.Fatalf("input+read=%d, want the full resident prompt 1000", got.InputTokens+got.CacheRead)
	}
	// The family now reads as having cache activity — the governor/warmth plane can see it.
	if !vcacheWindowHasCacheActivity(turns) {
		t.Fatal("codex family must register cache activity once the hit is normalized")
	}
}

// TestCodexAndClaudeCacheReadIdenticallyInObservePlane proves the normalization makes a
// codex turn and a Claude turn with the SAME real cache split land as byte-identical
// observe-plane rows — so "caching value working" means the same thing for both providers
// and neither is silently understated relative to the other.
func TestCodexAndClaudeCacheReadIdenticallyInObservePlane(t *testing.T) {
	// Codex: prompt_tokens=1000 includes the 800 hit (details).
	codex := agent.Usage{PromptTokens: 1000, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 800}}
	// Claude: input_tokens=200 is already the uncached remainder; the 800 hit is the
	// separate cache_read_input_tokens field. Same real split: 200 uncached + 800 cached.
	claude := agent.Usage{PromptTokens: 200, CacheReadInputTokens: 800}

	sCodex := &Server{metrics: newGatewayMetrics(time.Now())}
	sCodex.logInferenceTurn("c", "openai_responses", false, codex, "stop", time.Millisecond, false)
	sClaude := &Server{metrics: newGatewayMetrics(time.Now())}
	sClaude.logInferenceTurn("c", "anthropic_messages", false, claude, "end_turn", time.Millisecond, false)

	cx, _ := sCodex.metrics.vcacheTurnsSnapshot()
	cl, _ := sClaude.metrics.vcacheTurnsSnapshot()
	if len(cx) != 1 || len(cl) != 1 {
		t.Fatalf("want one turn each, got codex=%d claude=%d", len(cx), len(cl))
	}
	if cx[0].InputTokens != cl[0].InputTokens || cx[0].CacheRead != cl[0].CacheRead {
		t.Fatalf("codex and claude must read identically: codex(in=%d,read=%d) claude(in=%d,read=%d)",
			cx[0].InputTokens, cx[0].CacheRead, cl[0].InputTokens, cl[0].CacheRead)
	}
}

// TestAnthropicObservePlaneUnchangedByNormalization is the regression guard: for the
// Anthropic wire (no prompt_tokens_details; the hit is in cache_read_input_tokens), the
// normalized read/input are byte-identical to the raw fields the plane used before, so the
// fix cannot perturb the already-correct Claude path.
func TestAnthropicObservePlaneUnchangedByNormalization(t *testing.T) {
	s := &Server{metrics: newGatewayMetrics(time.Now())}
	u := agent.Usage{PromptTokens: 20, CacheReadInputTokens: 80, CacheCreationInputTokens: 5}
	s.logInferenceTurn("a", "anthropic_messages", false, u, "end_turn", time.Millisecond, false)

	turns, _ := s.metrics.vcacheTurnsSnapshot()
	if len(turns) != 1 {
		t.Fatalf("want 1 turn, got %d", len(turns))
	}
	got := turns[0]
	if got.InputTokens != 20 || got.CacheRead != 80 || got.CacheCreation != 5 {
		t.Fatalf("anthropic path perturbed by normalization: in=%d read=%d create=%d, want 20/80/5",
			got.InputTokens, got.CacheRead, got.CacheCreation)
	}
}
