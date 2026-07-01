package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// provider_cache_fak_placement_savings_test.go — the SAVINGS witness for the offensive
// cache-breakpoint placement (#806), the fak-SPECIFIC half of "does fak's caching save
// real money on the Claude/Anthropic path?".
//
// Placement CORRECTNESS is already witnessed in internal/agent/anthropic_cachebp_test.go
// (where the breakpoint lands, byte-identity, volatility). The pricing MODEL is witnessed in
// cache_pricing_test.go. What was MISSING (and what this file adds) is the isolated causal +
// economic witness that ties them together on the real wire transform:
//
//	a caller that sends NO cache_control (a raw OpenAI-shaped client, a minimal SDK, a
//	hand-rolled request) leaves provider prefix caching entirely on the table — its stable
//	head is re-prefilled at FULL price every turn. fak's offensive placement (gateway
//	messages.go:480, exercised here through the real Server.maybeCompactAnthropicRaw path)
//	splices a cache_control breakpoint onto that head, so the provider can serve turns 2..N
//	from cache at 0.1x instead of 1.0x.
//
// The provider's prompt-cache accounting rule is documented and deterministic: a request
// carrying a cache_control breakpoint whose keyed prefix was seen before bills that prefix
// as cache_read (0.1x); the first sight bills it as cache_creation (1.25x, 5m tier); a
// request with NO breakpoint caches nothing. providerCacheAccounting below implements THAT
// rule over the ACTUAL forwarded bytes, so the causal claim ("cache_read 0 -> X because fak
// placed the breakpoint") is derived from what fak really sends, not asserted.
//
// PROVENANCE (net-true-value standard, docs/standards/net-true-value.md):
//   - PLACEMENT + COUNTERFACTUAL are WITNESSED: the production transform on the real body,
//     and the same body with the lever off left byte-for-byte unchanged.
//   - CAUSALITY is WITNESSED over the forwarded bytes: cache_read is derived from the
//     presence/absence of the breakpoint fak did or did not add.
//   - MAGNITUDE is MODELED with the provider's DOCUMENTED economics (the shipped CachePricing
//     model + Anthropic's published 0.1x read / 1.25x 5m-write multipliers), NET of the
//     one-time write premium.
//   - SCOPE FENCE: this is the win for callers that send NO breakpoint. It is NOT a win over
//     Claude Code, which already marks its own head — there fak is identity (already_set) and
//     the provider cache is the CLIENT's, not fak's. This witness never claims that slice.

// rawClaudeCallerNoCacheControl builds a realistic /v1/messages body for a caller that set
// NO cache_control anywhere: a stable, byte-identical system head (headChars of plain policy
// prose — no per-request UUID or sub-day timestamp, so it is genuinely cacheable) plus a
// short volatile user turn. This is the exact population the offensive half (#806) targets.
func rawClaudeCallerNoCacheControl(t *testing.T, headChars int, userTurn string) []byte {
	t.Helper()
	head := strings.Repeat("You are a careful assistant. Follow the policy exactly. ", 1+headChars/56)
	if len(head) > headChars {
		head = head[:headChars]
	}
	raw, err := json.Marshal(map[string]any{
		"model":      "claude-opus-4-8",
		"max_tokens": 1024,
		"stream":     true,
		"system":     []map[string]any{{"type": "text", "text": head}},
		"messages":   []map[string]any{{"role": "user", "content": userTurn}},
	})
	if err != nil {
		t.Fatalf("marshal raw caller body: %v", err)
	}
	if bytes.Contains(raw, []byte("cache_control")) {
		t.Fatal("fixture sanity: raw caller body must carry NO cache_control")
	}
	return raw
}

// providerCacheAccounting is the DOCUMENTED Anthropic prompt-cache billing rule applied to a
// forwarded /v1/messages body: the cacheable prefix ends at the cache_control breakpoint. A
// body with no breakpoint caches nothing (all prompt tokens bill at full price). With a
// breakpoint, the prefix bills as cache_read when it was seen on a prior turn, else as
// cache_creation (5m write tier). Token estimates use the gateway's own 4-chars/token
// convention; the prefix is measured to the breakpoint's byte offset, so it is a CONSERVATIVE
// count (never overstates the cached span). This is the "mock upstream echo" the missing
// witness needed, applied directly to the bytes fak forwards.
func providerCacheAccounting(body []byte, prefixSeenBefore bool) CacheUsage {
	total := int(estimatedTokensFromBytes(len(body)))
	bp := bytes.Index(body, []byte(`"cache_control":{"type":"ephemeral"}`))
	if bp < 0 {
		// No breakpoint: the provider has nothing to key a cache on — full price, every turn.
		return CacheUsage{InputTokens: total}
	}
	prefix := int(estimatedTokensFromBytes(bp))
	if prefix > total {
		prefix = total
	}
	tail := total - prefix
	if prefixSeenBefore {
		return CacheUsage{InputTokens: tail, CacheReadTokens: prefix}
	}
	return CacheUsage{InputTokens: tail, CacheCreationTokens: prefix, WriteTTL: CacheTTL5m}
}

// TestFakPlacementUnlocksProviderCacheSavings is the fak-specific savings witness on the
// Claude path. It proves, end to end on the real wire transform:
//
//  1. WITHOUT fak (the lever off) a no-cache_control body is forwarded unchanged — the
//     counterfactual a raw caller actually gets: the provider caches nothing, every turn.
//  2. WITH fak (the default-on lever) the SAME body gets a cache_control breakpoint spliced
//     onto its stable head, byte-identically, so the provider serves turns 2..N from cache.
//  3. Priced over an N-turn session with the shipped CachePricing model at published
//     Anthropic economics, the placement is a NET saving after the one-time write premium,
//     break-even at turn 2 — and because the counterfactual caches nothing, 100% of that
//     saving is fak-attributable.
func TestFakPlacementUnlocksProviderCacheSavings(t *testing.T) {
	const (
		headChars = 8000 // ~2000-token stable system head (4 chars/token)
		userTurn  = "Please summarize the open incidents and propose next steps for the on-call."
		turns     = 8
	)
	raw := rawClaudeCallerNoCacheControl(t, headChars, userTurn)

	// --- (1) Counterfactual: the lever OFF forwards the raw body unchanged. ---
	reqOff, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode (off): %v", err)
	}
	orig := append([]byte(nil), reqOff.Raw...)
	anthropicPassthroughServer(0).maybeCompactAnthropicRaw(reqOff)
	if !bytes.Equal(reqOff.Raw, orig) {
		t.Fatal("lever OFF must forward the raw body byte-for-byte (the no-fak counterfactual)")
	}

	// --- (2) With fak: the DEFAULT-ON lever splices a breakpoint onto the stable head. ---
	reqOn, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode (on): %v", err)
	}
	anthropicPassthroughServer(DefaultCompactHistoryBudget).maybeCompactAnthropicRaw(reqOn)
	if bytes.Equal(reqOn.Raw, orig) {
		t.Fatal("fak must place a cache_control breakpoint on the stable head (the body must change)")
	}
	if !bytes.Contains(reqOn.Raw, []byte(`"cache_control":{"type":"ephemeral"}`)) {
		t.Fatalf("fak did not splice an ephemeral cache_control breakpoint:\n%s", reqOn.Raw)
	}
	// fak only INSERTS the breakpoint key: removing that exact span recovers the original
	// body byte-for-byte, so the cached prefix the provider keys on is untouched. (The
	// where-it-lands / prefix byte-identity is proven directly in anthropic_cachebp_test.go.)
	recovered := bytes.Replace(reqOn.Raw, []byte(`,"cache_control":{"type":"ephemeral"}`), nil, 1)
	if !bytes.Equal(recovered, orig) {
		t.Fatal("placement must only INSERT the breakpoint; the rest of the body must be byte-identical")
	}
	if _, err := agent.DecodeAnthropicMessagesRequest(reqOn.Raw); err != nil {
		t.Fatalf("placed body failed to re-decode as a valid request: %v", err)
	}

	// --- (3a) CAUSALITY over the forwarded bytes: replay N turns of the provider's billing
	//          rule against each arm's actual forwarded body. ---
	pricing := CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25} // Opus-4.8 example base rates
	const outputTokens = 200                                          // cancels in the saving; realistic per-turn cost

	var baselineUSD, fakUSD, cumSavingsAfterTurn1 float64
	var offCacheRead, onCacheRead, onCacheCreate int
	for turn := 1; turn <= turns; turn++ {
		seen := turn > 1

		// Causality control: the no-fak body carries no breakpoint, so the provider
		// caches nothing on every turn — cache_read stays 0.
		offCacheRead += providerCacheAccounting(reqOff.Raw, seen).CacheReadTokens

		// fak arm: the provider serves the prefix from cache (0.1x read) or writes it
		// on the first turn (1.25x, 5m tier); the volatile tail bills at full price.
		on := providerCacheAccounting(reqOn.Raw, seen)
		on.OutputTokens = outputTokens
		onCacheRead += on.CacheReadTokens
		onCacheCreate += on.CacheCreationTokens
		fakUSD += pricing.CostUSD(on)

		// Baseline: the SAME prompt with NO cache — the cached prefix AND the tail billed
		// at full input price. Same (prefix, tail) decomposition as the fak arm, so the
		// two arms differ ONLY in how the prefix is treated, and the tail/output cancel in
		// the difference — the saving is purely the prompt-cache effect fak unlocked.
		prefix := on.CacheReadTokens + on.CacheCreationTokens
		baseline := CacheUsage{InputTokens: prefix + on.InputTokens, OutputTokens: outputTokens}
		baselineUSD += pricing.CostUSD(baseline)

		if turn == 1 {
			cumSavingsAfterTurn1 = baselineUSD - fakUSD
		}
	}
	savingsUSD := baselineUSD - fakUSD

	// The no-fak arm caches NOTHING across the whole session — the provider had no breakpoint.
	if offCacheRead != 0 {
		t.Fatalf("no-fak counterfactual cache_read = %d, want 0 (a raw caller gets no provider cache)", offCacheRead)
	}
	// The fak arm turns cache_read from 0 into a real number — caused solely by the placement.
	if onCacheRead <= 0 || onCacheCreate <= 0 {
		t.Fatalf("fak arm cache_read=%d create=%d, want both > 0 (placement unlocked the cache)", onCacheRead, onCacheCreate)
	}
	// After turn 1 alone the one-time write premium makes fak MORE expensive — the honest cost.
	if cumSavingsAfterTurn1 >= 0 {
		t.Fatalf("turn-1 cumulative saving = %.8f, want negative (the one-time write premium)", cumSavingsAfterTurn1)
	}
	// By the end of the session the reads have repaid the write and then some.
	if savingsUSD <= 0 {
		t.Fatalf("net saving over %d turns = %.8f, want strictly positive", turns, savingsUSD)
	}

	// --- (3b) Fold the fak-unlocked cache through the SAME owner/mechanism split the guard
	//          exit summary and /metrics render, and cross-check the dollars. Because the
	//          counterfactual caches nothing, this whole provider slice is fak-attributable. ---
	sum := AdjudicationSummary{
		CachedPromptTokens:  uint64(onCacheRead),
		CacheCreationTokens: uint64(onCacheCreate),
	}
	ms := sum.MechanismSavings()
	if ms.ProviderTokenEquiv() <= 0 {
		t.Fatalf("net fak-unlocked token-equiv = %.2f, want positive", ms.ProviderTokenEquiv())
	}
	tokenEquivUSD := ms.ProviderTokenEquiv() * (pricing.InputPerMTokUSD / 1_000_000)
	if !approx(tokenEquivUSD, savingsUSD) {
		t.Fatalf("token-equiv $ (%.8f) and per-turn $ (%.8f) disagree", tokenEquivUSD, savingsUSD)
	}

	// Capture the witness so `go test -v -run TestFakPlacementUnlocksProviderCacheSavings`
	// prints a copy-pasteable proof block for the artifact.
	pct := 100.0 * savingsUSD / baselineUSD
	t.Log(strings.TrimSpace(fmt.Sprintf(`
FAK-SPECIFIC PROMPT-CACHE SAVINGS WITNESS (offensive placement #806, Claude/Anthropic path)
  scope         : one caller that sent NO cache_control; %d-turn session; 5m cache tier; Opus-4.8 example rates {$%.0f/$%.0f per MTok}
  placement     : WITNESSED — real Server.maybeCompactAnthropicRaw spliced an ephemeral breakpoint on the stable head, prefix byte-identical
  counterfactual: WITNESSED — same body, lever off => forwarded unchanged; no breakpoint => provider cache_read=0 across all %d turns
  causality     : WITNESSED — cache_read derived from the forwarded bytes: 0 without fak, %d tok with fak (creation %d tok on turn 1)
  magnitude     : MODELED   — shipped CachePricing + published Anthropic 0.1x read / 1.25x write economics, net of the write premium
  ---- economics ----
  baseline (no fak) : $%.6f   (head re-prefilled at full price every turn)
  with fak          : $%.6f
  net saving        : $%.6f  = %.1f%% of the prompt-cache spend over %d turns
  turn-1 saving     : $%.6f  (NEGATIVE — the honest one-time write premium; break-even at turn 2)
  attribution       : provider(read rebate) %.0f + provider(write premium) %.0f = net %.0f token-equiv, 100%% fak-unlocked`,
		turns, pricing.InputPerMTokUSD, pricing.OutputPerMTokUSD, turns,
		onCacheRead, onCacheCreate,
		baselineUSD, fakUSD, savingsUSD, pct, turns, cumSavingsAfterTurn1,
		ms.ProviderPromptCacheReadTokenEquiv, ms.ProviderPromptCacheWritePremiumTokenEquiv, ms.ProviderTokenEquiv())))
}
