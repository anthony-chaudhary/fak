package gateway

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func (s *Server) logInferenceTurn(traceID, wire string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted bool) {
	s.logInferenceTurnWithContextEvent(traceID, wire, stream, usage, finishReason, dur, compacted, compacted || s.consumeDecodedCtxViewEvent(traceID))
}

// logInferenceTurnAs is the principal-aware funnel entry (#14451): a route that
// holds the request context passes the authenticated keyset principal stamped
// by withAuth so the usage ledger keys the row on the presenting key, never on
// a client-chosen trace or header.
func (s *Server) logInferenceTurnAs(traceID, wire string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted, contextEvent bool, principal string) {
	if s == nil {
		return
	}
	s.logInferenceTurnForModelAs(traceID, wire, s.model, stream, usage, finishReason, dur, compacted, contextEvent, principal)
}

func (s *Server) logInferenceTurnWithContextEvent(traceID, wire string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted, contextEvent bool) {
	if s == nil {
		return
	}
	s.logInferenceTurnForModel(traceID, wire, s.model, stream, usage, finishReason, dur, compacted, contextEvent)
}

// logInferenceTurnForModel records an explicitly selected upstream model while
// the wire response can retain the client's public model alias.
func (s *Server) logInferenceTurnForModel(traceID, wire, model string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted, contextEvent bool) {
	s.logInferenceTurnForModelAs(traceID, wire, model, stream, usage, finishReason, dur, compacted, contextEvent, "")
}

// logInferenceTurnForModelAs is the shared per-turn chokepoint with the
// authenticated principal threaded through for the usage ledger (#14451).
func (s *Server) logInferenceTurnForModelAs(traceID, wire, model string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted, contextEvent bool, principal string) {
	if s == nil {
		return
	}
	// #2179: attribute this turn's cache-creation write to the 1h tier when the
	// managed-cache TTL-upgrade rung was already active for the session. GATEWAY-
	// ATTRIBUTED, not provider-reported — the Anthropic usage block never splits 5m
	// vs 1h creation tokens, so this is fak's own per-turn upgrade witness
	// (noteCtxValueTTL1h), read BEFORE this turn's write is folded into the total.
	s.metrics.recordCacheCreationTierSplit(usage.CacheCreationInputTokens, s.ttl1hActiveFor(traceID), s.ttl1hMessagePrefixFor(traceID))
	// Record this turn into the per-family live-observe window (#935) BEFORE the sink
	// gates below, so the per-family / governor / warmth view is populated even with
	// --log off and --debug-stats off. The family is the session/trace prefix; the token
	// axes are the provider's own counters (OBSERVED). This is purely observational — it
	// never feeds the request path (Law A2).
	//
	// The cache-read axis is the PROVIDER-NORMALIZED hit (CachedPromptTokens), not the
	// Anthropic-only cache_read_input_tokens field: OpenAI (chat + the codex Responses
	// wire) and Gemini report their prompt-cache hit in prompt_tokens_details, leaving
	// cache_read_input_tokens at 0, so reading the raw field made every codex/OpenAI cache
	// hit register as read=0 — the families/governor/warmth plane saw those sessions as
	// permanently COLD even when the upstream served a real cache. Pairing the normalized
	// read with the uncached remainder (UncachedPromptTokens — OpenAI/Gemini fold the hit
	// INTO prompt_tokens, so it is peeled back off) keeps input+read == the full resident
	// prompt on every provider, so a codex family's hit-rate/economics read identically to
	// a Claude family's rather than being understated by the double-counted cached span.
	cacheRead := usage.CachedPromptTokens()
	uncachedPrompt := usage.UncachedPromptTokens()
	// One timestamp shared by the native vcache row and the per-request usage
	// record (#10670), so the two join exactly instead of by a fragile
	// nearest-millisecond match.
	nowMillis := time.Now().UnixMilli()
	s.metrics.observeVCacheTurn(traceID, nowMillis,
		uncachedPrompt, cacheRead, usage.CacheCreationInputTokens)
	// The stable per-request usage record (#10670): exactly one row per
	// completed request, after the native plane above so its join receipts exist.
	usageRec, usageRecOK := s.metrics.recordUsageTurn(traceID, wire, stream, usage, nowMillis)
	// The billing-grade per-request usage ledger (#14451): one durable
	// hash-chained row per request, keyed on the authenticated principal when
	// the route carried one. Purely observational: a nil ledger or an append
	// failure never touches the request path.
	s.appendUsageLedgerRow(traceID, wire, model, usage, nowMillis, finishReason, principal)
	// Roll the per-session managed-context record (ctxvalue.go) on the same always-on
	// rung: every served turn, all wires, before any sink gating, so the long-session
	// context report is answerable even with --log and --debug-stats off.
	s.observeCtxValue(traceID, uncachedPrompt, cacheRead, usage.CacheCreationInputTokens,
		usage.CompletionTokens, contextEvent)
	// The per-turn human debug render (#793) fires independently of the JSON --log sink, so
	// --debug-stats works on a clean (--log off) terminal. It is a no-op unless debugStatsf is
	// wired, and reuses the #792 rolling health (read-only peek; no double-roll).
	if s.debugStatsf != nil {
		s.renderTurnDebugStats(traceID, wire, stream, finishReason,
			uncachedPrompt, usage.CompletionTokens, cacheRead, usage.CacheCreationInputTokens, compacted)
	}
	if s.logf == nil {
		return
	}
	ev := map[string]any{
		"event":                       "gateway_inference_turn",
		"wire":                        wire,
		"stream":                      stream,
		"model":                       model,
		"finish_reason":               finishReason,
		"duration_ms":                 float64(dur.Microseconds()) / 1000.0,
		"prompt_tokens":               usage.PromptTokens,
		"completion_tokens":           usage.CompletionTokens,
		"cached_prompt_tokens":        usage.CachedPromptTokens(),
		"cache_read_input_tokens":     usage.CacheReadInputTokens,
		"cache_creation_input_tokens": usage.CacheCreationInputTokens,
		"cache_creation_span":         cacheCreationSpanLabel(usage.CacheCreationInputTokens, s.ttl1hActiveFor(traceID), s.ttl1hMessagePrefixFor(traceID)),
		"total_tokens":                usage.TotalTokens,
		"compaction_fired":            compacted,
		"context_event":               contextEvent,
	}
	if trace := strings.TrimSpace(traceID); trace != "" {
		ev["trace_id"] = trace
		// Attribute the served turn to the tenant principal that owns the trace — the
		// keyset-bound org/project (#5332) — so the access log is joinable to
		// /v1/fak/events by trace_id. Emitted only for a NAMED owner; the single-tenant
		// loopback ("" owner / unbound trace) leaves the line's shape unchanged.
		if owner, ok := s.traceOwnerOf(trace); ok && owner != "" {
			ev["principal"] = owner
		}
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
	// The sibling per-request usage-record event (#10670) — same sink, same
	// trace attribution, emitted exactly once per completed request so a log
	// reader pairs the two lines by trace_id + request_ordinal.
	if usageRecOK {
		s.emitUsageRecordEvent(traceID, usageRec)
	}
}
