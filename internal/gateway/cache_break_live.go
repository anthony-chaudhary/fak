package gateway

import (
	"net/http"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// cache_break_live.go — the LIVE producer for the mid-conversation cache-break detector
// (#2847, Track C of epic #2834). The pure detector (internal/metrics/cache_break_detector.go)
// and the consumer sink (#2916, gatewayMetrics.recordCacheBreak) both shipped; this file is
// the call site between them, the one the closing commit fenced off.
//
// Hermes' "Prompt Caching Must Not Break" invariant is a review convention. fak sits in front
// of the wire and sees every turn, so the invariant can be STRUCTURAL: hash the turn's stable
// prefix (system prompt + tool schema + the already-sent history head), compare it against the
// session's established prefix, and — when it diverges — witness "cache broken here, +N tokens"
// and price the induced cache_creation. This runs on the Anthropic /v1/messages front door, so
// ANY harness (Claude Code, an SDK, a relay) gets the protection without cooperating.
//
// The lever is a closed deny/warn/off policy (Config.CacheBreak):
//   - off (default): the detector is never consulted; an unarmed gateway pays nothing.
//   - warn: a detected break is witnessed and PRICED, and the turn proceeds — the mutated
//     prefix reached the wire, so its cost is real and folds into the #2916 sink.
//   - deny: a detected break REFUSES the turn before it is forwarded (409 CACHE_BREAK), so the
//     warm prefix survives and the cost is AVOIDED rather than incurred. The detector keeps its
//     established baseline, so a caller that retries with the original prefix is clean again.
//
// The detector state is per session trace, content-free (three digests and a length — never
// prompt bytes), and bounded by the same generational cap the gateway's other per-session maps
// use. Compaction and the other fak-authored transforms run AFTER this observation and compare
// against the INBOUND prefix, so a sanctioned compaction rewrite is not misread as a mutation.
//
// Generation intent: gen/next foundation (the same classification its siblings cache_break.go
// and cache_break_detector.go carry). It advances toward "now" the moment a live serve path
// arms it and the priced CostTokens is corroborated against the provider-relayed
// cache_creation (internal/metrics/provider_cache.go).

// maxCacheBreakSessions bounds the per-trace detector table, matching the gateway's other
// per-session maps (maxResetHealthSessions / maxCoherenceSessions / maxCtxValueSessions). A
// long-lived `fak serve` must not grow detector state without bound; a busy-but-healthy gateway
// never reaches the cap, so only the long-idle tail is reclaimed (its prefix re-primes cold).
const maxCacheBreakSessions = 8192

// parseCacheBreakPolicyEnv resolves the cache-break lever from the Config value, letting the
// FAK_CACHE_BREAK env override (the fleet/A-B arm, mirroring FAK_ABLATE_PREFIX_GUARD). An
// unrecognized name folds to off via ParseCacheBreakPolicy, so a typo can never arm a denying
// gate. Keeping the env read here (not in metrics) leaves the pure detector env-free.
func parseCacheBreakPolicyEnv(cfg string) metrics.CacheBreakPolicy {
	raw := cfg
	if env := strings.TrimSpace(os.Getenv("FAK_CACHE_BREAK")); env != "" {
		raw = env
	}
	return metrics.ParseCacheBreakPolicy(strings.TrimSpace(raw))
}

// cacheBreakDetectorFor returns the session trace's detector, minting one under the configured
// policy on first use. Caller must hold s.cacheBreakMu. A generational reset (the same pattern
// resetHealth and ctxValue use) reclaims the whole table at the cap rather than tracking LRU.
func (s *Server) cacheBreakDetectorForLocked(trace string) *metrics.CacheBreakDetector {
	if s.cacheBreakDetectors == nil {
		s.cacheBreakDetectors = make(map[string]*metrics.CacheBreakDetector)
	}
	if d, ok := s.cacheBreakDetectors[trace]; ok {
		return d
	}
	if len(s.cacheBreakDetectors) >= maxCacheBreakSessions {
		s.cacheBreakDetectors = make(map[string]*metrics.CacheBreakDetector)
	}
	d := metrics.NewCacheBreakDetector(s.cacheBreakPolicy)
	s.cacheBreakDetectors[trace] = d
	return d
}

// cacheBreakTurnPrefix projects the inbound Anthropic request onto the detector's wire-order
// stable prefix: system prompt, then the serialized tool schema, then the already-sent history
// head. Token lengths are the caller's byte/4 estimates (EstimateTokens), so the induced
// cache_creation is priced from a real span rather than zero.
//
// The history head EXCLUDES the leading system message (it is already the System component) and
// serializes each message as a newline-terminated record. That shape is load-bearing: appending
// a turn extends the serialization as a byte-prefix of the previous one, which is exactly the
// property the detector's false-positive guard reads — a normal conversation appends, and an
// append must never be witnessed as a break.
func cacheBreakTurnPrefix(req *agent.AnthropicMessagesRequest) metrics.TurnPrefix {
	if req == nil {
		return metrics.TurnPrefix{}
	}
	system := req.System
	tools := serializeCacheBreakTools(req.Tools)
	history := serializeCacheBreakHistory(req, system)
	return metrics.TurnPrefix{
		System:        system,
		Tools:         tools,
		HistoryHead:   history,
		SystemTokens:  int64(EstimateTokens(system)),
		ToolsTokens:   int64(EstimateTokens(tools)),
		HistoryTokens: int64(EstimateTokens(history)),
	}
}

// serializeCacheBreakTools renders the tool schema deterministically in REQUEST order. Order is
// deliberately significant (a reordered schema is a real provider cache break, not a no-op), so
// this does NOT sort — it mirrors what the provider actually sees. Every field that can move a
// schema's bytes is included and each field is delimited so a byte shifting across a field
// boundary cannot forge an unchanged digest.
func serializeCacheBreakTools(tools []agent.ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	var b strings.Builder
	for _, t := range tools {
		b.WriteString(t.Type)
		b.WriteByte('|')
		b.WriteString(t.Function.Name)
		b.WriteByte('|')
		b.WriteString(t.Function.Description)
		b.WriteByte('|')
		b.Write(t.Function.Parameters)
		b.WriteByte('\n')
	}
	return b.String()
}

// serializeCacheBreakHistory renders the already-sent turns as newline-terminated records,
// skipping the leading system message the decoder prepends (its content is the System
// component). Tool calls and tool results are folded in so a rewritten call/result is a real
// altered_turn, not an invisible no-op.
func serializeCacheBreakHistory(req *agent.AnthropicMessagesRequest, system string) string {
	if req == nil || len(req.Messages) == 0 {
		return ""
	}
	start := 0
	if req.Messages[0].Role == agent.RoleSystem && req.Messages[0].Content == system {
		start = 1
	}
	var b strings.Builder
	for _, m := range req.Messages[start:] {
		b.WriteString(m.Role)
		b.WriteByte('|')
		b.WriteString(m.Content)
		for _, tc := range m.ToolCalls {
			b.WriteString("|call:")
			b.WriteString(tc.ID)
			b.WriteByte(':')
			b.WriteString(tc.Function.Name)
			b.WriteByte(':')
			b.WriteString(tc.Function.Arguments)
		}
		if m.ToolCallID != "" {
			b.WriteString("|result:")
			b.WriteString(m.ToolCallID)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// observeCacheBreak scores one inbound turn against the session's established prefix and folds
// the outcome: a warn-mode break is recorded on the #2916 sink; a deny-mode break returns
// denied=true so the caller refuses the turn before it reaches the wire. Nil-safe (a Server with
// no metrics, or the lever off, is a no-op) and lock-scoped to the per-session detector, which
// is not itself safe for concurrent use.
func (s *Server) observeCacheBreak(trace string, req *agent.AnthropicMessagesRequest) (metrics.CacheBreakVerdict, bool) {
	if s == nil || s.cacheBreakPolicy == metrics.CacheBreakPolicyOff {
		return metrics.CacheBreakVerdict{}, false
	}
	// The detector is not safe for concurrent use, so the score happens under the map lock;
	// the side effects (the metrics sink's own lock, and a log write) are deferred until after
	// it is released, keeping this lock free of nested locking and I/O.
	v := func() metrics.CacheBreakVerdict {
		s.cacheBreakMu.Lock()
		defer s.cacheBreakMu.Unlock()
		return s.cacheBreakDetectorForLocked(trace).Observe(cacheBreakTurnPrefix(req))
	}()
	if !v.Broken {
		return v, false
	}
	// The detector folded this break internally (Report/Avoided); mirror the INCURRED half onto
	// the gateway's per-session counter so /metrics and the guard exit summary agree. A denied
	// break is deliberately NOT recorded here: it never reached the wire, so it is avoided cost.
	if !v.Denied {
		s.metrics.recordCacheBreak(v.Event.Cause, v.Event.CostTokens)
	}
	if s.logf != nil {
		s.logf("gateway: %s (trace=%s)", v.Witness, trace)
	}
	return v, v.Denied
}

// cacheBreakRefusal writes the deny-mode refusal. 409 Conflict: the request is well-formed but
// conflicts with the session's established cache prefix; the client can retry with the original
// prefix (or a fresh trace) to proceed. The witness carries the measured induced cost.
func cacheBreakRefusal(w http.ResponseWriter, v metrics.CacheBreakVerdict) {
	writeErrCode(w, http.StatusConflict, "CACHE_BREAK", "cache break refused: "+v.Witness)
}
