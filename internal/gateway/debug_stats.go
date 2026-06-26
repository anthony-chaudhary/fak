package gateway

// debug_stats.go -- the per-turn cache & compaction debug render (#793). `fak guard
// --debug-stats` / `fak serve --debug-stats` wire Config.DebugStatsf to stderr; then every
// served turn prints ONE compact, payload-free line so an operator can watch turn-by-turn what
// the field otherwise reconstructs from a JSON --log: the request/cache token split, the
// cache-read rebate estimate, the compaction action, and the resetScore SHADOW health state.
//
// It REUSES the #792 per-session rolling health (reset_shadow.go): peekResetHealth scores the
// CURRENT rolling state WITHOUT mutating it (the roll happened on the compacted turn via
// observeResetHealth), so the debug line reports the same five-state verdict the metric does
// (healthy_cache / cache_decay / stale_prefix / cooldown / unknown_provider) and never
// double-counts. It is read-only and content-free: only counts, ratios, and the closed reason
// token are ever emitted, never a prompt byte.

import (
	"fmt"
	"strconv"
	"strings"
)

// peekResetHealth scores a session's CURRENT rolling compaction-health WITHOUT mutating it or
// minting a record. ok is false when the session has no rolling health yet (it has never been
// compacted), so the debug render reports health=n/a rather than a phantom unknown_provider.
func (s *Server) peekResetHealth(trace string) (ResetDecision, bool) {
	if s == nil || trace == "" {
		return ResetDecision{}, false
	}
	s.resetHealthMu.Lock()
	h := s.resetHealth[trace] // direct read: peeking must NOT create a record
	if h == nil {
		s.resetHealthMu.Unlock()
		return ResetDecision{}, false
	}
	st := h.state()
	s.resetHealthMu.Unlock()
	return DefaultResetPolicy().ResetScore(st), true
}

// renderTurnDebugStats emits one per-turn debug line when the sink is wired. It is the only
// caller of the formatter, and gates on debugStatsf so it is a byte-identical no-op when
// --debug-stats is off.
func (s *Server) renderTurnDebugStats(trace, wire string, stream bool, finish string, prompt, completion, cacheRead, cacheCreate int, compacted bool) {
	if s == nil || s.debugStatsf == nil {
		return
	}
	d, have := s.peekResetHealth(trace)
	s.debugStatsf("%s", formatTurnDebugStats(trace, wire, stream, finish, prompt, completion, cacheRead, cacheCreate, compacted, d, have))
}

// formatTurnDebugStats renders one served turn as a compact, payload-free key=val line. It is
// pure (no I/O, no state) so the format is unit-tested directly. health distinguishes the five
// resetScore states; when the session has no rolling health yet (have=false) it reads n/a and
// the score/recommend columns are inert dashes.
func formatTurnDebugStats(trace, wire string, stream bool, finish string, prompt, completion, cacheRead, cacheCreate int, compacted bool, d ResetDecision, have bool) string {
	if finish == "" {
		finish = "unknown"
	}
	compact := "none"
	if compacted {
		compact = "fired"
	}
	health, score, recommend := "n/a", "-", "-"
	if have {
		health = string(d.Reason)
		score = strconv.FormatFloat(d.Score, 'f', 2, 64)
		recommend = "no"
		if d.ShouldReset {
			recommend = "yes"
		}
	}
	// cache_hit/cache_rebate_tokens are OBSERVED/provider-counter projections. The hit ratio
	// shows whether the cached prefix is still landing; the rebate is token-equivalent savings
	// from cache reads at the published 0.1x read multiplier, before any model-specific $/MTok.
	requestTokens := cacheRead + prompt + cacheCreate
	hit := "0.00"
	if requestTokens > 0 {
		hit = strconv.FormatFloat(float64(cacheRead)/float64(requestTokens), 'f', 2, 64)
	}
	rebate := strconv.FormatFloat(float64(cacheRead)*(1-CacheReadMultiplier), 'f', 1, 64)
	var b strings.Builder
	fmt.Fprintf(&b, "fak-turn trace=%s wire=%s stream=%s finish=%s", debugField(trace), debugField(wire), boolFlag(stream), debugField(finish))
	fmt.Fprintf(&b, " request_tokens=%d prompt=%d completion=%d cache_read=%d cache_creation=%d cache_hit=%s cache_rebate_tokens=%s", requestTokens, prompt, completion, cacheRead, cacheCreate, hit, rebate)
	fmt.Fprintf(&b, " compact=%s health=%s reset_score=%s recommend=%s", compact, health, score, recommend)
	return b.String()
}

// debugField sanitizes a label for the single-line debug render: it can never carry prompt
// content (trace/wire/finish are kernel-minted tokens), but flattening whitespace keeps the line
// one row and parseable even if a caller passes an unexpected value. Empty renders as "-".
func debugField(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "-"
	}
	return strings.NewReplacer(" ", "_", "\t", "_", "\n", "_", "\r", "_").Replace(s)
}

func boolFlag(v bool) string {
	if v {
		return "1"
	}
	return "0"
}
