package gateway

// debug_stats.go -- the per-turn debug render (#793). `fak guard --debug-stats` /
// `fak serve --debug-stats` wire Config.DebugStatsf to stderr; then every served turn prints
// ONE compact, payload-free line whose FIRST job is to answer "did this turn work?" at a
// glance, not to dump the provider's raw usage counters.
//
// What it shows (and why):
//   - a one-word VERDICT (ok / warming / degraded / cold) — the glanceable "is fak working".
//   - the NET token saving this turn (saved=…) — the write-premium-aware number, computed by
//     the SAME engine /metrics (fak_vcache_*) and `fak vcache observe` use
//     (vcacheProofFromCounters), so the human line can never disagree with the scrape. This
//     is the honest fak-vs-no-cache value: a cold-write turn reads NEGATIVE until the reads
//     repay the write, where the old read-only rebate would have overstated it.
//   - the rolling compaction HEALTH and the compaction action.
//
// What it deliberately OMITS: the provider's own raw usage counters (cache_read /
// cache_creation / request_tokens / cache_hit). Those measure Anthropic's cache, not whether
// fak is doing its job, so they are noise on a glanceable line — they remain available for
// deep debugging in the JSON --log and on /metrics.
//
// It REUSES the #792 per-session rolling health (reset_shadow.go): peekResetHealth scores the
// CURRENT rolling state WITHOUT mutating it (the roll happened on the compacted turn via
// observeResetHealth), so the debug line reports the same five-state verdict the metric does
// (healthy_cache / cache_decay / stale_prefix / cooldown / unknown_provider) and never
// double-counts. It is read-only and content-free: only counts, ratios, and closed tokens are
// ever emitted, never a prompt byte.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
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

// formatTurnDebugStats renders one served turn as a compact, payload-free line that LEADS with
// the answer to "did this turn work?". It is pure (no I/O, no state) so the format is
// unit-tested directly. The verdict and saving are derived; the raw provider counters (prompt,
// completion, cache_read, cache_creation) are inputs to the math but never printed — they live
// in the JSON --log and on /metrics for deep debugging. health distinguishes the five
// resetScore states; when the session has no rolling health yet (have=false) it reads n/a.
//
// wire/stream/completion are retained in the signature (the JSON --log and callers carry them)
// but are intentionally not rendered on this glanceable line.
func formatTurnDebugStats(trace, wire string, stream bool, finish string, prompt, completion, cacheRead, cacheCreate int, compacted bool, d ResetDecision, have bool) string {
	if finish == "" {
		finish = "unknown"
	}
	compact := "none"
	if compacted {
		compact = "fired"
	}
	health := "n/a"
	if have {
		health = string(d.Reason)
	}
	// The NET token saving — write-premium-aware — from the SAME engine /metrics and
	// `fak vcache observe` use. prompt is the uncached input remainder (TelemetryRow.InputTokens);
	// cacheRead/cacheCreate are the provider read/write axes. A cold-write turn comes back REFUTED
	// with a negative saving (honest), where a read-only rebate would have overstated it.
	proof := vcacheProofFromCounters(uint64(maxNonNeg(prompt)), uint64(maxNonNeg(cacheRead)), uint64(maxNonNeg(cacheCreate)))
	verdict := turnVerdict(proof, d, have, cacheRead, cacheCreate)

	var b strings.Builder
	fmt.Fprintf(&b, "fak-turn trace=%s %s", debugField(trace), verdict)
	// saved=<net token-equiv> with the % of the uncached baseline it represents. Only meaningful
	// when the turn had cache activity; a cold turn (no read, no write) shows saved=0.
	if cacheRead > 0 || cacheCreate > 0 {
		fmt.Fprintf(&b, " saved=%s tok (%s%% of prompt)", HumanTokenEquiv(proof.SavedTokenEquiv), strconv.FormatFloat(proof.SavedPct, 'f', 0, 64))
	} else {
		b.WriteString(" saved=0 tok")
	}
	fmt.Fprintf(&b, " cache=%s compact=%s finish=%s", health, compact, debugField(finish))
	return b.String()
}

// turnVerdict folds the net-saving proof, the rolling reset health, and the cache activity into
// one glanceable word:
//   - cold     — no provider cache activity at all this turn (a first turn, or a non-cached path).
//   - degraded — the rolling health says the prefix is decaying / stale, or reset is recommended.
//   - ok       — a proven net saving on a healthy (or not-yet-scored) session.
//   - warming  — cache activity but no net saving yet (a cold write the later reads haven't repaid).
func turnVerdict(proof vcachegov.TelemetrySavingsProof, d ResetDecision, have bool, cacheRead, cacheCreate int) string {
	if cacheRead <= 0 && cacheCreate <= 0 {
		return "cold"
	}
	if have && (d.ShouldReset || d.Reason == ResetReasonStalePrefix || d.Reason == ResetReasonDecay) {
		return "degraded"
	}
	if proof.Status == vcachegov.ProofProven && proof.SavedTokenEquiv > 0 {
		return "ok"
	}
	return "warming"
}

// HumanTokenEquiv renders a token-equivalent count compactly and signed: 20712 -> "20.7k",
// -612 -> "-612", 1_250_000 -> "1.2M". It keeps the glanceable debug surfaces short without
// losing the sign that distinguishes a real saving from an unrepaid write premium. Exported so
// the `fak guard` exit summary formats the session saving identically to the per-turn line.
func HumanTokenEquiv(v float64) string {
	neg := v < 0
	a := v
	if neg {
		a = -a
	}
	var s string
	switch {
	case a >= 1_000_000:
		s = strconv.FormatFloat(a/1_000_000, 'f', 1, 64) + "M"
	case a >= 1_000:
		s = strconv.FormatFloat(a/1_000, 'f', 1, 64) + "k"
	default:
		s = strconv.FormatFloat(a, 'f', 0, 64)
	}
	if neg {
		return "-" + s
	}
	return s
}

// maxNonNeg clamps a possibly-negative count to 0 before the uint64 conversion the proof engine
// needs, so a planner that omits a count never wraps into a huge positive.
func maxNonNeg(n int) int {
	if n < 0 {
		return 0
	}
	return n
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
