package gateway

// debug_stats.go -- the per-turn debug render (#793). `fak guard --debug-stats` /
// `fak serve --debug-stats` wire Config.DebugStatsf to stderr; then every served turn prints
// ONE compact, payload-free line whose FIRST job is to answer "did this turn work?" at a
// glance, not to dump the provider's raw usage counters.
//
// What it shows (and why):
//   - a one-word VERDICT (ok / warming / degraded / cold) — the glanceable "is fak working".
//   - the provider NET token saving this turn (prov=…) plus the explicit fak slice
//     (fak=0 until this call site has a per-turn fak-authored token witness) — the
//     write-premium-aware number computed by the SAME engine /metrics (fak_vcache_*) and
//     `fak vcache observe` use (vcacheProofFromCounters), so the human line can never
//     disagree with the scrape. A cold-write turn reads NEGATIVE until later reads repay
//     the write, where the old read-only rebate would have overstated it.
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
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// turnSafetyDelta is ONE turn's adjudication SAFETY outcome — the calls the kernel BLOCKED and
// REPAIRED, and the inbound results it QUARANTINED, on that turn alone. It is the safety half of
// the per-turn fak-turn debug line, the counterpart to the cache/token VALUE the line already
// shows. topReason carries the dominant deny/quarantine reason (a closed-vocabulary token, e.g.
// POLICY_BLOCK) so the glanceable line can say WHY, not just how many. It is a per-turn delta, not
// a cumulative — recordTurnSafety stores exactly this turn's counts and takeTurnSafety clears them
// on read, so two deny turns each render blocked=1, never 1 then 2.
type turnSafetyDelta struct {
	blocked     int
	repaired    int
	quarantined int
	topReason   string
}

// any reports whether the delta has anything worth rendering — a clean ALLOW-everything turn has
// none of the three, so the fak-turn line stays byte-identical to a value-only turn.
func (d turnSafetyDelta) any() bool { return d.blocked > 0 || d.repaired > 0 || d.quarantined > 0 }

// foldTurnSafety derives a turn's safety delta from the SAME per-turn slices the proxy already
// computes — the proposed-call adjudications (adjs) and the inbound result admissions (results) —
// so the live line can never disagree with the in-band [fak] note or the /metrics counters that
// read the same verdicts. A non-admitted call is a BLOCK; a TRANSFORM is a REPAIR; a QUARANTINE
// verdict on an inbound result is a quarantine. topReason is the first blocked call's reason, else
// the first quarantine's, so the glanceable line names the most action-relevant cause.
func foldTurnSafety(adjs []ToolAdjudication, results []ResultAdmission) turnSafetyDelta {
	var d turnSafetyDelta
	for _, a := range adjs {
		switch {
		case !a.Admitted:
			d.blocked++
			if d.topReason == "" && a.Verdict.Reason != "" {
				d.topReason = a.Verdict.Reason
			}
		case a.Verdict.Kind == "TRANSFORM":
			d.repaired++
		}
	}
	for _, r := range results {
		if r.Verdict.Kind == "QUARANTINE" {
			d.quarantined++
			if d.topReason == "" && r.Verdict.Reason != "" {
				d.topReason = r.Verdict.Reason
			}
		}
	}
	return d
}

// recordTurnSafety stashes a turn's safety delta under its trace so the per-turn debug render can
// fold it into the same line that already carries the cache/token value. It is a no-op for an
// empty delta (a clean turn leaves no stash, so the render finds nothing and the line stays
// value-only) and bounds the map with the same reaper resetHealth uses, so an unbounded stream of
// distinct traces cannot grow it without limit. Called on both proxy paths right after the turn's
// calls/results are adjudicated, BEFORE the terminal inference observation that triggers the render.
func (s *Server) recordTurnSafety(trace string, adjs []ToolAdjudication, results []ResultAdmission) {
	if s == nil || trace == "" {
		return
	}
	d := foldTurnSafety(adjs, results)
	if !d.any() {
		return
	}
	s.turnSafetyMu.Lock()
	if s.turnSafety == nil {
		s.turnSafety = map[string]turnSafetyDelta{}
	}
	if len(s.turnSafety) >= maxResetHealthSessions {
		// Bound the stash: drop an arbitrary stale entry. The render clears on read, so this only
		// trips when many traces blocked something without a following rendered turn — a corner.
		for k := range s.turnSafety {
			delete(s.turnSafety, k)
			break
		}
	}
	s.turnSafety[trace] = d
	s.turnSafetyMu.Unlock()
}

// takeTurnSafety returns and CLEARS a trace's stashed safety delta. Clearing on read is what makes
// the rendered line a per-turn delta: the next turn starts from nothing, so a one-off block shows
// on exactly one line. The zero delta (any()==false) means this turn had no safety action.
func (s *Server) takeTurnSafety(trace string) turnSafetyDelta {
	if s == nil || trace == "" {
		return turnSafetyDelta{}
	}
	s.turnSafetyMu.Lock()
	d := s.turnSafety[trace]
	delete(s.turnSafety, trace)
	s.turnSafetyMu.Unlock()
	return d
}

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
	// Take (and clear) this turn's safety delta so the line carries the blocked/repaired/
	// quarantined half alongside the cache/token value. Clearing keeps it per-turn.
	safety := s.takeTurnSafety(trace)
	s.debugStatsf("%s", formatTurnDebugStats(trace, wire, stream, finish, prompt, completion, cacheRead, cacheCreate, compacted, d, have, safety))
}

// renderTurnDebugError emits one per-turn debug line on a FAILED turn — the missing half of the
// observable layer. renderTurnDebugStats only fires on a SUCCESSFUL completion, so without this
// a stalled / errored / fell-back turn produces no output on the default `--debug-stats` stderr
// line: the operator sees a frozen terminal with no signal. This makes the failure VISIBLE on the
// same line they already watch (no `--log` needed): `fak-turn trace=… FAILED reason=stalled
// wire=anthropic_messages after=61s`. Gated on debugStatsf so it is a no-op when --debug-stats is
// off. reason is classified from the planner error by debugErrorReason (the same closed kinds the
// upstream-error counter uses). elapsed is the wall-clock the turn ran before it failed.
func (s *Server) renderTurnDebugError(trace, wire string, err error, elapsed time.Duration) {
	if s == nil || s.debugStatsf == nil {
		return
	}
	s.debugStatsf("%s", formatTurnDebugError(trace, wire, debugErrorReason(err), elapsed))
}

// debugErrorReason maps a planner/proxy error to the closed reason token the FAILED debug line
// shows. It reuses upstreamErrorKind (the classifier the /metrics counter shares) so the line and
// the counter agree, then collapses the generic status kinds into one "status" token for the
// glanceable line — while keeping the operationally-distinct 4xx conditions (a rate limit, an auth
// failure, a permission denial) as their own glanceable tokens, because "we are being rate-limited"
// and "our credential is bad" call for very different operator action and should not both read
// "status". A nil error reads "error" (a failure with no typed cause — e.g. a stream that opened
// then produced no events).
func debugErrorReason(err error) string {
	switch upstreamErrorKind(err) {
	case "stalled":
		return "stalled"
	case "unreachable":
		return "unreachable"
	case "rate_limited":
		return "rate_limited"
	case "auth":
		return "auth"
	case "forbidden":
		return "forbidden"
	case "status_4xx", "status_5xx":
		return "status"
	case "oom":
		return "oom"
	default:
		return "error"
	}
}

// formatTurnDebugError renders one FAILED turn as a compact, payload-free line that LEADS with the
// failure verdict and its reason — the mirror of formatTurnDebugStats's success line. Pure (no I/O,
// no state) so the format is unit-tested directly. after= is rounded to whole seconds: the operator
// cares "it failed after ~a minute" (a stall) vs "instantly" (an unreachable/4xx), not millis.
func formatTurnDebugError(trace, wire, reason string, elapsed time.Duration) string {
	if reason == "" {
		reason = "error"
	}
	return fmt.Sprintf("fak-turn trace=%s FAILED reason=%s wire=%s after=%ds",
		debugField(trace), debugField(reason), debugField(wire), int(elapsed.Round(time.Second).Seconds()))
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
func formatTurnDebugStats(trace, wire string, stream bool, finish string, prompt, completion, cacheRead, cacheCreate int, compacted bool, d ResetDecision, have bool, safetyOpt ...turnSafetyDelta) string {
	if finish == "" {
		finish = "unknown"
	}
	var safety turnSafetyDelta
	if len(safetyOpt) > 0 {
		safety = safetyOpt[0]
	}
	compact := "none"
	if compacted {
		compact = "fired"
	}
	health := "n/a"
	if have {
		health = string(d.Reason)
	}
	// The provider-owned NET token saving — write-premium-aware — from the SAME engine
	// /metrics and `fak vcache observe` use. prompt is the uncached input remainder
	// (TelemetryRow.InputTokens); cacheRead/cacheCreate are the provider read/write axes.
	// A cold-write turn comes back REFUTED with a negative saving (honest), where a
	// read-only rebate would have overstated it. The per-turn line names this as `prov=`
	// and keeps `fak=0` until a per-turn fak-authored token-saving witness is available,
	// so the provider prompt cache cannot masquerade as a fak-authored cache win.
	proof := vcacheProofFromCounters(uint64(maxNonNeg(prompt)), uint64(maxNonNeg(cacheRead)), uint64(maxNonNeg(cacheCreate)))
	verdict := turnVerdict(proof, d, have, cacheRead, cacheCreate)

	var b strings.Builder
	fmt.Fprintf(&b, "fak-turn trace=%s %s", debugField(trace), verdict)
	// prov=<net token-equiv> with the % of the uncached baseline it represents. The fak
	// per-turn slice is explicitly 0 because this call site has no per-turn shed/KV witness;
	// session-level attribution folds those in from their own counters.
	if cacheRead > 0 || cacheCreate > 0 {
		fmt.Fprintf(&b, " prov=%s tok (%s%% of prompt) fak=0 tok", HumanTokenEquiv(proof.SavedTokenEquiv), strconv.FormatFloat(proof.SavedPct, 'f', 0, 64))
	} else {
		b.WriteString(" prov=0 tok fak=0 tok")
	}
	fmt.Fprintf(&b, " cache=%s compact=%s finish=%s", health, compact, debugField(finish))
	if safety.any() {
		fmt.Fprintf(&b, " safety=blocked:%d repaired:%d quarantined:%d", safety.blocked, safety.repaired, safety.quarantined)
		if safety.topReason != "" {
			fmt.Fprintf(&b, " reason=%s", debugField(safety.topReason))
		}
	}
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
