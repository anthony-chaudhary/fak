package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compactcohere"
)

// harness coherence is the gateway seam (#1132, child of #1131) that feeds the shipped
// compactcohere decision surface. On the Anthropic passthrough TWO context/cache managers ride
// the same wire blind to each other: fak (cache-PRESERVING byte-splice compaction that forwards
// the inbound cache_control prefix verbatim) and the Claude Code harness (cache-DESTROYING
// auto-compaction that rewrites its own messages[] near the context window, bursting the provider
// cache). This file captures a CONTENT-FREE digest of the inbound protected prefix BEFORE fak's
// request-side transforms, builds a compactcohere.TurnObservation per served turn, drives a
// per-trace compactcohere.Coordinator, and folds the verdict into a fak_harness_coherence_*
// metric family. The accumulators here are the SINGLE source the /metrics view and the operator
// line (#1135) both read, so the two surfaces can never disagree.

// inboundProtectedPrefixDigest returns a CONTENT-FREE digest of the inbound protected prefix —
// the bytes of the raw Anthropic /v1/messages body from the start through the FIRST cache_control
// breakpoint (the stable cached HEAD the provider reuses every turn). It hashes those bytes with
// SHA-256 and returns the hex digest; the prompt bytes themselves never leave this function, so a
// shadow log of the digest carries no content. Taken BEFORE fak's own request-side transforms
// (maybePlanAnthropicRaw / maybeCompactAnthropicRaw / maybeElide / maybeCompactInboundTools), which
// is the load-bearing invariant: fak forwards the inbound protected prefix VERBATIM, so a change in
// this digest between two turns can only be the harness rewriting its own history — never fak.
//
// An empty string means "no digest" (an empty body, or no cache_control breakpoint to anchor the
// protected head — a first-turn-shaped body with no stable cached prefix). Classify treats an empty
// digest as "unknown", so a turn with no anchor never spuriously reads as a harness rewrite.
func inboundProtectedPrefixDigest(raw []byte) string {
	if len(raw) == 0 {
		return ""
	}
	// The protected prefix runs from the start through the END of the first cache_control marker
	// object. We anchor on the literal cache_control key the Anthropic wire uses; everything up to
	// and including the close brace of the breakpoint object is the cached head. If no breakpoint is
	// present there is no stable cached prefix to protect, so there is nothing to digest.
	marker := []byte("cache_control")
	idx := bytes.Index(raw, marker)
	if idx < 0 {
		return ""
	}
	// Extend through the close of the breakpoint object so the digest covers the whole cached head,
	// not an arbitrary mid-object cut. cache_control values on the Anthropic wire are small objects
	// (e.g. {"type":"ephemeral"}); scan to the next '}' after the marker. A malformed body with no
	// closing brace falls back to the marker end — still a deterministic, content-free cut.
	end := idx + len(marker)
	if close := bytes.IndexByte(raw[end:], '}'); close >= 0 {
		end = end + close + 1
	}
	sum := sha256.Sum256(raw[:end])
	return hex.EncodeToString(sum[:])
}

// fakBailReasonFor maps a compaction CompactOutcome.Reason onto the compactcohere TurnObservation
// FakBailReason field, per #1132: "" (a clean fire) and "under_budget" (a healthy no-op — nothing
// to shed is not a failure) BOTH map to "" (no bail); any OTHER reason (prefix_mismatch, cached_span,
// window_no_drop, splice_failed, redecode_failed, no_breakpoint, …) is a real labeled bail and is
// carried through verbatim. A real bail is what the coordinator's yield-streak counts: fak wanted to
// shed tokens but could not, and a sustained streak is when the harness net is handed back.
func fakBailReasonFor(reason string) string {
	if reason == agent.CompactReasonNone || reason == agent.CompactReasonUnderBudget {
		return ""
	}
	return reason
}

// harnessCoherenceMetrics accumulates the per-trace coordinators and the cross-session counters the
// fak_harness_coherence_* family renders. It is the SINGLE source of truth: the /metrics scrape and
// the operator line (#1135) both fold these same numbers, so the two views can never disagree.
//
// One coordinator per trace (per served Claude Code session) carries the rolling, content-free
// prefix-event state; the counters below are the session-wide roll-up across every trace.
type harnessCoherenceMetrics struct {
	ttl time.Duration
	// residentCeiling / ceilingStreakToYield / nonHoldRewritesToReset are the resident-token
	// policy every per-trace coordinator is built with (#3158/#3159). residentCeiling is resolved
	// from FAK_CTX_YIELD_CEILING once at construction; the streaks use the package defaults.
	residentCeiling        int64
	ceilingStreakToYield   int
	nonHoldRewritesToReset int

	mu     sync.Mutex
	coords map[string]*coordEntry // trace -> per-session coordinator + last-turn wall clock

	// observedTurns is the denominator: served passthrough turns folded into a coordinator.
	observedTurns uint64
	// events counts each attributed compactcohere.PrefixEvent (stable|fak_cut|fak_world_break|
	// harness_rewrite|cold_ttl). harness_rewrite is the previously-invisible second-compactor event.
	events map[compactcohere.PrefixEvent]uint64
	// harnessRewrites / quarantineAtRisk / burstsObserved are the headline risk counts the family
	// surfaces: a harness rewrite bursts the provider cache; a quarantine-at-risk is a fak-sealed
	// span that may have been folded into the harness summary (the trust hole this policy surfaces);
	// a burst is any turn that (will) cost a provider cache_creation rebuild.
	harnessRewrites  uint64
	quarantineAtRisk uint64
	burstsObserved   uint64
	// prefixGuardTurns / prefixGuardStable / prefixGuardMutated / prefixGuardUnknown are
	// the prefix-determinism guard's witness counts (#2182, the runtime form of #1602/#1604):
	// served turns folded while the guard lever was armed, split by verdict — the inbound
	// protected-prefix digest matched the trace's baseline (stable), diverged from it
	// (mutated — the prefix burst the guard exists to catch before an operator relies on
	// provider-cache economics), or carried no breakpoint anchor to compare (unknown).
	// All stay 0 while the lever is off.
	prefixGuardTurns   uint64
	prefixGuardStable  uint64
	prefixGuardMutated uint64
	prefixGuardUnknown uint64
	// overCeilingTurns / rewriteNoDrops / recommendResets are the resident-token risk counts
	// (#3158/#3159): turns whose resident window exceeded the ceiling; harness rewrites judged
	// non-holding (window climbed back over the ceiling); turns recommending a hard reset
	// (cold_ttl OR the non-holding escalation). resetsFired counts coherence resets ACTUATED
	// through the ResetOnBudget seam (Phase C); it stays 0 when no host resetter is wired.
	overCeilingTurns uint64
	rewriteNoDrops   uint64
	recommendResets  uint64
	resetsFired      uint64
	// lastResident is the most recent turn's OBSERVED resident window size (input +
	// cache_creation + cache_read), surfaced as a gauge so an operator can watch it approach
	// the ceiling. Last-write-wins across traces (one session drives one trace).
	lastResident int64
	// posture is the CURRENT standing block/allow posture across all live traces (last-write-wins
	// per fold; a single Claude Code session drives one trace, so this is that session's stance).
	posture compactcohere.Posture
}

// coordEntry is one trace's coordinator plus the wall clock of its last served turn (for the idle
// gap the TTL signal needs) and the count of served turns folded so far (for the head-anchored
// session-length prior — see servedTurns).
type coordEntry struct {
	coord    *compactcohere.Coordinator
	lastTurn time.Time
	turns    uint64 // served passthrough turns folded into this trace so far
	// heldPeak is the largest OBSERVED resident window (input+cached, the coordinator's
	// ResidentTokens) this trace has ever reached — its demonstrated heaviness, the
	// held-volume signal the volume-aware head-anchored horizon reads (see headSessionPrior /
	// headHorizonHeavyResidentFloor). Peak (not last) so a single small turn cannot demote a
	// session that has proven it holds a large context; monotone within a trace's life.
	heldPeak int64
	// guardDigest is the prefix-determinism guard's last-accepted protected-prefix digest
	// for this trace (#2182). Written ONLY while the guard lever is armed (--prefix-guard /
	// FAK_ABLATE_PREFIX_GUARD=1), so an unarmed server carries zero extra state. Kept
	// separate from the coordinator's own rolling digest state so the guard's verdict is a
	// direct #1602-style baseline comparison, independent of the coordinator's richer
	// event attribution.
	guardDigest string
}

// maxCoherenceSessions bounds the per-trace coordinator map. observe mints one *coordEntry per
// served Claude Code session (trace) and, before this cap, nothing ever removed one — a long-lived
// `fak serve` grew coords without bound (monotonic heap growth, each entry holding a live
// *compactcohere.Coordinator with rolling state, #3450). The cap is far above any realistic live
// working set — it matches the gateway's other per-session maps (maxResetHealthSessions /
// maxCtxValueSessions / maxSessionPlanners, all 8192) — so a busy but healthy gateway never evicts
// a live session; it only reclaims the long-idle tail. Same unbounded-accumulation class the
// bounded A2A task store fixed, in the harness-coherence accumulator instead.
const maxCoherenceSessions = 8192

// evictColdestCoordLocked drops the coordinator whose last served turn is oldest — an active trace
// is re-stamped every turn (observe sets lastTurn), so only long-idle traces are ever candidates and
// a live session is never the victim. Ties break on trace id for determinism. The caller holds h.mu
// and coords is known non-empty. Mirrors session_activity.go's evictColdestLocked. Evicting a trace
// only costs it a fresh coordinator on its next turn (rolling prefix state re-primes), which for a
// trace idle past this many newer sessions is already effectively cold.
func (h *harnessCoherenceMetrics) evictColdestCoordLocked() {
	var victim string
	var best time.Time
	for id, e := range h.coords {
		if victim == "" || e.lastTurn.Before(best) || (e.lastTurn.Equal(best) && id < victim) {
			victim, best = id, e.lastTurn
		}
	}
	delete(h.coords, victim)
}

func newHarnessCoherenceMetrics(ttl time.Duration) *harnessCoherenceMetrics {
	if ttl <= 0 {
		ttl = compactcohere.DefaultProviderCacheTTL
	}
	return &harnessCoherenceMetrics{
		ttl:                    ttl,
		residentCeiling:        resolveCtxYieldCeiling(os.Getenv("FAK_CTX_YIELD_CEILING")),
		ceilingStreakToYield:   compactcohere.DefaultCeilingStreakToYield,
		nonHoldRewritesToReset: compactcohere.DefaultNonHoldRewritesToReset,
		coords:                 map[string]*coordEntry{},
		events:                 map[compactcohere.PrefixEvent]uint64{},
		posture:                compactcohere.PostureBlock,
	}
}

// resolveCtxYieldCeiling parses the FAK_CTX_YIELD_CEILING deployment knob (the resident-token
// high-water mark, #3158) from its raw env string. An empty, non-numeric, or non-positive value
// falls back to compactcohere.DefaultResidentCeiling (~160k) — the resident-token terms are always
// on with a sane default. An operator disables them by setting a very large ceiling the window
// never reaches. Pure (env read stays at the call site) so it is unit-tested without the env.
func resolveCtxYieldCeiling(raw string) int64 {
	v, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil || v <= 0 {
		return compactcohere.DefaultResidentCeiling
	}
	return v
}

// observe folds one served Anthropic passthrough turn into the trace's coordinator and updates the
// shared accumulators. trace keys the per-session coordinator; now is the served turn's wall clock
// (drives the idle-gap TTL signal). digest is the CONTENT-FREE inbound protected-prefix digest taken
// BEFORE fak's transforms; fakFired/fakBail describe fak's own compaction this turn (fakBail already
// mapped through fakBailReasonFor); sealed is whether fak sealed/quarantined a span in this turn's
// context; cacheRead/cacheCreate are the provider's OBSERVED counters, relayed verbatim. It returns
// the per-turn Decision so a caller (the debug line / a test) can read the verdict directly.
func (h *harnessCoherenceMetrics) observe(trace string, now time.Time, digest string, fakFired bool, fakBail string, fakWorldBreak, sealed bool, cacheRead, cacheCreate, inputTokens int64) compactcohere.Decision {
	if h == nil {
		return compactcohere.Decision{}
	}
	h.mu.Lock()
	defer h.mu.Unlock()

	entry := h.coords[trace]
	if entry == nil {
		if len(h.coords) >= maxCoherenceSessions {
			h.evictColdestCoordLocked()
		}
		entry = &coordEntry{coord: compactcohere.NewConfig(compactcohere.Config{
			TTL:                    h.ttl,
			ResidentCeiling:        h.residentCeiling,
			CeilingStreakToYield:   h.ceilingStreakToYield,
			NonHoldRewritesToReset: h.nonHoldRewritesToReset,
		})}
		h.coords[trace] = entry
	}
	var idle time.Duration
	if !entry.lastTurn.IsZero() {
		idle = now.Sub(entry.lastTurn)
	}
	entry.lastTurn = now
	entry.turns++

	obs := compactcohere.TurnObservation{
		InboundPrefixDigest: digest,
		FakCompactFired:     fakFired,
		FakBailReason:       fakBail,
		FakWorldBreak:       fakWorldBreak,
		SealedSpanPresent:   sealed,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheCreate,
		InputTokens:         inputTokens,
		IdleSinceLastTurn:   idle,
	}
	d := entry.coord.Observe(obs)

	h.observedTurns++
	h.events[d.Event]++
	if d.Event == compactcohere.EventHarnessRewrite {
		h.harnessRewrites++
	}
	if d.QuarantineAtRisk {
		h.quarantineAtRisk++
	}
	if d.BurstObserved {
		h.burstsObserved++
	}
	if d.OverCeiling {
		h.overCeilingTurns++
	}
	if d.RewriteNoDrop {
		h.rewriteNoDrops++
	}
	if d.Action == compactcohere.ActionRecommendReset {
		h.recommendResets++
	}
	h.lastResident = d.ResidentTokens
	if d.ResidentTokens > entry.heldPeak {
		entry.heldPeak = d.ResidentTokens
	}
	h.posture = d.HarnessPosture
	return d
}

// observePrefixGuard folds one served turn's prefix-determinism verdict (#2182): compare the
// inbound protected-prefix digest (the same content-free digest observe just folded, taken
// BEFORE any fak transform) against the trace's last-accepted baseline. Called ONLY while the
// prefix-guard lever is armed (Config.PrefixGuard / FAK_ABLATE_PREFIX_GUARD=1), and AFTER
// observe for the same turn, so the trace's coordEntry already exists. A mutated verdict
// re-primes the baseline, so a persistently NEW prefix counts one mutation, not one per turn —
// the same last-accepted-baseline discipline as cachemeta.PrefixStabilityTracker (#1602).
func (h *harnessCoherenceMetrics) observePrefixGuard(trace, digest string) {
	if h == nil {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.prefixGuardTurns++
	if digest == "" {
		// No breakpoint anchor ⇒ no protected prefix to hold deterministic. Never a
		// spurious mutation; the trace's baseline (if any) is left in place.
		h.prefixGuardUnknown++
		return
	}
	entry := h.coords[trace]
	if entry == nil {
		// observe folds first, so this only happens on an eviction race; treat as a fresh
		// baseline exactly like a first turn.
		if len(h.coords) >= maxCoherenceSessions {
			h.evictColdestCoordLocked()
		}
		entry = &coordEntry{}
		h.coords[trace] = entry
	}
	switch entry.guardDigest {
	case "":
		// First anchored turn primes the baseline; the prefix is trivially self-consistent.
		entry.guardDigest = digest
		h.prefixGuardStable++
	case digest:
		h.prefixGuardStable++
	default:
		entry.guardDigest = digest
		h.prefixGuardMutated++
	}
}

// observePrefixGuard is the gatewayMetrics-level, nil-safe accessor for the prefix-determinism
// guard fold, mirroring observeHarnessCoherence's shape so a Server built without metrics is safe.
func (m *gatewayMetrics) observePrefixGuard(trace, digest string) {
	if m == nil || m.harnessCoherence == nil {
		return
	}
	m.harnessCoherence.observePrefixGuard(trace, digest)
}

// recordCoherenceResetFired counts a coherence-triggered reset ACTUATED through the
// ResetOnBudget seam (Phase C actuation). The passthrough calls it the moment a live reset
// fires so fak_harness_coherence_reset_fired_total reflects real resets, not just recommendations.
func (h *harnessCoherenceMetrics) recordCoherenceResetFired() {
	if h == nil {
		return
	}
	h.mu.Lock()
	h.resetsFired++
	h.mu.Unlock()
}

// idleExceeds reports whether trace's last served turn (the same per-trace wall clock the
// cold-TTL signal folds) is older than the message-span cache TTL: h.ttl (the provider's short
// tier) normally, or a conservative one hour when the --managed-cache 1h upgrade is on. Message
// breakpoints ride the SHORT tier even under --managed-cache today (the 1h upgrade covers the
// stable head only, #2176), so the 1h threshold over-waits on purpose: a missed cold fire costs
// nothing new, a false cold claim would burst a warm cache. Unknown traces (first turn, or a
// server built without metrics) are never cold.
func (h *harnessCoherenceMetrics) idleExceeds(trace string, now time.Time, ttl1h bool) bool {
	if h == nil || trace == "" {
		return false
	}
	threshold := h.ttl
	if ttl1h {
		threshold = time.Hour
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	entry := h.coords[trace]
	if entry == nil || entry.lastTurn.IsZero() {
		return false
	}
	return now.Sub(entry.lastTurn) > threshold
}

// coldMessageSpanCache is the gatewayMetrics-level, nil-safe accessor for idleExceeds: has this
// trace provably idled past the message-breakpoint cache TTL since its last served turn? It is
// the OBSERVED (never guessed) cold-cache witness the --compact-anchor-head burst gate consumes:
// an expired message-span suffix re-bills cold this turn with or without a compaction, so a
// head-anchored fire on a cold trace carries no marginal cache penalty (#1407/#1408).
func (m *gatewayMetrics) coldMessageSpanCache(trace string, now time.Time, ttl1h bool) bool {
	if m == nil || m.harnessCoherence == nil {
		return false
	}
	return m.harnessCoherence.idleExceeds(trace, now, ttl1h)
}

// servedTurns reports how many served passthrough turns this trace has folded so far — the
// per-trace depth the head-anchored session-length prior reads as CurrentTurn. Because observe
// folds in the finalizer AFTER a turn's compaction runs, a turn N compaction sees the count from
// turn N-1, so CurrentTurn = servedTurns()+1 is the correct 1-based "this is turn N" index. An
// unknown trace (first turn, or a metrics-less server) reports 0.
func (h *harnessCoherenceMetrics) servedTurns(trace string) uint64 {
	return coordField(h, trace, func(e *coordEntry) uint64 { return e.turns })
}

// coordField reads one field of a trace's coordinator entry under the metrics lock. A nil
// receiver (metrics-less server), an empty trace, or a trace that has not been observed yet
// all report T's zero value — the conservative reading every caller's fallback prior wants.
// The per-trace accessors differ only in which field they pick, so the nil-safety and the
// locking live here once instead of being re-derived per field.
func coordField[T any](h *harnessCoherenceMetrics, trace string, pick func(*coordEntry) T) T {
	var zero T
	if h == nil || trace == "" {
		return zero
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	entry := h.coords[trace]
	if entry == nil {
		return zero
	}
	return pick(entry)
}

// servedTurnCount is the gatewayMetrics-level, nil-safe accessor for servedTurns: the per-trace
// served-turn depth the --compact-anchor-head burst gate consumes as CurrentTurn when it falls
// back to the assumed-session-length prior (an unbudgeted warm session with no wired turn horizon).
func (m *gatewayMetrics) servedTurnCount(trace string) uint64 {
	if m == nil || m.harnessCoherence == nil {
		return 0
	}
	return m.harnessCoherence.servedTurns(trace)
}

// heldResidentPeak reports the largest OBSERVED resident window (input+cached tokens) this trace has
// reached so far — the demonstrated held-volume the volume-aware head-anchored horizon reads to decide
// whether a trace is a heavy/long session (headSessionPrior / headHorizonHeavyResidentFloor). Folded in
// observe AFTER the turn, so a turn N compaction sees the peak through turn N-1 — the same one-turn lag
// as servedTurns, and the conservative direction (a not-yet-observed heavy turn keeps the base horizon).
// An unknown trace (first turn, or a metrics-less server) reports 0 ⇒ the conservative base horizon.
func (h *harnessCoherenceMetrics) heldResidentPeak(trace string) int64 {
	return coordField(h, trace, func(e *coordEntry) int64 { return e.heldPeak })
}

// heldResidentPeakTokens is the gatewayMetrics-level, nil-safe accessor for heldResidentPeak.
func (m *gatewayMetrics) heldResidentPeakTokens(trace string) int64 {
	if m == nil || m.harnessCoherence == nil {
		return 0
	}
	return m.harnessCoherence.heldResidentPeak(trace)
}

// harnessCoherenceInputs carries the per-turn facts the harness-coherence observation needs that
// are computed in handleAnthropicMessages (BEFORE fak's transforms) and must reach the streaming
// finalizers where the provider cache counters land. It threads through the stream functions so the
// observation is folded with the SAME content-free digest and fak-bail reason the buffered path uses.
type harnessCoherenceInputs struct {
	// inboundPrefixDigest is the content-free digest of the inbound protected prefix, taken before
	// any request-side transform.
	inboundPrefixDigest string
	// fakBail is fak's own compaction bail reason this turn, already mapped through fakBailReasonFor
	// ("" for a clean fire or a healthy under_budget no-op; the real reason for an actual bail).
	fakBail string
}

// observeHarnessCoherence folds one served Anthropic passthrough turn into the trace's coordinator,
// nil-safe at the gatewayMetrics layer so the call site (handleAnthropicMessages / the streaming
// passthrough) need not guard a Server built without metrics. It is the single entry point the
// passthrough uses; the accumulators it updates are the shared source /metrics and the operator
// line read.
func (m *gatewayMetrics) observeHarnessCoherence(trace string, now time.Time, digest string, fakFired bool, fakBail string, fakWorldBreak, sealed bool, cacheRead, cacheCreate, inputTokens int64) compactcohere.Decision {
	if m == nil || m.harnessCoherence == nil {
		return compactcohere.Decision{}
	}
	return m.harnessCoherence.observe(trace, now, digest, fakFired, fakBail, fakWorldBreak, sealed, cacheRead, cacheCreate, inputTokens)
}

// recordCoherenceResetFired is the gatewayMetrics-level, nil-safe accessor the actuation
// (maybeResetOnCoherence) calls when a coherence reset actually fires through the ResetOnBudget seam.
func (m *gatewayMetrics) recordCoherenceResetFired() {
	if m == nil || m.harnessCoherence == nil {
		return
	}
	m.harnessCoherence.recordCoherenceResetFired()
}

// observeHarnessCoherenceAndArm folds one served turn into the harness-coherence coordinator (as
// observeHarnessCoherence) and, when the returned Decision escalates a #3159 hard reset AND the host
// wired the opt-in ResetOnBudget, ARMS a coherence reset for trace so the next admitted turn
// actuates it (maybeResetOnCoherence). Arming here — at the call site, after the coordinator's own
// lock has been released inside observe — keeps the armed-latch lock off the coordinator's critical
// section. Gating on resetOnBudget != nil means an escalation on a gateway with no host resetter
// arms nothing while the recommendation still surfaces on /metrics. It returns the Decision.
func (s *Server) observeHarnessCoherenceAndArm(trace string, now time.Time, digest string, fakFired bool, fakBail string, fakWorldBreak, sealed bool, cacheRead, cacheCreate, inputTokens int64) compactcohere.Decision {
	d := s.metrics.observeHarnessCoherence(trace, now, digest, fakFired, fakBail, fakWorldBreak, sealed, cacheRead, cacheCreate, inputTokens)
	if s.prefixGuard {
		// #2182: the prefix-determinism guard rides the SAME per-turn seam (both the buffered
		// and streaming passthrough finalizers land here), so arming it changes gateway
		// behavior on every served turn — witnessed on fak_prefix_guard_* — without touching
		// the outbound bytes.
		s.metrics.observePrefixGuard(trace, digest)
	}
	if d.EscalateReset && s.resetOnBudget != nil {
		s.armCoherenceReset(trace)
	}
	return d
}

// harnessCoherenceSummary is the gatewayMetrics-level accessor for the operator-line roll-up,
// nil-safe so a bare Server still renders a zero summary.
func (m *gatewayMetrics) harnessCoherenceSummary() HarnessCoherenceSummary {
	if m == nil || m.harnessCoherence == nil {
		return HarnessCoherenceSummary{Posture: string(compactcohere.PostureBlock)}
	}
	return m.harnessCoherence.summary()
}

// harnessCoherenceSnapshot is a lock-free copy of the accumulators for rendering / the operator
// line. Both surfaces fold THIS struct, so a scrape and the exit line can never disagree.
type harnessCoherenceSnapshot struct {
	observedTurns    uint64
	events           map[compactcohere.PrefixEvent]uint64
	harnessRewrites  uint64
	quarantineAtRisk uint64
	burstsObserved   uint64
	overCeilingTurns uint64
	rewriteNoDrops   uint64
	recommendResets  uint64
	resetsFired      uint64
	lastResident     int64
	posture          compactcohere.Posture

	prefixGuardTurns   uint64
	prefixGuardStable  uint64
	prefixGuardMutated uint64
	prefixGuardUnknown uint64
}

func (h *harnessCoherenceMetrics) snapshot() harnessCoherenceSnapshot {
	out := harnessCoherenceSnapshot{
		events:  map[compactcohere.PrefixEvent]uint64{},
		posture: compactcohere.PostureBlock,
	}
	if h == nil {
		return out
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	out.observedTurns = h.observedTurns
	for k, v := range h.events {
		out.events[k] = v
	}
	out.harnessRewrites = h.harnessRewrites
	out.quarantineAtRisk = h.quarantineAtRisk
	out.burstsObserved = h.burstsObserved
	out.overCeilingTurns = h.overCeilingTurns
	out.rewriteNoDrops = h.rewriteNoDrops
	out.recommendResets = h.recommendResets
	out.resetsFired = h.resetsFired
	out.lastResident = h.lastResident
	out.posture = h.posture
	out.prefixGuardTurns = h.prefixGuardTurns
	out.prefixGuardStable = h.prefixGuardStable
	out.prefixGuardMutated = h.prefixGuardMutated
	out.prefixGuardUnknown = h.prefixGuardUnknown
	return out
}

// allEvents is the closed PrefixEvent set, emitted in a stable order at 0 so every panel exists
// before the first served turn (the same "emit-at-0" discipline the compaction family keeps).
var allEvents = []compactcohere.PrefixEvent{
	compactcohere.EventStable,
	compactcohere.EventFakCut,
	compactcohere.EventFakWorldBreak,
	compactcohere.EventHarnessRewrite,
	compactcohere.EventColdTTL,
}

// writeHarnessCoherenceMetrics renders the fak_harness_coherence_* family. It folds the SAME
// snapshot the operator line reads, so the two views agree by construction. The family is the
// gateway-visible form of the compactcohere decision surface: per-event counts (whose
// harness_rewrite bucket is the second-compactor event that was previously invisible), the
// quarantine-at-risk count (a fak seal that may have survived into a harness summary — the trust
// hole this policy exists to surface), the cache-creation-burst count, and the current standing
// PreCompact posture (block while fak copes; allow once fak's compaction has bailed for a streak).
func (h *harnessCoherenceMetrics) writeHarnessCoherenceMetrics(b *strings.Builder) {
	snap := h.snapshot()

	writeCounter(b, "fak_harness_coherence_turns_total",
		"WITNESSED (fak authored): served Anthropic passthrough turns folded into the harness-coherence coordinator. The denominator for the event family below.", int64(snap.observedTurns))

	writeHelpType(b, "fak_harness_coherence_events_total",
		"WITNESSED (fak authored): served turns by attributed prefix event (stable|fak_cut|fak_world_break|harness_rewrite|cold_ttl). harness_rewrite is the harness acting as a cache-DESTROYING second compactor (the inbound protected-prefix digest changed in a way fak never causes); cold_ttl is the provider cache going cold on an unchanged prefix. Attributed from CONTENT-FREE facts only (a prefix digest delta, fak's own compaction outcome, the provider's relayed cache counters, the idle gap).", "counter")
	for _, ev := range allEvents { // stable order; emit at 0 so the panel exists pre-first-turn
		fmt.Fprintf(b, "fak_harness_coherence_events_total{event=%q} %d\n", string(ev), snap.events[ev])
	}

	writeCounter(b, "fak_harness_coherence_harness_rewrites_total",
		"WITNESSED (fak authored): turns on which the HARNESS rewrote its own history (auto-compaction / /compact) — the inbound protected-prefix digest changed, which fak never causes (it forwards the prefix verbatim). Each bursts the provider cache (a cache_creation event), the opposite of what fak's cache-preserving compaction just worked to avoid.", int64(snap.harnessRewrites))

	writeCounter(b, "fak_harness_coherence_quarantine_at_risk_total",
		"WITNESSED (fak authored): harness-rewrite turns where a fak-sealed (quarantined) span preceded the rewrite — the poisoned span may have been folded into the harness summary, surviving the kernel's quarantine. The trust hole this policy exists to make observable (fak controls the wire, not the harness transcript).", int64(snap.quarantineAtRisk))

	writeCounter(b, "fak_harness_coherence_bursts_total",
		"WITNESSED (fak authored): turns that (will) cost a provider cache_creation burst — a harness rewrite or a cold-TTL rebuild. Lets an operator read the provider-cache cost of the two managers colliding.", int64(snap.burstsObserved))

	writeCounter(b, "fak_harness_coherence_over_ceiling_total",
		"WITNESSED (fak authored): served turns whose RESIDENT window (input + cache_creation + cache_read) exceeded the resident-token ceiling (FAK_CTX_YIELD_CEILING, default ~160k). A sustained streak yields the net back to the harness — fak is not bounding the window (#3158).", int64(snap.overCeilingTurns))

	writeCounter(b, "fak_harness_coherence_rewrite_no_drop_total",
		"WITNESSED (fak authored): harness rewrites judged NON-HOLDING — after the rewrite the resident window climbed back over the ceiling before the next rewrite, so the summary did not durably bound the window. The previously-invisible 'the compaction bailed' event (#3159).", int64(snap.rewriteNoDrops))

	writeCounter(b, "fak_harness_coherence_recommend_reset_total",
		"WITNESSED (fak authored): turns recommending a hard RESET — a cold-TTL rebuild (#774) or the non-holding-rewrite escalation (#3159). A recommendation surface; whether a reset is ACTUATED is reset_fired_total below.", int64(snap.recommendResets))

	writeCounter(b, "fak_harness_coherence_reset_fired_total",
		"WITNESSED (fak authored): coherence resets ACTUATED through the ResetOnBudget seam after the non-holding-rewrite escalation (#3159). Stays 0 when the host wired no resetter — actuation reuses the host's opt-in callback.", int64(snap.resetsFired))

	writeHelpType(b, "fak_harness_coherence_resident_tokens",
		"OBSERVED (relayed verbatim): the most recent served turn's RESIDENT window size (input + cache_creation + cache_read). A gauge an operator watches approach the ceiling; the size signal the yield and non-holding-rewrite policy (#3158/#3159) turn on.", "gauge")
	fmt.Fprintf(b, "fak_harness_coherence_resident_tokens %d\n", snap.lastResident)

	writeCounter(b, "fak_prefix_guard_turns_total",
		"WITNESSED (fak authored): served passthrough turns folded into the prefix-determinism guard (#2182, the runtime form of #1602/#1604) while the lever was armed (--prefix-guard on the gateway Config, or the FAK_ABLATE_PREFIX_GUARD=1 ablation arm). 0 means the guard is off. The denominator for the verdict counters below.", int64(snap.prefixGuardTurns))

	writeHelpType(b, "fak_prefix_guard_verdicts_total",
		"WITNESSED (fak authored): armed turns by prefix-determinism verdict — stable (the inbound protected-prefix digest matched the trace's last-accepted baseline), mutated (it diverged: the cacheable prefix burst, so provider-cache economics should NOT be relied on for this span), unknown (no cache_control anchor ⇒ no protected prefix to compare). Content-free: only the digest is compared, never prompt bytes.", "counter")
	fmt.Fprintf(b, "fak_prefix_guard_verdicts_total{verdict=%q} %d\n", "stable", snap.prefixGuardStable)
	fmt.Fprintf(b, "fak_prefix_guard_verdicts_total{verdict=%q} %d\n", "mutated", snap.prefixGuardMutated)
	fmt.Fprintf(b, "fak_prefix_guard_verdicts_total{verdict=%q} %d\n", "unknown", snap.prefixGuardUnknown)

	writeHelpType(b, "fak_harness_coherence_posture",
		"The CURRENT standing PreCompact posture the actuator (#1133, rung C) enforces when `fak guard` installs the Claude Code hook: 1 = block (exit 2 — suppress the harness's auto-compaction while fak's cache-preserving compaction is coping; the default), 0 = allow (exit 0 — fak's compaction has bailed for a sustained streak, so the harness is the only context net left).", "gauge")
	fmt.Fprintf(b, "fak_harness_coherence_posture %d\n", postureGauge(snap.posture))
}

func postureGauge(p compactcohere.Posture) int {
	if p == compactcohere.PostureBlock {
		return 1
	}
	return 0
}

// HarnessCoherenceSummary is the operator-line (#1135) roll-up of the harness-coherence family —
// the SAME numbers the fak_harness_coherence_* scrape reports, so the exit line and /metrics agree.
type HarnessCoherenceSummary struct {
	ObservedTurns    uint64            `json:"observed_turns"`
	Events           map[string]uint64 `json:"events,omitempty"`
	HarnessRewrites  uint64            `json:"harness_rewrites"`
	QuarantineAtRisk uint64            `json:"quarantine_at_risk"`
	BurstsObserved   uint64            `json:"bursts_observed"`
	OverCeiling      uint64            `json:"over_ceiling"`
	RewriteNoDrop    uint64            `json:"rewrite_no_drop"`
	RecommendResets  uint64            `json:"recommend_resets"`
	ResetsFired      uint64            `json:"resets_fired"`
	ResidentTokens   int64             `json:"resident_tokens"`
	Posture          string            `json:"posture"`
}

// summary folds the live accumulators into the operator-line roll-up. Every count is the SAME
// number the fak_harness_coherence_* scrape reports (both fold snapshot()), so the exit line can
// never disagree with the metrics — the explicit #1132 requirement.
func (h *harnessCoherenceMetrics) summary() HarnessCoherenceSummary {
	snap := h.snapshot()
	sum := HarnessCoherenceSummary{
		ObservedTurns:    snap.observedTurns,
		HarnessRewrites:  snap.harnessRewrites,
		QuarantineAtRisk: snap.quarantineAtRisk,
		BurstsObserved:   snap.burstsObserved,
		OverCeiling:      snap.overCeilingTurns,
		RewriteNoDrop:    snap.rewriteNoDrops,
		RecommendResets:  snap.recommendResets,
		ResetsFired:      snap.resetsFired,
		ResidentTokens:   snap.lastResident,
		Posture:          string(snap.posture),
	}
	if snap.observedTurns > 0 {
		ev := make(map[string]uint64, len(snap.events))
		keys := make([]string, 0, len(snap.events))
		for k := range snap.events {
			keys = append(keys, string(k))
		}
		sort.Strings(keys)
		for _, k := range keys {
			if v := snap.events[compactcohere.PrefixEvent(k)]; v > 0 {
				ev[k] = v
			}
		}
		if len(ev) > 0 {
			sum.Events = ev
		}
	}
	return sum
}
