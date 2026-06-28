package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
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
	// posture is the CURRENT standing block/allow posture across all live traces (last-write-wins
	// per fold; a single Claude Code session drives one trace, so this is that session's stance).
	posture compactcohere.Posture
}

// coordEntry is one trace's coordinator plus the wall clock of its last served turn (for the idle
// gap the TTL signal needs).
type coordEntry struct {
	coord    *compactcohere.Coordinator
	lastTurn time.Time
}

func newHarnessCoherenceMetrics(ttl time.Duration) *harnessCoherenceMetrics {
	if ttl <= 0 {
		ttl = compactcohere.DefaultProviderCacheTTL
	}
	return &harnessCoherenceMetrics{
		ttl:     ttl,
		coords:  map[string]*coordEntry{},
		events:  map[compactcohere.PrefixEvent]uint64{},
		posture: compactcohere.PostureBlock,
	}
}

// observe folds one served Anthropic passthrough turn into the trace's coordinator and updates the
// shared accumulators. trace keys the per-session coordinator; now is the served turn's wall clock
// (drives the idle-gap TTL signal). digest is the CONTENT-FREE inbound protected-prefix digest taken
// BEFORE fak's transforms; fakFired/fakBail describe fak's own compaction this turn (fakBail already
// mapped through fakBailReasonFor); sealed is whether fak sealed/quarantined a span in this turn's
// context; cacheRead/cacheCreate are the provider's OBSERVED counters, relayed verbatim. It returns
// the per-turn Decision so a caller (the debug line / a test) can read the verdict directly.
func (h *harnessCoherenceMetrics) observe(trace string, now time.Time, digest string, fakFired bool, fakBail string, fakWorldBreak, sealed bool, cacheRead, cacheCreate int64) compactcohere.Decision {
	if h == nil {
		return compactcohere.Decision{}
	}
	h.mu.Lock()
	defer h.mu.Unlock()

	entry := h.coords[trace]
	if entry == nil {
		entry = &coordEntry{coord: compactcohere.New(h.ttl)}
		h.coords[trace] = entry
	}
	var idle time.Duration
	if !entry.lastTurn.IsZero() {
		idle = now.Sub(entry.lastTurn)
	}
	entry.lastTurn = now

	obs := compactcohere.TurnObservation{
		InboundPrefixDigest: digest,
		FakCompactFired:     fakFired,
		FakBailReason:       fakBail,
		FakWorldBreak:       fakWorldBreak,
		SealedSpanPresent:   sealed,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheCreate,
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
	h.posture = d.HarnessPosture
	return d
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
func (m *gatewayMetrics) observeHarnessCoherence(trace string, now time.Time, digest string, fakFired bool, fakBail string, fakWorldBreak, sealed bool, cacheRead, cacheCreate int64) compactcohere.Decision {
	if m == nil || m.harnessCoherence == nil {
		return compactcohere.Decision{}
	}
	return m.harnessCoherence.observe(trace, now, digest, fakFired, fakBail, fakWorldBreak, sealed, cacheRead, cacheCreate)
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
	posture          compactcohere.Posture
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
	out.posture = h.posture
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

	writeHelpType(b, "fak_harness_coherence_posture",
		"The CURRENT standing PreCompact posture the actuator (#1133, rung C) would enforce: 1 = block (exit 2 — suppress the harness's auto-compaction while fak's cache-preserving compaction is coping; the default), 0 = allow (exit 0 — fak's compaction has bailed for a sustained streak, so the harness is the only context net left). A decision surface only until rung C wires the hook.", "gauge")
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
