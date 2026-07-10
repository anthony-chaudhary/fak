// Package cacheobs is the process-global observability tap for in-kernel KV-prefix
// reuse — the LIVE measurement of the "frozen-trajectory cache cliff"
// (docs/explainers/frozen-trajectory-cache-cliff.md).
//
// The in-kernel planner already computes, on every served turn, how many prompt tokens
// it prefilled (promptTokens) and how many of them it served from the cached KV prefix
// (the RadixAttention match). That ratio IS the realized cache-hit. Until now it reached
// only a log line; this tap accumulates it so the gateway can scrape it onto /metrics
// (the fak_gateway_kv_prefix_* family), making the cliff observable in any fak kernel run
// rather than a number you can only model offline.
//
// It mirrors the established process-global stats pattern (blob.Default, vdso.Default):
// a low-tier (foundation) leaf with no imports beyond the stdlib, fed by the hot path and
// read by the metrics renderer. The cliff is legible from two derived signals:
//
//   - reuse ratio = reusedTokens / promptTokens — the realized cache-hit. A single,
//     linear, append-only agent climbs toward ~1 (the frozen ceiling); flexibility, cold
//     fan-out, or a divergent prefix drives it down.
//   - cacheability ratio = cacheableTokens / promptTokens — the lookup-side could-be-served
//     rate (#3390, LMCache's lookup vs retrieve split): tokens that matched the prefix
//     index at lookup time, BEFORE eviction/admission decided what was actually servable.
//     Cacheability >= realized always; the gap between the two rates is the token weight
//     lost to eviction/admission, which a single realized rate silently folds into "miss".
//   - the per-regime turn buckets — frozen (reuse >= FrozenFloor), partial, cold
//     (reuse < ColdCeil) — show WHEN turns leave the frozen regime, which a single
//     cumulative ratio hides.
package cacheobs

import (
	"math"
	"sync"
)

// FrozenFloor / ColdCeil bucket a turn's reuse ratio into the three cliff regimes. A turn
// at or above FrozenFloor reused almost its whole prefix (the append-only ceiling); a turn
// below ColdCeil reused almost nothing (a cold first prefill, or a head-mutated / fanned-out
// turn that left the frozen regime). Between them is partial reuse.
const (
	FrozenFloor = 0.90
	ColdCeil    = 0.10
)

// ReuseRatioBuckets are the per-turn reuse-ratio histogram upper bounds (Prometheus
// `le` semantics: a turn lands in the first bucket whose bound is >= its ratio). The
// three-regime counters show WHEN turns leave the frozen regime; these buckets show
// the SHAPE of the partial range (bimodal vs uniform, #963) that a single scalar
// hides. Every observed turn is recorded — cumulative counts and an +Inf bucket equal
// to Turns fall out for free — while cold keeps its own strict (< ColdCeil) counter,
// so the le=0.1 bucket differs from ColdTurns only by turns at exactly ColdCeil.
var ReuseRatioBuckets = [...]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

// Default is the process-global observer the in-kernel planner feeds and the gateway
// scrapes. One per process, like blob.Default / vdso.Default.
var Default = New()

// Observer accumulates in-kernel KV-prefix reuse across served turns. Safe for concurrent
// use — the gateway may drive Complete (which feeds Observe) from many goroutines.
type Observer struct {
	mu           sync.Mutex
	turns        uint64
	promptTokens uint64
	reusedTokens uint64
	// cacheableTokens is the lookup-side half of the #3390 split: tokens that matched
	// the prefix index at lookup time, whether or not eviction/admission then let them
	// be served. Always >= reusedTokens (a served token was necessarily matched).
	cacheableTokens uint64
	frozen          uint64 // turns with reuse ratio >= FrozenFloor (the append-only ceiling)
	partial         uint64 // turns between ColdCeil and FrozenFloor
	cold            uint64 // turns with reuse ratio < ColdCeil (cold / head-mutated / fanned-out)
	// reuseHist[i] counts turns whose ratio fell in (ReuseRatioBuckets[i-1],
	// ReuseRatioBuckets[i]] — per-bucket (non-cumulative) so each increment touches
	// one slot; a renderer accumulates left-to-right to emit `le` lines.
	reuseHist [len(ReuseRatioBuckets)]uint64
}

// New returns a fresh observer (tests use it for isolation; production uses Default).
func New() *Observer { return &Observer{} }

// Observe records one served in-kernel turn: promptTokens prefilled, of which
// reusedPrefixTokens were served from the cached KV prefix (the planner's `matched`).
// A non-positive promptTokens is ignored (no turn to attribute); reusedPrefixTokens is
// clamped into [0, promptTokens] so a miscount can never push the ratio outside [0,1] or
// the reused total above the prompt total. With no lookup-side information the cacheable
// count defaults to the realized count — the tightest honest lower bound (never a
// fabricated lookup hit) — so CacheabilityRatio degrades to ReuseRatio for legacy taps.
func (o *Observer) Observe(promptTokens, reusedPrefixTokens int) {
	o.ObserveSplit(promptTokens, reusedPrefixTokens, reusedPrefixTokens)
}

// ObserveSplit records one served in-kernel turn with BOTH halves of the #3390 token
// hit-rate split (LMCache's lookup vs retrieve rates): promptTokens prefilled, of which
// cacheablePrefixTokens matched the prefix index at lookup time — before eviction/
// admission decided servability — and reusedPrefixTokens were actually served from the
// cached KV prefix. reusedPrefixTokens is clamped into [0, promptTokens] as in Observe;
// cacheablePrefixTokens is then clamped into [reusedPrefixTokens, promptTokens], because
// a token cannot be served without having matched, so the CacheabilityRatio >= ReuseRatio
// invariant holds by construction and the gap between them is exactly the eviction/
// admission loss. Regime buckets and the histogram stay keyed on the REALIZED ratio —
// the split adds the lookup rate beside them, it does not redefine them.
func (o *Observer) ObserveSplit(promptTokens, cacheablePrefixTokens, reusedPrefixTokens int) {
	if o == nil || promptTokens <= 0 {
		return
	}
	if reusedPrefixTokens < 0 {
		reusedPrefixTokens = 0
	}
	if reusedPrefixTokens > promptTokens {
		reusedPrefixTokens = promptTokens
	}
	if cacheablePrefixTokens < reusedPrefixTokens {
		cacheablePrefixTokens = reusedPrefixTokens
	}
	if cacheablePrefixTokens > promptTokens {
		cacheablePrefixTokens = promptTokens
	}
	ratio := float64(reusedPrefixTokens) / float64(promptTokens)
	o.mu.Lock()
	o.turns = saturatingAddU64(o.turns, 1)
	o.promptTokens = saturatingAddU64(o.promptTokens, uint64(promptTokens))
	o.reusedTokens = saturatingAddU64(o.reusedTokens, uint64(reusedPrefixTokens))
	o.cacheableTokens = saturatingAddU64(o.cacheableTokens, uint64(cacheablePrefixTokens))
	switch {
	case ratio >= FrozenFloor:
		o.frozen = saturatingAddU64(o.frozen, 1)
	case ratio < ColdCeil:
		o.cold = saturatingAddU64(o.cold, 1)
	default:
		o.partial = saturatingAddU64(o.partial, 1)
	}
	idx := len(ReuseRatioBuckets) - 1 // ratio is clamped to [0,1], so le=1.0 always catches
	for i, le := range ReuseRatioBuckets {
		if ratio <= le {
			idx = i
			break
		}
	}
	o.reuseHist[idx] = saturatingAddU64(o.reuseHist[idx], 1)
	o.mu.Unlock()
}

// saturatingAddU64 returns a+b clamped to math.MaxUint64 instead of silently
// wrapping past it. A real process will never prefill anywhere near 2^64 tokens,
// but the defined behavior at the ceiling must be a stuck-high saturation, never
// an undetected wrap back down to a small count.
func saturatingAddU64(a, b uint64) uint64 {
	sum := a + b
	if sum < a {
		return math.MaxUint64
	}
	return sum
}

// Stats is a point-in-time snapshot of the accumulated reuse.
type Stats struct {
	Turns        uint64
	PromptTokens uint64
	ReusedTokens uint64
	// CacheableTokens is the lookup-side half of the #3390 split: prompt tokens that
	// matched the prefix index at lookup time, before eviction/admission decided what
	// was servable. Always >= ReusedTokens; equal when every tap lacked lookup info.
	CacheableTokens uint64
	FrozenTurns     uint64
	PartialTurns    uint64
	ColdTurns       uint64
	// ReuseRatio is reusedTokens/promptTokens — the realized cache-hit across all observed
	// turns. 0 when no turns have prompt tokens yet (an idle process never reports a
	// phantom ratio).
	ReuseRatio float64
	// CacheabilityRatio is cacheableTokens/promptTokens — the lookup-side could-be-served
	// rate. CacheabilityRatio - ReuseRatio is the token-weighted rate lost to eviction/
	// admission. 0 when no turns have prompt tokens yet, like ReuseRatio.
	CacheabilityRatio float64
	// ReuseHistTurns[i] counts turns whose per-turn ratio fell in
	// (ReuseRatioBuckets[i-1], ReuseRatioBuckets[i]] (per-bucket, non-cumulative).
	// Every observed turn is counted, so the values sum to Turns.
	ReuseHistTurns [len(ReuseRatioBuckets)]uint64
}

// ColdCliffFinding is the stable code raised when a session has fallen off the
// frozen-trajectory cache cliff: the in-kernel KV-prefix reuse it was earning has
// collapsed, so the kernel is re-prefilling prompt it used to serve from the cached
// KV. It is a fixed string so a banner / /debug/vars reader can match on it (#3623).
const ColdCliffFinding = "PREFIX_COLD_CLIFF"

// Cold-cliff alarm thresholds. Deliberately conservative: a session's first prefill
// is always cold and ordinary partial reuse is healthy, so neither may trip the
// alarm — only an aggregate collapse does.
const (
	// ColdCliffMinTurns gates the alarm until enough turns are observed that an
	// unavoidable cold turn-1 prefill can no longer dominate the reading.
	ColdCliffMinTurns = 3
	// ColdCliffReuseFloor is the aggregate reuse-ratio floor: below it the kernel is
	// re-prefilling more than half of every prompt on average — the frozen ceiling is lost.
	ColdCliffReuseFloor = 0.50
	// ColdCliffColdFraction is the cold-turn fraction (ColdTurns/Turns) at or above which
	// the session is cold-dominated even if a few large warm turns prop the token-weighted
	// ratio up.
	ColdCliffColdFraction = 0.50
)

// ColdCliffVerdict is the result of evaluating a Stats snapshot for the frozen-trajectory
// cache cliff. A reader treats a Fired verdict — on the wire, the mere PRESENCE of this
// record, since a healthy session omits it — as the PREFIX_COLD_CLIFF alarm.
type ColdCliffVerdict struct {
	// Fired is true when the session has left the frozen regime. It is not serialized:
	// the record is only attached to a surface when it fired, so presence IS the alarm.
	Fired bool `json:"-"`
	// Finding is the stable alarm code (ColdCliffFinding) when Fired, else "".
	Finding string `json:"finding,omitempty"`
	// Reason names which signal tripped: "cold_fraction" (a majority of turns landed
	// cold) or "reuse_floor" (the token-weighted aggregate reuse fell below the floor).
	Reason string `json:"reason,omitempty"`
	// Turns / ReuseRatio / ColdFraction are the evidence the verdict was drawn from, so a
	// reader never has to re-derive why the alarm fired (or did not).
	Turns        uint64  `json:"turns"`
	ReuseRatio   float64 `json:"reuse_ratio"`
	ColdFraction float64 `json:"cold_fraction"`
}

// ColdCliff evaluates the snapshot for the frozen-trajectory cache cliff (#3623). It
// fires once at least ColdCliffMinTurns turns have been observed and EITHER a majority
// of them landed in the cold regime (ColdFraction >= ColdCliffColdFraction) OR the
// token-weighted aggregate reuse has fallen below ColdCliffReuseFloor. An idle or short
// session, and a frozen trajectory (reuse near 1, no cold turns), never fire. It reads
// the CUMULATIVE snapshot, not a per-turn delta: it answers "has this session's realized
// reuse collapsed" — exactly the cliff the metric exists to show.
func (s Stats) ColdCliff() ColdCliffVerdict {
	v := ColdCliffVerdict{Turns: s.Turns, ReuseRatio: s.ReuseRatio}
	if s.Turns > 0 {
		v.ColdFraction = float64(s.ColdTurns) / float64(s.Turns)
	}
	if s.Turns < ColdCliffMinTurns {
		return v // idle or still warming up: an unavoidable cold turn-1 must not alarm
	}
	switch {
	case v.ColdFraction >= ColdCliffColdFraction:
		v.Fired, v.Finding, v.Reason = true, ColdCliffFinding, "cold_fraction"
	case v.ReuseRatio < ColdCliffReuseFloor:
		v.Fired, v.Finding, v.Reason = true, ColdCliffFinding, "reuse_floor"
	}
	return v
}

// Snapshot returns the current accumulated stats. The ratio is derived under the lock so
// it is always consistent with the totals it is computed from.
func (o *Observer) Snapshot() Stats {
	if o == nil {
		return Stats{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	s := Stats{
		Turns:           o.turns,
		PromptTokens:    o.promptTokens,
		ReusedTokens:    o.reusedTokens,
		CacheableTokens: o.cacheableTokens,
		FrozenTurns:     o.frozen,
		PartialTurns:    o.partial,
		ColdTurns:       o.cold,
		ReuseHistTurns:  o.reuseHist,
	}
	if o.promptTokens > 0 {
		s.ReuseRatio = float64(o.reusedTokens) / float64(o.promptTokens)
		s.CacheabilityRatio = float64(o.cacheableTokens) / float64(o.promptTokens)
	}
	return s
}
