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
//   - the preempted vs cold miss-token split (#3895, vLLM's preempted_* counter split):
//     of the prompt tokens a turn had to re-prefill (promptTokens - reusedTokens), the
//     share SELF-INFLICTED by admission evicting a still-warm entry is booked separately
//     from genuine cold/capacity misses, so eviction pressure cannot masquerade as low
//     cache value.
//   - the eligibility-filtered denominator (#3391): eligible prompt tokens exclude each
//     turn's always-uncacheable share (the cold first prefill into an empty cache, or a
//     tap with prefix reuse disabled), so reused/eligible judges the cache only on tokens
//     it could possibly have served — the raw prompt denominator counts the unavoidable
//     cold head against the cache and unfairly depresses the hit-rate.
//
// ObserveLabeled (labeled.go) additionally attributes a turn to its (model, tenant)
// series (#3391) so a shared gateway can tell WHOSE traffic earns the reuse. Label rows
// book the SAME clamped per-turn values as the globals, so summing any column across
// LabeledSnapshot rows always reconciles exactly with the global Stats counter.
//
// ObserveTier / TierSnapshot (tiers.go) add the EXPLICIT-TIER axis (#6422): which cache —
// the in-process prefix tree, a shared KV store, an upstream provider's managed cache —
// served each access, with its own request / hit / miss / byte / latency totals and a
// coarse backend class, so multi-tier cache value can never be blended into one flattering
// aggregate and a tier with no collector reports as unsupported instead of as zero.
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
	// eligibleTokens is the eligibility-filtered denominator (#3391): prompt tokens that
	// COULD have been served from the cached KV prefix — the prompt total minus each
	// turn's always-uncacheable share (the cold first prefill into an empty cache, or a
	// tap with prefix reuse disabled). reused <= cacheable <= eligible <= prompt by
	// construction, so reused/eligible is the fair hit-rate the raw denominator depresses.
	// Legacy taps carry no eligibility witness and book their whole prompt as eligible,
	// degrading the filtered ratio to the raw one — never inflating it.
	eligibleTokens uint64
	// byLabel keys the #3391 per-(model, tenant) breakdown (labeled.go). Rows book the
	// same clamped per-turn values as the globals above, so cross-label sums always
	// reconcile; taps without labels land on the ("unknown","unknown") row.
	byLabel map[Labels]*labelTotals
	// preemptedReuseLostTokens / coldMissTokens split every turn's missed prompt tokens
	// (promptTokens - reusedTokens) by cause (#3895, vLLM's preempted_* counter split):
	// preempted is the SELF-INFLICTED share — admission evicted a still-warm entry and
	// forced this turn to recompute tokens the cache had already paid for — while cold
	// is everything else: a genuine cold / capacity miss, or a miss with no eviction
	// witness (attribution defaults to cold, never to a fabricated self-infliction).
	// preempted + cold == promptTokens - reusedTokens by construction.
	preemptedReuseLostTokens uint64
	coldMissTokens           uint64
	// srcLocalCompute / srcLocalHit / srcExternalTransfer decompose observed prompt tokens
	// along the ORTHOGONAL provenance axis (#3896, vLLM's by_source counter): WHERE a served
	// token's value came from — recomputed locally, served from a prefix already RESIDENT on
	// this box, or pulled across the fabric from the external / DISAGGREGATED KV tier. Fed by
	// ObserveBySource (bysource.go), independent of the depth-axis counters above; the three
	// always sum to the prompt tokens booked through that tap (the parts==total invariant).
	srcLocalCompute     uint64
	srcLocalHit         uint64
	srcExternalTransfer uint64
	// tierRows / tierStatus are the EXPLICIT-TIER axis (#6422, tiers.go): per
	// (tier, operation, backend class) request/hit/miss/error/byte/latency counters, plus
	// each tier's collection status so an UNSUPPORTED tier reports as such instead of as a
	// zero row indistinguishable from an implemented-but-idle one. Fed by ObserveTier and,
	// for the in-process prefix tier, by observeAttributed below — inside the same critical
	// section as the aggregate counters, so the two can never desync.
	tierRows   map[tierKey]*tierTotals
	tierStatus map[CacheTier]TierStatus
	// rejectedTierAccesses counts public ObserveTier calls rejected because at least one
	// dimension fell outside its closed vocabulary. Rejected accesses never open a row.
	rejectedTierAccesses uint64
	frozen               uint64 // turns with reuse ratio >= FrozenFloor (the append-only ceiling)
	partial              uint64 // turns between ColdCeil and FrozenFloor
	cold                 uint64 // turns with reuse ratio < ColdCeil (cold / head-mutated / fanned-out)
	// reuseHist[i] counts turns whose ratio fell in (ReuseRatioBuckets[i-1],
	// ReuseRatioBuckets[i]] — per-bucket (non-cumulative) so each increment touches
	// one slot; a renderer accumulates left-to-right to emit `le` lines.
	reuseHist [len(ReuseRatioBuckets)]uint64
}

// New returns a fresh observer (tests use it for isolation; production uses Default). It
// starts from this build's declared tier support (#6422, defaultTierStatus): which cache
// tiers have a collector at all is a property of the binary, not of one observer, so an
// unsupported tier reports as unsupported from the first snapshot rather than as a zero row.
func New() *Observer { return &Observer{tierStatus: defaultTierStatus()} }

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
// the split adds the lookup rate beside them, it does not redefine them. With no
// eviction witness the turn's missed tokens are attributed to the cold bucket (#3895):
// a tap that KNOWS the miss was self-inflicted uses ObservePreempted instead. With no
// eligibility witness the whole prompt books as eligible (#3391) — the filtered ratio
// degrades to the raw one; a tap that knows the turn's uncacheable share uses
// ObserveLabeled instead.
func (o *Observer) ObserveSplit(promptTokens, cacheablePrefixTokens, reusedPrefixTokens int) {
	o.observeAttributed(Labels{}, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, 0, promptTokens)
}

// ObservePreempted records one served in-kernel turn exactly like ObserveSplit and
// additionally attributes preemptedLostTokens of the turn's missed prompt tokens
// (promptTokens - reusedPrefixTokens) to SELF-INFLICTED eviction (#3895, vLLM's
// preempted_* counter split): admission evicted a still-warm entry, so this turn had to
// recompute tokens the cache had already paid for. The caller supplies the witness —
// typically the cache lifecycle, which knows the refetched key was evicted while warm;
// cacheobs never infers preemption on its own. preemptedLostTokens is clamped into
// [0, promptTokens - reusedPrefixTokens] after the ObserveSplit clamps (a served token
// was not lost to anything), and the remainder of the miss books to the cold bucket, so
// PreemptedReuseLostTokens + ColdMissTokens == PromptTokens - ReusedTokens always
// reconciles. Every pre-existing counter, regime bucket, histogram slot, and the
// CacheabilityRatio/ReuseRatio aggregates accumulate exactly as ObserveSplit — the
// attribution splits the miss, it never changes the aggregate's meaning.
func (o *Observer) ObservePreempted(promptTokens, cacheablePrefixTokens, reusedPrefixTokens, preemptedLostTokens int) {
	o.observeAttributed(Labels{}, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, preemptedLostTokens, promptTokens)
}

// observeAttributed is the shared accumulation core behind Observe / ObserveSplit /
// ObservePreempted / ObserveLabeled: the #3390 lookup-vs-realized clamps and counters,
// the #3895 miss-cause attribution (preempted vs cold) of the turn's un-reused prompt
// tokens, and the #3391 eligibility denominator plus (model, tenant) series row. Taps
// with no eligibility witness pass eligiblePromptTokens == promptTokens, degrading the
// filtered ratio to the raw one (an over-counted denominator can only UNDER-state the
// filtered hit-rate, never inflate it). eligiblePromptTokens is clamped into
// [cacheablePrefixTokens, promptTokens] AFTER the #3390 clamps: a token that matched the
// index at lookup was demonstrably cacheable, hence eligible — so even a stale witness
// (e.g. a prewarmed tree serving a "first" prefill) can never push reused/eligible
// above 1. The label row books the SAME clamped values as the globals, so cross-label
// sums always reconcile (the never-desync invariant).
func (o *Observer) observeAttributed(labels Labels, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, preemptedLostTokens, eligiblePromptTokens int) {
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
	if eligiblePromptTokens < cacheablePrefixTokens {
		eligiblePromptTokens = cacheablePrefixTokens
	}
	if eligiblePromptTokens > promptTokens {
		eligiblePromptTokens = promptTokens
	}
	missTokens := promptTokens - reusedPrefixTokens
	if preemptedLostTokens < 0 {
		preemptedLostTokens = 0
	}
	if preemptedLostTokens > missTokens {
		preemptedLostTokens = missTokens
	}
	labels = labels.normalized()
	ratio := float64(reusedPrefixTokens) / float64(promptTokens)
	o.mu.Lock()
	o.turns = saturatingAddU64(o.turns, 1)
	o.promptTokens = saturatingAddU64(o.promptTokens, uint64(promptTokens))
	o.reusedTokens = saturatingAddU64(o.reusedTokens, uint64(reusedPrefixTokens))
	o.cacheableTokens = saturatingAddU64(o.cacheableTokens, uint64(cacheablePrefixTokens))
	o.eligibleTokens = saturatingAddU64(o.eligibleTokens, uint64(eligiblePromptTokens))
	o.preemptedReuseLostTokens = saturatingAddU64(o.preemptedReuseLostTokens, uint64(preemptedLostTokens))
	o.coldMissTokens = saturatingAddU64(o.coldMissTokens, uint64(missTokens-preemptedLostTokens))
	lt := o.labelTotalsLocked(labels)
	lt.turns = saturatingAddU64(lt.turns, 1)
	lt.promptTokens = saturatingAddU64(lt.promptTokens, uint64(promptTokens))
	lt.eligibleTokens = saturatingAddU64(lt.eligibleTokens, uint64(eligiblePromptTokens))
	lt.reusedTokens = saturatingAddU64(lt.reusedTokens, uint64(reusedPrefixTokens))
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
	// #6422 tier axis: this turn WAS one read against the in-process KV-prefix tier, so book
	// it as such — the explicit-tier record every existing depth-axis tap now emits, without
	// its call sites changing. The verdict is the tier's own, not the turn's regime: a lookup
	// that returned any cached prefix hit the tier, one that returned none missed it (the
	// depth axis already reports HOW MUCH of the prefix came back). No byte or latency
	// witness is claimed — the planner measures tokens, not the tree's resident bytes or its
	// lookup latency — so those accesses book as unsized and untimed rather than as
	// zero-byte, zero-latency ones. Booked under the SAME lock as the aggregate counters
	// above so the tier totals can never desync from the aggregate they decompose.
	tierOutcome := OutcomeMiss
	if reusedPrefixTokens > 0 {
		tierOutcome = OutcomeHit
	}
	o.observeTierLocked(TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: tierOutcome, Backend: BackendMemory})
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
	// RejectedTierAccesses is the number of invalid public tier observations dropped whole.
	RejectedTierAccesses uint64
	// CacheableTokens is the lookup-side half of the #3390 split: prompt tokens that
	// matched the prefix index at lookup time, before eviction/admission decided what
	// was servable. Always >= ReusedTokens; equal when every tap lacked lookup info.
	CacheableTokens uint64
	// EligibleTokens is the eligibility-filtered denominator (#3391): prompt tokens that
	// COULD have been served from the cached KV prefix — the prompt total minus each
	// turn's always-uncacheable share (the cold first prefill into an empty cache, or a
	// tap with prefix reuse disabled). ReusedTokens <= CacheableTokens <= EligibleTokens
	// <= PromptTokens by construction; equal to PromptTokens when every tap lacked an
	// eligibility witness.
	EligibleTokens uint64
	// PreemptedReuseLostTokens is the SELF-INFLICTED share of the missed prompt tokens
	// (#3895, vLLM's preempted_* counter split): tokens a turn had to recompute because
	// admission evicted a still-warm entry, as witnessed by an ObservePreempted tap.
	// Kept strictly apart from ColdMissTokens so eviction pressure cannot masquerade
	// as low cache value. It sub-attributes the miss side; the CacheabilityRatio -
	// ReuseRatio aggregate keeps its exact pre-split meaning.
	PreemptedReuseLostTokens uint64
	// ColdMissTokens is every other missed prompt token: a genuine cold / capacity
	// miss, or a miss observed without an eviction witness (Observe / ObserveSplit
	// default the attribution here — the honest default, never a fabricated
	// self-infliction). PreemptedReuseLostTokens + ColdMissTokens ==
	// PromptTokens - ReusedTokens (barring saturation at MaxUint64).
	ColdMissTokens uint64
	FrozenTurns    uint64
	PartialTurns   uint64
	ColdTurns      uint64
	// ReuseRatio is reusedTokens/promptTokens — the realized cache-hit across all observed
	// turns. 0 when no turns have prompt tokens yet (an idle process never reports a
	// phantom ratio).
	ReuseRatio float64
	// CacheabilityRatio is cacheableTokens/promptTokens — the lookup-side could-be-served
	// rate. CacheabilityRatio - ReuseRatio is the token-weighted rate lost to eviction/
	// admission. 0 when no turns have prompt tokens yet, like ReuseRatio.
	CacheabilityRatio float64
	// EligibleReuseRatio is reusedTokens/eligibleTokens — the eligibility-filtered fair
	// hit-rate (#3391). ReuseRatio counts the always-uncacheable cold head in its
	// denominator and so under-states the cache's performance on the tokens it could
	// possibly have served; this ratio excludes that head. Always >= ReuseRatio, and
	// equal when no tap carried an eligibility witness. 0 when no eligible tokens have
	// been observed yet (an idle or reuse-disabled process never reports a phantom ratio).
	EligibleReuseRatio float64
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
		Turns:                    o.turns,
		PromptTokens:             o.promptTokens,
		ReusedTokens:             o.reusedTokens,
		RejectedTierAccesses:     o.rejectedTierAccesses,
		CacheableTokens:          o.cacheableTokens,
		EligibleTokens:           o.eligibleTokens,
		PreemptedReuseLostTokens: o.preemptedReuseLostTokens,
		ColdMissTokens:           o.coldMissTokens,
		FrozenTurns:              o.frozen,
		PartialTurns:             o.partial,
		ColdTurns:                o.cold,
		ReuseHistTurns:           o.reuseHist,
	}
	if o.promptTokens > 0 {
		s.ReuseRatio = float64(o.reusedTokens) / float64(o.promptTokens)
		s.CacheabilityRatio = float64(o.cacheableTokens) / float64(o.promptTokens)
	}
	if o.eligibleTokens > 0 {
		s.EligibleReuseRatio = float64(o.reusedTokens) / float64(o.eligibleTokens)
	}
	return s
}
