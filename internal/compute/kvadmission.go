package compute

// kvadmission.go — VALUE-AWARE KV ADMISSION: the miss-time decision that answers the dual
// of #2239's eviction question. Cost-aware eviction (kvcost.go) answers "which resident span
// is cheapest to LOSE"; it never answers "should this incoming span be cached AT ALL?"
// (issue #2672, epic #2236 matrix row M5). Today every miss inserts (ReplayKVCache /
// radixkv Insert), so a scan of one-shot cold spans evicts a proven-hot span just to cache
// pollutants that are never reused. Eviction alone cannot close this hole: by the time the
// hot span is the victim, the damage — admitting the pollutant — is already done. The classic
// fix every serious value-aware cache carries (W-TinyLFU / Caffeine, LRU-K/2Q/ARC, GDSF
// admission) is an ADMISSION FILTER: admit only if the newcomer is worth more than the victim
// it would displace.
//
// WHAT THIS FILE IS. The pure, deterministic, host-tractable DECISION — the missing
// value-aware member of internal/compute's admission family (discard_admit.go declines a
// forward pass known to be thrown away; prewarm_admit.go admits a forward pass known to be
// needed; this one declines to CACHE a span not worth its displacement). It reuses the
// family's KVSpanStats telemetry and the exact KVEvictionCost value-of-keeping score, so a
// candidate and the cheapest resident victim are ranked on one comparable axis. It moves no
// KV and holds no state: it depends only on its inputs, so it is byte-deterministic across
// machines — the repo's house form for a policy that must not drift with hardware.
//
// THE FENCES (inherited from the admission family, load-bearing here).
//   - Fail-open on unknowns. An unpriced candidate (Bytes ≤ 0, cost undefined) or an empty
//     resident set (no victim to protect) ALWAYS admits: never refuse to cache a span we
//     cannot PROVE is a pollutant. A rejected admission is only ever safe against a proven
//     colder-than-the-victim newcomer — the same fail-open contract discard_admit holds.
//   - No correctness surface. A bypassed admission only costs a cold prefill next time (the
//     status quo — today's miss already had to prefill); it never changes an answer. This is
//     a hit-rate / cost lever ONLY.
//   - Reduces to insert-always. The zero-value verdict is AdmitCache, so a default-constructed
//     decision reproduces today's insert-always behavior byte-identically; the aging-clock
//     admission path is inert when the pool does not age (clock ≤ 0). Wiring this in is purely
//     additive: an all-fail-open call site is identical to no gate at all.
//
// LIVE WIRING. internal/radixkv/admission.go now consults DecideKVAdmission before the real
// Insert / InsertSnapshot mutation whenever bounded token or snapshot residency would be
// displaced (#9311). It combines this value comparison with cacheprice's bounded frequency
// evidence and fails open when that telemetry is absent. The #2244 eviction-pressure
// benchmark remains the broader cross-policy measurement rung; this file stays the pure
// deterministic decision core.

// KVAdmitVerdict is the typed outcome of the value-aware KV admission decision.
type KVAdmitVerdict uint8

const (
	// AdmitCache is the fail-open verdict: cache the incoming span (evicting the cheapest
	// resident victim if the budget requires it), exactly as the insert-always path does
	// today. It is the ZERO VALUE, so a default-constructed decision caches — a call site
	// that never gates reproduces today's behavior byte-identically.
	AdmitCache KVAdmitVerdict = iota
	// AdmitBypass: serve the miss WITHOUT caching the span. The resident set is left intact,
	// so a proven-hot span is protected from being evicted for a colder newcomer. Costs only
	// a cold prefill next time — the status quo for a miss — never a wrong answer.
	AdmitBypass
)

// String renders the verdict for logs and the observability/witness surface.
func (v KVAdmitVerdict) String() string {
	switch v {
	case AdmitBypass:
		return "bypass"
	default:
		return "cache"
	}
}

// KVAdmitReason explains why a verdict was chosen — for the audit log and the witness. It is
// a closed vocabulary; policy code never parses a free-text reason.
type KVAdmitReason string

const (
	// ReasonUnpricedCandidate: the candidate has no tracked footprint (Bytes ≤ 0), so its
	// value is undefined — admit rather than refuse to cache a span we cannot prove is a
	// pollutant (the fail-open contract).
	ReasonUnpricedCandidate KVAdmitReason = "candidate_unpriced"
	// ReasonNoResidentVictim: the resident set is empty (no victim to displace), so caching
	// the candidate evicts nothing hot — admit (fail-open).
	ReasonNoResidentVictim KVAdmitReason = "no_resident_victim"
	// ReasonOutranksVictim: the candidate's value-of-keeping strictly exceeds the victim it
	// would displace — the W-TinyLFU rule, admit and let it replace the cheaper span.
	ReasonOutranksVictim KVAdmitReason = "outranks_victim"
	// ReasonClearsAgingClock: the candidate's value clears the current GDSF aging clock L
	// (#2668), so it is admitted even against a momentarily-cheaper victim. Inert when the
	// pool does not age (clock ≤ 0).
	ReasonClearsAgingClock KVAdmitReason = "clears_aging_clock"
	// ReasonColderThanVictim: the candidate is worth no more to keep than the resident victim
	// it would displace and does not clear the aging clock — bypass, so a one-shot scan cannot
	// evict a hotter span. This is the only verdict that protects residency.
	ReasonColderThanVictim KVAdmitReason = "colder_than_victim"
)

// KVAdmitDecision is the typed outcome for one candidate: the verdict plus the
// closed-vocabulary reason for the audit/witness surface.
type KVAdmitDecision struct {
	Verdict KVAdmitVerdict
	Reason  KVAdmitReason
}

// DecideKVAdmission decides whether an incoming (missed) span CAND should be cached, given the
// cheapest resident VICTIM it would displace under budget pressure and the pool's current GDSF
// aging clock. It admits iff CAND is worth strictly more to keep — per byte — than the victim
// it displaces (the W-TinyLFU "admit only if it beats the eviction candidate" rule), OR its
// value clears the aging clock L (the GDSF admission variant, #2668); otherwise it bypasses,
// leaving the hot resident span in place.
//
// The value axis is exactly KVEvictionCost (kvcost.go), so a candidate and a victim are ranked
// on one comparable scale — a long, frequently-reused, compact span outranks a short, one-shot,
// fat one. Callers signal "no resident victim" (an empty resident set) by passing a zero-value
// victim (Bytes ≤ 0).
//
// FAIL-OPEN on unknowns (never refuse to cache what we cannot prove is a pollutant):
//   - an unpriced candidate (Bytes ≤ 0) always admits;
//   - an empty resident set (victim Bytes ≤ 0) always admits.
//
// The aging-clock path is INERT when clock ≤ 0 (the pool does not age), so with no clock the
// decision reduces to the pure W-TinyLFU victim comparison — exactly the reduction discipline
// KVEvictionCostAged carries for AgeStamp == 0.
func DecideKVAdmission(cand, victim KVSpanStats, clock float64) KVAdmitDecision {
	// Fail-open 1 — unpriced candidate: value undefined, never refuse to cache it.
	if cand.Bytes <= 0 {
		return KVAdmitDecision{Verdict: AdmitCache, Reason: ReasonUnpricedCandidate}
	}
	// Fail-open 2 — empty resident set: caching displaces nothing hot, so admit.
	if victim.Bytes <= 0 {
		return KVAdmitDecision{Verdict: AdmitCache, Reason: ReasonNoResidentVictim}
	}
	candCost := KVEvictionCost(cand)
	victimCost := KVEvictionCost(victim)
	// W-TinyLFU: admit only if the newcomer is worth STRICTLY more to keep than the victim it
	// would replace. Equal value bypasses — never churn residency to swap equal-value spans.
	if candCost > victimCost {
		return KVAdmitDecision{Verdict: AdmitCache, Reason: ReasonOutranksVictim}
	}
	// GDSF admission: a candidate whose value clears the current aging clock L is admitted even
	// against a momentarily-cheaper victim. Gated on clock > 0 so a non-aging pool keeps the
	// pure W-TinyLFU behavior (otherwise every priced candidate would trivially clear clock 0).
	if clock > 0 && candCost >= clock {
		return KVAdmitDecision{Verdict: AdmitCache, Reason: ReasonClearsAgingClock}
	}
	// The candidate is no hotter than the victim and does not clear the clock: caching it would
	// evict a hotter (or equally-hot) resident span for a colder newcomer. Bypass — the
	// scan-resistance win.
	return KVAdmitDecision{Verdict: AdmitBypass, Reason: ReasonColderThanVictim}
}

// KVAdmissionCandidate is one miss the caller is deciding about: the incoming span CAND, the
// cheapest resident VICTIM it would displace (a zero-value victim means "empty resident set"),
// and the pool's current aging clock. Pure data so a resident pool can build it from what it
// already holds (PickEvictionVictim gives the victim; the pool carries the clock), without this
// tier-1 package importing the serve layer above it.
type KVAdmissionCandidate struct {
	Cand   KVSpanStats
	Victim KVSpanStats
	Clock  float64
}

// KVAdmissionPlanItem is one candidate's decision plus its original index, so a caller can map
// the plan back onto its own miss slice without re-deriving order.
type KVAdmissionPlanItem struct {
	Index    int
	Decision KVAdmitDecision
}

// KVAdmissionStats summarises a plan for observability and for the witness — the concrete
// scan-resistance win the gate realises.
type KVAdmissionStats struct {
	Candidates        int // total candidates decided
	Admitted          int // AdmitCache: cached (worth more than the victim, or fail-open)
	Bypassed          int // AdmitBypass: not cached — a colder newcomer refused
	HotSpansProtected int // bypasses whose protected victim was proven-hot (Hits > 0) — the pollutant-refusal win
}

// PlanKVAdmission decides every candidate and rolls the verdicts into per-candidate decisions
// plus aggregate stats. The decisions are index-aligned and 1:1 with the input (every candidate
// is decided exactly once). It is a pure fold over DecideKVAdmission, so it is deterministic for
// a given input and carries the same fail-open / fail-safe contract. A nil/empty input yields a
// nil item slice and a zero stats.
func PlanKVAdmission(cands []KVAdmissionCandidate) ([]KVAdmissionPlanItem, KVAdmissionStats) {
	if len(cands) == 0 {
		return nil, KVAdmissionStats{}
	}
	items := make([]KVAdmissionPlanItem, len(cands))
	stats := KVAdmissionStats{Candidates: len(cands)}
	for i, c := range cands {
		d := DecideKVAdmission(c.Cand, c.Victim, c.Clock)
		items[i] = KVAdmissionPlanItem{Index: i, Decision: d}
		if d.Verdict == AdmitBypass {
			stats.Bypassed++
			// A bypass that protects a proven-reused resident span is the headline
			// scan-resistance win: the pollutant did NOT evict a hot span.
			if c.Victim.Hits > 0 {
				stats.HotSpansProtected++
			}
			continue
		}
		stats.Admitted++
	}
	return items, stats
}
