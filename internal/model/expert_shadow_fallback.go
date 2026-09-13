package model

import "fmt"

// expert_shadow_fallback.go — mixed-precision cache-miss fallback for the bounded routed-expert
// ring (issue #1299, epic #5606). HOBBIT (arXiv 2411.01433) showed that replacing a cache-missed
// expert with a low-precision copy cuts load latency while preserving accuracy. On a Halo an NVMe
// miss costs ~ bytes / 7 GB/s and cannot overlap for a single token, so a RESIDENT low-precision
// "shadow" copy of a frequently-missed expert turns the miss from an unavoidable stall into a
// cheap approximate compute.
//
// What this rung closes. The ring (paging_ring.go, expert_ring_hal.go) has exactly one miss path:
// mk() builds the host source, it is uploaded, and if the budget refuses the caller falls back to
// PERMANENT halW residency. Every miss therefore pays full byte movement — there is no seam where a
// cheaper approximate answer could be substituted, and no accounting of what that substitution
// would cost in quality. This file is that seam: a bounded shadow cache of low-precision copies for
// the miss-HOT experts, and a per-miss policy that selects exact-from-disk vs approximate-from-shadow
// under an NMSE quality guard.
//
// Default-off, byte-for-byte. Nothing here is consulted unless a session explicitly enables it; the
// exact miss path is unchanged when the shadow cache is empty or disabled. Fail-closed: a shadow is
// used only when its NMSE is MEASURED and within the bound; an unmeasured or over-bound shadow is
// never selected, and the miss resolves exactly as it did before this file existed.
//
// Gold-plating boundary (issue #1299): no new quantization kernels — a shadow is described by the
// bytes an EXISTING quant path would produce, and the shadow's NMSE is supplied by the caller that
// ran that path; no quality benchmarking at scale; no change to the calibration budget. This file is
// the policy + bounded accounting; the copy itself reuses the existing quant paths.

// ShadowNMSECeiling is the expert NMSE bound a shadow must beat to be eligible. It is the same 0.20
// ceiling the ggufload quant-integrity validator applies to expert payloads
// (quantIntegrityExpertNMSECeiling) and the bound issue #1299 states, so a shadow measured by the
// artifact builder and selected here is judged by one shared number.
const ShadowNMSECeiling = 0.20

// ShadowExpert is one low-precision copy of a routed expert held in the shadow cache. It carries the
// IDENTITY (layer, expert), the resident byte cost the shadow cache budgets, the measured NMSE of the
// copy against the exact expert, and a per-shadow last-use clock the eviction policy ranks on.
//
// NMSE is a MEASURED value (never inferred) and shadows are admitted only through AdmitShadowExpert,
// which applies the guard, so an ExpertShadow with NMSEMeasured=false can never reach the policy's
// selection path.
type ShadowExpert struct {
	Layer  int `json:"layer"`
	Expert int `json:"expert"`
	// Bytes is the device-resident footprint of the low-precision copy — the shadow cache budgets
	// this, exactly as the ring budgets the exact weight.
	Bytes int64 `json:"bytes"`
	// NMSE is the measured normalized mean-squared error of this copy against the exact expert, and
	// NMSEMeasured records that the measurement actually happened. A zero NMSE with
	// NMSEMeasured=true is a perfect copy; NMSEMeasured=false is UNMEASURED and is never eligible.
	NMSE         float64 `json:"nmse"`
	NMSEMeasured bool    `json:"nmse_measured"`
	// lastUse is the shadow cache's recency clock value at this shadow's most recent use. It is
	// unexported: eviction order is the cache's own deterministic ranking, not caller state.
	lastUse uint64
}

// shadowKey identifies one routed expert within a layer.
type shadowKey struct {
	layer  int
	expert int
}

// ShadowCacheStats is the shadow cache's observable accounting. It is the object an operator reads
// to answer "is the shadow cache doing anything, and is it staying bounded?" — a cache with no
// budget, footprint, hit and eviction counters cannot answer that.
type ShadowCacheStats struct {
	// Enabled is false for the zero cache (default-off), where the honest answer is "no shadow path
	// runs". Bytes and Count are the instantaneous footprint; PeakBytes/PeakCount the high-water
	// marks, so PeakBytes <= BudgetBytes is the boundedness claim.
	Enabled     bool  `json:"enabled"`
	BudgetBytes int64 `json:"budget_bytes"`
	Bytes       int64 `json:"bytes"`
	Count       int   `json:"count"`
	PeakBytes   int64 `json:"peak_bytes"`
	PeakCount   int   `json:"peak_count"`
	// Hits are policy resolutions that selected a shadow, Exact the ones that fell through to the
	// exact miss path, GuardRejections the ones a shadow existed for but the NMSE guard refused.
	// Hits+Exact+GuardRejections == Lookups is the reconciliation identity.
	Hits            int `json:"hits"`
	Exact           int `json:"exact"`
	GuardRejections int `json:"guard_rejections"`
	Lookups         int `json:"lookups"`
	Evictions       int `json:"evictions"`
	Admissions      int `json:"admissions"`
}

// ShadowCache is the bounded set of low-precision expert copies the miss policy selects among. It is
// a PURE accounting object — it holds descriptors, not device handles — so the policy and the bound
// are unit-testable without a GPU, exactly as expert_spill_fit.go's sizing math is.
//
// The zero value is a disabled cache: BudgetBytes 0 admits nothing and every miss resolves exactly.
// A negative budget is treated as 0 (refuse everything) rather than unbounded.
type ShadowCache struct {
	budget  int64
	shadows map[shadowKey]ShadowExpert
	clock   uint64

	bytes     int64
	peakBytes int64
	peakCount int

	hits            int
	exact           int
	guardRejections int
	lookups         int
	evictions       int
	admissions      int
}

// NewShadowCache returns a shadow cache bounded to budgetBytes. A budget <= 0 returns a disabled
// cache — the default-off object — so a caller need not special-case "no shadow".
func NewShadowCache(budgetBytes int64) *ShadowCache {
	c := &ShadowCache{shadows: map[shadowKey]ShadowExpert{}}
	if budgetBytes > 0 {
		c.budget = budgetBytes
	}
	return c
}

// Enabled reports whether this cache can hold any shadow. A disabled cache never admits and never
// selects, so every miss takes the unchanged exact path.
func (c *ShadowCache) Enabled() bool { return c != nil && c.budget > 0 }

// Stats returns the cache's observable accounting.
func (c *ShadowCache) Stats() ShadowCacheStats {
	if c == nil {
		return ShadowCacheStats{}
	}
	return ShadowCacheStats{
		Enabled:         c.Enabled(),
		BudgetBytes:     c.budget,
		Bytes:           c.bytes,
		Count:           len(c.shadows),
		PeakBytes:       c.peakBytes,
		PeakCount:       c.peakCount,
		Hits:            c.hits,
		Exact:           c.exact,
		GuardRejections: c.guardRejections,
		Lookups:         c.lookups,
		Evictions:       c.evictions,
		Admissions:      c.admissions,
	}
}

// ShadowAdmitResult is the verdict of one AdmitShadowExpert call. Admitted is false whenever the
// guard or the bound refused, and Reason names which, so a caller can distinguish "the copy was too
// imprecise" from "the cache was too full" — two facts that demand different responses.
type ShadowAdmitResult struct {
	Admitted bool   `json:"admitted"`
	Reason   string `json:"reason"`
	// NMSE is the measured NMSE the guard judged, echoed for the caller's receipt.
	NMSE float64 `json:"nmse"`
	// Evicted lists the shadows displaced to make room, in eviction order.
	Evicted []ShadowExpert `json:"evicted,omitempty"`
}

// AdmitShadowExpert offers a measured low-precision copy to the cache. It FAILS CLOSED against the
// NMSE guard: an unmeasured copy (NMSEMeasured=false) or one whose NMSE exceeds ShadowNMSECeiling is
// refused and nothing is stored, so a shadow can never enter the cache without a proven quality
// bound. A copy larger than the whole budget is refused (it could never be resident).
//
// On admission the cache evicts its coldest shadows — least-recently used, ties broken by
// (layer, expert) so the choice is deterministic under Go's randomized map iteration — until the
// new copy fits. The new copy is treated as just-used.
func (c *ShadowCache) AdmitShadowExpert(s ShadowExpert) ShadowAdmitResult {
	if c == nil || !c.Enabled() {
		return ShadowAdmitResult{Admitted: false, Reason: "shadow cache disabled"}
	}
	if s.NMSE < 0 || !finiteFloat(s.NMSE) {
		return ShadowAdmitResult{Admitted: false, Reason: fmt.Sprintf("shadow NMSE %.6g is not a finite measurement", s.NMSE)}
	}
	if !s.NMSEMeasured {
		return ShadowAdmitResult{Admitted: false, Reason: "shadow NMSE unmeasured"}
	}
	if s.NMSE > ShadowNMSECeiling {
		return ShadowAdmitResult{Admitted: false, NMSE: s.NMSE,
			Reason: fmt.Sprintf("shadow NMSE %.6g exceeds ceiling %.6g", s.NMSE, ShadowNMSECeiling)}
	}
	if s.Bytes <= 0 {
		return ShadowAdmitResult{Admitted: false, Reason: fmt.Sprintf("shadow bytes %d must be positive", s.Bytes)}
	}
	if s.Bytes > c.budget {
		return ShadowAdmitResult{Admitted: false, NMSE: s.NMSE,
			Reason: fmt.Sprintf("shadow bytes %d exceed cache budget %d", s.Bytes, c.budget)}
	}
	key := shadowKey{s.Layer, s.Expert}
	// A re-admission of an existing expert replaces its copy (a fresher measurement), so the
	// footprint is adjusted rather than double-counted.
	if old, ok := c.shadows[key]; ok {
		c.bytes -= old.Bytes
		delete(c.shadows, key)
	}
	evicted := c.evictToFit(s.Bytes)
	c.clock++
	s.lastUse = c.clock
	c.shadows[key] = s
	c.bytes += s.Bytes
	c.admissions++
	if c.bytes > c.peakBytes {
		c.peakBytes = c.bytes
	}
	if len(c.shadows) > c.peakCount {
		c.peakCount = len(c.shadows)
	}
	return ShadowAdmitResult{Admitted: true, NMSE: s.NMSE, Evicted: evicted}
}

// evictToFit drops least-recently-used shadows until weightBytes more would fit. It returns the
// evicted descriptors in the order they left. The new copy has not yet been assigned a clock, so it
// can never be mistaken for the oldest.
func (c *ShadowCache) evictToFit(weightBytes int64) []ShadowExpert {
	var evicted []ShadowExpert
	for c.bytes+weightBytes > c.budget && len(c.shadows) > 0 {
		var (
			victim shadowKey
			found  bool
		)
		for k, s := range c.shadows {
			if !found || s.lastUse < c.shadows[victim].lastUse ||
				(s.lastUse == c.shadows[victim].lastUse && shadowKeyLess(k, victim)) {
				victim, found = k, true
			}
		}
		if !found {
			break
		}
		evicted = append(evicted, c.shadows[victim])
		c.bytes -= c.shadows[victim].Bytes
		delete(c.shadows, victim)
		c.evictions++
	}
	return evicted
}

// shadowKeyLess is the deterministic tie-break for equal last-use: lower layer, then lower expert.
func shadowKeyLess(a, b shadowKey) bool {
	if a.layer != b.layer {
		return a.layer < b.layer
	}
	return a.expert < b.expert
}

// ShadowLookup is the result of asking the cache for one expert's shadow: present is true only when
// a stored, guard-passing copy exists. It marks the copy used (advancing the eviction clock) so the
// lookup is the one place recency is earned.
func (c *ShadowCache) ShadowLookup(layer, expert int) (ShadowExpert, bool) {
	if c == nil || !c.Enabled() {
		return ShadowExpert{}, false
	}
	s, ok := c.shadows[shadowKey{layer, expert}]
	if !ok {
		return ShadowExpert{}, false
	}
	c.clock++
	s.lastUse = c.clock
	c.shadows[shadowKey{layer, expert}] = s
	return s, true
}

// ExpertMissPolicy is the per-miss selection rule. It answers the issue's core question — on a ring
// miss, pay the exact-from-disk latency, or compute the approximate shadow? — against a STATED
// latency/quality rule rather than a belief about which is faster.
//
// Default-off: the zero value has Enabled=false and every miss resolves exactly.
type ExpertMissPolicy struct {
	// Enabled turns the shadow path on. False (the zero value) is the pre-#1299 behavior exactly.
	Enabled bool
	// Cache is the bounded shadow store this policy selects from. It may be nil or empty, in which
	// case every miss resolves exactly (nothing to approximate with).
	Cache *ShadowCache
	// ShadowComputeCostBytes is the EQUIVALENT device-byte-movement cost of computing from the
	// shadow, expressed in the same units as the exact load. A shadow compute is cheap but not free,
	// and stating its cost in load-byte-equivalents is what lets the two be compared without a clock.
	ShadowComputeCostBytes int64
}

// ExpertMissResolution names the two closed outcomes of a miss.
type ExpertMissResolution int

const (
	// MissResolveExact loads the exact weight from its backing store — the unchanged path, and the
	// fail-closed outcome whenever the shadow is absent, unmeasured, over-bound, or not cheaper.
	MissResolveExact ExpertMissResolution = iota
	// MissResolveShadow computes from the resident low-precision copy.
	MissResolveShadow
)

func (r ExpertMissResolution) String() string {
	if r == MissResolveShadow {
		return "shadow"
	}
	return "exact"
}

// ExpertMissRequest is one ring miss presented to the policy: which expert missed and what the exact
// load would cost in device bytes and load latency, so the decision can be argued from the values in
// front of it.
type ExpertMissRequest struct {
	Layer  int
	Expert int
	// ExactLoadBytes / ExactLoadLatencyNanos are the exact-from-disk cost this miss would pay.
	ExactLoadBytes        int64
	ExactLoadLatencyNanos int64
}

// ExpertMissDecision is the sealed outcome: which resolution was chosen, the measured NMSE behind a
// shadow choice (0 for an exact choice), and a Reason that is ALWAYS populated — a policy that only
// explains its shadow choices turns every exact fallback into an unexplained default.
type ExpertMissDecision struct {
	Resolution ExpertMissResolution `json:"resolution"`
	// NMSE is the quality of the shadow actually selected; it is 0 when Resolution is exact.
	NMSE        float64 `json:"nmse"`
	ShadowBytes int64   `json:"shadow_bytes"`
	// LatencySavedNanos is the exact-load latency the shadow choice avoids, or 0 for exact.
	LatencySavedNanos int64  `json:"latency_saved_nanos"`
	Reason            string `json:"reason"`
}

// ResolveExpertMiss is the policy seam: given one ring miss, it selects exact-from-disk or
// approximate-from-shadow. It resolves SHADOW only when ALL of the following hold, so no single
// missing fact can silently degrade the answer:
//
//  1. the policy is enabled and the cache holds a shadow for this expert;
//  2. the shadow's NMSE is measured and within ShadowNMSECeiling (the cache already enforces this on
//     admission, and the policy re-checks it, because a decision this consequential must not depend
//     on the admission path having been the only writer);
//  3. the shadow compute is strictly cheaper than the exact load under the stated cost rule.
//
// Anything else resolves EXACT with a reason naming the failing clause. Because the cache only ever
// stores guard-passing copies, clause 2 is a defense-in-depth re-check, not the guard itself.
func (p ExpertMissPolicy) ResolveExpertMiss(req ExpertMissRequest) ExpertMissDecision {
	if !p.Enabled {
		return ExpertMissDecision{Resolution: MissResolveExact, Reason: "shadow fallback disabled"}
	}
	s, ok := p.Cache.ShadowLookup(req.Layer, req.Expert)
	if !ok {
		p.Cache.noteLookup(MissResolveExact, false)
		return ExpertMissDecision{Resolution: MissResolveExact,
			Reason: fmt.Sprintf("no shadow held for expert %d in layer %d", req.Expert, req.Layer)}
	}
	if !s.NMSEMeasured || !finiteFloat(s.NMSE) || s.NMSE < 0 || s.NMSE > ShadowNMSECeiling {
		p.Cache.noteLookup(MissResolveExact, true)
		return ExpertMissDecision{Resolution: MissResolveExact, NMSE: s.NMSE, ShadowBytes: s.Bytes,
			Reason: fmt.Sprintf("shadow NMSE %.6g not within measured ceiling %.6g; exact (fail closed)",
				s.NMSE, ShadowNMSECeiling)}
	}
	shadowCost := p.ShadowComputeCostBytes
	if shadowCost < 0 {
		shadowCost = 0
	}
	if shadowCost >= req.ExactLoadBytes {
		p.Cache.noteLookup(MissResolveExact, false)
		return ExpertMissDecision{Resolution: MissResolveExact, NMSE: s.NMSE, ShadowBytes: s.Bytes,
			Reason: fmt.Sprintf("shadow compute %d B not cheaper than exact load %d B; exact",
				shadowCost, req.ExactLoadBytes)}
	}
	p.Cache.noteLookup(MissResolveShadow, false)
	return ExpertMissDecision{
		Resolution:        MissResolveShadow,
		NMSE:              s.NMSE,
		ShadowBytes:       s.Bytes,
		LatencySavedNanos: req.ExactLoadLatencyNanos,
		Reason: fmt.Sprintf("shadow NMSE %.6g <= %.6g and compute %d B < exact load %d B; %d ns avoided",
			s.NMSE, ShadowNMSECeiling, shadowCost, req.ExactLoadBytes, req.ExactLoadLatencyNanos),
	}
}

// noteLookup folds one resolution into the cache's reconciliation counters. guardReject is true only
// when a guard rejected an EXISTING shadow, which is the fact the GuardRejections counter exists to
// surface; a plain "no shadow held" is an exact fallback, not a guard rejection.
func (c *ShadowCache) noteLookup(r ExpertMissResolution, guardReject bool) {
	if c == nil {
		return
	}
	c.lookups++
	switch {
	case r == MissResolveShadow:
		c.hits++
	case guardReject:
		c.guardRejections++
	default:
		c.exact++
	}
}

// finiteFloat reports whether v is neither NaN nor an infinity.
func finiteFloat(v float64) bool {
	return v == v && v-v == 0
}
