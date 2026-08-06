package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// expert_ring_hal.go — R0 of the activated-expert offload ladder (#5611, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): put the bounded pagedRing UNDER the session weight HAL for
// ROUTED expert weights, so a session's device residency for the expert bulk becomes a declared,
// bounded, observable object instead of an emergent side effect of a memo map.
//
// The problem this closes. A frontier MoE checkpoint activates a small fraction of its experts per
// token (GLM-5.2: top-8 of 256, ~3%), so the gap between STORED expert bytes and ACTIVATED expert
// bytes is the whole opportunity — and it is a RESIDENCY problem. But the serve path offered only
// two placements, both unbounded:
//
//   - --cpu-offload-experts → splitKernel (moe_offload.go): every expert weight runs on the HOST
//     kernel for the life of the session. No activated expert is ever device-resident — the floor,
//     not a cache.
//   - the device path → weightHALStaged (hal.go): every uploaded weight handle is memoized in halW
//     and NEVER evicted (halW is torn down only at Close). Device residency is therefore the union of
//     every expert activated since the session started, monotonically converging on the full expert
//     bulk. A memoizer, not a cache: no budget, no victim, no accounting.
//
// Neither has a bounded activated working set, so there was no seam where a residency policy, a
// prefetch, a pin-set or a byte budget could attach on the live path. This file is that seam.
//
// What it does. When a session declares ExpertRingBytes > 0, staging a ROUTED expert weight
// (isRoutedExpertWeight — `.mlp.experts.N.*`, NOT `.mlp.shared_experts.*`, keeping the #3212
// distinction) goes through pagedRing.stage instead of halW: the handle is admitted under the byte
// budget, the coldest unpinned expert is evicted and Freed to make room, and an evicted expert pages
// back IN on its next activation. Every other weight — dense projections, attention, the router,
// lm_head, and the SHARED expert every token uses — keeps its permanent halW residency, because those
// are activated every token and evicting them could only cost.
//
// Prior-art: activation-aware expert orchestration over a bounded GPU tier is ktransformers'
// GPU/CPU expert placement (https://github.com/kvcache-ai/ktransformers) and the 3-tier
// GPU/pinned-CPU/SSD expert cache the MoE-offload literature converged on; the route here is
// "borrow", not scratch-build. The distinct axis is DeepSeek EPLB, which balances experts ACROSS
// ranks rather than bounding residency WITHIN one device (#3886) — see the plan's non-goals.
//
// Why it reuses rather than invents. The byte budget, the LRU victim choice, the pinned exemption and
// the all-or-nothing admit are polymodel.Pool's — the same policy internal/residency uses for whole
// models — so `used <= budget` and pinned-never-evicted hold here by construction. pagedRing already
// witnesses that a ring-served GEMM is BIT-EQUAL to a resident one on both the hit and the miss path;
// stage() is that same lifecycle with the GEMM removed, so bit-exactness transfers to the HAL.
//
// Default is OFF, byte-for-byte. ExpertRingBytes defaults to 0, routedExpertRing then returns nil,
// and every weight takes the unchanged weightHALStaged path — so no existing session, witness or
// benchmark moves. The operator knob that SIZES the budget (--n-cpu-moe N / auto against measured
// device headroom, on the sizing math expert_spill_fit.go already ships) is R1/#5612; consulting the
// warm-start pin-set instead of a static pinned=false is R2/#5613; the async prefetch on the miss
// path is R3/#5614; the policy seam that replaces LRU on measured Belady regret is R4/#5615; the
// checkpoint tier under a miss is R5/#5616; the operator verb over ExpertRing() is R6/#5617.

// routedExpertRing returns the session's bounded routed-expert residency ring when `name` is a routed
// expert weight and this session declared a budget — nil otherwise, which is the default and which
// makes every caller fall through to the unchanged unbounded halW path. The ring is built lazily on
// the first routed-expert stage, so a session that never reaches an expert weight allocates nothing.
func (s *Session) routedExpertRing(name string) *pagedRing {
	if s == nil || s.ExpertRingBytes <= 0 || s.Backend == nil || s.halClosed {
		return nil
	}
	if !isRoutedExpertWeight(name) {
		return nil
	}
	if s.expertRing == nil {
		s.expertRing = newPagedRing(s.Backend, s.ExpertRingBytes)
		// R4/#5615: the victim ranking is fixed for the ring's whole life, because changing it
		// mid-flight would score one window under two policies and make the next measurement
		// uninterpretable. The zero value is LRU, so this is inert unless an operator promoted the
		// candidate on measured evidence (SelectExpertRingEvictPolicy).
		s.expertRing.policy = s.ExpertRingEvict
		// R2/#5613: seed the durable pin-set before the first staging, so turn 1 already pins the
		// prior's hot set. A session declaring neither knob gets no pin-set and the plain-LRU ring.
		s.warmStartExpertPins(s.expertRing)
	}
	return s.expertRing
}

// weightHALStagedBounded is weightHALStaged plus the routed-expert residency bound. `key` is the
// dtype-prefixed HAL cache key (so a Q4_K and a raw k-quant staging of the same tensor stay distinct
// residents, exactly as in halW); `name` is the canonical tensor name the routed-expert predicate
// reads; weightBytes is the RESIDENT device footprint the budget accounts (the QUANTIZED size, not an
// f32 expansion of it — Q4_K ~0.56 B/weight, so a ring sized for N f32 experts holds several times as
// many quantized ones).
//
// Order of resolution: an already-permanent halW resident wins (a weight promoted before the ring
// existed, or one the ring refused, is never staged twice); then the ring, if this is a routed expert
// under a budget; then the unchanged permanent staging. The ring's refusal path — a single weight
// larger than the whole budget (ErrTooLarge), or one that fits only by dropping a pinned resident
// (ErrPinnedNoRoom) — leaves the ring untouched and falls back to permanent residency rather than
// failing the forward: correctness never depends on the budget being generous. That fallback rebuilds
// the host source (stage already Freed its upload), which is the honest cost of a misconfigured
// budget and is rare by construction.
func (s *Session) weightHALStagedBounded(key, name string, mk func() compute.Tensor, dtype compute.Dtype, weightBytes int64) compute.Tensor {
	if s.halW != nil {
		if t, ok := s.halW[key]; ok {
			return t
		}
	}
	if r := s.routedExpertRing(name); r != nil {
		// R7/#5618: under a SHARED ring these four ring touches must not interleave with a peer
		// agent's staging, so they run inside one span. The span is reentrant, so the demand path's
		// outer stage-and-hold span (hal.go) nests here without deadlocking, and it is a no-op under
		// the per-session default — byte-for-byte the pre-R7 path.
		done := s.ringEnter(r)
		defer done()
		// R2/#5613: `pinned` comes from the durable, workload-personalized pin-set rather than the
		// constant false R0 passed, and the staging is folded into this turn's usage histogram so
		// the actuator repins against real routing. A session with no pin-set answers false and
		// observes nothing, which is R0's plain-LRU ring byte-for-byte.
		pinned := false
		if layer, expert, ok := routedExpertIdentity(name); ok {
			pinned = r.isExpertPinned(layer, expert)
			r.observeExpert(layer, expert)
			// R4/#5615: the ORDERED record the offline regret gauge replays. Independent of the
			// pin knobs — a session may want to MEASURE which victim policy its workload deserves
			// without running a pin-set at all.
			r.observeTrace(layer, expert, weightBytes)
		}
		if t, ok := r.stage(key, mk, dtype, weightBytes, pinned); ok {
			return t
		}
	}
	return s.weightHALStaged(key, mk, dtype)
}

// q4kResidentBytes / kQuantResidentBytes / q8ResidentBytes report the DEVICE-resident footprint of one
// staged weight, which is what the ring budget accounts. Q4_K and the raw k-quants stage their GGUF
// super-blocks verbatim, so the resident size is exactly the raw byte length; Q8_0 stages the int8
// codes plus the f32 per-block scales. A nil tensor reports 0 — the staged builder raises the uniform
// "missing weight" panic on its own, and the ring must not be handed a negative or guessed size.
func q4kResidentBytes(qt *q4kTensor) int64 {
	if qt == nil {
		return 0
	}
	return int64(len(qt.raw))
}

func kQuantResidentBytes(qt *kQuantTensor) int64 {
	if qt == nil {
		return 0
	}
	return int64(len(qt.raw))
}

func q8ResidentBytes(qt *q8Tensor) int64 {
	if qt == nil {
		return 0
	}
	return int64(len(qt.q)) + 4*int64(len(qt.d))
}

// ExpertRingStats is the activated-expert residency of one session, made countable. It is the object
// the plan's thesis asks for: "were this token's activated experts resident?" is answerable only if
// residency has a budget, a footprint and a hit/page-in/evict ledger. Enabled=false means this session
// runs the unbounded halW path, where the honest answer is "residency is whatever accumulated".
type ExpertRingStats struct {
	Enabled bool `json:"enabled"`
	// BudgetBytes is the declared ceiling; ResidentBytes the instantaneous footprint; PeakBytes the
	// high-water mark. PeakBytes <= BudgetBytes is the boundedness claim, and Evictions > 0 is what
	// proves the bound was actually exercised rather than merely never reached.
	BudgetBytes   int64 `json:"budget_bytes"`
	ResidentBytes int64 `json:"resident_bytes"`
	PeakBytes     int64 `json:"peak_bytes"`
	ResidentCount int   `json:"resident_count"`
	// PageIns are cold uploads, Hits resident reuses, Evictions page-outs. Hits/(Hits+PageIns) is the
	// activated-set hit rate a budget is tuned against.
	PageIns   int `json:"page_ins"`
	Hits      int `json:"hits"`
	Evictions int `json:"evictions"`
	// PinnedCount is how many experts the durable pin-set (R2/#5613) currently holds exempt from
	// eviction. 0 means plain LRU — either no pin-set was declared, or the prior was empty.
	PinnedCount int `json:"pinned_count"`
	// Prefetched is how many weights the activated-set prefetch (R3/#5614) staged ahead of their
	// GEMM. ActivatedExperts / ActivatedCovered are the COVERAGE meter: of the experts the router
	// activated and this ring could serve, how many the budget could actually hold. Covered <
	// Activated is the direct read on an undersized ring — the tuning signal a hit rate alone cannot
	// give, because a ring too small to hold one layer's top-k reports honest misses forever.
	Prefetched       int `json:"prefetched"`
	ActivatedExperts int `json:"activated_experts"`
	ActivatedCovered int `json:"activated_covered"`
	// AsyncOverlapped / AsyncWaited are the #5627 overlap meter, and they are populated ONLY on a
	// backend advertising compute.AsyncUploader — both stay 0 on cpu-ref and on every synchronous
	// backend, where a staged weight is visible the moment it is staged and there is nothing to
	// overlap. Overlapped counts transfers already landed when their weight was demanded (the bytes
	// moved underneath other work); Waited counts those the demand caught up with and paid for on
	// the critical path. A transfer evicted before it was ever demanded is in neither: nothing
	// computed against it, so it says nothing about overlap.
	AsyncOverlapped int `json:"async_overlapped"`
	AsyncWaited     int `json:"async_waited"`
}

// AsyncOverlapFraction is the share of fenced activated-expert transfers that had already landed
// when their weight was demanded — the number R3/#5614 asked for and could not produce, because
// before #5627 there was no handle to attribute a transfer to.
//
// It returns 0 when nothing was fenced, which is the honest reading for a synchronous backend:
// not "no overlap was achieved" but "overlap is not a thing this backend can report". Read it
// alongside AsyncOverlapped+AsyncWaited, which is the denominator, exactly as the coverage meter
// is read against ActivatedExperts.
func (s ExpertRingStats) AsyncOverlapFraction() float64 {
	fenced := s.AsyncOverlapped + s.AsyncWaited
	if fenced <= 0 {
		return 0
	}
	return float64(s.AsyncOverlapped) / float64(fenced)
}

// ExpertRing reports this session's bounded routed-expert residency. It returns the zero value (in
// particular Enabled=false) for a session with no ring — either because no budget was declared or
// because no routed expert has been staged yet.
// A session attached to a SHARED ring (R7/#5618) reports that ring's AGGREGATE residency — the
// budget, footprint and ledger of every attached agent together — because that is what bounds this
// session's experts. The per-agent split lives in SharedExpertRing.Stats.
func (s *Session) ExpertRing() ExpertRingStats {
	if s == nil || s.expertRing == nil {
		return ExpertRingStats{}
	}
	r := s.expertRing
	done := s.ringEnter(r)
	defer done()
	return r.stats()
}

// stats is the ring's own residency account, split out from Session.ExpertRing so the shared-ring
// owner (SharedExpertRing.Stats) can read the same numbers under its own lock without going through
// a session that may not exist any more.
func (r *pagedRing) stats() ExpertRingStats {
	return ExpertRingStats{
		Enabled:       true,
		BudgetBytes:   r.budget(),
		ResidentBytes: r.used(),
		PeakBytes:     r.peakUsed(),
		ResidentCount: r.residentCount(),
		PageIns:       r.pageIn,
		Hits:          r.hit,
		Evictions:     r.evict,
		PinnedCount:   r.pins.Len(),

		Prefetched:       r.prefetched,
		ActivatedExperts: r.activatedExperts,
		ActivatedCovered: r.activatedCovered,
		AsyncOverlapped:  r.asyncOverlapped,
		AsyncWaited:      r.asyncWaited,
	}
}
