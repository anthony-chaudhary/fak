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
		// pinned=false: R0 gives the ring plain LRU. Consulting the warm-start pin-set
		// (pagedRing.isExpertPinned, expert_warmpins.go) is R2/#5613.
		if t, ok := r.stage(key, mk, dtype, weightBytes, false); ok {
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
}

// ExpertRing reports this session's bounded routed-expert residency. It returns the zero value (in
// particular Enabled=false) for a session with no ring — either because no budget was declared or
// because no routed expert has been staged yet.
func (s *Session) ExpertRing() ExpertRingStats {
	if s == nil || s.expertRing == nil {
		return ExpertRingStats{}
	}
	r := s.expertRing
	return ExpertRingStats{
		Enabled:       true,
		BudgetBytes:   r.budget(),
		ResidentBytes: r.used(),
		PeakBytes:     r.peakUsed(),
		ResidentCount: r.residentCount(),
		PageIns:       r.pageIn,
		Hits:          r.hit,
		Evictions:     r.evict,
	}
}
