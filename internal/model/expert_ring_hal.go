package model

import (
	"errors"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

var errUntrackedExpertResidency = errors.New("routed expert has permanent residency outside the declared ring budget")

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
// the all-or-nothing admit are polymodel.Pool's, so `used <= budget` and pinned-never-evicted hold
// here by construction. pagedRing already witnesses that a ring-served GEMM is BIT-EQUAL to a
// resident one on both the hit and the miss path; stage() is that same lifecycle with the GEMM
// removed, so bit-exactness transfers to the HAL.
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
	if s == nil || !isRoutedExpertWeight(name) {
		return nil
	}
	s.validateCheckpointDeviceRingBudget(name)
	if s.ExpertRingBytes <= 0 || s.Backend == nil || s.halClosed {
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

func (s *Session) validateCheckpointDeviceRingBudget(name string) int64 {
	if s == nil || s.M == nil {
		return 0
	}
	declared := s.M.expertCheckpoint.DeviceRingBudget()
	if declared <= 0 {
		return 0
	}
	s.ensureOpenBackendSession()
	if s.ExpertRingBytes != declared {
		s.failExpertRing(s.expertRing, &ExpertCheckpointDeviceBudgetError{Tensor: name, Bytes: declared, Budget: s.ExpertRingBytes})
	}
	if s.expertRing != nil && s.expertRing.budget() != declared {
		s.failExpertRing(s.expertRing, &ExpertCheckpointDeviceBudgetError{Tensor: name, Bytes: declared, Budget: s.expertRing.budget()})
	}
	if s.Backend == nil {
		s.failExpertRing(s.expertRing, errUntrackedExpertResidency)
	}
	return declared
}

// weightHALStagedBounded is weightHALStaged plus the routed-expert residency bound. `key` is the
// dtype-prefixed HAL cache key (so a Q4_K and a raw k-quant staging of the same tensor stay distinct
// residents, exactly as in halW); `name` is the canonical tensor name the routed-expert predicate
// reads; weightBytes is the RESIDENT device footprint the budget accounts (the QUANTIZED size, not an
// f32 expansion of it — Q4_K ~0.56 B/weight, so a ring sized for N f32 experts holds several times as
// many quantized ones).
//
// A positive checkpoint-tier budget rejects permanent routed-expert residency and ring refusal
// under either execution policy. Device-only sessions also enforce their session-declared budget.
// Portable sessions without a checkpoint-tier budget retain the permanent-residency fallback.
func (s *Session) weightHALStagedBounded(key, name string, mk func() compute.Tensor, dtype compute.Dtype, weightBytes int64) compute.Tensor {
	declared := int64(0)
	if isRoutedExpertWeight(name) {
		declared = s.validateCheckpointDeviceRingBudget(name)
	}
	strict := isRoutedExpertWeight(name) && (declared > 0 || s.executionPolicy == ExecutionPolicyDeviceOnly && s.ExpertRingBytes > 0)
	if strict {
		s.ensureOpenBackendSession()
	}
	if s.halW != nil {
		if t, ok := s.halW[key]; ok {
			if strict {
				cause := errUntrackedExpertResidency
				if weightBytes > s.ExpertRingBytes {
					cause = errors.Join(cause, polymodel.ErrTooLarge)
				}
				s.failExpertRing(s.expertRing, cause)
			}
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
		if t, err := r.stageWithError(key, mk, dtype, weightBytes, pinned); err == nil {
			return t
		} else if strict {
			s.failExpertRing(r, err)
		}
	}
	// A routed expert that is outside a ring or refused by its bound remains session-local:
	// sharing it here would turn the deliberately bounded expert tier into an unbounded
	// model-lifetime memoizer. Dense/shared weights use model-lifetime immutable residency.
	if isRoutedExpertWeight(name) {
		if s.halW != nil {
			if t, ok := s.halW[key]; ok {
				return t
			}
		}
		t := s.Backend.Upload(mk(), dtype)
		if s.halW != nil {
			s.halW[key] = t
		}
		return t
	}
	return s.weightHALStaged(key, mk, dtype)
}

func (s *Session) failExpertRing(r *pagedRing, cause error) {
	backend := ""
	if s.Backend != nil {
		backend = s.Backend.Name()
	}
	err := &BackendForwardOperationError{
		Backend: backend,
		Forward: s.deviceOnlyForward(),
		Path:    "expert-ring",
		Layer:   -1,
		Stage:   "routed expert residency",
		Cause:   cause,
	}
	s.halFailure = err
	s.cancelRingSpan(r)
	s.closeBackendOwned()
	panic(err)
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
	if qt.lazy != nil {
		return int64(qt.lazy.Bytes)
	}
	return int64(len(qt.raw))
}

func kQuantResidentBytes(qt *kQuantTensor) int64 {
	if qt == nil {
		return 0
	}
	if qt.lazy != nil {
		return int64(qt.lazy.Bytes)
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
	// Policy and PolicyGeneration identify the policy-local measurement epoch. The traffic and
	// residency counters below remain cumulative across explicit policy swaps.
	Policy           ExpertRingEvictPolicy `json:"policy"`
	PolicyGeneration uint64                `json:"policy_generation"`
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
	// Lookups is every staging request the ring received and Refusals the ones it could not admit
	// (portable callers fall back to permanent residency; device-only callers fail closed). They are the
	// RECONCILIATION pair R6/#5617 reports against: Hits+PageIns+Refusals == Lookups is an identity
	// counted on both sides independently, so an operator surface can prove it is reading the ring's
	// own accounting rather than a parallel estimate. PageInBytes is the device bytes cold uploads
	// actually moved — the numerator of bytes-per-token, which PageIns alone cannot give because
	// routed projections differ in size and quantization.
	Lookups     int   `json:"lookups"`
	Refusals    int   `json:"refusals"`
	PageInBytes int64 `json:"page_in_bytes"`
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
	// StableCapBytes / StableBytes are the scan-resistant stable set (expert_ring_stable.go): the
	// pinned-on-admit ceiling and the bytes it holds. Both 0 when the stable set is off.
	StableCapBytes int64 `json:"stable_cap_bytes,omitempty"`
	StableBytes    int64 `json:"stable_bytes,omitempty"`
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
		Enabled:          true,
		Policy:           r.policy,
		PolicyGeneration: r.policyGeneration,
		BudgetBytes:      r.budget(),
		ResidentBytes:    r.used(),
		PeakBytes:        r.peakUsed(),
		ResidentCount:    r.residentCount(),
		PageIns:          r.pageIn,
		Hits:             r.hit,
		Evictions:        r.evict,
		Lookups:          r.lookups,
		Refusals:         r.refused,
		PageInBytes:      r.pageInBytes,
		StableCapBytes:   r.stableCap,
		StableBytes:      r.stableBytes,
		PinnedCount:      r.pins.Len(),

		Prefetched:       r.prefetched,
		ActivatedExperts: r.activatedExperts,
		ActivatedCovered: r.activatedCovered,
		AsyncOverlapped:  r.asyncOverlapped,
		AsyncWaited:      r.asyncWaited,
	}
}

// --- Paged-expert K-of-E window (borrow-port oss-port-strata-paged-experts) -------
//
// ADAPT of SGLang's paged-experts design (PR #42557@8e7601ae, Apache-2.0 — auto-K sizing, LRU
// waves, bit-identical vs unpaged; slot lookup fused into decide per #46457/#46567) and Strata's
// batched remote-expert-pool discipline (docs/DEEPSEEK41.md:54@46c152a1, MIT), extended onto the
// existing ring seams rather than built in parallel: the bounded pagedRing already holds a
// byte-budgeted working set of routed experts, and this leaf makes that working set an explicit
// K-of-E WINDOW with a planned LRU wave and an observed hit/miss split instead of an emergent
// property of byte-LRU. The window sizes itself from the declared budget (auto-K), plans how many
// of this layer's residents must leave when the step's picks arrive (the wave), and splits each
// pick into a hit (all three projections resident — compute immediately) or a miss (paged in
// through the ring's ordinary demand path, which routes through the CUDA copy stream on
// owner-capable backends). Nothing here stages bytes or touches a handle: dispatch order and math
// are untouched, so a decided layer stays bit-identical to an undecided one. Opt-in via
// FAK_V41_PAGED_EXPERTS=1 at the v41 routed-layer seam; default off is byte-for-byte unchanged.

// PagedExpertWindow states the K-of-E resident expert window from the declared ring budget and
// the measured cost of one whole expert.
type PagedExpertWindow struct {
	Experts        int   // E: routed experts per layer (Config.NumExperts)
	AvgExpertBytes int64 // resident bytes of ONE expert's three projections
	Budget         int64 // the session's declared expert-ring weight-byte budget
}

// K is the auto-sized window: as many whole experts as the budget buys, floor(Budget /
// AvgExpertBytes), clamped to [0, Experts]. A window that cannot be sized honestly — no experts,
// no measured cost, or no budget — answers 0 rather than guessing.
func (w PagedExpertWindow) K() int {
	if w.Experts <= 0 || w.AvgExpertBytes <= 0 || w.Budget <= 0 {
		return 0
	}
	k := int(w.Budget / w.AvgExpertBytes)
	if k > w.Experts {
		return w.Experts
	}
	return k
}

// PagedExpertWavePlan is the LRU wave size for one layer entry. demand is the number of DISTINCT
// experts the layer must hold once the step's picks are admitted — the union of the live ring's
// current residents and this step's incoming picks — and k the window. When demand outgrows the
// window, the excess leaves the ring as coldest unpinned residents; a demand that fits plans no
// wave. The plan is a per-layer prediction: the shared ring's global LRU chooses the actual
// victims, so the observed evictions are the witness, not the plan.
func PagedExpertWavePlan(demand, k int) int {
	if n := demand - k; n > 0 {
		return n
	}
	return 0
}

// PagedExpertSplit is one layer's accumulated paged-expert window ledger across the steps that
// decided it: how many picks the window served on a hit versus paged in on a miss, the demand it
// planned against, and the total LRU wave it predicted.
type PagedExpertSplit struct {
	Layer         int
	K             int // auto-sized window observed on the recorded steps
	Steps         int
	Demand        int // sum over steps of the distinct-expert demand (residents ∪ picks)
	Hits          int // picks whose three projections were all resident at decide time
	Misses        int // picks paged in through the ring's ordinary demand path
	WaveEvictions int // sum over steps of the planned LRU wave sizes
}

// pagedExpertState is the package-level window ledger, keyed by *Session exactly like
// crossLayerGateState: one mutex guards the map, entries are created lazily and only for a
// session that actually decided, so a default-off run allocates nothing. The documented leak
// until Close moves it onto the Session mirrors that precedent.
var pagedExpertState = struct {
	mu     sync.Mutex
	splits map[*Session]map[int]*PagedExpertSplit
}{splits: map[*Session]map[int]*PagedExpertSplit{}}

// PagedExpertSplitsFor returns s's accumulated paged-expert window splits, one entry per decided
// layer in layer order. A nil session, or one that never decided, reports the empty slice. Safe
// to call from another goroutine.
func PagedExpertSplitsFor(s *Session) []PagedExpertSplit {
	if s == nil {
		return nil
	}
	pagedExpertState.mu.Lock()
	defer pagedExpertState.mu.Unlock()
	byLayer := pagedExpertState.splits[s]
	if len(byLayer) == 0 {
		return nil
	}
	out := make([]PagedExpertSplit, 0, len(byLayer))
	for _, sp := range byLayer {
		out = append(out, *sp)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Layer < out[j].Layer })
	return out
}

// ResetPagedExpertSplits clears s's window ledger so a measurement can start from zero. A nil
// session is a no-op.
func ResetPagedExpertSplits(s *Session) {
	if s == nil {
		return
	}
	pagedExpertState.mu.Lock()
	delete(pagedExpertState.splits, s)
	pagedExpertState.mu.Unlock()
}

// recordPagedExpertSplit folds one decided layer entry into s's ledger.
func recordPagedExpertSplit(s *Session, layer, k, demand, hits, misses, wave int) {
	if s == nil {
		return
	}
	pagedExpertState.mu.Lock()
	byLayer := pagedExpertState.splits[s]
	if byLayer == nil {
		byLayer = map[int]*PagedExpertSplit{}
		pagedExpertState.splits[s] = byLayer
	}
	sp := byLayer[layer]
	if sp == nil {
		sp = &PagedExpertSplit{Layer: layer}
		byLayer[layer] = sp
	}
	sp.Steps++
	sp.K = k
	sp.Demand += demand
	sp.Hits += hits
	sp.Misses += misses
	sp.WaveEvictions += wave
	pagedExpertState.mu.Unlock()
}

// ringKeyExpertIdentity recovers the (layer, expert) identity behind a ring resident key. The ring
// keys a routed expert's projections under exactly two dtype prefixes — "q4k:" and "kquant-raw:" —
// built by expertWeight.halKey and the checkpoint tier's halKey to the same rule, so stripping
// either prefix yields the canonical weight name routedExpertIdentity parses. Any other key earns
// no opinion: the scan simply skips it.
func ringKeyExpertIdentity(key polymodel.ModelID) (layer, expert int, ok bool) {
	name := string(key)
	switch {
	case strings.HasPrefix(name, "q4k:"):
		name = name[len("q4k:"):]
	case strings.HasPrefix(name, "kquant-raw:"):
		name = name[len("kquant-raw:"):]
	default:
		return 0, 0, false
	}
	return routedExpertIdentity(name)
}

// pagedExpertDecide runs the K-of-E window's decide step for one layer entry: it sizes the window
// from the ring budget and the picked experts' own byte cost, counts this layer's distinct
// residents against the live ring, plans the LRU wave, and splits the picks into hits and misses.
// It records the split in the session ledger and changes NOTHING else — no bytes are staged, no
// handle is touched, dispatch order and math are untouched — so a decided layer stays bit-identical
// to an undecided one. The reads take their own reentrant ring span.
func (s *Session) pagedExpertDecide(layer int, picks []routePick, r *pagedRing, names [][3]string) {
	if s == nil || r == nil || s.M == nil || len(picks) == 0 || len(names) != len(picks) {
		return
	}
	done := s.ringEnter(r)
	defer done()
	// Auto-K sizing, measured on the first pick's own triple: one expert's three projections.
	var expertBytes int64
	sized := true
	for _, name := range names[0] {
		w, ok := s.resolveExpertWeight(name)
		if !ok {
			sized = false
			break
		}
		switch {
		case w.ck != nil:
			expertBytes += w.ck.bytes
		case w.q4 != nil:
			expertBytes += q4kResidentBytes(w.q4)
		default:
			expertBytes += kQuantResidentBytes(w.kq)
		}
	}
	win := PagedExpertWindow{Experts: s.M.Cfg.NumExperts, Budget: s.ExpertRingBytes}
	if sized && expertBytes > 0 {
		win.AvgExpertBytes = expertBytes
	}
	k := win.K()
	// Distinct experts the layer must hold after the step: the union of the live ring's residents
	// for THIS layer and this step's picks.
	incoming := make(map[int]struct{}, len(picks))
	for _, p := range picks {
		incoming[p.expert] = struct{}{}
	}
	// One expert is THREE resident keys, so the union is counted per DISTINCT expert, not per
	// projection key.
	counted := make(map[int]struct{})
	demand := 0
	for key := range r.resident {
		l, ex, ok := ringKeyExpertIdentity(key)
		if !ok || l != layer {
			continue
		}
		if _, in := incoming[ex]; in {
			continue
		}
		if _, done := counted[ex]; done {
			continue
		}
		counted[ex] = struct{}{}
		demand++
	}
	demand += len(incoming)
	wave := PagedExpertWavePlan(demand, k)
	hits, misses := 0, 0
	for slot := range picks {
		hit := true
		for _, name := range names[slot] {
			w, ok := s.resolveExpertWeight(name)
			if !ok || !r.isResident(w.halKey()) {
				hit = false
				break
			}
		}
		if hit {
			hits++
		} else {
			misses++
		}
	}
	recordPagedExpertSplit(s, layer, k, demand, hits, misses, wave)
}
