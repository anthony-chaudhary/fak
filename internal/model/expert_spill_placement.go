package model

import (
	"sort"
	"strings"
)

// expert_spill_placement.go — R1 of the activated-expert offload ladder (#5612, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): give the graded-spill sizing math a CALLER, and give the
// split kernel a GRADE.
//
// What was missing. Two halves of one knob shipped separately and never met:
//
//   - expert_spill_fit.go (#5281) computes N — how many MoE layers' experts to move to host RAM so
//     the device-resident remainder fits a measured byte budget — but ResolveExpertSpill and
//     AutoFitExpertSpill had no caller outside their own tests. Pure math, no consumer.
//   - splitKernel (moe_offload.go) routes each weight host-or-device by a predicate, but the only
//     predicate it was ever handed is isExpertWeight: a LAYER-BLIND, name-only test. So the live
//     placement was all-or-nothing — every MoE layer's experts on host, or none — with no way to
//     express the N the sizing math computes.
//
// This file joins them. It reads the layer ordinal back out of a canonical tensor name, orders the
// model's MoE layers, and turns "spill the first N" into a predicate splitKernel can run. It also
// builds the ExpertSpillBudget from a LOADED model's actual resident bytes (MoEResidentWeightBytes,
// resident_report.go), so the sizing input is measured rather than estimated, and derives the byte
// budget for R0's routed-expert ring (#5611) from the same arithmetic — which is what makes the
// ring operator-reachable at all.
//
// Prior-art: the graded host/device expert split is llama.cpp's `--n-cpu-moe N` (and the
// `-ot`/tensor-override family it generalizes) — the semantics are BORROWED verbatim, including
// "first N layers", so an operator carrying a working llama.cpp number gets the same placement here.
// The distinct axis is DeepEP / TensorRT-LLM expert-parallel dispatch, which spreads experts ACROSS
// ranks (#971, #3886) rather than choosing a host-vs-device home within one rank; the two compose
// and neither substitutes for the other.
//
// The three placements now compose instead of competing: the first N layers' experts run on the
// HOST kernel (spill), the remaining layers' experts are DEVICE-resident but BOUNDED by the ring
// (only the activated working set need be resident at once), and the dense base is device-resident
// and permanent. Default is unchanged: with ExpertSpillLayers <= 0 the predicate is exactly
// isExpertWeight, and with ExpertRingBytes == 0 there is no ring.

// expertLayerIndex reads the layer ordinal out of a canonical tensor name — "model.layers.<L>."
// (layerPrefix, weights.go) — which is what makes a layer-GRADED placement predicate possible from
// the name alone, exactly as the ungraded one works from the name alone. ok is false for any name
// with no layer segment (embeddings, lm_head, the final norm) or a non-numeric one; a caller must
// decide what an unnumbered weight means rather than being handed a silent 0, because layer 0 is
// the FIRST thing a graded spill moves.
func expertLayerIndex(name string) (int, bool) {
	const seg = "model.layers."
	i := strings.Index(name, seg)
	if i < 0 {
		return 0, false
	}
	rest := name[i+len(seg):]
	j := strings.IndexByte(rest, '.')
	if j <= 0 || j > 9 { // >9 digits is not a layer ordinal; refuse rather than overflow
		return 0, false
	}
	n := 0
	for k := 0; k < j; k++ {
		c := rest[k]
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

// MoEExpertLayers returns the ASCENDING layer ordinals that actually carry routed expert weights in
// this loaded model. It is the spill ORDER: `--n-cpu-moe N` moves the first N of these, so the list
// must be the real MoE layer set and not 0..NumLayers — a hybrid checkpoint's first
// FirstKDenseReplace layers are dense (moe.go), and counting them would silently spill N-k layers
// when the operator asked for N.
//
// It walks the same resident stores MoEResidentWeightBytes tallies, so the layer set and the byte
// accounting are derived from one source of truth: a layer appears here iff its expert bytes are in
// that partition's expert term.
func (m *Model) MoEExpertLayers() []int {
	if m == nil {
		return nil
	}
	seen := map[int]bool{}
	note := func(name string) {
		if !isRoutedExpertTensor(name) {
			return
		}
		if l, ok := expertLayerIndex(name); ok {
			seen[l] = true
		}
	}
	for name := range m.q4kw {
		note(name)
	}
	for name := range m.q8w {
		note(name)
	}
	for name := range m.kqw {
		note(name)
	}
	for name := range m.manifest {
		note(name)
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]int, 0, len(seen))
	for l := range seen {
		out = append(out, l)
	}
	sort.Ints(out)
	return out
}

// ExpertSpillBudgetFor builds the sizing input (ExpertSpillBudget) for THIS loaded model against a
// measured device byte budget — the bridge that #5281's math was missing. The three byte terms come
// from the model's real residency:
//
//	MoELayers           = len(MoEExpertLayers())          — the layers a spill can actually move
//	ExpertBytesPerLayer = routed expert bytes / MoELayers  — rounded UP, see below
//	DeviceBaseBytes     = the replicated remainder         — dense + attention + router + lm_head
//	                                                          + the always-on shared expert
//
// The per-layer cost is rounded UP so an uneven checkpoint (a layer with more experts, or a
// different quant) is never UNDER-counted: over-counting spills one layer too many, which costs
// throughput; under-counting admits a model that then OOMs on the device, which costs the serve.
// Fail-closed in the direction that keeps the serve alive.
//
// ok is false when nothing is resident (an unloaded model) or when the model has no routed-expert
// layers at all (dense, or an MoE whose experts are not in any resident store) — the caller then
// leaves placement exactly as it was rather than sizing against a zero footprint.
func (m *Model) ExpertSpillBudgetFor(deviceBudgetBytes int64) (ExpertSpillBudget, bool) {
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok {
		return ExpertSpillBudget{}, false
	}
	layers := m.MoEExpertLayers()
	if len(layers) == 0 || expert <= 0 {
		return ExpertSpillBudget{}, false
	}
	n := int64(len(layers))
	return ExpertSpillBudget{
		MoELayers:           len(layers),
		ExpertBytesPerLayer: (expert + n - 1) / n,
		DeviceBaseBytes:     replicated,
		BudgetBytes:         deviceBudgetBytes,
	}, true
}

// RingBytesAt reports the byte budget to hand R0's routed-expert ring (Session.ExpertRingBytes) when
// N layers spill to host: the device bytes left over after the always-resident dense base, capped at
// what the non-spilled expert layers would occupy if they were ALL resident.
//
// The cap is the point. Below it the ring is a genuine bound — the non-spilled layers' experts do
// not all fit, so only the activated working set is resident and the coldest pages out. At or above
// it every non-spilled expert fits and the ring never evicts, which is the honest answer for a model
// that fits: a bound that is never reached costs nothing. A budget smaller than the dense base
// yields 0, and 0 disables the ring, so a device that cannot even hold the base falls back to the
// unchanged path instead of thrashing a nonsensically small ring — the refusal belongs to preflight,
// not to the residency policy.
func (b ExpertSpillBudget) RingBytesAt(n int) int64 {
	if n < 0 {
		n = 0
	}
	if n > b.MoELayers {
		n = b.MoELayers
	}
	avail := b.BudgetBytes - b.DeviceBaseBytes
	if avail <= 0 {
		return 0
	}
	want := int64(b.MoELayers-n) * b.ExpertBytesPerLayer
	if want < avail {
		return want
	}
	return avail
}

// ExpertSpillPlacement is one resolved placement decision for a loaded model on a measured device: how
// many MoE layers spill to host, what that leaves device-resident, and how many bytes the routed
// expert ring gets. It is the single object a serve applies (ApplyExpertSpillPlacement) so the three
// coupled knobs — the split predicate's grade, the ring budget, and whether the split runs at all —
// can never drift out of agreement with each other.
type ExpertSpillPlacement struct {
	// Budget is the sizing input this plan was resolved against (measured, per ExpertSpillBudgetFor).
	Budget ExpertSpillBudget
	// Fit is the sizing result: SpillLayers is N, and Fits reports whether the device-resident
	// remainder is within the budget. Fits=false is a plan that spilled everything it could and STILL
	// does not fit the dense base — the caller refuses, it does not silently serve.
	Fit ExpertSpillFit
	// RingBytes is the routed-expert ring budget (Session.ExpertRingBytes) implied by this spill; 0
	// leaves the ring off and the device-side experts on the unbounded permanent path.
	RingBytes int64
	// spilled is Fit.SpillLayers already resolved into the layer-ordinal SET the split predicate
	// tests, computed ONCE here rather than per session. Resolving it walks every resident tensor
	// name (MoEExpertLayers), which on a 753B checkpoint is ~70k names — a per-session cost on a
	// path that builds a session per request. nil means ungraded (see spilledExpertLayers).
	spilled map[int]bool
}

// ResolveExpertSpillPlacement resolves the graded placement for this model against a measured device byte
// budget. userN < 0 selects AUTO (AutoFitExpertSpill: the smallest N that fits, fail-closed to the
// maximal spill when even that cannot); userN >= 0 is an explicit operator N and is REFUSED with
// *ExpertSpillRangeError when out of [0, MoELayers] — a typo must never silently clamp into a
// residency nobody asked for.
//
// ok=false (with a nil error) means this model has no gradeable expert residency — dense, unloaded,
// or no routed experts in any resident store — and the caller must leave placement untouched.
func (m *Model) ResolveExpertSpillPlacement(deviceBudgetBytes int64, userN int) (ExpertSpillPlacement, bool, error) {
	b, ok := m.ExpertSpillBudgetFor(deviceBudgetBytes)
	if !ok {
		return ExpertSpillPlacement{}, false, nil
	}
	var (
		fit ExpertSpillFit
		err error
	)
	if userN < 0 {
		fit, err = AutoFitExpertSpill(b)
	} else {
		fit, err = ResolveExpertSpill(b, userN)
	}
	if err != nil {
		return ExpertSpillPlacement{}, false, err
	}
	return ExpertSpillPlacement{
		Budget:    b,
		Fit:       fit,
		RingBytes: b.RingBytesAt(fit.SpillLayers),
		spilled:   m.spilledExpertLayers(fit.SpillLayers),
	}, true, nil
}

// spilledExpertLayers resolves "spill the first n MoE layers" into the layer-ordinal SET a
// placement predicate can test in O(1). nil is the UNGRADED answer and covers both endpoints —
// n <= 0 (spill nothing through the graded path) and n >= the MoE layer count (spilling every
// layer is exactly the layer-blind isExpertWeight) — so the caller has one nil check rather than
// two range checks, and the default path allocates nothing.
func (m *Model) spilledExpertLayers(n int) map[int]bool {
	if n <= 0 {
		return nil
	}
	layers := m.MoEExpertLayers()
	if n >= len(layers) {
		return nil
	}
	spilled := make(map[int]bool, n)
	for _, l := range layers[:n] {
		spilled[l] = true
	}
	return spilled
}

// gradedExpertSpillPredicate turns a resolved spill set into the predicate splitKernel runs.
//
// An empty set returns isExpertWeight itself — byte-for-byte the ungraded pre-#5612 predicate, not
// a wrapper that reproduces it — so the default path keeps its exact identity and cost. Otherwise
// an expert weight goes to host only when its layer is in the set; a NON-expert weight never does
// (dense, router, attention and lm_head stay on the device exactly as before), and an expert weight
// whose name carries no parseable layer ordinal spills — the memory-safe direction, since the
// device is the scarce side and an unplaceable name must not silently consume VRAM the plan did
// not budget.
func gradedExpertSpillPredicate(spilled map[int]bool) func(string) bool {
	if len(spilled) == 0 {
		return isExpertWeight
	}
	return func(name string) bool {
		if !isExpertWeight(name) {
			return false
		}
		l, ok := expertLayerIndex(name)
		if !ok {
			return true
		}
		return spilled[l]
	}
}

// ApplyExpertSpillPlacement installs a resolved plan on this session: the spill count grades the split
// predicate, and the ring budget bounds what the non-spilled experts may hold on the device. It
// turns the split ON only when the plan actually moves layers — a zero-layer spill leaves the
// all-device placement untouched, so an operator who asked for a ring but no spill gets exactly
// that, and a plan that spills nothing never quietly routes every expert GEMM to the host.
//
// Applying a plan REPLACES any earlier one, so a session re-planned mid-life cannot keep running
// the previous grade. It installs the predicate the plan already resolved rather than clearing it
// for a lazy rebuild: a served session is built per REQUEST, and rebuilding would re-walk every
// resident tensor name each time (spilledExpertLayers) for an answer that cannot change.
func (s *Session) ApplyExpertSpillPlacement(p ExpertSpillPlacement) {
	if s == nil {
		return
	}
	s.ExpertSpillLayers = p.Fit.SpillLayers
	s.ExpertRingBytes = p.RingBytes
	if p.Fit.SpillLayers > 0 {
		s.CPUOffloadExperts = true
	}
	s.spillOnHost = gradedExpertSpillPredicate(p.spilled)
}

// expertSpillOnHost is the placement predicate splitKernel runs — isExpertWeight GRADED by layer.
//
//	ExpertSpillLayers <= 0  -> isExpertWeight, byte-for-byte the ungraded pre-#5612 predicate
//	ExpertSpillLayers >= MoELayers -> also isExpertWeight (spilling every layer IS the ungraded case)
//	0 < N < MoELayers       -> only the FIRST N MoE layers' expert weights go to host
//
// "First N" is over the model's real MoE layer ordinals (MoEExpertLayers), matching llama.cpp's
// `--n-cpu-moe N` and the semantics expert_spill_fit.go sizes against. An expert weight whose name
// carries no parseable layer ordinal spills — the memory-safe direction, since the device is the
// scarce side and an unplaceable name must not silently consume VRAM the plan did not budget.
//
// The predicate is built once per session and memoized: it is consulted per GEMM, and rebuilding it
// would re-walk every resident tensor name on a 753B checkpoint each time. A session configured by
// ApplyExpertSpillPlacement already carries the resolved predicate and never reaches the walk here;
// this lazy arm serves a session whose grade was assigned to the field directly.
func (s *Session) expertSpillOnHost() func(string) bool {
	if s.spillOnHost == nil {
		s.spillOnHost = gradedExpertSpillPredicate(s.M.spilledExpertLayers(s.ExpertSpillLayers))
	}
	return s.spillOnHost
}
