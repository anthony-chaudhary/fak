package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// expert_ring_prefetch.go — R3 of the activated-expert offload ladder (#5614, epic #5606):
// tell the ring a layer's WHOLE activated set the moment the router names it, instead of letting
// the ring discover each expert one GEMM too late.
//
// The router produces all k picks for a layer BEFORE any of their GEMMs run; the ring never got to
// use that, because weightHALStagedBounded learns an expert exists only when its GEMM reaches for
// the gate projection. prefetchActivatedExperts stages the layer's activated set into the ring in
// route order at layer entry, so by the time the expert loop starts its weights are resident and
// every demand staging is a hit. The uploads are issued back to back with no GEMM between them —
// the seam an async H2D path needs.
//
// Two rules make it a prefetch rather than a thrash:
//   - DO NOT PREFETCH WHAT CANNOT STAY: only the longest PREFIX of the activated set that fits the
//     budget is staged. Running past the budget evicts what was just fetched, converting a late
//     miss into an early miss plus an eviction. Stopping at the prefix keeps the set ordered by the
//     router's own confidence.
//   - A PREFETCH IS A HINT, NOT A DEMAND: staging updates recency but earns NO heat under the R4
//     value-aware policy and is invisible to R2's usage histogram and R4's replay trace — those are
//     fed by the demand path only. A policy ranking on its own prefetcher's guesses would be
//     self-confirming: it would protect exactly what was speculated, whether or not anything used it.
//
// It also never falls back: weightHALStagedBounded answers a ring refusal by promoting the weight
// to PERMANENT halW residency, which is right for a demand but would let a misconfigured budget
// convert speculation into permanent residency and break R0's "no routed expert reaches halW"
// bound. On any refusal the prefetch simply stops.
//
// Honest limit: this issues the page-ins earlier, it does not yet OVERLAP them with compute —
// compute.Backend exposes no stream/event handle, so the plan's "fraction of page-in latency
// overlapped with compute" witness needs an async staging primitive, which is not this rung. What
// IS measured: the set is issued before the first GEMM, it costs zero extra page-ins, it stays
// inside the budget, and it leaves the R2/R4 evidence streams byte-for-byte unperturbed.
//
// Relation to #4300: that issue owns the PREDICTIVE layer (cross-token speculation, layer L+1
// before layer L retires); R3 is the deterministic floor beneath it and the seam it attaches to —
// prefetchActivatedExperts takes a pick list, and a predictor is just a different producer of one.

// ExpertPrefetchMode selects whether the ring is told a layer's activated set up front or discovers
// it one expert at a time.
//
// Unlike the other rungs' knobs, the ZERO value here turns the lookahead ON: the whole ring is
// opt-in behind ExpertRingBytes, so a session that declared nothing still gets nothing, and WITHIN
// a declared ring the prefetch is measured free (same page-ins, same bytes, same arithmetic —
// TestExpertRingPrefetchCostsNoExtraPageIns). ExpertPrefetchOnDemand exists because the plan's
// witness is a comparison at a fixed ring budget, which needs both arms reachable.
type ExpertPrefetchMode int

const (
	// ExpertPrefetchActivatedSet stages the prefix of each layer's routed top-k that the budget can
	// hold, at layer entry, before any expert GEMM runs.
	ExpertPrefetchActivatedSet ExpertPrefetchMode = iota
	// ExpertPrefetchOnDemand keeps the pre-R3 behaviour: each expert is staged when its own GEMM
	// reaches for it, serialized behind the expert before it.
	ExpertPrefetchOnDemand
	// ExpertPrefetchCrossLayer is ExpertPrefetchActivatedSet plus, on the V4.1 batched routed
	// layer, read-ahead hints for the predicted next-layer set (v41_cross_layer_prefetch.go).
	ExpertPrefetchCrossLayer
)

// String names the mode for a report or a log line.
func (m ExpertPrefetchMode) String() string {
	switch m {
	case ExpertPrefetchOnDemand:
		return "on-demand"
	case ExpertPrefetchCrossLayer:
		return "cross-layer"
	default:
		return "activated-set"
	}
}

// ringStaging is how this projection would be staged into the routed-expert ring: the dtype-prefixed
// key, the host-source builder, the dtype and the resident byte cost. It MIRRORS weightHALQ4K /
// weightHALKQuant (hal.go) exactly — same key, same dtype, same size — because a prefetched resident
// must be the very same ring entry the demand path then finds; TestExpertRingPrefetchCostsNoExtraPageIns
// guards that agreement (if the two drift, the demand staging misses and page-ins double).
//
// ok=false for a representation the ring has no staging for. Unlike the demand path this DECLINES
// rather than panics: a hint must never kill a forward the demand path would have served.
//
// A checkpoint-served projection (R5/#5616) reports the descriptor its own tier derived, built to
// this same rule: staging one issues the expert's stride read, so an activated set over a streamed
// checkpoint has its reads issued together at layer entry, and an expert the ring already holds is
// skipped by the isResident check below.
func (w expertWeight) ringStaging() (key string, mk func() compute.Tensor, dt compute.Dtype, bytes int64, ok bool) {
	switch {
	case w.ck != nil:
		return w.ck.key, w.ck.mk, w.ck.dt, w.ck.bytes, true
	case w.q4 != nil:
		qt := w.q4
		// Hints wrap resident raw bytes; lazy descriptors stay on the demand path,
		// which materializes under its own staging contract.
		if qt.lazy != nil {
			return "", nil, 0, 0, false
		}
		return w.halKey(), func() compute.Tensor {
			return compute.NewQ4K(compute.Default(), []int{qt.out, qt.in}, qt.raw)
		}, compute.Q4_K, q4kResidentBytes(qt), true
	case w.kq != nil:
		qt := w.kq
		// Hints wrap resident raw bytes; lazy descriptors stay on the demand path,
		// which materializes under its own staging contract.
		if qt.lazy != nil {
			return "", nil, 0, 0, false
		}
		if desc, ok := LookupQuantDescriptor(qt.kind); ok && desc.SupportsHAL() {
			return w.halKey(), func() compute.Tensor {
				return desc.NewHostTensor(qt.out, qt.in, qt.raw)
			}, desc.Dtype(), kQuantResidentBytes(qt), true
		}
	}
	return "", nil, 0, 0, false
}

// activatedExpertWeights resolves one routed expert's three projections through the SAME resolver
// the demand path uses (Session.resolveExpertWeight, hal.go) — resident stores first, then the
// R5/#5616 checkpoint tier — so the prefetch reaches for exactly the weights the GEMM will.
// ok=false if any projection is unreachable, in which case this expert is not a ring candidate.
//
// Resolving here is also what makes a checkpoint fault OVERLAPPABLE: the activated set is known at
// layer entry, so its reads are issued together instead of one GEMM apart.
func (s *Session) activatedExpertWeights(layer, expert int) ([3]expertWeight, bool) {
	var ws [3]expertWeight
	if s == nil || s.M == nil {
		return ws, false
	}
	for i, suffix := range [3]string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		w, ok := s.resolveExpertWeight(expertName(layer, expert, suffix))
		if !ok {
			return ws, false
		}
		ws[i] = w
	}
	return ws, true
}

// activatedExpertPlan is one resolved pick: its three staging descriptors and their total resident
// cost, computed before ANY of them is staged so the fit decision is made on the whole expert — an
// expert is admitted as a unit or not at all, since two of three projections is a fragment no GEMM
// can use.
type activatedExpertPlan struct {
	expert int
	bytes  int64
	stage  [3]struct {
		key   string
		mk    func() compute.Tensor
		dt    compute.Dtype
		bytes int64
	}
}

// prefetchActivatedExperts stages the prefix of this layer's activated set that the ring budget can
// hold, in route order. It is a no-op — allocating nothing and touching no counter — for a session
// with no ring, which is the default.
func (s *Session) prefetchActivatedExperts(layer int, picks []routePick) {
	if s == nil || len(picks) == 0 || s.ExpertRingBytes <= 0 || s.ExpertPrefetch == ExpertPrefetchOnDemand {
		return
	}
	// Probe through the same accessor the demand path uses, naming a weight that really exists, so
	// the ring is built (or declined) by exactly one rule rather than two.
	r := s.routedExpertRing(expertName(layer, picks[0].expert, "gate_proj.weight"))
	if r == nil {
		return
	}

	// Phase 1: resolve every pick before staging any of it, so the coverage meter counts the whole
	// activated set — including the tail the budget will refuse. A meter that only counted what fit
	// could never report a budget that fits nothing.
	plans := make([]activatedExpertPlan, 0, len(picks))
	for _, pk := range picks {
		ws, ok := s.activatedExpertWeights(layer, pk.expert)
		if !ok {
			continue // not a ring candidate (missing or unstageable projection)
		}
		p := activatedExpertPlan{expert: pk.expert}
		for i, w := range ws {
			key, mk, dt, b, ok := w.ringStaging()
			if !ok || b <= 0 {
				p.bytes = -1
				break
			}
			p.stage[i].key, p.stage[i].mk, p.stage[i].dt, p.stage[i].bytes = key, mk, dt, b
			p.bytes += b
		}
		if p.bytes <= 0 {
			continue
		}
		plans = append(plans, p)
	}
	if len(plans) == 0 {
		return
	}
	// Everything from here down mutates ring state, so under a shared ring it runs inside one span —
	// including the ring-wide `prefetching` flag, which would otherwise let a peer agent's DEMAND be
	// booked as this agent's hint. A no-op under the per-session default.
	release := s.ringEnter(r)
	defer release()
	r.activatedExperts += len(plans)

	// Phase 2: stage the longest prefix that fits. `prefetching` marks these stagings as hints, so
	// they take recency but earn no heat (see the file header).
	r.prefetching = true
	defer func() { r.prefetching = false }()

	// The batch-aware policy projects a larger EFFECTIVE hot-set ceiling when the admitted batch
	// justifies one (#1295); disabled or batch<=1 returns r.budget() byte-for-byte. The larger
	// ceiling admits a longer prefetched prefix — the policy's aggressiveness lever.
	effective := BatchAwareHotSetBytes(r.budget(), s.ExpertAdmittedBatch, s.ExpertRingBatchAware)
	var reserved int64
	for _, p := range plans {
		if reserved+p.bytes > effective {
			break // do not prefetch what cannot stay; the rest of the set is lower-confidence anyway
		}
		pinned := r.isExpertPinned(layer, p.expert)
		staged := 0
		for _, d := range p.stage {
			if r.isResident(d.key) {
				staged++ // already resident: it costs nothing and is already covered
				continue
			}
			if _, ok := r.stage(d.key, d.mk, d.dt, d.bytes, pinned); !ok {
				break // the ring refused (a durable pin left no room) — take the hint no further
			}
			r.prefetched++
			staged++
		}
		if staged < len(p.stage) {
			break // a partial expert is useless; stop rather than start another on top of it
		}
		reserved += p.bytes
		r.activatedCovered++
	}
}

// prefetchActivatedExperts is the layer-level entry the MoE FFN calls the instant its router returns.
// It runs only when the expert loop will actually take the device-HAL route that reaches the ring
// (routedExpertKQuantActive) — the host and Metal batched paths read their weights straight from the
// model, so prefetching for them would upload bytes nothing ever reads. GPT-OSS experts use the
// expertGPTOSS form rather than three SwiGLU projections and resolve to no ring candidate, so they
// fall out here too.
func prefetchActivatedExperts(m *Model, layer int, picks []routePick, mat matKernel) {
	if m == nil || m.Cfg.isGPTOSS() || !routedExpertKQuantActive(mat) {
		return
	}
	var sess *Session
	switch mk := mat.(type) {
	case sessionQ4KKernel:
		sess = mk.s
	case backendKernel:
		sess = mk.s
	}
	sess.prefetchActivatedExperts(layer, picks)
}
