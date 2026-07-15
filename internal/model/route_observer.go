package model

// route_observer.go — issue #2623, the MoE-routing analogue of AttnObserver (#852).
//
// route() in moe.go already computes, per token per layer, the top-k (expert, gate_weight)
// picks — then throws them away immediately after the weighted sum. This file emits them —
// and ONLY emits them — behind a default-off observer so the witnessed routing becomes
// available to the view layer (the per-span expert_hist descriptor, kvmmu) without
// perturbing the forward pass. It mirrors attn_observer.go exactly.
//
// INVARIANTS (the same byte-identical foundation AttnObserver depends on):
//   - nil observer == today's behavior: zero extra allocation, zero extra work in the hot
//     loop (every emission site is guarded behind `if m.routeObs != nil`, and mlpSeq only
//     stamps routePos when the observer is set).
//   - emission only: the router logits, softmax, top-k select, and the gate-weighted
//     accumulation are never touched. The observer receives a COPY of the picks, so even a
//     misbehaving observer cannot mutate the routePick slice the weighted sum reads.
//
// HONEST SCOPE (#2623, docs/notes/CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md §7). The
// routing signal is FREE but a WEAK descriptor: routing is a linear projection of the
// hidden state — a coarse geometric hash, not a domain label ("The Myth of Expert
// Specialization in MoEs": different models overlap only ~60% in experts on the same
// problem; prompt-phase routing does not predict generation-phase routing). So an expert id
// is admissible as a WITHIN-MODEL, WITHIN-RUN clustering/dedup key, NOT a portable semantic
// tag. A consumer must not read expert == topic; when query-relevance is wanted, attention
// (#2617) is the right signal, not router logits.

// RouteObserver receives one token's top-k expert routing for one (layer, tokenPos).
// experts[i] is a selected expert id and gateWeights[i] the gate weight it carried, in the
// model's selection order (the same order the weighted sum accumulates). tokenPos is the
// token's index within the sequence chunk the current forward pass processes — for a
// full-prefill pass (base 0) that is the absolute token position; on an incremental decode
// step it is the in-chunk index. The two slices are freshly allocated per call and owned by
// the observer — the caller never reads them again, so the observer may retain them. Both
// slices have len == the routed top-k for this token (NumExpertsPerTok, clamped to the
// candidate count).
type RouteObserver func(layer, tokenPos int, experts []int, gateWeights []float32)

// SetRouteObserver installs (or clears, with nil) the routing witness on this model.
// Default is nil — the unobserved forward pass is byte-identical to a model that never had
// this method called. Not safe to change concurrently with a forward pass; set it before
// the pass and clear it after (the same discipline SetAttnObserver requires).
func (m *Model) SetRouteObserver(obs RouteObserver) { m.routeObs = obs }

// RouteObserverSet reports whether a routing observer is currently installed. The hot path
// guards on m.routeObs != nil directly; this is for callers/tests.
func (m *Model) RouteObserverSet() bool { return m != nil && m.routeObs != nil }

// emitRouteObserved copies this token's (expert, gate-weight) picks and hands them to the
// installed observer. It is a no-op when no observer is set, so route()/glmRoute() may call
// it unconditionally: the per-call allocation happens ONLY behind the nil guard, preserving
// the zero-alloc-when-off invariant. tokenPos is read from routePos, which mlpSeq stamps.
func (m *Model) emitRouteObserved(layer int, picks []routePick) {
	if m == nil || m.routeObs == nil {
		return
	}
	experts := make([]int, len(picks))
	weights := make([]float32, len(picks))
	for i, pk := range picks {
		experts[i] = pk.expert
		weights[i] = pk.weight
	}
	m.routeObs(layer, m.routePos, experts, weights)
}
