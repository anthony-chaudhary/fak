package model

import "sort"

// v41_expert_policy.go — the V4.1 MoE expert-residency policy adapter for the unified-memory
// serving tier (anthony-chaudhary/fak#12952, parent #12640).
//
// The residency mechanism itself (resident hot set, separate prefill transient ring, descriptor
// swap, counters) lives in internal/ctxmmu/expert_residency.go. This file is the MODEL-SIDE half
// the issue's third acceptance criterion names: a RESUMABLE, per-layer trace generator that turns
// an observed V4.1 routing trace into per-layer histograms, and a deterministic warm-start PLAN
// (the ranked hot-set per layer) those histograms feed.
//
// Why a separate, resumable generator rather than one pass over the whole trace: a real appliance
// routing trace is large and is produced/consumed incrementally (a layer's routing is known as
// the forward reaches it). Streaming one layer at a time and keeping a resumable cursor lets a
// long capture be checkpointed and folded without holding the whole corpus, and lets the warm
// start consume exactly the layers a shard-decomposed serve is responsible for.
//
// Scope (honest): a standalone, host-free planner, OFF the live serve path. It copies no
// upstream bytes; it is a clean-room reimplementation of the studied trace-ranked warm-start
// lever. The real-workload warm-start hit-rate benefit is [HW-WITNESSED] and stays unchecked.

// V41LayerExpertHistogram is the per-layer expert-selection histogram for one V4.1 MoE layer:
// Counts[e] is how many times expert e was routed in this layer across the folded window.
// Unsized counts touches whose resident weight bytes could not be sized, so a silently-missing
// expert is visible rather than folded in as a zero.
type V41LayerExpertHistogram struct {
	Layer      int     `json:"layer"`
	NumExperts int     `json:"num_experts"`
	Counts     []int64 `json:"counts"`
	Touches    int64   `json:"touches"`
	Unsized    int64   `json:"unsized"`
}

// TopExperts returns the budget most-routed experts in deterministic (count desc, expert id
// asc) order. It is the per-layer warm-start selection: the experts a resident hot set should
// seed for this layer. A budget <= 0 yields nil.
func (h V41LayerExpertHistogram) TopExperts(budget int) []int {
	if budget <= 0 || len(h.Counts) == 0 {
		return nil
	}
	idx := make([]int, 0, len(h.Counts))
	for e, c := range h.Counts {
		if c > 0 {
			idx = append(idx, e)
		}
	}
	sort.Slice(idx, func(a, b int) bool {
		if h.Counts[idx[a]] != h.Counts[idx[b]] {
			return h.Counts[idx[a]] > h.Counts[idx[b]]
		}
		return idx[a] < idx[b]
	})
	if budget > len(idx) {
		budget = len(idx)
	}
	return idx[:budget]
}

// V41ExpertTraceGenerator folds an observed routing trace into per-layer histograms one layer
// at a time. It is RESUMABLE: Next returns the next layer (lowest not-yet-emitted layer id) and
// a false ok when the stream is exhausted; Reset rewinds to the first layer. The generator never
// mutates the trace it was built from.
type V41ExpertTraceGenerator struct {
	trace      ExpertAccessTrace
	numExperts int
	emitted    map[int]bool
	order      []int
	cursor     int
}

// NewV41ExpertTraceGenerator builds a generator over trace, sizing histograms to numExperts.
// Layers are emitted in ascending layer-id order. A non-positive numExperts derives the width
// from the largest expert id observed (so a caller need not know the model's expert count).
func NewV41ExpertTraceGenerator(trace ExpertAccessTrace, numExperts int) *V41ExpertTraceGenerator {
	layers := map[int]bool{}
	maxExpert := -1
	for _, ev := range trace.Events {
		if ev.Layer < 0 || ev.Expert < 0 {
			continue
		}
		layers[ev.Layer] = true
		if ev.Expert > maxExpert {
			maxExpert = ev.Expert
		}
	}
	if numExperts <= 0 {
		numExperts = maxExpert + 1
	}
	order := make([]int, 0, len(layers))
	for l := range layers {
		order = append(order, l)
	}
	sort.Ints(order)
	return &V41ExpertTraceGenerator{
		trace: trace, numExperts: numExperts, emitted: map[int]bool{}, order: order,
	}
}

// Progress reports how many layers have been emitted and how many remain — the resumable
// checkpoint a long capture records between calls.
func (g *V41ExpertTraceGenerator) Progress() (emitted, remaining int) {
	if g == nil {
		return 0, 0
	}
	return g.cursor, len(g.order) - g.cursor
}

// Next emits the next layer's histogram. ok is false once every layer has been emitted; call
// Reset to rewind. A touched expert id outside [0,numExperts) is skipped.
func (g *V41ExpertTraceGenerator) Next() (V41LayerExpertHistogram, bool) {
	if g == nil || g.cursor >= len(g.order) {
		return V41LayerExpertHistogram{}, false
	}
	layer := g.order[g.cursor]
	g.cursor++
	h := V41LayerExpertHistogram{Layer: layer, NumExperts: g.numExperts, Counts: make([]int64, g.numExperts)}
	for _, ev := range g.trace.Events {
		if ev.Layer != layer {
			continue
		}
		if ev.Expert < 0 || ev.Expert >= g.numExperts {
			continue
		}
		h.Counts[ev.Expert]++
		h.Touches++
	}
	h.Unsized = int64(g.trace.UnsizedTouches)
	g.emitted[layer] = true
	return h, true
}

// Reset rewinds the generator to the first layer, so a resumed run re-emits from the start.
func (g *V41ExpertTraceGenerator) Reset() {
	if g == nil {
		return
	}
	g.cursor = 0
	g.emitted = map[int]bool{}
}

// V41ExpertWarmStartPlan folds a whole trace into a deterministic per-layer warm-start plan:
// for every layer present in the trace, the budget hottest experts. It is the warm-start input
// the residency policy consumes (the ctxmmu policy's WarmStart takes these per-layer hot
// identities as its access seed). A budget <= 0 yields an empty plan.
func V41ExpertWarmStartPlan(trace ExpertAccessTrace, budget int) []V41LayerExpertHistogram {
	if budget <= 0 {
		return nil
	}
	g := NewV41ExpertTraceGenerator(trace, 0)
	plan := make([]V41LayerExpertHistogram, 0, len(g.order))
	for {
		h, ok := g.Next()
		if !ok {
			break
		}
		if len(h.TopExperts(budget)) == 0 {
			continue
		}
		plan = append(plan, h)
	}
	return plan
}
