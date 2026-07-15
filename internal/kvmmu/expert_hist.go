package kvmmu

import "github.com/anthony-chaudhary/fak/internal/model"

// expert_hist.go — issue #2623, the MoE-routing analogue of attention.go (#853).
//
// Rung 1 for routing (internal/model/route_observer.go, #2623) emits, per (layer, token),
// a COPY of that token's top-k (expert, gate_weight) picks — the free byproduct of every
// MoE FFN that route() otherwise discards. This file attributes that stream to SPANS: a
// token's picks are folded onto the Segment whose recorded [From, From+Len) range owns the
// token's position, building a per-span expert_hist descriptor (the multiset of experts the
// span's tokens routed to + the mean gate weight). It mirrors AttributeRow/AttentionObserver
// exactly, one accumulator per span, bounded per span like the attention accumulator.
//
// HONEST PROVENANCE (#2623, the load-bearing caution — do NOT strip it). expert_hist is a
// WITHIN-MODEL, WITHIN-RUN descriptor: routing is a linear projection of the hidden state, a
// coarse geometric hash, NOT a domain/topic label ("The Myth of Expert Specialization in
// MoEs": different models overlap only ~60% in experts on the same problem; prompt-phase
// routing does not predict generation-phase routing). It is admissible as a clustering/dedup
// key for spans of the SAME model in the SAME run — never as a portable semantic tag, and
// never as query-relevance (that is attention's job, #2617, not router logits). A consumer
// that reads an expert id as a topic contradicts this issue's explicit scope.

// expertHistCap bounds the distinct experts tracked per span, so one span's histogram is
// O(cap) regardless of how many tokens it owns or how many experts the model has. A pick on
// a (cap+1)-th distinct expert is counted in Overflow instead of growing the map — the
// bounded-memory guarantee the per-span accumulator shares with the attention trajectory
// ring (trajCap). 64 distinct experts is a generous window: a span's tokens concentrate on a
// small residency in practice, and the descriptor is a clustering key, not a full census.
const expertHistCap = 64

// ExpertHistForTest exposes the per-span cap to the external _test package so the
// bounded-size acceptance test can assert the bound without hardcoding the constant.
func ExpertHistForTest() int { return expertHistCap }

// ExpertHist is one span's witnessed routing histogram: counts[e] picks landed on expert e
// carrying gateSum[e] total gate weight (mean gate for e = gateSum[e]/counts[e]). Overflow
// counts picks dropped because the span already tracks expertHistCap distinct experts.
type ExpertHist struct {
	counts   map[int]int
	gateSum  map[int]float64
	Overflow int
}

// add folds one (expert, gate) pick into the histogram, respecting the per-span cap: a pick
// on an already-tracked expert always counts; a pick that would introduce a (cap+1)-th
// distinct expert bumps Overflow and is dropped rather than growing the map without bound.
func (h *ExpertHist) add(expert int, gate float64) {
	if h.counts == nil {
		h.counts = make(map[int]int)
		h.gateSum = make(map[int]float64)
	}
	if _, seen := h.counts[expert]; !seen && len(h.counts) >= expertHistCap {
		h.Overflow++
		return
	}
	h.counts[expert]++
	h.gateSum[expert] += gate
}

// snapshot returns a fresh copy of the histogram (the caller may retain it), the mean gate
// weight across every pick (Σ gateSum / Σ counts), and the overflow count.
func (h *ExpertHist) snapshot() (counts map[int]int, meanGate float64, overflow int) {
	counts = make(map[int]int, len(h.counts))
	var totalGate float64
	var totalCount int
	for e, c := range h.counts {
		counts[e] = c
		totalCount += c
		totalGate += h.gateSum[e]
	}
	if totalCount > 0 {
		meanGate = totalGate / float64(totalCount)
	}
	return counts, meanGate, h.Overflow
}

// AttributeRoute folds one token's routing picks (experts[i] with gate weight gateWeights[i])
// onto the live segment whose [From, From+Len) range owns tokenPos. It returns false if no
// live segment owns the position (a stale/out-of-range emission — the pick is dropped, never
// mis-attributed), true once the picks land on a span. Mirrors AttributeRow's single-cursor
// discipline: one lookup over the small live-segment ledger, then a fold of the whole row.
func (c *Context) AttributeRoute(tokenPos int, experts []int, gateWeights []float32) (attributed bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := len(experts)
	if len(gateWeights) < n {
		n = len(gateWeights)
	}
	if n == 0 {
		return false
	}
	var owner *Segment
	for _, s := range c.segs {
		if !s.Held && s.Len > 0 && tokenPos >= s.From && tokenPos < s.From+s.Len {
			owner = s
			break
		}
	}
	if owner == nil {
		return false
	}
	if owner.expertHist == nil {
		owner.expertHist = &ExpertHist{}
	}
	for i := 0; i < n; i++ {
		owner.expertHist.add(experts[i], float64(gateWeights[i]))
	}
	return true
}

// RouteObserver returns a model.RouteObserver that folds every emitted token's picks onto
// this Context's span ledger via AttributeRoute. Install it on the model for a forward pass
// (model.SetRouteObserver) to accumulate the per-span expert_hist across the turn; the layer
// dimension is summed into the span's histogram (a span's descriptor is over all layers, as
// the routing signal is a per-token property, not per-layer). Same single-flight discipline
// AttentionObserver assumes — install for the pass, do not race the eviction path.
func (c *Context) RouteObserver() model.RouteObserver {
	return func(_, tokenPos int, experts []int, gateWeights []float32) {
		c.AttributeRoute(tokenPos, experts, gateWeights)
	}
}

// ExpertHistogram returns span id's witnessed expert_hist as a fresh copy: expert id -> pick
// count, the mean gate weight across all picks, and the number of picks dropped past the
// per-span cap. ok is false for an unknown span or one that has routed nothing. The returned
// map is the caller's to keep — it aliases no ledger state. HONEST SCOPE: this is a
// within-model, within-run clustering/dedup key, NOT a semantic/topic label (see the file
// header and #2623).
func (c *Context) ExpertHistogram(id string) (counts map[int]int, meanGate float64, overflow int, ok bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.segs {
		if s.ID == id {
			if s.expertHist == nil {
				return nil, 0, 0, false
			}
			cnt, mean, of := s.expertHist.snapshot()
			return cnt, mean, of, true
		}
	}
	return nil, 0, 0, false
}
