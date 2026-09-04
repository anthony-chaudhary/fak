package kvmmu

import "github.com/anthony-chaudhary/fak/internal/model"

// expert_hist.go — within-model MoE expert routing attribution to spans.

// expertHistCap bounds the distinct experts tracked per span.
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

// RouteObserver returns a model.RouteObserver folding emitted token routing picks onto Context segments.
func (c *Context) RouteObserver() model.RouteObserver {
	return func(_, tokenPos int, experts []int, gateWeights []float32) {
		c.AttributeRoute(tokenPos, experts, gateWeights)
	}
}

// ExpertHistogram returns a copy of span's routing histogram, mean gate weight, and overflow count.
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
