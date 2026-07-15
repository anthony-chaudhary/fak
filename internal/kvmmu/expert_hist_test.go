package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// expert_hist_test.go — acceptance for issue #2623 (per-span MoE expert_hist attribution).
//
// Like attention_test.go these exercise the From/Len attribution partition directly through
// the public Context API (Append lays out spans; AttributeRoute / RouteObserver fold a
// token's picks; Quarantine evicts). They feed a KNOWN pick stream rather than running a
// forward pass — the attribution + bounding is what is under test, not the router numerics
// (those are the internal/model RouteObserver tests, #2623 rung 1).

func newRouteCtx(t *testing.T) *kvmmu.Context {
	t.Helper()
	m := model.NewSynthetic(synthCfg())
	return kvmmu.New(m.NewSession())
}

// TestAttributeRouteCorrectSpan: a token's picks land on the span whose [From, From+Len)
// range owns the token's position — never the neighbouring span.
func TestAttributeRouteCorrectSpan(t *testing.T) {
	c := newRouteCtx(t)
	c.Append("A", "toolA", []int{1, 2, 3}) // positions 0,1,2
	c.Append("B", "toolB", []int{4, 5, 6}) // positions 3,4,5

	// token at position 1 (owned by A) routed to experts {2,5}; token at position 4 (B) to {7}.
	if !c.AttributeRoute(1, []int{2, 5}, []float32{0.6, 0.4}) {
		t.Fatalf("position 1 attributed to no span; want span A")
	}
	if !c.AttributeRoute(4, []int{7}, []float32{0.9}) {
		t.Fatalf("position 4 attributed to no span; want span B")
	}

	aCounts, aMean, aOver, aOK := c.ExpertHistogram("A")
	if !aOK {
		t.Fatalf("span A has no expert_hist; want one")
	}
	if aCounts[2] != 1 || aCounts[5] != 1 || aOver != 0 {
		t.Errorf("A hist = %v (overflow %d); want experts 2 and 5 once each", aCounts, aOver)
	}
	// mean gate over A's two picks: (0.6+0.4)/2 = 0.5.
	if math.Abs(aMean-0.5) > 1e-6 {
		t.Errorf("A mean gate = %v, want 0.5", aMean)
	}
	if _, ok := aCounts[7]; ok {
		t.Errorf("A hist contains expert 7, which routed on span B — attribution crossed the boundary")
	}

	bCounts, bMean, _, bOK := c.ExpertHistogram("B")
	if !bOK || bCounts[7] != 1 || math.Abs(bMean-0.9) > 1e-6 {
		t.Errorf("B hist = %v mean %v; want expert 7 once, mean 0.9", bCounts, bMean)
	}
}

// TestExpertHistMultisetAndMean: repeated picks on the same expert accumulate as a multiset
// (count grows) and the mean gate is the pick-weighted average across all picks in the span.
func TestExpertHistMultisetAndMean(t *testing.T) {
	c := newRouteCtx(t)
	c.Append("A", "toolA", []int{1, 2, 3, 4}) // positions 0..3

	// four tokens, all in A: expert 2 routed on three of them, expert 9 on one.
	c.AttributeRoute(0, []int{2}, []float32{0.2})
	c.AttributeRoute(1, []int{2}, []float32{0.4})
	c.AttributeRoute(2, []int{2, 9}, []float32{0.5, 0.5})
	c.AttributeRoute(3, []int{9}, []float32{0.9})

	counts, mean, over, ok := c.ExpertHistogram("A")
	if !ok || over != 0 {
		t.Fatalf("A hist ok=%v overflow=%d; want ok, no overflow", ok, over)
	}
	if counts[2] != 3 || counts[9] != 2 {
		t.Errorf("multiset = %v; want expert 2 ×3, expert 9 ×2", counts)
	}
	// mean gate over all 5 picks: (0.2+0.4+0.5+0.5+0.9)/5 = 0.5.
	if math.Abs(mean-0.5) > 1e-6 {
		t.Errorf("mean gate = %v, want 0.5", mean)
	}
}

// TestExpertHistBounded is the #2623 named witness: a span that routes to more than
// expertHistCap distinct experts tracks at most cap of them and counts the rest in Overflow,
// so one span's histogram stays O(cap) regardless of how many experts its tokens touch.
func TestExpertHistBounded(t *testing.T) {
	c := newRouteCtx(t)
	histCap := kvmmu.ExpertHistForTest()
	nExperts := histCap + 40 // more distinct experts than the span may track

	c.Append("A", "toolA", []int{1, 2, 3}) // positions 0,1,2

	// spread the distinct experts across the span's three tokens (a real top-k is small, but
	// a span's tokens accumulate across the turn) — the span sees nExperts distinct experts.
	next := 0
	for pos := 0; pos < 3 && next < nExperts; pos++ {
		var experts []int
		var weights []float32
		for j := 0; j < nExperts/3+1 && next < nExperts; j++ {
			experts = append(experts, next)
			weights = append(weights, 0.5)
			next++
		}
		if !c.AttributeRoute(pos, experts, weights) {
			t.Fatalf("position %d attributed to no span", pos)
		}
	}

	counts, _, overflow, ok := c.ExpertHistogram("A")
	if !ok {
		t.Fatalf("span A has no expert_hist after routing")
	}
	if len(counts) != histCap {
		t.Errorf("tracked %d distinct experts; want the cap %d", len(counts), histCap)
	}
	if overflow != nExperts-histCap {
		t.Errorf("overflow = %d; want %d (%d distinct experts past the cap)", overflow, nExperts-histCap, nExperts-histCap)
	}
}

// TestRouteObserverFoldsOntoSpans wires the returned model.RouteObserver end to end: the
// closure folds each emitted token's picks onto the owning span, exactly as installing it on
// a model for a forward pass would (the emission itself is covered by the model-side tests).
func TestRouteObserverFoldsOntoSpans(t *testing.T) {
	c := newRouteCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1
	c.Append("B", "toolB", []int{3, 4}) // 2,3

	var obs model.RouteObserver = c.RouteObserver()
	obs(0, 0, []int{1, 4}, []float32{0.5, 0.5}) // layer 0, token 0 -> A
	obs(1, 0, []int{4}, []float32{1.0})         // layer 1, same token 0 -> A (layers sum)
	obs(0, 3, []int{2}, []float32{0.8})         // token 3 -> B

	aCounts, _, _, aOK := c.ExpertHistogram("A")
	if !aOK || aCounts[4] != 2 || aCounts[1] != 1 {
		t.Errorf("A hist = %v; want expert 4 ×2 (both layers) and expert 1 ×1", aCounts)
	}
	bCounts, _, _, bOK := c.ExpertHistogram("B")
	if !bOK || bCounts[2] != 1 {
		t.Errorf("B hist = %v; want expert 2 ×1", bCounts)
	}
}

// TestExpertHistClearsOnEvict mirrors the Attended eviction coherence: a quarantined span's
// routing histogram clears (its picks are moot once it leaves the cache).
func TestExpertHistClearsOnEvict(t *testing.T) {
	c := newRouteCtx(t)
	c.Append("A", "toolA", []int{1, 2, 3})
	c.AttributeRoute(0, []int{2, 5}, []float32{0.6, 0.4})
	if _, _, _, ok := c.ExpertHistogram("A"); !ok {
		t.Fatalf("span A has no expert_hist before evict; want one")
	}
	c.Quarantine("A")
	if _, _, _, ok := c.ExpertHistogram("A"); ok {
		t.Errorf("span A still has an expert_hist after eviction; want it cleared")
	}
}
