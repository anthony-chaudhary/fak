package ggufload

import (
	"math"
	"testing"
)

// specWidthLine is a verify cost of base+slope*width seconds.
type specWidthLine struct{ base, slope float64 }

func (l specWidthLine) at(width int) float64 { return l.base + l.slope*float64(width) }

// timedSpecWidth returns a chooser that has one verify timing for every width.
func timedSpecWidth(maxWidth int, cost specWidthLine) inkling2SpecWidth {
	c := newInkling2SpecWidth(maxWidth)
	for w := 1; w <= maxWidth; w++ {
		c.ObserveCost(w, cost.at(w))
	}
	return c
}

// nextAdaptive skips the periodic full-width probe so a test reads the policy's own choice.
func nextAdaptive(c *inkling2SpecWidth) int {
	w := c.Next()
	if w == c.max {
		w = c.Next()
	}
	return w
}

func TestInkling2SpecWidthSteadyChoice(t *testing.T) {
	rising := specWidthLine{base: 0.010, slope: 0.001}
	steep := specWidthLine{base: 0.002, slope: 0.004}
	flat := specWidthLine{base: 0.010}
	cases := []struct {
		name     string
		max      int
		cost     specWidthLine
		accepted []int
		want     int
	}{
		{"every draft accepted keeps the full width", 4, rising, []int{4}, 4},
		{"every draft accepted keeps the widest cap", inkling2SpecMaxDraft, rising, []int{inkling2SpecMaxDraft}, inkling2SpecMaxDraft},
		{"only the first draft accepted under steep cost shrinks to one", 4, steep, []int{1}, 1},
		{"nothing accepted under steep cost shrinks to one", 4, steep, []int{0}, 1},
		{"two accepted under rising cost shrinks to two", 4, rising, []int{2}, 2},
		{"flat cost with partial deep acceptance keeps the full width", 4, flat, []int{4, 1}, 4},
		{"steep cost with partial deep acceptance still shrinks", 4, steep, []int{4, 1}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := timedSpecWidth(tc.max, tc.cost)
			for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
				c.Observe(tc.max, tc.max, tc.accepted[i%len(tc.accepted)], tc.cost.at(tc.max))
			}
			if got := nextAdaptive(&c); got != tc.want {
				t.Fatalf("width = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestInkling2SpecWidthColdStartUsesConfiguredWidth(t *testing.T) {
	steep := specWidthLine{base: 0.002, slope: 0.004}
	c := timedSpecWidth(4, steep)
	if got := c.Next(); got != 4 {
		t.Fatalf("width with no acceptance history = %d, want 4", got)
	}
	for i := 1; i < inkling2SpecWidthMinVerifies; i++ {
		c.Observe(4, 4, 0, steep.at(4))
		if got := c.Next(); got != 4 {
			t.Fatalf("width after %d verifies = %d, want 4 until %d", i, got, inkling2SpecWidthMinVerifies)
		}
	}
	c.Observe(4, 4, 0, steep.at(4))
	if got := nextAdaptive(&c); got != 1 {
		t.Fatalf("width after %d verifies = %d, want 1", inkling2SpecWidthMinVerifies, got)
	}
}

func TestInkling2SpecWidthNeedsTwoTimedWidthsToFitALine(t *testing.T) {
	c := newInkling2SpecWidth(4)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(4, 4, 0, 0.018)
		if got := c.Next(); got != 4 {
			t.Fatalf("width with one timed width = %d, want 4", got)
		}
	}
	c.ObserveCost(1, 0.006)
	if got := nextAdaptive(&c); got != 1 {
		t.Fatalf("width once a second width is timed = %d, want 1", got)
	}
}

func TestInkling2SpecWidthIgnoresUnusableTimings(t *testing.T) {
	bad := []float64{0, -0.01, math.NaN(), math.Inf(1), math.Inf(-1)}
	c := newInkling2SpecWidth(4)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(1+i%4, 1+i%4, 0, bad[i%len(bad)])
		if got := c.Next(); got != 4 {
			t.Fatalf("width with no usable timing = %d, want 4", got)
		}
	}

	steep := specWidthLine{base: 0.002, slope: 0.004}
	c = timedSpecWidth(4, steep)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(4, 4, 1, steep.at(4))
	}
	for i := 0; i < 64; i++ {
		c.ObserveCost(1+i%4, bad[i%len(bad)])
	}
	if got := nextAdaptive(&c); got != 1 {
		t.Fatalf("width after unusable timings = %d, want the fitted choice 1", got)
	}
}

func TestInkling2SpecWidthFallsBackOnNonPositivePrice(t *testing.T) {
	c := newInkling2SpecWidth(4)
	c.ObserveCost(1, 0.030)
	c.ObserveCost(2, 0.016)
	c.ObserveCost(3, 0.002)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(3, 3, 0, 0.002)
	}
	for i := 0; i < 2*inkling2SpecWidthProbeEvery; i++ {
		if got := c.Next(); got != 4 {
			t.Fatalf("width when the fitted line goes negative inside the range = %d, want 4", got)
		}
	}
}

func TestInkling2SpecWidthStaysInBounds(t *testing.T) {
	for _, tc := range []struct{ configured, want int }{
		{-3, 1}, {0, 1}, {1, 1}, {4, 4}, {inkling2SpecMaxDraft, inkling2SpecMaxDraft}, {100, inkling2SpecMaxDraft},
	} {
		c := newInkling2SpecWidth(tc.configured)
		if got := c.Next(); got != tc.want {
			t.Fatalf("configured %d: cold width = %d, want %d", tc.configured, got, tc.want)
		}
	}
	var zero inkling2SpecWidth
	zero.Observe(1, 1, 1, 0.01)
	if got := zero.Next(); got != 1 {
		t.Fatalf("zero value width = %d, want 1", got)
	}

	timings := []float64{0.004, 0.02, 0, -1, math.NaN(), math.Inf(1), 1e-12, 1e9}
	for maxWidth := 1; maxWidth <= inkling2SpecMaxDraft; maxWidth++ {
		c := newInkling2SpecWidth(maxWidth)
		seed := uint32(maxWidth)
		for i := 0; i < 4096; i++ {
			seed = seed*1664525 + 1013904223
			width := int(seed>>8)%(inkling2SpecMaxDraft+4) - 2
			accepted := int(seed>>16)%(inkling2SpecMaxDraft+4) - 2
			c.Observe(width, width, accepted, timings[int(seed>>24)%len(timings)])
			if got := c.Next(); got < 1 || got > maxWidth {
				t.Fatalf("max %d step %d: width = %d", maxWidth, i, got)
			}
		}
	}
}

func TestInkling2SpecWidthIsDeterministic(t *testing.T) {
	run := func() []int {
		cost := specWidthLine{base: 0.006, slope: 0.002}
		c := timedSpecWidth(6, cost)
		seed := uint32(7)
		widths := make([]int, 0, 512)
		for i := 0; i < 512; i++ {
			w := c.Next()
			widths = append(widths, w)
			seed = seed*1664525 + 1013904223
			c.Observe(w, w, int(seed>>16)%(w+1), cost.at(w))
		}
		return widths
	}
	a, b := run(), run()
	distinct := map[int]bool{}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("step %d: width %d then %d for the same observations", i, a[i], b[i])
		}
		distinct[a[i]] = true
	}
	if len(distinct) < 2 {
		t.Fatalf("widths never varied: %v", distinct)
	}
}

func TestInkling2SpecWidthProbesFullWidthPeriodically(t *testing.T) {
	steep := specWidthLine{base: 0.002, slope: 0.004}
	c := timedSpecWidth(4, steep)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(4, 4, 0, steep.at(4))
	}
	full := 0
	for i := 0; i < 4*inkling2SpecWidthProbeEvery; i++ {
		w := c.Next()
		if w == 4 {
			full++
		} else if w != 1 {
			t.Fatalf("step %d: width = %d, want 1 or the full-width probe", i, w)
		}
		c.Observe(w, w, 0, steep.at(w))
	}
	if full != 4 {
		t.Fatalf("full-width probes in %d steps = %d, want 4", 4*inkling2SpecWidthProbeEvery, full)
	}
}

func TestInkling2SpecWidthWidensWhenAcceptanceRecovers(t *testing.T) {
	rising := specWidthLine{base: 0.010, slope: 0.001}
	c := timedSpecWidth(4, rising)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(4, 4, 0, rising.at(4))
	}
	if got := nextAdaptive(&c); got != 1 {
		t.Fatalf("width after rejected drafts = %d, want 1", got)
	}
	for i := 0; i < 40*inkling2SpecWidthProbeEvery; i++ {
		w := c.Next()
		c.Observe(w, w, w, rising.at(w))
	}
	if got := nextAdaptive(&c); got != 4 {
		t.Fatalf("width after acceptance recovered = %d, want 4", got)
	}
}

func TestInkling2SpecWidthLongAcceptedRunKeepsFullWidth(t *testing.T) {
	rising := specWidthLine{base: 0.010, slope: 0.001}
	c := timedSpecWidth(4, rising)
	for i := 0; i < 20000; i++ {
		w := c.Next()
		if w != 4 {
			t.Fatalf("step %d: width = %d, want 4 while every draft is accepted", i, w)
		}
		c.Observe(w, w, w, rising.at(w))
	}
}

func TestInkling2SpecWidthTimesTheDraftedRows(t *testing.T) {
	steep := specWidthLine{base: 0.002, slope: 0.004}
	c := newInkling2SpecWidth(4)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		drafted := 1 + i%2
		c.Observe(4, drafted, 0, steep.at(drafted))
	}
	if got := nextAdaptive(&c); got != 1 {
		t.Fatalf("width with timings at the drafted widths 1 and 2 = %d, want 1", got)
	}
}

func TestInkling2SpecStepWidthIsFixedWithoutAChooser(t *testing.T) {
	steep := specWidthLine{base: 0.002, slope: 0.004}
	c := timedSpecWidth(4, steep)
	for i := 0; i < 4*inkling2SpecWidthMinVerifies; i++ {
		c.Observe(4, 4, 0, steep.at(4))
	}
	for i := 0; i < 2*inkling2SpecWidthProbeEvery; i++ {
		if got := inkling2SpecStepWidth(nil, 4); got != 4 {
			t.Fatalf("width with no chooser = %d, want the depth 4", got)
		}
	}
	if got := inkling2SpecStepWidth(&c, 4); got != 1 {
		t.Fatalf("width with a chooser = %d, want 1", got)
	}
	if got := inkling2SpecStepWidth(&c, 0); got != 0 {
		t.Fatalf("width at depth 0 = %d, want 0", got)
	}
}

func TestInkling2SpecAdaptIsOptIn(t *testing.T) {
	for value, want := range map[string]bool{"": false, "0": false, "true": false, "2": false, "1": true} {
		t.Setenv("FAK_INKLING2_SPEC_ADAPT", value)
		if got := inkling2SpecAdapt(); got != want {
			t.Fatalf("FAK_INKLING2_SPEC_ADAPT=%q: adapt = %v, want %v", value, got, want)
		}
	}
}

func TestInkling2SpecWidthStepDoesNotAllocate(t *testing.T) {
	cost := specWidthLine{base: 0.006, slope: 0.002}
	c := timedSpecWidth(inkling2SpecMaxDraft, cost)
	allocs := testing.AllocsPerRun(1000, func() {
		w := c.Next()
		c.Observe(w, w, w/2, cost.at(w))
	})
	if allocs != 0 {
		t.Fatalf("allocations per step = %v, want 0", allocs)
	}
}

func BenchmarkInkling2SpecWidth(b *testing.B) {
	cost := specWidthLine{base: 0.006, slope: 0.002}
	c := timedSpecWidth(inkling2SpecMaxDraft, cost)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		w := c.Next()
		c.Observe(w, w, i%(w+1), cost.at(w))
	}
}
