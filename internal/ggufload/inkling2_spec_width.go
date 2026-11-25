package ggufload

import (
	"math"
	"os"
)

const (
	// inkling2SpecWidthMinVerifies is how many verify outcomes the chooser sees before
	// it leaves the configured width.
	inkling2SpecWidthMinVerifies = 8
	// inkling2SpecWidthProbeEvery makes one step in this many run the configured width,
	// which is the only way positions past a narrowed width are sampled again.
	inkling2SpecWidthProbeEvery = 16
	// inkling2SpecWidthWindow caps the weight of past samples, so both estimates follow
	// the recent steps of a long run.
	inkling2SpecWidthWindow = 64
)

// inkling2SpecAdapt reports whether the speculative loop picks its draft width per
// step; only FAK_INKLING2_SPEC_ADAPT=1 selects it.
func inkling2SpecAdapt() bool { return os.Getenv("FAK_INKLING2_SPEC_ADAPT") == "1" }

// inkling2SpecWidth picks how many drafted tokens the next speculative step verifies:
// the width with the most expected kept tokens per second of verify. It returns the
// configured width until it has inkling2SpecWidthMinVerifies outcomes and verify
// timings at two distinct widths.
type inkling2SpecWidth struct {
	max        int
	verifies   int
	sinceProbe int
	// tried[i] counts verifies that drafted position i, kept[i] those that kept it.
	tried, kept [inkling2SpecMaxDraft]float64
	cost        inkling2SpecCostFit
}

// inkling2SpecCostFit is the running least-squares fit of verify seconds against
// draft width.
type inkling2SpecCostFit struct {
	n, x, y, xx, xy float64
	timed           uint16
}

func newInkling2SpecWidth(maxWidth int) inkling2SpecWidth {
	return inkling2SpecWidth{max: min(max(maxWidth, 1), inkling2SpecMaxDraft)}
}

// Next returns the draft width of the next step, from 1 to the configured width.
func (c *inkling2SpecWidth) Next() int {
	full := max(c.max, 1)
	if c.verifies < inkling2SpecWidthMinVerifies {
		return full
	}
	if c.sinceProbe++; c.sinceProbe >= inkling2SpecWidthProbeEvery {
		c.sinceProbe = 0
		return full
	}
	base, slope, ok := c.cost.line(full)
	if !ok {
		return full
	}
	// the verify always yields the target's own next token, hence the leading 1
	expected, chance := 1.0, 1.0
	best, width := 0.0, full
	for w := 1; w <= full; w++ {
		if c.tried[w-1] > 0 {
			chance = c.kept[w-1] / c.tried[w-1]
		}
		expected += chance
		if rate := expected / (base + slope*float64(w)); rate > best {
			best, width = rate, w
		}
	}
	return width
}

// Observe records one verify of a step that asked the drafter for width tokens, got
// drafted of them, kept the first accepted and took verifySeconds. Acceptance counts
// against the asked width, which is what Next chooses between; the time belongs to
// the drafted rows the verify ran.
func (c *inkling2SpecWidth) Observe(width, drafted, accepted int, verifySeconds float64) {
	c.ObserveCost(drafted, verifySeconds)
	if width < 1 || width > c.max || accepted < 0 {
		return
	}
	accepted = min(accepted, width)
	for i := 0; i < width; i++ {
		if c.tried[i] >= inkling2SpecWidthWindow {
			c.tried[i] /= 2
			c.kept[i] /= 2
		}
		c.tried[i]++
		if i < accepted {
			c.kept[i]++
		}
	}
	if c.verifies < inkling2SpecWidthMinVerifies {
		c.verifies++
	}
}

// inkling2SpecStepWidth is the draft width of the next step: depth with no chooser.
func inkling2SpecStepWidth(c *inkling2SpecWidth, depth int) int {
	if c == nil {
		return depth
	}
	return min(c.Next(), depth)
}

// ObserveCost records the time of a verify whose outcome says nothing about the
// drafter, such as a warm-up run.
func (c *inkling2SpecWidth) ObserveCost(width int, verifySeconds float64) {
	if width < 1 || width > c.max {
		return
	}
	c.cost.observe(width, verifySeconds)
}

func (f *inkling2SpecCostFit) observe(width int, seconds float64) {
	if math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds <= 0 {
		return
	}
	if f.n >= inkling2SpecWidthWindow {
		f.n, f.x, f.y, f.xx, f.xy = f.n/2, f.x/2, f.y/2, f.xx/2, f.xy/2
	}
	w := float64(width)
	f.n++
	f.x += w
	f.y += seconds
	f.xx += w * w
	f.xy += w * seconds
	f.timed |= 1 << width
}

// line returns the fitted seconds = base + slope*width, and false when fewer than two
// widths are timed or the line is not a positive finite price at every width up to full.
func (f *inkling2SpecCostFit) line(full int) (base, slope float64, ok bool) {
	if f.timed&(f.timed-1) == 0 {
		return 0, 0, false
	}
	// once every width but one has decayed away the spread is rounding noise
	spread := f.n*f.xx - f.x*f.x
	if !(spread > 1e-9*f.n*f.xx) {
		return 0, 0, false
	}
	slope = (f.n*f.xy - f.x*f.y) / spread
	base = (f.y - slope*f.x) / f.n
	first, last := base+slope, base+slope*float64(full)
	if !(first > 0) || !(last > 0) || math.IsInf(first, 0) || math.IsInf(last, 0) {
		return 0, 0, false
	}
	return base, slope, true
}
