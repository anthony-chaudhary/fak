package systembaseline

import (
	"math"
	"time"
)

// CadencePolicy bounds adaptive process-census sampling. The sampler never
// stretches beyond Maximum, because doing so would silently lose churn coverage.
type CadencePolicy struct {
	Minimum            time.Duration
	Maximum            time.Duration
	MaximumDutyPercent float64
}

// CadenceController derives the next census interval from witnessed cost.
type CadenceController struct {
	policy     CadencePolicy
	effective  time.Duration
	overloaded bool
}

func NewCadenceController(p CadencePolicy) *CadenceController {
	if p.Minimum <= 0 {
		p.Minimum = 10 * time.Millisecond
	}
	if p.Maximum < p.Minimum {
		p.Maximum = p.Minimum
	}
	if p.MaximumDutyPercent <= 0 || p.MaximumDutyPercent > 100 {
		p.MaximumDutyPercent = 10
	}
	return &CadenceController{policy: p, effective: p.Minimum}
}
func (c *CadenceController) Observe(cost time.Duration) time.Duration {
	// A 25% reserve absorbs process-churn variance between one census and the next;
	// without it a controller that targets the ceiling oscillates just above policy.
	required := time.Duration(math.Ceil(float64(cost) * 125 / c.policy.MaximumDutyPercent))
	if required < c.policy.Minimum {
		required = c.policy.Minimum
	}
	c.overloaded = required > c.policy.Maximum
	if required > c.policy.Maximum {
		required = c.policy.Maximum
	}
	c.effective = required
	return required
}
func (c *CadenceController) Effective() time.Duration { return c.effective }
func (c *CadenceController) Overloaded() bool         { return c.overloaded }

// StableProcessCache caches identity-stable process metadata without confusing
// PID reuse. Dynamic CPU/RSS counters intentionally remain uncached.
type StableProcessCache struct{ entries map[int]stableProcess }
type stableProcess struct {
	start uint64
	image string
}

func (c *StableProcessCache) Apply(samples []ProcessSample) (hits, misses int) {
	if c.entries == nil {
		c.entries = make(map[int]stableProcess)
	}
	seen := make(map[int]bool, len(samples))
	for i := range samples {
		p := &samples[i]
		seen[p.PID] = true
		old, ok := c.entries[p.PID]
		if ok && old.start == p.StartID {
			hits++
			if p.Image == "" {
				p.Image = old.image
			}
		} else {
			misses++
			c.entries[p.PID] = stableProcess{p.StartID, p.Image}
		}
	}
	for pid := range c.entries {
		if !seen[pid] {
			delete(c.entries, pid)
		}
	}
	return hits, misses
}
