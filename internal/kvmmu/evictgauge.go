package kvmmu

// evictgauge.go — retained attention mass gauge for eviction decisions.

// LastRetainedMass returns the hit-quality reading of the most recent eviction or selection
// decision: fraction of observed attention mass kept by surviving spans.
func (c *Context) LastRetainedMass() RetainedMassGauge { return c.lastRetained }

// gradeRetention evaluates retention for the survivors against the pre-decision ledger.
func (c *Context) gradeRetention(before *AttentionAccumulator, cost map[string]int) {
	if before == nil {
		c.lastRetained = RetainedMassGauge{}
		return
	}
	c.lastRetained = RetainedMass(before, c.liveSpanIDs(), cost)
}

// liveMassLedger captures undecayed mass and token costs for live unheld spans.
// Returns (nil, nil) if no span has positive cumulative mass.
func (c *Context) liveMassLedger() (*AttentionAccumulator, map[string]int) {
	var total float64
	for _, s := range c.segs {
		if s.Held || s.Len == 0 {
			continue
		}
		total += s.Cumulative
	}
	if total <= 0 {
		return nil, nil
	}
	mass := make(map[string]float64, len(c.segs))
	cost := make(map[string]int, len(c.segs))
	for _, s := range c.segs {
		if s.Held || s.Len == 0 {
			continue
		}
		mass[s.ID] += s.Cumulative
		cost[s.ID] += s.Len
	}
	acc := NewAttentionAccumulator(1, 0)
	acc.Observe(mass)
	return acc, cost
}

// liveSpanIDs returns IDs of currently resident unheld spans.
func (c *Context) liveSpanIDs() []string {
	out := make([]string, 0, len(c.segs))
	for _, s := range c.segs {
		if s.Held || s.Len == 0 {
			continue
		}
		out = append(out, s.ID)
	}
	return out
}
