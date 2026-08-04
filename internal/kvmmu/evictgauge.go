package kvmmu

// evictgauge.go — issue #5123, the productization rung of the #3901 retained-mass gauge.
//
// #3901 shipped RetainedMass (recall.go) as a PURE, observability-only function: it grades a
// kept span set against the #855 accumulator's observed attention mass. What it did not have
// was a caller at the boundary its own seam names — "emitted alongside the eviction/selection
// decision." ApplyPlan, EvictColdest and EvictUnderBudget each returned only what they DROPPED
// and said nothing about how much of the observed mass the SURVIVORS carried, so the
// hit-QUALITY reading existed as a function nobody invoked.
//
// WIRED INTO THE DECISION, NOT OFFERED BESIDE IT. The grading happens inside those three entry
// points, so every caller of theirs emits the gauge with no change of its own — including the
// live served path, where the context planner's elision reaches ApplyPlan through
// agent.InKernelPlanner.ElideKVSpans. A parallel "...WithGauge" entry point would have replaced
// one uninvoked function with another; the reading is instead a property of the decision, read
// back with LastRetainedMass.
//
// DOES NOT CHANGE WHAT IS EVICTED. The selection logic is untouched: this reads the span ledger
// before the decision and grades the survivors after it. Every entry point keeps its exact
// signature and its exact choice of victims, so the gauge is additive and never a policy input.
//
// WHY THE LEDGER IS READ BEFORE THE DECISION. evict() clears a dropped span's rolling
// accumulators — Cumulative, EMA and the trajectory ring all go to zero, because a held span can
// no longer be attended to and its history is moot to the controller (kvmmu.go). That is right
// for the controller and fatal for the gauge: read AFTER the decision, the denominator would
// have already shrunk to the survivors and the fraction would read a meaningless 1.0 — precisely
// the failure recall.go warns about when it asks for a post-hoc (λ=1, no-Forget) accumulator. So
// the mass and token cost of every span live AT the decision are captured first, and that
// pre-decision set (kept + dropped) is the denominator.
//
// FAIL-CLOSED, INHERITED RATHER THAN RE-IMPLEMENTED. The reduction is the shipped RetainedMass
// over a post-hoc accumulator built from that capture, so the fail-closed rule comes with it:
// with no observer installed (#852 is default-off, every Cumulative is 0) the total mass is 0
// and the gauge reads Available=false with every ratio 0 — never a 0/0 NaN, never a spurious
// 1.0. No observer ⇒ no gauge, the eviction still happens, and the decision reports honestly
// that it has no quality reading to give.
//
// WHICH SCALAR. The gauge reads Cumulative, the undecayed per-span mass CloseTurn folds out of
// the in-flight Attended at a turn boundary. That is deliberately the same closed-turn ledger
// the coldest-by-EMA selection reads, so the gauge grades the decision on the evidence the
// decision itself saw. Mass from a turn not yet closed is in neither.

// LastRetainedMass is the #3901 hit-QUALITY reading of the most recent eviction/selection
// decision this context took: the fraction of the observed attention mass the SURVIVORS carried,
// Σ(kept mass) / Σ(all observed mass) over the set that was live when the decision was taken.
// Fraction is the retained_mass_fraction the #3901 seam defines and TokenFraction is what the
// survivors cost, so an operator reads "the kept spans captured F of the mass at T of the
// tokens."
//
// Available is the fail-closed bit and MUST be consulted before reporting the bare number: false
// means no attention was witnessed, so every ratio is 0 and "the survivors captured none of the
// mass" is indistinguishable from "no observer, no reading" without it. A context that has taken
// no decision yet also reads Available=false.
func (c *Context) LastRetainedMass() RetainedMassGauge { return c.lastRetained }

// gradeRetention emits the gauge for the decision that just ran: the survivors (read after it,
// so exactly the spans still resident) graded against the pre-decision ledger `before` captured
// by liveMassLedger. A nil `before` is the fail-closed path — no mass was observed, so the
// decision is recorded as having no reading rather than a fabricated one, without walking the
// ledger again.
func (c *Context) gradeRetention(before *AttentionAccumulator, cost map[string]int) {
	if before == nil {
		c.lastRetained = RetainedMassGauge{}
		return
	}
	c.lastRetained = RetainedMass(before, c.liveSpanIDs(), cost)
}

// liveMassLedger captures the candidate set a decision is about to rule on: each live span's
// undecayed Cumulative mass folded into a post-hoc accumulator (λ=1, no Forget — the posture
// RetainedMass requires so a dropped span stays in the denominator), plus its resident token
// cost. Held / zero-length spans are excluded: they left the cache at an EARLIER decision and
// their accumulators were already cleared, so counting them would pad the denominator with zeros
// rather than with mass. Masses and costs are summed per id, so a repeated span id folds into
// one honest entry instead of shadowing.
//
// Returns (nil, nil) when no live span carries any mass — the default-off observer case (#852),
// which is the common one on a served path. RetainedMass already fails closed on a nil
// accumulator, so short-circuiting here keeps the un-observed decision free of the map, the
// accumulator and the snapshot it would otherwise allocate to report nothing.
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

// liveSpanIDs lists the spans still resident — the gauge's kept set. Read AFTER a decision runs,
// so it is exactly the survivors: evict marks a dropped span Held and sets its Len to 0.
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
