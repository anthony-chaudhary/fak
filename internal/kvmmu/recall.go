package kvmmu

import "sort"

// recall.go — issue #3901, the hit-QUALITY dual of internal/cacheobs's hit-RATE.
//
// cacheobs answers "did we serve from cache?" (reuse RATE = reused/prompt). It cannot
// answer "was the set we KEPT the valuable one?" — that is a QUALITY question, and the
// only witness for it is the attention mass each span actually drew (the #851 epic's
// carrier #852 → attributor #853 → accumulator #855). Rung 4 (#855, accumulator.go) folds
// that stream into a per-span undecayed Cumulative mass. This file reduces those masses
// against a retention/eviction decision into ONE gauge:
//
//	retained_mass_fraction = Σ(cumulative mass of KEPT spans) / Σ(cumulative mass of ALL observed spans)
//
// It grades "after eviction the kept spans captured 92% of the observed attention mass at
// 40% of the tokens." Rate says WHETHER we hit; recall says HOW MUCH OF THE VALUE the
// retained set carried — orthogonal axes.
//
// OBSERVABILITY ONLY. The gauge reads the #855 accumulator and a kept-id set; it never
// selects, evicts, or perturbs a forward pass. Feed it a post-hoc accumulator (λ=1, no
// Forget) so an evicted span stays in the denominator — otherwise "all observed spans"
// would silently shrink to the survivors and the fraction would read a meaningless 1.0.
//
// FAIL-CLOSED. When no attention was observed (the observer is default-off, #852, so every
// span's Cumulative is 0) there is no honest quality to report: Available is false and
// every ratio is 0. This matches attn_observer's default-off / zero-alloc invariant — no
// observer installed ⇒ no gauge, never a fabricated 0/0 → NaN or a spurious 1.0.

// RetainedMassGauge is the hit-QUALITY reading of a retained/selected span set: the
// fraction of the total observed attention mass the KEPT spans captured, plus the token
// fraction those spans cost, so an operator reads "Fraction of the mass at TokenFraction
// of the tokens." Available is the fail-closed bit: false means no attention was observed
// and no ratio is meaningful.
type RetainedMassGauge struct {
	Fraction      float64 `json:"fraction"`       // Σ(kept mass) / Σ(all observed mass), in [0,1]; 0 when unavailable
	KeptMass      float64 `json:"kept_mass"`      // Σ cumulative mass over the kept spans
	TotalMass     float64 `json:"total_mass"`     // Σ cumulative mass over every observed span (kept + dropped)
	KeptTokens    int     `json:"kept_tokens"`    // Σ resident tokens of the kept spans (0 when cost unknown)
	TotalTokens   int     `json:"total_tokens"`   // Σ resident tokens of every observed span (0 when cost unknown)
	TokenFraction float64 `json:"token_fraction"` // KeptTokens / TotalTokens, in [0,1]; 0 when cost unknown
	Available     bool    `json:"available"`      // false when no attention was observed (fail-closed)
}

// RetainedMass grades a kept span set against the #855 accumulator's observed attention.
// The denominator is Σ Cumulative over every span the accumulator has seen (feed it
// post-hoc — λ=1, no Forget — so evicted spans stay counted, matching "all observed
// spans"); the numerator is Σ Cumulative over the surviving ids. cost (span id → resident
// tokens) is optional: with it the gauge also reports the token fraction; a nil/partial
// map leaves the unknown token counts at 0.
//
// Fail-closed: a nil accumulator, or one that has observed zero total mass (no observer
// installed — attn_observer is default-off, #852), yields Available=false and every ratio
// 0 — never a 0/0 NaN or a spurious 1.0. A kept id the accumulator never saw contributes 0
// mass (a pinned-but-never-attended span is honest dead weight to the numerator), never an
// error. Pure and deterministic: the same (snapshot, kept set, cost) yields the same gauge.
func RetainedMass(acc *AttentionAccumulator, kept []string, cost map[string]int) RetainedMassGauge {
	var g RetainedMassGauge
	if acc == nil {
		return g
	}
	snap := acc.Snapshot()
	for _, s := range snap {
		g.TotalMass += s.Cumulative
		g.TotalTokens += cost[s.ID]
	}
	if g.TotalMass <= 0 {
		// No attention observed over any span: fail closed — no gauge, no token totals.
		return RetainedMassGauge{}
	}
	keptSet := make(map[string]struct{}, len(kept))
	for _, id := range kept {
		keptSet[id] = struct{}{}
	}
	for _, s := range snap {
		if _, ok := keptSet[s.ID]; ok {
			g.KeptMass += s.Cumulative
			g.KeptTokens += cost[s.ID]
		}
	}
	g.Fraction = g.KeptMass / g.TotalMass
	if g.TotalTokens > 0 {
		g.TokenFraction = float64(g.KeptTokens) / float64(g.TotalTokens)
	}
	g.Available = true
	return g
}

// TopKSpansByMass returns the k span ids carrying the most cumulative attention mass (ties
// broken by id for determinism), most-massive first. Keeping exactly this set is the
// mass-maximizing choice of k spans: because span masses are non-negative and additive
// across disjoint spans, no other k-subset can capture more total mass — so RetainedMass
// over this set is the CEILING any k-span retention can reach. It is the reference an
// eviction policy is graded against: a controller that keeps the k warmest spans hits this
// ceiling; one that keeps a colder set falls short of it by a measurable mass gap.
//
// Returns every observed span (fewer than k) when the accumulator has seen fewer than k,
// and nil for a nil accumulator or k <= 0.
func TopKSpansByMass(acc *AttentionAccumulator, k int) []string {
	if acc == nil || k <= 0 {
		return nil
	}
	snap := acc.Snapshot()
	sort.SliceStable(snap, func(i, j int) bool {
		if snap[i].Cumulative != snap[j].Cumulative {
			return snap[i].Cumulative > snap[j].Cumulative
		}
		return snap[i].ID < snap[j].ID
	})
	if k > len(snap) {
		k = len(snap)
	}
	out := make([]string, k)
	for i := 0; i < k; i++ {
		out[i] = snap[i].ID
	}
	return out
}
