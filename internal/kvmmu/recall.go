package kvmmu

import "sort"

// recall.go — hit-quality retention mass measurement over observed attention.

// RetainedMassGauge measures the attention mass fraction and token cost retained by kept spans.
type RetainedMassGauge struct {
	Fraction      float64 `json:"fraction"`       // fraction of observed mass in [0,1]
	KeptMass      float64 `json:"kept_mass"`      // cumulative mass of kept spans
	TotalMass     float64 `json:"total_mass"`     // cumulative mass across all observed spans
	KeptTokens    int     `json:"kept_tokens"`    // resident tokens of kept spans
	TotalTokens   int     `json:"total_tokens"`   // resident tokens across all observed spans
	TokenFraction float64 `json:"token_fraction"` // token ratio in [0,1]
	Available     bool    `json:"available"`      // false when no attention was observed (fail-closed)
}

// RetainedMass grades a kept span set against accumulated observed attention.
// Fails closed if accumulator is nil or total observed mass is zero.
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

// TopKSpansByMass returns up to k span IDs carrying the most cumulative attention mass.
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
