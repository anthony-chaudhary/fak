package dispatchtick

import (
	"math"
	"sort"
	"strings"
)

// DispatchLaneCandidate is the normalized input to unattended whole-lane scoring.
type DispatchLaneCandidate struct {
	Lane       string
	Priority   int
	StepBudget int
	Count      int
	Core       bool
}

// DispatchLaneScorer is one independent whole-lane selection signal.
type DispatchLaneScorer interface {
	Name() string
	Weight() float64
	Score(DispatchLaneCandidate) float64
}

type DispatchLaneScorerFunc struct {
	ScorerName   string
	ScorerWeight float64
	Fn           func(DispatchLaneCandidate) float64
}

func (s DispatchLaneScorerFunc) Name() string                          { return s.ScorerName }
func (s DispatchLaneScorerFunc) Weight() float64                       { return s.ScorerWeight }
func (s DispatchLaneScorerFunc) Score(c DispatchLaneCandidate) float64 { return s.Fn(c) }

type DispatchLaneScorerRegistry struct{ scorers []DispatchLaneScorer }

func NewDispatchLaneScorerRegistry(scorers ...DispatchLaneScorer) *DispatchLaneScorerRegistry {
	r := &DispatchLaneScorerRegistry{}
	for _, s := range scorers {
		r.Register(s)
	}
	return r
}

func (r *DispatchLaneScorerRegistry) Register(s DispatchLaneScorer) {
	if s == nil || strings.TrimSpace(s.Name()) == "" || s.Weight() <= 0 || math.IsNaN(s.Weight()) {
		return
	}
	r.scorers = append(r.scorers, s)
}

// Order ranks lanes by weighted score. Values are clamped before weighting and
// lexical lane order remains the deterministic exact-score tie break.
func (r *DispatchLaneScorerRegistry) Order(cands []DispatchLaneCandidate) []DispatchLaneCandidate {
	type scored struct {
		c     DispatchLaneCandidate
		score float64
	}
	out := make([]scored, 0, len(cands))
	for _, c := range cands {
		if strings.TrimSpace(c.Lane) == "" {
			continue
		}
		s := scored{c: c}
		for _, scorer := range r.scorers {
			v := scorer.Score(c)
			if math.IsNaN(v) || v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			s.score += v * scorer.Weight()
		}
		out = append(out, s)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		return out[i].c.Lane < out[j].c.Lane
	})
	ordered := make([]DispatchLaneCandidate, len(out))
	for i := range out {
		ordered[i] = out[i].c
	}
	return ordered
}

func DefaultDispatchLaneScorers(highPriority bool, maxStepBudget, maxCount int) *DispatchLaneScorerRegistry {
	norm := func(v, max int) float64 {
		if max <= 0 || v <= 0 {
			return 0
		}
		return float64(v) / float64(max)
	}
	r := NewDispatchLaneScorerRegistry()
	// Weights encode the old lexicographic invariants: each leading signal weighs
	// more than the maximum combined score of every signal below it.
	if highPriority {
		r.Register(DispatchLaneScorerFunc{"has-work", 16, func(c DispatchLaneCandidate) float64 {
			if c.Count > 0 {
				return 1
			}
			return 0
		}})
		r.Register(DispatchLaneScorerFunc{"priority", 8, func(c DispatchLaneCandidate) float64 { return norm(c.Priority, PriorityWeightP0) }})
	}
	r.Register(DispatchLaneScorerFunc{"core", 4, func(c DispatchLaneCandidate) float64 {
		if c.Core {
			return 1
		}
		return 0
	}})
	r.Register(DispatchLaneScorerFunc{"step-budget", 2, func(c DispatchLaneCandidate) float64 { return norm(c.StepBudget, maxStepBudget) }})
	r.Register(DispatchLaneScorerFunc{"count", 1, func(c DispatchLaneCandidate) float64 { return norm(c.Count, maxCount) }})
	return r
}
