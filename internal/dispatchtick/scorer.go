package dispatchtick

import (
	"math"
	"sort"
)

// LaneScorer is one additive dispatch-selection signal. Scores are clamped to
// [0,1] before Weight is applied, so a faulty plugin cannot dominate the pick.
type LaneScorer interface {
	Name() string
	Weight() float64
	Score(LaneCandidate) float64
}

// LaneScorerFunc adapts a function into a named weighted scorer.
type LaneScorerFunc struct {
	ScorerName   string
	ScorerWeight float64
	Fn           func(LaneCandidate) float64
}

func (s LaneScorerFunc) Name() string                  { return s.ScorerName }
func (s LaneScorerFunc) Weight() float64               { return s.ScorerWeight }
func (s LaneScorerFunc) Score(c LaneCandidate) float64 { return s.Fn(c) }

// LaneScorerRegistry is the additive extension seam for dispatch ranking.
type LaneScorerRegistry struct{ scorers []LaneScorer }

func NewLaneScorerRegistry(scorers ...LaneScorer) *LaneScorerRegistry {
	r := &LaneScorerRegistry{}
	for _, s := range scorers {
		r.Register(s)
	}
	return r
}

func (r *LaneScorerRegistry) Register(s LaneScorer) {
	if s == nil || s.Name() == "" || s.Weight() <= 0 || math.IsNaN(s.Weight()) {
		return
	}
	r.scorers = append(r.scorers, s)
}

// Order returns candidates by descending weighted score, retaining the existing
// issue-number recency policy for exact ties.
func (r *LaneScorerRegistry) Order(cands []LaneCandidate, preferNewest bool) []int {
	type scored struct {
		candidate LaneCandidate
		score     float64
	}
	ordered := make([]scored, len(cands))
	for i, c := range cands {
		ordered[i].candidate = c
		for _, s := range r.scorers {
			v := s.Score(c)
			if math.IsNaN(v) || v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			ordered[i].score += v * s.Weight()
		}
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].score != ordered[j].score {
			return ordered[i].score > ordered[j].score
		}
		return numberTiebreak(ordered[i].candidate.Number, ordered[j].candidate.Number, preferNewest)
	})
	return orderedNumbers(len(ordered), func(i int) int { return ordered[i].candidate.Number })
}

var defaultLaneScorers = NewLaneScorerRegistry(LaneScorerFunc{
	ScorerName: "priority", ScorerWeight: 1,
	Fn: func(c LaneCandidate) float64 { return float64(c.Weight) / PriorityWeightP0 },
})
