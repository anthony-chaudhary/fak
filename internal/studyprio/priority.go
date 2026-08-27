// Package studyprio ranks uncovered mechanisms with deterministic hard gates and dependencies.
package studyprio

import (
	"errors"
	"sort"
)

type Scores struct{ Centrality, NativeImpact, EndToEndValue, Evidence, Recurrence, DependencyUnlock, ImplementationCost, HardwareCost, Risk, Duplication int }
type Frame struct{ For, Problem, Today, BetterBecause, Witness, Centrality, P1P4 string }
type Candidate struct {
	ID, Category, Horizon string
	Scores                Scores
	HardGatePass          bool
	GateReason            string
	Dependencies          []string
	Frame                 Frame
}
type Ranked struct {
	Candidate Candidate
	Score     int
	Priority  string
}

var ErrInvalid = errors.New("studyprio: invalid ledger")

func total(s Scores) int {
	return s.Centrality + s.NativeImpact + s.EndToEndValue + s.Evidence + s.Recurrence + s.DependencyUnlock - s.ImplementationCost - s.HardwareCost - s.Risk - s.Duplication
}
func Rank(v []Candidate) ([]Ranked, error) {
	by := map[string]Candidate{}
	for _, c := range v {
		if c.ID == "" || c.Frame.For == "" || c.Frame.Witness == "" || c.Horizon == "" || c.Category == "" {
			return nil, ErrInvalid
		}
		if _, ok := by[c.ID]; ok {
			return nil, ErrInvalid
		}
		by[c.ID] = c
	}
	state := map[string]int{}
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return ErrInvalid
		}
		if state[id] == 2 {
			return nil
		}
		c, ok := by[id]
		if !ok {
			return ErrInvalid
		}
		state[id] = 1
		for _, d := range c.Dependencies {
			if err := visit(d); err != nil {
				return err
			}
		}
		state[id] = 2
		return nil
	}
	for id := range by {
		if err := visit(id); err != nil {
			return nil, err
		}
	}
	var out []Ranked
	for _, c := range v {
		score := total(c.Scores)
		p := "deferred"
		if !c.HardGatePass {
			p = "rejected"
		} else if score >= 20 {
			p = "P0"
		} else if score >= 10 {
			p = "P1"
		} else if score >= 0 {
			p = "P2"
		}
		out = append(out, Ranked{c, score, p})
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Candidate.HardGatePass != b.Candidate.HardGatePass {
			return a.Candidate.HardGatePass
		}
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		return a.Candidate.ID < b.Candidate.ID
	})
	rankedByID := map[string]Ranked{}
	for _, r := range out {
		rankedByID[r.Candidate.ID] = r
	}
	ordered := make([]Ranked, 0, len(out))
	emitted := map[string]bool{}
	var emit func(string)
	emit = func(id string) {
		if emitted[id] {
			return
		}
		for _, dep := range rankedByID[id].Candidate.Dependencies {
			emit(dep)
		}
		emitted[id] = true
		ordered = append(ordered, rankedByID[id])
	}
	for _, r := range out {
		emit(r.Candidate.ID)
	}
	out = ordered
	return out, nil
}
