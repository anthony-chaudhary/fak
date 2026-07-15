package ablate

import (
	"errors"
	"fmt"
	"sort"
)

const (
	SweepPlanMain     = "main"
	SweepPlanPairwise = "pairwise"
	SweepPlanGreedy   = "greedy"
)

// BuildSweepPlan builds a bounded plan. ranked lists concepts from largest measured
// main effect to smallest; pairwise requires that measured ranking rather than guessing.
func BuildSweepPlan(mode string, top int, ranked []string) ([]FeatureConfig, error) {
	known := KnownFeatures()
	switch mode {
	case SweepPlanMain:
		return buildMainPlan(known), nil
	case SweepPlanPairwise:
		if top < 2 {
			return nil, errors.New("ablate: pairwise sweep plan requires --top >= 2")
		}
		if len(ranked) == 0 {
			return nil, errors.New("ablate: pairwise sweep plan requires measured main-effect ranking")
		}
		if top > len(ranked) {
			top = len(ranked)
		}
		selected := append([]string(nil), ranked[:top]...)
		for _, token := range selected {
			if _, ok := registeredConcept(token); !ok {
				return nil, fmt.Errorf("ablate: unknown ranked concept %q", token)
			}
		}
		out := buildMainPlan(known)
		for i := 0; i < len(selected); i++ {
			for j := i + 1; j < len(selected); j++ {
				c := allOff("pair-"+selected[i]+"+"+selected[j], known)
				c.apply(selected[i], true)
				c.apply(selected[j], true)
				out = append(out, c)
			}
		}
		return out, nil
	case SweepPlanGreedy:
		return nil, errors.New("ablate: greedy sweep plan requires iterative measurements and is not yet wired")
	default:
		return nil, fmt.Errorf("ablate: unknown sweep plan %q (want main or pairwise)", mode)
	}
}

func buildMainPlan(tokens []string) []FeatureConfig {
	out := []FeatureConfig{allOff("all-off", tokens)}
	for _, token := range tokens {
		c := allOff(token, tokens)
		c.apply(token, true)
		out = append(out, c)
	}
	return out
}

func allOff(name string, tokens []string) FeatureConfig {
	c := FeatureConfig{Name: name}
	for _, token := range tokens {
		c.apply(token, false)
	}
	return c
}

// RankMainEffects returns concept tokens ordered by absolute token-equivalent effect, then token.
func RankMainEffects(rep *Report) []string {
	type scored struct {
		token string
		score float64
	}
	var rows []scored
	for _, run := range rep.Runs {
		if _, ok := registeredConcept(run.ArmID); !ok {
			continue
		}
		score := run.TotalTokenEquiv()
		if score < 0 {
			score = -score
		}
		rows = append(rows, scored{run.ArmID, score})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].score == rows[j].score {
			return rows[i].token < rows[j].token
		}
		return rows[i].score > rows[j].score
	})
	out := make([]string, len(rows))
	for i := range rows {
		out[i] = rows[i].token
	}
	return out
}
