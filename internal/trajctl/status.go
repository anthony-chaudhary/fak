package trajctl

import "fmt"

const StatusSchema = "fak-trajctl-status/1"

type WitnessCount struct {
	WitnessRung WitnessRung `json:"WitnessRung"`
	Count       int         `json:"count"`
}
type Status struct {
	Schema    string           `json:"schema"`
	Open      []ObjectiveCurve `json:"open"`
	Witnesses []WitnessCount   `json:"witnesses"`
}

func (s State) Status() Status {
	out := Status{Schema: StatusSchema, Open: s.OpenCurves().Objectives}
	counts := map[WitnessRung]int{}
	for _, score := range s.Scores {
		counts[score.Witness]++
	}
	for _, w := range []WitnessRung{W3, W2, W1, W0} {
		if counts[w] > 0 {
			out.Witnesses = append(out.Witnesses, WitnessCount{w, counts[w]})
		}
	}
	return out
}

func (s Status) Empty() bool { return len(s.Open) == 0 && len(s.Witnesses) == 0 }

func (s Status) Lines() []string {
	if s.Empty() {
		return nil
	}
	lines := []string{"trajectory:"}
	for _, o := range s.Open {
		lines = append(lines, fmt.Sprintf("  %s  %s  %.2f (%+.2f)", o.ObjectiveID, o.Signal, o.Latest, o.Delta))
	}
	if len(s.Witnesses) > 0 {
		line := "  witnesses:"
		for _, w := range s.Witnesses {
			line += fmt.Sprintf(" %s=%d", w.WitnessRung, w.Count)
		}
		lines = append(lines, line)
	}
	return lines
}
