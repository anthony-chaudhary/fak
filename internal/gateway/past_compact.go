package gateway

import "strconv"

const (
	PastCompactActionCheckpoint = "checkpoint"
	PastCompactActionRotate     = "rotate"
	PastCompactActionPark       = "park"
	pastCompactEscalateAt       = 3
)

// PastCompactEscalation is the structured guard/dev-ex state emitted after
// repeated past-inversion nudges. A zero Consecutive value means the trace reset.
type PastCompactEscalation struct {
	Trace       string `json:"trace"`
	Consecutive int    `json:"consecutive"`
	Action      string `json:"action"`
	Escalated   bool   `json:"escalated"`
}

func (s *Server) observePastCompact(trace string, past, checkpointed bool) PastCompactEscalation {
	if s == nil || trace == "" {
		return PastCompactEscalation{}
	}
	s.pastCompactMu.Lock()
	defer s.pastCompactMu.Unlock()
	if s.pastCompactRuns == nil {
		s.pastCompactRuns = map[string]int{}
	}
	if checkpointed || !past {
		delete(s.pastCompactRuns, trace)
		return PastCompactEscalation{Trace: trace}
	}
	n := s.pastCompactRuns[trace] + 1
	if len(s.pastCompactRuns) >= maxResetHealthSessions {
		for k := range s.pastCompactRuns {
			if k != trace {
				delete(s.pastCompactRuns, k)
				break
			}
		}
	}
	s.pastCompactRuns[trace] = n
	action := PastCompactActionCheckpoint
	if n >= pastCompactEscalateAt {
		action = PastCompactActionRotate
	}
	if n >= pastCompactEscalateAt*2 {
		action = PastCompactActionPark
	}
	return PastCompactEscalation{Trace: trace, Consecutive: n, Action: action, Escalated: n >= pastCompactEscalateAt}
}

func formatPastCompactEscalation(v PastCompactEscalation) string {
	if !v.Escalated {
		return ""
	}
	return " escalation=past-compact-repeat consecutive=" + strconv.Itoa(v.Consecutive) + " next_action=" + v.Action
}
