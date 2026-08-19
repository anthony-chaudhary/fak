package gateway

import "net/http"

const FleetResponseSchema = "fak-fleet-status/1"

type FleetSessionCard struct {
	TraceID    string `json:"trace_id"`
	Run        string `json:"run"`
	Reason     string `json:"reason,omitempty"`
	Priority   int    `json:"priority"`
	TurnsLeft  int    `json:"turns_left"`
	TokensLeft int    `json:"tokens_left"`
	Rev        uint64 `json:"rev"`
}
type FleetRollup struct {
	Sessions       int `json:"sessions"`
	Running        int `json:"running"`
	Blocked        int `json:"blocked"`
	Throttled      int `json:"throttled"`
	BudgetPressure int `json:"budget_pressure"`
}
type FleetStatusResponse struct {
	Schema   string             `json:"schema"`
	Sessions []FleetSessionCard `json:"sessions"`
	Rollup   FleetRollup        `json:"rollup"`
}

func fleetStatus(states []SessionState) FleetStatusResponse {
	out := FleetStatusResponse{Schema: FleetResponseSchema, Sessions: make([]FleetSessionCard, 0, len(states))}
	out.Rollup.Sessions = len(states)
	for _, state := range states {
		card := FleetSessionCard{TraceID: state.TraceID, Run: state.Run, Reason: state.Reason, Priority: state.Priority, TurnsLeft: state.Budget.TurnsLeft, TokensLeft: state.Budget.TokensLeft, Rev: state.Rev}
		out.Sessions = append(out.Sessions, card)
		switch state.Run {
		case "running":
			out.Rollup.Running++
		case "blocked", "paused", "stopped":
			out.Rollup.Blocked++
		case "throttled":
			out.Rollup.Throttled++
		}
		if state.Budget.TurnsLeft == 0 || state.Budget.TokensLeft == 0 {
			out.Rollup.BudgetPressure++
		}
	}
	return out
}
func (s *Server) handleFakFleet(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	if s.listSessions == nil {
		writeErr(w, http.StatusNotFound, "session list is not configured")
		return
	}
	writeJSON(w, http.StatusOK, fleetStatus(s.listSessions(r.Context())))
}
