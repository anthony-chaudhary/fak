package resume

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

const ResumeAnchorSchema = "fak-resume-anchor/1"

type ResumeAnchor struct {
	Schema      string                  `json:"schema"`
	Session     string                  `json:"session"`
	ObjectiveID string                  `json:"objective_id,omitempty"`
	Objective   string                  `json:"objective,omitempty"`
	Curve       *trajctl.ObjectiveCurve `json:"curve,omitempty"`
	Plan        []trajctl.PlanPhase     `json:"plan,omitempty"`
	Present     bool                    `json:"present"`
}

// BuildResumeAnchor selects the newest objective carrying witnessed score rows
// from this session. It is pure: callers read the ledger and stamp launch time.
func BuildResumeAnchor(session string, st trajctl.State) ResumeAnchor {
	out := ResumeAnchor{Schema: ResumeAnchorSchema, Session: session}
	var objectiveID string
	for i := len(st.Scores) - 1; i >= 0; i-- {
		row := st.Scores[i]
		if row.SessionID == session && row.Witness != trajctl.W0 {
			objectiveID = row.ObjectiveID
			break
		}
	}
	if objectiveID == "" {
		return out
	}
	obj, ok := st.Objectives[objectiveID]
	if !ok {
		return out
	}
	curve, ok := st.CurveFor(objectiveID)
	if !ok {
		return out
	}
	out.ObjectiveID, out.Objective, out.Plan = obj.ID, obj.Statement, append([]trajctl.PlanPhase(nil), obj.Plan...)
	out.Curve, out.Present = &curve, true
	return out
}

func (a ResumeAnchor) Prompt() string {
	if !a.Present || a.Curve == nil {
		return ""
	}
	phases := make([]string, 0, len(a.Plan))
	for _, p := range a.Plan {
		phases = append(phases, strings.TrimSpace(p.ID+" "+p.Title))
	}
	return fmt.Sprintf("fresh resume anchor (independently re-read; do not inherit stale transcript direction):\nobjective [%s]: %s\nwitnessed curve: %s latest=%.2f delta=%+.2f � %s\nplan state: %s\nre-read this anchor, checkpoint against it, then continue.",
		a.ObjectiveID, a.Objective, a.Curve.Signal, a.Curve.Latest, a.Curve.Delta, a.Curve.Detail, strings.Join(phases, "; "))
}
