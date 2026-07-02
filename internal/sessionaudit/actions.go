package sessionaudit

import "fmt"

const CompactActionPlanSchema = "fak.session_audit.actions.v1"

type CompactActionPlan struct {
	Schema        string              `json:"schema"`
	SummarySchema string              `json:"summary_schema"`
	Generated     string              `json:"generated"`
	Scope         CompactScope        `json:"scope"`
	Counts        CompactActionCounts `json:"counts"`
	Actions       []CompactAction     `json:"actions"`
	Correctness   string              `json:"correctness"`
}

type CompactActionCounts struct {
	Total  int `json:"total"`
	High   int `json:"high"`
	Medium int `json:"medium"`
}

type CompactAction struct {
	ID              string   `json:"id"`
	Kind            string   `json:"kind"`
	Severity        string   `json:"severity"`
	Target          string   `json:"target"`
	Session         string   `json:"session,omitempty"`
	Namespace       string   `json:"namespace,omitempty"`
	Command         string   `json:"command"`
	WitnessCommands []string `json:"witness_commands,omitempty"`
	Reason          string   `json:"reason"`
	Evidence        string   `json:"evidence"`
}

// BuildCompactActionPlan lowers the compact session-audit recommendations into a
// machine-readable action ledger. It does not mutate sessions or routes; it names
// the next operator action and the witness command that should prove it helped.
func BuildCompactActionPlan(rep CompactReport) CompactActionPlan {
	plan := CompactActionPlan{
		Schema:        CompactActionPlanSchema,
		SummarySchema: rep.Schema,
		Generated:     rep.Generated,
		Scope:         rep.Scope,
		Correctness:   "actions are advisory until a session/context witness proves an effect; no model route or session is changed by this report",
	}
	for _, rec := range rep.Recommendations {
		action, ok := compactActionFromRecommendation(rep, rec)
		if !ok {
			continue
		}
		plan.Actions = append(plan.Actions, action)
		switch action.Severity {
		case "high":
			plan.Counts.High++
		case "medium":
			plan.Counts.Medium++
		}
	}
	plan.Counts.Total = len(plan.Actions)
	return plan
}

func compactActionFromRecommendation(rep CompactReport, rec CompactRecommendation) (CompactAction, bool) {
	switch rec.Kind {
	case "opus_cost_pressure":
		return CompactAction{
			ID:       "keep_fable_default",
			Kind:     rec.Kind,
			Severity: rec.Severity,
			Target:   "model_route:fable_default",
			Command:  rec.Action,
			WitnessCommands: []string{
				"fak session-audit summary --here --json",
				"fak session audit --days 7 --json",
			},
			Reason:   rec.Reason,
			Evidence: rec.Evidence,
		}, true
	case "long_context_pressure":
		action := CompactAction{
			ID:       "checkpoint_reset_top_long_context",
			Kind:     rec.Kind,
			Severity: rec.Severity,
			Target:   "session",
			Command:  rec.Action,
			WitnessCommands: []string{
				"fak vcache context-witness",
				"fak vcache score --json",
				"fak session-audit summary --here --json",
			},
			Reason:   rec.Reason,
			Evidence: rec.Evidence,
		}
		if len(rep.TopLongContext) > 0 {
			top := rep.TopLongContext[0]
			action.Session = top.Session
			action.Namespace = top.Namespace
			action.Target = fmt.Sprintf("session:%s/%s", top.Namespace, top.Session)
		}
		return action, true
	default:
		return CompactAction{}, false
	}
}
