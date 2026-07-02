package sessionaudit

import (
	"fmt"
	"strings"
)

const CompactActionPlanSchema = "fak.session_audit.actions.v1"

type CompactActionPlan struct {
	Schema        string              `json:"schema"`
	SummarySchema string              `json:"summary_schema"`
	Generated     string              `json:"generated"`
	Scope         CompactScope        `json:"scope"`
	Counts        CompactActionCounts `json:"counts"`
	Gate          CompactActionGate   `json:"gate,omitempty,omitzero"`
	Actions       []CompactAction     `json:"actions"`
	Correctness   string              `json:"correctness"`
}

type CompactActionCounts struct {
	Total  int `json:"total"`
	High   int `json:"high"`
	Medium int `json:"medium"`
}

type CompactActionGate struct {
	Threshold string `json:"threshold,omitempty"`
	Verdict   string `json:"verdict,omitempty"`
	Refused   int    `json:"refused,omitempty"`
	Reason    string `json:"reason,omitempty"`
}

func (g CompactActionGate) IsZero() bool {
	return g.Threshold == "" && g.Verdict == "" && g.Refused == 0 && g.Reason == ""
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

// ApplyCompactActionGate stamps a guard verdict onto plan. threshold is "high",
// "medium", or "none"; unknown thresholds return false so CLI callers can emit a
// usage error. A refused gate is evidence that the audited window should block a
// new high-cost/long-context launch until a witness confirms the pressure changed.
func ApplyCompactActionGate(plan CompactActionPlan, threshold string) (CompactActionPlan, bool) {
	thresholdRank, ok := compactSeverityRank(threshold)
	if !ok {
		return plan, false
	}
	threshold = compactSeverityName(threshold)
	gate := CompactActionGate{
		Threshold: threshold,
		Verdict:   "allow",
		Reason:    "no action meets the configured severity threshold",
	}
	if thresholdRank == 0 {
		gate.Reason = "gate disabled"
		plan.Gate = gate
		return plan, true
	}
	for _, action := range plan.Actions {
		rank, ok := compactSeverityRank(action.Severity)
		if ok && rank >= thresholdRank {
			gate.Refused++
		}
	}
	if gate.Refused > 0 {
		gate.Verdict = "refuse"
		gate.Reason = "recent session audit surfaced action(s) at or above the configured severity threshold"
	}
	plan.Gate = gate
	return plan, true
}

func compactSeverityRank(severity string) (int, bool) {
	severity = strings.ToLower(strings.TrimSpace(severity))
	switch severity {
	case "", "none", "off":
		return 0, true
	case "medium":
		return 1, true
	case "high":
		return 2, true
	default:
		return 0, false
	}
}

func CompactActionSeverityRank(severity string) (int, bool) {
	return compactSeverityRank(severity)
}

func compactSeverityName(severity string) string {
	severity = strings.ToLower(strings.TrimSpace(severity))
	switch severity {
	case "", "none", "off":
		return "none"
	default:
		return severity
	}
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
