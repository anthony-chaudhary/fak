package ctxplan

// PlanStepDecision is the reconciliation result for one stable StepID. Outcome
// reuses the closed ObjectiveOutcome vocabulary so every non-preserved result
// has the same refusal semantics as ReconcileObjective.
type PlanStepDecision struct {
	StepID      string           `json:"step_id"`
	Outcome     ObjectiveOutcome `json:"outcome"`
	BeforeIndex int              `json:"before_index"`
	AfterIndex  int              `json:"after_index"`
}

// PlanReconciliation records the per-step result of comparing two plan pins.
// Decisions is keyed by StepID. AllMatch is true only when both pins verify and
// every ID, text, and position is preserved.
type PlanReconciliation struct {
	Decisions map[string]PlanStepDecision `json:"decisions"`
	AllMatch  bool                        `json:"all_match"`
}

// ReconcilePlan compares stable step identities across two ordered plan pins.
// A missing step is dropped, a new or ambiguous step requires the user, and a
// text or position change is drifted. Duplicate/empty IDs are ambiguous and
// therefore never pass silently.
func ReconcilePlan(before, after PlanPin) PlanReconciliation {
	result := PlanReconciliation{Decisions: make(map[string]PlanStepDecision)}
	beforeByID, beforeDup := indexPlanSteps(before.Steps)
	afterByID, afterDup := indexPlanSteps(after.Steps)

	for id, old := range beforeByID {
		decision := PlanStepDecision{StepID: id, BeforeIndex: old.index, AfterIndex: -1}
		current, exists := afterByID[id]
		switch {
		case id == "" || beforeDup[id] || afterDup[id]:
			decision.Outcome = ObjectiveQueryUser
		case !exists:
			decision.Outcome = ObjectiveDropped
		case old.step.Text != current.step.Text || old.index != current.index:
			decision.Outcome = ObjectiveDrifted
		default:
			decision.Outcome = ObjectivePreserved
		}
		if exists {
			decision.AfterIndex = current.index
		}
		result.Decisions[id] = decision
	}
	for id, current := range afterByID {
		if _, exists := beforeByID[id]; exists {
			continue
		}
		result.Decisions[id] = PlanStepDecision{
			StepID: id, Outcome: ObjectiveQueryUser, BeforeIndex: -1, AfterIndex: current.index,
		}
	}

	result.AllMatch = before.Verify() && after.Verify() && len(before.Steps) == len(after.Steps)
	if result.AllMatch {
		for _, decision := range result.Decisions {
			if decision.Outcome != ObjectivePreserved {
				result.AllMatch = false
				break
			}
		}
	}
	return result
}

type indexedPlanStep struct {
	step  StepPin
	index int
}

func indexPlanSteps(steps []StepPin) (map[string]indexedPlanStep, map[string]bool) {
	indexed := make(map[string]indexedPlanStep, len(steps))
	duplicates := make(map[string]bool)
	for i, step := range steps {
		if _, exists := indexed[step.StepID]; exists {
			duplicates[step.StepID] = true
			continue
		}
		indexed[step.StepID] = indexedPlanStep{step: step, index: i}
	}
	return indexed, duplicates
}

// PlanLogEntry is one observed before/after plan transition.
type PlanLogEntry struct {
	Before PlanPin `json:"before"`
	After  PlanPin `json:"after"`
}

// PlanLog is an append-only sequence of plan transitions.
type PlanLog struct {
	Entries []PlanLogEntry `json:"entries"`
}

// Replay independently reconciles every recorded transition and reports
// whether all of them preserved the complete plan.
func (l PlanLog) Replay() ([]PlanReconciliation, bool) {
	results := make([]PlanReconciliation, 0, len(l.Entries))
	allMatch := true
	for _, entry := range l.Entries {
		result := ReconcilePlan(entry.Before, entry.After)
		results = append(results, result)
		if !result.AllMatch {
			allMatch = false
		}
	}
	return results, allMatch
}
