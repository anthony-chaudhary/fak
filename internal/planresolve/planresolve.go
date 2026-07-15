package planresolve

import (
	"context"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

// Disposition is the closed plan-content decision vocabulary.
type Disposition string

const (
	AutoApprove Disposition = "AUTO_APPROVE"
	AutoRefuse  Disposition = "AUTO_REFUSE"
	Escalate    Disposition = "ESCALATE"
)

// Reason is the closed operator-gate/refusal vocabulary.
type Reason string

const (
	ReasonApproved                Reason = "PLAN_APPROVED"
	ReasonTreeCollision           Reason = "PLAN_TREE_COLLISION"
	ReasonWrongDirection          Reason = "PLAN_WRONG_DIRECTION"
	ReasonCoreLockRequired        Reason = "PLAN_CORE_LOCK_REQUIRED"
	ReasonDoneUnverifiable        Reason = "PLAN_DONE_UNVERIFIABLE"
	ReasonIrreversibleUnwitnessed Reason = "PLAN_IRREVERSIBLE_UNWITNESSED"
	ReasonOracleFailure           Reason = "PLAN_ORACLE_FAILURE"
)

// OracleResult is the typed output of one read-only plan-content oracle.
type OracleResult struct {
	OK      bool   `json:"ok"`
	Reason  Reason `json:"reason,omitempty"`
	Witness string `json:"witness,omitempty"`
}

// OracleSet contains the read-only tree, direction, and done-criterion seams.
type OracleSet interface {
	TreeDisjoint(context.Context, []string) (OracleResult, error)
	DirectionAllowed(context.Context, []string) (OracleResult, error)
	DoneVerifiable(context.Context, string) (OracleResult, error)
}

// StepResult records structural reversibility evidence for one plan step.
type StepResult struct {
	Step      operatorquestion.PlanStep      `json:"step"`
	Class     adjudicator.ReversibilityClass `json:"class"`
	Witnessed bool                           `json:"witnessed"`
}

// Verdict is the plan decision plus bounded evidence and appeal route.
type Verdict struct {
	Disposition Disposition              `json:"disposition"`
	Reason      Reason                   `json:"reason"`
	Triage      choicetriage.Disposition `json:"triage,omitempty"`
	Witnesses   []string                 `json:"witnesses,omitempty"`
	Steps       []StepResult             `json:"steps,omitempty"`
	Appeal      string                   `json:"appeal,omitempty"`
}

// Resolve adjudicates plan content from oracle output, never the plan's self-assessment.
func Resolve(ctx context.Context, q operatorquestion.OperatorQuestion, oracles OracleSet) (Verdict, error) {
	if q.Kind != operatorquestion.PlanApproval || q.Plan == nil {
		return Verdict{}, fmt.Errorf("planresolve: PLAN_APPROVAL with structured plan is required")
	}
	if oracles == nil {
		return Verdict{}, fmt.Errorf("planresolve: oracle set is required")
	}
	checks := []func(context.Context) (OracleResult, error){
		func(ctx context.Context) (OracleResult, error) { return oracles.TreeDisjoint(ctx, q.Plan.FileTree) },
		func(ctx context.Context) (OracleResult, error) { return oracles.DirectionAllowed(ctx, q.Plan.FileTree) },
		func(ctx context.Context) (OracleResult, error) {
			return oracles.DoneVerifiable(ctx, q.Plan.DoneCriterion)
		},
	}
	verdict := Verdict{Disposition: AutoApprove, Reason: ReasonApproved}
	for _, check := range checks {
		result, err := check(ctx)
		if err != nil {
			return Verdict{Disposition: AutoRefuse, Reason: ReasonOracleFailure, Appeal: appeal(ReasonOracleFailure)}, err
		}
		if result.Witness != "" {
			verdict.Witnesses = append(verdict.Witnesses, result.Witness)
		}
		if !result.OK {
			reason := result.Reason
			if reason == "" {
				reason = ReasonOracleFailure
			}
			verdict.Disposition, verdict.Reason, verdict.Appeal = AutoRefuse, reason, appeal(reason)
			return verdict, nil
		}
	}
	for _, step := range q.Plan.Steps {
		envelope := adjudicator.ClassifyReversibility(step.Tool, step.Args)
		witnessed := strings.TrimSpace(step.Witness) != ""
		verdict.Steps = append(verdict.Steps, StepResult{Step: step, Class: envelope.Class, Witnessed: witnessed})
		if envelope.Class != adjudicator.ReversibilityReversible && !witnessed {
			verdict.Disposition = Escalate
			verdict.Reason = ReasonIrreversibleUnwitnessed
			verdict.Triage = choicetriage.HumanResidual
			verdict.Appeal = appeal(verdict.Reason)
			return verdict, nil
		}
	}
	return verdict, nil
}

func appeal(reason Reason) string {
	return "fak complain --reason " + string(reason)
}
