package planresolve

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

type oracleStub struct {
	tree, direction, done OracleResult
	calls                 []string
}

func (o *oracleStub) TreeDisjoint(context.Context, []string) (OracleResult, error) {
	o.calls = append(o.calls, "tree")
	return o.tree, nil
}
func (o *oracleStub) DirectionAllowed(context.Context, []string) (OracleResult, error) {
	o.calls = append(o.calls, "direction")
	return o.direction, nil
}
func (o *oracleStub) DoneVerifiable(context.Context, string) (OracleResult, error) {
	o.calls = append(o.calls, "done")
	return o.done, nil
}

func planQuestion(step operatorquestion.PlanStep) operatorquestion.OperatorQuestion {
	return operatorquestion.OperatorQuestion{
		Kind: operatorquestion.PlanApproval,
		Plan: &operatorquestion.Plan{
			FileTree:      []string{"internal/planresolve/**"},
			Steps:         []operatorquestion.PlanStep{step},
			DoneCriterion: "dos verify plans/operator.md phase-1",
		},
	}
}

func cleanOracles() *oracleStub {
	return &oracleStub{
		tree:      OracleResult{OK: true, Witness: "dos arbitrate: DISJOINT"},
		direction: OracleResult{OK: true, Witness: "architest: direction allowed"},
		done:      OracleResult{OK: true, Witness: "dos verify: criterion parseable"},
	}
}

func TestTreeCollisionAutoRefuses(t *testing.T) {
	o := cleanOracles()
	o.tree = OracleResult{OK: false, Reason: ReasonTreeCollision, Witness: "lease L7 intersects internal/planresolve/**"}
	got, err := Resolve(context.Background(), planQuestion(operatorquestion.PlanStep{Text: "edit leaf", Tool: "Read"}), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != AutoRefuse || got.Reason != ReasonTreeCollision || got.Appeal == "" {
		t.Fatalf("got %+v", got)
	}
	if !reflect.DeepEqual(o.calls, []string{"tree"}) {
		t.Fatalf("fail-fast calls=%v", o.calls)
	}
}

func TestReversibleDisjointVerifiedPlanAutoApproves(t *testing.T) {
	o := cleanOracles()
	got, err := Resolve(context.Background(), planQuestion(operatorquestion.PlanStep{
		Text: "inspect policy", Tool: "Read", Args: map[string]any{"path": "policy.json"},
	}), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != AutoApprove || got.Reason != ReasonApproved || len(got.Witnesses) != 3 {
		t.Fatalf("got %+v", got)
	}
	if !reflect.DeepEqual(o.calls, []string{"tree", "direction", "done"}) {
		t.Fatalf("oracle calls=%v", o.calls)
	}
}

func TestIrreversibleUnwitnessedStepEscalates(t *testing.T) {
	o := cleanOracles()
	got, err := Resolve(context.Background(), planQuestion(operatorquestion.PlanStep{
		Text: "publish release", Tool: "Bash", Args: map[string]any{"command": "git push origin main"},
	}), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != Escalate || got.Reason != ReasonIrreversibleUnwitnessed || got.Triage != choicetriage.HumanResidual {
		t.Fatalf("got %+v", got)
	}
	if len(got.Steps) != 1 || got.Steps[0].Witnessed {
		t.Fatalf("missing structural step evidence: %+v", got.Steps)
	}
}

func TestIrreversibleWitnessedStepCanApprove(t *testing.T) {
	o := cleanOracles()
	got, err := Resolve(context.Background(), planQuestion(operatorquestion.PlanStep{
		Text: "publish release", Tool: "Bash", Args: map[string]any{"command": "git push origin main"}, Witness: "operator-approved release ticket #1",
	}), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.Disposition != AutoApprove {
		t.Fatalf("got %+v", got)
	}
}

func TestWrongDirectionAndUnverifiableDoneRefuseWithClosedReasons(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*oracleStub)
		want   Reason
	}{
		{"direction", func(o *oracleStub) { o.direction = OracleResult{Reason: ReasonCoreLockRequired} }, ReasonCoreLockRequired},
		{"done", func(o *oracleStub) { o.done = OracleResult{Reason: ReasonDoneUnverifiable} }, ReasonDoneUnverifiable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := cleanOracles()
			tc.mutate(o)
			got, err := Resolve(context.Background(), planQuestion(operatorquestion.PlanStep{Tool: "Read"}), o)
			if err != nil {
				t.Fatal(err)
			}
			if got.Disposition != AutoRefuse || got.Reason != tc.want {
				t.Fatalf("got %+v", got)
			}
		})
	}
}
