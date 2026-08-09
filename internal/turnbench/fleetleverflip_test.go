package turnbench

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

type marginalFixtureRung struct {
	name string
	deny map[string]bool
}

func (r marginalFixtureRung) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if r.deny[c.Tool] {
		return abi.Verdict{Kind: abi.VerdictDeny, By: r.name, Reason: abi.ReasonPolicyBlock}
	}
	return abi.Verdict{By: r.name}
}
func (marginalFixtureRung) Caps() []abi.Capability { return nil }

func fixtureMarginalInput(trace *Trace) DivHistInput {
	trace.SliceID = "fixture"
	return DivHistInput{
		Trace: trace,
		Arms: []PolicyArm{{Name: "reference", Policy: adjudicator.Policy{Allow: map[string]bool{
			"first_harm": true, "late_harm": true, "read": true,
		}}}},
		RefName: "reference",
	}
}

func TestRunFleetLeverFlipFrontierGatesHarmfulSinkCredit(t *testing.T) {
	trace := &Trace{Calls: []Call{
		{Tool: "first_harm", Args: json.RawMessage(`{}`), Meta: map[string]string{HarmfulSinkMetaKey: "true"}},
		{Tool: "late_harm", Args: json.RawMessage(`{}`), Meta: map[string]string{HarmfulSinkMetaKey: "true"}},
	}}
	chain := []abi.Adjudicator{
		marginalFixtureRung{name: "first", deny: map[string]bool{"first_harm": true}},
		marginalFixtureRung{name: "late", deny: map[string]bool{"late_harm": true}},
	}

	report, err := runFleetLeverFlipWithChain(context.Background(), []DivHistInput{fixtureMarginalInput(trace)}, chain)
	if err != nil {
		t.Fatal(err)
	}
	if report.ModelCallsSpent != 0 {
		t.Fatalf("model_calls_spent=%d, want replay-only zero", report.ModelCallsSpent)
	}
	if report.Schema != "fak.turnbench.rung-marginal.v1" {
		t.Fatalf("schema=%q", report.Schema)
	}
	rows := map[string]RungMarginalRow{}
	for _, row := range report.Rows {
		rows[row.Rung] = row
	}
	first := rows["first"]
	if first.InjectionsAdmitted != 1 {
		t.Fatalf("first injections_admitted=%d, want 1; post-frontier late sink must receive no credit", first.InjectionsAdmitted)
	}
	if first.DeniesDelta != -1 || first.ExactCells != 0 || first.BoundedCells != 1 || !first.NeedsLiveRevalidation {
		t.Fatalf("first row=%+v, want delta=-1 bounded=1 live-revalidation", first)
	}
	late := rows["late"]
	if late.InjectionsAdmitted != 1 || late.DeniesDelta != -1 || late.BoundedCells != 1 {
		t.Fatalf("late row=%+v, want one frontier sink admitted", late)
	}
}

func TestRunFleetLeverFlipDeterministicAndExactCells(t *testing.T) {
	trace := &Trace{Calls: []Call{{Tool: "read", Args: json.RawMessage(`{}`)}}}
	chain := []abi.Adjudicator{marginalFixtureRung{name: "unused", deny: map[string]bool{"never": true}}}
	corpus := []DivHistInput{fixtureMarginalInput(trace)}
	a, err := runFleetLeverFlipWithChain(context.Background(), corpus, chain)
	if err != nil {
		t.Fatal(err)
	}
	b, err := runFleetLeverFlipWithChain(context.Background(), corpus, chain)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Rows) != 1 || len(b.Rows) != 1 || a.Rows[0] != b.Rows[0] {
		t.Fatalf("non-deterministic rows: a=%+v b=%+v", a.Rows, b.Rows)
	}
	if a.Rows[0].ExactCells != 1 || a.Rows[0].BoundedCells != 0 || a.Rows[0].NeedsLiveRevalidation {
		t.Fatalf("exact row=%+v", a.Rows[0])
	}
}
