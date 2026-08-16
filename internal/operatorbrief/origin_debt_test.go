package operatorbrief

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestOriginDebtSeparatesTaskAndSessionWitnessDebt(t *testing.T) {
	report := Fold(Inputs{DebtWitnesses: []DebtWitnessRecord{
		{Source: "session", ID: "s-9", Debt: DebtFoundLate, Detail: "shape failure found during review", Evidence: "session-checkpoint"},
		{Source: "task", ID: "t-2", Debt: DebtCaughtAtOrigin, Detail: "artifact refused before handoff", Evidence: "task-witness"},
	}})

	if len(report.OriginDebt) != 1 || report.OriginDebt[0].ID != "t-2" {
		t.Fatalf("origin_debt = %+v, want task t-2", report.OriginDebt)
	}
	if len(report.LateFoundDebt) != 1 || report.LateFoundDebt[0].ID != "s-9" {
		t.Fatalf("late_found_debt = %+v, want session s-9", report.LateFoundDebt)
	}

	rendered := RenderCompact(report)
	for _, want := range []string{
		"origin_debt:", "task t-2: artifact refused before handoff (task-witness)",
		"late_found_debt:", "session s-9: shape failure found during review (session-checkpoint)",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}

	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"origin_debt"`, `"late_found_debt"`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("JSON missing %s: %s", want, b)
		}
	}
}
