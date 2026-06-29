package loopdrive

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseRenderRoundTrip(t *testing.T) {
	src := []byte(`---
loop: issue-1176
witness: commit-audit # checked by the loop gate
budget: { max_iters: 20, max_tokens: 4000 }
---
# Objective
Ship the GOAL.md loop spec.

# Plan
- [x] Add a parser
- [ ] Wire the driver

# Scratch / last-refusal
- NOT_YET previous refusal
`)
	spec, err := Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Loop != "issue-1176" || spec.Witness != "commit-audit" || spec.Budget.MaxIters != 20 || spec.Budget.MaxTokens != 4000 {
		t.Fatalf("frontmatter = %+v", spec)
	}
	if spec.Objective != "Ship the GOAL.md loop spec." {
		t.Fatalf("objective = %q", spec.Objective)
	}
	idx, item, ok := spec.NextUnchecked()
	if !ok || idx != 1 || item.Text != "Wire the driver" {
		t.Fatalf("next unchecked = (%d,%+v,%v)", idx, item, ok)
	}
	again, err := Parse(spec.Render())
	if err != nil {
		t.Fatalf("parse rendered spec: %v\n%s", err, spec.Render())
	}
	if !reflect.DeepEqual(again, spec) {
		t.Fatalf("render round-trip drifted\n got: %+v\nwant: %+v", again, spec)
	}
}

func TestTemplateIsParseable(t *testing.T) {
	spec, err := Parse(Template("nightly/dispatch"))
	if err != nil {
		t.Fatal(err)
	}
	if spec.Loop != "nightly/dispatch" || spec.Witness != "commit-audit" || spec.Budget.MaxIters != 20 {
		t.Fatalf("template spec = %+v", spec)
	}
	if _, _, ok := spec.NextUnchecked(); !ok {
		t.Fatalf("template must include one unchecked first step: %+v", spec.Plan)
	}
}

func TestParseRejectsMissingRequiredFields(t *testing.T) {
	cases := []string{
		"# Objective\nmissing frontmatter\n",
		"---\nwitness: none\n---\n# Objective\nx\n",
		"---\nloop: l\n---\n# Objective\nx\n",
		"---\nloop: l\nwitness: none\n---\n# Plan\n- [ ] x\n",
	}
	for _, src := range cases {
		if _, err := Parse([]byte(src)); err == nil {
			t.Fatalf("Parse(%q) returned nil error", src)
		}
	}
}

func TestParseBudgetRejectsBadMaxIters(t *testing.T) {
	_, err := Parse([]byte(`---
loop: l
witness: none
budget: { max_iters: nope }
---
# Objective
x
`))
	if err == nil || !strings.Contains(err.Error(), "max_iters") {
		t.Fatalf("err = %v, want max_iters parse failure", err)
	}
}

func TestDecideStopsOnWitnessedExitGate(t *testing.T) {
	d := Decide(PolicyInput{Witnessed: true, Iterations: 99, MaxIters: 1})
	if d.Action != ActionStopWitnessed || d.Reason != ReasonWitnessedDone {
		t.Fatalf("decision = %+v, want witnessed stop", d)
	}
}

func TestDecideStopsOnBudgets(t *testing.T) {
	cases := []PolicyInput{
		{Iterations: 2, MaxIters: 2},
		{TokensUsed: 100, MaxTokens: 100},
		{NowUnixNano: 200, DeadlineUnixNano: 200},
	}
	for _, in := range cases {
		if d := Decide(in); d.Action != ActionStopBudget || d.Reason != ReasonBudgetSpent {
			t.Fatalf("Decide(%+v) = %+v, want budget stop", in, d)
		}
	}
}

func TestDecideRunsWhenBudgetRemains(t *testing.T) {
	d := Decide(PolicyInput{Iterations: 1, MaxIters: 2, TokensUsed: 99, MaxTokens: 100, NowUnixNano: 99, DeadlineUnixNano: 100})
	if d.Action != ActionRunTurn {
		t.Fatalf("decision = %+v, want run turn", d)
	}
}

func TestNextWorkDoesNotTreatChecklistAsWitness(t *testing.T) {
	spec := Spec{Plan: []PlanItem{{Checked: true, Text: "done"}}}
	idx, item, unchecked := spec.NextWork()
	if unchecked || idx != 1 || !strings.Contains(item.Text, "witness") {
		t.Fatalf("NextWork = (%d,%+v,%v), want witness-settling pseudo item", idx, item, unchecked)
	}
}
