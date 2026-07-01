package devexmeter

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGatePassesOnlyOnStrictDrop(t *testing.T) {
	issue := Issue{Number: 2166, Labels: []string{"dev-ex", "friction/retry-after-refusal"}}
	rows := []Row{
		{Issue: 2166, Class: "retry-after-refusal", Window: "before", Value: 12, Samples: 2, Source: "baseline"},
		{Issue: 2166, Class: "retry-after-refusal", Window: "after", Value: 7, Samples: 2, Source: "post-merge"},
	}
	got := GateIssue(issue, rows)
	if !got.OK || got.Verdict != VerdictPass {
		t.Fatalf("GateIssue = %+v, want PASS", got)
	}
	if got.Before == nil || got.After == nil || *got.Before != 12 || *got.After != 7 {
		t.Fatalf("before/after = %v/%v, want 12/7", got.Before, got.After)
	}
	if got.Delta == nil || *got.Delta != -5 {
		t.Fatalf("delta = %v, want -5", got.Delta)
	}
}

func TestGateHoldsFlatMeterNotYet(t *testing.T) {
	issue := Issue{Number: 99, Labels: []string{"track/F-integration-tooling", "dev-ex", "friction:re-read-waste"}}
	rows := []Row{
		{Issue: 99, Class: "re-read-waste", Window: "before", Value: 4},
		{Issue: 99, Class: "re-read-waste", Window: "after", Value: 4},
	}
	got := GateIssue(issue, rows)
	if got.OK || got.Verdict != VerdictNotYet {
		t.Fatalf("GateIssue = %+v, want NOT_YET", got)
	}
	if len(got.MissingWitness) != 1 || !strings.Contains(got.MissingWitness[0], "strict drop") {
		t.Fatalf("missing witness = %v, want strict-drop reason", got.MissingWitness)
	}
}

func TestGateRequiresBeforeAndAfterWindows(t *testing.T) {
	issue := Issue{Number: 100, Labels: []string{"dev-ex"}, Body: "Friction-Class: livelock"}
	got := GateIssue(issue, []Row{{Issue: 100, Class: "livelock", Window: "before", Value: 2}})
	if got.OK || got.Verdict != VerdictNotYet {
		t.Fatalf("GateIssue = %+v, want NOT_YET", got)
	}
	if len(got.MissingWitness) != 1 || !strings.Contains(got.MissingWitness[0], "after meter window") {
		t.Fatalf("missing witness = %v, want after-window gap", got.MissingWitness)
	}
}

func TestGateSkipsWhenIssueHasNoFrictionClaim(t *testing.T) {
	noDevEx := GateIssue(Issue{Number: 1, Labels: []string{"bug"}}, nil)
	if !noDevEx.OK || noDevEx.Verdict != VerdictSkip {
		t.Fatalf("non-dev-ex gate = %+v, want SKIP/OK", noDevEx)
	}
	noClass := GateIssue(Issue{Number: 2, Labels: []string{"dev-ex"}}, nil)
	if !noClass.OK || noClass.Verdict != VerdictSkip {
		t.Fatalf("dev-ex without class = %+v, want SKIP/OK", noClass)
	}
}

func TestFoldRowsWeightedBySamples(t *testing.T) {
	rows := []Row{
		{Issue: 12, Class: "retry", Window: "before", Value: 10, Samples: 3, Source: "a"},
		{Issue: 12, Class: "retry", Window: "before", Value: 4, Samples: 1, Source: "b"},
		{Issue: 12, Class: "retry", Window: "after", Value: 5, Samples: 2, Source: "c"},
		{Issue: 13, Class: "retry", Window: "after", Value: 99},
		{Class: "retry", Window: "after", Value: 7, Samples: 2, Source: "class-wide"},
	}
	fold := FoldRows(rows, 12, "retry")
	if fold.Before.Rows != 2 || fold.Before.Samples != 4 || fold.Before.Value != 8.5 {
		t.Fatalf("before = %+v, want rows=2 samples=4 value=8.5", fold.Before)
	}
	if fold.After.Rows != 2 || fold.After.Samples != 4 || fold.After.Value != 6 {
		t.Fatalf("after = %+v, want rows=2 samples=4 value=6", fold.After)
	}
	if strings.Join(fold.After.Sources, ",") != "c,class-wide" {
		t.Fatalf("sources = %v, want sorted source names", fold.After.Sources)
	}
}

func TestParseLedgerValidatesRows(t *testing.T) {
	data := []byte(`{"schema":"fak.devexmeter.row.v1","issue":1,"class":"Retry_After_Refusal","window":"before","value":9,"samples":3}

{"issue":1,"class":"retry-after-refusal","window":"after","value":5}
`)
	rows, err := ParseLedger(data)
	if err != nil {
		t.Fatalf("ParseLedger: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(rows))
	}
	if _, err := ParseLedger([]byte(`{"class":"x","window":"during","value":1}` + "\n")); err == nil {
		t.Fatal("ParseLedger accepted an unknown window")
	}
	if _, err := ParseLedger([]byte(`{"class":"x","window":"before","value":-1}` + "\n")); err == nil {
		t.Fatal("ParseLedger accepted a negative value")
	}
}

func TestParseIssueAcceptsGitHubLabelObjects(t *testing.T) {
	raw := []byte(`{"number":2166,"body":"body","labels":[{"name":"dev-ex"},{"name":"friction-class:retry-after-refusal"}]}`)
	issue, err := ParseIssue(raw)
	if err != nil {
		t.Fatalf("ParseIssue: %v", err)
	}
	class, ok := ClassFromIssue(issue)
	if !ok || class != "retry-after-refusal" {
		t.Fatalf("ClassFromIssue = %q/%v, want retry-after-refusal/true", class, ok)
	}
	encoded, err := json.Marshal(GateIssue(issue, []Row{
		{Issue: 2166, Class: class, Window: "before", Value: 3},
		{Issue: 2166, Class: class, Window: "after", Value: 2},
	}))
	if err != nil || !strings.Contains(string(encoded), GateSchema) {
		t.Fatalf("marshal gate result = %q, %v", encoded, err)
	}
}
