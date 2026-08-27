package issuefanout

import (
	"reflect"
	"testing"
)

func TestFileLiveRefusesMissingProjectDenominator(t *testing.T) {
	p, err := Build(Input{Title: "x", Leaf: "x", SpineRef: "abc", Max: MinFanout})
	if err != nil {
		t.Fatal(err)
	}
	_, err = FileLive(p, nil, LiveOptions{Runner: func([]string) (string, string, bool) { t.Fatal("runner called"); return "", "", false }})
	if err == nil {
		t.Fatal("missing denominator accepted")
	}
}

func TestFileLivePreflightsWholeBatchBeforeFirstCreate(t *testing.T) {
	p, err := Build(Input{
		Title:              "x",
		Leaf:               "x",
		SpineRef:           "abc",
		ParentIssue:        36,
		ParentBaseline:     100,
		CompletionStandard: "demo",
		Max:                MinFanout,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Corrupt only the last candidate. A streaming contract check would create
	// the first two issues before discovering this failure.
	p.Candidates[len(p.Candidates)-1].RequiredModelTier = ""
	calls := 0
	got, err := FileLive(p, nil, LiveOptions{Runner: func([]string) (string, string, bool) {
		calls++
		return "https://github.com/o/r/issues/1", "", true
	}})
	if err == nil {
		t.Fatal("corrupt final candidate accepted")
	}
	if calls != 0 {
		t.Fatalf("runner calls = %d, want zero before whole-batch validation", calls)
	}
	if !reflect.DeepEqual(got, LiveResult{}) {
		t.Fatalf("refusal leaked a partial live result: %+v", got)
	}
}
