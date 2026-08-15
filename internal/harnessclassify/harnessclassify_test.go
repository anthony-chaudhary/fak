package harnessclassify

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestExplicitTaskBeatsProjectAndInference(t *testing.T) {
	got, err := Classify(Input{Path: "main.go", Task: "draft contract citations", TaskDomain: "integrated", ProjectDomain: "legal"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Domain != "integrated" || got.Source != "task-declaration" || got.Confidence != 1 {
		t.Fatalf("got=%#v", got)
	}
}

func TestInferenceAutoSelectsOnlyCorroboratedUniqueDomain(t *testing.T) {
	legal, err := Classify(Input{Path: "matter.docx", Task: "draft deposition brief"})
	if err != nil {
		t.Fatal(err)
	}
	if legal.Domain != "legal" || legal.NeedsDecision || legal.Confidence < .8 {
		t.Fatalf("legal=%#v", legal)
	}
	coding, err := Classify(Input{Path: "main.go", Task: "implement compile bug"})
	if err != nil {
		t.Fatal(err)
	}
	if coding.Domain != "coding" || coding.NeedsDecision {
		t.Fatalf("coding=%#v", coding)
	}
}

func TestAmbiguityReturnsOneBoundedDecision(t *testing.T) {
	got, err := Classify(Input{Path: "main.go", Task: "implement contract brief"})
	if err != nil {
		t.Fatal(err)
	}
	if !got.NeedsDecision || got.Domain != "" || got.DecisionRequest == nil {
		t.Fatalf("got=%#v", got)
	}
	if !reflect.DeepEqual(got.DecisionRequest.Choices, []string{"coding", "legal"}) {
		t.Fatalf("choices=%v", got.DecisionRequest.Choices)
	}
	if got.DecisionRequest.Scope == "" {
		t.Fatal("missing choice scope")
	}
}

func TestRememberedChoiceIsScopedExpiringAndExplainable(t *testing.T) {
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	base, _ := Classify(Input{Path: "matter", Task: "review", Now: now})
	choice := Choice{Domain: "legal", Scope: base.ContextKey, ExpiresAt: now.Add(time.Hour), Reason: "operator confirmed matter work"}
	got, err := Classify(Input{Path: "matter", Task: "review", Choice: &choice, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if got.Domain != "legal" || got.Source != "remembered-choice" || len(got.Evidence) != 1 {
		t.Fatalf("got=%#v", got)
	}
	choice.ExpiresAt = now
	if _, err := Classify(Input{Path: "matter", Task: "review", Choice: &choice, Now: now}); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("expired err=%v", err)
	}
	choice.ExpiresAt = now.Add(time.Hour)
	choice.Scope = "ctx:wrong"
	if _, err := Classify(Input{Path: "matter", Task: "review", Choice: &choice, Now: now}); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("scope err=%v", err)
	}
}

func TestLegalNeverReceivesCodingTagCorpus(t *testing.T) {
	corpus := []Input{
		{Path: "brief.docx", Task: "draft deposition citations"},
		{Path: "matter.pdf", Task: "review contract brief"},
		{Path: "evidence.pdf", Task: "prepare deposition matter", Signals: map[string]string{"project": "legal"}},
	}
	for _, in := range corpus {
		got, err := Classify(in)
		if err != nil {
			t.Fatal(err)
		}
		if got.Domain != "legal" || got.NeedsDecision {
			t.Fatalf("input=%#v got=%#v", in, got)
		}
	}
}

type corpusCase struct {
	Path     string `json:"path"`
	Task     string `json:"task"`
	Want     string `json:"want"`
	Decision bool   `json:"decision"`
}

func TestDomainCorpusHasNoFalseAutoSelection(t *testing.T) {
	raw, err := os.ReadFile("testdata/domain-corpus.json")
	if err != nil {
		t.Fatal(err)
	}
	var corpus []corpusCase
	if err := json.Unmarshal(raw, &corpus); err != nil {
		t.Fatal(err)
	}
	falseSelections := 0
	for _, tc := range corpus {
		got, err := Classify(Input{Path: tc.Path, Task: tc.Task})
		if err != nil {
			t.Fatal(err)
		}
		if tc.Decision {
			if !got.NeedsDecision || got.Domain != "" {
				falseSelections++
				t.Errorf("ambiguous case auto-selected: %#v => %#v", tc, got)
			}
		} else if got.NeedsDecision || got.Domain != tc.Want {
			falseSelections++
			t.Errorf("classified %#v => %#v", tc, got)
		}
	}
	if falseSelections != 0 {
		t.Fatalf("false selections=%d/%d", falseSelections, len(corpus))
	}
}

func BenchmarkClassify(b *testing.B) {
	in := Input{Path: "brief.docx", Task: "draft deposition citations"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Classify(in); err != nil {
			b.Fatal(err)
		}
	}
}
