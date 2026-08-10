package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func corpusFixture(t *testing.T) []ghCorpusIssue {
	t.Helper()
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	xs := make([]ghCorpusIssue, 60)
	for i := range xs {
		n := 1000 + i
		xs[i] = ghCorpusIssue{Number: n, Title: fmt.Sprintf("issue %d", n), Body: fmt.Sprintf("References #1000 from %d", n), State: "OPEN", Labels: []ghLabel{{Name: "class:dev"}}, CreatedAt: base.Add(time.Duration(i) * time.Hour), UpdatedAt: base.Add(time.Duration(i) * time.Hour), URL: "https://example.invalid/issues/x"}
	}
	return xs
}

func TestCorpusFreezeVerifyAndBlindGrade(t *testing.T) {
	d := t.TempDir()
	in := filepath.Join(d, "in.json")
	pub := filepath.Join(d, "public.json")
	ans := filepath.Join(d, "answers.json")
	rep := filepath.Join(d, "report.json")
	b, _ := json.Marshal(corpusFixture(t))
	if err := os.WriteFile(in, b, 0644); err != nil {
		t.Fatal(err)
	}
	if err := freezeCorpus(in, pub, ans, rep, "test-source"); err != nil {
		t.Fatal(err)
	}
	if err := verifyCorpusArtifacts(pub, ans, rep); err != nil {
		t.Fatal(err)
	}
	ab, _ := os.ReadFile(ans)
	var gold answerBundle
	if err := json.Unmarshal(ab, &gold); err != nil {
		t.Fatal(err)
	}
	for _, answer := range gold.Answers {
		seen := map[string]bool{}
		for _, label := range answer.Labels {
			if seen[label] {
				t.Fatalf("duplicate label %q in %s", label, answer.ID)
			}
			seen[label] = true
		}
	}
	s := submission{Schema: "fak-microcontext-submission/1", CorpusSHA256: gold.CorpusSHA256, Answers: gold.Answers, Aggregates: gold.Aggregates}
	sb, _ := json.Marshal(s)
	sp := filepath.Join(d, "submission.json")
	gp := filepath.Join(d, "grade.json")
	_ = os.WriteFile(sp, sb, 0644)
	if err := gradeSubmissionFiles(ans, sp, gp); err != nil {
		t.Fatal(err)
	}
	pb, _ := os.ReadFile(pub)
	if !bytesContainAny(pb, []string{`"state"`, `"labels"`, `"closed_at"`}) {
		t.Fatal("source fields missing from public corpus")
	}
	if bytesContainAny(pb, []string{`"references"`, `"duplicate_targets"`, `"scope_contradictions"`, `"aggregates"`}) {
		t.Fatal("derived answer fields leaked into public corpus")
	}
}

func TestCorpusGraderRejectsQualityMiss(t *testing.T) {
	gold := answerBundle{Schema: answerSchema, CorpusSHA256: "x", Answers: []issueAnswer{{ID: "issue-1", Split: "test", State: "OPEN", Labels: []string{"bug"}}}, Aggregates: aggregateAnswers{StateCounts: map[string]int{"OPEN": 1}, LabelCounts: map[string]int{"bug": 1}}}
	got := submission{Schema: "fak-microcontext-submission/1", CorpusSHA256: "x", Answers: []issueAnswer{{ID: "issue-1", Split: "test", State: "CLOSED"}}, Aggregates: gold.Aggregates}
	r := gradeSubmission(gold, got)
	if r.QualityPass || r.FalsePositiveFacts == 0 || r.FalseNegativeFacts == 0 {
		t.Fatalf("accepted miss: %+v", r)
	}
}

func TestCorpusSplitHasThreeStablePartitions(t *testing.T) {
	seen := map[string]bool{}
	for n := 1; n < 500; n++ {
		s := corpusSplit(n)
		seen[s] = true
		if s != corpusSplit(n) {
			t.Fatal("unstable split")
		}
	}
	for _, s := range []string{"train", "tune", "test"} {
		if !seen[s] {
			t.Fatalf("missing %s", s)
		}
	}
}
