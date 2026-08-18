package workflowoutcomestudy

import "testing"

func TestAnalyzeRequiresEquivalentCompletedBlindPairs(t *testing.T) {
	s := Study{Schema: Schema, StudyID: "s", Tasks: []Task{{ID: "T1", Class: "serial", Prompt: "inspect and report", Rubric: []string{"correct"}}}, Results: []ArmResult{
		{TaskID: "T1", Arm: "direct", CandidateID: "A", ProducerID: "producer-a", Model: "sol", Effort: "high", BudgetTokens: 1000, Completed: true, ElapsedMS: 10, ArtifactDigest: "sha256:a", Usage: Usage{InputTokens: 10}},
		{TaskID: "T1", Arm: "workflow", CandidateID: "B", ProducerID: "producer-b", Model: "sol", Effort: "high", BudgetTokens: 1000, Completed: true, ElapsedMS: 20, ArtifactDigest: "sha256:b", Usage: Usage{InputTokens: 20}},
	}, Grades: []Grade{{TaskID: "T1", CandidateID: "A", GraderID: "grader", Correctness: 3, Usefulness: 2, Rationale: "ok"}, {TaskID: "T1", CandidateID: "B", GraderID: "grader", Correctness: 4, Usefulness: 3, Rationale: "better"}}}
	r, err := Analyze(s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.GainClaimReady || r.CompletePairs != 1 || r.BlindGrades != 2 || r.Arms["workflow"].Correctness != 4 {
		t.Fatalf("report=%+v", r)
	}
}

func TestAnalyzeRejectsUnequalEnvelope(t *testing.T) {
	s := Study{Schema: Schema, StudyID: "s", Tasks: []Task{{ID: "T", Prompt: "p", Rubric: []string{"r"}}}, Results: []ArmResult{{TaskID: "T", Arm: "direct", CandidateID: "A", ProducerID: "producer-a", Model: "sol", Effort: "high", BudgetTokens: 1}, {TaskID: "T", Arm: "workflow", CandidateID: "B", ProducerID: "producer-b", Model: "sol", Effort: "xhigh", BudgetTokens: 1}}}
	if _, err := Analyze(s); err == nil {
		t.Fatal("unequal effort accepted")
	}
}
