package livecodebench

import (
	"encoding/json"
	"testing"
)

// TestBuildTestOutputPromptGolden pins the exact predict-the-output prompt shape for
// a problem carrying starter code — a golden test so a drift in the request the CLI
// sends to the gateway is caught here rather than in a live run.
func TestBuildTestOutputPromptGolden(t *testing.T) {
	p := Problem{
		QuestionID:  "q1",
		Scenario:    ScenarioTestOutputPrediction,
		Prompt:      "Read an integer n and print n doubled.",
		StarterCode: "def solve(n):\n    return n * 2",
	}
	got, err := BuildTestOutputPrompt(p, "21")
	if err != nil {
		t.Fatalf("BuildTestOutputPrompt: %v", err)
	}
	want := "Predict the exact output this program produces for the given input. " +
		"Return only the output, with no explanation.\n\n" +
		"## Problem\nRead an integer n and print n doubled.\n\n" +
		"## Code\ndef solve(n):\n    return n * 2\n\n" +
		"## Input\n21\n\n" +
		"## Output\n"
	if got != want {
		t.Fatalf("prompt drift:\n got=%q\nwant=%q", got, want)
	}
}

// TestBuildTestOutputPromptNoStarter drops the ## Code section when the problem has no
// starter code — nothing to show, so the header must not appear.
func TestBuildTestOutputPromptNoStarter(t *testing.T) {
	p := Problem{QuestionID: "q2", Prompt: "Echo the input line."}
	got, err := BuildTestOutputPrompt(p, "hello")
	if err != nil {
		t.Fatalf("BuildTestOutputPrompt: %v", err)
	}
	if want := "## Input\nhello\n\n## Output\n"; got[len(got)-len(want):] != want {
		t.Fatalf("tail drift:\n got=%q\nwant suffix=%q", got, want)
	}
	if containsSection(got, "## Code") {
		t.Fatalf("no-starter prompt must not carry a ## Code section: %q", got)
	}
}

// TestBuildTestOutputPromptRefuses covers the two empty-input refusals.
func TestBuildTestOutputPromptRefuses(t *testing.T) {
	if _, err := BuildTestOutputPrompt(Problem{Prompt: "  "}, "1"); err == nil {
		t.Fatal("expected refusal for empty problem prompt")
	}
	if _, err := BuildTestOutputPrompt(Problem{Prompt: "x"}, "  "); err == nil {
		t.Fatal("expected refusal for empty input")
	}
}

// TestGradeTestOutput checks the normalized comparison: trailing whitespace and line
// endings are ignored, but a genuinely different output is a miss.
func TestGradeTestOutput(t *testing.T) {
	if !GradeTestOutput("42\n", "42") {
		t.Error("trailing newline should not matter")
	}
	if !GradeTestOutput("a \r\nb\n", "a\nb") {
		t.Error("CRLF and trailing spaces should be normalized")
	}
	if GradeTestOutput("42", "43") {
		t.Error("different outputs must not match")
	}
	if GradeTestOutput("a b", "ab") {
		t.Error("internal whitespace is significant")
	}
}

// TestGradeTestOutputProblemMultiInput exercises the all-or-nothing per-problem rule:
// every input right is a pass; one wrong input is a miss for the whole problem.
func TestGradeTestOutputProblemMultiInput(t *testing.T) {
	p := Problem{
		QuestionID: "q1",
		PublicTests: []TestCase{
			{Input: "1", Output: "2"},
			{Input: "3", Output: "6"},
		},
	}
	all, err := GradeTestOutputProblem(p, []string{"2", "6"})
	if err != nil {
		t.Fatalf("GradeTestOutputProblem: %v", err)
	}
	if !all.Correct || len(all.Cases) != 2 {
		t.Fatalf("all-correct problem should be Correct with 2 cases: %+v", all)
	}
	partial, err := GradeTestOutputProblem(p, []string{"2", "7"})
	if err != nil {
		t.Fatalf("GradeTestOutputProblem: %v", err)
	}
	if partial.Correct {
		t.Fatal("a problem with one wrong input must not be Correct")
	}
	if !partial.Cases[0].Correct || partial.Cases[1].Correct {
		t.Fatalf("per-case correctness wrong: %+v", partial.Cases)
	}
}

// TestGradeTestOutputProblemRefuses covers no-test-case and count-mismatch refusals.
func TestGradeTestOutputProblemRefuses(t *testing.T) {
	if _, err := GradeTestOutputProblem(Problem{QuestionID: "q"}, nil); err == nil {
		t.Fatal("expected refusal for a problem with no test cases")
	}
	p := Problem{QuestionID: "q", PublicTests: []TestCase{{Input: "1", Output: "2"}}}
	if _, err := GradeTestOutputProblem(p, []string{"2", "3"}); err == nil {
		t.Fatal("expected refusal for prediction/input count mismatch")
	}
}

// TestBuildTestOutputSummary checks the summary accuracy and the evidence gate.
func TestBuildTestOutputSummary(t *testing.T) {
	problems := []TestOutputProblem{
		{QuestionID: "a", Correct: true},
		{QuestionID: "b", Correct: false},
		{QuestionID: "c", Correct: true},
		{QuestionID: "d", Correct: false},
	}
	s, err := BuildTestOutputSummary("m", problems)
	if err != nil {
		t.Fatalf("BuildTestOutputSummary: %v", err)
	}
	if s.Correct != 2 || s.Accuracy != 0.5 {
		t.Fatalf("accuracy = %v (correct %d), want 0.5 (2)", s.Accuracy, s.Correct)
	}
	if s.Scenario != ScenarioTestOutputPrediction {
		t.Fatalf("scenario = %q", s.Scenario)
	}
	if s.ResultClaimAllowed || s.EvidenceClass != EvidenceLocalUngraded {
		t.Fatalf("evidence gate broken: claim=%v class=%q", s.ResultClaimAllowed, s.EvidenceClass)
	}
	if _, err := BuildTestOutputSummary("m", nil); err == nil {
		t.Fatal("expected refusal for empty problem set")
	}
	if _, err := BuildTestOutputSummary("m", []TestOutputProblem{{QuestionID: "  "}}); err == nil {
		t.Fatal("expected refusal for blank question_id")
	}
}

// TestTestOutputSummaryJSONGolden pins the v1 wire shape of the summary artifact:
// the schema tag fak.livecodebench.testoutput.v1 freezes the field names and order,
// so a rename or type change is caught here rather than by a downstream artifact
// reader parsing an already-written summary.
func TestTestOutputSummaryJSONGolden(t *testing.T) {
	s, err := BuildTestOutputSummary("m1", []TestOutputProblem{
		{
			QuestionID: "q1",
			Cases:      []TestOutputCase{{Input: "1", Expected: "2", Predicted: "2", Correct: true}},
			Correct:    true,
		},
	})
	if err != nil {
		t.Fatalf("BuildTestOutputSummary: %v", err)
	}
	got, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"schema":"fak.livecodebench.testoutput.v1","scenario":"testoutputprediction","model":"m1",` +
		`"problems":[{"question_id":"q1","cases":[{"input":"1","expected":"2","predicted":"2","correct":true}],` +
		`"correct":true}],"correct":1,"accuracy":1,"evidence_class":"local-ungraded","result_claim_allowed":false}`
	if string(got) != want {
		t.Fatalf("summary v1 JSON drift:\n got=%s\nwant=%s", got, want)
	}
}

// containsSection reports whether s contains the given section header on its own line.
func containsSection(s, header string) bool {
	for _, ln := range splitLines(s) {
		if ln == header {
			return true
		}
	}
	return false
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	out = append(out, s[start:])
	return out
}
