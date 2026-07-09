package livecodebench

import "testing"

// execProblem is the fixture used by the code-execution golden tests: a program plus
// one input whose exact output the model must predict.
func execProblem() Problem {
	return Problem{
		QuestionID:  "q1",
		Scenario:    ScenarioCodeExecution,
		StarterCode: "def solve(n):\n    return n * 2",
		PublicTests: []TestCase{{Input: "21", Output: "42"}},
	}
}

// TestBuildCodeExecutionPromptGoldenDirect pins the exact non-CoT (direct) prompt so a
// drift in the request the CLI sends to the gateway is caught here, not in a live run.
func TestBuildCodeExecutionPromptGoldenDirect(t *testing.T) {
	got, err := BuildCodeExecutionPrompt(execProblem(), "21", false)
	if err != nil {
		t.Fatalf("BuildCodeExecutionPrompt: %v", err)
	}
	want := "Determine the exact output this program produces when run on the given input. " +
		"Return only the output, with no explanation.\n\n" +
		"## Program\ndef solve(n):\n    return n * 2\n\n" +
		"## Input\n21\n\n" +
		"## Output\n"
	if got != want {
		t.Fatalf("direct prompt drift:\n got=%q\nwant=%q", got, want)
	}
}

// TestBuildCodeExecutionPromptGoldenCoT pins the exact CoT prompt (the --cot mode): it
// shares the Program/Input body but leads with the reason-step-by-step instruction.
func TestBuildCodeExecutionPromptGoldenCoT(t *testing.T) {
	got, err := BuildCodeExecutionPrompt(execProblem(), "21", true)
	if err != nil {
		t.Fatalf("BuildCodeExecutionPrompt: %v", err)
	}
	want := "Determine the exact output this program produces when run on the given input. " +
		"Reason step by step about what the program does, then give the final output on the last line.\n\n" +
		"## Program\ndef solve(n):\n    return n * 2\n\n" +
		"## Input\n21\n\n" +
		"## Output\n"
	if got != want {
		t.Fatalf("cot prompt drift:\n got=%q\nwant=%q", got, want)
	}
}

// TestBuildCodeExecutionPromptCoTDiffersFromDirect guards the toggle itself: the two
// modes must not render identical prompts, else --cot would be a no-op.
func TestBuildCodeExecutionPromptCoTDiffersFromDirect(t *testing.T) {
	direct, err := BuildCodeExecutionPrompt(execProblem(), "21", false)
	if err != nil {
		t.Fatalf("direct: %v", err)
	}
	cot, err := BuildCodeExecutionPrompt(execProblem(), "21", true)
	if err != nil {
		t.Fatalf("cot: %v", err)
	}
	if direct == cot {
		t.Fatal("--cot must change the prompt; direct and CoT modes rendered identically")
	}
}

// TestBuildCodeExecutionPromptRefuses covers the two empty-input refusals.
func TestBuildCodeExecutionPromptRefuses(t *testing.T) {
	if _, err := BuildCodeExecutionPrompt(Problem{StarterCode: "  "}, "1", false); err == nil {
		t.Fatal("expected refusal for empty program")
	}
	if _, err := BuildCodeExecutionPrompt(Problem{StarterCode: "x"}, "  ", true); err == nil {
		t.Fatal("expected refusal for empty input")
	}
}

// TestGradeCodeExecution checks the normalized comparison: trailing whitespace and
// line endings are ignored, but a genuinely different output is a miss.
func TestGradeCodeExecution(t *testing.T) {
	if !GradeCodeExecution("42\n", "42") {
		t.Error("trailing newline should not matter")
	}
	if GradeCodeExecution("42", "43") {
		t.Error("different output must be a miss")
	}
}

// TestGradeCodeExecutionProblem covers all-or-nothing per-problem scoring and the two
// refusals (no test cases, prediction-count mismatch).
func TestGradeCodeExecutionProblem(t *testing.T) {
	p := Problem{
		QuestionID:  "q2",
		StarterCode: "def f(x):\n    return x",
		PublicTests: []TestCase{{Input: "1", Output: "1"}, {Input: "2", Output: "2"}},
	}
	got, err := GradeCodeExecutionProblem(p, []string{"1", "2"})
	if err != nil {
		t.Fatalf("GradeCodeExecutionProblem: %v", err)
	}
	if !got.Correct {
		t.Fatal("all inputs correct should score the problem correct")
	}
	miss, err := GradeCodeExecutionProblem(p, []string{"1", "9"})
	if err != nil {
		t.Fatalf("GradeCodeExecutionProblem: %v", err)
	}
	if miss.Correct {
		t.Fatal("one wrong input must make the problem a miss (all-or-nothing)")
	}
	if _, err := GradeCodeExecutionProblem(Problem{QuestionID: "q3"}, nil); err == nil {
		t.Fatal("expected refusal for a problem with no test cases")
	}
	if _, err := GradeCodeExecutionProblem(p, []string{"1"}); err == nil {
		t.Fatal("expected refusal for prediction-count mismatch")
	}
}

// TestBuildCodeExecutionSummary checks the summary accuracy, the separately-recorded
// CoT tag, the evidence gate, and the empty/blank refusals.
func TestBuildCodeExecutionSummary(t *testing.T) {
	problems := []CodeExecutionProblem{
		{QuestionID: "a", Correct: true},
		{QuestionID: "b", Correct: false},
	}
	for _, cot := range []bool{false, true} {
		s, err := BuildCodeExecutionSummary("m", cot, problems)
		if err != nil {
			t.Fatalf("BuildCodeExecutionSummary(cot=%v): %v", cot, err)
		}
		if s.CoT != cot {
			t.Fatalf("summary must record its CoT mode: got %v want %v", s.CoT, cot)
		}
		if s.Correct != 1 || s.Accuracy != 0.5 {
			t.Fatalf("accuracy: got correct=%d accuracy=%v want 1, 0.5", s.Correct, s.Accuracy)
		}
		if s.Scenario != ScenarioCodeExecution {
			t.Fatalf("scenario: got %q want %q", s.Scenario, ScenarioCodeExecution)
		}
		if s.EvidenceClass != EvidenceLocalUngraded || s.ResultClaimAllowed {
			t.Fatalf("result must stay evidence-gated: class=%q allowed=%v", s.EvidenceClass, s.ResultClaimAllowed)
		}
	}
	if _, err := BuildCodeExecutionSummary("m", false, nil); err == nil {
		t.Fatal("expected refusal for no graded problems")
	}
	if _, err := BuildCodeExecutionSummary("m", false, []CodeExecutionProblem{{QuestionID: "  "}}); err == nil {
		t.Fatal("expected refusal for blank question_id")
	}
}
