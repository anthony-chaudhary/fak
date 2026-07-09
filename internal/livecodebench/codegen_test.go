package livecodebench

import (
	"fmt"
	"math"
	"testing"
)

// fence wraps code in a python-tagged closed fence so ExtractCode returns it.
func fence(code string) string {
	return "reasoning...\n```python\n" + code + "\n```\n"
}

// TestScoreCodegenEndToEnd drives the codegen scenario end to end: two problems,
// n=5 completions each, a deterministic grader that passes when the extracted
// code contains "return ok". It asserts the per-problem AND summary pass@1 /
// pass@5 match the unbiased estimator, that a NoCode sample is scored a miss
// (not handed to the grader), and that n / temperature are recorded.
func TestScoreCodegenEndToEnd(t *testing.T) {
	problems := []Problem{
		{QuestionID: "q1", Scenario: ScenarioCodeGeneration},
		{QuestionID: "q2", Scenario: ScenarioCodeGeneration},
	}
	// q1: 3 of 5 pass (two prose-only NoCode samples are automatic misses).
	// q2: 0 of 5 pass.
	completions := [][]string{
		{
			fence("return ok"),
			fence("return ok"),
			fence("return ok"),
			"no fence here, prose only",   // NoCode -> miss, never graded
			"still prose, still no fence", // NoCode -> miss, never graded
		},
		{
			fence("return wrong"),
			fence("return wrong"),
			fence("return wrong"),
			fence("return wrong"),
			fence("return wrong"),
		},
	}
	graderCalls := 0
	grade := func(_ Problem, code string) (bool, error) {
		graderCalls++
		return code == "return ok", nil
	}

	rep, err := ScoreCodegen(CodegenConfig{N: 5, Temperature: CodegenDefaultTemperature}, problems, completions, grade)
	if err != nil {
		t.Fatalf("ScoreCodegen: %v", err)
	}

	// The two NoCode samples must never reach the grader: 5 + 3 = 8 graded.
	if graderCalls != 8 {
		t.Fatalf("grader calls = %d, want 8 (NoCode samples must not be graded)", graderCalls)
	}
	if rep.N != 5 || rep.Temperature != CodegenDefaultTemperature {
		t.Fatalf("recorded n/temperature = %d/%v, want 5/%v", rep.N, rep.Temperature, CodegenDefaultTemperature)
	}
	if rep.Scenario != ScenarioCodeGeneration {
		t.Fatalf("scenario = %q, want %q", rep.Scenario, ScenarioCodeGeneration)
	}
	if len(rep.Problems) != 2 {
		t.Fatalf("per-problem rows = %d, want 2", len(rep.Problems))
	}

	// q1: n=5, c=3, extracted=3. pass@1 = 3/5; pass@5 = 1 (n-c=2 < 5).
	q1 := rep.Problems[0]
	if q1.Samples != 5 || q1.Correct != 3 || q1.Extracted != 3 {
		t.Fatalf("q1 tally = n%d c%d ex%d, want n5 c3 ex3", q1.Samples, q1.Correct, q1.Extracted)
	}
	wantP1, _ := PassAtK(5, 3, 1)
	wantP5, _ := PassAtK(5, 3, 5)
	if !closeEnough(q1.Pass1, wantP1) || !closeEnough(q1.Pass5, wantP5) {
		t.Fatalf("q1 pass@1/pass@5 = %v/%v, want %v/%v", q1.Pass1, q1.Pass5, wantP1, wantP5)
	}

	// q2: n=5, c=0. pass@1 = pass@5 = 0.
	q2 := rep.Problems[1]
	if q2.Correct != 0 || q2.Pass1 != 0 || q2.Pass5 != 0 {
		t.Fatalf("q2 = c%d p1%v p5%v, want all-zero", q2.Correct, q2.Pass1, q2.Pass5)
	}

	// Summary is the mean of the per-problem rates.
	wantSumP1 := (wantP1 + 0) / 2
	wantSumP5 := (wantP5 + 0) / 2
	if !closeEnough(rep.Summary.Pass1, wantSumP1) || !closeEnough(rep.Summary.Pass5, wantSumP5) {
		t.Fatalf("summary pass@1/pass@5 = %v/%v, want %v/%v", rep.Summary.Pass1, rep.Summary.Pass5, wantSumP1, wantSumP5)
	}
	if rep.Summary.Problems != 2 || rep.Summary.Generations != 10 || rep.Summary.Graded != 8 {
		t.Fatalf("summary counts = p%d g%d graded%d, want p2 g10 graded8", rep.Summary.Problems, rep.Summary.Generations, rep.Summary.Graded)
	}
}

// TestScoreCodegenMergesStarter checks the starter-code merge path: a bare
// completion that omits the starter signature is completed with it before
// grading, so the grader sees the whole program.
func TestScoreCodegenMergesStarter(t *testing.T) {
	problems := []Problem{{QuestionID: "q1", StarterCode: "def solve(x):"}}
	completions := [][]string{{
		fence("    return x"),
		fence("    return x"),
		fence("    return x"),
		fence("    return x"),
		fence("    return x"),
	}}
	grade := func(_ Problem, code string) (bool, error) {
		// The grader must receive the starter prepended.
		want := "def solve(x):\n    return x"
		return code == want, nil
	}
	rep, err := ScoreCodegen(CodegenConfig{N: 5}, problems, completions, grade)
	if err != nil {
		t.Fatalf("ScoreCodegen: %v", err)
	}
	if rep.Problems[0].Correct != 5 {
		t.Fatalf("starter-merged correct = %d, want 5 (grader must see the merged program)", rep.Problems[0].Correct)
	}
}

func TestScoreCodegenErrors(t *testing.T) {
	ok := []Problem{{QuestionID: "q1"}}
	good := func(_ Problem, _ string) (bool, error) { return true, nil }
	five := [][]string{{fence("a"), fence("a"), fence("a"), fence("a"), fence("a")}}

	if _, err := ScoreCodegen(CodegenConfig{N: 5}, ok, five, nil); err == nil {
		t.Fatal("nil grader must error")
	}
	if _, err := ScoreCodegen(CodegenConfig{N: 5}, nil, nil, good); err == nil {
		t.Fatal("no problems must error")
	}
	if _, err := ScoreCodegen(CodegenConfig{N: 5}, ok, [][]string{}, good); err == nil {
		t.Fatal("mismatched completion/problem counts must error")
	}
	// Fewer than five samples: pass@5 undefined.
	if _, err := ScoreCodegen(CodegenConfig{N: 5}, ok, [][]string{{fence("a"), fence("a")}}, good); err == nil {
		t.Fatal("n<5 must error (pass@5 undefined)")
	}
	// Grader error propagates.
	boom := func(_ Problem, _ string) (bool, error) { return false, fmt.Errorf("sandbox down") }
	if _, err := ScoreCodegen(CodegenConfig{N: 5}, ok, five, boom); err == nil {
		t.Fatal("grader error must propagate")
	}
}

func closeEnough(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
