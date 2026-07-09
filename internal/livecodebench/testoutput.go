package livecodebench

import (
	"fmt"
	"strings"
)

// testoutput.go is the test-output-prediction scenario (epic #2085, issue #2098):
// the model is shown a problem and one program input and must predict the exact
// program output, which is graded against the problem's expected output. Three
// pieces are pure here so they are unit-tested without a model call:
// BuildTestOutputPrompt (the predict-the-output request the CLI sends to the gateway,
// one per input), GradeTestOutputProblem (per-problem correctness over a problem's
// inputs, all-or-nothing like upstream), and BuildTestOutputSummary (the run's
// summary accuracy). Like the rest of the package the result is evidence-gated —
// ResultClaimAllowed stays false; the same predictions must be graded by the
// official lcb_runner before any accuracy is claimable.

// TestOutputSummarySchema tags a test-output-prediction summary artifact.
const TestOutputSummarySchema = "fak.livecodebench.testoutput.v1"

// BuildTestOutputPrompt renders the test-output-prediction request for ONE input:
// the problem, its code (starter_code, when the problem carries it), and the specific
// input, asking the model for the exact program output and nothing else. This is the
// upstream output-prediction prompt shape (problem + code + input -> output) rendered
// purely so it is unit-tested without a model call. It refuses an empty problem prompt
// or an empty input — there is nothing to predict from, or nothing to predict for,
// otherwise.
func BuildTestOutputPrompt(p Problem, input string) (string, error) {
	if strings.TrimSpace(p.Prompt) == "" {
		return "", fmt.Errorf("livecodebench test-output: problem prompt is empty (nothing to predict from)")
	}
	if strings.TrimSpace(input) == "" {
		return "", fmt.Errorf("livecodebench test-output: input is empty (nothing to predict for)")
	}
	var b strings.Builder
	b.WriteString("Predict the exact output this program produces for the given input. ")
	b.WriteString("Return only the output, with no explanation.\n\n")
	b.WriteString("## Problem\n")
	b.WriteString(strings.TrimSpace(p.Prompt))
	if strings.TrimSpace(p.StarterCode) != "" {
		b.WriteString("\n\n## Code\n")
		b.WriteString(strings.TrimSpace(p.StarterCode))
	}
	b.WriteString("\n\n## Input\n")
	b.WriteString(strings.TrimSpace(input))
	b.WriteString("\n\n## Output\n")
	return b.String(), nil
}

// normalizeTestOutput canonicalizes a program output for comparison: it normalizes
// line endings, trims trailing whitespace on each line, and drops a trailing blank
// line, mirroring the upstream output-prediction grader which compares stripped
// stdout. Internal whitespace is preserved (spacing inside a line is significant).
func normalizeTestOutput(s string) string {
	lines := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
	for i := range lines {
		lines[i] = strings.TrimRight(lines[i], " \t")
	}
	return strings.TrimRight(strings.Join(lines, "\n"), "\n")
}

// GradeTestOutput compares one predicted output against the expected output under
// normalizeTestOutput (line-ending- and trailing-whitespace-insensitive), returning
// true on an exact normalized match.
func GradeTestOutput(predicted, expected string) bool {
	return normalizeTestOutput(predicted) == normalizeTestOutput(expected)
}

// TestOutputCase is the graded result for one problem input: the input, its expected
// output, the model's predicted output, and whether they matched under GradeTestOutput.
type TestOutputCase struct {
	Input     string `json:"input"`
	Expected  string `json:"expected"`
	Predicted string `json:"predicted"`
	Correct   bool   `json:"correct"`
}

// TestOutputProblem is the per-problem prediction result: every input's predicted vs
// expected output and whether the model got ALL of them right. A problem is Correct
// only when every one of its inputs is predicted correctly — a partially-right
// multi-input problem is a miss, matching upstream all-or-nothing scoring.
type TestOutputProblem struct {
	QuestionID string           `json:"question_id"`
	Cases      []TestOutputCase `json:"cases"`
	Correct    bool             `json:"correct"`
}

// TestOutputSummary is the machine-readable result of a test-output-prediction run:
// the per-problem correctness rows and the summary accuracy (fraction of problems all
// of whose inputs were predicted correctly), plus the evidence gate.
type TestOutputSummary struct {
	Schema             string              `json:"schema"`
	Scenario           Scenario            `json:"scenario"`
	Model              string              `json:"model,omitempty"`
	Problems           []TestOutputProblem `json:"problems"`
	Correct            int                 `json:"correct"`
	Accuracy           float64             `json:"accuracy"`
	EvidenceClass      string              `json:"evidence_class"`
	ResultClaimAllowed bool                `json:"result_claim_allowed"`
}

// GradeTestOutputProblem grades one problem's predictions: predicted[i] is the model's
// output for the problem's i-th public test-case input, graded against that case's
// expected Output. It refuses when the problem carries no test cases (nothing to
// predict) or when the prediction count does not match the input count (a prediction
// is required for every input — a short slice would silently score missing inputs as
// passes). The returned problem is Correct only when every input matched.
func GradeTestOutputProblem(p Problem, predicted []string) (TestOutputProblem, error) {
	tests := p.PublicTests
	if len(tests) == 0 {
		return TestOutputProblem{}, fmt.Errorf("livecodebench test-output: problem %q has no test cases to predict", p.QuestionID)
	}
	if len(predicted) != len(tests) {
		return TestOutputProblem{}, fmt.Errorf("livecodebench test-output: problem %q has %d inputs but %d predictions", p.QuestionID, len(tests), len(predicted))
	}
	cases := make([]TestOutputCase, len(tests))
	all := true
	for i, tc := range tests {
		ok := GradeTestOutput(predicted[i], tc.Output)
		cases[i] = TestOutputCase{Input: tc.Input, Expected: tc.Output, Predicted: predicted[i], Correct: ok}
		if !ok {
			all = false
		}
	}
	return TestOutputProblem{QuestionID: p.QuestionID, Cases: cases, Correct: all}, nil
}

// BuildTestOutputSummary aggregates graded per-problem predictions into a run summary:
// the accuracy is the fraction of problems all of whose inputs the model predicted
// correctly. It refuses an empty set (no problems graded) or a problem with a blank
// question_id. Evidence is gated — ResultClaimAllowed stays false until the official
// lcb_runner grades the same predictions.
func BuildTestOutputSummary(model string, problems []TestOutputProblem) (TestOutputSummary, error) {
	if len(problems) == 0 {
		return TestOutputSummary{}, fmt.Errorf("livecodebench test-output: no graded problems to summarize")
	}
	correct := 0
	for i, p := range problems {
		if strings.TrimSpace(p.QuestionID) == "" {
			return TestOutputSummary{}, fmt.Errorf("livecodebench test-output: problem %d question_id is required", i)
		}
		if p.Correct {
			correct++
		}
	}
	return TestOutputSummary{
		Schema:             TestOutputSummarySchema,
		Scenario:           ScenarioTestOutputPrediction,
		Model:              strings.TrimSpace(model),
		Problems:           problems,
		Correct:            correct,
		Accuracy:           float64(correct) / float64(len(problems)),
		EvidenceClass:      EvidenceLocalUngraded,
		ResultClaimAllowed: false,
	}, nil
}
