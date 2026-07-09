package livecodebench

import (
	"fmt"
	"strings"
)

// codeexecution.go is the code-execution scenario (epic #2085, issue #2099): the
// model is shown a program and one input and must reason about the program's
// behavior to predict the exact output of running it. An optional chain-of-thought
// mode (the upstream --cot_code_execution flag) toggles whether the request asks the
// model to reason step by step before answering or to answer directly. Three pieces
// are pure here so they are unit-tested without a model call: BuildCodeExecutionPrompt
// (the predict-the-output request the CLI sends to the gateway, with the CoT toggle),
// GradeCodeExecutionProblem (per-problem correctness over a problem's inputs,
// all-or-nothing like upstream), and BuildCodeExecutionSummary (the run's summary
// accuracy, tagged with the CoT mode so a CoT and a non-CoT run are recorded
// separately). Like the rest of the package the result is evidence-gated —
// ResultClaimAllowed stays false; the same predictions must be graded by the official
// lcb_runner before any accuracy is claimable.

// CodeExecutionSummarySchema tags a code-execution summary artifact.
const CodeExecutionSummarySchema = "fak.livecodebench.codeexecution.v1"

// The two leading instructions the CoT toggle chooses between. Direct mode asks for
// the output and nothing else; CoT mode asks the model to reason step by step first
// (the upstream code-execution chain-of-thought prompt shape). They share the same
// Code/Input body so a golden test pins both modes against a single problem.
const (
	codeExecDirectInstruction = "Determine the exact output this program produces when run on the given input. " +
		"Return only the output, with no explanation."
	codeExecCoTInstruction = "Determine the exact output this program produces when run on the given input. " +
		"Reason step by step about what the program does, then give the final output on the last line."
)

// BuildCodeExecutionPrompt renders the code-execution request for ONE input: the
// program and the specific input, asking the model for the exact output of executing
// the program on that input. When cot is true the request asks the model to reason
// step by step before answering (the upstream chain-of-thought mode); when false it
// asks for the output directly. This is the upstream code-execution prompt shape
// (program + input -> output) rendered purely so it is unit-tested without a model
// call. It refuses an empty program or an empty input — there is nothing to execute,
// or nothing to execute it on, otherwise.
func BuildCodeExecutionPrompt(p Problem, input string, cot bool) (string, error) {
	if strings.TrimSpace(p.StarterCode) == "" {
		return "", fmt.Errorf("livecodebench code-execution: program is empty (nothing to execute)")
	}
	if strings.TrimSpace(input) == "" {
		return "", fmt.Errorf("livecodebench code-execution: input is empty (nothing to execute on)")
	}
	instruction := codeExecDirectInstruction
	if cot {
		instruction = codeExecCoTInstruction
	}
	var b strings.Builder
	b.WriteString(instruction)
	b.WriteString("\n\n## Program\n")
	b.WriteString(strings.TrimSpace(p.StarterCode))
	b.WriteString("\n\n## Input\n")
	b.WriteString(strings.TrimSpace(input))
	b.WriteString("\n\n## Output\n")
	return b.String(), nil
}

// GradeCodeExecution compares one predicted output against the expected output under
// normalizeTestOutput (line-ending- and trailing-whitespace-insensitive), returning
// true on an exact normalized match.
func GradeCodeExecution(predicted, expected string) bool {
	return normalizeTestOutput(predicted) == normalizeTestOutput(expected)
}

// CodeExecutionCase is the graded result for one problem input: the input, its
// expected output, the model's predicted output, and whether they matched under
// GradeCodeExecution.
type CodeExecutionCase struct {
	Input     string `json:"input"`
	Expected  string `json:"expected"`
	Predicted string `json:"predicted"`
	Correct   bool   `json:"correct"`
}

// CodeExecutionProblem is the per-problem execution result: every input's predicted vs
// expected output and whether the model got ALL of them right. A problem is Correct
// only when every one of its inputs is predicted correctly — a partially-right
// multi-input problem is a miss, matching upstream all-or-nothing scoring.
type CodeExecutionProblem struct {
	QuestionID string              `json:"question_id"`
	Cases      []CodeExecutionCase `json:"cases"`
	Correct    bool                `json:"correct"`
}

// CodeExecutionSummary is the machine-readable result of a code-execution run: the
// per-problem correctness rows and the summary accuracy (fraction of problems all of
// whose inputs were predicted correctly), tagged with the CoT mode the run used so a
// CoT and a non-CoT run are recorded separately, plus the evidence gate.
type CodeExecutionSummary struct {
	Schema             string                 `json:"schema"`
	Scenario           Scenario               `json:"scenario"`
	Model              string                 `json:"model,omitempty"`
	CoT                bool                   `json:"cot"`
	Problems           []CodeExecutionProblem `json:"problems"`
	Correct            int                    `json:"correct"`
	Accuracy           float64                `json:"accuracy"`
	EvidenceClass      string                 `json:"evidence_class"`
	ResultClaimAllowed bool                   `json:"result_claim_allowed"`
}

// GradeCodeExecutionProblem grades one problem's predictions: predicted[i] is the
// model's output for the problem's i-th public test-case input, graded against that
// case's expected Output. It refuses when the problem carries no test cases (nothing
// to execute) or when the prediction count does not match the input count (a
// prediction is required for every input — a short slice would silently score missing
// inputs as passes). The returned problem is Correct only when every input matched.
func GradeCodeExecutionProblem(p Problem, predicted []string) (CodeExecutionProblem, error) {
	tests := p.PublicTests
	if len(tests) == 0 {
		return CodeExecutionProblem{}, fmt.Errorf("livecodebench code-execution: problem %q has no test cases to execute", p.QuestionID)
	}
	if len(predicted) != len(tests) {
		return CodeExecutionProblem{}, fmt.Errorf("livecodebench code-execution: problem %q has %d inputs but %d predictions", p.QuestionID, len(tests), len(predicted))
	}
	cases := make([]CodeExecutionCase, len(tests))
	all := true
	for i, tc := range tests {
		ok := GradeCodeExecution(predicted[i], tc.Output)
		cases[i] = CodeExecutionCase{Input: tc.Input, Expected: tc.Output, Predicted: predicted[i], Correct: ok}
		if !ok {
			all = false
		}
	}
	return CodeExecutionProblem{QuestionID: p.QuestionID, Cases: cases, Correct: all}, nil
}

// BuildCodeExecutionSummary aggregates graded per-problem executions into a run
// summary: the accuracy is the fraction of problems all of whose inputs the model
// predicted correctly. The cot flag is recorded verbatim so a CoT run and a non-CoT
// run over the same problems produce two separately-tagged summaries. It refuses an
// empty set (no problems graded) or a problem with a blank question_id. Evidence is
// gated — ResultClaimAllowed stays false until the official lcb_runner grades the
// same predictions.
func BuildCodeExecutionSummary(model string, cot bool, problems []CodeExecutionProblem) (CodeExecutionSummary, error) {
	if len(problems) == 0 {
		return CodeExecutionSummary{}, fmt.Errorf("livecodebench code-execution: no graded problems to summarize")
	}
	correct := 0
	for i, p := range problems {
		if strings.TrimSpace(p.QuestionID) == "" {
			return CodeExecutionSummary{}, fmt.Errorf("livecodebench code-execution: problem %d question_id is required", i)
		}
		if p.Correct {
			correct++
		}
	}
	return CodeExecutionSummary{
		Schema:             CodeExecutionSummarySchema,
		Scenario:           ScenarioCodeExecution,
		Model:              strings.TrimSpace(model),
		CoT:                cot,
		Problems:           problems,
		Correct:            correct,
		Accuracy:           float64(correct) / float64(len(problems)),
		EvidenceClass:      EvidenceLocalUngraded,
		ResultClaimAllowed: false,
	}, nil
}
