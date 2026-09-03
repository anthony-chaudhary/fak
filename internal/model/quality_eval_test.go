package model

import (
	"errors"
	"math"
	"strings"
	"testing"
	"time"
)

func TestQualityEval_ValidateCodeOutput(t *testing.T) {
	tests := []struct {
		name          string
		requiredFunc  string
		output        string
		wantPass      bool
		wantReasonSub string
	}{
		{
			name:         "valid python function",
			requiredFunc: "is_prime",
			output:       "def is_prime(n):\n    if n <= 1:\n        return False\n    return True\n",
			wantPass:     true,
		},
		{
			name:         "valid python function in markdown fences",
			requiredFunc: "fibonacci",
			output:       "```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    return 1\n```",
			wantPass:     true,
		},
		{
			name:          "missing function definition",
			requiredFunc:  "calculate_total",
			output:        "def compute_sum(a, b):\n    return a + b\n",
			wantPass:      false,
			wantReasonSub: "missing function definition",
		},
		{
			name:          "missing return statement",
			requiredFunc:  "print_info",
			output:        "def print_info(msg):\n    print(msg)\n",
			wantPass:      false,
			wantReasonSub: "missing return statement",
		},
		{
			name:          "invalid indentation body at base indent",
			requiredFunc:  "do_work",
			output:        "def do_work(x):\nreturn x * 2\n",
			wantPass:      false,
			wantReasonSub: "invalid indentation",
		},
		{
			name:          "empty required function name",
			requiredFunc:  "",
			output:        "def foo():\n    return 1\n",
			wantPass:      false,
			wantReasonSub: "missing required function name",
		},
		{
			name:          "empty function body",
			requiredFunc:  "do_nothing",
			output:        "def do_nothing():\n",
			wantPass:      false,
			wantReasonSub: "missing return statement",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pass, reason := ValidateCodeOutput(tc.requiredFunc, tc.output)
			if pass != tc.wantPass {
				t.Fatalf("ValidateCodeOutput(%q) pass = %v, want %v (reason: %q)", tc.requiredFunc, pass, tc.wantPass, reason)
			}
			if tc.wantReasonSub != "" && !strings.Contains(strings.ToLower(reason), strings.ToLower(tc.wantReasonSub)) {
				t.Errorf("ValidateCodeOutput reason = %q, want substring %q", reason, tc.wantReasonSub)
			}
		})
	}
}

func TestQualityEval_ValidateJSONOutput(t *testing.T) {
	tests := []struct {
		name           string
		requiredFields []string
		output         string
		wantPass       bool
		wantReasonSub  string
	}{
		{
			name:           "valid json object",
			requiredFields: []string{"name", "age"},
			output:         `{"name": "Alice", "age": 30}`,
			wantPass:       true,
		},
		{
			name:           "valid json in code fences",
			requiredFields: []string{"name", "age"},
			output:         "```json\n{\n  \"name\": \"Bob\",\n  \"age\": 25\n}\n```",
			wantPass:       true,
		},
		{
			name:           "valid json surrounded by prose",
			requiredFields: []string{"result"},
			output:         "Result is here:\n{\"result\": \"success\"}\nEnd of transmission.",
			wantPass:       true,
		},
		{
			name:           "invalid json syntax",
			requiredFields: []string{"name"},
			output:         "{name: invalid_json}",
			wantPass:       false,
			wantReasonSub:  "invalid json",
		},
		{
			name:           "missing required field",
			requiredFields: []string{"name", "role"},
			output:         `{"name": "Alice"}`,
			wantPass:       false,
			wantReasonSub:  "missing required field",
		},
		{
			name:           "empty string field",
			requiredFields: []string{"name"},
			output:         `{"name": ""}`,
			wantPass:       false,
			wantReasonSub:  "field name is empty",
		},
		{
			name:           "null field",
			requiredFields: []string{"name"},
			output:         `{"name": null}`,
			wantPass:       false,
			wantReasonSub:  "null",
		},
		{
			name:           "empty output",
			requiredFields: []string{"name"},
			output:         "   ",
			wantPass:       false,
			wantReasonSub:  "empty output",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pass, reason := ValidateJSONOutput(tc.requiredFields, tc.output)
			if pass != tc.wantPass {
				t.Fatalf("ValidateJSONOutput() pass = %v, want %v (reason: %q)", pass, tc.wantPass, reason)
			}
			if tc.wantReasonSub != "" && !strings.Contains(strings.ToLower(reason), strings.ToLower(tc.wantReasonSub)) {
				t.Errorf("ValidateJSONOutput reason = %q, want substring %q", reason, tc.wantReasonSub)
			}
		})
	}
}

func TestQualityEval_ValidateLogicOutput(t *testing.T) {
	tests := []struct {
		name          string
		expected      string
		output        string
		wantPass      bool
		wantReasonSub string
	}{
		{
			name:     "exact answer present",
			expected: "32",
			output:   "The next number in the sequence is 32.",
			wantPass: true,
		},
		{
			name:     "case insensitive match",
			expected: "Paris",
			output:   "The capital of France is paris.",
			wantPass: true,
		},
		{
			name:          "incorrect answer",
			expected:      "32",
			output:        "The answer is 64.",
			wantPass:      false,
			wantReasonSub: "not found",
		},
		{
			name:          "empty output",
			expected:      "32",
			output:        "",
			wantPass:      false,
			wantReasonSub: "empty",
		},
		{
			name:          "empty expected answer",
			expected:      "",
			output:        "32",
			wantPass:      false,
			wantReasonSub: "expected answer is empty",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pass, reason := ValidateLogicOutput(tc.expected, tc.output)
			if pass != tc.wantPass {
				t.Fatalf("ValidateLogicOutput() pass = %v, want %v (reason: %q)", pass, tc.wantPass, reason)
			}
			if tc.wantReasonSub != "" && !strings.Contains(strings.ToLower(reason), strings.ToLower(tc.wantReasonSub)) {
				t.Errorf("ValidateLogicOutput reason = %q, want substring %q", reason, tc.wantReasonSub)
			}
		})
	}
}

func TestQualityEval_DefaultQualitySuite(t *testing.T) {
	suite := DefaultQualitySuite()
	if len(suite) < 3 {
		t.Fatalf("DefaultQualitySuite returned %d tasks, want at least 3", len(suite))
	}

	kindsFound := make(map[TaskKind]bool)
	for _, task := range suite {
		if task.ID == "" {
			t.Errorf("task has empty ID")
		}
		if task.Prompt == "" {
			t.Errorf("task %s has empty prompt", task.ID)
		}
		if task.Description == "" {
			t.Errorf("task %s has empty description", task.ID)
		}
		if task.Validate == nil {
			t.Errorf("task %s has nil Validate function", task.ID)
		}
		kindsFound[task.Kind] = true
	}

	if !kindsFound[TaskKindCode] {
		t.Errorf("missing code task in suite")
	}
	if !kindsFound[TaskKindJSON] {
		t.Errorf("missing json task in suite")
	}
	if !kindsFound[TaskKindLogic] {
		t.Errorf("missing logic task in suite")
	}
}

func TestQualityEval_RunQualityEvaluation_ValidOutputs(t *testing.T) {
	suite := DefaultQualitySuite()

	mockGenerate := func(prompt string) (string, int, int, error) {
		time.Sleep(1 * time.Millisecond)
		switch {
		case strings.Contains(prompt, "fibonacci"):
			return "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n", 40, 15, nil
		case strings.Contains(prompt, "Alice"):
			return `{"name": "Alice", "age": 30}`, 20, 8, nil
		case strings.Contains(prompt, "sequence"):
			return "The answer is 32.", 10, 4, nil
		default:
			return "ok", 5, 2, nil
		}
	}

	report, err := RunQualityEvaluation(suite, mockGenerate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.TotalTasks != len(suite) {
		t.Errorf("TotalTasks = %d, want %d", report.TotalTasks, len(suite))
	}
	if report.PassedTasks != len(suite) {
		t.Errorf("PassedTasks = %d, want %d", report.PassedTasks, len(suite))
	}
	if report.PassRate != 100.0 {
		t.Errorf("PassRate = %f, want 100.0", report.PassRate)
	}
	if report.TotalTokens != 70 {
		t.Errorf("TotalTokens = %d, want 70", report.TotalTokens)
	}
	if report.DraftAccepted != 27 {
		t.Errorf("DraftAccepted = %d, want 27", report.DraftAccepted)
	}
	if len(report.Results) != len(suite) {
		t.Errorf("len(Results) = %d, want %d", len(report.Results), len(suite))
	}

	for _, res := range report.Results {
		if !res.Pass {
			t.Errorf("task %s failed unexpectedly: %s", res.TaskID, res.Reason)
		}
		if res.TokensPerSec <= 0 {
			t.Errorf("task %s TokensPerSec = %f, want > 0", res.TaskID, res.TokensPerSec)
		}
		if res.Duration <= 0 {
			t.Errorf("task %s Duration = %v, want > 0", res.TaskID, res.Duration)
		}
	}
}

func TestQualityEval_RunQualityEvaluation_FailingOutputs(t *testing.T) {
	suite := DefaultQualitySuite()

	mockFailingGenerate := func(prompt string) (string, int, int, error) {
		switch {
		case strings.Contains(prompt, "fibonacci"):
			return "def fibonacci(n):\npass\n", 10, 2, nil
		case strings.Contains(prompt, "Alice"):
			return "This is not json.", 15, 5, nil
		case strings.Contains(prompt, "sequence"):
			return "The answer is 32.", 10, 3, nil
		default:
			return "", 0, 0, nil
		}
	}

	report, err := RunQualityEvaluation(suite, mockFailingGenerate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if report.TotalTasks != 3 {
		t.Errorf("TotalTasks = %d, want 3", report.TotalTasks)
	}
	if report.PassedTasks != 1 {
		t.Errorf("PassedTasks = %d, want 1", report.PassedTasks)
	}
	if report.PassRate >= 100.0 {
		t.Errorf("PassRate = %f, want non-100%% (< 100.0)", report.PassRate)
	}
	expectedPassRate := (1.0 / 3.0) * 100.0
	if math.Abs(report.PassRate-expectedPassRate) > 1e-4 {
		t.Errorf("PassRate = %f, want approx %f", report.PassRate, expectedPassRate)
	}

	for _, res := range report.Results {
		if !res.Pass && res.Reason == "" {
			t.Errorf("task %s failed but reason is empty", res.TaskID)
		}
	}
}

func TestQualityEval_RunQualityEvaluation_GeneratorError(t *testing.T) {
	suite := DefaultQualitySuite()
	genErr := errors.New("inference generation failed")

	mockErrGenerate := func(prompt string) (string, int, int, error) {
		return "", 0, 0, genErr
	}

	report, err := RunQualityEvaluation(suite, mockErrGenerate)
	if err == nil {
		t.Fatalf("expected error from RunQualityEvaluation, got nil")
	}
	if report != nil {
		t.Errorf("expected nil report on generator error, got %+v", report)
	}
	if !strings.Contains(err.Error(), "inference generation failed") {
		t.Errorf("error = %v, want substring %q", err, "inference generation failed")
	}
}

func TestQualityEval_RunQualityEvaluation_NilGenerator(t *testing.T) {
	_, err := RunQualityEvaluation(nil, nil)
	if err == nil {
		t.Fatalf("expected error for nil generator, got nil")
	}
}

func TestQualityEval_RunQualityEvaluation_EmptyTasks(t *testing.T) {
	report, err := RunQualityEvaluation([]QualityTask{}, func(prompt string) (string, int, int, error) {
		return "ok", 1, 1, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if report.TotalTasks != 0 || report.PassedTasks != 0 || report.PassRate != 0.0 {
		t.Errorf("unexpected report for empty tasks: %+v", report)
	}
}
