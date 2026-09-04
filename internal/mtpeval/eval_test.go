package mtpeval

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestValidatePythonCode(t *testing.T) {
	validCode := `def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
`
	if ok, reason := ValidatePythonCode(validCode); !ok {
		t.Fatalf("expected valid python code to pass, failed with: %s", reason)
	}

	validWithFence := "```python\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    return -1\n```"
	if ok, reason := ValidatePythonCode(validWithFence); !ok {
		t.Fatalf("expected code with fences to pass, failed with: %s", reason)
	}

	validWithPreambleAndFences := "Here is the code:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\nHope that helps!"
	if ok, reason := ValidatePythonCode(validWithPreambleAndFences); !ok {
		t.Fatalf("expected code with preamble and fences to pass, failed with: %s", reason)
	}

	validWithEscapedQuotes := `def greet(name):
    msg = "He said \"hello\""
    return msg
`
	if ok, reason := ValidatePythonCode(validWithEscapedQuotes); !ok {
		t.Fatalf("expected code with escaped quotes to pass, failed with: %s", reason)
	}

	// Delimiter mismatch
	unbalanced := `def broken(x):
    return (x + [1, 2)
`
	if ok, _ := ValidatePythonCode(unbalanced); ok {
		t.Fatal("expected unbalanced delimiter to fail")
	}

	// Missing colon
	noColon := `def broken(x)
    return x
`
	if ok, _ := ValidatePythonCode(noColon); ok {
		t.Fatal("expected missing def colon to fail")
	}

	// No indentation in body
	noIndent := `def broken(x):
return x
`
	if ok, _ := ValidatePythonCode(noIndent); ok {
		t.Fatal("expected unindented body to fail")
	}
}

func TestValidateJSONSchema(t *testing.T) {
	schema := &JSONSchema{
		RequiredKeys: []string{"name", "score", "active", "tags"},
		KeyTypes: map[string]string{
			"name":   "string",
			"score":  "number",
			"active": "boolean",
			"tags":   "array",
		},
	}

	validJSON := `{"name": "test_spec", "score": 95.5, "active": true, "tags": ["mtp", "q38"]}`
	if ok, reason := ValidateJSONSchema(validJSON, schema); !ok {
		t.Fatalf("expected valid JSON to pass, failed: %s", reason)
	}

	validJSONWithFences := "```json\n{\"name\": \"test_spec\", \"score\": 95.5, \"active\": true, \"tags\": [\"mtp\"]}\n```"
	if ok, reason := ValidateJSONSchema(validJSONWithFences, schema); !ok {
		t.Fatalf("expected valid JSON with fences to pass, failed: %s", reason)
	}

	// Syntax error
	syntaxErr := `{"name": "broken", "score": 95.5,`
	if ok, _ := ValidateJSONSchema(syntaxErr, schema); ok {
		t.Fatal("expected syntax error to fail")
	}

	// Missing key
	missingKey := `{"name": "test_spec", "score": 95.5, "tags": []}`
	if ok, _ := ValidateJSONSchema(missingKey, schema); ok {
		t.Fatal("expected missing required key to fail")
	}

	// Wrong type
	wrongType := `{"name": "test_spec", "score": "ninety-five", "active": true, "tags": []}`
	if ok, _ := ValidateJSONSchema(wrongType, schema); ok {
		t.Fatal("expected wrong type for score to fail")
	}
}

func TestValidateMathExact(t *testing.T) {
	if ok, reason := ValidateMathExact("391", "391"); !ok {
		t.Fatalf("exact string match failed: %s", reason)
	}

	if ok, reason := ValidateMathExact("The answer is 391.", "391"); !ok {
		t.Fatalf("embedded token match failed: %s", reason)
	}

	if ok, reason := ValidateMathExact("Alice", "Alice"); !ok {
		t.Fatalf("logic answer match failed: %s", reason)
	}

	if ok, reason := ValidateMathExact("The result is -42.", "-42"); !ok {
		t.Fatalf("negative math match failed: %s", reason)
	}

	if ok, _ := ValidateMathExact("The answer is 42.", "391"); ok {
		t.Fatal("incorrect answer should fail")
	}
}

// deterministicMockGenerator produces valid outputs for DefaultSmokeSuite
type deterministicMockGenerator struct {
	tpsMultiplier float64
	acceptancePct float64
}

func (m *deterministicMockGenerator) Generate(ctx context.Context, task EvalCase) (output string, proposed int, accepted int, totalTokens int, duration time.Duration, err error) {
	proposed = 40
	accepted = int(float64(proposed) * (m.acceptancePct / 100.0))
	totalTokens = 50
	durMs := float64(totalTokens) / (m.tpsMultiplier * 30.0) * 1000.0
	duration = time.Duration(durMs) * time.Millisecond
	if duration <= 0 {
		duration = 10 * time.Millisecond
	}

	switch task.Category {
	case CategoryCode:
		output = "def " + strings.ReplaceAll(task.ID, "code_", "") + "(x):\n    # implementation\n    return x\n"
	case CategoryJSON:
		if task.ID == "json_user_profile" {
			output = `{"id": 42, "username": "spec_user", "roles": ["admin"], "active": true}`
		} else {
			output = `{"engine": "fak-native", "tps": 36.5, "acceptance_rate": 0.85, "passed": true}`
		}
	case CategoryMath:
		output = "Answer is " + task.Expected
	}

	return output, proposed, accepted, totalTokens, duration, nil
}

func TestDeterministicSmokeEvaluation(t *testing.T) {
	mock := &deterministicMockGenerator{
		tpsMultiplier: 1.0,
		acceptancePct: 75.0,
	}

	gates := QualityGates{
		MinTPS:                   20.0,
		MinDraftAcceptancePct:    60.0,
		RequireAllFunctionalPass: true,
	}

	suite := DefaultSmokeSuite()
	report, err := RunEvaluation(context.Background(), mock, suite, gates)
	if err != nil {
		t.Fatalf("RunEvaluation failed: %v", err)
	}

	if !report.Passed {
		t.Fatalf("expected evaluation to pass gates, failed: %v", report.GateFailures)
	}
	if report.PassCount != len(suite) {
		t.Fatalf("pass count = %d, want %d", report.PassCount, len(suite))
	}
	if report.FailCount != 0 {
		t.Fatalf("fail count = %d, want 0", report.FailCount)
	}
	if report.OverallTPS < gates.MinTPS {
		t.Fatalf("overall TPS %.2f < %.2f", report.OverallTPS, gates.MinTPS)
	}
	if report.OverallAcceptancePct < gates.MinDraftAcceptancePct {
		t.Fatalf("overall acceptance %.2f%% < %.2f%%", report.OverallAcceptancePct, gates.MinDraftAcceptancePct)
	}
}

func TestQualityGatesFailures(t *testing.T) {
	// Mock with low acceptance rate (40%) and low TPS
	mock := &deterministicMockGenerator{
		tpsMultiplier: 0.2, // ~6 tok/s
		acceptancePct: 40.0,
	}

	gates := QualityGates{
		MinTPS:                   25.0,
		MinDraftAcceptancePct:    65.0,
		RequireAllFunctionalPass: true,
	}

	report, err := RunEvaluation(context.Background(), mock, DefaultSmokeSuite(), gates)
	if err != nil {
		t.Fatalf("RunEvaluation failed: %v", err)
	}

	if report.Passed {
		t.Fatal("expected report to fail gates")
	}
	if len(report.GateFailures) < 2 {
		t.Fatalf("expected at least 2 gate failures (TPS and acceptance), got %v", report.GateFailures)
	}
}

func TestQualityGatesMaxDurationFailure(t *testing.T) {
	mock := &deterministicMockGenerator{
		tpsMultiplier: 1.0,
		acceptancePct: 75.0,
	}

	gates := QualityGates{
		MinTPS:                   10.0,
		MinDraftAcceptancePct:    50.0,
		RequireAllFunctionalPass: true,
		MaxDurationMsPerTask:     0.001, // extremely tight threshold to trigger failure
	}

	report, err := RunEvaluation(context.Background(), mock, DefaultSmokeSuite(), gates)
	if err != nil {
		t.Fatalf("RunEvaluation failed: %v", err)
	}
	if report.Passed {
		t.Fatal("expected report to fail MaxDurationMsPerTask gate")
	}
	foundDurationFailure := false
	for _, f := range report.GateFailures {
		if strings.Contains(f, "exceeded max duration") {
			foundDurationFailure = true
			break
		}
	}
	if !foundDurationFailure {
		t.Fatalf("expected max duration gate failure in %v", report.GateFailures)
	}
}

func TestRunEvaluationContextCancellation(t *testing.T) {
	mock := &deterministicMockGenerator{
		tpsMultiplier: 1.0,
		acceptancePct: 75.0,
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := RunEvaluation(ctx, mock, DefaultSmokeSuite(), DefaultQualityGates())
	if err == nil {
		t.Fatal("expected context canceled error, got nil")
	}
}

func TestRunEvaluationNilGenerator(t *testing.T) {
	_, err := RunEvaluation(context.Background(), nil, DefaultSmokeSuite(), DefaultQualityGates())
	if err == nil {
		t.Fatal("expected error for nil generator")
	}
}
