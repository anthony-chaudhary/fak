package mtpeval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// TaskCategory specifies the domain of evaluation.
type TaskCategory string

const (
	CategoryCode TaskCategory = "Code"
	CategoryJSON TaskCategory = "JSON"
	CategoryMath TaskCategory = "Math"
)

// JSONSchema specifies requirements for strict JSON validation.
type JSONSchema struct {
	RequiredKeys []string          `json:"required_keys"`
	KeyTypes     map[string]string `json:"key_types,omitempty"` // "string", "number", "boolean", "array", "object"
}

// EvalCase defines one evaluation test case.
type EvalCase struct {
	ID         string       `json:"id"`
	Name       string       `json:"name"`
	Category   TaskCategory `json:"category"`
	Prompt     string       `json:"prompt"`
	Expected   string       `json:"expected,omitempty"`    // For Math/Logic exact match
	JSONSchema *JSONSchema  `json:"json_schema,omitempty"` // For JSON validation
}

// SpeculativeGenerator defines the contract for models or harnesses under evaluation.
type SpeculativeGenerator interface {
	Generate(ctx context.Context, task EvalCase) (output string, proposed int, accepted int, totalTokens int, duration time.Duration, err error)
}

// TaskResult records the outcome of a single evaluation case.
type TaskResult struct {
	ID                 string       `json:"id"`
	Name               string       `json:"name"`
	Category           TaskCategory `json:"category"`
	Prompt             string       `json:"prompt"`
	Output             string       `json:"output"`
	FunctionalPass     bool         `json:"functional_pass"`
	FailureReason      string       `json:"failure_reason,omitempty"`
	ProposedTokens     int          `json:"proposed_tokens"`
	AcceptedTokens     int          `json:"accepted_tokens"`
	DraftAcceptancePct float64      `json:"draft_acceptance_pct"`
	TotalTokens        int          `json:"total_tokens"`
	DurationMs         float64      `json:"duration_ms"`
	TPS                float64      `json:"tps"`
}

// QualityGates defines the checkable pass/fail thresholds.
type QualityGates struct {
	MinTPS                   float64 `json:"min_tps"`
	MinDraftAcceptancePct    float64 `json:"min_draft_acceptance_pct"`
	RequireAllFunctionalPass bool    `json:"require_all_functional_pass"`
	MaxDurationMsPerTask     float64 `json:"max_duration_ms_per_task,omitempty"`
}

// DefaultQualityGates returns the standard quality criteria for MTP speculative evaluation.
func DefaultQualityGates() QualityGates {
	return QualityGates{
		MinTPS:                   15.0,
		MinDraftAcceptancePct:    50.0,
		RequireAllFunctionalPass: true,
		MaxDurationMsPerTask:     5000.0,
	}
}

// EvaluationReport contains aggregated multi-task results and gate verdicts.
type EvaluationReport struct {
	Passed               bool         `json:"passed"`
	GateFailures         []string     `json:"gate_failures,omitempty"`
	Tasks                []TaskResult `json:"tasks"`
	TotalTokens          int          `json:"total_tokens"`
	OverallTPS           float64      `json:"overall_tps"`
	OverallAcceptancePct float64      `json:"overall_acceptance_pct"`
	TotalDurationMs      float64      `json:"total_duration_ms"`
	PassCount            int          `json:"pass_count"`
	FailCount            int          `json:"fail_count"`
}

// DefaultSmokeSuite returns deterministic multi-task quality and smoke test cases.
func DefaultSmokeSuite() []EvalCase {
	return []EvalCase{
		// Code validation cases
		{
			ID:       "code_factorial",
			Name:     "Python Factorial Function",
			Category: CategoryCode,
			Prompt:   "Write a Python function factorial(n) that computes the factorial of n recursively or iteratively.",
		},
		{
			ID:       "code_binary_search",
			Name:     "Python Binary Search Function",
			Category: CategoryCode,
			Prompt:   "Write a Python function binary_search(arr, target) that returns the index of target or -1.",
		},
		{
			ID:       "code_reverse_string",
			Name:     "Python String Reversal Function",
			Category: CategoryCode,
			Prompt:   "Write a Python function reverse_string(s) that returns the reversed string.",
		},

		// JSON strict schema validation cases
		{
			ID:       "json_user_profile",
			Name:     "User Profile JSON",
			Category: CategoryJSON,
			Prompt:   "Output a JSON object with user details including id, username, roles, and active status.",
			JSONSchema: &JSONSchema{
				RequiredKeys: []string{"id", "username", "roles", "active"},
				KeyTypes: map[string]string{
					"id":       "number",
					"username": "string",
					"roles":    "array",
					"active":   "boolean",
				},
			},
		},
		{
			ID:       "json_spec_benchmark",
			Name:     "Speculative Benchmark Result JSON",
			Category: CategoryJSON,
			Prompt:   "Output a JSON object reporting benchmark results with engine, tps, acceptance_rate, and passed.",
			JSONSchema: &JSONSchema{
				RequiredKeys: []string{"engine", "tps", "acceptance_rate", "passed"},
				KeyTypes: map[string]string{
					"engine":          "string",
					"tps":             "number",
					"acceptance_rate": "number",
					"passed":          "boolean",
				},
			},
		},

		// Math / Logic exact match cases
		{
			ID:       "math_mult",
			Name:     "Multiplication Arithmetic",
			Category: CategoryMath,
			Prompt:   "What is 23 multiplied by 17?",
			Expected: "391",
		},
		{
			ID:       "math_fib",
			Name:     "Fibonacci Value",
			Category: CategoryMath,
			Prompt:   "What is the 10th Fibonacci number (where F(0)=0, F(1)=1, F(2)=1)?",
			Expected: "55",
		},
		{
			ID:       "logic_ordering",
			Name:     "Transitive Ordering Logic",
			Category: CategoryMath,
			Prompt:   "If Alice is taller than Bob, and Bob is taller than Charlie, who is the tallest person?",
			Expected: "Alice",
		},
	}
}

// isEscaped returns true if character at idx is preceded by an odd number of backslashes.
func isEscaped(s string, idx int) bool {
	numBackslashes := 0
	for j := idx - 1; j >= 0 && s[j] == '\\'; j-- {
		numBackslashes++
	}
	return numBackslashes%2 == 1
}

func isWordChar(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_'
}

// ValidatePythonCode checks Python syntax integrity: balanced brackets, function header, and body indentation.
func ValidatePythonCode(output string) (bool, string) {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return false, "empty Python code output"
	}

	// Remove optional markdown code fences if present
	lines := strings.Split(trimmed, "\n")
	var codeLines []string
	inCode := false
	sawFence := false
	for _, line := range lines {
		l := strings.TrimSpace(line)
		if strings.HasPrefix(l, "```") {
			sawFence = true
			inCode = !inCode
			continue
		}
		if inCode {
			codeLines = append(codeLines, line)
		}
	}
	if !sawFence {
		codeLines = lines
	}

	codeText := strings.Join(codeLines, "\n")

	// 1. Check balanced delimiters
	var stack []rune
	matching := map[rune]rune{')': '(', ']': '[', '}': '{'}
	inSingleQuote := false
	inDoubleQuote := false
	inComment := false

	for i, r := range codeText {
		if r == '#' && !inSingleQuote && !inDoubleQuote {
			inComment = true
			continue
		}
		if r == '\n' {
			inComment = false
			continue
		}
		if inComment {
			continue
		}

		if r == '\'' && !inDoubleQuote {
			if !isEscaped(codeText, i) {
				inSingleQuote = !inSingleQuote
			}
			continue
		}
		if r == '"' && !inSingleQuote {
			if !isEscaped(codeText, i) {
				inDoubleQuote = !inDoubleQuote
			}
			continue
		}
		if inSingleQuote || inDoubleQuote {
			continue
		}

		if r == '(' || r == '[' || r == '{' {
			stack = append(stack, r)
		} else if closing, ok := matching[r]; ok {
			if len(stack) == 0 || stack[len(stack)-1] != closing {
				return false, fmt.Sprintf("unbalanced delimiter %c", r)
			}
			stack = stack[:len(stack)-1]
		}
	}

	if len(stack) > 0 {
		return false, fmt.Sprintf("unclosed delimiter %c", stack[len(stack)-1])
	}
	if inSingleQuote || inDoubleQuote {
		return false, "unterminated string quote"
	}

	// 2. Check function definition: def <name>(...):
	defPattern := regexp.MustCompile(`(?m)^\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*:`)
	if !defPattern.MatchString(codeText) {
		return false, "missing valid function definition header 'def <name>(...):'"
	}

	// 3. Check for indentation in function body
	sawDef := false
	sawBody := false
	for _, l := range codeLines {
		trimmedL := strings.TrimSpace(l)
		if trimmedL == "" || strings.HasPrefix(trimmedL, "#") {
			continue
		}
		if strings.HasPrefix(trimmedL, "def ") {
			sawDef = true
			continue
		}
		if sawDef {
			if strings.HasPrefix(l, "    ") || strings.HasPrefix(l, "\t") || strings.HasPrefix(l, "  ") {
				sawBody = true
				break
			}
		}
	}
	if sawDef && !sawBody {
		return false, "function body missing proper indentation"
	}

	return true, ""
}

// ValidateJSONSchema checks that the string is valid JSON and complies with required keys and types.
func ValidateJSONSchema(output string, schema *JSONSchema) (bool, string) {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return false, "empty JSON output"
	}

	// Remove optional markdown code fences if present
	lines := strings.Split(trimmed, "\n")
	var jsonLines []string
	inCode := false
	sawFence := false
	for _, line := range lines {
		l := strings.TrimSpace(line)
		if strings.HasPrefix(l, "```") {
			sawFence = true
			inCode = !inCode
			continue
		}
		if inCode {
			jsonLines = append(jsonLines, line)
		}
	}
	if sawFence && len(jsonLines) > 0 {
		trimmed = strings.Join(jsonLines, "\n")
	}

	// Extract JSON block if surrounded by markdown fences or text
	jsonStr := strings.TrimSpace(trimmed)
	if start := strings.Index(jsonStr, "{"); start >= 0 {
		if end := strings.LastIndex(jsonStr, "}"); end > start {
			jsonStr = jsonStr[start : end+1]
		}
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return false, fmt.Sprintf("invalid JSON syntax: %v", err)
	}

	if schema == nil {
		return true, ""
	}

	// Check required keys
	for _, reqKey := range schema.RequiredKeys {
		val, exists := parsed[reqKey]
		if !exists {
			return false, fmt.Sprintf("missing required JSON key %q", reqKey)
		}

		if expectedType, ok := schema.KeyTypes[reqKey]; ok {
			switch expectedType {
			case "string":
				if _, ok := val.(string); !ok {
					return false, fmt.Sprintf("key %q must be string, got %T", reqKey, val)
				}
			case "number":
				if _, ok := val.(float64); !ok {
					return false, fmt.Sprintf("key %q must be number, got %T", reqKey, val)
				}
			case "boolean":
				if _, ok := val.(bool); !ok {
					return false, fmt.Sprintf("key %q must be boolean, got %T", reqKey, val)
				}
			case "array":
				if _, ok := val.([]interface{}); !ok {
					return false, fmt.Sprintf("key %q must be array, got %T", reqKey, val)
				}
			case "object":
				if _, ok := val.(map[string]interface{}); !ok {
					return false, fmt.Sprintf("key %q must be object, got %T", reqKey, val)
				}
			}
		}
	}

	return true, ""
}

// ValidateMathExact checks whether the output accurately contains the expected result.
func ValidateMathExact(output, expected string) (bool, string) {
	trimmedOutput := strings.TrimSpace(output)
	trimmedExpected := strings.TrimSpace(expected)
	if trimmedOutput == "" {
		return false, "empty math/logic output"
	}

	// Direct match or contains expected token/value
	if strings.EqualFold(trimmedOutput, trimmedExpected) {
		return true, ""
	}

	// Check word boundaries for exact match in generated text
	prefixBoundary := `\b`
	if len(trimmedExpected) > 0 && !isWordChar(rune(trimmedExpected[0])) {
		prefixBoundary = `(?:\s|^)`
	}
	suffixBoundary := `\b`
	if len(trimmedExpected) > 0 && !isWordChar(rune(trimmedExpected[len(trimmedExpected)-1])) {
		suffixBoundary = `(?:\s|$)`
	}
	pattern := fmt.Sprintf(`(?i)%s%s%s`, prefixBoundary, regexp.QuoteMeta(trimmedExpected), suffixBoundary)
	re, err := regexp.Compile(pattern)
	if err == nil && re.MatchString(trimmedOutput) {
		return true, ""
	}

	// Numerical parsing check if both are numbers
	if outNum, err1 := strconv.ParseFloat(trimmedOutput, 64); err1 == nil {
		if expNum, err2 := strconv.ParseFloat(trimmedExpected, 64); err2 == nil {
			if math.Abs(outNum-expNum) < 1e-6 {
				return true, ""
			}
		}
	}

	return false, fmt.Sprintf("expected %q, but output did not match (got %q)", trimmedExpected, trimmedOutput)
}

// ValidateTaskOutput applies the category-specific functional validator.
func ValidateTaskOutput(task EvalCase, output string) (bool, string) {
	switch task.Category {
	case CategoryCode:
		return ValidatePythonCode(output)
	case CategoryJSON:
		return ValidateJSONSchema(output, task.JSONSchema)
	case CategoryMath:
		return ValidateMathExact(output, task.Expected)
	default:
		if strings.TrimSpace(output) == "" {
			return false, "empty output"
		}
		return true, ""
	}
}

// RunEvaluation evaluates a speculative generator across all cases in the suite.
func RunEvaluation(ctx context.Context, gen SpeculativeGenerator, suite []EvalCase, gates QualityGates) (*EvaluationReport, error) {
	if gen == nil {
		return nil, errors.New("eval: generator is required")
	}
	if len(suite) == 0 {
		suite = DefaultSmokeSuite()
	}

	var results []TaskResult
	totalTokens := 0
	totalProposed := 0
	totalAccepted := 0
	var totalDuration time.Duration

	passCount := 0
	failCount := 0

	for _, tc := range suite {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		out, proposed, accepted, tokens, dur, err := gen.Generate(ctx, tc)
		durMs := float64(dur.Milliseconds())
		if durMs <= 0 {
			durMs = float64(dur.Nanoseconds()) / 1e6
		}

		var tps float64
		if dur.Seconds() > 0 {
			tps = float64(tokens) / dur.Seconds()
		}

		var acceptPct float64
		if proposed > 0 {
			acceptPct = (float64(accepted) / float64(proposed)) * 100.0
		}

		valid := false
		failureReason := ""
		if err != nil {
			failureReason = err.Error()
		} else {
			valid, failureReason = ValidateTaskOutput(tc, out)
		}

		if valid {
			passCount++
		} else {
			failCount++
		}

		res := TaskResult{
			ID:                 tc.ID,
			Name:               tc.Name,
			Category:           tc.Category,
			Prompt:             tc.Prompt,
			Output:             out,
			FunctionalPass:     valid,
			FailureReason:      failureReason,
			ProposedTokens:     proposed,
			AcceptedTokens:     accepted,
			DraftAcceptancePct: acceptPct,
			TotalTokens:        tokens,
			DurationMs:         durMs,
			TPS:                tps,
		}
		results = append(results, res)

		totalTokens += tokens
		totalProposed += proposed
		totalAccepted += accepted
		totalDuration += dur
	}

	totalDurSec := totalDuration.Seconds()
	overallTPS := 0.0
	if totalDurSec > 0 {
		overallTPS = float64(totalTokens) / totalDurSec
	}

	overallAcceptPct := 0.0
	if totalProposed > 0 {
		overallAcceptPct = (float64(totalAccepted) / float64(totalProposed)) * 100.0
	}

	// Evaluate Checkable Quality Gates
	var gateFailures []string
	if gates.RequireAllFunctionalPass && failCount > 0 {
		gateFailures = append(gateFailures, fmt.Sprintf("%d/%d tasks failed functional validation", failCount, len(suite)))
	}
	if gates.MinTPS > 0 && overallTPS < gates.MinTPS {
		gateFailures = append(gateFailures, fmt.Sprintf("overall TPS %.2f < threshold %.2f", overallTPS, gates.MinTPS))
	}
	if gates.MinDraftAcceptancePct > 0 && overallAcceptPct < gates.MinDraftAcceptancePct {
		gateFailures = append(gateFailures, fmt.Sprintf("overall draft acceptance %.1f%% < threshold %.1f%%", overallAcceptPct, gates.MinDraftAcceptancePct))
	}
	if gates.MaxDurationMsPerTask > 0 {
		var slowTasks []string
		for _, res := range results {
			if res.DurationMs > gates.MaxDurationMsPerTask {
				slowTasks = append(slowTasks, fmt.Sprintf("%s (%.1fms > %.1fms)", res.ID, res.DurationMs, gates.MaxDurationMsPerTask))
			}
		}
		if len(slowTasks) > 0 {
			gateFailures = append(gateFailures, fmt.Sprintf("%d task(s) exceeded max duration of %.1fms: %s", len(slowTasks), gates.MaxDurationMsPerTask, strings.Join(slowTasks, ", ")))
		}
	}

	passed := len(gateFailures) == 0

	totalDurMs := float64(totalDuration.Milliseconds())
	if totalDurMs <= 0 && totalDuration > 0 {
		totalDurMs = float64(totalDuration.Nanoseconds()) / 1e6
	}

	return &EvaluationReport{
		Passed:               passed,
		GateFailures:         gateFailures,
		Tasks:                results,
		TotalTokens:          totalTokens,
		OverallTPS:           overallTPS,
		OverallAcceptancePct: overallAcceptPct,
		TotalDurationMs:      totalDurMs,
		PassCount:            passCount,
		FailCount:            failCount,
	}, nil
}
