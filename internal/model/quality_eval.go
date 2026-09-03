package model

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// TaskKind classifies the nature of the evaluation task.
type TaskKind string

const (
	TaskKindCode  TaskKind = "code"
	TaskKindJSON  TaskKind = "json"
	TaskKindLogic TaskKind = "logic"
)

// QualityTask represents an individual quality evaluation item.
type QualityTask struct {
	ID          string                             `json:"id"`
	Kind        TaskKind                           `json:"kind"`
	Prompt      string                             `json:"prompt"`
	Description string                             `json:"description"`
	Validate    func(output string) (bool, string) `json:"-"`
}

// TaskResult holds the execution and validation outcome of a single task.
type TaskResult struct {
	TaskID        string        `json:"task_id"`
	Kind          TaskKind      `json:"kind"`
	Output        string        `json:"output"`
	Pass          bool          `json:"pass"`
	Reason        string        `json:"reason,omitempty"`
	Tokens        int           `json:"tokens"`
	DraftAccepted int           `json:"draft_accepted"`
	Duration      time.Duration `json:"duration"`
	TokensPerSec  float64       `json:"tokens_per_sec"`
}

// QualityReport summarizes the multi-task evaluation run.
type QualityReport struct {
	TotalTasks    int          `json:"total_tasks"`
	PassedTasks   int          `json:"passed_tasks"`
	PassRate      float64      `json:"pass_rate"`
	TotalTokens   int          `json:"total_tokens"`
	DraftAccepted int          `json:"draft_accepted"`
	Results       []TaskResult `json:"results"`
}

// ValidateCodeOutput verifies that output defines the required function,
// contains a return statement, and preserves valid indentation in the function body.
func ValidateCodeOutput(requiredFunc string, output string) (bool, string) {
	if strings.TrimSpace(requiredFunc) == "" {
		return false, "missing required function name"
	}
	targetDef := "def " + strings.TrimSpace(requiredFunc)
	lines := strings.Split(output, "\n")

	defIndex := -1
	defIndent := 0

	for i, line := range lines {
		if idx := strings.Index(line, targetDef); idx != -1 {
			prefix := line[:idx]
			if strings.TrimSpace(prefix) == "" {
				defIndex = i
				defIndent = countLeadingWhitespace(prefix)
				break
			}
		}
	}

	if defIndex == -1 {
		return false, fmt.Sprintf("missing function definition: %s", requiredFunc)
	}

	if !strings.Contains(output, "return") {
		return false, "missing return statement"
	}

	hasIndentedBody := false
	for i := defIndex + 1; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "```") {
			continue
		}
		indent := countLeadingWhitespace(lines[i])
		if !hasIndentedBody {
			if indent <= defIndent {
				return false, "invalid indentation: body must be indented"
			}
			hasIndentedBody = true
		}
		if strings.Contains(lines[i], "return") && indent <= defIndent {
			return false, "invalid indentation: return statement must be indented"
		}
	}

	if !hasIndentedBody {
		return false, "invalid indentation: empty function body"
	}

	return true, ""
}

func countLeadingWhitespace(s string) int {
	indent := 0
	for _, r := range s {
		if r == ' ' {
			indent++
		} else if r == '\t' {
			indent += 4
		} else {
			break
		}
	}
	return indent
}

// ValidateJSONOutput extracts JSON from output (stripping code fences if any),
// unmarshals into map[string]any, and asserts all required fields exist and are non-empty.
func ValidateJSONOutput(requiredFields []string, output string) (bool, string) {
	clean := strings.TrimSpace(output)
	if clean == "" {
		return false, "empty output"
	}

	// Strip code fences if present: ```json ... ``` or ``` ... ```
	if start := strings.Index(clean, "```"); start != -1 {
		rest := clean[start+3:]
		if nl := strings.Index(rest, "\n"); nl != -1 {
			rest = rest[nl+1:]
		}
		if end := strings.Index(rest, "```"); end != -1 {
			clean = strings.TrimSpace(rest[:end])
		} else {
			clean = strings.TrimSpace(rest)
		}
	}

	var data map[string]any
	if err := json.Unmarshal([]byte(clean), &data); err != nil {
		firstBrace := strings.Index(clean, "{")
		lastBrace := strings.LastIndex(clean, "}")
		if firstBrace != -1 && lastBrace > firstBrace {
			slice := clean[firstBrace : lastBrace+1]
			if err2 := json.Unmarshal([]byte(slice), &data); err2 != nil {
				return false, fmt.Sprintf("invalid JSON: %v", err)
			}
		} else {
			return false, fmt.Sprintf("invalid JSON: %v", err)
		}
	}

	if data == nil {
		return false, "parsed JSON is null"
	}

	for _, field := range requiredFields {
		val, exists := data[field]
		if !exists {
			return false, fmt.Sprintf("missing required field: %s", field)
		}
		if val == nil {
			return false, fmt.Sprintf("field %s is null", field)
		}
		switch v := val.(type) {
		case string:
			if strings.TrimSpace(v) == "" {
				return false, fmt.Sprintf("field %s is empty", field)
			}
		case []any:
			if len(v) == 0 {
				return false, fmt.Sprintf("field %s is empty list", field)
			}
		case map[string]any:
			if len(v) == 0 {
				return false, fmt.Sprintf("field %s is empty object", field)
			}
		}
	}

	return true, ""
}

// ValidateLogicOutput checks that output contains the expected answer string (trimmed, case-insensitively).
func ValidateLogicOutput(expectedAnswer string, output string) (bool, string) {
	trimmedExpected := strings.TrimSpace(expectedAnswer)
	if trimmedExpected == "" {
		return false, "expected answer is empty"
	}
	trimmedOutput := strings.TrimSpace(output)
	if trimmedOutput == "" {
		return false, "output is empty"
	}

	if !strings.Contains(strings.ToLower(trimmedOutput), strings.ToLower(trimmedExpected)) {
		return false, fmt.Sprintf("expected %q not found in output", trimmedExpected)
	}

	return true, ""
}

// DefaultQualitySuite returns standard deterministic tasks across code, json, and logic kinds.
func DefaultQualitySuite() []QualityTask {
	return []QualityTask{
		{
			ID:          "code-fibonacci",
			Kind:        TaskKindCode,
			Prompt:      "Write a Python function named fibonacci(n) that returns the n-th Fibonacci number. Include a return statement.",
			Description: "Deterministic Python code generation with function definition, body indentation, and return statement.",
			Validate: func(out string) (bool, string) {
				return ValidateCodeOutput("fibonacci", out)
			},
		},
		{
			ID:          "json-user-info",
			Kind:        TaskKindJSON,
			Prompt:      `Extract user data from: "Alice is a 30-year-old developer." Return valid JSON with "name" and "age" keys.`,
			Description: "Deterministic JSON extraction with required non-empty fields.",
			Validate: func(out string) (bool, string) {
				return ValidateJSONOutput([]string{"name", "age"}, out)
			},
		},
		{
			ID:          "logic-next-sequence",
			Kind:        TaskKindLogic,
			Prompt:      "What is the next number in the sequence: 2, 4, 8, 16, ...? Answer with just the number.",
			Description: "Deterministic math sequence reasoning.",
			Validate: func(out string) (bool, string) {
				return ValidateLogicOutput("32", out)
			},
		},
	}
}

// RunQualityEvaluation runs each task through the generator, measures duration and tokens per second,
// evaluates outputs against validators, and compiles a QualityReport.
func RunQualityEvaluation(
	tasks []QualityTask,
	generate func(prompt string) (output string, tokens int, draftAccepted int, err error),
) (*QualityReport, error) {
	if generate == nil {
		return nil, fmt.Errorf("generate function must not be nil")
	}

	results := make([]TaskResult, 0, len(tasks))
	passedCount := 0
	sumTokens := 0
	sumDraftAccepted := 0

	for _, task := range tasks {
		start := time.Now()
		out, tok, draft, err := generate(task.Prompt)
		if err != nil {
			return nil, fmt.Errorf("task %s execution failed: %w", task.ID, err)
		}
		elapsed := time.Since(start)

		var tps float64
		if elapsed.Seconds() > 0 {
			tps = float64(tok) / elapsed.Seconds()
		}

		pass := true
		reason := ""
		if task.Validate != nil {
			pass, reason = task.Validate(out)
		}

		if pass {
			passedCount++
		}
		sumTokens += tok
		sumDraftAccepted += draft

		results = append(results, TaskResult{
			TaskID:        task.ID,
			Kind:          task.Kind,
			Output:        out,
			Pass:          pass,
			Reason:        reason,
			Tokens:        tok,
			DraftAccepted: draft,
			Duration:      elapsed,
			TokensPerSec:  tps,
		})
	}

	var passRate float64
	if len(tasks) > 0 {
		passRate = (float64(passedCount) / float64(len(tasks))) * 100.0
	}

	report := &QualityReport{
		TotalTasks:    len(tasks),
		PassedTasks:   passedCount,
		PassRate:      passRate,
		TotalTokens:   sumTokens,
		DraftAccepted: sumDraftAccepted,
		Results:       results,
	}

	return report, nil
}
