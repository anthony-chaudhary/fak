package workflows

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// GitHubContext represents the subset of GitHub Actions event context
// needed to evaluate the project-board-sync filter predicate.
type GitHubContext struct {
	EventName string
	Action    string
	LabelName string
}

func (c GitHubContext) Get(path string) string {
	switch path {
	case "github.event_name":
		return c.EventName
	case "github.event.action":
		return c.Action
	case "github.event.label.name":
		return c.LabelName
	default:
		return ""
	}
}

// locateWorkflowFile finds project-board-sync.yml from test working directory or source tree.
func locateWorkflowFile(t *testing.T) string {
	t.Helper()
	candidates := []string{
		"project-board-sync.yml",
		filepath.Join(".github", "workflows", "project-board-sync.yml"),
		filepath.Join("..", "..", ".github", "workflows", "project-board-sync.yml"),
	}
	for _, cand := range candidates {
		if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
			abs, err := filepath.Abs(cand)
			if err == nil {
				return abs
			}
			return cand
		}
	}
	if _, file, _, ok := runtime.Caller(0); ok {
		cand := filepath.Join(filepath.Dir(file), "project-board-sync.yml")
		if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
			return cand
		}
	}
	wd, err := os.Getwd()
	if err == nil {
		cur := wd
		for i := 0; i < 6; i++ {
			cand := filepath.Join(cur, ".github", "workflows", "project-board-sync.yml")
			if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
				return cand
			}
			parent := filepath.Dir(cur)
			if parent == cur {
				break
			}
			cur = parent
		}
	}
	t.Fatalf("could not locate project-board-sync.yml")
	return ""
}

// extractJobIf extracts the collapsed expression string and line positions from add-to-board job.
func extractJobIf(t *testing.T, content string) (expr string, ifLine int, runsOnLine int) {
	t.Helper()
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")

	inJobs := false
	inAddJob := false
	inIf := false
	var ifParts []string

	for idx, line := range lines {
		lineNum := idx + 1
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(line, "jobs:") {
			inJobs = true
			continue
		}
		if inJobs && strings.HasPrefix(line, "  add-to-board:") {
			inAddJob = true
			continue
		}
		if inAddJob {
			// If we reached a new top-level or new job key (2 spaces indentation)
			if len(line) > 2 && line[0] == ' ' && line[1] == ' ' && line[2] != ' ' && !strings.HasPrefix(line, "  add-to-board:") {
				break
			}
			if strings.HasPrefix(line, "    if:") {
				ifLine = lineNum
				inIf = true
				rest := strings.TrimSpace(strings.TrimPrefix(line, "    if:"))
				if rest != "" && rest != ">-" && rest != ">" && rest != "|" && rest != "|-" {
					ifParts = append(ifParts, rest)
				}
				continue
			}
			if strings.HasPrefix(line, "    runs-on:") {
				runsOnLine = lineNum
				inIf = false
				continue
			}
			if inIf {
				// Continuation line in folded scalar indented at least 6 spaces
				if strings.HasPrefix(line, "      ") {
					ifParts = append(ifParts, trimmed)
				} else {
					inIf = false
				}
			}
		}
	}

	if ifLine == 0 {
		t.Fatalf("job-level if: condition not found under add-to-board")
	}
	if runsOnLine == 0 {
		t.Fatalf("runs-on: not found under add-to-board")
	}
	if len(ifParts) == 0 {
		t.Fatalf("job-level if: expression body is empty")
	}

	return strings.Join(ifParts, " "), ifLine, runsOnLine
}

// EvaluateActionsExpression evaluates a boolean expression consisting of comparisons joined by '||'.
func EvaluateActionsExpression(expr string, ctx GitHubContext) (bool, error) {
	clauses := strings.Split(expr, "||")
	for _, raw := range clauses {
		clause := strings.TrimSpace(raw)
		if clause == "" {
			continue
		}

		var isNotEqual bool
		var left, right string
		if idx := strings.Index(clause, "!="); idx != -1 {
			isNotEqual = true
			left = strings.TrimSpace(clause[:idx])
			right = strings.TrimSpace(clause[idx+2:])
		} else if idx := strings.Index(clause, "=="); idx != -1 {
			isNotEqual = false
			left = strings.TrimSpace(clause[:idx])
			right = strings.TrimSpace(clause[idx+2:])
		} else {
			return false, fmt.Errorf("unsupported operator in clause: %q", clause)
		}

		right = strings.Trim(right, "'\"")
		leftVal := ctx.Get(left)

		matched := strings.EqualFold(leftVal, right)
		if isNotEqual {
			if !matched {
				return true, nil // short-circuit OR
			}
		} else {
			if matched {
				return true, nil // short-circuit OR
			}
		}
	}
	return false, nil
}

func TestProjectBoardSyncPlacement(t *testing.T) {
	path := locateWorkflowFile(t)
	contentBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read workflow file: %v", err)
	}
	content := string(contentBytes)

	expr, ifLine, runsOnLine := extractJobIf(t, content)

	if ifLine >= runsOnLine {
		t.Fatalf("job-level if: at line %d must be placed before runs-on: at line %d", ifLine, runsOnLine)
	}

	expectedSubstrings := []string{
		"github.event_name == 'workflow_dispatch'",
		"github.event.action != 'labeled'",
		"github.event.label.name == 'gen/now'",
		"github.event.label.name == 'gen/next'",
		"github.event.label.name == 'gen/second-next'",
		"github.event.label.name == 'gen/future'",
	}
	for _, sub := range expectedSubstrings {
		if !strings.Contains(expr, sub) {
			t.Errorf("extracted if: expression missing expected clause %q", sub)
		}
	}
}

func TestProjectBoardSyncEvaluator(t *testing.T) {
	path := locateWorkflowFile(t)
	contentBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read workflow file: %v", err)
	}
	expr, _, _ := extractJobIf(t, string(contentBytes))

	testCases := []struct {
		name     string
		ctx      GitHubContext
		expected bool
	}{
		{
			name:     "workflow_dispatch",
			ctx:      GitHubContext{EventName: "workflow_dispatch"},
			expected: true,
		},
		{
			name:     "issues opened",
			ctx:      GitHubContext{EventName: "issues", Action: "opened"},
			expected: true,
		},
		{
			name:     "issues reopened",
			ctx:      GitHubContext{EventName: "issues", Action: "reopened"},
			expected: true,
		},
		{
			name:     "issues transferred",
			ctx:      GitHubContext{EventName: "issues", Action: "transferred"},
			expected: true,
		},
		{
			name:     "issues labeled with gen/now",
			ctx:      GitHubContext{EventName: "issues", Action: "labeled", LabelName: "gen/now"},
			expected: true,
		},
		{
			name:     "issues labeled with gen/next",
			ctx:      GitHubContext{EventName: "issues", Action: "labeled", LabelName: "gen/next"},
			expected: true,
		},
		{
			name:     "issues labeled with gen/second-next",
			ctx:      GitHubContext{EventName: "issues", Action: "labeled", LabelName: "gen/second-next"},
			expected: true,
		},
		{
			name:     "issues labeled with gen/future",
			ctx:      GitHubContext{EventName: "issues", Action: "labeled", LabelName: "gen/future"},
			expected: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got, err := EvaluateActionsExpression(expr, tc.ctx)
			if err != nil {
				t.Fatalf("evaluation failed: %v", err)
			}
			if got != tc.expected {
				t.Errorf("eval(%+v) = %v; want %v", tc.ctx, got, tc.expected)
			}
		})
	}

	nonGenLabels := []string{
		"bug",
		"performance",
		"priority/P0",
		"qwen",
		"class:dev",
		"vulkan",
		"speculative-decoding",
	}

	for _, label := range nonGenLabels {
		label := label
		t.Run("issues labeled with non-gen "+label, func(t *testing.T) {
			ctx := GitHubContext{EventName: "issues", Action: "labeled", LabelName: label}
			got, err := EvaluateActionsExpression(expr, ctx)
			if err != nil {
				t.Fatalf("evaluation failed: %v", err)
			}
			if got != false {
				t.Errorf("eval(%+v) = true; want false (skipped)", ctx)
			}
		})
	}
}

func TestProjectBoardSyncBulkLabelingQueueCostBound(t *testing.T) {
	path := locateWorkflowFile(t)
	contentBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read workflow file: %v", err)
	}
	expr, _, _ := extractJobIf(t, string(contentBytes))

	// Simulate issue creation flow:
	// 1 open event + 10 non-gen labels + 1 gen/now label
	events := []GitHubContext{
		{EventName: "issues", Action: "opened"},
		{EventName: "issues", Action: "labeled", LabelName: "bug"},
		{EventName: "issues", Action: "labeled", LabelName: "performance"},
		{EventName: "issues", Action: "labeled", LabelName: "priority/P0"},
		{EventName: "issues", Action: "labeled", LabelName: "qwen"},
		{EventName: "issues", Action: "labeled", LabelName: "class:dev"},
		{EventName: "issues", Action: "labeled", LabelName: "vulkan"},
		{EventName: "issues", Action: "labeled", LabelName: "speculative-decoding"},
		{EventName: "issues", Action: "labeled", LabelName: "area:ci"},
		{EventName: "issues", Action: "labeled", LabelName: "backend"},
		{EventName: "issues", Action: "labeled", LabelName: "triage"},
		{EventName: "issues", Action: "labeled", LabelName: "gen/now"},
	}

	allocatedRunners := 0
	skippedEvents := 0

	for _, evt := range events {
		admitted, err := EvaluateActionsExpression(expr, evt)
		if err != nil {
			t.Fatalf("evaluate event %+v: %v", evt, err)
		}
		if admitted {
			allocatedRunners++
		} else {
			skippedEvents++
		}
	}

	if allocatedRunners != 2 {
		t.Fatalf("expected exactly 2 runner allocations (1 open + 1 gen-sync), got %d", allocatedRunners)
	}
	if skippedEvents != 10 {
		t.Fatalf("expected exactly 10 skipped non-gen label events, got %d", skippedEvents)
	}
}
