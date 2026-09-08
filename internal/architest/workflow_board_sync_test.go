package architest

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type gitHubEventContext struct {
	EventName string
	Action    string
	LabelName string
}

func (c gitHubEventContext) Get(field string) string {
	switch field {
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

// evalProjectBoardSyncIf is the reference Go model of the project-board-sync add-to-board if condition:
// github.event_name != 'issues' || github.event.action != 'labeled' || startsWith(github.event.label.name, 'gen/')
func evalProjectBoardSyncIf(ctx gitHubEventContext) bool {
	return ctx.EventName != "issues" || ctx.Action != "labeled" || strings.HasPrefix(ctx.LabelName, "gen/")
}

// evalActionsExpr parses and evaluates a disjunction (||) of GitHub Actions expression clauses
// extracted directly from the workflow file.
func evalActionsExpr(expr string, ctx gitHubEventContext) (bool, error) {
	clauses := strings.Split(expr, "||")
	for _, raw := range clauses {
		clause := strings.TrimSpace(raw)
		if clause == "" {
			continue
		}

		if strings.HasPrefix(clause, "startsWith(") && strings.HasSuffix(clause, ")") {
			inner := strings.TrimSuffix(strings.TrimPrefix(clause, "startsWith("), ")")
			parts := strings.Split(inner, ",")
			if len(parts) != 2 {
				return false, fmt.Errorf("malformed startsWith clause: %q", clause)
			}
			target := strings.TrimSpace(parts[0])
			prefix := strings.Trim(strings.TrimSpace(parts[1]), "'\"")
			val := ctx.Get(target)
			if strings.HasPrefix(val, prefix) {
				return true, nil
			}
			continue
		}

		if idx := strings.Index(clause, "!="); idx != -1 {
			left := strings.TrimSpace(clause[:idx])
			right := strings.Trim(strings.TrimSpace(clause[idx+2:]), "'\"")
			leftVal := ctx.Get(left)
			if leftVal != right {
				return true, nil
			}
			continue
		}

		if idx := strings.Index(clause, "=="); idx != -1 {
			left := strings.TrimSpace(clause[:idx])
			right := strings.Trim(strings.TrimSpace(clause[idx+2:]), "'\"")
			leftVal := ctx.Get(left)
			if leftVal == right {
				return true, nil
			}
			continue
		}

		return false, fmt.Errorf("unsupported expression clause: %q", clause)
	}
	return false, nil
}

// extractJobIfClause extracts the collapsed if: expression and line numbers from the add-to-board job.
func extractJobIfClause(t *testing.T, content string) (expr string, ifLine int, runsOnLine int) {
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
			// Stop if a new sibling job begins (indentation 2 spaces)
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

func TestProjectBoardSync(t *testing.T) {
	root := repoRoot(t)
	workflowPath := filepath.Join(root, ".github", "workflows", "project-board-sync.yml")

	data, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatalf("read %s: %v", workflowPath, err)
	}
	content := strings.ReplaceAll(string(data), "\r\n", "\n")

	const expectedConditionContract = `    if: >-
      github.event_name != 'issues' ||
      github.event.action != 'labeled' ||
      startsWith(github.event.label.name, 'gen/')`

	t.Run("ConditionPresence", func(t *testing.T) {
		if !strings.Contains(content, expectedConditionContract) {
			t.Fatalf("project-board-sync.yml does not contain expected condition contract:\n%s", expectedConditionContract)
		}

		expr, ifLine, runsOnLine := extractJobIfClause(t, content)
		if ifLine >= runsOnLine {
			t.Fatalf("job if: at line %d must be placed before runs-on: at line %d", ifLine, runsOnLine)
		}

		expectedSubstrings := []string{
			"github.event_name != 'issues'",
			"github.event.action != 'labeled'",
			"startsWith(github.event.label.name, 'gen/')",
		}
		for _, sub := range expectedSubstrings {
			if !strings.Contains(expr, sub) {
				t.Errorf("extracted if: expression missing expected clause %q; got: %s", sub, expr)
			}
		}
	})

	t.Run("BooleanLogic", func(t *testing.T) {
		expr, _, _ := extractJobIfClause(t, content)

		testCases := []struct {
			name     string
			ctx      gitHubEventContext
			expected bool
		}{
			{
				name:     "workflow_dispatch (allowed)",
				ctx:      gitHubEventContext{EventName: "workflow_dispatch"},
				expected: true,
			},
			{
				name:     "issues.opened (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "opened"},
				expected: true,
			},
			{
				name:     "issues.reopened (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "reopened"},
				expected: true,
			},
			{
				name:     "issues.transferred (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "transferred"},
				expected: true,
			},
			{
				name:     "issues.labeled with gen/now (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "gen/now"},
				expected: true,
			},
			{
				name:     "issues.labeled with gen/next (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "gen/next"},
				expected: true,
			},
			{
				name:     "issues.labeled with gen/second-next (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "gen/second-next"},
				expected: true,
			},
			{
				name:     "issues.labeled with gen/future (allowed)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "gen/future"},
				expected: true,
			},
			{
				name:     "issues.labeled with bug (denied / skipped)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "bug"},
				expected: false,
			},
			{
				name:     "issues.labeled with priority/P0 (denied / skipped)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "priority/P0"},
				expected: false,
			},
			{
				name:     "issues.labeled with compute (denied / skipped)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "compute"},
				expected: false,
			},
			{
				name:     "issues.labeled with documentation (denied / skipped)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "documentation"},
				expected: false,
			},
			{
				name:     "issues.labeled with area/ci (denied / skipped)",
				ctx:      gitHubEventContext{EventName: "issues", Action: "labeled", LabelName: "area/ci"},
				expected: false,
			},
		}

		for _, tc := range testCases {
			tc := tc
			t.Run(tc.name, func(t *testing.T) {
				// Evaluate against extracted YAML expression
				gotExpr, err := evalActionsExpr(expr, tc.ctx)
				if err != nil {
					t.Fatalf("evalActionsExpr failed: %v", err)
				}
				if gotExpr != tc.expected {
					t.Errorf("evalActionsExpr(%+v) = %v; want %v", tc.ctx, gotExpr, tc.expected)
				}

				// Verify reference Go logic also agrees
				gotModel := evalProjectBoardSyncIf(tc.ctx)
				if gotModel != tc.expected {
					t.Errorf("evalProjectBoardSyncIf(%+v) = %v; want %v", tc.ctx, gotModel, tc.expected)
				}
			})
		}
	})

	t.Run("LabelEventStormCoalescing", func(t *testing.T) {
		expr, _, _ := extractJobIfClause(t, content)

		// Simulate an issue creation sequence:
		// 1 issue opened event
		// 8 non-gen classification label events (storm)
		// 1 gen/now milestone label event
		events := []gitHubEventContext{
			{EventName: "issues", Action: "opened"},
			{EventName: "issues", Action: "labeled", LabelName: "bug"},
			{EventName: "issues", Action: "labeled", LabelName: "priority/P0"},
			{EventName: "issues", Action: "labeled", LabelName: "compute"},
			{EventName: "issues", Action: "labeled", LabelName: "area/ci"},
			{EventName: "issues", Action: "labeled", LabelName: "documentation"},
			{EventName: "issues", Action: "labeled", LabelName: "performance"},
			{EventName: "issues", Action: "labeled", LabelName: "triage"},
			{EventName: "issues", Action: "labeled", LabelName: "vulkan"},
			{EventName: "issues", Action: "labeled", LabelName: "gen/now"},
		}

		admitted := 0
		skipped := 0

		for _, evt := range events {
			pass, err := evalActionsExpr(expr, evt)
			if err != nil {
				t.Fatalf("evalActionsExpr(%+v): %v", evt, err)
			}
			if pass {
				admitted++
			} else {
				skipped++
			}
		}

		if admitted != 2 {
			t.Errorf("admitted runs = %d; want 2 (1 opened + 1 gen/now)", admitted)
		}
		if skipped != 8 {
			t.Errorf("skipped runs = %d; want 8 (all non-gen label storm events coalesced)", skipped)
		}
	})
}
