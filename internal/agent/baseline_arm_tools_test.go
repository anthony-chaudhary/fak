package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type baselineArmTestPlanner struct {
	toolName    string
	toolOutputs []string
}

func (p *baselineArmTestPlanner) Model() string { return "baseline-arm-test-model" }

func (p *baselineArmTestPlanner) Complete(_ context.Context, messages []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	tool := p.toolName
	if tool == "" {
		tool = "Read"
	}
	for _, m := range messages {
		if m.Role == RoleTool && (m.Name == tool || strings.EqualFold(m.Name, tool)) {
			p.toolOutputs = append(p.toolOutputs, m.Content)
			return &Completion{Message: Message{Role: RoleAssistant, Content: "read completed: " + m.Content}}, nil
		}
	}
	return &Completion{
		Message: Message{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{
					ID:   "call-read-1",
					Type: "function",
					Function: Func{
						Name:      tool,
						Arguments: `{"file_path":"test.txt"}`,
					},
				},
			},
		},
	}, nil
}

func TestBaselineArmToolCatalogExecution(t *testing.T) {
	dir := t.TempDir()
	testContent := "hello test baseline arm execution content 12345"
	testFilePath := filepath.Join(dir, "test.txt")
	if err := os.WriteFile(testFilePath, []byte(testContent), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cat, err := ArmCodeTools(dir)
	if err != nil {
		t.Fatalf("ArmCodeTools: %v", err)
	}
	t.Cleanup(DisarmCodeTools)

	ctx := context.Background()
	planner := &baselineArmTestPlanner{toolName: "Read"}

	res, _, err := Run(ctx, planner, "read test.txt", 4, WithToolCatalog(cat))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	baseM := res.Baseline
	if baseM.ToolErrors != 0 {
		t.Errorf("baseM.ToolErrors = %d, want 0", baseM.ToolErrors)
	}
	if baseM.ToolCalls <= 0 {
		t.Errorf("baseM.ToolCalls = %d, want > 0", baseM.ToolCalls)
	}

	if len(planner.toolOutputs) < 2 {
		t.Fatalf("expected at least 2 tool outputs recorded, got %d", len(planner.toolOutputs))
	}
	baselineToolOutput := planner.toolOutputs[1]
	if !strings.Contains(baselineToolOutput, testContent) {
		t.Errorf("baseline tool output does not contain test.txt content: %q", baselineToolOutput)
	}
	if strings.Contains(baselineToolOutput, "unknown tool") {
		t.Errorf("baseline tool output contains unknown tool error: %q", baselineToolOutput)
	}

	fakM := res.Fak
	if fakM.ToolErrors != 0 {
		t.Errorf("fakM.ToolErrors = %d, want 0", fakM.ToolErrors)
	}
	if fakM.ToolCalls <= 0 {
		t.Errorf("fakM.ToolCalls = %d, want > 0", fakM.ToolCalls)
	}
	fakToolOutput := planner.toolOutputs[0]
	if !strings.Contains(fakToolOutput, testContent) {
		t.Errorf("fak tool output does not contain test.txt content: %q", fakToolOutput)
	}

	t.Run("lowercase tool name normalization", func(t *testing.T) {
		lowerPlanner := &baselineArmTestPlanner{toolName: "read"}
		resLower, _, errLower := Run(ctx, lowerPlanner, "read test.txt", 4, WithToolCatalog(cat))
		if errLower != nil {
			t.Fatalf("Run lowercase: %v", errLower)
		}
		if resLower.Baseline.ToolErrors != 0 {
			t.Errorf("resLower.Baseline.ToolErrors = %d, want 0", resLower.Baseline.ToolErrors)
		}
		if resLower.Baseline.ToolCalls <= 0 {
			t.Errorf("resLower.Baseline.ToolCalls = %d, want > 0", resLower.Baseline.ToolCalls)
		}
		if len(lowerPlanner.toolOutputs) < 2 {
			t.Fatalf("expected at least 2 tool outputs, got %d", len(lowerPlanner.toolOutputs))
		}
		if !strings.Contains(lowerPlanner.toolOutputs[1], testContent) {
			t.Errorf("lowercase baseline output missing test content: %q", lowerPlanner.toolOutputs[1])
		}
	})
}
