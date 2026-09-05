package agent

import (
	"strings"
	"testing"
)

func TestSeedSystemPromptCodingToolsFallback(t *testing.T) {
	DisarmCodeTools()
	defer DisarmCodeTools()

	t.Run("no system prompt and no coding tools: returns SystemPrompt", func(t *testing.T) {
		DisarmCodeTools()
		cfg := resolveRunConfig(nil)
		got := cfg.seedSystemPrompt()
		if !strings.Contains(got, SystemPrompt) {
			t.Fatalf("expected SystemPrompt in %q", got)
		}
		if strings.Contains(got, CodeAgentSystemPrompt) {
			t.Fatalf("did not expect CodeAgentSystemPrompt in %q", got)
		}
		if !strings.Contains(ownedAgentSystemPrompt(), SystemPrompt) {
			t.Fatalf("expected ownedAgentSystemPrompt to contain SystemPrompt")
		}
	})

	t.Run("coding tools armed via ArmCodeTools: returns CodeAgentSystemPrompt", func(t *testing.T) {
		_, err := ArmCodeTools(t.TempDir())
		if err != nil {
			t.Fatalf("ArmCodeTools: %v", err)
		}
		defer DisarmCodeTools()

		cfg := resolveRunConfig(nil)
		got := cfg.seedSystemPrompt()
		if !strings.Contains(got, CodeAgentSystemPrompt) {
			t.Fatalf("expected CodeAgentSystemPrompt in %q", got)
		}
		if strings.Contains(got, SystemPrompt) {
			t.Fatalf("did not expect SystemPrompt in %q", got)
		}
		if !strings.Contains(ownedAgentSystemPrompt(), CodeAgentSystemPrompt) {
			t.Fatalf("expected ownedAgentSystemPrompt to contain CodeAgentSystemPrompt")
		}
	})

	t.Run("coding tools provided via WithToolCatalog: returns CodeAgentSystemPrompt", func(t *testing.T) {
		DisarmCodeTools()
		tools := []ToolDef{
			{Function: ToolDefFunction{Name: "Read"}},
		}
		cfg := resolveRunConfig([]RunOption{WithToolCatalog(tools)})
		got := cfg.seedSystemPrompt()
		if !strings.Contains(got, CodeAgentSystemPrompt) {
			t.Fatalf("expected CodeAgentSystemPrompt in %q", got)
		}
		if strings.Contains(got, SystemPrompt) {
			t.Fatalf("did not expect SystemPrompt in %q", got)
		}
	})

	t.Run("WithSystemPrompt provided: returns custom prompt even if coding tools are armed", func(t *testing.T) {
		_, err := ArmCodeTools(t.TempDir())
		if err != nil {
			t.Fatalf("ArmCodeTools: %v", err)
		}
		defer DisarmCodeTools()

		custom := "custom prompt"
		cfg := resolveRunConfig([]RunOption{WithSystemPrompt(custom)})
		got := cfg.seedSystemPrompt()
		if !strings.Contains(got, custom) {
			t.Fatalf("expected custom prompt in %q", got)
		}
		if strings.Contains(got, CodeAgentSystemPrompt) {
			t.Fatalf("did not expect CodeAgentSystemPrompt in %q", got)
		}
		if strings.Contains(got, SystemPrompt) {
			t.Fatalf("did not expect SystemPrompt in %q", got)
		}
	})
}
