package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAgentMCPFlagDefaultsOn(t *testing.T) {
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse empty args: %v", err)
	}
	if af.mcpTools == nil || !*af.mcpTools {
		t.Fatal("--mcp-tools defaulted off for agent")
	}

	fs2, af2 := newAgentFlagSet()
	if err := fs2.Parse([]string{"--mcp-tools=false"}); err != nil {
		t.Fatalf("Parse --mcp-tools=false: %v", err)
	}
	if af2.mcpTools == nil || *af2.mcpTools {
		t.Fatal("--mcp-tools=false did not disable for agent")
	}
}

func TestAgentMCPChatFlagDefaultsOn(t *testing.T) {
	fs, cf := newChatFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse empty args: %v", err)
	}
	if cf.mcpTools == nil || !*cf.mcpTools {
		t.Fatal("--mcp-tools defaulted off for chat")
	}

	fs2, cf2 := newChatFlagSet()
	if err := fs2.Parse([]string{"--mcp-tools=false"}); err != nil {
		t.Fatalf("Parse --mcp-tools=false: %v", err)
	}
	if cf2.mcpTools == nil || *cf2.mcpTools {
		t.Fatal("--mcp-tools=false did not disable for chat")
	}
}

func TestAgentMCPArmingAddsToCatalog(t *testing.T) {
	mcpCatalog, err := agent.ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools: %v", err)
	}
	defer agent.DisarmMCPTools()

	found := make(map[string]bool)
	for _, tool := range mcpCatalog {
		found[tool.Function.Name] = true
	}

	for _, want := range []string{"fak_read", "fak_tools_search", "fak_adjudicate", "fak_syscall", "fak_capabilities"} {
		if !found[want] {
			t.Errorf("expected %q in armed MCP catalog, got %v", want, found)
		}
	}
}
