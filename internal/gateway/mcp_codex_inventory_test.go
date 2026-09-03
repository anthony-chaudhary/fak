package gateway

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestResponsesToolsDeduplicateMCPNamespaces(t *testing.T) {
	inbound := []responsesTool{
		{
			Type:        "function",
			Name:        "mcp__fak__fak_read",
			Description: "read tool via mcp fak",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
		{
			Type:        "function",
			Name:        "mcp__fak_guard__fak_read",
			Description: "read tool via mcp fak_guard",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
	}

	defs := responsesToolsToToolDefs(inbound)
	if len(defs) != 1 {
		t.Fatalf("expected exactly 1 tool def, got %d: %+v", len(defs), defs)
	}
	if defs[0].Function.Name != "mcp__fak_guard__fak_read" {
		t.Fatalf("expected tool name mcp__fak_guard__fak_read, got %q", defs[0].Function.Name)
	}
	if defs[0].Function.Description != "read tool via mcp fak_guard" {
		t.Errorf("expected canonical fak_guard description to be preserved, got %q", defs[0].Function.Description)
	}

	// Also verify reverse order collapses identically into canonical fak_guard.
	reverse := []responsesTool{inbound[1], inbound[0]}
	defsRev := responsesToolsToToolDefs(reverse)
	if len(defsRev) != 1 {
		t.Fatalf("expected exactly 1 tool def for reversed input, got %d: %+v", len(defsRev), defsRev)
	}
	if defsRev[0].Function.Name != "mcp__fak_guard__fak_read" {
		t.Fatalf("expected tool name mcp__fak_guard__fak_read for reversed input, got %q", defsRev[0].Function.Name)
	}
}

func TestCodexGuardedToolInventoryNoDuplicates(t *testing.T) {
	goldenInventory := []responsesTool{
		{Type: "function", Name: "exec_command", Description: "Run shell command"},
		{Type: "function", Name: "apply_patch", Description: "Apply file modifications"},
		{Type: "web_search"},
		// Guard MCP tools
		{Type: "function", Name: "mcp__fak_guard__fak_adjudicate", Description: "Adjudicate tool call"},
		{Type: "function", Name: "mcp__fak_guard__fak_admit", Description: "Admit tool call"},
		{Type: "function", Name: "mcp__fak_guard__fak_syscall", Description: "Kernel syscall"},
		{Type: "function", Name: "mcp__fak_guard__fak_read", Description: "Read files"},
		{Type: "function", Name: "mcp__fak_guard__fak_changes", Description: "Inspect working tree changes"},
		{Type: "function", Name: "mcp__fak_guard__fak_memory_drivers", Description: "List memory drivers"},
		{Type: "function", Name: "mcp__fak_guard__fak_memory_explain", Description: "Explain memory"},
		{Type: "function", Name: "mcp__fak_guard__fak_memory_run", Description: "Run memory driver"},
		{Type: "function", Name: "mcp__fak_guard__fak_trajquery", Description: "Query trajectory"},
		{Type: "function", Name: "mcp__fak_guard__fak_tools_search", Description: "Search tools"},
		{Type: "function", Name: "mcp__fak_guard__fak_feature_query", Description: "Query features"},
		{Type: "function", Name: "mcp__fak_guard__fak_capabilities", Description: "Query capabilities"},
		{Type: "function", Name: "mcp__fak_guard__fak_context_value", Description: "Query context value"},
		{Type: "function", Name: "mcp__fak_guard__fak_context_spans", Description: "Query context spans"},
		{Type: "function", Name: "mcp__fak_guard__fak_context_restore", Description: "Restore context"},
		{Type: "function", Name: "mcp__fak_guard__fak_resume_history", Description: "Query resume history"},
		// Duplicate definitions under mcp__fak__ namespace
		{Type: "function", Name: "mcp__fak__fak_adjudicate", Description: "Duplicate adjudicate"},
		{Type: "function", Name: "mcp__fak__fak_admit", Description: "Duplicate admit"},
		{Type: "function", Name: "mcp__fak__fak_syscall", Description: "Duplicate syscall"},
		{Type: "function", Name: "mcp__fak__fak_read", Description: "Duplicate read"},
		{Type: "function", Name: "mcp__fak__fak_changes", Description: "Duplicate changes"},
		{Type: "function", Name: "mcp__fak__fak_memory_drivers", Description: "Duplicate memory drivers"},
		{Type: "function", Name: "mcp__fak__fak_memory_explain", Description: "Duplicate memory explain"},
		{Type: "function", Name: "mcp__fak__fak_memory_run", Description: "Duplicate memory run"},
		{Type: "function", Name: "mcp__fak__fak_trajquery", Description: "Duplicate trajquery"},
		{Type: "function", Name: "mcp__fak__fak_tools_search", Description: "Duplicate tools search"},
		{Type: "function", Name: "mcp__fak__fak_feature_query", Description: "Duplicate feature query"},
		{Type: "function", Name: "mcp__fak__fak_capabilities", Description: "Duplicate capabilities"},
		{Type: "function", Name: "mcp__fak__fak_context_value", Description: "Duplicate context value"},
		{Type: "function", Name: "mcp__fak__fak_context_spans", Description: "Duplicate context spans"},
		{Type: "function", Name: "mcp__fak__fak_context_restore", Description: "Duplicate context restore"},
		{Type: "function", Name: "mcp__fak__fak_resume_history", Description: "Duplicate resume history"},
		// Standalone tool only under mcp__fak__
		{Type: "function", Name: "mcp__fak__standalone_tool", Description: "Standalone tool"},
		// Third-party tool
		{Type: "function", Name: "mcp__github__create_issue", Description: "Create GitHub issue"},
		// Exact duplicate name entry
		{Type: "function", Name: "exec_command", Description: "Duplicate exec_command"},
	}

	defs := responsesToolsToToolDefs(goldenInventory)

	seen := make(map[string]int)
	var gotNames []string
	for _, d := range defs {
		if d.Type == "function" {
			seen[d.Function.Name]++
			gotNames = append(gotNames, d.Function.Name)
		}
	}

	for name, count := range seen {
		if count > 1 {
			t.Errorf("duplicate tool name %q appeared %d times in tool defs", name, count)
		}
		if strings.HasPrefix(name, mcpFakPrefix) {
			guardName := mcpFakGuardPrefix + strings.TrimPrefix(name, mcpFakPrefix)
			if seen[guardName] > 0 {
				t.Errorf("tool %q was not suppressed despite canonical %q being present", name, guardName)
			}
		}
	}

	goldenExpectedNames := []string{
		"exec_command",
		"apply_patch",
		"mcp__fak_guard__fak_adjudicate",
		"mcp__fak_guard__fak_admit",
		"mcp__fak_guard__fak_syscall",
		"mcp__fak_guard__fak_read",
		"mcp__fak_guard__fak_changes",
		"mcp__fak_guard__fak_memory_drivers",
		"mcp__fak_guard__fak_memory_explain",
		"mcp__fak_guard__fak_memory_run",
		"mcp__fak_guard__fak_trajquery",
		"mcp__fak_guard__fak_tools_search",
		"mcp__fak_guard__fak_feature_query",
		"mcp__fak_guard__fak_capabilities",
		"mcp__fak_guard__fak_context_value",
		"mcp__fak_guard__fak_context_spans",
		"mcp__fak_guard__fak_context_restore",
		"mcp__fak_guard__fak_resume_history",
		"mcp__fak__standalone_tool",
		"mcp__github__create_issue",
	}

	if !reflect.DeepEqual(gotNames, goldenExpectedNames) {
		t.Fatalf("golden tool def names mismatch:\ngot:  %v\nwant: %v", gotNames, goldenExpectedNames)
	}
}
