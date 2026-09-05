package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestToolDialect_Normalization(t *testing.T) {
	tests := []struct {
		wireName      string
		wantCanonical string
		wantServer    string
		wantDialect   ToolDialect
	}{
		// Claude MCP canonical
		{
			wireName:      "mcp__fak_guard__fak_context_restore",
			wantCanonical: "fak_context_restore",
			wantServer:    "fak_guard",
			wantDialect:   DialectClaudeMCPCanonical,
		},
		{
			wireName:      "mcp__fak_guard__fak_context_spans",
			wantCanonical: "fak_context_spans",
			wantServer:    "fak_guard",
			wantDialect:   DialectClaudeMCPCanonical,
		},
		{
			wireName:      "mcp__fak_guard__exec_command",
			wantCanonical: "exec_command",
			wantServer:    "fak_guard",
			wantDialect:   DialectClaudeMCPCanonical,
		},

		// Claude MCP standard
		{
			wireName:      "mcp__fak__fak_context_restore",
			wantCanonical: "fak_context_restore",
			wantServer:    "fak",
			wantDialect:   DialectClaudeMCP,
		},
		{
			wireName:      "mcp__custom__read_file",
			wantCanonical: "read_file",
			wantServer:    "custom",
			wantDialect:   DialectClaudeMCP,
		},

		// Colon MCP
		{
			wireName:      "mcp:fak:fak_context_restore",
			wantCanonical: "fak_context_restore",
			wantServer:    "fak",
			wantDialect:   DialectColonMCP,
		},
		{
			wireName:      "mcp:custom_srv:query_db",
			wantCanonical: "query_db",
			wantServer:    "custom_srv",
			wantDialect:   DialectColonMCP,
		},

		// OpenAI func
		{
			wireName:      "functions.fak_context_restore",
			wantCanonical: "fak_context_restore",
			wantServer:    "",
			wantDialect:   DialectOpenAIFunc,
		},
		{
			wireName:      "functions.exec_command",
			wantCanonical: "exec_command",
			wantServer:    "",
			wantDialect:   DialectOpenAIFunc,
		},

		// OpenCode double
		{
			wireName:      "fak_fak_context_restore",
			wantCanonical: "fak_context_restore",
			wantServer:    "fak",
			wantDialect:   DialectOpenCode,
		},
		{
			wireName:      "fak_fak_read",
			wantCanonical: "fak_read",
			wantServer:    "fak",
			wantDialect:   DialectOpenCode,
		},
		{
			wireName:      "srv_srv_my_tool",
			wantCanonical: "srv_my_tool",
			wantServer:    "srv",
			wantDialect:   DialectOpenCode,
		},

		// OpenCode single matching known tools
		{
			wireName:      "bash_exec_command",
			wantCanonical: "exec_command",
			wantServer:    "bash",
			wantDialect:   DialectOpenCode,
		},

		// Bare tools
		{
			wireName:      "fak_context_restore",
			wantCanonical: "fak_context_restore",
			wantServer:    "",
			wantDialect:   DialectBare,
		},
		{
			wireName:      "fak_context_spans",
			wantCanonical: "fak_context_spans",
			wantServer:    "",
			wantDialect:   DialectBare,
		},
		{
			wireName:      "exec_command",
			wantCanonical: "exec_command",
			wantServer:    "",
			wantDialect:   DialectBare,
		},

		// Unregistered tool
		{
			wireName:      "unregistered_tool",
			wantCanonical: "unregistered_tool",
			wantServer:    "",
			wantDialect:   DialectBare,
		},
		{
			wireName:      "",
			wantCanonical: "",
			wantServer:    "",
			wantDialect:   DialectBare,
		},
	}

	for _, tc := range tests {
		t.Run(tc.wireName, func(t *testing.T) {
			canonical, server, dialect := NormalizeToolDialect(tc.wireName)
			if canonical != tc.wantCanonical {
				t.Errorf("canonical = %q, want %q", canonical, tc.wantCanonical)
			}
			if server != tc.wantServer {
				t.Errorf("server = %q, want %q", server, tc.wantServer)
			}
			if dialect != tc.wantDialect {
				t.Errorf("dialect = %v, want %v", dialect, tc.wantDialect)
			}

			norm := NormalizeToolName(tc.wireName)
			if norm != tc.wantCanonical {
				t.Errorf("NormalizeToolName(%q) = %q, want %q", tc.wireName, norm, tc.wantCanonical)
			}
		})
	}
}

func TestToolDialect_Formatting(t *testing.T) {
	tests := []struct {
		canonical string
		dialect   ToolDialect
		want      string
	}{
		{"fak_context_restore", DialectClaudeMCPCanonical, "mcp__fak_guard__fak_context_restore"},
		{"fak_context_restore", DialectClaudeMCP, "mcp__fak__fak_context_restore"},
		{"fak_context_restore", DialectOpenAIFunc, "functions.fak_context_restore"},
		{"fak_context_restore", DialectOpenCode, "fak_fak_context_restore"},
		{"fak_context_restore", DialectColonMCP, "mcp:fak:fak_context_restore"},
		{"fak_context_restore", DialectBare, "fak_context_restore"},

		{"fak_context_spans", DialectClaudeMCPCanonical, "mcp__fak_guard__fak_context_spans"},
		{"fak_context_spans", DialectClaudeMCP, "mcp__fak__fak_context_spans"},
		{"fak_context_spans", DialectOpenAIFunc, "functions.fak_context_spans"},
		{"fak_context_spans", DialectOpenCode, "fak_fak_context_spans"},
		{"fak_context_spans", DialectColonMCP, "mcp:fak:fak_context_spans"},
		{"fak_context_spans", DialectBare, "fak_context_spans"},
	}

	for _, tc := range tests {
		got := FormatToolDialect(tc.canonical, tc.dialect)
		if got != tc.want {
			t.Errorf("FormatToolDialect(%q, %v) = %q, want %q", tc.canonical, tc.dialect, got, tc.want)
		}
	}
}

func TestToolDialect_Detection(t *testing.T) {
	tests := []struct {
		name  string
		tools []string
		want  ToolDialect
	}{
		{"empty", nil, DialectBare},
		{"bare tools", []string{"read_file", "write_file"}, DialectBare},
		{"claude canonical", []string{"mcp__fak_guard__bash", "read"}, DialectClaudeMCPCanonical},
		{"claude standard", []string{"mcp__custom__read_file"}, DialectClaudeMCP},
		{"colon mcp", []string{"mcp:fak:exec_command"}, DialectColonMCP},
		{"openai func", []string{"functions.exec_command"}, DialectOpenAIFunc},
		{"opencode double", []string{"fak_fak_read"}, DialectOpenCode},
		{"opencode single", []string{"bash_exec_command"}, DialectOpenCode},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := DetectToolDialectFromNames(tc.tools)
			if got != tc.want {
				t.Errorf("DetectToolDialectFromNames(%v) = %v, want %v", tc.tools, got, tc.want)
			}
			var toolDefs []agent.ToolDef
			for _, n := range tc.tools {
				toolDefs = append(toolDefs, agent.ToolDef{
					Type: "function",
					Function: agent.ToolDefFunction{
						Name: n,
					},
				})
			}
			gotDefs := DetectToolDialect(toolDefs)
			if gotDefs != tc.want {
				t.Errorf("DetectToolDialect(%v) = %v, want %v", tc.tools, gotDefs, tc.want)
			}
		})
	}
}

func TestSelfHealToolDefs(t *testing.T) {
	// 1. Tool already present: no duplicate added
	t.Run("no duplicate when already present", func(t *testing.T) {
		tools := []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "mcp__fak_guard__fak_context_restore"}},
		}
		res := SelfHealToolDefs(tools, nil, true)
		if len(res) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(res))
		}
	})

	// 2. Force restore with ClaudeMCPCanonical dialect
	t.Run("inject with claude canonical dialect", func(t *testing.T) {
		tools := []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "mcp__fak_guard__bash"}},
		}
		res := SelfHealToolDefs(tools, nil, true)
		if len(res) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(res))
		}
		injected := res[1]
		if injected.Function.Name != "mcp__fak_guard__fak_context_restore" {
			t.Errorf("injected name = %q, want mcp__fak_guard__fak_context_restore", injected.Function.Name)
		}
		if !strings.Contains(injected.Function.Description, "Restore dropped context") {
			t.Errorf("expected restore description, got %q", injected.Function.Description)
		}
	})

	// 3. Force restore with ColonMCP dialect
	t.Run("inject with colon mcp dialect", func(t *testing.T) {
		tools := []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "mcp:fak:exec_command"}},
		}
		res := SelfHealToolDefs(tools, nil, true)
		if len(res) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(res))
		}
		if res[1].Function.Name != "mcp:fak:fak_context_restore" {
			t.Errorf("injected name = %q, want mcp:fak:fak_context_restore", res[1].Function.Name)
		}
	})

	// 4. Force restore with OpenCode dialect
	t.Run("inject with opencode dialect", func(t *testing.T) {
		tools := []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "fak_fak_read"}},
		}
		res := SelfHealToolDefs(tools, nil, true)
		if len(res) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(res))
		}
		if res[1].Function.Name != "fak_fak_context_restore" {
			t.Errorf("injected name = %q, want fak_fak_context_restore", res[1].Function.Name)
		}
	})

	// 5. Detect from message reference when forceRestore is false
	t.Run("detect from message reference", func(t *testing.T) {
		tools := []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "functions.exec_command"}},
		}
		msgs := []agent.Message{
			{Role: agent.RoleTool, Content: "...[fak: tool output elided; recover original via fak_context_restore id=sha256:abcd]..."},
		}
		res := SelfHealToolDefs(tools, msgs, false)
		if len(res) != 2 {
			t.Fatalf("expected 2 tools, got %d", len(res))
		}
		if res[1].Function.Name != "functions.fak_context_restore" {
			t.Errorf("injected name = %q, want functions.fak_context_restore", res[1].Function.Name)
		}
	})

	// 6. Detect spans tool reference in messages
	t.Run("detect spans reference", func(t *testing.T) {
		tools := []agent.ToolDef{
			{Type: "function", Function: agent.ToolDefFunction{Name: "mcp__fak_guard__bash"}},
		}
		msgs := []agent.Message{
			{Role: agent.RoleUser, Content: "please call fak_context_spans to see dropped tasks"},
		}
		res := SelfHealToolDefs(tools, msgs, false)
		var foundSpans bool
		for _, tDef := range res {
			if NormalizeToolName(tDef.Function.Name) == "fak_context_spans" {
				foundSpans = true
				if tDef.Function.Name != "mcp__fak_guard__fak_context_spans" {
					t.Errorf("expected dialect formatted name mcp__fak_guard__fak_context_spans, got %q", tDef.Function.Name)
				}
			}
		}
		if !foundSpans {
			t.Fatalf("fak_context_spans not injected: %+v", res)
		}
	})
}
