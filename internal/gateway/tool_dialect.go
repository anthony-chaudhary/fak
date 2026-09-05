package gateway

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// ToolDialect classifies the client-side tool namespace and prefix convention
// (e.g. Claude Code MCP, OpenAI functions, OpenCode, Colon MCP, or Bare).
type ToolDialect int

const (
	DialectBare ToolDialect = iota
	DialectClaudeMCPCanonical
	DialectClaudeMCP
	DialectOpenAIFunc
	DialectOpenCode
	DialectColonMCP
)

func (d ToolDialect) String() string {
	switch d {
	case DialectClaudeMCPCanonical:
		return "claude_mcp_canonical"
	case DialectClaudeMCP:
		return "claude_mcp"
	case DialectOpenAIFunc:
		return "openai_func"
	case DialectOpenCode:
		return "opencode"
	case DialectColonMCP:
		return "colon_mcp"
	default:
		return "bare"
	}
}

var knownCanonicalTools = map[string]bool{
	"fak_context_restore": true,
	"fak_context_spans":   true,
	"fak_context_value":   true,
	"fak_resume_history":  true,
	"fak_read":            true,
	"fak_syscall":         true,
	"fak_adjudicate":      true,
	"fak_admit":           true,
	"fak_changes":         true,
	"fak_revoke":          true,
	"fak_session_reset":   true,
	"fak_context_change":  true,
	"fak_memory_drivers":  true,
	"fak_memory_explain":  true,
	"fak_memory_run":      true,
	"fak_tools_search":    true,
	"fak_feature_query":   true,
	"fak_capabilities":    true,
	"fak_arch_check":      true,
	"fak_index":           true,
	"fak_unlease":         true,
	"fak_translate":       true,
	"fak_lattice_search":  true,
	"fak_index_graph":     true,
	"exec_command":        true,
	"read_file":           true,
	"write_file":          true,
	"edit_file":           true,
	"apply_patch":         true,
	"view_file":           true,
	"run_test":            true,
	"dos_verify":          true,
}

func init() {
	for _, td := range toolDescriptors() {
		if name, ok := td["name"].(string); ok && name != "" {
			knownCanonicalTools[name] = true
		}
	}
}

// NormalizeToolDialect decomposes a client wire tool name into its canonical tool name,
// optional server name, and detected dialect.
func NormalizeToolDialect(wireName string) (canonical string, server string, dialect ToolDialect) {
	wireName = strings.TrimSpace(wireName)
	if wireName == "" {
		return "", "", DialectBare
	}

	// 1. OpenAI / Codex prefix: "functions.<tool>"
	if strings.HasPrefix(wireName, "functions.") {
		canonical = strings.TrimPrefix(wireName, "functions.")
		return canonical, "", DialectOpenAIFunc
	}

	// 2. Claude Code MCP canonical guard prefix: "mcp__fak_guard__<tool>"
	if strings.HasPrefix(wireName, "mcp__fak_guard__") {
		canonical = strings.TrimPrefix(wireName, "mcp__fak_guard__")
		return canonical, "fak_guard", DialectClaudeMCPCanonical
	}

	// 3. Claude Code MCP generic prefix: "mcp__<server>__<tool>"
	if strings.HasPrefix(wireName, "mcp__") {
		rest := strings.TrimPrefix(wireName, "mcp__")
		if idx := strings.Index(rest, "__"); idx > 0 && idx+2 < len(rest) {
			server = rest[:idx]
			canonical = rest[idx+2:]
			if server == "fak_guard" {
				return canonical, server, DialectClaudeMCPCanonical
			}
			return canonical, server, DialectClaudeMCP
		}
	}

	// 4. Colon MCP prefix: "mcp:<server>:<tool>"
	if strings.HasPrefix(wireName, "mcp:") {
		rest := strings.TrimPrefix(wireName, "mcp:")
		if idx := strings.Index(rest, ":"); idx > 0 && idx+1 < len(rest) {
			server = rest[:idx]
			canonical = rest[idx+1:]
			return canonical, server, DialectColonMCP
		}
	}

	// 5. OpenCode double fak prefix: "fak_fak_<tool>"
	if strings.HasPrefix(wireName, "fak_fak_") {
		canonical = strings.TrimPrefix(wireName, "fak_")
		return canonical, "fak", DialectOpenCode
	}

	// 6. If already recognized as a known canonical bare tool, return as Bare.
	if knownCanonicalTools[wireName] {
		return wireName, "", DialectBare
	}

	// 7. OpenCode double server prefix: "<server>_<server>_<tool>"
	if idx1 := strings.Index(wireName, "_"); idx1 > 0 {
		srvName := wireName[:idx1]
		rest := wireName[idx1+1:]
		if strings.HasPrefix(rest, srvName+"_") {
			return rest, srvName, DialectOpenCode
		}
	}

	// 8. OpenCode single prefix: "<server>_<tool>" matching known canonical tools.
	// Filter out harness prefixes (allow_, deny_, selfmod_, transform_, witness_).
	for k := range knownCanonicalTools {
		suffix := "_" + k
		if strings.HasSuffix(wireName, suffix) {
			server = strings.TrimSuffix(wireName, suffix)
			if server != "" && !isExcludedServerPrefix(server) {
				return k, server, DialectOpenCode
			}
		}
	}

	return wireName, "", DialectBare
}

func isExcludedServerPrefix(prefix string) bool {
	switch prefix {
	case "allow", "deny", "selfmod", "transform", "witness":
		return true
	default:
		return false
	}
}

// NormalizeToolName strips namespace and dialect prefixes, returning the canonical tool name.
func NormalizeToolName(wireName string) string {
	canonical, _, _ := NormalizeToolDialect(wireName)
	if canonical == "" {
		return strings.TrimSpace(wireName)
	}
	return canonical
}

// FormatToolDialect formats a canonical tool name according to the target ToolDialect.
func FormatToolDialect(canonical string, dialect ToolDialect) string {
	canonical = strings.TrimSpace(canonical)
	if canonical == "" {
		return ""
	}
	switch dialect {
	case DialectClaudeMCPCanonical:
		return "mcp__fak_guard__" + canonical
	case DialectClaudeMCP:
		return "mcp__fak__" + canonical
	case DialectOpenAIFunc:
		return "functions." + canonical
	case DialectOpenCode:
		return "fak_" + canonical
	case DialectColonMCP:
		return "mcp:fak:" + canonical
	default:
		return canonical
	}
}

// DetectToolDialectFromNames determines the predominant tool dialect from a slice of tool names.
func DetectToolDialectFromNames(names []string) ToolDialect {
	for _, name := range names {
		name = strings.TrimSpace(name)
		if strings.HasPrefix(name, "mcp__fak_guard__") {
			return DialectClaudeMCPCanonical
		}
	}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if strings.HasPrefix(name, "mcp__") {
			return DialectClaudeMCP
		}
	}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if strings.HasPrefix(name, "mcp:") {
			return DialectColonMCP
		}
	}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if strings.HasPrefix(name, "functions.") {
			return DialectOpenAIFunc
		}
	}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if _, _, d := NormalizeToolDialect(name); d == DialectOpenCode {
			return DialectOpenCode
		}
	}
	return DialectBare
}

// DetectToolDialect inspects a slice of agent.ToolDef and determines the tool dialect.
func DetectToolDialect(tools []agent.ToolDef) ToolDialect {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name != "" {
			names = append(names, t.Function.Name)
		}
	}
	return DetectToolDialectFromNames(names)
}

func messageReferencesRestore(content string) bool {
	return strings.Contains(content, "fak_context_restore") ||
		strings.Contains(content, "recover original via") ||
		strings.Contains(content, "[fak: tool output elided") ||
		strings.Contains(content, "fak_context_restore id=")
}

func messagesReferenceRestore(messages []agent.Message) bool {
	for _, m := range messages {
		if messageReferencesRestore(m.Content) {
			return true
		}
		for _, tc := range m.ToolCalls {
			if NormalizeToolName(tc.Function.Name) == "fak_context_restore" {
				return true
			}
		}
	}
	return false
}

func messagesReferenceSpans(messages []agent.Message) bool {
	for _, m := range messages {
		if strings.Contains(m.Content, "fak_context_spans") {
			return true
		}
		for _, tc := range m.ToolCalls {
			if NormalizeToolName(tc.Function.Name) == "fak_context_spans" {
				return true
			}
		}
	}
	return false
}

func detectDialectFromMessages(messages []agent.Message) ToolDialect {
	for _, m := range messages {
		if strings.Contains(m.Content, "mcp__fak_guard__") {
			return DialectClaudeMCPCanonical
		}
		if strings.Contains(m.Content, "mcp__") {
			return DialectClaudeMCP
		}
		if strings.Contains(m.Content, "mcp:fak:") {
			return DialectColonMCP
		}
		if strings.Contains(m.Content, "functions.fak_context_restore") {
			return DialectOpenAIFunc
		}
		if strings.Contains(m.Content, "fak_fak_context_restore") {
			return DialectOpenCode
		}
	}
	return DialectBare
}

func findFloorToolDef(name string) (agent.ToolDef, bool) {
	for _, t := range MCPFloorToolDefs() {
		if t.Function.Name == name {
			return t, true
		}
	}
	return agent.ToolDef{}, false
}

// SelfHealToolDefs ensures required kernel tool definitions (e.g. fak_context_restore,
// fak_context_spans) are present in the tools list when referenced in messages or when
// elision is active. Injected tools are formatted matching the detected dialect.
func SelfHealToolDefs(tools []agent.ToolDef, messages []agent.Message, forceRestore bool) []agent.ToolDef {
	if len(tools) == 0 && !forceRestore {
		return tools
	}
	hasRestore := false
	for _, t := range tools {
		if NormalizeToolName(t.Function.Name) == "fak_context_restore" {
			hasRestore = true
			break
		}
	}

	needsRestore := forceRestore
	if !needsRestore {
		needsRestore = messagesReferenceRestore(messages)
	}

	dialect := DetectToolDialect(tools)
	if dialect == DialectBare && len(tools) == 0 {
		dialect = detectDialectFromMessages(messages)
	}

	out := append([]agent.ToolDef(nil), tools...)

	if !hasRestore && needsRestore {
		toolName := FormatToolDialect("fak_context_restore", dialect)
		restoreDef, ok := findFloorToolDef("fak_context_restore")
		if !ok {
			restoreDef = agent.ToolDef{
				Type: "function",
				Function: agent.ToolDefFunction{
					Name:        toolName,
					Description: "Restore dropped context by content-addressed sha256 id. Returns verbatim stashed bytes plus orientation; optional trace_id defaults to the current guarded session. Read-only and trust-gated.",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"id":{"type":"string","description":"the content-address handle (sha256 hex) a compaction tombstone embedded as id=<hex>, or a recall page digest"},"trace_id":{"type":"string","description":"session trace id; omitted uses the gateway default trace"}},"required":["id"]}`),
				},
			}
		} else {
			restoreDef.Function.Name = toolName
		}
		out = append(out, restoreDef)
	}

	// Self-heal context spans when referenced in messages
	if messagesReferenceSpans(messages) {
		hasSpans := false
		for _, t := range out {
			if NormalizeToolName(t.Function.Name) == "fak_context_spans" {
				hasSpans = true
				break
			}
		}
		if !hasSpans {
			spansName := FormatToolDialect("fak_context_spans", dialect)
			spansDef, ok := findFloorToolDef("fak_context_spans")
			if !ok {
				spansDef = agent.ToolDef{
					Type: "function",
					Function: agent.ToolDefFunction{
						Name:        spansName,
						Description: "List dropped context spans available to fak_context_restore. Returns safe metadata (id, excerpt, bytes, evidence edges, suppression, restorable) without content or paging; sealed or tombstoned spans remain listed but not restorable. Optional trace_id defaults to the current guarded session; unknown traces return Count 0.",
						Parameters:  json.RawMessage(`{"type":"object","properties":{"trace_id":{"type":"string","description":"session trace id; omitted uses the gateway default trace"}},"additionalProperties":false}`),
					},
				}
			} else {
				spansDef.Function.Name = spansName
			}
			out = append(out, spansDef)
		}
	}

	return out
}
