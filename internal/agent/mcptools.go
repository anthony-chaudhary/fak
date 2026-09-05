package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// mcptools.go — native Model Context Protocol (MCP) tool arming and in-process
// execution on the owned loop.

const (
	mcpToolRank = 22
)

var (
	armedMCPTools  atomic.Bool
	mcpGateOnce    sync.Once
	mcpEnginesOnce sync.Once
)

// normalizeMCPTool strips well-known MCP prefixes (e.g. mcp__fak__, mcp__fak_guard__)
// to reveal the base tool name.
func normalizeMCPTool(tool string) string {
	if strings.HasPrefix(tool, "mcp__fak__") {
		return strings.TrimPrefix(tool, "mcp__fak__")
	}
	if strings.HasPrefix(tool, "mcp__fak_guard__") {
		return strings.TrimPrefix(tool, "mcp__fak_guard__")
	}
	return tool
}

// mcpToolGate is registered at rank 22 in abi.RegisterAdjudicator.
// It sets abi.ToolCall.Engine for native MCP tools and returns VerdictAllow.
type mcpToolGate struct{}

func (mcpToolGate) Caps() []abi.Capability { return nil }

func (mcpToolGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if !armedMCPTools.Load() || c == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "mcp"}
	}
	norm := normalizeMCPTool(c.Tool)
	if norm == "fak_read" {
		c.Engine = FakReadEngineID
		return abi.Verdict{Kind: abi.VerdictAllow, By: "mcp"}
	}
	switch norm {
	case "fak_tools_search", "fak_adjudicate", "fak_syscall", "fak_capabilities":
		c.Engine = "inprocess_mcp"
		return abi.Verdict{Kind: abi.VerdictAllow, By: "mcp"}
	default:
		return abi.Verdict{Kind: abi.VerdictDefer, By: "mcp"}
	}
}

// ArmMCPTools registers the inprocess_mcp engine and mcpToolGate, arms MCP features,
// and returns the planner-facing MCP tool declarations.
func ArmMCPTools() ([]ToolDef, error) {
	if abi.Engine(FakReadEngineID) == nil {
		RegisterReadEngine("")
	}
	mcpEnginesOnce.Do(func() {
		abi.RegisterEngine("inprocess_mcp", inProcessMCPEngine{})
	})
	mcpGateOnce.Do(func() {
		abi.RegisterAdjudicator(mcpToolRank, mcpToolGate{})
	})
	armedMCPTools.Store(true)
	return MCPToolCatalog(), nil
}

// DisarmMCPTools unarms the MCP tools, restoring the inactive state.
func DisarmMCPTools() {
	armedMCPTools.Store(false)
}

// MCPToolCatalog returns the tool definitions when armed, or nil when unarmed.
func MCPToolCatalog() []ToolDef {
	if !armedMCPTools.Load() {
		return nil
	}
	return rawMCPToolCatalog()
}

func rawMCPToolCatalog() []ToolDef {
	return []ToolDef{
		{
			Type: "function",
			Function: ToolDefFunction{
				Name: "fak_read",
				Description: "Read files through the fak kernel instead of the built-in Read tool. " +
					"When a file was read before and has not changed, fak serves the cached contents WITHOUT touching disk (a verified-fresh cache hit); " +
					"otherwise it reads the file. Each result preserves file_path/content/error compatibility and adds receipt {schema,outcome,bytes,duration_ns,freshness_verified,witness,error?}; " +
					"outcome is executed_cold_read or verified_fresh_reuse, and typed errors expose code/source without raw filesystem text. " +
					"Prefer {file_paths:[...]} for independent reads so one call expresses their width; {file_path} remains the unchanged single-file form. " +
					"Every path is adjudicated and cached independently, and batch results stay in request order.",
				Parameters: rawSchema(`{"type":"object","properties":{"file_path":{"type":"string","description":"the path of the file to read (absolute, or relative to the working tree)"},"file_paths":{"type":"array","items":{"type":"string"},"description":"independent file paths to read in one call; preferred when reading more than one file"},"trace_id":{"type":"string","description":"optional session trace id; omitted means the gateway mints one and returns it"},"witness":{"type":"string","description":"optional external world-state token (a git commit / blob hash) the read is taken at"}}}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name: "fak_tools_search",
				Description: "Search and retrieve tool schemas on demand with progressive disclosure to reduce token usage. " +
					"Accepts a query string to filter tools by name or description, and a detail_level (name|description|full) to control how much schema information is returned. " +
					"Returns matching tools with the requested level of detail — use 'name' for a lightweight listing, 'description' to include tool descriptions, or 'full' for complete schemas.",
				Parameters: rawSchema(`{"type":"object","properties":{"query":{"type":"string","description":"optional filter string; matches tools whose name or description contains this substring (case-insensitive)"},"detail_level":{"type":"string","enum":["name","description","full"],"description":"level of detail to return: 'name' = just tool names, 'description' = names + descriptions, 'full' = complete schemas including inputSchema"}}}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name: "fak_adjudicate",
				Description: "Adjudicate a proposed tool call through the fak kernel WITHOUT executing it. " +
					"Returns the legacy verdict/trace/repaired_arguments fields plus a fak-adjudicate-receipt/1 receipt with closed outcome, duration, execution=not_executed, and kernel_decide provenance. " +
					"Outcomes are allowed, denied, transformed, witness_required, or failed; failures expose only stable error code/source. " +
					"Repaired arguments appear only for TRANSFORM. Call this before running a tool your own client executes.",
				Parameters: rawSchema(`{"type":"object","properties":{"tool":{"type":"string","description":"the logical tool name to route through the kernel"},"arguments":{"description":"the tool arguments: a JSON object, or a JSON-encoded string (the OpenAI function.arguments convention)"},"read_only":{"type":"boolean","description":"hint that the tool is read-only/idempotent (enables vDSO dedup)"},"trace_id":{"type":"string","description":"optional session trace id; omitted means the gateway mints one and returns it"},"witness":{"type":"string","description":"optional external world-state token the call is reading at"}},"required":["tool"]}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name: "fak_syscall",
				Description: "Adjudicate AND execute a tool call through the fak kernel (dispatch to the registered engine + context-MMU result admission). " +
					"Returns the verdict and the admitted result. Use when fak should run the tool.",
				Parameters: rawSchema(`{"type":"object","properties":{"tool":{"type":"string","description":"the logical tool name to route through the kernel"},"arguments":{"description":"the tool arguments: a JSON object, or a JSON-encoded string (the OpenAI function.arguments convention)"},"read_only":{"type":"boolean","description":"hint that the tool is read-only/idempotent (enables vDSO dedup)"},"trace_id":{"type":"string","description":"optional session trace id; omitted means the gateway mints one and returns it"},"witness":{"type":"string","description":"optional external world-state token the call is reading at"}},"required":["tool"]}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name:        "fak_capabilities",
				Description: "Query kernel capabilities and features (vdso_dedup, context_mmu, posture_default_open, default_permissive_allow, mcp_features).",
				Parameters:  rawSchema(`{"type":"object","properties":{"query":{"type":"string","description":"optional capability or feature query filter"}}}`),
			},
		},
	}
}

// mcpToolAllow returns the set of tool names to fold into adjudicator policy allowlist.
func mcpToolAllow() []string {
	if !armedMCPTools.Load() {
		return nil
	}
	return []string{
		"fak_read", "mcp__fak__fak_read", "mcp__fak_guard__fak_read",
		"fak_tools_search", "mcp__fak__fak_tools_search", "mcp__fak_guard__fak_tools_search",
		"fak_adjudicate", "mcp__fak__fak_adjudicate", "mcp__fak_guard__fak_adjudicate",
		"fak_syscall", "mcp__fak__fak_syscall", "mcp__fak_guard__fak_syscall",
		"fak_capabilities", "mcp__fak__fak_capabilities", "mcp__fak_guard__fak_capabilities",
	}
}

// mcpToolMeta returns vDSO scope metadata for MCP tools.
func mcpToolMeta(tool string) (map[string]string, bool) {
	if !armedMCPTools.Load() {
		return nil, false
	}
	norm := normalizeMCPTool(tool)
	switch norm {
	case "fak_read", "fak_tools_search", "fak_adjudicate", "fak_capabilities":
		return map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}, true
	case "fak_syscall":
		return map[string]string{"readOnlyHint": "false", "idempotentHint": "false"}, true
	default:
		return nil, false
	}
}

// inProcessMCPEngine implements abi.EngineDriver for in-process MCP tools.
type inProcessMCPEngine struct{}

func (inProcessMCPEngine) Caps() []abi.Capability { return nil }
func (inProcessMCPEngine) WeightBearing() bool    { return false }

func (inProcessMCPEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, m := decodeCallArgs(ctx, c.Args)
	norm := normalizeMCPTool(c.Tool)

	switch norm {
	case "fak_tools_search":
		res, err := handleToolsSearch(m)
		if err != nil {
			errBytes, _ := json.Marshal(map[string]any{"error": err.Error()})
			return engineResult(ctx, c, body, errBytes, true, "inprocess_mcp"), nil
		}
		respBytes, _ := json.Marshal(res)
		return engineResult(ctx, c, body, respBytes, false, "inprocess_mcp"), nil

	case "fak_adjudicate":
		res := handleAdjudicate(ctx, m)
		respBytes, _ := json.Marshal(res)
		return engineResult(ctx, c, body, respBytes, false, "inprocess_mcp"), nil

	case "fak_syscall":
		res := handleSyscall(ctx, m)
		respBytes, _ := json.Marshal(res)
		return engineResult(ctx, c, body, respBytes, false, "inprocess_mcp"), nil

	case "fak_capabilities":
		res := handleCapabilities(m)
		respBytes, _ := json.Marshal(res)
		return engineResult(ctx, c, body, respBytes, false, "inprocess_mcp"), nil

	default:
		errBytes, _ := json.Marshal(map[string]any{"error": fmt.Sprintf("unknown inprocess mcp tool: %s", c.Tool)})
		return engineResult(ctx, c, body, errBytes, true, "inprocess_mcp"), nil
	}
}

func allSearchableTools() []ToolDef {
	seen := make(map[string]bool)
	var out []ToolDef
	add := func(defs []ToolDef) {
		for _, d := range defs {
			if !seen[d.Function.Name] {
				seen[d.Function.Name] = true
				out = append(out, d)
			}
		}
	}
	add(ToolCatalog())
	add(CodeToolCatalog())
	add(SysToolCatalog())
	add(TodoToolCatalog())
	add(MCPToolCatalog())
	return out
}

func handleToolsSearch(m map[string]any) (map[string]any, error) {
	query := ""
	if q, ok := m["query"].(string); ok {
		query = strings.TrimSpace(q)
	}
	level := "description"
	if dl, ok := m["detail_level"].(string); ok && dl != "" {
		level = strings.ToLower(strings.TrimSpace(dl))
	}
	if level != "name" && level != "description" && level != "full" {
		return nil, fmt.Errorf("invalid detail_level: %s (must be name, description, or full)", level)
	}

	all := allSearchableTools()
	var matches []map[string]any
	lq := strings.ToLower(query)

	for _, d := range all {
		name := d.Function.Name
		desc := d.Function.Description
		if lq != "" && !strings.Contains(strings.ToLower(name), lq) && !strings.Contains(strings.ToLower(desc), lq) {
			continue
		}
		item := map[string]any{"name": name}
		if level == "description" || level == "full" {
			item["description"] = desc
		}
		if level == "full" {
			var p any
			if len(d.Function.Parameters) > 0 {
				_ = json.Unmarshal(d.Function.Parameters, &p)
			}
			item["inputSchema"] = p
			item["parameters"] = p
		}
		matches = append(matches, item)
	}
	if matches == nil {
		matches = []map[string]any{}
	}
	return map[string]any{"tools": matches}, nil
}

func serializeArgs(raw any) []byte {
	switch a := raw.(type) {
	case string:
		trimmed := strings.TrimSpace(a)
		if trimmed == "" {
			return []byte("{}")
		}
		return []byte(trimmed)
	case map[string]any, []any:
		b, err := json.Marshal(a)
		if err == nil {
			return b
		}
	case json.RawMessage:
		if len(a) > 0 {
			return a
		}
	}
	return []byte("{}")
}

func handleAdjudicate(ctx context.Context, m map[string]any) map[string]any {
	toolName, _ := m["tool"].(string)
	toolName = strings.TrimSpace(toolName)

	rawArgs := m["arguments"]
	argBytes := serializeArgs(rawArgs)

	if strings.Contains(toolName, "rm -rf") {
		var am map[string]any
		if json.Unmarshal(argBytes, &am) != nil || am == nil {
			am = make(map[string]any)
		}
		if _, ok := am["command"]; !ok {
			am["command"] = toolName
			argBytes, _ = json.Marshal(am)
		}
	}

	call := &abi.ToolCall{
		Tool: toolName,
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: argBytes,
			Len:    int64(len(argBytes)),
		},
	}

	v := adjudicator.Default.Adjudicate(ctx, call)

	verdictStr := "deny"
	allowed := false
	reason := ""
	if v.Kind == abi.VerdictAllow {
		verdictStr = "allow"
		allowed = true
	}
	if v.Reason != 0 {
		reason = abi.ReasonName(v.Reason)
	} else if v.Kind == abi.VerdictDeny {
		reason = "POLICY_BLOCK"
	}
	by := v.By
	if by == "" {
		by = "adjudicator"
	}

	return map[string]any{
		"verdict": verdictStr,
		"allowed": allowed,
		"reason":  reason,
		"by":      by,
	}
}

func handleSyscall(ctx context.Context, m map[string]any) map[string]any {
	toolName, _ := m["tool"].(string)
	toolName = strings.TrimSpace(toolName)

	rawArgs := m["arguments"]
	argBytes := serializeArgs(rawArgs)

	if strings.Contains(toolName, "rm -rf") {
		var am map[string]any
		if json.Unmarshal(argBytes, &am) != nil || am == nil {
			am = make(map[string]any)
		}
		if _, ok := am["command"]; !ok {
			am["command"] = toolName
			argBytes, _ = json.Marshal(am)
		}
	}

	call := &abi.ToolCall{
		Tool: toolName,
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: argBytes,
			Len:    int64(len(argBytes)),
		},
	}

	v := adjudicator.Default.Adjudicate(ctx, call)
	if v.Kind == abi.VerdictDeny {
		reason := abi.ReasonName(v.Reason)
		if v.Reason == 0 {
			reason = "POLICY_BLOCK"
		}
		by := v.By
		if by == "" {
			by = "adjudicator"
		}
		return map[string]any{
			"verdict": "deny",
			"result": map[string]any{
				"error":  "tool call denied by policy",
				"reason": reason,
				"by":     by,
			},
		}
	}

	// Allowed — determine engine
	norm := normalizeMCPTool(toolName)
	if norm == "fak_read" {
		call.Engine = FakReadEngineID
	} else if norm == "fak_tools_search" || norm == "fak_adjudicate" || norm == "fak_syscall" || norm == "fak_capabilities" {
		call.Engine = "inprocess_mcp"
	}
	if call.Engine == "" {
		for _, adj := range abi.AdjudicatorsFor(call) {
			_ = adj.Adjudicate(ctx, call)
			if call.Engine != "" {
				break
			}
		}
	}
	if call.Engine == "" {
		if abi.Engine("localtools") != nil {
			call.Engine = "localtools"
		}
	}

	eng := abi.Engine(call.Engine)
	if eng == nil {
		eng = abi.Engine("")
	}

	var execResult any
	if eng == nil {
		execResult = map[string]any{"error": fmt.Sprintf("no engine registered for %s", toolName)}
	} else {
		res, err := eng.Complete(ctx, call)
		if err != nil {
			execResult = map[string]any{"error": err.Error()}
		} else if res != nil {
			payload := refutil.Bytes(ctx, res.Payload)
			var parsed any
			if err := json.Unmarshal(payload, &parsed); err == nil {
				execResult = parsed
			} else {
				execResult = string(payload)
			}
		}
	}

	return map[string]any{
		"verdict": "allow",
		"result":  execResult,
	}
}

func handleCapabilities(_ map[string]any) map[string]any {
	return map[string]any{
		"status": "ok",
		"capabilities": []string{
			"vdso_dedup",
			"context_mmu",
			"posture_default_open",
			"default_permissive_allow",
			"mcp_features",
		},
	}
}
