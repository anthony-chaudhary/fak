package projectassets

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Subagent role constants for OpenCode multi-agent workflows.
const (
	SubagentRoleExplore        = "explore"
	SubagentRoleResearcher     = "researcher"
	SubagentRoleScout          = "scout"
	SubagentRoleWorker         = "worker"
	SubagentRoleDeepReason     = "deep-reason"
	SubagentRoleTester         = "tester"
	SubagentRoleCrossValidator = "cross-validator"
	SubagentRoleReviewer       = "reviewer"
	SubagentRoleIssueAuditor   = "issue-auditor"
	SubagentRoleLander         = "lander"
	SubagentRoleGeneral        = "general"
	SubagentRoleBuild          = "build"
)

// ToolDescriptor represents an MCP tool definition advertised to an agent.
type ToolDescriptor struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Category    string         `json:"category,omitempty"`
	InputSchema map[string]any `json:"inputSchema,omitempty"`
}

// SubagentTokenSavingsReceipt documents token and schema savings from progressive disclosure.
type SubagentTokenSavingsReceipt struct {
	SubagentRole          string   `json:"subagent_role"`
	ToolsBefore           int      `json:"tools_before"`
	ToolsAfter            int      `json:"tools_after"`
	EstimatedTokensBefore int      `json:"estimated_tokens_before"`
	EstimatedTokensAfter  int      `json:"estimated_tokens_after"`
	SavingsPercent        float64  `json:"savings_percent"`
	ReductionTargetMet    bool     `json:"reduction_target_met"`
	AllowedTools          []string `json:"allowed_tools"`
	FilteredOutTools      []string `json:"filtered_out_tools"`
}

// MutationTools defines tools that alter the workspace filesystem.
var MutationTools = map[string]bool{
	"edit":        true,
	"write":       true,
	"apply_patch": true,
}

// ExecutionTools defines tools that execute system commands.
var ExecutionTools = map[string]bool{
	"bash": true,
}

// ReadOnlySearchTools defines safe retrieval tools exposed to exploratory subagents.
var ReadOnlySearchTools = map[string]bool{
	"fak_read":         true,
	"read":             true,
	"grep":             true,
	"glob":             true,
	"fak_tools_search": true,
	"dos_answer":       true,
	"webfetch":         true,
}

// WorkerAllowedTools defines the curated progressive disclosure toolset for implementation workers.
// Unneeded governance, context, and recursive delegation tools are excluded to save >60% prompt tokens.
var WorkerAllowedTools = map[string]bool{
	"edit":        true,
	"write":       true,
	"apply_patch": true,
	"bash":        true,
	"read":        true,
	"fak_read":    true,
	"grep":        true,
	"glob":        true,
	"todowrite":   true,
}

// TesterAllowedTools defines the curated progressive disclosure toolset for testing agents.
// Mutation tools are strictly forbidden while test execution and verification tools are exposed.
var TesterAllowedTools = map[string]bool{
	"bash":             true,
	"read":             true,
	"fak_read":         true,
	"grep":             true,
	"glob":             true,
	"dos_verify":       true,
	"dos_commit_audit": true,
	"todowrite":        true,
}

// NormalizeSubagentRole cleans and standardizes a subagent role identifier.
func NormalizeSubagentRole(role string) string {
	r := strings.ToLower(strings.TrimSpace(role))
	r = strings.TrimPrefix(r, "opencode:")
	r = strings.TrimPrefix(r, "subagent:")
	r = strings.TrimPrefix(r, "agent:")
	return r
}

// IsToolAllowed returns whether toolName is permitted under the capability floor of role.
func IsToolAllowed(role string, toolName string) bool {
	normRole := NormalizeSubagentRole(role)
	normTool := strings.ToLower(strings.TrimSpace(toolName))

	switch normRole {
	case SubagentRoleExplore, SubagentRoleResearcher, SubagentRoleScout:
		// Exploratory agents: read-only search tools only.
		// Strictly forbid file mutations (edit, write, apply_patch) and command execution (bash).
		if MutationTools[normTool] || ExecutionTools[normTool] {
			return false
		}
		return ReadOnlySearchTools[normTool] || normTool == "todowrite"

	case SubagentRoleReviewer, SubagentRoleIssueAuditor:
		// Review/audit agents: read-only analysis without file mutation or arbitrary execution.
		if MutationTools[normTool] || ExecutionTools[normTool] {
			return false
		}
		return ReadOnlySearchTools[normTool] ||
			normTool == "dos_commit_audit" ||
			normTool == "dos_review" ||
			normTool == "dos_verify" ||
			normTool == "todowrite"

	case SubagentRoleTester, SubagentRoleCrossValidator:
		// Tester agents: test execution (bash) and inspection, but NO file modifications.
		if MutationTools[normTool] {
			return false
		}
		return TesterAllowedTools[normTool]

	case SubagentRoleWorker, SubagentRoleDeepReason, SubagentRoleLander:
		// Implementation agents: curated implementation toolset without prompt bloat.
		// Excludes orchestration (task) and unrelated governance tools.
		return WorkerAllowedTools[normTool]

	case "plan":
		// Planning agents: read-only investigation and coordination. No mutations.
		if MutationTools[normTool] {
			return false
		}
		return ReadOnlySearchTools[normTool] || normTool == "todowrite" || normTool == "task"

	case SubagentRoleBuild, SubagentRoleGeneral, "":
		// Default / top-level coordinator roles: full tool access.
		return true

	default:
		// For unknown subagents, enforce fail-closed principle on mutation tools.
		return !MutationTools[normTool]
	}
}

// FilterToolsForSubagent returns only the tool descriptors permitted for the subagent role.
func FilterToolsForSubagent(role string, tools []ToolDescriptor) []ToolDescriptor {
	if len(tools) == 0 {
		return nil
	}
	var res []ToolDescriptor
	for _, td := range tools {
		if IsToolAllowed(role, td.Name) {
			res = append(res, td)
		}
	}
	return res
}

// FilterToolNamesForSubagent filters a slice of tool names for a subagent role.
func FilterToolNamesForSubagent(role string, toolNames []string) []string {
	if len(toolNames) == 0 {
		return nil
	}
	var res []string
	for _, name := range toolNames {
		if IsToolAllowed(role, name) {
			res = append(res, name)
		}
	}
	return res
}

// FilterMCPToolMapsForSubagent filters dynamic map-based tool descriptors (e.g. from gateway).
func FilterMCPToolMapsForSubagent(role string, tools []map[string]any) []map[string]any {
	if len(tools) == 0 {
		return nil
	}
	var res []map[string]any
	for _, td := range tools {
		name, _ := td["name"].(string)
		if IsToolAllowed(role, name) {
			res = append(res, td)
		}
	}
	return res
}

// EvaluateSubagentTokenSavings computes the schema and estimated prompt-prefix token savings.
func EvaluateSubagentTokenSavings(role string, tools []ToolDescriptor) SubagentTokenSavingsReceipt {
	filtered := FilterToolsForSubagent(role, tools)

	// Approximate token cost per tool definition schema:
	// A typical tool descriptor in JSON is ~250 tokens (~1,000 bytes).
	// For high fidelity, we compute token estimate from descriptor serialization.
	beforeTokens := estimateToolTokens(tools)
	afterTokens := estimateToolTokens(filtered)

	var savingsPct float64
	if beforeTokens > 0 {
		savingsPct = float64(beforeTokens-afterTokens) / float64(beforeTokens) * 100.0
	}

	var allowed []string
	allowedMap := make(map[string]bool)
	for _, f := range filtered {
		allowed = append(allowed, f.Name)
		allowedMap[f.Name] = true
	}

	var filteredOut []string
	for _, t := range tools {
		if !allowedMap[t.Name] {
			filteredOut = append(filteredOut, t.Name)
		}
	}

	return SubagentTokenSavingsReceipt{
		SubagentRole:          role,
		ToolsBefore:           len(tools),
		ToolsAfter:            len(filtered),
		EstimatedTokensBefore: beforeTokens,
		EstimatedTokensAfter:  afterTokens,
		SavingsPercent:        savingsPct,
		ReductionTargetMet:    savingsPct >= 60.0,
		AllowedTools:          allowed,
		FilteredOutTools:      filteredOut,
	}
}

func estimateToolTokens(tools []ToolDescriptor) int {
	if len(tools) == 0 {
		return 0
	}
	// Measure serialized length: ~4 characters per token in JSON schema representation.
	data, err := json.Marshal(tools)
	if err != nil || len(data) == 0 {
		return len(tools) * 250
	}
	tokens := len(data) / 4
	if tokens < len(tools)*200 {
		tokens = len(tools) * 200
	}
	return tokens
}

// ExtractSubagentFromRequest extracts caller subagent identity declared in OpenCode HTTP requests.
func ExtractSubagentFromRequest(req *http.Request) string {
	if req == nil {
		return ""
	}

	// 1. Dedicated custom headers
	for _, h := range []string{
		"X-OpenCode-Subagent",
		"X-Subagent-Type",
		"X-Subagent-Role",
		"X-Agent-Role",
		"X-Fak-Subagent",
	} {
		if val := strings.TrimSpace(req.Header.Get(h)); val != "" {
			return NormalizeSubagentRole(val)
		}
	}

	// 2. User-Agent inspection (e.g., "opencode/1.0 (subagent:explore)")
	if ua := req.Header.Get("User-Agent"); ua != "" {
		if r := extractSubagentFromString(ua); r != "" {
			return r
		}
	}

	// 3. Query parameter
	if req.URL != nil {
		q := req.URL.Query()
		for _, param := range []string{"subagent", "agent_role", "role", "subagent_type"} {
			if val := strings.TrimSpace(q.Get(param)); val != "" {
				return NormalizeSubagentRole(val)
			}
		}
	}

	// 4. Inspect body payload if present and accessible
	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil && len(bodyBytes) > 0 {
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			if role := ExtractSubagentFromPayload(bodyBytes); role != "" {
				return role
			}
		}
	}

	return ""
}

var subagentPattern = regexp.MustCompile(`(?i)(?:subagent|subagent_type|agent_role|agent)[\s:=]+["']?([a-zA-Z0-9_\-]+)["']?`)

func extractSubagentFromString(str string) string {
	matches := subagentPattern.FindStringSubmatch(str)
	if len(matches) > 1 {
		return NormalizeSubagentRole(matches[1])
	}
	return ""
}

// ExtractSubagentFromPayload parses JSON body for declared subagent identity.
func ExtractSubagentFromPayload(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return ""
	}

	for _, key := range []string{"subagent", "subagent_type", "agent_role", "subagent_role"} {
		if val, ok := parsed[key].(string); ok && strings.TrimSpace(val) != "" {
			return NormalizeSubagentRole(val)
		}
	}

	// Check metadata nested object
	if meta, ok := parsed["metadata"].(map[string]interface{}); ok {
		for _, key := range []string{"subagent", "subagent_type", "agent_role"} {
			if val, ok := meta[key].(string); ok && strings.TrimSpace(val) != "" {
				return NormalizeSubagentRole(val)
			}
		}
	}

	return ""
}

// EnsureOpenCodeSubagentPermissions injects role-specific tool permissions into opencode.json.
func EnsureOpenCodeSubagentPermissions(root string) (bool, error) {
	if root == "" {
		root = "."
	}
	configPath := filepath.Join(root, "opencode.json")

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("read opencode.json: %w", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return false, fmt.Errorf("parse opencode.json: %w", err)
	}
	if raw == nil {
		raw = make(map[string]interface{})
	}

	agentMap, ok := raw["agent"].(map[string]interface{})
	if !ok || agentMap == nil {
		agentMap = make(map[string]interface{})
		raw["agent"] = agentMap
	}

	modified := false

	// Target tool configurations for key subagents
	roleTools := map[string]map[string]interface{}{
		SubagentRoleExplore: {
			"edit":        false,
			"write":       false,
			"apply_patch": false,
			"bash":        false,
			"grep":        true,
			"glob":        true,
			"read":        true,
			"fak_read":    true,
		},
		SubagentRoleResearcher: {
			"edit":        false,
			"write":       false,
			"apply_patch": false,
			"bash":        false,
			"grep":        true,
			"glob":        true,
			"read":        true,
			"fak_read":    true,
		},
		SubagentRoleScout: {
			"edit":        false,
			"write":       false,
			"apply_patch": false,
			"bash":        false,
			"grep":        true,
			"glob":        true,
			"read":        true,
			"fak_read":    true,
		},
		SubagentRoleTester: {
			"edit":        false,
			"write":       false,
			"apply_patch": false,
			"bash":        true,
			"grep":        true,
			"glob":        true,
			"read":        true,
			"fak_read":    true,
		},
		SubagentRoleWorker: {
			"edit":        true,
			"write":       true,
			"apply_patch": true,
			"bash":        true,
			"grep":        true,
			"glob":        true,
			"read":        true,
			"fak_read":    true,
		},
	}

	for role, targetTools := range roleTools {
		existingEntry, hasEntry := agentMap[role]
		var entryMap map[string]interface{}
		if hasEntry {
			if em, ok := existingEntry.(map[string]interface{}); ok {
				entryMap = em
			}
		}
		if entryMap == nil {
			entryMap = make(map[string]interface{})
			agentMap[role] = entryMap
			modified = true
		}

		existingTools, hasTools := entryMap["tools"]
		if !hasTools {
			entryMap["tools"] = targetTools
			modified = true
		} else {
			exBytes, _ := json.Marshal(existingTools)
			tarBytes, _ := json.Marshal(targetTools)
			if string(exBytes) != string(tarBytes) {
				entryMap["tools"] = targetTools
				modified = true
			}
		}
	}

	if modified {
		out, err := json.MarshalIndent(raw, "", "  ")
		if err != nil {
			return false, fmt.Errorf("serialize opencode.json: %w", err)
		}
		if err := os.WriteFile(configPath, append(out, '\n'), 0644); err != nil {
			return false, fmt.Errorf("write opencode.json: %w", err)
		}
	}

	return modified, nil
}
