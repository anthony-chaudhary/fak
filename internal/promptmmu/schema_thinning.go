package promptmmu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// DefaultHotTools lists the 6 core tools always advertised to local models.
var DefaultHotTools = []string{
	"Read",
	"Edit",
	"Write",
	"Bash",
	"Glob",
	"Grep",
}

// IsHotTool checks whether a tool belongs to the default hot tool set.
func IsHotTool(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "read", "edit", "write", "bash", "glob", "grep":
		return true
	default:
		return false
	}
}

// FakToolsSearchDef returns the canonical minified JSON for fak_tools_search.
func FakToolsSearchDef() []byte {
	return []byte(`{"type":"function","function":{"name":"fak_tools_search","description":"Search available cold tools by keyword to fault their schemas into context.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Keyword to search tool names/descriptions."}},"required":["query"]}}}`)
}

// MinifyToolSchema strips verbose documentation, redundant titles, and empty fields
// from a tool's JSON schema to minimize prompt token footprint on local models.
func MinifyToolSchema(rawJSON []byte) ([]byte, error) {
	if len(bytes.TrimSpace(rawJSON)) == 0 {
		return nil, fmt.Errorf("promptmmu: empty schema input")
	}

	var data any
	if err := json.Unmarshal(rawJSON, &data); err != nil {
		return nil, fmt.Errorf("promptmmu: invalid JSON: %w", err)
	}

	cleanNode(data)

	out, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("promptmmu: failed to marshal minified schema: %w", err)
	}
	return out, nil
}

// cleanNode recursively minifies maps and slices in place.
func cleanNode(node any) {
	switch v := node.(type) {
	case map[string]any:
		// 1. Strip redundant title
		delete(v, "title")

		// 2. Truncate long descriptions (max 80 chars)
		if descVal, ok := v["description"]; ok {
			if descStr, isStr := descVal.(string); isStr {
				v["description"] = truncateDesc(descStr, 80)
			}
		}

		// 3. Drop empty examples or null defaults
		if exVal, ok := v["examples"]; ok {
			if exVal == nil {
				delete(v, "examples")
			} else if sliceVal, isSlice := exVal.([]any); isSlice && len(sliceVal) == 0 {
				delete(v, "examples")
			}
		}
		if defVal, ok := v["default"]; ok && defVal == nil {
			delete(v, "default")
		}

		// Recurse on remaining keys
		for _, child := range v {
			cleanNode(child)
		}

	case []any:
		for _, child := range v {
			cleanNode(child)
		}
	}
}

func truncateDesc(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxLen {
		return s
	}
	// Truncate at maxLen-3 and append "..."
	return strings.TrimSpace(s[:maxLen-3]) + "..."
}

// ToolEntry is an internal representation of an OpenAI-format tool declaration.
type ToolEntry struct {
	Type     string         `json:"type"`
	Function map[string]any `json:"function"`
}

// ThinLocalTools filters an inbound array of tool declarations down to active hot tools
// plus any previously faulted cold tools, injecting fak_tools_search when cold tools exist.
func ThinLocalTools(rawToolsJSON []byte, hotTools []string, faultedTools []string) ([]byte, []string, error) {
	trimmed := bytes.TrimSpace(rawToolsJSON)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return rawToolsJSON, nil, nil
	}

	var rawEntries []map[string]any
	if err := json.Unmarshal(rawToolsJSON, &rawEntries); err != nil {
		return nil, nil, fmt.Errorf("promptmmu: cannot decode tools array: %w", err)
	}

	activeSet := make(map[string]bool)
	for _, h := range hotTools {
		activeSet[strings.ToLower(strings.TrimSpace(h))] = true
	}
	for _, f := range faultedTools {
		activeSet[strings.ToLower(strings.TrimSpace(f))] = true
	}

	var keptEntries []map[string]any
	var prunedNames []string

	for _, entry := range rawEntries {
		name := extractToolName(entry)
		if name == "" {
			keptEntries = append(keptEntries, entry)
			continue
		}

		if activeSet[strings.ToLower(name)] {
			cleanNode(entry)
			keptEntries = append(keptEntries, entry)
		} else {
			prunedNames = append(prunedNames, name)
		}
	}

	// If any tools were pruned/cold, inject fak_tools_search so the agent can discover them
	if len(prunedNames) > 0 {
		var searchEntry map[string]any
		_ = json.Unmarshal(FakToolsSearchDef(), &searchEntry)
		keptEntries = append(keptEntries, searchEntry)
	}

	out, err := json.Marshal(keptEntries)
	if err != nil {
		return nil, nil, fmt.Errorf("promptmmu: failed to marshal thinned tools: %w", err)
	}

	return out, prunedNames, nil
}

func extractToolName(entry map[string]any) string {
	if fn, ok := entry["function"].(map[string]any); ok {
		if name, ok := fn["name"].(string); ok {
			return name
		}
	}
	if name, ok := entry["name"].(string); ok {
		return name
	}
	return ""
}

// FaultInColdTools searches available tool definitions for matches to query,
// promoting them into the active tool set for subsequent turns.
func FaultInColdTools(query string, catalog map[string]string, activeSet map[string]bool) []string {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}

	var newlyFaulted []string
	for name, desc := range catalog {
		lowerName := strings.ToLower(name)
		if activeSet[lowerName] {
			continue
		}

		if strings.Contains(lowerName, q) || strings.Contains(strings.ToLower(desc), q) {
			newlyFaulted = append(newlyFaulted, name)
			activeSet[lowerName] = true
		}
	}

	return newlyFaulted
}
