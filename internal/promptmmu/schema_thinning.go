package promptmmu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
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

// FaultedToolRecord tracks turn lifecycle and invocation history for a cold tool
// faulted into the prompt context.
type FaultedToolRecord struct {
	Name            string `json:"name"`
	FaultedTurn     int    `json:"faulted_turn"`
	LastInvokedTurn int    `json:"last_invoked_turn"`
}

// FaultedToolTracker manages turn-decay and LRU aging for faulted cold tool schemas.
type FaultedToolTracker struct {
	MaxIdleTurns int                           `json:"max_idle_turns"` // default 5 if <= 0
	records      map[string]*FaultedToolRecord // lowercased tool name -> record
}

// NewFaultedToolTracker initializes a FaultedToolTracker with the specified max idle turns.
// If maxIdleTurns <= 0, defaults to 5.
func NewFaultedToolTracker(maxIdleTurns int) *FaultedToolTracker {
	if maxIdleTurns <= 0 {
		maxIdleTurns = 5
	}
	return &FaultedToolTracker{
		MaxIdleTurns: maxIdleTurns,
		records:      make(map[string]*FaultedToolRecord),
	}
}

// RecordFault registers or updates a faulted tool at currentTurn.
// Sets FaultedTurn to currentTurn. If LastInvokedTurn is 0, sets LastInvokedTurn to currentTurn.
func (t *FaultedToolTracker) RecordFault(toolName string, currentTurn int) {
	if t == nil {
		return
	}
	trimmed := strings.TrimSpace(toolName)
	if trimmed == "" {
		return
	}
	if t.records == nil {
		t.records = make(map[string]*FaultedToolRecord)
	}
	lower := strings.ToLower(trimmed)
	rec, exists := t.records[lower]
	if !exists {
		t.records[lower] = &FaultedToolRecord{
			Name:            trimmed,
			FaultedTurn:     currentTurn,
			LastInvokedTurn: currentTurn,
		}
		return
	}
	rec.FaultedTurn = currentTurn
	if rec.LastInvokedTurn == 0 {
		rec.LastInvokedTurn = currentTurn
	}
	if rec.Name == "" {
		rec.Name = trimmed
	}
}

// RecordInvocation updates LastInvokedTurn to currentTurn if tool is tracked.
func (t *FaultedToolTracker) RecordInvocation(toolName string, currentTurn int) {
	if t == nil || len(t.records) == 0 {
		return
	}
	lower := strings.ToLower(strings.TrimSpace(toolName))
	if rec, ok := t.records[lower]; ok {
		rec.LastInvokedTurn = currentTurn
	}
}

// PruneColdTools computes which tools have been idle for >= MaxIdleTurns turns
// (idle turns = currentTurn - max(FaultedTurn, LastInvokedTurn)). Evicts them from
// records and returns the pruned tool names in deterministic sorted order.
func (t *FaultedToolTracker) PruneColdTools(currentTurn int) []string {
	if t == nil || len(t.records) == 0 {
		return nil
	}
	maxIdle := t.MaxIdleTurns
	if maxIdle <= 0 {
		maxIdle = 5
	}

	var pruned []string
	for key, rec := range t.records {
		lastActive := rec.FaultedTurn
		if rec.LastInvokedTurn > lastActive {
			lastActive = rec.LastInvokedTurn
		}
		idleTurns := currentTurn - lastActive
		if idleTurns >= maxIdle {
			prunedName := rec.Name
			if prunedName == "" {
				prunedName = key
			}
			pruned = append(pruned, prunedName)
			delete(t.records, key)
		}
	}
	sort.Strings(pruned)
	return pruned
}

// ActiveFaultedTools returns the currently active faulted tool names, sorted
// alphabetically for deterministic prompt caching.
func (t *FaultedToolTracker) ActiveFaultedTools() []string {
	if t == nil || len(t.records) == 0 {
		return []string{}
	}
	active := make([]string, 0, len(t.records))
	for _, rec := range t.records {
		if rec.Name != "" {
			active = append(active, rec.Name)
		}
	}
	sort.Strings(active)
	return active
}

// IsActive returns true if the tool is currently tracked.
func (t *FaultedToolTracker) IsActive(toolName string) bool {
	if t == nil || len(t.records) == 0 {
		return false
	}
	lower := strings.ToLower(strings.TrimSpace(toolName))
	_, ok := t.records[lower]
	return ok
}

// ActiveSet returns a map of lowercased active faulted tool names.
func (t *FaultedToolTracker) ActiveSet() map[string]bool {
	set := make(map[string]bool)
	if t == nil {
		return set
	}
	for k := range t.records {
		set[k] = true
	}
	return set
}

// FaultInColdToolsWithTracker searches catalog for tools matching query, records
// fault in tracker for each new tool, and returns newly faulted tool names.
func FaultInColdToolsWithTracker(query string, catalog map[string]string, tracker *FaultedToolTracker, currentTurn int) []string {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" || tracker == nil {
		return nil
	}

	keys := make([]string, 0, len(catalog))
	for name := range catalog {
		keys = append(keys, name)
	}
	sort.Strings(keys)

	var newlyFaulted []string
	for _, name := range keys {
		lowerName := strings.ToLower(name)
		if tracker.IsActive(lowerName) {
			continue
		}

		desc := catalog[name]
		if strings.Contains(lowerName, q) || strings.Contains(strings.ToLower(desc), q) {
			tracker.RecordFault(name, currentTurn)
			newlyFaulted = append(newlyFaulted, name)
		}
	}

	return newlyFaulted
}
