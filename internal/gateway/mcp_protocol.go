package gateway

import (
	"encoding/json"
	"sync"
)

// MCPToolDescriptor represents an MCP tool descriptor struct with schema, annotations,
// and granular capability hints.
type MCPToolDescriptor struct {
	Name            string              `json:"name"`
	Description     string              `json:"description,omitempty"`
	InputSchema     json.RawMessage     `json:"inputSchema,omitempty"`
	Annotations     *mcpToolAnnotations `json:"annotations,omitempty"`
	Strict          bool                `json:"strict,omitempty"`
	DestructiveHint bool                `json:"destructive_hint,omitempty"`
	OpenWorldHint   bool                `json:"open_world_hint,omitempty"`
}

// ToMap converts an MCPToolDescriptor to its map representation.
func (td MCPToolDescriptor) ToMap() map[string]any {
	m := map[string]any{
		"name": td.Name,
	}
	if td.Description != "" {
		m["description"] = td.Description
	}
	if len(td.InputSchema) > 0 {
		m["inputSchema"] = td.InputSchema
	}
	if td.Annotations != nil {
		m["annotations"] = td.Annotations.toMap()
	}
	if td.Strict {
		m["strict"] = true
	}
	if td.DestructiveHint {
		m["destructive_hint"] = true
	}
	if td.OpenWorldHint {
		m["open_world_hint"] = true
	}
	return m
}

// ToolListRepresentation formats the tool descriptor for MCP tools/list responses.
func (td MCPToolDescriptor) ToolListRepresentation() map[string]any {
	return td.ToMap()
}

// ToolDescriptor represents a tool descriptor with schema and capability annotations.
type ToolDescriptor struct {
	Name            string              `json:"name"`
	Description     string              `json:"description,omitempty"`
	InputSchema     json.RawMessage     `json:"inputSchema,omitempty"`
	Annotations     *mcpToolAnnotations `json:"annotations,omitempty"`
	Strict          bool                `json:"strict,omitempty"`
	DestructiveHint bool                `json:"destructive_hint,omitempty"`
	OpenWorldHint   bool                `json:"open_world_hint,omitempty"`
}

// ToMap converts a ToolDescriptor to its map representation.
func (td ToolDescriptor) ToMap() map[string]any {
	m := map[string]any{
		"name": td.Name,
	}
	if td.Description != "" {
		m["description"] = td.Description
	}
	if len(td.InputSchema) > 0 {
		m["inputSchema"] = td.InputSchema
	}
	if td.Annotations != nil {
		m["annotations"] = td.Annotations.toMap()
	}
	if td.Strict {
		m["strict"] = true
	}
	if td.DestructiveHint {
		m["destructive_hint"] = true
	}
	if td.OpenWorldHint {
		m["open_world_hint"] = true
	}
	return m
}

// ToolListRepresentation formats the tool descriptor for MCP tools/list responses.
func (td ToolDescriptor) ToolListRepresentation() map[string]any {
	return td.ToMap()
}

// ToolRegistration represents a registration struct representing tool definitions.
type ToolRegistration struct {
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	InputSchema     json.RawMessage `json:"input_schema,omitempty"`
	DestructiveHint bool            `json:"destructive_hint,omitempty"`
	OpenWorldHint   bool            `json:"open_world_hint,omitempty"`
}

// ToolDefinition represents a schema/registration struct for tool definitions.
type ToolDefinition struct {
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	InputSchema     json.RawMessage `json:"inputSchema,omitempty"`
	DestructiveHint bool            `json:"destructive_hint,omitempty"`
	OpenWorldHint   bool            `json:"open_world_hint,omitempty"`
}

var (
	registeredMCPToolsLock sync.RWMutex
	registeredMCPTools     = make(map[string]MCPToolDescriptor)
	registeredMCPToolOrder []string
)

// RegisterMCPTool registers an MCP tool descriptor into the MCP protocol registry.
func RegisterMCPTool(td MCPToolDescriptor) {
	registeredMCPToolsLock.Lock()
	defer registeredMCPToolsLock.Unlock()
	if _, exists := registeredMCPTools[td.Name]; !exists {
		registeredMCPToolOrder = append(registeredMCPToolOrder, td.Name)
	}
	registeredMCPTools[td.Name] = td
}

// RegisterTool registers an MCP tool descriptor into the MCP protocol registry.
func RegisterTool(td MCPToolDescriptor) {
	RegisterMCPTool(td)
}

// RegisteredMCPTools returns all registered tool descriptors in registration order.
func RegisteredMCPTools() []MCPToolDescriptor {
	registeredMCPToolsLock.RLock()
	defer registeredMCPToolsLock.RUnlock()
	out := make([]MCPToolDescriptor, 0, len(registeredMCPToolOrder))
	for _, name := range registeredMCPToolOrder {
		out = append(out, registeredMCPTools[name])
	}
	return out
}

// RegisteredTools is an alias for RegisteredMCPTools.
func RegisteredTools() []MCPToolDescriptor {
	return RegisteredMCPTools()
}

// RegisteredMCPToolListRepresentations returns the tool list representations of all registered tools.
func RegisteredMCPToolListRepresentations() []map[string]any {
	tools := RegisteredMCPTools()
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.ToolListRepresentation())
	}
	return out
}

// RegisteredToolListRepresentations is an alias for RegisteredMCPToolListRepresentations.
func RegisteredToolListRepresentations() []map[string]any {
	return RegisteredMCPToolListRepresentations()
}

// GetRegisteredMCPTool returns a registered tool by name.
func GetRegisteredMCPTool(name string) (MCPToolDescriptor, bool) {
	registeredMCPToolsLock.RLock()
	defer registeredMCPToolsLock.RUnlock()
	td, ok := registeredMCPTools[name]
	return td, ok
}

// ResetRegisteredMCPTools resets the registered tools to the core tools palette.
func ResetRegisteredMCPTools() {
	registeredMCPToolsLock.Lock()
	defer registeredMCPToolsLock.Unlock()
	registeredMCPTools = make(map[string]MCPToolDescriptor)
	registeredMCPToolOrder = nil
	for _, tool := range CoreToolsPalette {
		registeredMCPToolOrder = append(registeredMCPToolOrder, tool.Name)
		registeredMCPTools[tool.Name] = tool
	}
}

func init() {
	ResetRegisteredMCPTools()
}
