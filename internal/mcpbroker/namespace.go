package mcpbroker

import (
	"encoding/json"
	"strings"
)

// MCPTool represents an MCP tool definition discovered from an upstream MCP server.
type MCPTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// NamespaceTool returns the deterministic namespaced tool name for an MCP server:
// "mcp__" + serverID + "__" + toolName
func NamespaceTool(serverID, toolName string) string {
	return "mcp__" + serverID + "__" + toolName
}

// ParseNamespacedTool parses a namespaced tool name into its component serverID and raw toolName.
// Returns ok=false if the name does not match the "mcp__<serverID>__<toolName>" convention.
func ParseNamespacedTool(name string) (serverID, toolName string, ok bool) {
	if !strings.HasPrefix(name, "mcp__") {
		return "", "", false
	}
	rest := name[len("mcp__"):]
	idx := strings.Index(rest, "__")
	if idx <= 0 || idx+2 >= len(rest) {
		return "", "", false
	}
	serverID = rest[:idx]
	toolName = rest[idx+2:]
	return serverID, toolName, true
}

// RegisterServerTools registers a list of discovered MCP tools into the broker with namespacing.
// It filters tools according to the server's AllowedTools and DeniedTools policies.
// Tools filtered out by policy are skipped. Returns the list of successfully registered
// namespaced tool names.
func RegisterServerTools(b *Broker, serverID string, tools []MCPTool, handler ToolHandler) ([]string, error) {
	return b.RegisterServerTools(serverID, tools, handler)
}

// RegisterServerTools registers discovered MCP tools into the broker with namespacing on the broker.
func (b *Broker) RegisterServerTools(serverID string, tools []MCPTool, handler ToolHandler) ([]string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return nil, ErrBrokerClosed
	}

	srv, hasSrv := b.servers[serverID]
	var registered []string

	for _, tool := range tools {
		if tool.Name == "" {
			continue
		}

		namespacedName := NamespaceTool(serverID, tool.Name)

		// Check server policies if server config exists
		if hasSrv {
			// Check denylist
			denied := false
			for _, d := range srv.DeniedTools {
				if d == tool.Name || d == namespacedName {
					denied = true
					break
				}
			}
			if denied {
				continue
			}

			// Check allowlist
			if len(srv.AllowedTools) > 0 {
				allowed := false
				for _, a := range srv.AllowedTools {
					if a == tool.Name || a == namespacedName {
						allowed = true
						break
					}
				}
				if !allowed {
					continue
				}
			}
		}

		reg := ToolRegistration{
			Name:        namespacedName,
			ServerID:    serverID,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
			ReadOnly:    hasSrv && srv.ReadOnly,
			Handler:     handler,
		}

		// Save or update registration
		b.tools[namespacedName] = reg
		registered = append(registered, namespacedName)
	}

	return registered, nil
}
