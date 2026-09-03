package zaitask

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValidateToolCall(t *testing.T) {
	defs := []ToolDefinition{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "search_kb",
				Description: "Search knowledge base",
				Parameters: json.RawMessage(`{
					"type": "object",
					"required": ["query", "limit"],
					"properties": {
						"query": {"type": "string"},
						"limit": {"type": "integer"},
						"tags":  {"type": "array"}
					}
				}`),
			},
		},
	}

	t.Run("conforming call", func(t *testing.T) {
		call := ToolCall{
			Function: ToolCallFunction{
				Name:      "search_kb",
				Arguments: json.RawMessage(`{"query": "GLM architecture", "limit": 10, "tags": ["glm", "flash"]}`),
			},
		}
		if err := ValidateToolCall(call, defs); err != nil {
			t.Fatalf("expected valid call to pass, got: %v", err)
		}
	})

	t.Run("missing required argument", func(t *testing.T) {
		call := ToolCall{
			Function: ToolCallFunction{
				Name:      "search_kb",
				Arguments: json.RawMessage(`{"query": "GLM architecture"}`),
			},
		}
		err := ValidateToolCall(call, defs)
		if err == nil || !strings.Contains(err.Error(), "missing required argument \"limit\"") {
			t.Fatalf("expected missing limit error, got: %v", err)
		}
	})

	t.Run("type mismatch", func(t *testing.T) {
		call := ToolCall{
			Function: ToolCallFunction{
				Name:      "search_kb",
				Arguments: json.RawMessage(`{"query": 12345, "limit": 10}`),
			},
		}
		err := ValidateToolCall(call, defs)
		if err == nil || !strings.Contains(err.Error(), "must be a string") {
			t.Fatalf("expected string type mismatch error, got: %v", err)
		}
	})

	t.Run("unknown tool name", func(t *testing.T) {
		call := ToolCall{
			Function: ToolCallFunction{
				Name:      "nonexistent_tool",
				Arguments: json.RawMessage(`{}`),
			},
		}
		err := ValidateToolCall(call, defs)
		if err == nil || !strings.Contains(err.Error(), "not found in tool definitions") {
			t.Fatalf("expected unknown tool error, got: %v", err)
		}
	})
}
