package agent

import "strings"

// hostedToolTypes are the Responses API tool types the provider executes itself.
// Only a Responses-wire upstream can run them; a chat-completions upstream has no
// shape to receive them.
var hostedToolTypes = map[string]bool{
	"web_search":       true,
	"file_search":      true,
	"code_interpreter": true,
	"computer_use":     true,
	"local_shell":      true,
	"shell":            true,
	"image_generation": true,
	"mcp":              true,
}

// hostedToolTypePrefixes admit the dated and preview variants of a hosted tool
// (web_search_preview, web_search_2025_08_26, computer_use_preview).
var hostedToolTypePrefixes = []string{"web_search_", "computer_use_"}

// IsHostedToolType reports whether typ names a provider-executed Responses tool.
func IsHostedToolType(typ string) bool {
	if hostedToolTypes[typ] {
		return true
	}
	for _, prefix := range hostedToolTypePrefixes {
		if strings.HasPrefix(typ, prefix) && len(typ) > len(prefix) {
			return true
		}
	}
	return false
}
