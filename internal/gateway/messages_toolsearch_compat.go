package gateway

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const retiredToolSearchBeta = "tool-search-2025-09-17"

// filterRetiredAnthropicBetas removes protocol revisions Anthropic has retired while
// preserving every other client-negotiated beta in its original order.
func filterRetiredAnthropicBetas(raw string) string {
	parts := strings.Split(raw, ",")
	kept := parts[:0]
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" || part == retiredToolSearchBeta {
			continue
		}
		kept = append(kept, part)
	}
	return strings.Join(kept, ",")
}

// migrateRetiredToolSearch upgrades the retired Claude Code descriptor as one
// protocol unit with the header filter above. The canonical Tools view intentionally
// contains only function tools; Raw is the authoritative passthrough body.
func migrateRetiredToolSearch(req *agent.AnthropicMessagesRequest) bool {
	if req == nil || len(req.Raw) == 0 {
		return false
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(req.Raw, &body); err != nil {
		return false
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(body["tools"], &tools); err != nil {
		return false
	}
	changed := false
	for i, raw := range tools {
		migrated, ok := migrateToolSearchDescriptor(raw)
		if ok {
			tools[i], changed = migrated, true
		}
	}
	if !changed {
		return false
	}
	migratedTools, err := json.Marshal(tools)
	if err != nil {
		return false
	}
	body["tools"] = migratedTools
	migrated, err := json.Marshal(body)
	if err != nil {
		return false
	}
	req.Raw = migrated
	return true
}

func migrateToolSearchDescriptor(raw json.RawMessage) (json.RawMessage, bool) {
	var tool map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tool); err != nil {
		return raw, false
	}
	var typ string
	_ = json.Unmarshal(tool["type"], &typ)
	if typ != "tool_search_tool_20250917" {
		return raw, false
	}
	tool["type"] = json.RawMessage(`"` + toolSearchToolType + `"`)
	tool["name"] = json.RawMessage(`"` + toolSearchToolName + `"`)
	migrated, err := json.Marshal(tool)
	if err != nil {
		return raw, false
	}
	return migrated, true
}
