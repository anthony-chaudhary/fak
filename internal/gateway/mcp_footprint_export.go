package gateway

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// mcp_footprint_export.go — a read-only view of the MCP tool registry for the
// always-sent-floor scorecard (#3230, epic #3229).
//
// The registry (toolDescriptors) is the single source of truth for what fak's MCP
// server advertises; this renders it as []agent.ToolDef so the floor can be PRICED
// with the same estimator the gateway already uses (agent.RequestFootprint), never
// a second one. It returns the FULL, unfiltered registry — the floor a client pays
// when no --expose allowlist is in force (exposeAllow == nil), which is the default.

// MCPFloorToolDefs renders every registered fak MCP tool descriptor as an
// agent.ToolDef carrying the exact bytes a client sees on tools/list: the tool
// name, description, and its inputSchema (mapped to the ToolDef Parameters slot,
// which agent.RequestFootprint prices identically to a Messages tool schema). The
// mapping is byte-faithful: a json.RawMessage inputSchema is copied verbatim so no
// re-marshal whitespace churn can move the priced byte count.
func MCPFloorToolDefs() []agent.ToolDef {
	descs := toolDescriptors()
	out := make([]agent.ToolDef, 0, len(descs))
	for _, d := range descs {
		name, _ := d["name"].(string)
		desc, _ := d["description"].(string)
		var params json.RawMessage
		switch s := d["inputSchema"].(type) {
		case json.RawMessage:
			params = s
		case []byte:
			params = s
		case nil:
			// tool with no declared inputSchema — priced as zero parameter bytes
		default:
			if b, err := json.Marshal(s); err == nil {
				params = b
			}
		}
		out = append(out, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        name,
				Description: desc,
				Parameters:  params,
			},
		})
	}
	return out
}
