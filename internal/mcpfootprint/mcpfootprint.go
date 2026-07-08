// Package mcpfootprint prices the always-sent MCP tool-schema floor — the fixed
// per-turn token tax every registered tool adds to every API call, whether or not
// the tool is ever selected.
//
// It is the consumer #3230 (epic #3229) wires onto the shipped-but-unwired
// agent.RequestFootprint primitive: the SAME char-walk and the SAME ~4-char/token
// divisor, so this number can never drift from EstimateAnthropicTokens. Pricing is
// deterministic and offline — no live model call — so it is a stable scorecard the
// epic can ratchet down as cold schemas are deferred (#3231/#3232).
package mcpfootprint

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Price decomposes a tool set into an agent.Footprint whose Tools (== Floor here,
// since there is no System/History) bucket is the always-sent MCP floor and whose
// PerTool slice is each tool's per-call byte/token cost. It reuses
// agent.RequestFootprint verbatim — an otherwise-empty request whose only cost is
// the tool schemas — so there is exactly ONE estimator in the tree.
func Price(tools []agent.ToolDef) agent.Footprint {
	return agent.RequestFootprint(&agent.AnthropicMessagesRequest{Tools: tools})
}

// PerToolSorted returns the footprint's per-tool costs largest-first, ties broken by
// name so the order is deterministic — the "where the bytes went" ranking a distiller
// reads first.
func PerToolSorted(fp agent.Footprint) []agent.ToolFootprint {
	out := append([]agent.ToolFootprint(nil), fp.PerTool...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Bytes != out[j].Bytes {
			return out[i].Bytes > out[j].Bytes
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Render is a compact human table: the total floor (est. tokens + exact bytes +
// provenance) then each tool largest-first. Deterministic for a given tool set.
func Render(fp agent.Footprint) string {
	var b strings.Builder
	fmt.Fprintf(&b, "mcp-footprint: %d tools · floor %d est. tokens (%d bytes, %s)\n",
		fp.ToolCount, fp.Tools.Tokens, fp.Tools.Bytes, fp.Provenance)
	for _, t := range PerToolSorted(fp) {
		fmt.Fprintf(&b, "  %6d tok  %7d B  %s\n", t.Tokens, t.Bytes, t.Name)
	}
	return b.String()
}
