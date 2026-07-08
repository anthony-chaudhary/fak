package mcpfootprint

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func td(name, desc, params string) agent.ToolDef {
	return agent.ToolDef{
		Type:     "function",
		Function: agent.ToolDefFunction{Name: name, Description: desc, Parameters: json.RawMessage(params)},
	}
}

// TestPricePartitionsExactly proves Price is a faithful partition of the schema
// bytes: the Tools bucket equals the sum of the per-tool bytes, ToolCount matches
// the input, Floor == Tools (no System/History present), and the provenance label
// is carried through. If any of these drifts, the scorecard would be lying.
func TestPricePartitionsExactly(t *testing.T) {
	tools := []agent.ToolDef{
		td("a", "short", `{"type":"object"}`),
		td("bbb", "a longer description here", `{"type":"object","properties":{"x":{"type":"string"}}}`),
	}
	fp := Price(tools)
	if fp.ToolCount != len(tools) {
		t.Fatalf("ToolCount=%d want %d", fp.ToolCount, len(tools))
	}
	sum := 0
	for _, pt := range fp.PerTool {
		sum += pt.Bytes
	}
	if sum != fp.Tools.Bytes {
		t.Fatalf("per-tool bytes sum=%d != Tools.Bytes=%d", sum, fp.Tools.Bytes)
	}
	if fp.Floor.Bytes != fp.Tools.Bytes {
		t.Fatalf("Floor.Bytes=%d != Tools.Bytes=%d (no system expected)", fp.Floor.Bytes, fp.Tools.Bytes)
	}
	if fp.Provenance != agent.FootprintProvenance {
		t.Fatalf("provenance=%q want %q", fp.Provenance, agent.FootprintProvenance)
	}
}

// TestPerToolSortedDescending proves the ranking is largest-first and deterministic.
func TestPerToolSortedDescending(t *testing.T) {
	tools := []agent.ToolDef{
		td("small", "x", `{}`),
		td("big", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", `{"type":"object","properties":{"y":{"type":"string"}}}`),
	}
	got := PerToolSorted(Price(tools))
	for i := 1; i < len(got); i++ {
		if got[i-1].Bytes < got[i].Bytes {
			t.Fatalf("not sorted descending at %d: %+v", i, got)
		}
	}
	if got[0].Name != "big" {
		t.Fatalf("largest tool = %q want big", got[0].Name)
	}
}

// TestRealFakMCPFloor is the witness: it prices fak's ACTUAL MCP registry (via the
// gateway export) and proves the number is a non-trivial, faithful floor. The logged
// value is the #3230 baseline the epic (#3229) ratchets down as cold schemas defer.
func TestRealFakMCPFloor(t *testing.T) {
	defs := gateway.MCPFloorToolDefs()
	if len(defs) == 0 {
		t.Fatal("fak MCP registry priced as 0 tools — export is not seeing the registry")
	}
	fp := Price(defs)
	sum := 0
	for _, pt := range fp.PerTool {
		sum += pt.Bytes
	}
	if sum != fp.Tools.Bytes {
		t.Fatalf("real registry not a faithful partition: sum=%d != Tools.Bytes=%d", sum, fp.Tools.Bytes)
	}
	if fp.Tools.Tokens <= 0 {
		t.Fatalf("real registry floor priced as %d tokens", fp.Tools.Tokens)
	}
	t.Logf("fak MCP always-sent floor: %d tools, %d est. tokens (%d bytes, %s)",
		fp.ToolCount, fp.Tools.Tokens, fp.Tools.Bytes, fp.Provenance)
	top := PerToolSorted(fp)
	for i := 0; i < min(5, len(top)); i++ {
		t.Logf("  top %d: %5d tok  %s", i+1, top[i].Tokens, top[i].Name)
	}
}
