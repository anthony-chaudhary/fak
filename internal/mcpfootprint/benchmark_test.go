package mcpfootprint

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// syntheticTools generates n distinct tool definitions with realistic schemas and descriptions.
func syntheticTools(n int) []agent.ToolDef {
	tools := make([]agent.ToolDef, n)
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("mcp_tool_synthetic_%03d", i)
		desc := fmt.Sprintf("Description for synthetic MCP tool %d: handles execution, context observation, and state management.", i)
		params := fmt.Sprintf(`{"type":"object","properties":{"arg_%d":{"type":"string","description":"Parameter %d"},"flag":{"type":"boolean"}},"required":["arg_%d"]}`, i, i, i)
		tools[i] = agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        name,
				Description: desc,
				Parameters:  json.RawMessage(params),
			},
		}
	}
	return tools
}

// BenchmarkPrice measures footprint calculation on production MCP tool definitions.
func BenchmarkPrice(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	if len(defs) == 0 {
		b.Fatal("empty MCP tool definitions")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fp := Price(defs)
		if fp.ToolCount != len(defs) {
			b.Fatalf("unexpected tool count: got %d want %d", fp.ToolCount, len(defs))
		}
	}
}

// BenchmarkPriceSyntheticScaling measures footprint calculation throughput across varied tool counts.
func BenchmarkPriceSyntheticScaling(b *testing.B) {
	for _, count := range []int{10, 50, 100} {
		b.Run(fmt.Sprintf("tools=%d", count), func(b *testing.B) {
			tools := syntheticTools(count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fp := Price(tools)
				if fp.ToolCount != count {
					b.Fatalf("unexpected tool count: got %d want %d", fp.ToolCount, count)
				}
			}
		})
	}
}

// BenchmarkPerToolSorted measures ranking and sorting of per-tool footprint costs.
func BenchmarkPerToolSorted(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	fp := Price(defs)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sorted := PerToolSorted(fp)
		if len(sorted) != len(defs) {
			b.Fatalf("unexpected sorted count: got %d want %d", len(sorted), len(defs))
		}
	}
}

// BenchmarkRender measures formatting and rendering of human-readable footprint tables.
func BenchmarkRender(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	fp := Price(defs)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := Render(fp)
		if len(s) == 0 {
			b.Fatal("unexpected empty render output")
		}
	}
}

// BenchmarkDescriptionFootprint measures calculation of the description-only footprint floor.
func BenchmarkDescriptionFootprint(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dfp := DescriptionFootprint(defs)
		if dfp.ToolCount != len(defs) {
			b.Fatalf("unexpected description tool count: got %d want %d", dfp.ToolCount, len(defs))
		}
	}
}

// BenchmarkDescriptionTokens measures extraction of total estimated description tokens.
func BenchmarkDescriptionTokens(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toks := DescriptionTokens(defs)
		if toks <= 0 {
			b.Fatalf("unexpected non-positive description tokens: %d", toks)
		}
	}
}

// BenchmarkPerToolDescription measures individual description pricing and largest-first ranking.
func BenchmarkPerToolDescription(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ptd := PerToolDescription(defs)
		if len(ptd) != len(defs) {
			b.Fatalf("unexpected per-tool description count: got %d want %d", len(ptd), len(defs))
		}
	}
}

// BenchmarkCheckFloor measures floor gate evaluation against the committed ceiling.
func BenchmarkCheckFloor(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	fp := Price(defs)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := CheckFloor(fp); err != nil {
			b.Fatalf("unexpected floor check failure: %v", err)
		}
	}
}

// BenchmarkCheckDescriptions measures description gate evaluation against the committed ceiling.
func BenchmarkCheckDescriptions(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := CheckDescriptions(defs); err != nil {
			b.Fatalf("unexpected description check failure: %v", err)
		}
	}
}

// BenchmarkWeightedBill measures joining per-tool footprints with invocation frequencies and ranking.
func BenchmarkWeightedBill(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	fp := Price(defs)

	freq := make(map[string]float64, len(fp.PerTool))
	for idx, pt := range fp.PerTool {
		switch idx % 4 {
		case 0:
			freq[pt.Name] = 12.5 // frequent
		case 1:
			freq[pt.Name] = 0.25 // rare
		case 2:
			freq[pt.Name] = 0.0 // observed idle
		case 3:
			// omitted -> triggers fallback 1.0
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bill := WeightedBill(fp.PerTool, freq)
		if len(bill) != len(fp.PerTool) {
			b.Fatalf("unexpected bill length: got %d want %d", len(bill), len(fp.PerTool))
		}
	}
}

// BenchmarkTotalBill measures summation of weighted footprint contributions.
func BenchmarkTotalBill(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	fp := Price(defs)

	freq := make(map[string]float64, len(fp.PerTool))
	for idx, pt := range fp.PerTool {
		if idx%2 == 0 {
			freq[pt.Name] = 3.5
		}
	}
	bill := WeightedBill(fp.PerTool, freq)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total := TotalBill(bill)
		if total <= 0 {
			b.Fatalf("unexpected non-positive total bill: %f", total)
		}
	}
}

// BenchmarkFootprintLifecycle measures end-to-end pricing, gating, weighting, and formatting.
func BenchmarkFootprintLifecycle(b *testing.B) {
	defs := gateway.MCPFloorToolDefs()
	freq := map[string]float64{
		"fak_memory_run": 5.0,
		"fak_admit":      10.0,
		"fak_trajquery":  1.5,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fp := Price(defs)
		if err := CheckFloor(fp); err != nil {
			b.Fatalf("floor check failed: %v", err)
		}
		if err := CheckDescriptions(defs); err != nil {
			b.Fatalf("description check failed: %v", err)
		}
		sorted := PerToolSorted(fp)
		bill := WeightedBill(sorted, freq)
		total := TotalBill(bill)
		render := Render(fp)
		if total <= 0 || len(render) == 0 {
			b.Fatalf("lifecycle produced invalid outputs: total=%f render_len=%d", total, len(render))
		}
	}
}

// TestBenchmarkSanity verifies that the footprint benchmark suite executes cleanly.
func TestBenchmarkSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkFootprintLifecycle)
	if res.N <= 0 {
		t.Fatalf("expected iterations > 0, got %d", res.N)
	}
}
