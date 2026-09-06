package promptmmu

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

var (
	sinkPruneResult    PruneResult
	sinkBytes          []byte
	sinkStrings        []string
	sinkBreakpointPlan BreakpointPlan
	sinkBool           bool
	sinkString         string
	sinkInt            int
	sinkAudit          SerializationAudit
	sinkToolPlan       ToolPlan
)

func makeBenchmarkTools(n int, breakpointIdx int) []map[string]any {
	tools := make([]map[string]any, n)
	for i := 0; i < n; i++ {
		tools[i] = tool(fmt.Sprintf("tool_%03d", i), i == breakpointIdx)
	}
	return tools
}

func makeBenchmarkSystemBlocks(n int, breakpointIdx int) []map[string]any {
	blocks := make([]map[string]any, n)
	for i := 0; i < n; i++ {
		blockType := "skills"
		if i == 0 {
			blockType = "system"
		} else if i%3 == 0 {
			blockType = "memory"
		}
		blocks[i] = systemBlock(
			fmt.Sprintf("block_%03d", i),
			blockType,
			fmt.Sprintf("Instruction content for block %d with detailed directives.", i),
			i == breakpointIdx,
		)
	}
	return blocks
}

// BenchmarkCompactInboundTools benchmarks page translation and tail compaction
// of tools arrays while proving byte-exact prefix preservation.
func BenchmarkCompactInboundTools(b *testing.B) {
	sizes := []struct {
		name     string
		count    int
		bpIdx    int
		dropFrom int
	}{
		{"Small_3Tools", 3, 0, 2},
		{"Medium_20Tools", 20, 5, 10},
		{"Large_100Tools", 100, 20, 50},
	}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			tools := makeBenchmarkTools(tc.count, tc.bpIdx)
			raw := body(b, tools, false)
			dropNames := make(map[string]bool)
			for i := tc.dropFrom; i < tc.count; i++ {
				dropNames[fmt.Sprintf("tool_%03d", i)] = true
			}
			plan := ToolPlan{Drop: dropNames}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkPruneResult = CompactInboundTools(raw, plan, nil)
			}
		})
	}

	b.Run("WithDecodeRecheck_Medium20Tools", func(b *testing.B) {
		tools := makeBenchmarkTools(20, 5)
		raw := body(b, tools, false)
		dropNames := make(map[string]bool)
		for i := 10; i < 20; i++ {
			dropNames[fmt.Sprintf("tool_%03d", i)] = true
		}
		plan := ToolPlan{Drop: dropNames}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkPruneResult = CompactInboundTools(raw, plan, okDecode)
		}
	})
}

// BenchmarkCompactInboundSystem benchmarks system block page translation and compaction,
// pruning non-cached segments strictly after the cache_control breakpoint.
func BenchmarkCompactInboundSystem(b *testing.B) {
	sizes := []struct {
		name     string
		count    int
		bpIdx    int
		dropFrom int
	}{
		{"Small_5Blocks", 5, 1, 3},
		{"Medium_25Blocks", 25, 5, 12},
	}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			blocks := makeBenchmarkSystemBlocks(tc.count, tc.bpIdx)
			raw := systemBody(b, blocks)
			dropNames := make(map[string]bool)
			for i := tc.dropFrom; i < tc.count; i++ {
				dropNames[fmt.Sprintf("block_%03d", i)] = true
			}
			plan := BlockPlan{Block: "skills", Drop: dropNames}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkPruneResult = CompactInboundSystem(raw, plan, nil)
			}
		})
	}
}

// BenchmarkArraySplicePoints benchmarks address lookup and breakpoint boundary resolution
// within serialized JSON payloads.
func BenchmarkArraySplicePoints(b *testing.B) {
	tools := makeBenchmarkTools(20, 5)
	toolsRaw := body(b, tools, false)

	blocks := makeBenchmarkSystemBlocks(20, 5)
	sysRaw := systemBody(b, blocks)

	b.Run("ToolsArray_Lookup", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var ok bool
			sinkInt, sinkInt, sinkInt, ok = ArraySplicePoints(toolsRaw, "tools")
			if !ok {
				b.Fatal("ArraySplicePoints failed")
			}
		}
	})

	b.Run("SystemArray_Lookup", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var ok bool
			sinkInt, sinkInt, sinkInt, ok = ArraySplicePoints(sysRaw, "system")
			if !ok {
				b.Fatal("ArraySplicePoints failed")
			}
		}
	})

	b.Run("WithReason_Lookup", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkInt, sinkInt, sinkInt, sinkString = ArraySplicePointsWithReason(toolsRaw, "tools")
			if sinkString != ArrayOffsetsResolved {
				b.Fatal("ArraySplicePointsWithReason failed")
			}
		}
	})
}

// BenchmarkPlanBreakpoints benchmarks translating segment sequences and cache stability
// state into memory page residency plans (ProtectedPrefix, MutableTail, UnsafeToCompact).
func BenchmarkPlanBreakpoints(b *testing.B) {
	makeSegments := func(n int) []cachemeta.PromptSegment {
		segs := make([]cachemeta.PromptSegment, n)
		for i := 0; i < n; i++ {
			kind := cachemeta.SegMessage
			if i == 0 {
				kind = cachemeta.SegStable
			} else if i == 1 {
				kind = cachemeta.SegToolSchema
			} else if i == n/2 {
				kind = cachemeta.SegSealed
			}
			segs[i] = cachemeta.PromptSegment{
				Kind:    kind,
				Tokens:  int64(20 + i),
				Content: []byte(fmt.Sprintf("segment_content_%03d", i)),
			}
		}
		return segs
	}

	for _, count := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("%d_Segments", count), func(b *testing.B) {
			baseline := makeSegments(count)
			tracker := cachemeta.NewPrefixStabilityTracker("5m", abi.ScopeAgent)
			tracker.Observe(baseline)

			mutated := makeSegments(count)
			mutated[count/2].Content = []byte("mutated_content")
			score := tracker.Observe(mutated)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBreakpointPlan = PlanBreakpoints(mutated, score)
			}
		})
	}
}

// BenchmarkCheckStableEdit benchmarks validating proposed edit address spans
// against protected prefix and sealed memory segments.
func BenchmarkCheckStableEdit(b *testing.B) {
	plan := BreakpointPlan{
		ProtectedPrefix: Span{Start: 0, End: 10},
		MutableTail:     Span{Start: 10, End: 30},
		UnsafeToCompact: []UnsafeSpan{
			{Span: Span{Start: 15, End: 17}, Kind: cachemeta.SegSealed, Reason: "sealed-quarantine"},
			{Span: Span{Start: 10, End: 11}, Kind: cachemeta.SegMessage, Reason: "first-divergent"},
		},
		State: cachemeta.PrefixMutated,
	}

	b.Run("SafeMutableTailEdit", func(b *testing.B) {
		edit := Span{Start: 12, End: 14}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool, sinkString = CheckStableEdit(plan, edit)
		}
	})

	b.Run("UnsafeProtectedPrefixEdit", func(b *testing.B) {
		edit := Span{Start: 8, End: 12}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool, sinkString = CheckStableEdit(plan, edit)
		}
	})

	b.Run("UnsafeSealedSpanEdit", func(b *testing.B) {
		edit := Span{Start: 16, End: 18}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool, sinkString = CheckStableEdit(plan, edit)
		}
	})
}

// BenchmarkFaultedToolTracker_PruneColdTools benchmarks turn-decay eviction of cold tools
// from the prompt context memory tracker.
func BenchmarkFaultedToolTracker_PruneColdTools(b *testing.B) {
	const batchSize = 64
	initTracker := func() *FaultedToolTracker {
		tr := NewFaultedToolTracker(5)
		for j := 0; j < 50; j++ {
			name := fmt.Sprintf("cold_tool_%02d", j)
			tr.RecordFault(name, 1)
			if j%2 == 0 {
				tr.RecordInvocation(name, 10)
			}
		}
		return tr
	}

	trackers := make([]*FaultedToolTracker, batchSize)
	for k := range trackers {
		trackers[k] = initTracker()
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % batchSize
		if idx == 0 && i > 0 {
			b.StopTimer()
			for k := range trackers {
				trackers[k] = initTracker()
			}
			b.StartTimer()
		}
		sinkStrings = trackers[idx].PruneColdTools(10)
	}
}

// BenchmarkFaultedToolTracker_RecordLifecycle benchmarks tracking cold tool faults,
// invocations, and retrieving active tool names.
func BenchmarkFaultedToolTracker_RecordLifecycle(b *testing.B) {
	tracker := NewFaultedToolTracker(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolName := fmt.Sprintf("tool_%d", i%30)
		tracker.RecordFault(toolName, i)
		tracker.RecordInvocation(toolName, i+1)
		if i%10 == 0 {
			sinkStrings = tracker.ActiveFaultedTools()
		}
	}
}

// BenchmarkMinifyToolSchema benchmarks compressing verbose OpenAI-format tool schemas
// by removing redundant titles, empty defaults/examples, and truncating descriptions.
func BenchmarkMinifyToolSchema(b *testing.B) {
	raw := []byte(`{
		"type": "function",
		"function": {
			"name": "benchmark_tool_operation",
			"title": "Benchmark Tool Operation That Contains Redundant Title Information",
			"description": "This is a verbose tool description explaining the function parameters, purpose, and invocation contract in extensive detail that easily exceeds eighty characters.",
			"parameters": {
				"type": "object",
				"title": "Parameters Object Title",
				"properties": {
					"targetPath": {
						"type": "string",
						"title": "Target File Path",
						"description": "The absolute or relative file path to the destination target file that will be accessed by the tool.",
						"default": null,
						"examples": []
					},
					"options": {
						"type": "object",
						"title": "Execution Options",
						"properties": {
							"timeoutMs": {"type": "integer", "default": 5000},
							"retries": {"type": "integer", "default": null}
						}
					}
				},
				"required": ["targetPath"]
			}
		}
	}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		sinkBytes, err = MinifyToolSchema(raw)
		if err != nil {
			b.Fatalf("MinifyToolSchema failed: %v", err)
		}
	}
}

// BenchmarkThinLocalTools benchmarks filtering inbound tool schemas down to hot tools
// plus faulted cold tools, injecting fak_tools_search for evicted definitions.
func BenchmarkThinLocalTools(b *testing.B) {
	entries := make([]map[string]any, 20)
	for i := 0; i < 20; i++ {
		name := fmt.Sprintf("tool_%02d", i)
		if i < 6 {
			name = DefaultHotTools[i]
		}
		entries[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        name,
				"title":       "Title for " + name,
				"description": "Description for tool " + name + " with some extra text for testing.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"arg": map[string]any{"type": "string"},
					},
				},
			},
		}
	}
	rawTools, err := json.Marshal(entries)
	if err != nil {
		b.Fatalf("marshal tools: %v", err)
	}

	hot := DefaultHotTools
	faulted := []string{"tool_06", "tool_07"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBytes, sinkStrings, err = ThinLocalTools(rawTools, hot, faulted)
		if err != nil {
			b.Fatalf("ThinLocalTools failed: %v", err)
		}
	}
}

// BenchmarkFaultInColdTools benchmarks keyword search and fault-in mapping
// over cold tool definitions.
func BenchmarkFaultInColdTools(b *testing.B) {
	catalog := make(map[string]string, 100)
	for i := 0; i < 100; i++ {
		catalog[fmt.Sprintf("tool_%03d", i)] = fmt.Sprintf("Performs operation %d including file search and database access.", i)
	}

	baseActive := map[string]bool{
		"read":  true,
		"write": true,
		"edit":  true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		active := make(map[string]bool, len(baseActive)+4)
		for k, v := range baseActive {
			active[k] = v
		}
		sinkStrings = FaultInColdTools("database access", catalog, active)
	}
}

// BenchmarkToolPlanForRequest benchmarks policy-to-plan mapping combining advertised
// tools, unconditional denial closures, and agent self-drop requests.
func BenchmarkToolPlanForRequest(b *testing.B) {
	advertised := make([]string, 30)
	for i := 0; i < 30; i++ {
		advertised[i] = fmt.Sprintf("tool_%02d", i)
	}
	selfDrop := []string{"tool_05", "tool_10", "tool_15"}
	req := ToolPlanRequest{
		Advertised: advertised,
		SelfDrop:   selfDrop,
	}
	denies := func(tool string) bool {
		return tool == "tool_01" || tool == "tool_02" || tool == "tool_20"
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkToolPlan = ToolPlanForRequest(req, denies)
	}
}

// BenchmarkAuditSerialization benchmarks byte-by-byte address scanning to detect
// prompt serialization divergence and isolate changed byte ranges.
func BenchmarkAuditSerialization(b *testing.B) {
	orig := []byte(`{"model":"claude-3-opus","system":[{"type":"text","text":"you are an agent with long instructions that remain stable across multiple turns"}],"messages":[{"role":"user","content":"turn 1"}]}`)
	mutated := []byte(`{"model":"claude-3-opus","system":[{"type":"text","text":"you are an agent with long instructions that remain stable across multiple turns"}],"messages":[{"role":"user","content":"turn 2 - edited"}]}`)

	b.Run("IdenticalPrefixAndBody", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkAudit = AuditSerialization(orig, orig)
		}
	})

	b.Run("DivergentTailRange", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkAudit = AuditSerialization(orig, mutated)
		}
	})
}
