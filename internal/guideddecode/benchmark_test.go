// Package guideddecode benchmarks measure vocabulary mask computation,
// bitset matching throughput, envelope byte walking, and reasoning
// boundary latch emissions for constrained token generation.
package guideddecode

import (
	"fmt"
	"strconv"
	"testing"
)

var (
	sinkAllowed map[byte]bool
	sinkBitset  ByteBitset
	sinkUnc     bool
	sinkBool    bool
	sinkSlice   []bool
)

// BenchmarkAllowedNextBytes_PrefixPhases measures mask computation across each
// structural region of tool-call envelope decoding.
// Operating envelope: 6 structural decode stages (PreSkeleton to DeadEnd).
// Allocation budget: <= 2 allocs/op for map return.
// Latency ceiling: P50 < 300ns, P99 < 1.5µs across envelope stages.
func BenchmarkAllowedNextBytes_PrefixPhases(b *testing.B) {
	schema := ToolSchema{
		Names: []string{"get_weather", "get_forecast", "list_files", "read_file", "search_web"},
	}

	cases := []struct {
		name   string
		prefix string
	}{
		{"PreSkeleton", `{"na`},
		{"EnumBranch", preLit},
		{"MidName", preLit + "get_"},
		{"SuffixSkeleton", preLit + "get_weather" + `","arg`},
		{"Unconstrained", preLit + "get_weather" + sufLit},
		{"DeadEnd", preLit + "unknown_tool"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			prefix := []byte(tc.prefix)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkAllowed = AllowedNextBytes(prefix, schema)
			}
		})
	}
}

// BenchmarkAllowedNextByteBitset_PrefixPhases measures zero-allocation bitset
// mask computation across each structural region of tool-call envelope decoding.
// Operating envelope: 6 structural decode stages from prefix to unconstrained payload.
// Allocation budget: 0 allocs/op for bitset mask calculation.
// Latency ceiling: P50 < 80ns, P99 < 400ns.
func BenchmarkAllowedNextByteBitset_PrefixPhases(b *testing.B) {
	schema := ToolSchema{
		Names: []string{"get_weather", "get_forecast", "list_files", "read_file", "search_web"},
	}

	cases := []struct {
		name   string
		prefix string
	}{
		{"PreSkeleton", `{"na`},
		{"EnumBranch", preLit},
		{"MidName", preLit + "get_"},
		{"SuffixSkeleton", preLit + "get_weather" + `","arg`},
		{"Unconstrained", preLit + "get_weather" + sufLit},
		{"DeadEnd", preLit + "unknown_tool"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			prefix := []byte(tc.prefix)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkBitset, sinkUnc = AllowedNextByteBitset(prefix, schema)
			}
		})
	}
}

// BenchmarkAllowedNextBytes_SchemaScaling measures matcher performance at the
// tool-name selection branch as the registered tool schema scales from 5 to 100 tools.
// Operating envelope: tool schema scaling across 5, 25, and 100 registered tools.
// Allocation budget: <= 2 allocs/op for returned mask map.
// Latency ceiling: P50 < 500ns, P99 < 2.5µs at 100 tools.
func BenchmarkAllowedNextBytes_SchemaScaling(b *testing.B) {
	for _, count := range []int{5, 25, 100} {
		b.Run(fmt.Sprintf("%d_tools", count), func(b *testing.B) {
			names := make([]string, count)
			for i := 0; i < count; i++ {
				names[i] = "tool_service_endpoint_" + strconv.Itoa(i)
			}
			schema := ToolSchema{Names: names}
			prefix := []byte(preLit + "tool_service_endpoint_")

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkAllowed = AllowedNextBytes(prefix, schema)
			}
		})
	}
}

// BenchmarkAllowedNextBytes_EnvelopeByteWalk simulates a full production decode
// loop, stepping byte-by-byte through an entire valid envelope prefix until
// the unconstrained arguments payload is reached.
// Operating envelope: byte-by-byte iteration over full valid envelope prefix (~65 bytes).
// Allocation budget: 0 allocs/op across all byte positions.
// Latency ceiling: P50 < 2µs, P99 < 10µs for complete envelope walk.
func BenchmarkAllowedNextBytes_EnvelopeByteWalk(b *testing.B) {
	schema := ToolSchema{
		Names: []string{"get_weather", "get_forecast", "list_files"},
	}
	envelope := []byte(preLit + "get_weather" + sufLit + `{"city":"San Francisco"}}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j <= len(envelope); j++ {
			sinkBitset, sinkUnc = AllowedNextByteBitset(envelope[:j], schema)
		}
	}
}

// BenchmarkAllowedNextBytes_EnvelopeByteWalk_MapLegacy benchmarks the legacy map-allocating
// return path for historical comparison.
// Operating envelope: byte-by-byte envelope walk with legacy map allocation.
// Allocation budget: <= 2 allocs/byte.
// Latency ceiling: P50 < 10µs, P99 < 50µs.
func BenchmarkAllowedNextBytes_EnvelopeByteWalk_MapLegacy(b *testing.B) {
	schema := ToolSchema{
		Names: []string{"get_weather", "get_forecast", "list_files"},
	}
	envelope := []byte(preLit + "get_weather" + sufLit + `{"city":"San Francisco"}}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j <= len(envelope); j++ {
			sinkAllowed = AllowedNextBytes(envelope[:j], schema)
		}
	}
}

// BenchmarkMaskArm_Emit benchmarks streaming token emission through the reasoning
// boundary latch in pre-marker, transition, post-marker, and split-boundary conditions.
// Operating envelope: single token evaluation across thinking, transition, and post-marker phases.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 40ns, P99 < 200ns.
func BenchmarkMaskArm_Emit(b *testing.B) {
	b.Run("PreMarkerThinking", func(b *testing.B) {
		a := NewMaskArm("</think>", InactiveUntilMarker)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool = a.Emit("thinking")
		}
	})

	b.Run("MarkerTransition", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			a := NewMaskArm("</think>", InactiveUntilMarker)
			a.Emit("step")
			sinkBool = a.Emit("</think>")
		}
	})

	b.Run("PostMarkerEmissions", func(b *testing.B) {
		a := NewMaskArm("</think>", InactiveUntilMarker)
		a.Emit("</think>")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkBool = a.Emit("token")
		}
	})

	b.Run("SplitMarkerTokens", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			a := NewMaskArm("</think>", InactiveUntilMarker)
			_ = a.Emit("preface ")
			_ = a.Emit("</thi")
			sinkBool = a.Emit("nk>")
		}
	})
}

// BenchmarkMaskActiveByToken benchmarks whole-stream token array evaluation
// across various stream lengths and marker configurations.
// Operating envelope: stream lengths of 10, 100, and 500 tokens.
// Allocation budget: <= 1 alloc/op for active token slice.
// Latency ceiling: P50 < 1µs, P99 < 5µs at 100 tokens.
func BenchmarkMaskActiveByToken(b *testing.B) {
	for _, size := range []int{10, 100, 500} {
		b.Run(fmt.Sprintf("%d_tokens_with_reasoning", size), func(b *testing.B) {
			tokens := make([]string, size)
			for i := 0; i < size; i++ {
				tokens[i] = "tok"
			}
			tokens[0] = "<think>"
			markerPos := size / 2
			tokens[markerPos] = "</think>"
			tokens[markerPos+1] = `{"name":"`

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkSlice = MaskActiveByToken(tokens, "</think>", InactiveUntilMarker)
			}
		})
	}

	b.Run("100_tokens_markerless_active", func(b *testing.B) {
		tokens := make([]string, 100)
		for i := range tokens {
			tokens[i] = "tok"
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkSlice = MaskActiveByToken(tokens, "</think>", ActiveWhenNoMarker)
		}
	})
}

// BenchmarkGuidedDecode_Parallel evaluates concurrent AllowedNextBytes computation across
// multiple worker goroutines.
// Operating envelope: parallel evaluation across GOMAXPROCS workers on structured tool prefixes.
// Allocation budget: <= 2 allocs/op per mask calculation.
// Latency ceiling: P50 < 300ns/op, P99 < 1.5µs/op under concurrent execution.
func BenchmarkGuidedDecode_Parallel(b *testing.B) {
	schema := ToolSchema{
		Names: []string{"get_weather", "get_forecast", "list_files", "read_file", "search_web"},
	}
	prefix := []byte(preLit + "get_")
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res := AllowedNextBytes(prefix, schema)
			if len(res) == 0 {
				b.Fatal("unexpected empty allowed next bytes")
			}
		}
	})
}

// TestBenchmarksRun executes core benchmarks for a small iteration count to guarantee
// they complete without panicking.
func TestBenchmarksRun(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark runner in -short mode")
	}

	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"BenchmarkAllowedNextBytes_PrefixPhases", BenchmarkAllowedNextBytes_PrefixPhases},
		{"BenchmarkAllowedNextByteBitset_PrefixPhases", BenchmarkAllowedNextByteBitset_PrefixPhases},
		{"BenchmarkMaskArm_Emit", BenchmarkMaskArm_Emit},
		{"BenchmarkGuidedDecode_Parallel", BenchmarkGuidedDecode_Parallel},
	}

	for _, bm := range benchmarks {
		t.Run(bm.name, func(t *testing.T) {
			res := testing.Benchmark(bm.fn)
			if res.N <= 0 {
				t.Fatalf("benchmark %s performed 0 iterations", bm.name)
			}
		})
	}
}
