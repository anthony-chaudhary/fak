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
