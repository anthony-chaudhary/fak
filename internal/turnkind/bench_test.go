package turnkind

import "testing"

// BenchmarkTurnKind exercises the turn kind classification logic across
// representative turn profiles (mechanical continuation, new ask, error, mixed).
func BenchmarkTurnKind(b *testing.B) {
	cases := [][]Block{
		{{Type: BlockToolResult}},
		{{Type: BlockToolResult}, {Type: BlockToolResult}, {Type: BlockToolResult}},
		{{Type: BlockToolResult}, {Type: BlockText}},
		{{Type: BlockToolResult, IsError: true}},
		{{Type: BlockText}},
		{{Type: "unrecognized_custom_block"}},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, c := range cases {
			_ = Classify(c)
		}
	}
}
