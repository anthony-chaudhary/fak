package turnkind

import "testing"

// TestClassify table-tests the four structural shapes plus the precedence
// rules (#3307): a new ask outranks everything, an errored tool_result
// outranks mechanical, and purity is required for the mechanical verdict.
func TestClassify(t *testing.T) {
	cases := []struct {
		name   string
		blocks []Block
		want   Kind
	}{
		// The four base shapes.
		{name: "nil blocks", blocks: nil, want: KindUnknown},
		{name: "empty blocks", blocks: []Block{}, want: KindUnknown},
		{name: "only text", blocks: []Block{{Type: BlockText}}, want: KindNewAsk},
		{name: "only image", blocks: []Block{{Type: BlockImage}}, want: KindNewAsk},
		{name: "only document", blocks: []Block{{Type: BlockDocument}}, want: KindNewAsk},
		{name: "single clean tool_result", blocks: []Block{{Type: BlockToolResult}}, want: KindMechanical},
		{
			name:   "several clean tool_results",
			blocks: []Block{{Type: BlockToolResult}, {Type: BlockToolResult}, {Type: BlockToolResult}},
			want:   KindMechanical,
		},
		{
			name:   "single errored tool_result",
			blocks: []Block{{Type: BlockToolResult, IsError: true}},
			want:   KindErrorContinuation,
		},

		// Precedence: a new ask wins outright, even over an error.
		{
			name:   "text plus clean tool_result",
			blocks: []Block{{Type: BlockToolResult}, {Type: BlockText}},
			want:   KindNewAsk,
		},
		{
			name:   "text plus errored tool_result",
			blocks: []Block{{Type: BlockToolResult, IsError: true}, {Type: BlockText}},
			want:   KindNewAsk,
		},
		{
			name:   "image plus clean tool_results",
			blocks: []Block{{Type: BlockToolResult}, {Type: BlockImage}, {Type: BlockToolResult}},
			want:   KindNewAsk,
		},

		// Precedence: one error taints the whole continuation.
		{
			name:   "errored plus clean tool_results",
			blocks: []Block{{Type: BlockToolResult}, {Type: BlockToolResult, IsError: true}, {Type: BlockToolResult}},
			want:   KindErrorContinuation,
		},
		{
			name:   "errored tool_result plus unrecognized block",
			blocks: []Block{{Type: "tool_use"}, {Type: BlockToolResult, IsError: true}},
			want:   KindErrorContinuation,
		},

		// Purity: an unrecognized block demotes a would-be mechanical turn.
		{name: "only tool_use", blocks: []Block{{Type: "tool_use"}}, want: KindUnknown},
		{
			name:   "clean tool_result plus unrecognized block",
			blocks: []Block{{Type: BlockToolResult}, {Type: "some_future_block"}},
			want:   KindUnknown,
		},

		// IsError is meaningless off a tool_result: it must not taint or rescue.
		{
			name:   "unrecognized block with stray IsError",
			blocks: []Block{{Type: "tool_use", IsError: true}},
			want:   KindUnknown,
		},
		{
			name:   "text block with stray IsError still a new ask",
			blocks: []Block{{Type: BlockText, IsError: true}},
			want:   KindNewAsk,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.blocks); got != tc.want {
				t.Fatalf("Classify(%v) = %v, want %v", tc.blocks, got, tc.want)
			}
		})
	}
}

// TestKindString pins the stable shadow-log labels, including the fallback
// for an out-of-range Kind.
func TestKindString(t *testing.T) {
	cases := []struct {
		k    Kind
		want string
	}{
		{KindUnknown, "unknown"},
		{KindNewAsk, "new_ask"},
		{KindErrorContinuation, "error_continuation"},
		{KindMechanical, "mechanical"},
		{Kind(99), "unknown"},
	}
	for _, tc := range cases {
		if got := tc.k.String(); got != tc.want {
			t.Errorf("Kind(%d).String() = %q, want %q", int(tc.k), got, tc.want)
		}
	}
}
