package compute

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestPrefillBatchPositionRange(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name          string
		start, tokens int
		reject        bool
		infer         bool
	}{
		{"exclusive-end-overflow", maxInt, 1, true, false},
		{"last-token-overflow", maxInt, 2, true, false},
		{"largest-representable-end", maxInt - 2, 2, false, false},
		{"ordinary", 3, 2, false, false},
		{"inferred-token-overflow", maxInt, 2, true, true},
		{"inferred-token-boundary", maxInt - 2, 2, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := NewF32(Default(), []int{tc.tokens, 2}, make([]float32, tc.tokens*2))
			w := NewF32(Default(), []int{2, 2}, []float32{1, 0, 0, 1})
			args := PrefillBatchArgs{X: x, Wq: w, Wk: w, Wv: w, NumHeads: 1, NumKVHeads: 1, HeadDim: 2, Tokens: tc.tokens, StartPos: tc.start}
			if tc.infer {
				args.Tokens = 0
			}
			_, _, err := validatePrefillBatchArgs(&args)
			if (err != nil) != tc.reject {
				t.Fatalf("rejected=%v, want %v (error=%v)", err != nil, tc.reject, err)
			}
		})
	}
}
