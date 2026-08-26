package agent

import (
	"encoding/json"
	"testing"
)

func TestEstimatedElementTokensBoundsRange(t *testing.T) {
	elems := []json.RawMessage{
		json.RawMessage(`{"role":"user","content":"first"}`),
		json.RawMessage(`{"role":"assistant","content":"second response"}`),
		json.RawMessage(`{"role":"user","content":"third"}`),
	}
	all := estimateElementTokens(elems[0]) + estimateElementTokens(elems[1]) + estimateElementTokens(elems[2])

	tests := []struct {
		name       string
		start, end int
		want       int
	}{
		{name: "all with outer bounds", start: -4, end: 20, want: all},
		{name: "middle", start: 1, end: 2, want: estimateElementTokens(elems[1])},
		{name: "anchor before messages", start: 0, end: 0, want: 0},
		{name: "inverted", start: 2, end: 1, want: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := estimatedElementTokens(elems, test.start, test.end); got != test.want {
				t.Fatalf("estimatedElementTokens(%d, %d) = %d, want %d", test.start, test.end, got, test.want)
			}
		})
	}
}
