package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func checkQwen35PromptPanelEnds(t *testing.T, n int, ends []int) {
	t.Helper()
	if n < qwen35PromptPanelQuantum {
		if ends != nil {
			t.Fatalf("n=%d: want nil, got %v", n, ends)
		}
		return
	}
	maxW := metalgemm.PromptPanelMaxTokens
	if want := (n + maxW - 1) / maxW; len(ends) != want {
		t.Fatalf("n=%d: panels=%d want %d (%v)", n, len(ends), want, ends)
	}
	if ends[len(ends)-1] != n {
		t.Fatalf("n=%d: last end %d != n", n, ends[len(ends)-1])
	}
	lo, hi, prev := n+1, -1, 0
	for i, e := range ends {
		w := e - prev
		if w < qwen35PromptPanelQuantum || w > maxW {
			t.Fatalf("n=%d: panel %d width %d outside [%d,%d] (%v)", n, i, w, qwen35PromptPanelQuantum, maxW, ends)
		}
		lo, hi, prev = min(lo, w), max(hi, w), e
	}
	if hi-lo > 1 {
		t.Fatalf("n=%d: widths span %d..%d (%v)", n, lo, hi, ends)
	}
}

func TestQwen35PromptPanelEndsTable(t *testing.T) {
	cases := []struct {
		n    int
		want []int
	}{
		{31, nil},
		{32, []int{32}},
		{33, []int{33}},
		{127, []int{127}},
		{128, []int{128}},
		{129, []int{65, 129}},
		{482, []int{121, 242, 362, 482}},
		{497, []int{125, 249, 373, 497}},
		{512, []int{128, 256, 384, 512}},
		{2050, nil},
		{4224, nil},
	}
	for _, c := range cases {
		got := qwen35PromptPanelEnds(c.n)
		checkQwen35PromptPanelEnds(t, c.n, got)
		if c.want == nil {
			continue
		}
		if len(got) != len(c.want) {
			t.Fatalf("n=%d: got %v want %v", c.n, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("n=%d: got %v want %v", c.n, got, c.want)
			}
		}
	}
}

func TestQwen35PromptPanelEndsExhaustive(t *testing.T) {
	for n := 0; n <= 9000; n++ {
		checkQwen35PromptPanelEnds(t, n, qwen35PromptPanelEnds(n))
	}
}
