package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func checkQwen35PromptPanelEnds(t *testing.T, n, maxW int, ends []int) {
	t.Helper()
	if n < qwen35PromptPanelQuantum {
		if ends != nil {
			t.Fatalf("n=%d: want nil, got %v", n, ends)
		}
		return
	}
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
		{256, []int{256}},
		{257, []int{129, 257}},
		{482, []int{241, 482}},
		{497, []int{249, 497}},
		{512, []int{256, 512}},
		{1000, []int{250, 500, 750, 1000}},
		{2050, nil},
		{4224, nil},
	}
	for _, c := range cases {
		got := qwen35PromptPanelEnds(c.n, 256)
		checkQwen35PromptPanelEnds(t, c.n, 256, got)
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
	for _, maxW := range []int{qwen35PromptPanelNarrowRows, metalgemm.PromptPanelMaxTokens} {
		for n := 0; n <= 9000; n++ {
			checkQwen35PromptPanelEnds(t, n, maxW, qwen35PromptPanelEnds(n, maxW))
		}
	}
}

// TestQwen35PromptPanelMaxRowsGatesByModelSize pins the command-buffer-time gate:
// Qwen3.5-4B-class models take the full native ceiling, larger ones (Qwen3.5-9B,
// Qwen3.8-27B) keep 128-row panels, and no width ever exceeds the native guard.
func TestQwen35PromptPanelMaxRowsGatesByModelSize(t *testing.T) {
	for _, c := range []struct {
		name           string
		layers, hidden int
		want           int
	}{
		{"synthetic", 4, 256, metalgemm.PromptPanelMaxTokens},
		{"qwen3.5-4b", 32, 2560, metalgemm.PromptPanelMaxTokens},
		{"qwen3.5-9b", 32, 4096, qwen35PromptPanelNarrowRows},
		{"qwen3.8-27b", 64, 5120, qwen35PromptPanelNarrowRows},
	} {
		got := qwen35PromptPanelMaxRows(Config{NumLayers: c.layers, HiddenSize: c.hidden})
		if got != c.want || got > metalgemm.PromptPanelMaxTokens {
			t.Fatalf("%s: max rows=%d, want %d (native ceiling %d)", c.name, got, c.want, metalgemm.PromptPanelMaxTokens)
		}
	}
	if got := qwen35PromptPanelEnds(512, qwen35PromptPanelMaxRows(Config{NumLayers: 64, HiddenSize: 5120})); len(got) != 4 {
		t.Fatalf("27B-class 512-token prompt panels=%v, want four 128-row panels", got)
	}
}
