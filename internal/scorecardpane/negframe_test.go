package scorecardpane

import (
	"github.com/anthony-chaudhary/fak/internal/negframe"
	"strings"
	"testing"
)

func TestNegframePaneOrdersClipsAndRenders(t *testing.T) {
	fs := []negframe.Finding{{Path: "z.md", Line: 2, Category: negframe.Prohibition, Text: strings.Repeat("x", 100), Hint: "h"}, {Path: "b.md", Line: 3, Category: negframe.Prohibition, Text: "mechanical b", Suggest: "remember"}, {Path: "a.md", Line: 9, Category: negframe.Prohibition, Text: "mechanical a", Suggest: "remember"}}
	p := BuildNegframePane(fs, 3)
	if p.Debt != 2 || p.Rows[0].Path != "a.md" || p.Rows[1].Path != "b.md" || p.Rows[2].Tier != "JUDGEMENT" {
		t.Fatalf("pane=%+v", p)
	}
	if len(p.Rows[2].Text) > NegframeTextWidth || !strings.HasSuffix(p.Rows[2].Text, "...") {
		t.Fatalf("clip=%q", p.Rows[2].Text)
	}
	got := RenderNegframePane(p)
	for _, want := range []string{"negation-tax debt=2", "MECHANICAL a.md:9", "JUDGEMENT z.md:2"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q: %s", want, got)
		}
	}
}
func TestNegframePaneEmptyIsCleanZero(t *testing.T) {
	p := BuildNegframePane(nil, 5)
	if p.Debt != 0 || len(p.Rows) != 0 || RenderNegframePane(p) != "negation-tax debt=0" {
		t.Fatalf("pane=%+v render=%q", p, RenderNegframePane(p))
	}
}
