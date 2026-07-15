package scorecardpane

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

const NegframeTextWidth = 72

type NegframePaneRow struct {
	Path     string `json:"path"`
	Line     int    `json:"line"`
	Category string `json:"category"`
	Tier     string `json:"tier"`
	Text     string `json:"text"`
}
type NegframePane struct {
	Debt int               `json:"debt"`
	Rows []NegframePaneRow `json:"rows"`
}

func BuildNegframePane(findings []negframe.Finding, topN int) NegframePane {
	var p NegframePane
	ordered := append([]negframe.Finding(nil), findings...)
	for _, f := range ordered {
		if f.Mechanical() {
			p.Debt++
		}
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		mi, mj := ordered[i].Mechanical(), ordered[j].Mechanical()
		if mi != mj {
			return mi
		}
		if ordered[i].Path != ordered[j].Path {
			return ordered[i].Path < ordered[j].Path
		}
		return ordered[i].Line < ordered[j].Line
	})
	if topN <= 0 || topN > len(ordered) {
		topN = len(ordered)
	}
	for _, f := range ordered[:topN] {
		p.Rows = append(p.Rows, NegframePaneRow{Path: f.Path, Line: f.Line, Category: string(f.Category), Tier: tierName(f), Text: clipPaneText(f.Text, NegframeTextWidth)})
	}
	return p
}
func tierName(f negframe.Finding) string {
	if f.Mechanical() {
		return "MECHANICAL"
	}
	return "JUDGEMENT"
}
func clipPaneText(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	if n < 4 {
		return s[:n]
	}
	return s[:n-3] + "..."
}
func RenderNegframePane(p NegframePane) string {
	var b strings.Builder
	fmt.Fprintf(&b, "negation-tax debt=%d", p.Debt)
	for _, r := range p.Rows {
		fmt.Fprintf(&b, "\n  %s %s:%d [%s] %s", r.Tier, r.Path, r.Line, r.Category, r.Text)
	}
	return b.String()
}
