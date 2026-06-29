package supportmaturityscore

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
)

// Row is the per-cell support read-out: one model-family x backend cell folded
// across the scorecard grid, the support-maturity ladder, the declared target,
// and the routed next action.
type Row struct {
	Family     string `json:"family"`
	Backend    string `json:"backend"`
	Support    string `json:"support"`
	Rung       string `json:"rung"`
	RungLabel  string `json:"rung_label"`
	Witness    string `json:"witness"`
	Regime     string `json:"regime"`
	RegimeName string `json:"regime_name"`
	Target     string `json:"target"`
	TargetName string `json:"target_name"`
	Loop       string `json:"loop"`
	NextAction string `json:"next_action"`
}

// ReadOut folds the live coverage grid into the per-cell support read-out. It is
// deterministic because covmatrix.Grid is deterministic and every lowering it composes is
// total.
func ReadOut() []Row {
	cells := covmatrix.Grid()
	rows := make([]Row, 0, len(cells))
	for _, c := range cells {
		rung := supportmaturity.FromSupport(c.Support)
		target := declaredTarget(c)
		na := supportmaturity.NextActionFor(rung)
		rows = append(rows, Row{
			Family:     c.Family,
			Backend:    c.Backend,
			Support:    string(c.Support),
			Rung:       rung.String(),
			RungLabel:  rung.Label(),
			Witness:    supportmaturity.WitnessFor(rung).String(),
			Regime:     na.Regime.String(),
			RegimeName: na.Regime.Name(),
			Target:     target.String(),
			TargetName: target.Label(),
			Loop:       na.Loop.String(),
			NextAction: na.Action,
		})
	}
	return rows
}

// FilterReadOut narrows rows to a family substring and/or backend exact match. Empty
// filters match everything.
func FilterReadOut(rows []Row, family, backend string) []Row {
	family = strings.ToLower(strings.TrimSpace(family))
	backend = strings.ToLower(strings.TrimSpace(backend))
	if family == "" && backend == "" {
		return rows
	}
	out := make([]Row, 0, len(rows))
	for _, r := range rows {
		if family != "" && !strings.Contains(strings.ToLower(r.Family), family) {
			continue
		}
		if backend != "" && strings.ToLower(r.Backend) != backend {
			continue
		}
		out = append(out, r)
	}
	return out
}

// RenderReadOut renders rows as a deterministic ASCII table for the human `fak support`
// view.
func RenderReadOut(rows []Row) string {
	header := []string{"FAMILY", "BACKEND", "RUNG", "REGIME", "TARGET", "WITNESS", "NEXT-ACTION"}
	cols := func(r Row) []string {
		return []string{
			r.Family,
			r.Backend,
			r.Rung + " " + r.RungLabel,
			r.Regime + " " + r.RegimeName,
			r.Target + " " + r.TargetName,
			r.Witness,
			r.NextAction,
		}
	}

	width := make([]int, len(header))
	for i, h := range header {
		width[i] = len(h)
	}
	for _, r := range rows {
		for i, c := range cols(r) {
			if len(c) > width[i] {
				width[i] = len(c)
			}
		}
	}

	var b strings.Builder
	b.WriteString("fak support -- per-cell support read-out (rung . regime . target . next-action . witness)\n")
	b.WriteString("folded from internal/covmatrix (scorecard grid) + internal/supportmaturity (ladder/router); deterministic.\n\n")
	writeReadOutRow(&b, header, width)
	rule := make([]string, len(header))
	for i := range rule {
		rule[i] = strings.Repeat("-", width[i])
	}
	writeReadOutRow(&b, rule, width)
	for _, r := range rows {
		writeReadOutRow(&b, cols(r), width)
	}
	if len(rows) == 0 {
		b.WriteString("\n(no cells match the filter)\n")
		return b.String()
	}
	fmt.Fprintf(&b, "\n%d cell(s). The WITNESS column cites the non-author evidence that proves each rung; "+
		"the NEXT-ACTION routes to the dev-loop for that cell's regime.\n", len(rows))
	return b.String()
}

func writeReadOutRow(b *strings.Builder, c []string, width []int) {
	for i, s := range c {
		if i == len(c)-1 {
			b.WriteString(s)
			break
		}
		b.WriteString(s)
		b.WriteString(strings.Repeat(" ", width[i]-len(s)+2))
	}
	b.WriteString("\n")
}
