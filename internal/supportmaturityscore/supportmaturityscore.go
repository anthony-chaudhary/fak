// Package supportmaturityscore grades the covmatrix support rungs as a scorecard.
//
// growth_debt is intentionally narrow: only silently undefined cells are hard debt.
// This card is stricter. It answers "how much of the declared model x backend grid is
// actually mature support?" A SUPPORTED cell is mature; PROOF-PATH-ONLY, FENCED, and
// UNDEFINED cells are honest positions on the ladder, but still support-maturity debt.
package supportmaturityscore

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// Schema is the control-pane schema id for this scorecard.
	Schema = "fak-support-maturity-scorecard/1"
	// DebtKey is the headline integer the control pane folds.
	DebtKey = "support_maturity_debt"
)

// Build folds the live coverage matrix into a support-maturity scorecard payload.
func Build() scorecard.Payload {
	cells := covmatrix.Grid()
	counts := countBy(cells)
	supported := counts[covmatrix.Supported]
	total := len(cells)

	var defects []string
	for _, c := range cells {
		if c.Support == covmatrix.Supported {
			continue
		}
		defects = append(defects, fmt.Sprintf("%s x %s is %s", c.Family, c.Backend, c.Support))
	}

	kpi := scorecard.KPI{
		Key:     "supported_cell_coverage",
		Group:   "maturity",
		Score:   pct(supported, total),
		Detail:  fmt.Sprintf("%d/%d model x backend cells are SUPPORTED", supported, total),
		Defects: defects,
	}

	return scorecard.Fold(Schema, []scorecard.KPI{kpi}, DebtKey, nil, scorecard.Messages{
		Finding: fmt.Sprintf("%d model x backend cell(s) below the SUPPORTED maturity rung",
			len(defects)),
		FindingClean:    fmt.Sprintf("all %d model x backend cells are SUPPORTED", total),
		NextAction:      "retire support-maturity debt by moving proof-path/fenced/undefined cells to SUPPORTED with a CI witness",
		NextActionClean: "hold the line: re-run the support-maturity scorecard on every model/backend change",
		ExtraCorpus: map[string]any{
			"families":             len(covmatrix.Families),
			"backends":             len(covmatrix.Backends),
			"cells":                total,
			"supported":            supported,
			"proof_path_only":      counts[covmatrix.ProofPathOnly],
			"fenced":               counts[covmatrix.Fenced],
			"undefined":            counts[covmatrix.Undefined],
			"support_coverage_pct": scorecard.Round1(pct(supported, total)),
		},
	})
}

func countBy(cells []covmatrix.Cell) map[covmatrix.Support]int {
	m := map[covmatrix.Support]int{}
	for _, c := range cells {
		m[c.Support]++
	}
	return m
}

func pct(n, total int) float64 {
	if total == 0 {
		return 100
	}
	return 100 * float64(n) / float64(total)
}
