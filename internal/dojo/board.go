package dojo

// board.go holds the pure cross-lever LEADERBOARD fold (#957): one row per lever
// rolled up from a run's scored episodes — the verdict distribution, the mean
// calibration error, the worst metric, and the letter grade — sorted worst-first
// so an operator sees which predictor is least calibrated at a glance. It is a
// projection over Report.Episodes (each episode already carries its Lever), so it
// needs no durable-ledger change: the board is the cross-lever view of ONE run,
// the trend is the across-tick view the ledger already keeps.

import (
	"fmt"
	"sort"
	"strings"
)

// BoardRow is one lever's rollup across the episodes it produced in a run.
type BoardRow struct {
	Lever        string  `json:"lever"`
	Episodes     int     `json:"episodes"`
	Measured     int     `json:"measured"`
	Calibrated   int     `json:"calibrated"`
	OverClaim    int     `json:"over_claim"`
	UnderClaim   int     `json:"under_claim"`
	Unmeasured   int     `json:"unmeasured"`
	MeanCalibErr float64 `json:"mean_calib_err"`
	Grade        string  `json:"grade"`
	WorstMetric  string  `json:"worst_metric,omitempty"` // the metric with the largest calib-err
	WorstCalib   float64 `json:"worst_calib_err"`
}

// Board is the cross-lever leaderboard for one run.
type Board struct {
	Schema string     `json:"schema"`
	Rows   []BoardRow `json:"rows"`
}

// BoardSchema tags the board envelope.
const BoardSchema = "fak-dojo-board/1"

// BoardFromEpisodes folds a run's scored episodes into one row per lever. It is
// pure and total: an empty input yields an empty board; a lever whose episodes
// were all UNMEASURED grades n/a (never a vacuous A). Rows are sorted worst-first
// by mean calibration error (a measured lever always sorts ahead of an
// all-unmeasured one), then by lever name for determinism.
func BoardFromEpisodes(episodes []Episode) Board {
	type acc struct {
		row      BoardRow
		sumCE    float64
		worstCE  float64
		worstMet string
	}
	byLever := map[string]*acc{}
	var order []string
	for _, e := range episodes {
		a, ok := byLever[e.Lever]
		if !ok {
			a = &acc{row: BoardRow{Lever: e.Lever}}
			byLever[e.Lever] = a
			order = append(order, e.Lever)
		}
		a.row.Episodes++
		if e.Verdict == VerdictUnmeasured {
			a.row.Unmeasured++
			continue
		}
		a.row.Measured++
		a.sumCE += e.CalibErr
		switch e.Verdict {
		case VerdictCalibrated:
			a.row.Calibrated++
		case VerdictOverClaim:
			a.row.OverClaim++
		case VerdictUnderClaim:
			a.row.UnderClaim++
		}
		if e.CalibErr > a.worstCE {
			a.worstCE = e.CalibErr
			a.worstMet = e.Metric
		}
	}

	band := DefaultCalibBand()
	rows := make([]BoardRow, 0, len(order))
	for _, lever := range order {
		a := byLever[lever]
		if a.row.Measured > 0 {
			a.row.MeanCalibErr = a.sumCE / float64(a.row.Measured)
			a.row.Grade = band.grade(a.row.MeanCalibErr)
		} else {
			a.row.Grade = gradeNA
		}
		a.row.WorstMetric = a.worstMet
		a.row.WorstCalib = a.worstCE
		rows = append(rows, a.row)
	}

	sort.SliceStable(rows, func(i, j int) bool {
		// A measured lever sorts ahead of an all-unmeasured one (we want the graded
		// rows first); among measured, worst calib-err first; then lever name.
		mi, mj := rows[i].Measured > 0, rows[j].Measured > 0
		if mi != mj {
			return mi
		}
		if rows[i].MeanCalibErr != rows[j].MeanCalibErr {
			return rows[i].MeanCalibErr > rows[j].MeanCalibErr
		}
		return rows[i].Lever < rows[j].Lever
	})
	return Board{Schema: BoardSchema, Rows: rows}
}

// RenderBoard produces the human leaderboard table.
func RenderBoard(b Board) string {
	lines := []string{
		"dojo board — cross-lever calibration leaderboard (worst-first)",
		"",
		fmt.Sprintf("  %-15s %5s %5s %5s %5s %5s %9s %-3s  %s",
			"lever", "eps", "meas", "calb", "over", "undr", "calib_err", "grd", "worst_metric"),
	}
	for _, r := range b.Rows {
		lines = append(lines, fmt.Sprintf("  %-15s %5d %5d %5d %5d %5d %9.3f %-3s  %s",
			truncate(r.Lever, 15), r.Episodes, r.Measured, r.Calibrated, r.OverClaim, r.UnderClaim,
			r.MeanCalibErr, r.Grade, r.WorstMetric))
	}
	if len(b.Rows) == 0 {
		lines = append(lines, "  (no episodes — register a lever and a scenario, then re-run)")
	}
	return strings.Join(lines, "\n")
}
