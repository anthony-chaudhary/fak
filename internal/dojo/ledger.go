package dojo

// ledger.go holds the durable, append-only history surface: one JSONL row per
// dojo tick (a flattened projection of the folded report) and the per-tick trend
// of the mean calibration error against the most recent prior row. This is the
// loop's memory — the gym answers "are our predictors getting better calibrated
// over time" only because the calibration error is trended, not re-derived from
// scratch each run. Mirrors internal/cadencereport's ledger.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// LedgerRow is one durable, append-only history line (a flattened projection of
// the folded report, so the ledger is a self-describing time series).
type LedgerRow struct {
	Schema       string  `json:"schema"`
	Date         string  `json:"date"`
	Commit       string  `json:"commit"`
	GeneratedAt  string  `json:"generated_at"`
	Verdict      string  `json:"verdict"`
	LeverCount   int     `json:"lever_count"`
	EpisodeCount int     `json:"episode_count"`
	Measured     int     `json:"measured"`
	Calibrated   int     `json:"calibrated"`
	MeanCalibErr float64 `json:"mean_calib_err"`
	Grade        string  `json:"grade"`
}

// Trend is the per-tick delta vs the previous ledger row: did the mean
// calibration error fall (improved), rise (regressed), or hold (flat)?
type Trend struct {
	PrevDate       string  `json:"prev_date"`
	PrevCommit     string  `json:"prev_commit"`
	Direction      string  `json:"direction"` // improved | regressed | flat | new
	CalibErrFrom   float64 `json:"calib_err_from"`
	CalibErrTo     float64 `json:"calib_err_to"`
	CalibErrDelta  float64 `json:"calib_err_delta"`
	CalibratedFrom int     `json:"calibrated_from"`
	CalibratedTo   int     `json:"calibrated_to"`
	Summary        string  `json:"summary"`
}

// trendDisplayEpsilon is the smallest mean-calib-err delta the trend summary can
// render as nonzero at its %.3f precision (the rounding boundary, 0.0005). A
// change below it prints "+0.000", so the direction is reported "flat" rather
// than a regressed/improved label the operator cannot see in the numbers.
const trendDisplayEpsilon = 5e-4

// RowFromReport projects a folded report into one durable ledger row.
func RowFromReport(r Report) LedgerRow {
	return LedgerRow{
		Schema:       LedgerSchema,
		Date:         r.Date,
		Commit:       r.Commit,
		GeneratedAt:  r.GeneratedAt,
		Verdict:      r.Verdict,
		LeverCount:   r.LeverCount,
		EpisodeCount: r.EpisodeCount,
		Measured:     r.Measured,
		Calibrated:   r.Calibrated,
		MeanCalibErr: r.MeanCalibErr,
		Grade:        r.Grade,
	}
}

// ParseLedger parses an append-only JSONL ledger, tolerating blank lines and
// skipping any line that is not a valid row (so a hand-edit can't crash the
// reader). Rows are returned in file order.
func ParseLedger(content string) []LedgerRow {
	var rows []LedgerRow
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		// Reject rows that are not our schema. A foreign JSONL (e.g. another
		// tool's history with its own "date" field) unmarshals cleanly into
		// LedgerRow — extra fields are dropped, Date survives — so it would
		// otherwise pollute the trend. Committed rows are written via
		// RowFromReport, which stamps Schema=LedgerSchema, so this only drops
		// lines that were never ours.
		if row.Schema != LedgerSchema {
			continue
		}
		if row.Date == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// TrendVsLast computes the per-tick trend of `row` against the most recent prior
// row in `prior`. With no prior row the trend is "new" (the first tick
// establishes the series). Direction reads the mean calibration error: a fall is
// an improvement (the predictors got closer to billed reality).
func TrendVsLast(row LedgerRow, prior []LedgerRow) Trend {
	last, ok := latestBefore(row, prior)
	if !ok {
		return Trend{
			Direction:    "new",
			CalibErrTo:   row.MeanCalibErr,
			CalibratedTo: row.Calibrated,
			Summary: fmt.Sprintf("first dojo tick (mean calib-err %.3f, grade %s, %d/%d calibrated)",
				row.MeanCalibErr, row.Grade, row.Calibrated, row.Measured),
		}
	}
	delta := row.MeanCalibErr - last.MeanCalibErr
	// The direction must agree with the magnitude the summary renders at %.3f: a
	// delta that rounds to +0.000 reads as a contradiction if we still call it
	// "regressed"/"improved" (a real dogfood finding — a +0.00045 corpus drift
	// printed "calibration regressed +0.000 (0.341->0.341)"). So a change finer
	// than the displayed precision is "flat", not a direction the operator can't
	// see. trendDisplayEpsilon is the %.3f rounding boundary.
	dir := "flat"
	if delta <= -trendDisplayEpsilon {
		dir = "improved"
	} else if delta >= trendDisplayEpsilon {
		dir = "regressed"
	}
	return Trend{
		PrevDate:       last.Date,
		PrevCommit:     last.Commit,
		Direction:      dir,
		CalibErrFrom:   last.MeanCalibErr,
		CalibErrTo:     row.MeanCalibErr,
		CalibErrDelta:  delta,
		CalibratedFrom: last.Calibrated,
		CalibratedTo:   row.Calibrated,
		Summary: fmt.Sprintf("calibration %s %+.3f (%.3f->%.3f) vs %s; %d/%d episodes calibrated",
			dir, delta, last.MeanCalibErr, row.MeanCalibErr, last.Date, row.Calibrated, row.Measured),
	}
}

// latestBefore returns the most recent prior row that STRICTLY PRECEDES `row`,
// comparing by (date, then generated_at). Two rows are excluded: one with the
// exact same generated_at as `row` (idempotent re-append, mirroring the cadence
// ledger's rule), and any row that is not strictly before `row` in (date,
// generated_at) order — so a `--date` backfill (a 2026-06-20 row appended when a
// 2026-06-27 row already exists) trends against the prior 06-20 data, never the
// 06-27 FUTURE row. Without the strict-before filter, a backfill would print
// "improved ... vs 2026-06-27" while stamping a 06-20 tick — a direction computed
// against data from the future, contradicting this function's name and doc.
func latestBefore(row LedgerRow, prior []LedgerRow) (LedgerRow, bool) {
	cands := make([]LedgerRow, 0, len(prior))
	for _, p := range prior {
		if p.GeneratedAt != "" && p.GeneratedAt == row.GeneratedAt {
			continue
		}
		// Keep only rows strictly before `row`: an earlier date, or the same date
		// with an earlier generated_at. A same-date row whose generated_at does not
		// sort before `row`'s (including the empty-string case) cannot be proven to
		// precede it, so it is not a valid prior.
		if !(p.Date < row.Date || (p.Date == row.Date && p.GeneratedAt < row.GeneratedAt)) {
			continue
		}
		cands = append(cands, p)
	}
	if len(cands) == 0 {
		return LedgerRow{}, false
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].Date != cands[j].Date {
			return cands[i].Date < cands[j].Date
		}
		return cands[i].GeneratedAt < cands[j].GeneratedAt
	})
	return cands[len(cands)-1], true
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline); the
// caller appends it with a newline. Keeping the rendering pure makes the writer
// testable without touching disk.
func AppendLedgerLine(row LedgerRow) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
