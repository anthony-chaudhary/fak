package projectreport

// ledger.go is the durable-history half of the project report: the append-only JSONL
// ledger row (a flattened board snapshot), the per-tick trend vs the previous row, and
// the WithTrend attach. It mirrors internal/milestonereport's ledger/trend split and
// delegates the shared plumbing to internal/jsonlledger (parse / latest-prior) and
// internal/trendreport (line marshaller), so the trend accrues on the weekly cadence
// tick exactly like the milestone climb + roadmap do. The fold (projectreport.go) stays
// pure of trend state; the trend is attached here and rides the report envelope + card.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/generation"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// LedgerSchema tags each durable history row so a reader can validate the line.
const LedgerSchema = "fak.project-ledger/1"

// DefaultLedgerRel is the committed, append-only history ledger (one JSONL row per
// project tick). It lives under docs/ so it is durable trunk evidence, not a
// regenerable build artifact — the sibling of docs/milestones/history.jsonl.
const DefaultLedgerRel = "docs/project/history.jsonl"

// LedgerRow is one durable, append-only history line: a flattened projection of the
// board distribution (total, unclassified count, and the four Generation horizon
// columns) so the ledger is a self-describing time series.
type LedgerRow struct {
	Schema          string `json:"schema"`
	Date            string `json:"date"`
	Commit          string `json:"commit"`
	GeneratedAt     string `json:"generated_at"`
	Verdict         string `json:"verdict"`
	Finding         string `json:"finding"`
	Measured        bool   `json:"measured"`
	Total           int    `json:"total"`
	Unclassified    int    `json:"unclassified"`
	Now             int    `json:"now"`
	Next            int    `json:"next"`
	SecondNext      int    `json:"second_next"`
	Future          int    `json:"future"`
	GenUnclassified int    `json:"gen_unclassified"`
}

// RowFromReport projects a folded report into one durable ledger row, normalizing the
// Generation distribution into the four stable horizon columns (anything off the
// closed now/next/second-next/future vocabulary, including the (unset) bucket, folds
// into gen_unclassified).
func RowFromReport(r Report) LedgerRow {
	row := LedgerRow{
		Schema:       LedgerSchema,
		Date:         r.Date,
		Commit:       r.Commit,
		GeneratedAt:  r.GeneratedAt,
		Verdict:      r.Verdict,
		Finding:      r.Finding,
		Measured:     r.Measured,
		Total:        r.Total,
		Unclassified: len(r.Unclassified),
	}
	for k, n := range r.ByGeneration {
		switch normalizeGeneration(k) {
		case "now":
			row.Now += n
		case "next":
			row.Next += n
		case "second-next":
			row.SecondNext += n
		case "future":
			row.Future += n
		default:
			row.GenUnclassified += n
		}
	}
	return row
}

// normalizeGeneration lowers a board Generation value to the closed horizon
// vocabulary, treating anything else (a custom option, or the (unset) bucket) as
// unclassified. It mirrors milestonereport.normalizeGeneration so the two PM surfaces
// bucket horizons alike.
func normalizeGeneration(g string) string { return generation.Normalize(g) }

// ParseLedger parses an append-only JSONL ledger, tolerating blank lines and skipping
// any line that is not a valid dated row (a hand-edit can't crash the reader). Rows are
// returned in file order.
func ParseLedger(content string) []LedgerRow {
	return jsonlledger.Parse(content, func(r LedgerRow) bool { return r.Date != "" })
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline); the caller
// appends it with a newline. Keeping the rendering pure makes the writer testable
// without touching disk.

// Trend is the per-tick delta vs the previous ledger row. The direction is driven by
// the unclassified count (lower is better), then by the tracked item total — a board
// that shed unclassified items OR grew is "improved"; one that gained unclassified
// items is "regressed". With no prior row the trend is "new".
type Trend struct {
	PrevDate          string `json:"prev_date"`
	PrevCommit        string `json:"prev_commit"`
	Direction         string `json:"direction"` // improved | regressed | flat | new
	TotalFrom         int    `json:"total_from"`
	TotalTo           int    `json:"total_to"`
	TotalDelta        int    `json:"total_delta"`
	UnclassifiedFrom  int    `json:"unclassified_from"`
	UnclassifiedTo    int    `json:"unclassified_to"`
	UnclassifiedDelta int    `json:"unclassified_delta"`
	Summary           string `json:"summary"`
}

// TrendVsLast computes the per-tick trend of row against the most recent prior row.
func TrendVsLast(row LedgerRow, prior []LedgerRow) Trend {
	last, ok := latestBefore(row, prior)
	if !ok {
		return Trend{
			Direction:      "new",
			TotalTo:        row.Total,
			UnclassifiedTo: row.Unclassified,
			Summary:        fmt.Sprintf("first project tick (%d item(s), %d unclassified)", row.Total, row.Unclassified),
		}
	}
	totalDelta := row.Total - last.Total
	unclDelta := row.Unclassified - last.Unclassified
	dir := "flat"
	switch {
	case unclDelta < 0 || (unclDelta == 0 && totalDelta > 0):
		dir = "improved"
	case unclDelta > 0:
		dir = "regressed"
	}
	return Trend{
		PrevDate:          last.Date,
		PrevCommit:        last.Commit,
		Direction:         dir,
		TotalFrom:         last.Total,
		TotalTo:           row.Total,
		TotalDelta:        totalDelta,
		UnclassifiedFrom:  last.Unclassified,
		UnclassifiedTo:    row.Unclassified,
		UnclassifiedDelta: unclDelta,
		Summary: fmt.Sprintf("board %s: items %+d (%d->%d), unclassified %+d (%d->%d) vs %s",
			dir, totalDelta, last.Total, row.Total, unclDelta, last.Unclassified, row.Unclassified, last.Date),
	}
}

// latestBefore returns the most recent prior row by (date, then generated_at),
// excluding a row with the exact same generated_at (idempotent re-append).
func latestBefore(row LedgerRow, prior []LedgerRow) (LedgerRow, bool) {
	return jsonlledger.LatestBefore(row, prior,
		func(r LedgerRow) string { return r.Date },
		func(r LedgerRow) string { return r.GeneratedAt })
}

// WithTrend attaches a per-tick trend to the report, returning the reconciled copy. It
// is deliberately non-verdict-touching: the project fold's OK/ACTION/UNMEASURED verdict
// is a measured fact about the board's CURRENT distribution, and a trend is an
// orthogonal time-series signal surfaced alongside it (the report card + --json), not a
// second gate.
func (r Report) WithTrend(t Trend) Report {
	r.Trend = &t
	return r
}
