// Package skipledger is a pure fold over one dispatchorder.Result tick into
// an auditable ledger: one row per candidate (skipped or selected), naming
// the issue, lane, reason, and timestamp, plus -- for a skip -- whether it
// was safety- or capacity-related. It never decides anything
// dispatchorder.Plan didn't already decide; it only records the WHY so rate
// loss is auditable later (#1776). Pure: same Result + clock in, same rows
// out; zero I/O, zero clock reads.
package skipledger

import "github.com/anthony-chaudhary/fak/internal/dispatchorder"

// Category is the closed safety/capacity split over dispatchorder's closed
// reason vocabulary. A collision-risk skip is a correctness/safety hazard
// (running both this tick would risk a concurrent edit on overlapping
// scope); every other skip reason -- a worker already live, a cooldown
// window, a stale duplicate collapsed to its fresher twin, or a generation
// outside this window -- is about scheduling/resource capacity, not safety.
// A selected (DispKeep) row carries no category; it was not skipped.
type Category string

const (
	CategoryNone     Category = ""
	CategorySafety   Category = "safety"
	CategoryCapacity Category = "capacity"
)

// Row is one candidate's ledger entry for this tick.
type Row struct {
	Issue         string   `json:"issue"`
	Lane          string   `json:"lane,omitempty"`
	Disposition   string   `json:"disposition"`
	Reason        string   `json:"reason"`
	Category      Category `json:"category,omitempty"`
	Skipped       bool     `json:"skipped"`
	TimestampUnix int64    `json:"timestamp_unix"`
}

// Report is the full ledger for one tick.
type Report struct {
	Rows          []Row `json:"rows"`
	SkippedCount  int   `json:"skipped_count"`
	SelectedCount int   `json:"selected_count"`
}

// Record folds one dispatchorder.Result into ledger rows, stamped with
// nowUnix (the caller supplies the clock as data; this package never reads
// one). Every candidate in res.Order gets exactly one row, in the same order
// Result.Order already carries (kept units first).
func Record(res dispatchorder.Result, nowUnix int64) Report {
	rep := Report{Rows: make([]Row, 0, len(res.Order))}
	for _, r := range res.Order {
		row := Row{
			Issue:         r.ID,
			Lane:          r.Lane,
			Disposition:   string(r.Disposition),
			Reason:        r.Reason,
			TimestampUnix: nowUnix,
		}
		if r.Disposition == dispatchorder.DispKeep {
			rep.SelectedCount++
		} else {
			row.Skipped = true
			row.Category = categorize(r.Disposition)
			rep.SkippedCount++
		}
		rep.Rows = append(rep.Rows, row)
	}
	return rep
}

// categorize applies the closed safety/capacity split documented on Category.
func categorize(d dispatchorder.Disposition) Category {
	if d == dispatchorder.DispCollisionRisk {
		return CategorySafety
	}
	return CategoryCapacity
}
