// Package closurerate folds a ledger of issue-close records into three
// counters that separate THROUGHPUT (how much got closed) from HONESTY (how
// much of that close activity is backed by a witness).
//
// A high close count alone is not evidence of real work: an issue can be
// closed with a genuine diff/commit/test pointer, or closed with nothing to
// show. This package makes that split visible so a large ClosureRate cannot
// hide a pile of unwitnessed closes.
//
// The core Fold is pure and deterministic: it takes the window length as a
// parameter and never reads the clock, so the same ledger always yields the
// same Report. It is stdlib-only and off the hot path.
package closurerate

import (
	"fmt"
	"sort"
	"strings"
)

// CloseRecord is one issue-close observation in the ledger.
//
// Witness is the honesty bit: it is true only when the close is backed by a
// concrete pointer to work — a diff, a commit SHA, or a passing test. A close
// with no such pointer (Witness == false) is a claimed-without-witness close,
// the drift the net-effectiveness audit flagged.
//
// Witness is only meaningful when Closed is true. A record that is still open
// (Closed == false) is never counted as witnessed, whatever its Witness value.
type CloseRecord struct {
	Issue      int    // issue number, for identification only
	Closed     bool   // whether the issue is closed
	HasWitness bool   // whether the close points at a diff/commit/test
	Note       string // optional human note (e.g. the witness pointer); ignored by Fold
}

// Report is the folded view of a close ledger. THROUGHPUT (Total, Closed,
// ClosureRate, ClosesPerHour) is kept separate from HONESTY
// (WitnessedCloseRate, ClaimedWithoutWitness) by construction.
type Report struct {
	// Throughput.
	Total         int     // number of records in the ledger
	Closed        int     // number of closed records
	ClosureRate   float64 // Closed / Total, in [0,1]; 0 when Total == 0
	WindowHours   float64 // the window the rates are computed over (0 = unset)
	ClosesPerHour float64 // Closed / WindowHours; 0 when WindowHours <= 0

	// Honesty.
	Witnessed             int     // closed records that carry a witness
	WitnessedCloseRate    float64 // Witnessed / Closed, in [0,1]; 0 when Closed == 0
	ClaimedWithoutWitness int     // closed records with NO witness (Closed - Witnessed)
}

// Fold computes the Report for a ledger over a window of windowHours.
//
// It is pure: no clock, no I/O, no mutation of the input. windowHours <= 0
// leaves the per-hour throughput at zero (rate-per-window is simply not
// reported) while still computing the ratio-based counters. An empty ledger
// folds to an all-zero Report with no divide-by-zero.
func Fold(records []CloseRecord, windowHours float64) Report {
	r := Report{
		Total:       len(records),
		WindowHours: windowHours,
	}
	for _, rec := range records {
		if !rec.Closed {
			continue
		}
		r.Closed++
		if rec.HasWitness {
			r.Witnessed++
		}
	}
	r.ClaimedWithoutWitness = r.Closed - r.Witnessed

	if r.Total > 0 {
		r.ClosureRate = float64(r.Closed) / float64(r.Total)
	}
	if r.Closed > 0 {
		r.WitnessedCloseRate = float64(r.Witnessed) / float64(r.Closed)
	}
	if windowHours > 0 {
		r.ClosesPerHour = float64(r.Closed) / windowHours
	}
	return r
}

// String renders the Report with THROUGHPUT and HONESTY in separate blocks, so
// a reader cannot mistake a high close count for witnessed work.
func (r Report) String() string {
	var b strings.Builder
	b.WriteString("closure-rate report\n")

	b.WriteString("  throughput:\n")
	fmt.Fprintf(&b, "    closes:        %d / %d closed (%.1f%%)\n",
		r.Closed, r.Total, r.ClosureRate*100)
	if r.WindowHours > 0 {
		fmt.Fprintf(&b, "    closes/hour:   %.2f over %.1fh window\n",
			r.ClosesPerHour, r.WindowHours)
	} else {
		b.WriteString("    closes/hour:   n/a (no window)\n")
	}

	b.WriteString("  honesty:\n")
	fmt.Fprintf(&b, "    witnessed:     %d / %d closes (%.1f%%)\n",
		r.Witnessed, r.Closed, r.WitnessedCloseRate*100)
	fmt.Fprintf(&b, "    claimed w/o witness: %d\n", r.ClaimedWithoutWitness)

	return b.String()
}

// Line renders the Report as one compact, greppable line — throughput first,
// honesty second, with the claimed-without-witness count last so it is never
// buried by a big close number.
func (r Report) Line() string {
	return fmt.Sprintf(
		"closure=%.1f%% (%d/%d) closes/h=%.2f | witnessed=%.1f%% (%d/%d) claimed-no-witness=%d",
		r.ClosureRate*100, r.Closed, r.Total, r.ClosesPerHour,
		r.WitnessedCloseRate*100, r.Witnessed, r.Closed, r.ClaimedWithoutWitness,
	)
}

// SortedByIssue returns the records sorted by issue number, leaving the input
// untouched. Handy for rendering a deterministic ledger dump alongside a
// Report.
func SortedByIssue(records []CloseRecord) []CloseRecord {
	out := make([]CloseRecord, len(records))
	copy(out, records)
	sort.Slice(out, func(i, j int) bool { return out[i].Issue < out[j].Issue })
	return out
}
