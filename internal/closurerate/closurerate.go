// Package closurerate folds an issue-close ledger into throughput and witness honesty metrics.
package closurerate

import (
	"fmt"
	"sort"
	"strings"
)

// CloseRecord tracks an issue observation in the ledger along with witness verification.
type CloseRecord struct {
	Issue      int
	Closed     bool
	HasWitness bool
	Note       string
}

// Report holds aggregated throughput and honesty metrics for a ledger over a time window.
type Report struct {
	Total                 int
	Closed                int
	ClosureRate           float64
	WindowHours           float64
	ClosesPerHour         float64
	Witnessed             int
	WitnessedCloseRate    float64
	ClaimedWithoutWitness int
}

// Fold aggregates close records into a Report without clock reads or input mutation.
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

// String renders throughput and honesty sections as human-readable multiline text.
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

// Line formats the report into a single concise log line highlighting unbacked claims.
func (r Report) Line() string {
	return fmt.Sprintf(
		"closure=%.1f%% (%d/%d) closes/h=%.2f | witnessed=%.1f%% (%d/%d) claimed-no-witness=%d",
		r.ClosureRate*100, r.Closed, r.Total, r.ClosesPerHour,
		r.WitnessedCloseRate*100, r.Witnessed, r.Closed, r.ClaimedWithoutWitness,
	)
}

// SortedByIssue produces a copy of records ordered by ascending issue identifier.
func SortedByIssue(records []CloseRecord) []CloseRecord {
	out := make([]CloseRecord, len(records))
	copy(out, records)
	sort.Slice(out, func(i, j int) bool { return out[i].Issue < out[j].Issue })
	return out
}
