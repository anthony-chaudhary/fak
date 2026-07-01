// Package issuecost is a pure fold over per-issue worker-cost rows so the
// 400-issues/hour throughput model can be driven by MEASURED distributions
// (median + p95 of worker elapsed time) instead of a guessed constant.
//
// # What one row records
//
// One IssueCost is what a dispatch worker spent on a single GitHub issue: the
// issue number, the wall-clock elapsed seconds, how many attempts it took, and
// the final Outcome (shipped | blocked | abandoned). A durable ledger is just a
// sequence of these rows appended over time; the fold below turns that ledger
// into a Report an operator can read.
//
// # Percentile method
//
// The elapsed-time median and p95 are NOT re-derived here. They reuse the
// NEAREST-RANK percentiles in internal/fleetmetrics (the "C = 1" variant), so
// there is exactly one percentile method in the tree and a fixture verified
// against fleetmetrics' hand-computed ranks stays valid here. For a slice of N
// elapsed values sorted ascending and a percentile p in [0,100]:
//
//	rank  = ceil( (p/100) * N )   clamped to [1, N]   (1-indexed)
//	value = sorted[rank-1]
//
// Nearest-rank always returns a value actually present in the input, which
// makes hand-verifying a fixture trivial. Edge cases inherited from
// fleetmetrics: empty input -> 0 for every percentile; a single value -> that
// value for every percentile.
//
// # Determinism
//
// The fold takes no wall clock: Summary/Median/P95 are pure functions of the
// rows passed in, so a fixture ledger produces the SAME median and p95 on every
// run. The only I/O is the thin JSONL reader (ParseLedger) and the row appender
// (AppendRow); both are optional conveniences and neither is on the hot path.
// Everything imports only the stdlib and internal/fleetmetrics.
package issuecost

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
)

// Outcome is the terminal disposition of a worker's run on one issue.
type Outcome string

const (
	// Shipped: the worker landed a commit that closed the issue.
	Shipped Outcome = "shipped"
	// Blocked: the worker stopped on an external blocker (peer break, quota,
	// missing hardware) without shipping.
	Blocked Outcome = "blocked"
	// Abandoned: the worker gave up (retry budget exhausted, no-op, crash)
	// without either shipping or naming a blocker.
	Abandoned Outcome = "abandoned"
)

// Valid reports whether o is one of the three recognized outcomes.
func (o Outcome) Valid() bool {
	switch o {
	case Shipped, Blocked, Abandoned:
		return true
	default:
		return false
	}
}

// IssueCost is one durable ledger row: what a worker spent on a single issue.
// ElapsedSec is wall-clock seconds; Attempts is how many tries it took (>= 1
// for any real run); Outcome is the terminal disposition.
type IssueCost struct {
	Issue      int     `json:"issue"`
	ElapsedSec float64 `json:"elapsed_sec"`
	Attempts   int     `json:"attempts"`
	Outcome    Outcome `json:"outcome"`
}

// Report is the operator-facing summary of a ledger: the median and p95 of
// elapsed seconds over all rows, the total attempts across every row, and a
// count per outcome. N is the number of rows folded.
type Report struct {
	N             int             `json:"n"`
	MedianSec     float64         `json:"median_sec"`
	P95Sec        float64         `json:"p95_sec"`
	TotalAttempts int             `json:"total_attempts"`
	OutcomeCounts map[Outcome]int `json:"outcome_counts"`
}

// elapsed projects the ElapsedSec field out of a slice of rows.
func elapsed(rows []IssueCost) []float64 {
	ds := make([]float64, len(rows))
	for i, r := range rows {
		ds[i] = r.ElapsedSec
	}
	return ds
}

// Median returns the nearest-rank 50th percentile of elapsed seconds over rows,
// via fleetmetrics. Empty rows -> 0.
func Median(rows []IssueCost) float64 {
	return fleetmetrics.Percentiles(elapsed(rows), 50)[50]
}

// P95 returns the nearest-rank 95th percentile of elapsed seconds over rows,
// via fleetmetrics. Empty rows -> 0.
func P95(rows []IssueCost) float64 {
	return fleetmetrics.Percentiles(elapsed(rows), 95)[95]
}

// Summary folds a ledger into a Report. It computes median and p95 in a single
// fleetmetrics pass, sums attempts, and tallies outcomes. It is pure: no wall
// clock, no I/O, and it does not mutate rows. OutcomeCounts is always non-nil
// (empty map on empty input) so callers can index it without a nil check.
func Summary(rows []IssueCost) Report {
	m := fleetmetrics.Percentiles(elapsed(rows), 50, 95)
	rep := Report{
		N:             len(rows),
		MedianSec:     m[50],
		P95Sec:        m[95],
		OutcomeCounts: map[Outcome]int{},
	}
	for _, r := range rows {
		rep.TotalAttempts += r.Attempts
		rep.OutcomeCounts[r.Outcome]++
	}
	return rep
}

// Render formats a Report for operator output as a single multi-metric line,
// e.g.:
//
//	per-issue worker cost (n=20): median=100.0s p95=190.0s attempts=27 shipped=14 blocked=4 abandoned=2
//
// An empty ledger (N == 0) is reported explicitly rather than as a misleading
// 0s/0s row. Outcomes are listed in a fixed order (shipped, blocked, abandoned)
// so the line is deterministic.
func (r Report) Render() string {
	if r.N == 0 {
		return "per-issue worker cost (n=0): no rows"
	}
	return fmt.Sprintf(
		"per-issue worker cost (n=%d): median=%.1fs p95=%.1fs attempts=%d shipped=%d blocked=%d abandoned=%d",
		r.N, r.MedianSec, r.P95Sec, r.TotalAttempts,
		r.OutcomeCounts[Shipped], r.OutcomeCounts[Blocked], r.OutcomeCounts[Abandoned],
	)
}

// ParseLedger reads a JSONL ledger (one IssueCost per non-blank line) and
// returns the rows in file order. Blank lines are skipped. A line that is not
// valid IssueCost JSON, or that carries an unrecognized Outcome, is an error
// that names the 1-indexed line number so a corrupt ledger is easy to locate.
// The returned slice is safe to hand straight to Summary.
func ParseLedger(data []byte) ([]IssueCost, error) {
	var rows []IssueCost
	sc := bufio.NewScanner(bytes.NewReader(data))
	// Allow long rows without a scanner-buffer overflow on wide ledgers.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := bytes.TrimSpace(sc.Bytes())
		if len(raw) == 0 {
			continue
		}
		var r IssueCost
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, fmt.Errorf("issuecost: line %d: %w", line, err)
		}
		if !r.Outcome.Valid() {
			return nil, fmt.Errorf("issuecost: line %d: unknown outcome %q", line, r.Outcome)
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("issuecost: scan: %w", err)
	}
	return rows, nil
}

// AppendRow marshals one row to a single JSONL line (trailing newline) and
// appends it to prior, returning the new ledger bytes. It is the inverse of a
// ParseLedger read for one row and does no filesystem I/O itself, so a caller
// controls where the bytes land. An invalid Outcome is refused so a corrupt row
// never reaches the durable ledger.
func AppendRow(prior []byte, r IssueCost) ([]byte, error) {
	if !r.Outcome.Valid() {
		return nil, fmt.Errorf("issuecost: refusing row with unknown outcome %q", r.Outcome)
	}
	b, err := json.Marshal(r)
	if err != nil {
		return nil, fmt.Errorf("issuecost: marshal issue %d: %w", r.Issue, err)
	}
	out := make([]byte, 0, len(prior)+len(b)+1)
	out = append(out, prior...)
	out = append(out, b...)
	out = append(out, '\n')
	return out, nil
}

// SortedByIssue returns a copy of rows sorted ascending by issue number, leaving
// the caller's slice untouched. It is a convenience for stable rendering of a
// ledger folded from concurrent workers that appended out of order.
func SortedByIssue(rows []IssueCost) []IssueCost {
	out := make([]IssueCost, len(rows))
	copy(out, rows)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Issue < out[j].Issue })
	return out
}
