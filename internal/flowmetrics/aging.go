// This file holds the aging in-flight READOUT: the rendering half of the
// aging_wip KPI. The fold lives in flowmetrics.go, the grading in report.go.
//
// WHY IT EXISTS. `kpiAgingWIP` already counts the started-but-unclosed issues
// that have been quiet for longer than AgingWIPDays and defects past
// AgingWIPCeiling, but a COUNT is not a target: "86 issues are rotting" tells an
// operator that finish-before-start is worth doing and nothing at all about
// which issue to finish. This file names the set — oldest first, with the three
// facts needed to pick one up: how long it has been quiet, how many commits
// already went into it, and which lanes those commits touched. Aging work is the
// cheapest throughput in the system precisely because those commits exist.
//
// ONE DEFINITION. The threshold filter lives in `stalledWIP` and is called by
// both the KPI and the readout, so the number in the defect line and the number
// of rows an operator can act on cannot drift apart. That identity is the
// issue's done condition, and report_test pins it.

package flowmetrics

import (
	"fmt"
	"io"
	"strings"
	"time"
)

// stalledWIP narrows the in-flight list to the rows the aging_wip KPI counts:
// started, still open, and quiet for longer than AgingWIPDays. It inherits
// AgingWIP's oldest-first order.
//
// The filter is on AGE, and the input is already sorted by age descending, so
// the result is a prefix of AgingWIP — but the loop stays explicit rather than
// slicing at the first young row, because a future ordering change would then
// silently truncate the set instead of failing a test.
func stalledWIP(spans []Span, now time.Time) []Span {
	cut := float64(AgingWIPDays) * 24
	rows := AgingWIP(spans, now, 0)
	out := make([]Span, 0, len(rows))
	for _, s := range rows {
		if s.AgeHours(now) > cut {
			out = append(out, s)
		}
	}
	return out
}

// RenderAging writes the aging in-flight readout for a report Build already
// produced. It takes the whole Report rather than a row slice so the count it
// prints is by construction the same fold that produced the aging_wip KPI
// value: a caller cannot pass a list from one corpus and a total from another.
//
// now must be the same instant the report was built at; ages are measured
// against it.
func RenderAging(w io.Writer, rep Report, now time.Time) {
	if rep.AgingTotal <= 0 {
		fmt.Fprintf(w, "aging in-flight: none — no started issue has been quiet for over %dd\n", AgingWIPDays)
		return
	}
	over := ""
	if rep.AgingTotal > AgingWIPCeiling {
		over = fmt.Sprintf(", %d over the cap of %d", rep.AgingTotal-AgingWIPCeiling, AgingWIPCeiling)
	}
	fmt.Fprintf(w, "aging in-flight: %d issue(s) started but unclosed for over %dd%s — finish these before starting anything new\n",
		rep.AgingTotal, AgingWIPDays, over)
	for _, s := range rep.Aging {
		lanes := strings.Join(s.Leaves, ",")
		if lanes == "" {
			// An in-flight issue whose commits carried no `(fak <leaf>)`
			// trailer is a real state, not a render bug, so it prints as
			// an explicit dash rather than as an empty column.
			lanes = "-"
		}
		fmt.Fprintf(w, "  #%-6d %6.1fd  %3d commit(s)  %-20s %s\n",
			s.Issue, s.AgeHours(now)/24, s.Commits, lanes, clipTitle(s.Title, 56))
	}
	if shown := len(rep.Aging); shown < rep.AgingTotal {
		fmt.Fprintf(w, "  ... %d more not listed (showing the %d oldest)\n", rep.AgingTotal-shown, shown)
	}
}

// clipTitle bounds one title to max runes so a pathological issue title cannot
// wrap the readout into unreadability. Cutting on runes rather than bytes keeps
// a multi-byte title from being sliced mid-character.
func clipTitle(s string, max int) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) <= max {
		return string(r)
	}
	return string(r[:max-3]) + "..."
}
