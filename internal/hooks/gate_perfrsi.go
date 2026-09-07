package hooks

import (
	"sort"
	"strings"
)

var perfCodePrefixes = []string{
	"internal/compute/",
	"internal/model/",
	"benchmarks/",
}

// gatePerformanceRSINudge advises when performance or optimization code is changed
// without consulting/referencing the performance-RSI scorecard, attaching
// performance-RSI evidence, or supplying a "performance-rsi:" / "perfrsi:" trailer.
func gatePerformanceRSINudge(d *StagedDiff) ([]Finding, error) {
	if d == nil {
		return nil, nil
	}

	var matched []string
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		for _, prefix := range perfCodePrefixes {
			if strings.HasPrefix(p, prefix) {
				matched = append(matched, p)
				break
			}
		}
	}

	d.NoteCandidates("PERFORMANCE_RSI_NUDGE", len(matched), "touched performance file(s)")

	if len(matched) == 0 {
		return nil, nil
	}

	// Suppression 1: any staged path attaching performance-RSI evidence or scorecard.
	for _, raw := range d.StagedPaths {
		p := strings.ToLower(strings.ReplaceAll(raw, "\\", "/"))
		if strings.Contains(p, "perfrsiscore") ||
			strings.Contains(p, "performance-rsi") ||
			strings.Contains(p, "perfrsi") {
			return nil, nil
		}
	}

	// Suppression 2: any added line referencing the scorecard, evidence, or trailer.
	for _, al := range d.AddedLines() {
		lower := strings.ToLower(al.Text)
		if strings.Contains(lower, "performance-rsi:") ||
			strings.Contains(lower, "perfrsi:") ||
			strings.Contains(lower, "performance-rsi") ||
			strings.Contains(lower, "perfrsiscore") ||
			strings.Contains(lower, "allow_no_perfrsi_nudge") ||
			strings.Contains(lower, "allow-no-perfrsi-nudge") {
			return nil, nil
		}
	}

	sort.Strings(matched)

	var findings []Finding
	for _, file := range matched {
		findings = append(findings, Finding{
			Gate:     "PERFORMANCE_RSI_NUDGE",
			File:     file,
			Line:     0,
			Detail:   `performance/optimization code in "` + file + `" modified without referencing performance-RSI scorecard or attaching evidence; run ` + "`fak performance-rsi-scorecard`" + ` or add a "performance-rsi:" trailer to silence`,
			Advisory: true,
		})
	}

	return findings, nil
}
