package conceptcatalog

import (
	"regexp"
	"sort"
	"strings"
)

// Volatile whole-tree counts (fak-private#3072).
//
// The scorecard README renders two kinds of number. Most are CATALOG facts: rows,
// verdicts, definitions, twin-pairs, glossary anchors, the per-KPI clarity table. They
// move only when someone edits the concept corpus or breaks a grounding token, and a
// committed README that disagrees with them is genuinely stale.
//
// A handful are WHOLE-TREE counts: how many confusable tokens the generator discovers by
// walking internal/ and cmd/, how many of those the catalog positions, the coverage debt
// and legacy score derived from them. Every ordinary feature commit that adds a Go
// identifier moves them, yet RelevantPath (rightly) does not gate ordinary .go commits,
// so pinning them byte for byte turned trunk red on every SHA after the last regenerate.
//
// freshEqual therefore compares the README with those volatile fields masked. Generation
// still writes the live numbers; only the freshness verdict stops treating their drift as
// staleness. Catalog/semantic drift - including a clarity defect, which surfaces in the
// per-KPI table - stays a hard failure.

const volatileMask = "<volatile>"

var (
	// Headline rows whose whole value cell is a whole-tree count.
	volatileHeadlinePrefixes = []string{
		"| **Disambiguation-debt (drive to 0)** |",
		"| **Confusable tokens positioned (covered / discovered)** |",
		"| Legacy bounded score (saturates; not the driver) |",
		"namespace coverage  [",
	}
	// The chart title keeps its catalog concept count; score and debt are volatile.
	volatileChartTitle = regexp.MustCompile(`^(concept-disambiguation chart - \d+ concepts) - score .*$`)
	// "  plan             #######....... 465/561" inside the text chart's coverage block.
	volatileChartFamily = regexp.MustCompile(`^  (\S+)\s+[#.]+ \d+/\d+$`)
	// "| plan | 465 | 561 | 96 |" inside the "## Coverage by family" table.
	volatileTableFamily = regexp.MustCompile(`^\| ([^|]+) \| \d+ \| \d+ \| \d+ \|$`)
)

// freshEqual is the freshness comparison for one tracked artifact: checkout line endings
// are always normalized, and the README's volatile whole-tree counts are masked.
func freshEqual(tracked string, actual, expected []byte) bool {
	if generatedBytesEqual(actual, expected) {
		return true
	}
	if tracked != GeneratedReadme {
		return false
	}
	a := maskVolatileCounts(string(normalizeGeneratedNewlines(actual)))
	e := maskVolatileCounts(string(normalizeGeneratedNewlines(expected)))
	return a == e
}

// maskVolatileCounts replaces every whole-tree count in a rendered README with a fixed
// mask. The per-family coverage lists are ordered by their unpositioned gap, which is
// itself volatile, so each contiguous run is reduced to its family names and sorted: the
// SET of families stays a compared catalog fact, their counts and order do not.
func maskVolatileCounts(doc string) string {
	lines := strings.Split(doc, "\n")
	out := make([]string, 0, len(lines))
	var run []string
	flush := func() {
		sort.Strings(run)
		out = append(out, run...)
		run = run[:0]
	}
	for _, ln := range lines {
		if m := volatileChartFamily.FindStringSubmatch(ln); m != nil {
			run = append(run, "  "+m[1]+" "+volatileMask)
			continue
		}
		if m := volatileTableFamily.FindStringSubmatch(ln); m != nil {
			run = append(run, "| "+m[1]+" | "+volatileMask+" |")
			continue
		}
		flush()
		out = append(out, maskVolatileLine(ln))
	}
	flush()
	return strings.Join(out, "\n")
}

func maskVolatileLine(ln string) string {
	for _, p := range volatileHeadlinePrefixes {
		if strings.HasPrefix(ln, p) {
			return p + " " + volatileMask
		}
	}
	if m := volatileChartTitle.FindStringSubmatch(ln); m != nil {
		return m[1] + " - " + volatileMask
	}
	return ln
}
