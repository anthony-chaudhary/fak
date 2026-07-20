// Coverage (#2532) is the defaults honesty meter: the fleet-wide adoption
// scorecard for the two spine-first defaults, scored from WITNESSES only.
//
// docs/spine-first-defaults.md states two defaults for any new unit of work.
// `Adoption` (adoption.go) already measures default 2 in isolation — did a leaf
// file its 3..50+ follow-ons. It cannot answer default 1 at all, and it takes
// the shipped-leaf set on faith from the caller. Coverage closes both halves and
// folds them into the two rates the issue names:
//
//   - spine coverage  = leaves with a spine witness / all new leaves in the window
//   - fan-out coverage = leaves that cleared MinFanout / leaves WITH a spine
//
// The fan-out denominator is the SPINE set, not the leaf set, and that is
// deliberate: default 2 fires "the moment a spine ships", so a leaf with no
// spine has no fan-out obligation yet. Scoring it against all leaves would
// double-count a missing spine as a fan-out failure too, and the two rates would
// stop being independently actionable.
//
// Every input is a witness a machine gathered — paths git says were ADDED in the
// window, the tracked file list, and marker keys read out of issue bodies. None
// of it is a self-report that the defaults were followed, which is the whole
// point: a scorecard that trusts claims is false-green.
//
// Pure, like the rest of this leaf. The caller runs git and gh; Coverage only
// decides the standing.

package issuefanout

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// CoverageSchema identifies the machine-readable adoption scorecard.
const CoverageSchema = "fak.issue-fanout-coverage.v1"

// DefaultCoverageScanCap bounds the tracker export the scorecard reads.
//
// It is deliberately NOT DefaultDedupeCap (300). That bound exists for --live's
// anti-spam dedupe, where scanning the newest issues is exactly right. A coverage
// scan is the opposite shape: fan-out markers cluster in OLDER issues, so a
// 300-issue window over this tracker returns zero markers and the meter reports a
// confident 0% fan-out coverage that is simply wrong. Reusing the dedupe bound
// here was a measured false-red, which is why the two constants are separate.
//
// The value is sized off the real tracker: a 2026-07-20 run examined 5278 issues,
// so a 5000 cap still truncated and reported NOT PROVEN out of the box. A default
// that cannot certify its own repo is the wrong default, hence the headroom.
const DefaultCoverageScanCap = 20000

// leafPathRE matches a tracked path that belongs to an internal leaf, capturing
// the leaf name. A leaf is a directory directly under internal/.
var leafPathRE = regexp.MustCompile(`^internal/([a-z0-9]+)/`)

// markerKeyRE pulls a fan-out marker key out of an issue body. The filed marker
// is an HTML comment in issuecontract's `fak-*-key` grammar (LiveBody emits
// `fak-issuefanout-key`; older filed issues carry `fak-fanout-key`), so the scan
// keys off the VALUE shape — `fanout-<leaf>-<slug>` — rather than one spelling
// of the comment name, and therefore keeps counting issues filed before the
// marker name settled.
var markerKeyRE = regexp.MustCompile(`fanout-[a-z0-9]+-[a-z0-9-]+`)

// LeafWitness is the gathered evidence for one new leaf in the window: whether
// the tree carries the two artifacts doctrine calls a library leaf's spine — a
// test that drives the real object, plus a runnable verb to capture a live run
// of. Both must be present; a test with no runnable path is not a spine.
type LeafWitness struct {
	Leaf    string `json:"leaf"`
	HasTest bool   `json:"has_test"` // a *_test.go inside internal/<leaf>/
	HasVerb bool   `json:"has_verb"` // a cmd/ shell that makes the leaf runnable
}

// LeafCoverage is one leaf's evidence row in the scorecard.
type LeafCoverage struct {
	Leaf        string `json:"leaf"`
	HasTest     bool   `json:"has_test"`
	HasVerb     bool   `json:"has_verb"`
	HasSpine    bool   `json:"has_spine"`    // HasTest && HasVerb
	FanoutFiled int    `json:"fanout_filed"` // distinct fanout-<leaf>-* marker keys
	ClearsFloor bool   `json:"clears_floor"` // HasSpine && FanoutFiled >= MinFanout
	Gap         int    `json:"gap"`          // follow-ons still owed (spine leaves only)
}

// CoverageReport is the two-rate adoption scorecard with per-leaf evidence rows.
//
// OK is the one-bit gate a pipeline drives: both rates are perfect. Rates are
// reported alongside their denominators so a caller never reads a vacuous 0/0 as
// success — an empty denominator yields rate 0, never 1.0, so an empty window
// cannot render as a false-green 100%.
type CoverageReport struct {
	Schema    string `json:"schema"`
	MinFanout int    `json:"min_fanout"`

	NewLeaves      int     `json:"new_leaves"`      // denominator of the spine rate
	SpineWitnessed int     `json:"spine_witnessed"` // numerator of the spine rate
	SpineCoverage  float64 `json:"spine_coverage"`

	Spines         int     `json:"spines"`         // denominator of the fan-out rate
	FanoutCleared  int     `json:"fanout_cleared"` // numerator of the fan-out rate
	FanoutCoverage float64 `json:"fanout_coverage"`

	// Scan provenance: how much of the tracker the marker witness actually saw.
	// A truncated scan cannot prove the fan-out rate, so it is reported, not hidden.
	ScannedIssues int  `json:"scanned_issues"`
	ScanCap       int  `json:"scan_cap"`
	ScanTruncated bool `json:"scan_truncated"`

	OK            bool           `json:"ok"`
	Leaves        []LeafCoverage `json:"leaves"`                   // one evidence row per new leaf, leaf-sorted
	SpineGaps     []string       `json:"spine_gaps"`               // new leaves with no spine witness
	FanoutGaps    []string       `json:"fanout_gaps"`              // spines below the fan-out floor
	OrphanMarkers []string       `json:"orphan_markers,omitempty"` // markers matching no new leaf
}

// CoverageScan records how much of the tracker the marker witness saw: the
// number of issues examined and the bound the export was run under. It travels
// with the fold so the report can say whether the fan-out rate is provable, not
// just what it computed.
type CoverageScan struct {
	Issues int // issues actually examined
	Cap    int // the limit the export ran under (0 = DefaultCoverageScanCap)
}

// Coverage folds the gathered witnesses into the two rates.
//
// Marker counting reuses Adoption, so the longest-prefix-match rule (a leaf whose
// name prefixes another's never steals its count) and orphan detection stay in
// one tested place. Orphans are scored against ALL new leaves, not just spines:
// a marker filed against a leaf that has no spine is a real marker, not an
// orphan, and reporting it as one would invent a bookkeeping failure.
//
// A scan that hit its cap is marked truncated and forces OK false: the export may
// have missed the very issues that would clear a leaf, so the honest verdict is
// "not proven", never a green computed from a window that was cut short.
func Coverage(witnesses []LeafWitness, markerKeys []string, scan CoverageScan) CoverageReport {
	// Collapse duplicate leaves, keeping the strongest witness seen for each —
	// a gatherer that reports internal/foo/foo.go and internal/foo/foo_test.go
	// as separate rows must not have the second row erase the first's evidence.
	byLeaf := map[string]LeafWitness{}
	var names []string
	for _, w := range witnesses {
		leaf := strings.TrimSpace(w.Leaf)
		if leaf == "" {
			continue
		}
		prev, seen := byLeaf[leaf]
		if !seen {
			names = append(names, leaf)
		}
		byLeaf[leaf] = LeafWitness{
			Leaf:    leaf,
			HasTest: prev.HasTest || w.HasTest,
			HasVerb: prev.HasVerb || w.HasVerb,
		}
	}
	sort.Strings(names)

	counts := Adoption(names, markerKeys)
	filed := map[string]int{}
	for _, l := range counts.Leaves {
		filed[l.Leaf] = l.FanoutFiled
	}

	scanCap := scan.Cap
	if scanCap <= 0 {
		scanCap = DefaultCoverageScanCap
	}
	rep := CoverageReport{
		Schema:        CoverageSchema,
		MinFanout:     MinFanout,
		NewLeaves:     len(names),
		OrphanMarkers: counts.OrphanMarkers,
		SpineGaps:     []string{},
		FanoutGaps:    []string{},
		ScannedIssues: scan.Issues,
		ScanCap:       scanCap,
		ScanTruncated: scan.Issues >= scanCap,
	}
	for _, name := range names {
		w := byLeaf[name]
		row := LeafCoverage{
			Leaf:        name,
			HasTest:     w.HasTest,
			HasVerb:     w.HasVerb,
			HasSpine:    w.HasTest && w.HasVerb,
			FanoutFiled: filed[name],
		}
		if row.HasSpine {
			rep.SpineWitnessed++
			rep.Spines++
			row.ClearsFloor = row.FanoutFiled >= MinFanout
			if row.ClearsFloor {
				rep.FanoutCleared++
			} else {
				row.Gap = MinFanout - row.FanoutFiled
				rep.FanoutGaps = append(rep.FanoutGaps, name)
			}
		} else {
			// No spine yet: default 2 has not fired, so this leaf owes no
			// follow-ons and is not a fan-out gap. It is a SPINE gap.
			rep.SpineGaps = append(rep.SpineGaps, name)
		}
		rep.Leaves = append(rep.Leaves, row)
	}

	rep.SpineCoverage = rate(rep.SpineWitnessed, rep.NewLeaves)
	rep.FanoutCoverage = rate(rep.FanoutCleared, rep.Spines)
	rep.OK = len(rep.SpineGaps) == 0 && len(rep.FanoutGaps) == 0 && !rep.ScanTruncated
	return rep
}

// rate is num/den, with an empty denominator scored 0 rather than 1 so a window
// with nothing in it never renders as a false-green 100%.
func rate(num, den int) float64 {
	if den <= 0 {
		return 0
	}
	return float64(num) / float64(den)
}

// NewLeavesFromPaths reduces the paths git reports as ADDED in the window to the
// set of internal leaves they introduce, deduplicated and sorted. Paths outside
// internal/<leaf>/ are ignored.
func NewLeavesFromPaths(paths []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range paths {
		m := leafPathRE.FindStringSubmatch(strings.TrimSpace(strings.ReplaceAll(p, `\`, "/")))
		if m == nil || seen[m[1]] {
			continue
		}
		seen[m[1]] = true
		out = append(out, m[1])
	}
	sort.Strings(out)
	return out
}

// NewLeavesInWindow reduces the window's ADDED paths to the leaves that are
// genuinely new, by subtracting every leaf already witnessed before the window.
//
// A leaf that merely GAINED a file inside the window is not new — it already
// existed — and counting it inflates the spine denominator with long-lived
// leaves that were never subject to the spine default. Measured on this repo, a
// 14-day window reported 315 "new" leaves against 163 real ones (1.9x), which
// renders as a falsely low spine coverage: the false-RED dual of the false-green
// this meter exists to prevent.
//
// An empty addedBefore degenerates to NewLeavesFromPaths, which is the correct
// reading only when the window spans the repo's whole history.
func NewLeavesInWindow(addedInWindow, addedBefore []string) []string {
	pre := map[string]bool{}
	for _, leaf := range NewLeavesFromPaths(addedBefore) {
		pre[leaf] = true
	}
	out := []string{}
	for _, leaf := range NewLeavesFromPaths(addedInWindow) {
		if !pre[leaf] {
			out = append(out, leaf)
		}
	}
	return out
}

// verbFileNamesLeaf reports whether a cmd/fak file base name names the leaf.
//
// A verb shell routinely splits the leaf name into underscore-separated words —
// cmd/fak/issue_fanout.go is the shell for the issuefanout leaf — so an exact
// prefix match silently scores those leaves as having no runnable verb, and thus
// no spine. That is not a rounding error: it dropped issuefanout itself (19
// filed follow-ons) out of the fan-out denominator entirely.
//
// So the match ignores underscores in the file name while still requiring the
// leaf to end on a word boundary. That boundary is what stops a short leaf from
// claiming a longer leaf's shell: benchloop.go does not witness bench (no break
// after "bench"), while bench_loop.go and benchloop's own file do.
func verbFileNamesLeaf(base, leaf string) bool {
	if leaf == "" {
		return false
	}
	i := 0
	for j := 0; j < len(base); j++ {
		if base[j] == '_' {
			if i == len(leaf) {
				return true // leaf consumed exactly, at a word break
			}
			continue
		}
		if i == len(leaf) || base[j] != leaf[i] {
			return false
		}
		i++
	}
	return i == len(leaf)
}

// WitnessLeaves scores each new leaf's spine artifacts against the tracked file
// list. A leaf is test-witnessed by any internal/<leaf>/*_test.go, and
// verb-witnessed by a non-test Go file under cmd/ that carries the leaf name —
// either the cmd/fak/<leaf>*.go verb shell or a cmd/<leaf>/ binary.
func WitnessLeaves(newLeaves, trackedFiles []string) []LeafWitness {
	out := make([]LeafWitness, 0, len(newLeaves))
	for _, leaf := range newLeaves {
		w := LeafWitness{Leaf: leaf}
		testPrefix := "internal/" + leaf + "/"
		verbDir := "cmd/fak/"
		binPrefix := "cmd/" + leaf + "/"
		for _, f := range trackedFiles {
			f = strings.TrimSpace(strings.ReplaceAll(f, `\`, "/"))
			if !strings.HasSuffix(f, ".go") {
				continue
			}
			isTest := strings.HasSuffix(f, "_test.go")
			switch {
			case isTest && strings.HasPrefix(f, testPrefix):
				w.HasTest = true
			case isTest:
				// a test elsewhere is not this leaf's spine test
			case strings.HasPrefix(f, binPrefix):
				w.HasVerb = true
			case strings.HasPrefix(f, verbDir):
				if verbFileNamesLeaf(strings.TrimSuffix(strings.TrimPrefix(f, verbDir), ".go"), leaf) {
					w.HasVerb = true
				}
			}
		}
		out = append(out, w)
	}
	return out
}

// ExtractMarkerKeys pulls every fan-out marker key out of the given issue bodies
// (a `gh issue list --json number,body` export), deduplicated and sorted.
func ExtractMarkerKeys(issues []Issue) []string {
	seen := map[string]bool{}
	var out []string
	for _, iss := range issues {
		for _, k := range markerKeyRE.FindAllString(iss.Body, -1) {
			if seen[k] {
				continue
			}
			seen[k] = true
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// RenderCoverage prints the scorecard for a human: the two headline rates, one
// evidence row per new leaf, and the named offenders behind each rate.
func RenderCoverage(r CoverageReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "spine+fanout adoption scorecard (defaults honesty meter)\n")
	fmt.Fprintf(&b, "  spine coverage:   %s  (%d/%d new leaves carry a spine witness)\n",
		pct(r.SpineWitnessed, r.NewLeaves), r.SpineWitnessed, r.NewLeaves)
	fmt.Fprintf(&b, "  fan-out coverage: %s  (%d/%d spines filed >=%d follow-ons)\n",
		pct(r.FanoutCleared, r.Spines), r.FanoutCleared, r.Spines, r.MinFanout)
	fmt.Fprintf(&b, "  marker scan:      %d issue(s) examined (cap %d)\n", r.ScannedIssues, r.ScanCap)
	if r.ScanTruncated {
		fmt.Fprintf(&b, "  NOT PROVEN: the scan hit its cap — fan-out markers live in OLDER issues, so this\n"+
			"              rate is a floor, not a measurement. Re-run with a larger --scan-cap.\n")
	}
	if len(r.Leaves) > 0 {
		fmt.Fprintf(&b, "evidence:\n")
	}
	for _, l := range r.Leaves {
		spine := "no spine"
		if l.HasSpine {
			spine = "spine   "
		}
		fmt.Fprintf(&b, "  [%s] %-24s test=%-5v verb=%-5v %d/%d follow-on(s)\n",
			spine, l.Leaf, l.HasTest, l.HasVerb, l.FanoutFiled, r.MinFanout)
	}
	if len(r.SpineGaps) > 0 {
		fmt.Fprintf(&b, "spine gaps (new leaf, no runnable spine witness): %s\n", strings.Join(r.SpineGaps, ", "))
	}
	if len(r.FanoutGaps) > 0 {
		fmt.Fprintf(&b, "fan-out gaps (spine shipped, backlog not filed): %s\n", strings.Join(r.FanoutGaps, ", "))
	}
	if len(r.OrphanMarkers) > 0 {
		fmt.Fprintf(&b, "orphan markers (fan-out filed, no new leaf): %s\n", strings.Join(r.OrphanMarkers, ", "))
	}
	return b.String()
}

// pct renders a rate, naming an empty denominator rather than printing 0% as if
// it were a measured failure.
func pct(num, den int) string {
	if den <= 0 {
		return "n/a  "
	}
	return fmt.Sprintf("%5.1f%%", 100*float64(num)/float64(den))
}
