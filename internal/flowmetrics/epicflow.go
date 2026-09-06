package flowmetrics

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// This file answers the "time to X percent complete" question, which needs a
// definition of PERCENT that is not a self-report.
//
// Two candidate definitions exist in an issue body, and they are not equally
// honest:
//
//  1. The checkbox ratio (`- [x]` over `- [x]`+`- [ ]`). Cheap, but it is a
//     self-report: a box is ticked by whoever edits the body, with no witness,
//     and nothing forces a tick when the work lands. Measured on this repo the
//     tick rate is ~1.4%, which is not a 1.4% completion rate — it is evidence
//     that nobody ticks boxes. Treat it as a LOWER BOUND only.
//
//  2. The child-issue closure ratio, for the `- [ ] #1234` form. Each child's
//     closedAt is an independent, dated fact, so the ratio is auditable AND it
//     carries timestamps — which is what makes time-to-25/50/75% computable at
//     all. This is the definition to prefer whenever children exist.
//
// EpicProgress reports both and marks which one is load-bearing, so a caller
// can never quietly present a self-report as a witnessed measurement.

// childRefRE matches a task-list item that names a child issue, the only form
// whose completion can be independently witnessed. It is applied one line at a
// time, to lines that have already had fenced blocks removed.
var childRefRE = regexp.MustCompile(`^[ \t]*[-*+]\s+\[[ xX]\][^\n]*?#([0-9]+)\b`)

// unfencedLines splits a body into lines with fenced code blocks dropped, using
// the same fence rule ParseChecklist applies.
//
// Sharing the rule is the point: when the counts skipped fences and the child
// extraction did not, an issue quoting a checklist as an EXAMPLE reported two
// task items and three children, and the phantom child then counted as
// permanently-open work against the epic forever.
func unfencedLines(body string) []string {
	var out []string
	inFence := false
	for _, line := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		if fenceRe.MatchString(line) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		out = append(out, line)
	}
	return out
}

// EpicProgress is one aggregate issue's completion picture.
type EpicProgress struct {
	Issue int    `json:"issue"`
	Title string `json:"title,omitempty"`

	// Checked/Unchecked are the raw self-reported task-list counts.
	Checked   int `json:"checked"`
	Unchecked int `json:"unchecked"`

	// Children are the distinct `#N` issues named by task-list items, and
	// ChildrenClosed how many of those are closed per the issue record.
	Children       []int `json:"children,omitempty"`
	ChildrenClosed int   `json:"children_closed"`

	// Fraction is completion in [0,1], -1 when there is nothing to measure.
	Fraction float64 `json:"fraction"`
	// Basis is "children" (witnessed) or "checkbox" (self-reported) or
	// "none". A caller MUST surface this next to Fraction.
	Basis string `json:"basis"`

	// Milestones holds the first instant each percent threshold was
	// reached, derived from child close timestamps. Only present when
	// Basis=="children". A threshold not yet reached is absent.
	Milestones map[int]time.Time `json:"milestones,omitempty"`
	// HoursTo mirrors Milestones as hours from the epic's open, which is
	// the form an operator compares across epics.
	HoursTo map[int]float64 `json:"hours_to,omitempty"`
}

// DefaultThresholds are the percent-complete marks tracked by default. 100 is
// included so "time to done" comes out of the same fold as time-to-half rather
// than needing a separate code path.
var DefaultThresholds = []int{25, 50, 75, 100}

// ParseTaskList returns the checked count, unchecked count, and the distinct
// child issue numbers named by task-list items in a body.
//
// The counts come from ParseChecklist (dod.go) rather than a second regex pair,
// and the child scan walks the same fence-stripped lines, so the two can never
// disagree about what is in scope. Child-ref extraction stays here since it is
// specific to the witnessed-progress question, and is fenced by ForeignRefFloor
// for the same reason commit refs are — a task line citing an upstream project's
// #14762 is not a child of this repo.
func ParseTaskList(body string) (checked, unchecked int, children []int) {
	list := ParseChecklist(body)
	checked, unchecked = list.Checked, list.Unchecked
	seen := map[int]bool{}
	for _, line := range unfencedLines(body) {
		m := childRefRE.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		n, err := strconv.Atoi(m[1])
		if err != nil || n <= 0 || n >= ForeignRefFloor || seen[n] {
			continue
		}
		seen[n] = true
		children = append(children, n)
	}
	sort.Ints(children)
	return checked, unchecked, children
}

// BuildEpicProgress folds one issue against a lookup of every known issue so
// child closure can be resolved. Children not present in the lookup are kept in
// Children (they are still declared work) but counted as not-closed, and that
// is the conservative direction: an unresolvable child never inflates progress.
//
// thresholds may be nil to use DefaultThresholds.
func BuildEpicProgress(iss Issue, byNumber map[int]Issue, thresholds []int) EpicProgress {
	if len(thresholds) == 0 {
		thresholds = DefaultThresholds
	}
	ep := EpicProgress{Issue: iss.Number, Title: iss.Title, Fraction: -1, Basis: "none"}
	ep.Checked, ep.Unchecked, ep.Children = ParseTaskList(iss.Body)

	if len(ep.Children) > 0 {
		// Collect the close instants we can witness, in order, so the
		// k-th close is the moment k/N was reached.
		var closes []time.Time
		for _, n := range ep.Children {
			child, ok := byNumber[n]
			if !ok || !child.Closed() {
				continue
			}
			closes = append(closes, *child.ClosedAt)
		}
		ep.ChildrenClosed = len(closes)
		sort.Slice(closes, func(a, b int) bool { return closes[a].Before(closes[b]) })
		total := len(ep.Children)
		ep.Fraction = float64(ep.ChildrenClosed) / float64(total)
		ep.Basis = "children"
		ep.Milestones = map[int]time.Time{}
		ep.HoursTo = map[int]float64{}
		for _, pct := range thresholds {
			// need is the smallest child count that reaches pct%.
			// Ceiling division keeps 50% of 3 children at 2, never
			// 1: a threshold is reached when it is MET or passed.
			need := (pct*total + 99) / 100
			if need < 1 {
				need = 1
			}
			if need > len(closes) {
				continue
			}
			at := closes[need-1]
			ep.Milestones[pct] = at
			ep.HoursTo[pct] = hours(iss.CreatedAt, at)
		}
		return ep
	}

	if tot := ep.Checked + ep.Unchecked; tot > 0 {
		// No child refs: fall back to the self-report, clearly labelled.
		// No milestones are emitted, because a tick carries no date and
		// inventing one from updatedAt would be a fabricated timestamp.
		ep.Fraction = float64(ep.Checked) / float64(tot)
		ep.Basis = "checkbox"
	}
	return ep
}

// IsAggregate reports whether an issue is aggregate work rather than a leaf,
// using only signals present on the issue itself: an `epic` label, an
// epic/track-shaped title, or a task list with at least minItems items.
//
// This deliberately mirrors the LOOSER of the repo's two existing epic rules
// (internal/issuehygiene's label-or-title test) and adds the task-list arm,
// because for a FLOW measurement a false positive costs one extra progress row
// while a false negative hides a whole project's worth of WIP.
func IsAggregate(iss Issue, minItems int) bool {
	for _, l := range iss.Labels {
		if l == "epic" {
			return true
		}
	}
	t := strings.ToLower(iss.Title)
	for _, p := range []string{"epic", "[epic]", "track(", "[track]"} {
		if strings.HasPrefix(t, p) {
			return true
		}
	}
	c, u, _ := ParseTaskList(iss.Body)
	return minItems > 0 && c+u >= minItems
}

// Summary returns a formatted one-line progress string for the aggregate.
// For basis=="children", it reports fraction, basis, children closed/total,
// and hours to each reached default threshold (25%, 50%, 75%, 100%).
// For basis=="checkbox", it reports fraction, checked/total items, basis,
// and emits no milestone timestamps.
// For basis=="none", it reports unmeasurable.
func (ep EpicProgress) Summary() string {
	switch ep.Basis {
	case "children":
		total := len(ep.Children)
		var reached []string
		for _, pct := range DefaultThresholds {
			if h, ok := ep.HoursTo[pct]; ok {
				reached = append(reached, fmt.Sprintf("%d%%: %.1fh", pct, h))
			}
		}
		milestones := ""
		if len(reached) > 0 {
			milestones = "; hours to " + strings.Join(reached, ", ")
		}
		return fmt.Sprintf("#%d: %.0f%% (%d/%d children, basis=children%s)",
			ep.Issue, ep.Fraction*100, ep.ChildrenClosed, total, milestones)
	case "checkbox":
		total := ep.Checked + ep.Unchecked
		return fmt.Sprintf("#%d: %.0f%% (%d/%d checkboxes, basis=checkbox)",
			ep.Issue, ep.Fraction*100, ep.Checked, total)
	default:
		return fmt.Sprintf("#%d: unmeasurable (basis=none)", ep.Issue)
	}
}

// OpenCount returns the count of unfinished items for sorting unmeasurable epics:
// open children when children exist, otherwise unchecked checkboxes.
func (ep EpicProgress) OpenCount() int {
	if len(ep.Children) > 0 {
		return len(ep.Children) - ep.ChildrenClosed
	}
	return ep.Unchecked
}

// EpicsProgress folds every open aggregate issue against known issues, sorted
// deterministically by issue number.
func EpicsProgress(issues []Issue, byNumber map[int]Issue) []EpicProgress {
	var out []EpicProgress
	for _, iss := range issues {
		if iss.Closed() || !IsAggregate(iss, 5) {
			continue
		}
		out = append(out, BuildEpicProgress(iss, byNumber, nil))
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Issue < out[j].Issue
	})
	return out
}

// UnmeasurableAggregates returns open aggregate issues whose basis is not
// "children" (i.e. checkbox-only or none), sorted worst-first by open count,
// capped at limit (<=0 means no limit).
func UnmeasurableAggregates(epics []EpicProgress, limit int) []EpicProgress {
	var unmeasurable []EpicProgress
	for _, ep := range epics {
		if ep.Basis != "children" {
			unmeasurable = append(unmeasurable, ep)
		}
	}
	sort.Slice(unmeasurable, func(i, j int) bool {
		oi, oj := unmeasurable[i].OpenCount(), unmeasurable[j].OpenCount()
		if oi != oj {
			return oi > oj
		}
		return unmeasurable[i].Issue < unmeasurable[j].Issue
	})
	if limit > 0 && len(unmeasurable) > limit {
		unmeasurable = unmeasurable[:limit]
	}
	return unmeasurable
}
