package contextq

import (
	"fmt"
	"sort"
	"strings"
)

// WorkingSetDiff is the source-set delta between two context queries materialized
// over the SAME image: which working-set handles a second query added, dropped, or
// kept relative to a baseline, plus the sealed/poisoned pages each side refused to
// materialize. A "handle" is a working-set slice identified by (Step, Role,
// Descriptor) — the stable evidence coordinate, independent of byte length or which
// view materialized it. This is the typed core behind `fak debug --cmd context-diff`
// (issue #427): a diff over evidence handles, not over rendered prose.
type WorkingSetDiff struct {
	BaseQuery string     `json:"base_query"`
	NextQuery string     `json:"next_query"`
	Added     []SliceRef `json:"added"`     // in next, not in base
	Removed   []SliceRef `json:"removed"`   // in base, not in next
	Unchanged []SliceRef `json:"unchanged"` // in both (the base side's slice)
	// RefusedBase/RefusedNext are the typed refusals (sealed-page / poisoned)
	// surfaced on each side — the raw-evidence-expansion guard the issue calls for:
	// a refused handle is reported, never silently dropped or force-materialized.
	RefusedBase []Refusal `json:"refused_base,omitempty"`
	RefusedNext []Refusal `json:"refused_next,omitempty"`
}

func sliceHandle(s SliceRef) string {
	return fmt.Sprintf("%d\x1f%s\x1f%s", s.Step, s.Role, s.Descriptor)
}

// DiffWorkingSets computes the source-set delta from base to next. Both Results are
// expected to come from contextq.Query over the same image; the diff is purely over
// the returned working-set handles and refusals, so it makes no assumption about how
// either side was materialized.
func DiffWorkingSets(base, next Result) WorkingSetDiff {
	baseByHandle := make(map[string]SliceRef, len(base.Slices))
	for _, s := range base.Slices {
		baseByHandle[sliceHandle(s)] = s
	}
	nextByHandle := make(map[string]bool, len(next.Slices))

	d := WorkingSetDiff{
		BaseQuery:   base.Query,
		NextQuery:   next.Query,
		RefusedBase: base.Refused,
		RefusedNext: next.Refused,
	}
	for _, s := range next.Slices {
		nextByHandle[sliceHandle(s)] = true
		if _, ok := baseByHandle[sliceHandle(s)]; ok {
			d.Unchanged = append(d.Unchanged, s)
		} else {
			d.Added = append(d.Added, s)
		}
	}
	for _, s := range base.Slices {
		if !nextByHandle[sliceHandle(s)] {
			d.Removed = append(d.Removed, s)
		}
	}
	sortSlices(d.Added)
	sortSlices(d.Removed)
	sortSlices(d.Unchanged)
	return d
}

func sortSlices(s []SliceRef) {
	sort.Slice(s, func(i, j int) bool {
		if s[i].Step != s[j].Step {
			return s[i].Step < s[j].Step
		}
		return s[i].Descriptor < s[j].Descriptor
	})
}

// Markdown renders the diff as a human-readable transcript (the JSON Result pair is
// the machine artifact; this is the operator-facing companion the issue asks for).
func (d WorkingSetDiff) Markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# context source-set diff\n\n")
	fmt.Fprintf(&b, "- base query: %q\n- next query: %q\n\n", d.BaseQuery, d.NextQuery)
	fmt.Fprintf(&b, "| change | step | role | descriptor | bytes |\n|---|---|---|---|---|\n")
	row := func(tag string, s SliceRef) {
		fmt.Fprintf(&b, "| %s | %d | %s | %s | %d |\n", tag, s.Step, s.Role, s.Descriptor, s.Bytes)
	}
	for _, s := range d.Added {
		row("+ added", s)
	}
	for _, s := range d.Removed {
		row("- removed", s)
	}
	for _, s := range d.Unchanged {
		row("= unchanged", s)
	}
	fmt.Fprintf(&b, "\nadded=%d removed=%d unchanged=%d\n", len(d.Added), len(d.Removed), len(d.Unchanged))
	if len(d.RefusedBase)+len(d.RefusedNext) > 0 {
		fmt.Fprintf(&b, "\n## refused (raw-evidence expansion blocked)\n\n")
		refused := func(side string, rs []Refusal) {
			for _, r := range rs {
				fmt.Fprintf(&b, "- [%s] step %d %s %q: %s\n", side, r.Step, r.Role, r.Descriptor, r.Reason)
			}
		}
		refused("base", d.RefusedBase)
		refused("next", d.RefusedNext)
	}
	return b.String()
}
