package fleet

import (
	"fmt"
	"strings"
)

// stateOrder is the render order for the per-state summary: usable states first,
// the two trouble states (down, unknown) last.
var stateOrder = []State{StateLive, StateIdle, StateDraining, StateDown, StateUnknown}

// Render draws a compact, operator-readable status frame from a Snapshot. The hard
// property it guarantees: it STAYS READABLE AT 100+ BOXES. Without all it never
// prints one line per box — it summarizes (counts by state and class, the version
// picture, a capped attention list), so a 5-box fleet and a 500-box fleet produce a
// frame of the same bounded height. all appends the per-box table for when an
// operator wants every row. Output is plain ASCII so it diffs cleanly and renders the
// same on every host.
func Render(snap Snapshot, all bool, width int) string {
	if width < 40 {
		width = 72
	}
	var b strings.Builder
	title := fmt.Sprintf("fleet - %d box(es) - readiness %d/100", snap.Total, snap.Score)
	b.WriteString(ruleTop(title, width) + "\n\n")

	fmt.Fprintf(&b, "REACHABLE  %d/%d\n", snap.Reachable, snap.Total)

	b.WriteString("STATE     ")
	wrote := false
	for _, st := range stateOrder {
		if n := snap.ByState[st]; n > 0 {
			fmt.Fprintf(&b, " %s=%d", st, n)
			wrote = true
		}
	}
	if !wrote {
		b.WriteString(" (none)")
	}
	b.WriteString("\n")

	if len(snap.ByClass) > 0 {
		fmt.Fprintf(&b, "CLASS      %s\n", classLine(snap.ByClass, 8))
	}

	switch {
	case snap.ModalVersion != "":
		off := snap.Reachable - modalCount(snap)
		line := fmt.Sprintf("VERSION    %s", snap.ModalVersion)
		if off > 0 {
			line += fmt.Sprintf("  (%d reachable box(es) on other/none)", off)
		}
		b.WriteString(line + "\n")
	case snap.Reachable > 0:
		b.WriteString("VERSION    (none reported)\n")
	}

	// GPU UTIL is the fleet COMPUTE-utilization line — busy/total GPUs and the
	// GPU-weighted busy percent. Printed only when at least one reachable box reported
	// a GPU stat (snap.GPUUtil != nil), so a fleet with no util producers shows no line
	// rather than a misleading "0%". idle = total - busy is the wasted-silicon count
	// the founding 1/8 case is about.
	if g := snap.GPUUtil; g != nil {
		fmt.Fprintf(&b, "GPU UTIL   busy=%d/%d idle=%d (%d%%)\n", g.Busy, g.Total, g.Total-g.Busy, g.UtilPct)
	}

	b.WriteString("\nATTENTION\n")
	for _, it := range snap.Attention {
		fmt.Fprintf(&b, "  [%s] %s\n", strings.ToUpper(it.Level), it.Title)
		if it.Detail != "" {
			fmt.Fprintf(&b, "        %s\n", it.Detail)
		}
	}

	if all && len(snap.Rows) > 0 {
		b.WriteString("\nBOXES\n")
		for _, r := range snap.Rows {
			ver := Dash(r.Version)
			note := r.Note
			if r.Err != "" {
				note = r.Err
			}
			fmt.Fprintf(&b, "  %-18s %-12s %-9s %-10s %s\n", r.ID, Dash(r.Class), r.State, ver, note)
		}
	}

	b.WriteString("\n" + strings.Repeat("=", width))
	return b.String()
}

// classLine renders the by-class counts, capping at max entries (already sorted by
// count) with a "(+N more)" tail so the line stays bounded even when class is used as
// a high-cardinality label across a large fleet — the same cap rule the attention
// lists use, on the one summary aggregate that is otherwise width-unbounded.
func classLine(rows []countRow, max int) string {
	parts := make([]string, 0, max+1)
	for i, c := range rows {
		if i >= max {
			parts = append(parts, fmt.Sprintf("(+%d more)", len(rows)-max))
			break
		}
		parts = append(parts, fmt.Sprintf("%s=%d", c.Key, c.Count))
	}
	return strings.Join(parts, " ")
}

// modalCount returns how many reachable boxes are on the modal version.
func modalCount(snap Snapshot) int {
	for _, v := range snap.Versions {
		if v.Key == snap.ModalVersion {
			return v.Count
		}
	}
	return 0
}

func ruleTop(title string, width int) string {
	head := "== " + title + " "
	if len(head) >= width {
		return head
	}
	return head + strings.Repeat("=", width-len(head))
}

// Dash renders an empty string as "-" for a tidy column.
func Dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
