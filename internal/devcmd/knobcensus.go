package devcmd

// fak index knobs — the KNOB CENSUS (#2210, epic #2208). Emits the generated
// inventory of every user-facing behavior knob in the tree (guard/session/
// account/model/fleet flags + FAK_* env, plus #2199's context slice), each
// carrying a verdict from the closed two-token vocabulary INTENT | HOUSEKEEPING
// with its disposition, route coverage (which surfaces expose the control — the
// issue's "each INTENT row names its route" witness), owner epic, and file:line
// provenance. --json for tooling;
// the table for humans. It is data-only for now — a query, not a gate; the
// enforcing ratchet (HOUSEKEEPING count non-increasing) lands after two stable
// runs, per the issue.

import (
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/knobcensus"
)

func indexKnobs(stdout, stderr io.Writer, root string, asJSON bool) int {
	census, err := knobcensus.Scan(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak index knobs: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, census, "fak index knobs")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "SURFACE\tNAME\tVERDICT\tDISPOSITION\tROUTE\tOWNER\tPROVENANCE\n")
	for _, k := range census.Knobs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s:%d\n",
			k.Surface, k.Name, k.Verdict, k.Disposition, k.Route(), k.OwnerEpic, k.File, k.Line)
	}
	if rc := flushTab(tw, stderr, "fak index knobs"); rc != 0 {
		return rc
	}
	fmt.Fprintf(stdout, "\nINTENT: %d (promote — #2208)   HOUSEKEEPING: %d (automate — #2198)\n",
		census.Intent, census.Housekeeping)
	return 0
}
