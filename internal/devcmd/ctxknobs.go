package devcmd

// fak index ctxknobs — the MANUAL-OVERLAY COUNTER (#2199, epic #2198). Emits
// the generated inventory of every context-touching flag, env lookup, and
// context-management skill in the tree, each classified operator-debug (fine)
// or user-required (defect) with file:line provenance. --json for tooling; the
// table for humans. The ratchet that refuses a NEW user-required knob lives in
// internal/ctxknobs (TestNoNewUserRequiredKnobs, under `make ci`).

import (
	"fmt"
	"io"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/ctxknobs"
)

func indexCtxKnobs(stdout, stderr io.Writer, root string, asJSON bool) int {
	inv, err := ctxknobs.Scan(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak index ctxknobs: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, inv, "fak index ctxknobs")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "KIND\tNAME\tCLASS\tPROVENANCE\n")
	for _, k := range inv.Knobs {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s:%d\n", k.Kind, k.Name, k.Class, k.File, k.Line)
	}
	if rc := flushTab(tw, stderr, "fak index ctxknobs"); rc != 0 {
		return rc
	}
	fmt.Fprintf(stdout, "\nuser-required: %d (ratchet floor %d)   operator-debug: %d\n",
		inv.UserRequired, ctxknobs.BaselineCount(), inv.OperatorDebug)
	return 0
}
