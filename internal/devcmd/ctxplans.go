package devcmd

// fak index ctxplans — the CONTEXT-PLAN-REQUIRED advisory lint (#2202, epic #2198).
// Emits the enumerated context-touching surfaces (cmd/fak verbs + in-repo skills),
// each marked DECLARED (carries a `//fak:ctxplan enters=/pages=/warms=` directive) or
// UNDECLARED (debt), and the advisory undeclared-surface count. This is L7 as code —
// "every surface declares its context plan". Advisory only; never a hard gate. The
// ratchet-free witness (a fixture verb with no declaration raising the debt by one)
// lives in internal/ctxplans, run under `make ci`. --json for tooling; the table for humans.

import (
	"fmt"
	"io"
	"text/tabwriter"

	ctxplans "github.com/anthony-chaudhary/fak/internal/ctxplanlint"
)

func indexCtxPlans(stdout, stderr io.Writer, root string, asJSON bool) int {
	rep, err := ctxplans.Scan(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak index ctxplans: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak index ctxplans")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "KIND\tSURFACE\tCONTEXT PLAN\tPROVENANCE\n")
	for _, s := range rep.Surfaces {
		status := "UNDECLARED"
		prov := ""
		if s.Declared {
			status = "declared"
			prov = fmt.Sprintf("%s:%d", s.File, s.Line)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", s.Kind, s.Name, status, prov)
	}
	if rc := flushTab(tw, stderr, "fak index ctxplans"); rc != 0 {
		return rc
	}
	fmt.Fprintf(stdout, "\ndeclared verbs: %d (floor 10)   undeclared-surface debt: %d (advisory)\n",
		rep.DeclaredVerbs, rep.Debt)
	return 0
}
