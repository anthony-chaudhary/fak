package devcmd

// `fak index execaudit` — the shell over devindex.AuditExecutables (#5648).
//
// The package holds the derivation; this half is only the CLI surface and the exit
// code. The exit code IS the verdict, which is what makes the audit usable as a gate
// rather than a report nobody reads:
//
//	0  every executable package has an adjacent test and a real invocation edge
//	   (or a live, reasoned pin admits it)
//	1  a package is untested or unreachable, or a pin has gone stale, or the
//	   executable domain could not be established at all
//
// The could-not-establish-domain case deliberately shares the failing exit code with
// a red audit: a toolchain that cannot enumerate the module must never be mistaken
// for a module with nothing wrong in it.

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func indexExecAudit(stdout, stderr io.Writer, rootDir string, asJSON bool, limit int) int {
	res, err := devindex.AuditExecutables(devindex.ExecAuditOptions{
		Root: rootDir,
		Pins: devindex.ExecAuditPins,
	})
	if asJSON {
		// Emit the result even on the fail-closed path: the JSON carries the status
		// and the reason, and a caller parsing it must be able to SEE the refusal
		// rather than infer it from an empty document.
		if code := encodeJSONOrFail(stdout, stderr, res, "fak index execaudit"); code != 0 {
			return code
		}
		if err != nil || res.Status != "ok" {
			return 1
		}
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak index execaudit: %s: %v\n", res.Status, err)
		return 1
	}

	fmt.Fprintf(stdout, "executable domain: %d main packages — %d with an adjacent test, %d reached from outside\n",
		res.Domain, res.Tested, res.Reached)

	rows := make([]devindex.ExecPackage, 0, len(res.Packages))
	for _, p := range res.Packages {
		if p.Status != devindex.ExecStatusOK {
			rows = append(rows, p)
		}
	}
	if limit > 0 && len(rows) > limit {
		rows = rows[:limit]
	}
	if len(rows) > 0 {
		fmt.Fprintln(stdout)
		tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "PACKAGE\tSTATUS\tTEST\tREACHED BY")
		for _, p := range rows {
			test := "none"
			if p.HasTest {
				test = fmt.Sprintf("%d file(s)", p.TestFiles)
			}
			by := "nothing outside the package"
			if len(p.Evidence) > 0 {
				by = strings.Join(evidenceClasses(p), ",")
			}
			if p.Status == devindex.ExecStatusPinned {
				by += " [pinned: " + p.PinReason + "]"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", p.Dir, p.Status, test, by)
		}
		if code := flushTab(tw, stderr, "fak index execaudit"); code != 0 {
			return code
		}
	}
	for _, ex := range res.Exceptions {
		if ex.Stale {
			fmt.Fprintf(stdout, "\nSTALE PIN %s: %s\n", ex.Package, ex.Why)
		}
	}
	if res.Status == "ok" {
		fmt.Fprintln(stdout, "\naudit ok — every executable is tested and reached.")
		return 0
	}
	fmt.Fprintf(stdout, "\naudit FAILED — %s\n", res.Reason)
	return 1
}

// evidenceClasses lists the distinguished edge kinds that reached a package, sorted
// so the readout is stable. Which class holds is the actionable part: a package
// reached only by a doc example needs different wiring than one with a build target.
func evidenceClasses(p devindex.ExecPackage) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(p.Evidence))
	for _, e := range p.Evidence {
		if !seen[string(e.Class)] {
			seen[string(e.Class)] = true
			out = append(out, string(e.Class))
		}
	}
	sort.Strings(out)
	return out
}
