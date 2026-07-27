package main

// headless_closing.go — the operator-facing shell over internal/headlesslint.ScanClosing,
// the CLOSING-SHAPE dual of `fak headless-lint --leftovers` (headless.go).
//
//	fak headless-lint --closing [--override] [--json] "…final summary…"
//
// --leftovers asks whether a run FILED the follow-ups it narrated. --closing asks the
// other run-level question about the SAME final summary: does it CLOSE in a shape the
// operator can scan? A turn that ends on a trailing multi-sentence prose paragraph
// (over the fold's word threshold, no leading list marker) buries the verdict and the
// next step in text the operator has to re-read to act on — that is CLOSING_PROSE_WALL,
// exit 1. A final block that already leads with a bullet, a short single-line closer,
// and the operator escape (--override, "this prose closer is intentional") each exit 0.
//
// The fold itself is pure and lives in internal/headlesslint/closing.go; this file only
// renders it and maps the verdict to an exit code, exactly like runHeadlessLeftovers.

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
)

// runHeadlessClosing is the RUN-LEVEL closing fold behind `fak headless-lint --closing`:
// exit 1 (CLOSING_PROSE_WALL) when the final block is a prose wall and --override was not
// given, 0 otherwise.
func runHeadlessClosing(stdout, stderr io.Writer, text string, override, asJSON bool) int {
	rep := headlesslint.ScanClosing(text, override)
	if asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak headless-lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderClosingReport(rep))
	}
	if rep.Refused() {
		return 1
	}
	return 0
}

// renderClosingReport is the human-readable view of the closing fold: the verdict, the
// doctrine it enforces, the offending trailing block, and the re-cast that clears it.
func renderClosingReport(rep headlesslint.ClosingReport) string {
	if !rep.Refused() {
		escape := ""
		if rep.Overridden {
			escape = " (operator override)"
		}
		return fmt.Sprintf("fak headless-lint --closing: clean%s — the final block closes in a scannable shape; doctrine: %q\n",
			escape, rep.Doctrine)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak headless-lint --closing: %s — the final block is a trailing prose wall.\n", rep.Verdict)
	fmt.Fprintf(&b, "  doctrine: %q\n\n", rep.Doctrine)
	if rep.Hit != nil {
		fmt.Fprintf(&b, "  line %-4d %d sentence(s), %d word(s)\n", rep.Hit.Line, rep.Hit.Sentences, rep.Hit.Words)
		fmt.Fprintf(&b, "           %q\n", rep.Hit.Excerpt)
	}
	fmt.Fprintf(&b, "\n  instead: %s\n", rep.Resolve)
	fmt.Fprintln(&b, "  (--override is the escape when the prose closer is deliberate)")
	return b.String()
}
