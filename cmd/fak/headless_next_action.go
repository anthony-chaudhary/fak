package main

// `fak headless-lint --next-action` — the STOP-SHAPE fold, sibling of --leftovers and
// --closing over the same final summary. It enforces the AGENTS.md rule "There is no
// blocked state: name the next action (spawn / queue / route around / wait / hand off),
// never a terminal block." via headlesslint.ScanNextAction. A summary that stops in
// terminal-block grammar ("blocked", "cannot proceed", "stuck on") and names no next
// action (a next-step phrase, a stated intent, or a cited ticket) exits 1; otherwise 0.
// There is deliberately no --override arm: the grammar of the stop is the evidence.

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
)

func runHeadlessNextAction(stdout, stderr io.Writer, text string, asJSON bool) int {
	rep := headlesslint.ScanNextAction(text)
	if asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak headless-lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderNextActionReport(rep))
	}
	if rep.Refused() {
		return 1
	}
	return 0
}

// renderNextActionReport is the human-readable view of the stop-shape fold. A clean
// report that still contained block grammar names the phrase that discharged it, so the
// verdict is legible rather than trusted.
func renderNextActionReport(rep headlesslint.NextActionReport) string {
	if !rep.Refused() {
		if rep.Blocked > 0 {
			return fmt.Sprintf("fak headless-lint --next-action: clean — %d block phrase(s), next action named: %q; doctrine: %q\n",
				rep.Blocked, rep.Next, rep.Doctrine)
		}
		return fmt.Sprintf("fak headless-lint --next-action: clean — no terminal-block grammar; doctrine: %q\n", rep.Doctrine)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak headless-lint --next-action: %s — %d block phrase(s) and no next action named.\n", rep.Verdict, rep.Blocked)
	fmt.Fprintf(&b, "  doctrine: %q\n\n", rep.Doctrine)
	for _, h := range rep.Hits {
		fmt.Fprintf(&b, "  line %-4d %q  (%s)\n", h.Line, h.Match, h.Excerpt)
	}
	fmt.Fprintf(&b, "\n  instead: %s\n", rep.Resolve)
	return b.String()
}
