package main

// fak headless-lint — the operator-facing shell over internal/headlesslint, the
// sensor-side dual of `fak choice-triage`. It scans an agent's final-output text
// for operator-directed "pesky notes" — "do you want me to push?", "let me know
// if you'd like the docs", "TODO: handle this later", "please review" — and
// types each into a closed anti-pattern Class with the remediation an autonomous
// worker takes instead of asking (TAKE_OBVIOUS / FRESH_CONTEXT / FILE_TICKET /
// HUMAN_RESIDUAL, inherited from choicetriage).
//
//	fak headless-lint --self-test [--json]     -> scan the built-in corpus; exit 0 iff every case scans as labeled
//	fak headless-lint --file out.txt [--json]  -> scan a file ("-" = stdin)
//	fak headless-lint "…text…" [--json]        -> scan literal text from the args
//	… | fak headless-lint [--json]             -> scan stdin
//
// Exit codes suit `fak headless-lint … && ship`: 0 = clean (headless-safe), 1 =
// operator-directed notes found (or a failed self-test), 2 = usage. The pure
// scanner is internal/headlesslint; its self-test corpus is asserted by
// internal/headlesslint/headlesslint_test.go.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
)

func cmdHeadlessLint(argv []string) {
	os.Exit(runHeadlessLint(os.Stdout, os.Stderr, os.Stdin, argv))
}

func runHeadlessLint(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("headless-lint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		selfTest    = fs.Bool("self-test", false, "scan the built-in corpus and exit 0 iff every case scans as labeled")
		asJSON      = fs.Bool("json", false, "emit the Report as JSON")
		file        = fs.String("file", "", `read the text to scan from this path ("-" = stdin)`)
		leftovers   = fs.Bool("leftovers", false, "run the RUN-LEVEL end-of-run fold: refuse a final summary that narrates deferred work while zero gh issues were filed")
		closing     = fs.Bool("closing", false, "run the RUN-LEVEL closing-shape fold: refuse a final summary whose last block is a trailing prose wall instead of scannable bullets")
		issuesFiled = fs.Int("issues-filed", 0, "with --leftovers: how many gh issues the run filed (or resolved) during its lifetime (the doctrine cross-check)")
		override    = fs.Bool("override", false, `with --leftovers/--closing: the operator escape ("genuinely nothing left" / "this prose closer is deliberate") — forces clean`)
	)
	fs.Usage = func() { fmt.Fprint(stderr, headlessLintUsage) }
	if !parseFlags(fs, argv) {
		return 2
	}

	if *selfTest {
		return runHeadlessSelfTest(stdout, *asJSON)
	}

	text, err := readHeadlessSource(stdin, *file, fs.Args())
	if err != nil {
		fmt.Fprintf(stderr, "fak headless-lint: %v\n", err)
		return 1
	}
	if strings.TrimSpace(text) == "" {
		fmt.Fprint(stderr, headlessLintUsage)
		return 2
	}

	if *leftovers {
		return runHeadlessLeftovers(stdout, stderr, text, *issuesFiled, *override, *asJSON)
	}
	// The closing-shape dual of --leftovers (headless_closing.go): same final summary,
	// the other run-level question — does it CLOSE in a shape the operator can scan?
	if *closing {
		return runHeadlessClosing(stdout, stderr, text, *override, *asJSON)
	}

	rep := headlesslint.Scan(text)
	if *asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak headless-lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderHeadlessReport(rep))
	}
	if rep.Count > 0 {
		return 1
	}
	return 0
}

// runHeadlessLeftovers is the RUN-LEVEL fold behind `fak headless-lint --leftovers`:
// the operator-facing surface over headlesslint.ScanLeftovers, enforcing the AGENTS.md
// spine-first rule "End of run: file the leftovers, don't narrate them." (#3670). It
// scans a run's final summary for deferred / out-of-scope narration and cross-checks
// --issues-filed; exit 1 (LEFTOVERS_UNFILED) when leftovers are narrated but the run
// filed zero issues and --override was not given, 0 otherwise.
func runHeadlessLeftovers(stdout, stderr io.Writer, text string, issuesFiled int, override, asJSON bool) int {
	rep := headlesslint.ScanLeftovers(text, issuesFiled, override)
	if asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak headless-lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderLeftoversReport(rep))
	}
	if rep.Refused() {
		return 1
	}
	return 0
}

// renderLeftoversReport is the human-readable view of the run-level fold.
func renderLeftoversReport(rep headlesslint.LeftoversReport) string {
	if !rep.Refused() {
		return fmt.Sprintf("fak headless-lint --leftovers: clean — %d narrated leftover(s), %d issue(s) filed; doctrine: %q\n",
			rep.Narrated, rep.IssuesFiled, rep.Doctrine)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak headless-lint --leftovers: %s — %d leftover(s) narrated but 0 gh issue(s) filed.\n", rep.Verdict, rep.Narrated)
	fmt.Fprintf(&b, "  doctrine: %q\n\n", rep.Doctrine)
	for _, h := range rep.Hits {
		fmt.Fprintf(&b, "  line %-4d %q\n", h.Line, h.Match)
	}
	fmt.Fprintf(&b, "\n  instead: %s\n", rep.Resolve)
	return b.String()
}

// readHeadlessSource resolves the text to scan: --file (or "-" for stdin), then
// literal positional args, then stdin when neither is given.
func readHeadlessSource(stdin io.Reader, file string, args []string) (string, error) {
	switch {
	case file == "-":
		b, err := io.ReadAll(stdin)
		return string(b), err
	case file != "":
		b, err := os.ReadFile(file)
		return string(b), err
	case len(args) > 0:
		return strings.Join(args, " "), nil
	default:
		b, err := io.ReadAll(stdin)
		return string(b), err
	}
}

func runHeadlessSelfTest(stdout io.Writer, asJSON bool) int {
	cases, passed := headlesslint.RunFixture()
	allOK := passed == len(cases)
	if asJSON {
		_ = writeIndentedJSON(stdout, map[string]any{
			"cases":  cases,
			"passed": passed,
			"total":  len(cases),
			"ok":     allOK,
		})
	} else {
		for _, c := range cases {
			mark := "ok"
			if !c.OK {
				mark = "MISMATCH"
			}
			exp := "dirty"
			if c.Clean {
				exp = "clean"
			}
			fmt.Fprintf(stdout, "  %-26s expect=%-6s got=%-17s %s\n", c.Name, exp, c.Got, mark)
		}
		fmt.Fprintf(stdout, "headless-lint self-test: %d/%d scanned as labeled\n", passed, len(cases))
	}
	if allOK {
		return 0
	}
	return 1
}

// renderHeadlessReport is the human-readable view: one block per finding with
// the offending line, why it pages a human, and what to do instead.
func renderHeadlessReport(rep headlesslint.Report) string {
	if rep.Count == 0 {
		return "fak headless-lint: clean — no operator-directed notes; the output is headless-safe.\n"
	}
	var b strings.Builder
	plural := "note"
	if rep.Count != 1 {
		plural = "notes"
	}
	fmt.Fprintf(&b, "fak headless-lint: %d operator-directed %s — an autonomous worker acts, decides, tickets, or escalates; it never asks.\n\n", rep.Count, plural)
	for _, f := range rep.Findings {
		fmt.Fprintf(&b, "  line %-4d [%s]  %q\n", f.Line, f.Class, f.Match)
		fmt.Fprintf(&b, "           why:     %s\n", f.Reason)
		fmt.Fprintf(&b, "           instead: %s — %s\n\n", f.Disposition, f.Resolve)
	}
	fmt.Fprintf(&b, "verdict: %s (%d", rep.Verdict, rep.Count)
	if rep.NeedsHuman > 0 {
		fmt.Fprintf(&b, ", %d genuine escalation(s) to route", rep.NeedsHuman)
	}
	fmt.Fprint(&b, ")\n")
	return b.String()
}

const headlessLintUsage = `fak headless-lint — scan agent output for operator-directed notes and type each one.

usage:
  fak headless-lint --self-test [--json]
  fak headless-lint --file out.txt [--json]      ("-" reads the text from stdin)
  fak headless-lint "…text…" [--json]
  … | fak headless-lint [--json]
  fak headless-lint --leftovers [--issues-filed N] [--override] "…final summary…"
  fak headless-lint --closing [--override] [--json] "…final summary…"

run-level closing-shape fold (--closing):
  Enforces the AGENTS.md rule "Close operator-facing turns with scannable bullets,
  verdict first; make the last line a bullet carrying the next checkable step."
  Refuses (exit 1, CLOSING_PROSE_WALL) a final summary whose LAST block is a trailing
  prose wall — a long paragraph with no leading bullet, burying the verdict and the
  next step. A bulleted final block or a short single-line closer is clean; --override
  is the escape when the prose closer is deliberate.

run-level end-of-run fold (--leftovers):
  Enforces the AGENTS.md rule "End of run: file the leftovers, don't narrate them."
  Refuses (exit 1) a final summary that narrates deferred / out-of-scope follow-ups
  ("two more things worth doing", "out of scope", "follow-up", "left to do", "TODO")
  while --issues-filed is 0. Pass --issues-filed N once the follow-ups are filed as
  open gh issues, or --override for a genuine "nothing left".

the closed anti-pattern vocabulary (Class -> what to do instead):
  PERMISSION_ASK         asks permission for the obvious next step      -> TAKE_OBVIOUS
  PREFERENCE_ASK         asks the human to choose between options       -> FRESH_CONTEXT
  CLARIFICATION_REQUEST  asks the human to clarify before proceeding    -> FRESH_CONTEXT
  REVIEW_REQUEST         asks a human to review the work                -> FRESH_CONTEXT
  CONFIRMATION_WAIT      blocks on a human confirmation                 -> TAKE_OBVIOUS
  DEFERRED_WORK          punts real work with no bounded ticket         -> FILE_TICKET
  SUGGESTION_PUNT        hands the operator advice instead of acting    -> FRESH_CONTEXT
  OPEN_OFFER             a dangling "let me know if…" offer             -> TAKE_OBVIOUS

  a line naming authority (release/auth/policy) folds to HUMAN_RESIDUAL — a typed
  escalation to route to another agent/model/tool, never an inline question.

verdict & exit code:
  clean (0)              no operator-directed note — headless-safe
  operator_directed (1)  at least one note assumes a human is watching
  usage (2)              a bad flag or empty input
`
