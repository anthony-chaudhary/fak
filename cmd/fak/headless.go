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
		selfTest = fs.Bool("self-test", false, "scan the built-in corpus and exit 0 iff every case scans as labeled")
		asJSON   = fs.Bool("json", false, "emit the Report as JSON")
		file     = fs.String("file", "", `read the text to scan from this path ("-" = stdin)`)
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
