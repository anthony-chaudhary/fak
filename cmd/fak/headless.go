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
// operator-directed notes found (or a failed self-test), 2 = usage, 3 = --leftovers
// could not establish the run's issues-filed count from evidence (unknown, which is
// deliberately neither a pass nor a refusal). The pure scanner is
// internal/headlesslint; its self-test corpus is asserted by
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
		selfTest       = fs.Bool("self-test", false, "scan the built-in corpus and exit 0 iff every case scans as labeled")
		asJSON         = fs.Bool("json", false, "emit the Report as JSON")
		file           = fs.String("file", "", `read the text to scan from this path ("-" = stdin)`)
		leftovers      = fs.Bool("leftovers", false, "run the RUN-LEVEL end-of-run fold: refuse a final summary that narrates deferred work while zero gh issues were filed")
		closing        = fs.Bool("closing", false, "run the RUN-LEVEL closing-shape fold: refuse a final summary whose last block is a trailing prose wall instead of scannable bullets")
		issuesFiled    = fs.Int("issues-filed", 0, "with --leftovers: DEPRECATED self-report of how many gh issues the run filed during its lifetime; used only when --transcript is absent, and superseded by it")
		transcriptPath = fs.String("transcript", "", "with --leftovers: count the issues the run filed from THIS session transcript's tool-use evidence (JSONL); authoritative over --issues-filed")
		override       = fs.Bool("override", false, `with --leftovers/--closing: the operator escape ("genuinely nothing left" / "this prose closer is deliberate") — forces clean`)
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
		return runHeadlessLeftovers(stdout, stderr, text, *issuesFiled, *transcriptPath, *override, *asJSON)
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
// the operator-facing surface over headlesslint.ScanLeftoversEvidence, enforcing the
// AGENTS.md spine-first rule "End of run: file the leftovers, don't narrate them."
// (#3670). It scans a run's final summary for deferred / out-of-scope narration and
// cross-checks how many issues the run filed.
//
// Where that cross-check COMES FROM is the point (#5425). --transcript makes it evidence:
// the count is walked out of the run's own tool_use inputs (guard_leftovers.go), so a
// summary claiming "I filed them" while the transcript shows no `gh issue create` is
// still refused, and a `--issues-filed 5` alongside a transcript holding two filings
// scores two. --issues-filed survives only as the fallback for callers with no
// transcript to hand over, and the report discloses which of the two produced the number.
//
// Exit: 1 (LEFTOVERS_UNFILED) when leftovers are narrated and the filing count is KNOWN
// to be zero; 3 (filing count unknown) when a transcript was requested but yielded no
// usable evidence — an unknown count is not a zero, so it is neither a pass nor a
// refusal; 0 otherwise.
func runHeadlessLeftovers(stdout, stderr io.Writer, text string, issuesFiled int, transcriptPath string, override, asJSON bool) int {
	filed := headlesslint.ClaimedIssuesFiled(issuesFiled)
	if strings.TrimSpace(transcriptPath) != "" {
		filed = guardIssuesFiledEvidence(transcriptPath)
		if issuesFiled > 0 {
			// Keep the superseded claim visible: the gap between what a run says it
			// filed and what its record shows is the measurement this fold exists for.
			filed = filed.Supersedes(issuesFiled)
		}
	}
	rep := headlesslint.ScanLeftoversEvidence(text, filed, override)
	if asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak headless-lint: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderLeftoversReport(rep, transcriptPath))
	}
	switch {
	case rep.Refused():
		return 1
	case rep.Undecided():
		return 3
	default:
		return 0
	}
}

// renderLeftoversReport is the human-readable view of the run-level fold. Every arm
// names the PROVENANCE of the count, because "2 issues filed" means something different
// depending on whether the run's transcript showed it or the run asserted it.
func renderLeftoversReport(rep headlesslint.LeftoversReport, transcriptPath string) string {
	count, known := rep.FiledCount()
	switch {
	case rep.Undecided():
		var b strings.Builder
		fmt.Fprintf(&b, "fak headless-lint --leftovers: %s — %d leftover(s) narrated and the issues-filed count is UNKNOWN.\n", rep.Verdict, rep.Narrated)
		fmt.Fprintf(&b, "  no usable tool-use evidence at %q — unknown is not zero, so this is neither a pass nor a refusal.\n", transcriptPath)
		appendLeftoversEvidence(&b, rep)
		return b.String()
	case !rep.Refused():
		// A clean report can still carry an unknown count (nothing was narrated, so the
		// count never mattered). Say "unknown" rather than printing a zero for it.
		filed := fmt.Sprintf("%d issue(s) filed (count from: %s)", count, rep.IssuesFiledSource)
		if !known {
			filed = fmt.Sprintf("issues-filed count unknown (source: %s)", rep.IssuesFiledSource)
		}
		return fmt.Sprintf("fak headless-lint --leftovers: clean — %d narrated leftover(s), %s; doctrine: %q\n",
			rep.Narrated, filed, rep.Doctrine)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak headless-lint --leftovers: %s — %d leftover(s) narrated but %d gh issue(s) filed (count from: %s).\n",
		rep.Verdict, rep.Narrated, count, rep.IssuesFiledSource)
	if rep.IssuesFiledClaimed != nil && *rep.IssuesFiledClaimed > count {
		fmt.Fprintf(&b, "  the run claimed %d filed; its transcript evidences %d. The evidence is what counts.\n", *rep.IssuesFiledClaimed, count)
	}
	appendLeftoversEvidence(&b, rep)
	return b.String()
}

// appendLeftoversEvidence writes the shared evidence tail of a leftovers report: the
// doctrine being applied, every matched transcript line, and the remedy. The undecided and
// the refusing reports open differently but must close with the SAME evidence, so an
// operator reading either one sees the same lines and the same instruction.
func appendLeftoversEvidence(b *strings.Builder, rep headlesslint.LeftoversReport) {
	fmt.Fprintf(b, "  doctrine: %q\n\n", rep.Doctrine)
	for _, h := range rep.Hits {
		fmt.Fprintf(b, "  line %-4d %q\n", h.Line, h.Match)
	}
	fmt.Fprintf(b, "\n  instead: %s\n", rep.Resolve)
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
  fak headless-lint --leftovers [--transcript session.jsonl] [--issues-filed N] [--override] "…final summary…"
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
  while the run filed zero gh issues. --override is the escape for a genuine
  "nothing left".

  --transcript session.jsonl COUNTS the filings from evidence: the run's own tool_use
  inputs are walked for issue-creating calls ("fak issue create", "gh issue create",
  native issue-creation tools, nested parallel calls), ignoring prose — so narrating
  "I filed an issue" cannot satisfy the fold, and a claim larger than the record
  scores the record. This is the authoritative source; it supersedes --issues-filed,
  and the report names which source produced the number (issues_filed_source).

  --issues-filed N is the DEPRECATED fallback for callers with no transcript: it is a
  number the audited run asserts about itself, which is the hole --transcript closes.

  When --transcript yields no usable evidence (missing/unreadable/empty) the count is
  UNKNOWN, not zero: the fold reports leftovers_filing_unknown and exits 3 rather than
  refusing, because "we could not tell" is a different fact from "the run filed none".

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
  filing unknown (3)     --leftovers only: leftovers narrated and the issues-filed
                         count could not be established from evidence (not a zero)
`
