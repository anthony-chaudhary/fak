package main

// fak negate — the negation-operator surface: the callable primitive the negation program
// (docs/notes/GLOBAL-WORKSPACE-NEGATION-OPERATOR-AEO-2026-07.md) describes as "detect, resolve,
// substitute", exposed over internal/negframe. Each subcommand is a thin CLI shell over the
// library, so the operator is scriptable off the request path and golden-testable:
//
//   fak negate detect  --text/-       locate negatively-framed spans (negframe.Classify).
//                                      exit 1 if any negative is in play, 0 if the prose is clean.
//   fak negate resolve "not shared"    resolve the positive complement over the L2 domain registry
//                                      (--domain to name the domain; --list to dump the registry).
//                                      exit 0 when a positive is resolved, 1 on UNKNOWN.
//   fak negate reframe --text/-        flip unambiguous negative idioms to their positive inverse
//                                      in token space (negframe.Reframe), leaving load-bearing
//                                      judgement-tier prose byte-identical. exit 0.
//
// The triad IS the operator: detect that a negative is in play, resolve its positive residual, or
// substitute the positive form back — the three parts the design note names, one verb.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// negateFlagSet builds a ContinueOnError FlagSet whose usage/errors render to stderr — the shared
// constructor the three subcommands use so they follow the standard cmd/fak flag convention.
func negateFlagSet(name string, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	return fs
}

func cmdNegate(args []string) {
	if len(args) == 0 {
		negateUsage(os.Stderr)
		os.Exit(2)
	}
	var code int
	switch args[0] {
	case "detect":
		code = runNegateDetect(os.Stdin, os.Stdout, os.Stderr, args[1:])
	case "resolve":
		code = runNegateResolve(os.Stdout, os.Stderr, args[1:])
	case "reframe":
		code = runNegateReframe(os.Stdin, os.Stdout, os.Stderr, args[1:])
	case "-h", "--help", "help":
		negateUsage(os.Stdout)
		return
	default:
		fmt.Fprintf(os.Stderr, "fak negate: unknown subcommand %q\n", args[0])
		negateUsage(os.Stderr)
		os.Exit(2)
	}
	if code != 0 {
		os.Exit(code)
	}
}

func negateUsage(w io.Writer) {
	fmt.Fprint(w, `fak negate — the negation operator: detect, resolve, substitute.

usage:
  fak negate detect  [--text S | --file F | -]  [--json]
      Locate negatively-framed spans (negframe.Classify). Exit 1 if any is found.
  fak negate resolve ["not X" | --negated X] [--domain D] [--json] [--list]
      Resolve the positive complement of "not X" over the L2 domain registry.
      Exit 0 when a positive is resolved (exact or candidate set), 1 on UNKNOWN.
  fak negate reframe [--text S | --file F | -]  [--json]
      Flip unambiguous negative idioms to their positive inverse (negframe.Reframe).

Input for detect/reframe defaults to stdin when no --text/--file is given.
`)
}

// runNegateDetect reads prose and reports every negatively-framed span negframe.Classify finds.
// It is the functional detection API (#4461): a scriptable front to the same lint the scorecard
// folds. Exit 1 == a negative is in play (so `fak negate detect … && …` gates on clean prose).
func runNegateDetect(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := negateFlagSet("negate detect", stderr)
	text := fs.String("text", "", `text to scan, or "-" for stdin (default: stdin)`)
	file := fs.String("file", "", "read the text from this file instead of --text/stdin")
	path := fs.String("path", "<stdin>", "provenance label recorded on each finding's Path")
	asJSON := fs.Bool("json", false, "emit the findings as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	input, err := readShapeInput(*text, *file, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak negate detect: %v\n", err)
		return 2
	}
	findings := negframe.Classify(*path, string(input))
	if *asJSON {
		encoded := make([]map[string]any, 0, len(findings))
		for _, finding := range findings {
			encoded = append(encoded, map[string]any{
				"line":       finding.Line,
				"span":       finding.Span,
				"category":   finding.Category,
				"mechanical": finding.Mechanical(),
			})
		}
		negateJSON(stdout, map[string]any{
			"present":  len(findings) > 0,
			"findings": encoded,
		})
	} else {
		writeFindingsHuman(stdout, findings)
	}
	if len(findings) > 0 {
		return 1
	}
	return 0
}

// writeFindingsHuman renders Classify findings as a compact operator block.
func writeFindingsHuman(w io.Writer, findings []negframe.Finding) {
	if len(findings) == 0 {
		fmt.Fprintln(w, "negate detect: clean (no negatively-framed spans)")
		return
	}
	mech := 0
	for _, f := range findings {
		if f.Mechanical() {
			mech++
		}
	}
	fmt.Fprintf(w, "negate detect: %d finding(s) — %d mechanical, %d judgement\n", len(findings), mech, len(findings)-mech)
	for _, f := range findings {
		tier := "judgement"
		fix := f.Hint
		if f.Mechanical() {
			tier = "mechanical"
			fix = "-> " + f.Suggest
		}
		fmt.Fprintf(w, "  %s:%d [%s/%s] %s\n", f.Path, f.Line, f.Category, tier, f.Span)
		if fix != "" {
			fmt.Fprintf(w, "      %s\n", fix)
		}
	}
}

// runNegateResolve resolves the positive complement of "not X" over the L2 registry. The term
// comes from a positional ("not shared") — StripNegation peels the marker — or from --negated.
// --list dumps the registry. Exit 0 when a positive is resolved, 1 on UNKNOWN (fail-closed).
func runNegateResolve(stdout, stderr io.Writer, argv []string) int {
	fs := negateFlagSet("negate resolve", stderr)
	negated := fs.String("negated", "", `the term X in "not X" (alternative to the positional argument)`)
	domain := fs.String("domain", "", "name the enumerable domain (default: infer from the term)")
	asJSON := fs.Bool("json", false, "emit the Resolution as JSON")
	list := fs.Bool("list", false, "list the registered domains and their members, then exit")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *list {
		writeDomainsHuman(stdout)
		return 0
	}

	term := strings.TrimSpace(*negated)
	if term == "" {
		term = strings.TrimSpace(strings.Join(fs.Args(), " "))
	}
	if term == "" {
		fmt.Fprintln(stderr, "fak negate resolve: nothing to resolve (give \"not X\" or --negated X); see --list")
		return 2
	}
	// A positional "not X" carries the negation marker; --negated is the bare atom. Peel a leading
	// marker either way so both spellings resolve the same term.
	if atom, ok := negframe.StripNegation(term); ok {
		term = atom
	}

	res := negframe.Resolve(term, *domain)
	if *asJSON {
		negateJSON(stdout, res)
	} else {
		writeResolutionHuman(stdout, res)
	}
	if res.Resolved() {
		return 0
	}
	return 1
}

// writeResolutionHuman renders a Resolution as a one-block operator answer.
func writeResolutionHuman(w io.Writer, r negframe.Resolution) {
	switch r.Kind {
	case negframe.Exact:
		fmt.Fprintf(w, "resolve: not %q over domain %q => %q (exact)\n", r.Negated, r.Domain, r.Positive)
	case negframe.Candidates:
		fmt.Fprintf(w, "resolve: not %q over domain %q => one of {%s} (candidates)\n",
			r.Negated, r.Domain, strings.Join(r.Members, ", "))
	default:
		fmt.Fprintf(w, "resolve: not %q => UNKNOWN (%s)\n", r.Negated, r.Reason)
	}
}

// writeDomainsHuman dumps the complement registry for `--list`.
func writeDomainsHuman(w io.Writer) {
	fmt.Fprintln(w, "negate resolve: registered domains")
	for _, d := range negframe.Domains() {
		fmt.Fprintf(w, "  %-14s {%s}\n", d.Name, strings.Join(d.Members, ", "))
	}
}

// runNegateReframe flips unambiguous negative idioms to their positive inverse in token space and
// prints the rewritten text (the substitution half of the operator, L0). --json emits the
// ReframeResult telemetry instead of the bare text.
func runNegateReframe(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	fs := negateFlagSet("negate reframe", stderr)
	text := fs.String("text", "", `text to reframe, or "-" for stdin (default: stdin)`)
	file := fs.String("file", "", "read the text from this file instead of --text/stdin")
	asJSON := fs.Bool("json", false, "emit the ReframeResult (text + telemetry) as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	input, err := readShapeInput(*text, *file, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak negate reframe: %v\n", err)
		return 2
	}
	res := negframe.ReframePass(string(input))
	if *asJSON {
		negateJSON(stdout, res)
	} else {
		fmt.Fprint(stdout, res.Text)
		if !strings.HasSuffix(res.Text, "\n") {
			fmt.Fprintln(stdout)
		}
	}
	return 0
}

// emitJSON marshals v as indented JSON to w (the shared --json emitter for this verb).
func negateJSON(w io.Writer, v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Fprintln(w, string(b))
}
