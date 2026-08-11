package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func cmdDisambiguation(args []string) {
	os.Exit(runDisambiguation(os.Stdout, os.Stderr, args))
}

func runDisambiguation(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak disambiguation schema [--json] [--self-test]\n       fak disambiguation query <canonical-term> [--json]\n       fak disambiguation query --self-test [--json]")
		return 2
	}
	switch args[0] {
	case "schema":
		return runDisambiguationSchema(stdout, stderr, args[1:])
	case "query":
		return runDisambiguationQuery(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak disambiguation: unknown command %q (want schema or query)\n", args[0])
		return 2
	}
}

func runDisambiguationSchema(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak disambiguation schema", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOutput := fs.Bool("json", false, "emit the schema descriptor as JSON")
	selfTest := fs.Bool("self-test", false, "run the hermetic complete/required-omission contract witness")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak disambiguation schema: positional arguments are not accepted")
		return 2
	}

	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if *selfTest {
		report, err := disambiguation.RunSelfTest()
		if err != nil {
			fmt.Fprintf(stderr, "disambiguation schema self-test: FAIL: %v\n", err)
			return 1
		}
		if *jsonOutput {
			if err := enc.Encode(report); err != nil {
				fmt.Fprintf(stderr, "encode self-test report: %v\n", err)
				return 1
			}
			return 0
		}
		fmt.Fprintf(stdout, "PASS %s: complete record accepted; %d required omissions rejected\n", report.Schema, len(report.OmissionsRejected))
		return 0
	}

	if *jsonOutput {
		if err := enc.Encode(disambiguation.Descriptor()); err != nil {
			fmt.Fprintf(stderr, "encode schema descriptor: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "%s (strict exact-version JSON; run with --json for required fields or --self-test for the contract witness)\n", disambiguation.EntrySchemaVersion)
	return 0
}

func runDisambiguationQuery(stdout, stderr io.Writer, args []string) int {
	var jsonOutput, selfTest bool
	var terms []string
	for _, arg := range args {
		switch arg {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		default:
			if strings.HasPrefix(arg, "-") {
				fmt.Fprintf(stderr, "fak disambiguation query: unknown option %q\n", arg)
				return 2
			}
			terms = append(terms, arg)
		}
	}
	if selfTest {
		if len(terms) != 0 {
			fmt.Fprintln(stderr, "fak disambiguation query: --self-test does not accept a term")
			return 2
		}
		report, err := disambiguation.RunQuerySelfTest()
		if err != nil {
			fmt.Fprintf(stderr, "disambiguation query self-test: FAIL: %v\n", err)
			return 1
		}
		if jsonOutput {
			return encodeDisambiguationJSON(stdout, stderr, report)
		}
		fmt.Fprintf(stdout, "PASS %s: alias %q resolved to canonical term %q with a complete %s record\n", report.Schema, report.MatchedAlias, report.CanonicalTerm, report.EntrySchema)
		return 0
	}
	if len(terms) != 1 {
		fmt.Fprintln(stderr, "usage: fak disambiguation query <term> [--json]")
		return 2
	}
	response, err := disambiguation.Resolve(terms[0])
	if err != nil {
		if errors.Is(err, disambiguation.ErrCanonicalTermNotFound) {
			fmt.Fprintf(stderr, "fak disambiguation query: %v\n", err)
			return 3
		}
		fmt.Fprintf(stderr, "fak disambiguation query: %v\n", err)
		return 1
	}
	if jsonOutput {
		return encodeDisambiguationJSON(stdout, stderr, response)
	}
	fmt.Fprintf(stdout, "%s — %s\n", response.Entry.Identity.CanonicalTerm, response.Entry.Definition)
	return 0
}

func encodeDisambiguationJSON(stdout, stderr io.Writer, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		fmt.Fprintf(stderr, "encode disambiguation JSON: %v\n", err)
		return 1
	}
	return 0
}
