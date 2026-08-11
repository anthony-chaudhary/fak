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
	case "ownership":
		return runDisambiguationOwnership(stdout, stderr, args[1:])
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
	var scope disambiguation.Scope
	var terms []string
	for n := 0; n < len(args); n++ {
		arg := args[n]
		switch arg {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		case "--scope-kind", "--scope-value":
			if n+1 >= len(args) {
				fmt.Fprintf(stderr, "fak disambiguation query: %s requires a value\n", arg)
				return 2
			}
			n++
			if arg == "--scope-kind" {
				scope.Kind = args[n]
			} else {
				scope.Value = args[n]
			}
		default:
			if strings.HasPrefix(arg, "-") {
				fmt.Fprintf(stderr, "fak disambiguation query: unknown option %q\n", arg)
				return 2
			}
			terms = append(terms, arg)
		}
	}
	if (scope.Kind == "") != (scope.Value == "") {
		fmt.Fprintln(stderr, "fak disambiguation query: --scope-kind and --scope-value are required together")
		return 2
	}
	if selfTest {
		if len(terms) != 0 || scope.Kind != "" {
			fmt.Fprintln(stderr, "fak disambiguation query: --self-test does not accept a term or scope")
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
	var response disambiguation.QueryResponse
	var err error
	if scope.Kind != "" {
		response, err = disambiguation.ResolveScoped(terms[0], scope)
	} else {
		response, err = disambiguation.Resolve(terms[0])
	}
	if err != nil {
		if errors.Is(err, disambiguation.ErrScopeRequired) {
			fmt.Fprintf(stderr, "fak disambiguation query: %v; use --scope-kind and --scope-value\n", err)
			return 3
		}
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

func runDisambiguationOwnership(stdout, stderr io.Writer, args []string) int {
	jsonOutput, selfTest := false, false
	for _, arg := range args {
		switch arg {
		case "--json":
			jsonOutput = true
		case "--self-test":
			selfTest = true
		default:
			fmt.Fprintf(stderr, "fak disambiguation ownership: unknown option %q\n", arg)
			return 2
		}
	}
	if !selfTest {
		fmt.Fprintln(stderr, "usage: fak disambiguation ownership --self-test [--json]")
		return 2
	}
	report := disambiguation.OwnershipSelfCheck()
	if jsonOutput {
		if code := encodeDisambiguationJSON(stdout, stderr, report); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "ownership self-test: accepted=%t rejected_leaf=%t rejected_lane=%t\n", report.AcceptedFixture, report.RejectedLeaf, report.RejectedLane)
	}
	if !report.OK {
		return 1
	}
	return 0
}
