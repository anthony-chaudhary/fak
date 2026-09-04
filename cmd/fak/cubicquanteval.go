package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/cubicquanteval"
)

func cmdCubicQuantEval(argv []string) {
	os.Exit(runCubicQuantEval(os.Stdout, os.Stderr, argv))
}

func runCubicQuantEval(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cubicquanteval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "path to cubicquanteval fixture JSON ('-' for stdin)")
	scope := fs.String("scope", cubicquanteval.ScopeReconstruction, "evaluation scope (scalar-reconstruction, model-quality, hardware-performance)")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak cubicquanteval --input <fixture.json> [--scope <scope>] [--json]")
		return 2
	}

	var raw []byte
	var err error
	if *input == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(*input)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak cubicquanteval: read input: %v\n", err)
		return 2
	}

	req := cubicquanteval.Request{
		Scope:       *scope,
		FixtureJSON: raw,
	}
	result := cubicquanteval.Evaluate(req)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak cubicquanteval: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "CUBICQUANT EVALUATION\n")
		fmt.Fprintf(stdout, "  outcome: %s\n", result.Outcome)
		fmt.Fprintf(stdout, "  reason:  %s\n", result.Reason)
		fmt.Fprintf(stdout, "  detail:  %s\n", result.Detail)
		if len(result.Rows) > 0 {
			fmt.Fprintf(stdout, "  rows:    %d evaluated\n", len(result.Rows))
		}
	}

	switch result.Outcome {
	case cubicquanteval.Supported:
		return 0
	case cubicquanteval.Delegate:
		return 4
	case cubicquanteval.Unsupported:
		return 3
	default:
		return 1
	}
}
