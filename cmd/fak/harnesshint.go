package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnesshint"
)

func cmdHarnessHint(argv []string) {
	os.Exit(runHarnessHint(os.Stdout, os.Stderr, argv))
}

func runHarnessHint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak harnesshint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "model identifier or alias to resolve hints for (e.g. gemini-3.8-flash, gpt-4o)")
	asJSON := fs.Bool("json", false, "emit scope hint as JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	hint := harnesshint.ResolveHint(*model, nil)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(hint); err != nil {
			fmt.Fprintf(stderr, "fak harnesshint: json encode: %v\n", err)
			return 2
		}
		return 0
	}

	fmt.Fprintf(stdout, "Model:        %s\n", hint.Model)
	fmt.Fprintf(stdout, "Canonical:    %s\n", hint.CanonicalModel)
	fmt.Fprintf(stdout, "Posture:      %s\n", hint.Posture)
	fmt.Fprintf(stdout, "Max Turns:    %d\n", hint.MaxTurnsRecommended)
	fmt.Fprintf(stdout, "Decompose:    %t\n", hint.DecompositionRecommended)
	fmt.Fprintf(stdout, "Context:      %d tokens\n", hint.ContextBudgetRecommended)
	fmt.Fprintf(stdout, "Advisory:     %s\n", hint.Advisory)
	fmt.Fprintf(stdout, "Provenance:   %s\n", hint.Provenance)
	return 0
}
