package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/bitnetmeta"
)

func cmdBitnetmeta(argv []string) {
	os.Exit(runBitnetmeta(os.Stdout, os.Stderr, argv))
}

func runBitnetmeta(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak bitnetmeta", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "path to BitNet metadata JSON descriptor (or '-' for stdin)")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak bitnetmeta --input <file.json> [--json]")
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
		fmt.Fprintf(stderr, "fak bitnetmeta: read input: %v\n", err)
		return 2
	}

	caps := bitnetmeta.DefaultCapabilities()
	result := bitnetmeta.ParseAndAdjudicate(raw, caps)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak bitnetmeta: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "BITNET METADATA ADJUDICATION\n")
		fmt.Fprintf(stdout, "  outcome: %s\n", result.Outcome)
		fmt.Fprintf(stdout, "  reason:  %s\n", result.Reason)
		fmt.Fprintf(stdout, "  detail:  %s\n", result.Detail)
		if result.Descriptor != nil {
			fmt.Fprintf(stdout, "  artifact: %s (%s@%s)\n", result.Descriptor.Artifact.ID, result.Descriptor.Artifact.Format, result.Descriptor.Artifact.Version)
			fmt.Fprintf(stdout, "  semantic: %s (%s)\n", result.Descriptor.Weights.Semantic, result.Descriptor.Weights.Label)
		}
	}

	switch result.Outcome {
	case bitnetmeta.OutcomeAccept:
		return 0
	case bitnetmeta.OutcomeAbstain:
		return 3
	case bitnetmeta.OutcomeDelegate:
		return 4
	case bitnetmeta.OutcomeRefuse:
		return 1
	default:
		return 1
	}
}
