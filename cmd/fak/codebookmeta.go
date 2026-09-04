package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/codebookmeta"
)

func cmdCodebookmeta(argv []string) {
	os.Exit(runCodebookmeta(os.Stdout, os.Stderr, argv))
}

func runCodebookmeta(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak codebookmeta", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "path to codebook metadata JSON descriptor (or '-' for stdin)")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak codebookmeta --input <file.json> [--json]")
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
		fmt.Fprintf(stderr, "fak codebookmeta: read input: %v\n", err)
		return 2
	}

	caps := codebookmeta.DefaultCapability()
	result, _ := codebookmeta.Parse(raw, caps)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak codebookmeta: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "CODEBOOK METADATA ADJUDICATION\n")
		fmt.Fprintf(stdout, "  outcome: %s\n", result.Outcome)
		fmt.Fprintf(stdout, "  reason:  %s\n", result.Reason)
		if result.Detail != "" {
			fmt.Fprintf(stdout, "  detail:  %s\n", result.Detail)
		}
		if result.Descriptor != nil {
			fmt.Fprintf(stdout, "  artifact: %s\n", result.Descriptor.Provenance.ArtifactID)
			fmt.Fprintf(stdout, "  codebook: %s (%s@%s)\n", result.Descriptor.Codebook.ID, result.Descriptor.Codebook.Kind, result.Descriptor.Codebook.Version)
			fmt.Fprintf(stdout, "  packing:  %s@%s (%d-bit)\n", result.Descriptor.Packing.ID, result.Descriptor.Packing.Version, result.Descriptor.Packing.BitsPerIndex)
		}
	}

	switch result.Outcome {
	case codebookmeta.OutcomeSupported:
		return 0
	case codebookmeta.OutcomeDelegate:
		return 4
	case codebookmeta.OutcomeRefused:
		return 1
	default:
		return 1
	}
}
