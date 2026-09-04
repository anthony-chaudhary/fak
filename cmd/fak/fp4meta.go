package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fp4meta"
)

func cmdFp4meta(argv []string) {
	os.Exit(runFp4meta(os.Stdout, os.Stderr, argv))
}

func runFp4meta(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak fp4meta", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "path to FP4 metadata JSON descriptor (or '-' for stdin)")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak fp4meta --input <file.json> [--json]")
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
		fmt.Fprintf(stderr, "fak fp4meta: read input: %v\n", err)
		return 2
	}

	caps := fp4meta.DefaultCapabilities()
	desc, result, err := fp4meta.Parse(raw, caps)

	if *asJSON {
		payload := struct {
			Descriptor *fp4meta.Descriptor `json:"descriptor,omitempty"`
			Result     fp4meta.Result      `json:"result"`
			Error      string              `json:"error,omitempty"`
		}{
			Result: result,
		}
		if err == nil {
			payload.Descriptor = &desc
		} else {
			payload.Error = err.Error()
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak fp4meta: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "FP4 METADATA ADJUDICATION\n")
		fmt.Fprintf(stdout, "  outcome: %s\n", result.Outcome)
		fmt.Fprintf(stdout, "  reason:  %s\n", result.Reason)
		if result.Detail != "" {
			fmt.Fprintf(stdout, "  detail:  %s\n", result.Detail)
		}
		if err == nil {
			fmt.Fprintf(stdout, "  variant: %s\n", desc.Variant)
			fmt.Fprintf(stdout, "  format:  %s@%s\n", desc.Artifact.Format, desc.Artifact.Version)
			fmt.Fprintf(stdout, "  recipe:  %s@%s\n", desc.Recipe.ID, desc.Recipe.Version)
		}
	}

	switch result.Outcome {
	case fp4meta.OutcomeAccept:
		return 0
	case fp4meta.OutcomeAbstain:
		return 3
	case fp4meta.OutcomeDelegate:
		return 4
	case fp4meta.OutcomeRefuse:
		return 1
	default:
		return 1
	}
}
