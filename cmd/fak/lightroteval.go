package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/lightroteval"
)

func cmdLightRotEval(argv []string) {
	os.Exit(runLightRotEval(os.Stdout, os.Stderr, argv))
}

func runLightRotEval(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak lightroteval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "path to LightRot evaluation JSON fixture ('-' for stdin)")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak lightroteval --input <fixture.json> [--json]")
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
		fmt.Fprintf(stderr, "fak lightroteval: read input: %v\n", err)
		return 2
	}

	var req lightroteval.Request
	if err := json.Unmarshal(raw, &req); err != nil || req.ContractVersion == "" {
		var fix struct {
			Request lightroteval.Request `json:"request"`
		}
		if err2 := json.Unmarshal(raw, &fix); err2 == nil && fix.Request.ContractVersion != "" {
			req = fix.Request
		} else if err != nil {
			fmt.Fprintf(stderr, "fak lightroteval: decode request: %v\n", err)
			return 1
		}
	}

	result := lightroteval.Evaluate(req)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak lightroteval: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "LIGHTROT EVALUATION\n")
		fmt.Fprintf(stdout, "  contract: %s\n", result.ContractVersion)
		fmt.Fprintf(stdout, "  outcome:  %s\n", result.Outcome)
		fmt.Fprintf(stdout, "  reason:   %s\n", result.Reason)
		fmt.Fprintf(stdout, "  evidence: %s\n", result.Evidence)
		if result.Delegate != "" {
			fmt.Fprintf(stdout, "  delegate: %s\n", result.Delegate)
		}
		if len(result.Candidates) > 0 {
			fmt.Fprintf(stdout, "  candidates: %d evaluated\n", len(result.Candidates))
			for _, c := range result.Candidates {
				fmt.Fprintf(stdout, "    - %s (%s): acc=%.4f mse=%.6f\n", c.ID, c.Role, c.Metrics.ReconstructionAccuracy, c.Metrics.MSE)
			}
		}
	}

	switch result.Outcome {
	case lightroteval.OutcomeSupported:
		return 0
	case lightroteval.OutcomeDelegate:
		return 4
	case lightroteval.OutcomeUnsupported:
		return 3
	default:
		return 1
	}
}
