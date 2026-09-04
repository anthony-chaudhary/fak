package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/kvquantquality"
)

func cmdKVQuantQuality(argv []string) {
	os.Exit(runKVQuantQuality(os.Stdout, os.Stderr, argv))
}

func runKVQuantQuality(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak kvquantquality", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "path to KV quantization quality evaluation JSON fixture ('-' for stdin)")
	asJSON := fs.Bool("json", false, "emit report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak kvquantquality --input <fixture.json> [--json]")
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
		fmt.Fprintf(stderr, "fak kvquantquality: read input: %v\n", err)
		return 2
	}

	reportBytes, err := kvquantquality.EvaluateJSON(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak kvquantquality: evaluation error: %v\n", err)
		return 1
	}

	var rep kvquantquality.Report
	if err := json.Unmarshal(reportBytes, &rep); err != nil {
		fmt.Fprintf(stderr, "fak kvquantquality: decode report: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak kvquantquality: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "KV QUANTIZATION QUALITY EVALUATION\n")
		fmt.Fprintf(stdout, "  contract: %s\n", rep.Contract)
		fmt.Fprintf(stdout, "  fixture:  %s\n", rep.FixtureID)
		fmt.Fprintf(stdout, "  outcome:  %s\n", rep.Outcome)
		fmt.Fprintf(stdout, "  reason:   %s\n", rep.Reason)
		if rep.Detail != "" {
			fmt.Fprintf(stdout, "  detail:   %s\n", rep.Detail)
		}
		fmt.Fprintf(stdout, "  baseline: %s\n", rep.BaselinePrecision)
		fmt.Fprintf(stdout, "  quant:    %s\n", rep.QuantizedPrecision)
		fmt.Fprintf(stdout, "  metrics:  row_jsd=%.5f output_jsd=%.5f task_drop=%.5f\n", rep.Metrics.RowJSD, rep.Metrics.OutputJSD, rep.Metrics.TaskDrop)
	}

	switch rep.Outcome {
	case kvquantquality.OutcomeSupported:
		return 0
	case kvquantquality.OutcomeDelegate:
		return 4
	case kvquantquality.OutcomeRefused:
		return 3
	default:
		return 1
	}
}
